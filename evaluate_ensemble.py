"""Ensemble para Cassava Leaf Disease Classification.

Este script carrega varios checkpoints gerados por train_experiments.py,
faz media ponderada das probabilidades e avalia no split de validacao ou
gera submission.csv no split de teste.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CassavaDataset
from dataset_ext import CassavaTestDataset
from models.model_factory import build_model
from utils.transforms import build_transforms
from utils.tta import predict_logits


DEFAULT_CLASS_NAMES = ["0", "1", "2", "3", "4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble de checkpoints Cassava")

    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Lista de caminhos para best_model.pth.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--tta", action="store_true",
                        help="Usa TTA em todos os modelos, caso --tta_modes nao seja passado.")
    parser.add_argument("--tta_modes", nargs="*", choices=["none", "tta"], default=None,
                        help="Um modo por checkpoint. Ex.: --tta_modes none tta tta")
    parser.add_argument("--weights", nargs="*", type=float, default=None,
                        help="Um peso por checkpoint. Ex.: --weights 0.4 0.3 0.3")

    parser.add_argument("--val_size", type=float, default=None,
                        help="Sobrescreve val_size. Se omitido, usa args do 1o checkpoint ou 0.15.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Sobrescreve seed. Se omitido, usa args do 1o checkpoint ou 42.")
    parser.add_argument("--val_csv", type=str, default=None,
                        help="CSV opcional com split de validacao. Se omitido, tenta val_split.csv ao lado do 1o checkpoint.")
    parser.add_argument("--force_num_classes", type=int, default=None)

    return parser.parse_args()


def safe_torch_load(path: str, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def validate_args(args: argparse.Namespace) -> None:
    n = len(args.checkpoints)

    if args.tta_modes is not None and len(args.tta_modes) > 0 and len(args.tta_modes) != n:
        raise ValueError(
            f"--tta_modes precisa ter {n} itens, um por checkpoint. Recebido: {len(args.tta_modes)}"
        )

    if args.weights is not None and len(args.weights) > 0 and len(args.weights) != n:
        raise ValueError(
            f"--weights precisa ter {n} itens, um por checkpoint. Recebido: {len(args.weights)}"
        )

    for ckpt in args.checkpoints:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint nao encontrado: {ckpt}")


def normalize_weights(weights: List[float] | None, n: int) -> np.ndarray:
    if weights is None or len(weights) == 0:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.array(weights, dtype=np.float64)

    if np.any(w < 0):
        raise ValueError("Os pesos do ensemble nao podem ser negativos.")
    if np.sum(w) <= 0:
        raise ValueError("A soma dos pesos precisa ser positiva.")
    return w / np.sum(w)


def get_tta_modes(args: argparse.Namespace) -> List[bool]:
    n = len(args.checkpoints)
    if args.tta_modes is None or len(args.tta_modes) == 0:
        return [bool(args.tta)] * n
    return [mode == "tta" for mode in args.tta_modes]


def first_checkpoint_args(checkpoint_path: str, device: torch.device) -> Dict:
    ckpt = safe_torch_load(checkpoint_path, device)
    return ckpt.get("args", {})


def load_val_dataframe(args: argparse.Namespace, first_ckpt_args: Dict) -> pd.DataFrame:
    if args.val_csv is not None:
        if not os.path.exists(args.val_csv):
            raise FileNotFoundError(f"--val_csv nao encontrado: {args.val_csv}")
        return pd.read_csv(args.val_csv).reset_index(drop=True)

    first_ckpt_dir = Path(args.checkpoints[0]).resolve().parent
    candidate = first_ckpt_dir / "val_split.csv"
    if candidate.exists():
        print(f"Usando val_split.csv do primeiro checkpoint: {candidate}")
        return pd.read_csv(candidate).reset_index(drop=True)

    csv_path = os.path.join(args.data_dir, "train.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"train.csv nao encontrado: {csv_path}")

    seed = args.seed if args.seed is not None else first_ckpt_args.get("seed", 42)
    val_size = args.val_size if args.val_size is not None else first_ckpt_args.get("val_size", 0.15)

    df = pd.read_csv(csv_path)
    _, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df["label"],
    )
    return val_df.reset_index(drop=True)


def load_test_dataframe(data_dir: str) -> pd.DataFrame:
    sample_path = os.path.join(data_dir, "sample_submission.csv")
    test_dir = os.path.join(data_dir, "test_images")

    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)[["image_id"]].copy().reset_index(drop=True)

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"test_images nao encontrado: {test_dir}")

    files = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    return pd.DataFrame({"image_id": files})


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    force_num_classes: int | None = None,
) -> Tuple[torch.nn.Module, Dict, Dict]:
    ckpt = safe_torch_load(checkpoint_path, device)
    ckpt_args = ckpt.get("args", {})

    model_name = ckpt.get("model_name") or ckpt_args.get("model", "resnet50")
    image_size = int(ckpt_args.get("image_size", 224))
    drop_rate = float(ckpt_args.get("drop_rate", 0.0))
    num_classes = int(force_num_classes or ckpt.get("num_classes", 5))

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        image_size=image_size,
        drop_rate=drop_rate,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, ckpt, ckpt_args


def make_loader_for_model(
    df: pd.DataFrame,
    args: argparse.Namespace,
    ckpt_args: Dict,
    split: str,
) -> DataLoader:
    image_size = int(ckpt_args.get("image_size", 224))
    segmentation = bool(ckpt_args.get("segmentation", False))

    transform = build_transforms(
        train=False,
        image_size=image_size,
        augmentation="none",
        use_segmentation=segmentation,
        frequency_aug=False,
    )

    pin = torch.cuda.is_available()

    if split == "val":
        img_dir = os.path.join(args.data_dir, "train_images")
        ds = CassavaDataset(df, img_dir=img_dir, transform=transform)
    else:
        img_dir = os.path.join(args.data_dir, "test_images")
        ds = CassavaTestDataset(df, img_dir=img_dir, transform=transform)

    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )


@torch.no_grad()
def predict_probs_val(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool,
) -> np.ndarray:
    probs_all = []
    for images, labels in tqdm(loader, desc="Validacao do modelo", leave=False):
        images = images.to(device, non_blocking=True)
        logits = predict_logits(model, images, tta=use_tta)
        probs = torch.softmax(logits, dim=1)
        probs_all.append(probs.cpu().numpy())
    return np.concatenate(probs_all, axis=0)


@torch.no_grad()
def predict_probs_test(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool,
) -> np.ndarray:
    probs_all = []
    for images, names in tqdm(loader, desc="Predicao teste do modelo", leave=False):
        images = images.to(device, non_blocking=True)
        logits = predict_logits(model, images, tta=use_tta)
        probs = torch.softmax(logits, dim=1)
        probs_all.append(probs.cpu().numpy())
    return np.concatenate(probs_all, axis=0)


def save_val_outputs(
    out_dir: str,
    val_df: pd.DataFrame,
    probs: np.ndarray,
    class_names: List[str],
    config: Dict,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    y_true = val_df["label"].values.astype(int)
    y_pred = probs.argmax(axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }

    with open(os.path.join(out_dir, "ensemble_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "ensemble_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    pred_df = pd.DataFrame(probs, columns=[f"prob_{c}" for c in class_names])
    pred_df.insert(0, "pred_label", y_pred.astype(int))
    pred_df.insert(0, "label", y_true.astype(int))
    pred_df.insert(0, "image_id", val_df["image_id"].values)
    pred_df.to_csv(os.path.join(out_dir, "ensemble_val_predictions.csv"), index=False)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        os.path.join(out_dir, "ensemble_confusion_matrix.csv")
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(out_dir, "ensemble_classification_report.csv")
    )

    print("\nMetricas do ensemble:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nArquivos salvos em: {out_dir}")


def save_test_outputs(
    out_dir: str,
    test_df: pd.DataFrame,
    probs: np.ndarray,
    class_names: List[str],
    config: Dict,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    y_pred = probs.argmax(axis=1).astype(int)

    sub = pd.DataFrame({
        "image_id": test_df["image_id"].values,
        "label": y_pred,
    })
    sub_path = os.path.join(out_dir, "submission.csv")
    sub.to_csv(sub_path, index=False)

    prob_df = pd.DataFrame(probs, columns=[f"prob_{c}" for c in class_names])
    prob_df.insert(0, "pred_label", y_pred)
    prob_df.insert(0, "image_id", test_df["image_id"].values)
    prob_df.to_csv(os.path.join(out_dir, "ensemble_test_probabilities.csv"), index=False)

    with open(os.path.join(out_dir, "ensemble_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Submission salva em: {sub_path}")
    print(f"Probabilidades salvas em: {os.path.join(out_dir, 'ensemble_test_probabilities.csv')}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    weights = normalize_weights(args.weights, len(args.checkpoints))
    tta_flags = get_tta_modes(args)

    first_args = first_checkpoint_args(args.checkpoints[0], device)

    if args.split == "val":
        df = load_val_dataframe(args, first_args)
    else:
        df = load_test_dataframe(args.data_dir)

    probs_ensemble = None
    model_summaries = []
    class_names = DEFAULT_CLASS_NAMES

    for i, checkpoint_path in enumerate(args.checkpoints):
        print("\n" + "=" * 80)
        print(f"Modelo {i + 1}/{len(args.checkpoints)}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Peso no ensemble: {weights[i]:.4f}")
        print(f"TTA: {tta_flags[i]}")

        model, ckpt, ckpt_args = load_model_from_checkpoint(
            checkpoint_path,
            device=device,
            force_num_classes=args.force_num_classes,
        )

        model_name = ckpt.get("model_name") or ckpt_args.get("model", "resnet50")
        image_size = int(ckpt_args.get("image_size", 224))
        segmentation = bool(ckpt_args.get("segmentation", False))
        this_class_names = ckpt.get("class_names", DEFAULT_CLASS_NAMES)
        class_names = [str(c) for c in this_class_names]

        print(f"Modelo: {model_name}")
        print(f"Image size: {image_size}")
        print(f"Segmentacao: {segmentation}")

        loader = make_loader_for_model(df, args, ckpt_args, args.split)

        if args.split == "val":
            probs = predict_probs_val(model, loader, device, use_tta=tta_flags[i])
        else:
            probs = predict_probs_test(model, loader, device, use_tta=tta_flags[i])

        if probs_ensemble is None:
            probs_ensemble = weights[i] * probs
        else:
            probs_ensemble += weights[i] * probs

        model_summaries.append({
            "checkpoint": checkpoint_path,
            "weight": float(weights[i]),
            "tta": bool(tta_flags[i]),
            "model_name": model_name,
            "image_size": image_size,
            "segmentation": segmentation,
            "best_acc": float(ckpt.get("best_acc", -1.0)) if ckpt.get("best_acc", None) is not None else None,
            "best_macro_f1": float(ckpt.get("best_macro_f1", -1.0)) if ckpt.get("best_macro_f1", None) is not None else None,
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    config = {
        "split": args.split,
        "checkpoints": args.checkpoints,
        "weights": weights.tolist(),
        "tta_modes": ["tta" if f else "none" for f in tta_flags],
        "model_summaries": model_summaries,
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }

    if args.split == "val":
        if "label" not in df.columns:
            raise ValueError("Split val precisa ter coluna 'label'.")
        save_val_outputs(args.out_dir, df, probs_ensemble, class_names, config)
    else:
        save_test_outputs(args.out_dir, df, probs_ensemble, class_names, config)


if __name__ == "__main__":
    main()
