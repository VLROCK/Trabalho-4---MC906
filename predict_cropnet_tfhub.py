"""Gera probabilidades do CropNet para val/test do Cassava.

O CropNet do TensorFlow Hub retorna 6 classes:
    0 cbb      -> Kaggle 0
    1 cbsd     -> Kaggle 1
    2 cgm      -> Kaggle 2
    3 cmd      -> Kaggle 3
    4 healthy  -> Kaggle 4
    5 unknown  -> sem classe no Kaggle 2020

Para entrar no ensemble de 5 classes, este script usa as cinco primeiras
probabilidades e, por padrão, renormaliza a soma para 1.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


CROPNET_HANDLE = "https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2"
CROPNET_CLASSES_6 = ["cbb", "cbsd", "cgm", "cmd", "healthy", "unknown"]
KAGGLE_CLASS_NAMES = ["0", "1", "2", "3", "4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predição CropNet TF Hub para Cassava")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--handle", type=str, default=CROPNET_HANDLE,
                        help="Handle TF Hub ou caminho local para um SavedModel/KerasLayer compatível.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Opcional: TFHUB_CACHE_DIR. Útil no cluster.")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Reservado para compatibilidade. TensorFlow usa pipeline interno simples aqui.")

    parser.add_argument("--val_csv", type=str, default=None,
                        help="CSV explícito do split de validação. Deve ter image_id,label.")
    parser.add_argument("--reference_checkpoint", type=str, default=None,
                        help="Usa val_split.csv na pasta do checkpoint informado.")
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--center_crop", action="store_true",
                        help="Aplica crop quadrado central antes de redimensionar para 224.")
    parser.add_argument("--crop_tta", action="store_true",
                        help="Usa crops/flip simples e média das probabilidades. Mais lento.")
    parser.add_argument("--no_renormalize_5class", action="store_true",
                        help="Não renormaliza as 5 classes após remover unknown.")

    return parser.parse_args()


def load_val_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if args.val_csv is not None:
        if not os.path.exists(args.val_csv):
            raise FileNotFoundError(f"--val_csv não encontrado: {args.val_csv}")
        return pd.read_csv(args.val_csv).reset_index(drop=True)

    if args.reference_checkpoint is not None:
        ckpt_dir = Path(args.reference_checkpoint).resolve().parent
        candidate = ckpt_dir / "val_split.csv"
        if candidate.exists():
            print(f"Usando val_split.csv do checkpoint de referência: {candidate}")
            return pd.read_csv(candidate).reset_index(drop=True)
        print(f"Aviso: não encontrei {candidate}; vou recriar split via train_test_split.")

    csv_path = os.path.join(args.data_dir, "train.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"train.csv não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    _, val_df = train_test_split(
        df,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=df["label"],
    )
    return val_df.reset_index(drop=True)


def load_test_dataframe(data_dir: str) -> pd.DataFrame:
    sample_path = os.path.join(data_dir, "sample_submission.csv")
    test_dir = os.path.join(data_dir, "test_images")

    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)[["image_id"]].copy().reset_index(drop=True)

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"test_images não encontrado: {test_dir}")

    files = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    return pd.DataFrame({"image_id": files})


def read_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Imagem não encontrada ou inválida: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def center_square_crop(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return image[y0:y0 + side, x0:x0 + side]


def five_crops(image: np.ndarray, crop_ratio: float = 0.875) -> List[np.ndarray]:
    """Crops quadrados: centro + 4 cantos.

    Para imagens 800x600, primeiro pega um crop quadrado máximo 600x600.
    Depois faz crops internos de ~87.5% para simular uma TTA leve.
    """
    image = center_square_crop(image)
    h, w = image.shape[:2]
    side = int(min(h, w) * crop_ratio)
    side = max(1, side)

    positions = [
        (0, 0),
        (0, w - side),
        (h - side, 0),
        (h - side, w - side),
        ((h - side) // 2, (w - side) // 2),
    ]

    crops = []
    for y0, x0 in positions:
        crops.append(image[y0:y0 + side, x0:x0 + side])
    return crops


def preprocess_single(image: np.ndarray, center_crop: bool = True) -> np.ndarray:
    if center_crop:
        image = center_square_crop(image)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    return image


def build_views(image: np.ndarray, center_crop: bool, crop_tta: bool) -> List[np.ndarray]:
    if not crop_tta:
        return [preprocess_single(image, center_crop=center_crop)]

    views: List[np.ndarray] = []
    for crop in five_crops(image):
        base = preprocess_single(crop, center_crop=False)
        views.append(base)
        views.append(np.ascontiguousarray(base[:, ::-1, :]))  # horizontal flip
    return views


def iter_batches(items: List[Tuple[str, List[np.ndarray]]], batch_size: int) -> Iterable[Tuple[List[str], np.ndarray, List[int]]]:
    """Agrupa views em batches, preservando a qual image_id cada view pertence."""
    names: List[str] = []
    arrays: List[np.ndarray] = []
    counts: List[int] = []

    flat_names: List[str] = []
    flat_views: List[np.ndarray] = []
    for image_id, views in items:
        for view in views:
            flat_names.append(image_id)
            flat_views.append(view)

    for start in range(0, len(flat_views), batch_size):
        end = min(start + batch_size, len(flat_views))
        yield flat_names[start:end], np.stack(flat_views[start:end], axis=0), []


def convert_6_to_5(probs6: np.ndarray, renormalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    probs5 = probs6[:, :5].astype(np.float64)
    unknown = probs6[:, 5].astype(np.float64)

    if renormalize:
        denom = probs5.sum(axis=1, keepdims=True)
        denom = np.clip(denom, 1e-12, None)
        probs5 = probs5 / denom

    return probs5.astype(np.float32), unknown.astype(np.float32)


def predict_cropnet(args: argparse.Namespace, df: pd.DataFrame, img_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        os.environ["TFHUB_CACHE_DIR"] = args.cache_dir

    import tensorflow as tf
    import tensorflow_hub as hub

    print(f"Carregando CropNet: {args.handle}")
    classifier = hub.KerasLayer(args.handle, trainable=False)

    image_ids = df["image_id"].tolist()
    summed: dict[str, np.ndarray] = {img_id: np.zeros(6, dtype=np.float64) for img_id in image_ids}
    counts: dict[str, int] = {img_id: 0 for img_id in image_ids}

    current_names: List[str] = []
    current_batch: List[np.ndarray] = []

    def flush_batch() -> None:
        nonlocal current_names, current_batch
        if not current_batch:
            return
        batch = np.stack(current_batch, axis=0).astype(np.float32)
        probs = classifier(tf.convert_to_tensor(batch, dtype=tf.float32)).numpy()
        for name, p in zip(current_names, probs):
            summed[name] += p.astype(np.float64)
            counts[name] += 1
        current_names = []
        current_batch = []

    for image_id in tqdm(image_ids, desc="CropNet inferência"):
        path = os.path.join(img_dir, image_id)
        image = read_rgb(path)
        views = build_views(image, center_crop=args.center_crop or args.crop_tta, crop_tta=args.crop_tta)

        for view in views:
            current_names.append(image_id)
            current_batch.append(view)
            if len(current_batch) >= args.batch_size:
                flush_batch()

    flush_batch()

    probs6 = []
    for image_id in image_ids:
        if counts[image_id] == 0:
            raise RuntimeError(f"Nenhuma view gerada para {image_id}")
        probs6.append(summed[image_id] / counts[image_id])

    probs6_arr = np.stack(probs6, axis=0).astype(np.float32)
    probs5, unknown = convert_6_to_5(probs6_arr, renormalize=not args.no_renormalize_5class)
    return probs5, unknown


def save_outputs(args: argparse.Namespace, df: pd.DataFrame, probs5: np.ndarray, unknown: np.ndarray) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    pred = probs5.argmax(axis=1).astype(int)
    prefix = "cropnet_val" if args.split == "val" else "cropnet_test"

    out_df = pd.DataFrame(probs5, columns=[f"prob_{i}" for i in range(5)])
    out_df.insert(0, "cropnet_unknown_prob", unknown)
    out_df.insert(0, "pred_label", pred)
    if "label" in df.columns:
        out_df.insert(0, "label", df["label"].values.astype(int))
    out_df.insert(0, "image_id", df["image_id"].values)

    pred_path = os.path.join(args.out_dir, f"{prefix}_predictions.csv")
    out_df.to_csv(pred_path, index=False)

    config = {
        "handle": args.handle,
        "split": args.split,
        "center_crop": bool(args.center_crop),
        "crop_tta": bool(args.crop_tta),
        "renormalize_5class": not bool(args.no_renormalize_5class),
        "batch_size": args.batch_size,
        "classes_6": CROPNET_CLASSES_6,
        "kaggle_mapping": {str(i): i for i in range(5)},
    }
    with open(os.path.join(args.out_dir, "cropnet_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Predições CropNet salvas em: {pred_path}")

    if args.split == "val":
        y_true = df["label"].values.astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, average="macro")),
            "mean_unknown_prob": float(np.mean(unknown)),
            "median_unknown_prob": float(np.median(unknown)),
        }
        with open(os.path.join(args.out_dir, "cropnet_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        cm = confusion_matrix(y_true, pred, labels=list(range(5)))
        pd.DataFrame(cm, index=KAGGLE_CLASS_NAMES, columns=KAGGLE_CLASS_NAMES).to_csv(
            os.path.join(args.out_dir, "cropnet_confusion_matrix.csv")
        )
        report = classification_report(
            y_true,
            pred,
            labels=list(range(5)),
            target_names=KAGGLE_CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).transpose().to_csv(os.path.join(args.out_dir, "cropnet_classification_report.csv"))

        print("Métricas CropNet:")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    else:
        sub = pd.DataFrame({"image_id": df["image_id"].values, "label": pred})
        sub_path = os.path.join(args.out_dir, "submission_cropnet.csv")
        sub.to_csv(sub_path, index=False)
        print(f"Submission só com CropNet salva em: {sub_path}")


def main() -> None:
    args = parse_args()

    if args.split == "val":
        df = load_val_dataframe(args)
        img_dir = os.path.join(args.data_dir, "train_images")
    else:
        df = load_test_dataframe(args.data_dir)
        img_dir = os.path.join(args.data_dir, "test_images")

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {img_dir}")

    probs5, unknown = predict_cropnet(args, df, img_dir)
    save_outputs(args, df, probs5, unknown)


if __name__ == "__main__":
    main()
