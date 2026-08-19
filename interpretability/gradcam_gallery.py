#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gera mapas Grad-CAM e painéis para interpretabilidade.

Exemplo:
python interpretability/gradcam_gallery.py \
  --checkpoint "pesos/experimentos/efficientnet_b3_cutmix_label_smoothing_img300_segmentation/best_model.pth" \
  --data_dir "data/cassava_data" \
  --split val \
  --predictions_csv "pesos/experimentos/ensemble_direct_plus_cropnet_w040_060/ensemble_val_predictions.csv" \
  --out_dir "interpretability_outputs/gradcam_effb3_seg" \
  --groups correct_high_conf wrong_high_conf uncertain low_true_prob \
  --num_per_group 6 \
  --device cuda
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.model_factory import build_model
from utils.transforms import build_transforms

CLASS_NAMES = {0: "CBB", 1: "CBSD", 2: "CGM", 3: "CMD", 4: "Healthy"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "train", "test"])
    parser.add_argument("--split_csv", type=str, default=None)
    parser.add_argument("--predictions_csv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--groups", nargs="+", default=["correct_high_conf", "wrong_high_conf", "uncertain", "low_true_prob"],
        choices=["correct_high_conf", "wrong_high_conf", "uncertain", "low_true_prob", "per_class", "random"]
    )
    parser.add_argument("--num_per_group", type=int, default=6)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--segmentation", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--target_layer", type=str, default="auto")
    parser.add_argument("--target_class", type=str, default="pred", choices=["pred", "true"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def namespace_or_dict_to_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return {}


def load_checkpoint_and_args(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = namespace_or_dict_to_dict(ckpt.get("args", {})) if isinstance(ckpt, dict) else {}
    return ckpt, ckpt_args


def get_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                break
        else:
            sd = ckpt
    else:
        sd = ckpt
    clean = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean[k] = v
    return clean


def resolve_model_config(args, ckpt_args):
    model_name = args.model or ckpt_args.get("model", ckpt_args.get("model_name", "resnet50"))
    image_size = args.image_size or int(ckpt_args.get("image_size", 224))
    drop_rate = float(ckpt_args.get("drop_rate", 0.0))
    if args.segmentation == "auto":
        use_segmentation = bool(ckpt_args.get("segmentation", False))
    elif args.segmentation == "on":
        use_segmentation = True
    else:
        use_segmentation = False
    return model_name, image_size, drop_rate, use_segmentation


def build_and_load_model(args, ckpt, ckpt_args, device):
    model_name, image_size, drop_rate, use_segmentation = resolve_model_config(args, ckpt_args)
    model = build_model(model_name, num_classes=5, pretrained=False, image_size=image_size, drop_rate=drop_rate)
    state_dict = get_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Aviso] Chaves faltando: {len(missing)}", missing[:10])
    if unexpected:
        print(f"[Aviso] Chaves inesperadas: {len(unexpected)}", unexpected[:10])
    model.to(device)
    model.eval()
    return model, model_name, image_size, use_segmentation


def find_last_conv_layer(model: nn.Module) -> Tuple[str, nn.Module]:
    last_name, last_module = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name, last_module = name, module
    if last_module is None:
        raise RuntimeError("Não encontrei camada Conv2d para Grad-CAM.")
    return last_name, last_module


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        raise ValueError(f"Camada '{name}' não encontrada. Últimas camadas: {list(modules.keys())[-25:]}")
    return modules[name]


def choose_target_layer(model: nn.Module, model_name: str, target_layer: str) -> Tuple[str, nn.Module]:
    if target_layer != "auto":
        return target_layer, get_module_by_name(model, target_layer)
    name = model_name.lower()
    if "resnet" in name or "resnext" in name:
        if hasattr(model, "layer4"):
            return "layer4[-1]", model.layer4[-1]
        if hasattr(model, "model") and hasattr(model.model, "layer4"):
            return "model.layer4[-1]", model.model.layer4[-1]
    if "efficientnet" in name:
        if hasattr(model, "features"):
            return "features[-1]", model.features[-1]
        if hasattr(model, "model") and hasattr(model.model, "features"):
            return "model.features[-1]", model.model.features[-1]
    if "convnext" in name:
        if hasattr(model, "features"):
            return "features[-1]", model.features[-1]
        if hasattr(model, "model") and hasattr(model.model, "features"):
            return "model.features[-1]", model.model.features[-1]
    return find_last_conv_layer(model)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        for h in self.handles:
            h.remove()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        pred = int(probs.argmax(dim=1).item())
        if class_idx is None:
            class_idx = pred
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=False)
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks do Grad-CAM não capturaram ativações/gradientes.")
        acts = self.activations
        grads = self.gradients
        if acts.ndim != 4:
            raise RuntimeError(f"A camada escolhida retornou shape {tuple(acts.shape)}; use camada [B,C,H,W].")
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = F.relu(cam)[0].detach().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam, logits.detach(), probs.detach(), pred


def read_rgb_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (1 - alpha) * rgb.astype(np.float32) + alpha * heatmap.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def get_prob_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("prob_")]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def add_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    prob_cols = get_prob_columns(df)
    if len(prob_cols) >= 5:
        probs = df[prob_cols].to_numpy(dtype=float)
        df["pred_label"] = probs.argmax(axis=1)
        df["pred_confidence"] = probs.max(axis=1)
        df["entropy"] = -(probs * np.log(probs + 1e-12)).sum(axis=1)
        if "label" in df.columns:
            labels = df["label"].astype(int).to_numpy()
            df["true_label_probability"] = probs[np.arange(len(df)), labels]
            df["is_correct"] = df["pred_label"].astype(int) == df["label"].astype(int)
    else:
        if "pred_label" in df.columns and "label" in df.columns:
            df["is_correct"] = df["pred_label"].astype(int) == df["label"].astype(int)
        if "pred_confidence" not in df.columns:
            df["pred_confidence"] = np.nan
        if "entropy" not in df.columns:
            df["entropy"] = np.nan
        if "true_label_probability" not in df.columns:
            df["true_label_probability"] = np.nan
    return df


def load_split_dataframe(args, checkpoint_path: str) -> pd.DataFrame:
    data_dir = Path(args.data_dir)
    if args.split == "test":
        sample_path = data_dir / "sample_submission.csv"
        if sample_path.exists():
            return pd.read_csv(sample_path)
        return pd.DataFrame({"image_id": sorted(os.listdir(data_dir / "test_images"))})
    if args.split_csv:
        return pd.read_csv(args.split_csv)
    ckpt_parent = Path(checkpoint_path).resolve().parent
    candidate = ckpt_parent / f"{args.split}_split.csv"
    if candidate.exists():
        return pd.read_csv(candidate)
    candidate = data_dir / f"{args.split}_split.csv"
    if candidate.exists():
        return pd.read_csv(candidate)
    return pd.read_csv(data_dir / "train.csv")


def merge_predictions(split_df: pd.DataFrame, predictions_csv: Optional[str]) -> pd.DataFrame:
    df = split_df.copy()
    if predictions_csv is not None:
        pred_df = pd.read_csv(predictions_csv)
        if "image_id" not in pred_df.columns:
            raise ValueError("predictions_csv precisa ter coluna image_id.")
        df = df.merge(pred_df, on="image_id", how="inner", suffixes=("", "_pred"))
        if "label_pred" in df.columns and "label" not in df.columns:
            df["label"] = df["label_pred"]
    return add_prediction_columns(df)


def select_samples(df: pd.DataFrame, groups: List[str], n: int, seed: int) -> pd.DataFrame:
    selected = []
    for group in groups:
        tmp = df.copy()
        tmp["selection_group"] = group
        if group == "correct_high_conf":
            if "is_correct" in tmp.columns:
                tmp = tmp[tmp["is_correct"] == True]
            tmp = tmp.sort_values("pred_confidence", ascending=False, na_position="last").head(n)
        elif group == "wrong_high_conf":
            if "is_correct" in tmp.columns:
                tmp = tmp[tmp["is_correct"] == False]
            tmp = tmp.sort_values("pred_confidence", ascending=False, na_position="last").head(n)
        elif group == "uncertain":
            if "entropy" in tmp.columns:
                tmp = tmp.sort_values("entropy", ascending=False, na_position="last").head(n)
            else:
                tmp = tmp.sample(min(n, len(tmp)), random_state=seed)
        elif group == "low_true_prob":
            if "true_label_probability" in tmp.columns:
                tmp = tmp.sort_values("true_label_probability", ascending=True, na_position="last").head(n)
            else:
                tmp = tmp.sample(min(n, len(tmp)), random_state=seed)
        elif group == "per_class":
            parts = []
            if "label" in tmp.columns:
                for _, part in tmp.groupby("label"):
                    parts.append(part.sample(min(n, len(part)), random_state=seed))
                tmp = pd.concat(parts, ignore_index=True) if parts else tmp.head(0)
            else:
                tmp = tmp.sample(min(n, len(tmp)), random_state=seed)
        elif group == "random":
            tmp = tmp.sample(min(n, len(tmp)), random_state=seed)
        selected.append(tmp)
    out = pd.concat(selected, ignore_index=True) if selected else df.head(0)
    return out.drop_duplicates(subset=["image_id", "selection_group"]).reset_index(drop=True)


def format_title(row, pred_from_model: int, prob_from_model: float, target_class: int) -> str:
    img = row.get("image_id", "")
    group = row.get("selection_group", "")
    true_lab = row.get("label", None)
    pred_lab = row.get("pred_label", pred_from_model)
    ens_conf = row.get("pred_confidence", np.nan)
    true_text = f"true={int(true_lab)} {CLASS_NAMES.get(int(true_lab), '')}" if pd.notna(true_lab) else "true=?"
    pred_text = f"sel_pred={int(pred_lab)} {CLASS_NAMES.get(int(pred_lab), '')}" if pd.notna(pred_lab) else "sel_pred=?"
    model_text = f"model_pred={pred_from_model} {CLASS_NAMES.get(pred_from_model, '')} ({prob_from_model:.3f})"
    target_text = f"cam_target={target_class} {CLASS_NAMES.get(target_class, '')}"
    conf_text = f"sel_conf={float(ens_conf):.3f}" if pd.notna(ens_conf) else ""
    return f"{img}\n{group} | {true_text} | {pred_text} {conf_text}\n{model_text} | {target_text}"


def save_panel(original, overlay, cam, title, out_path, dpi=160):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), dpi=dpi)
    axes[0].imshow(original); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(cam, cmap="turbo"); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title("Sobreposição"); axes[2].axis("off")
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_contact_sheet(image_paths: List[Path], out_path: Path, cols=2, dpi=160):
    if not image_paths:
        return
    thumbs = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        thumbs.append((p.name, img))
    if not thumbs:
        return
    rows = math.ceil(len(thumbs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.2), dpi=dpi)
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (name, img) in zip(axes, thumbs):
        ax.imshow(img)
        ax.set_title(name.replace("_", " "), fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    print("Dispositivo:", device)
    ckpt, ckpt_args = load_checkpoint_and_args(args.checkpoint, device)
    model, model_name, image_size, use_segmentation = build_and_load_model(args, ckpt, ckpt_args, device)
    layer_name, target_layer = choose_target_layer(model, model_name, args.target_layer)
    print("Modelo:", model_name)
    print("Image size:", image_size)
    print("Segmentação:", use_segmentation)
    print("Camada Grad-CAM:", layer_name)
    transform = build_transforms(train=False, image_size=image_size, augmentation="none", use_segmentation=use_segmentation, frequency_aug=False)
    split_df = load_split_dataframe(args, args.checkpoint)
    df = merge_predictions(split_df, args.predictions_csv)
    samples = select_samples(df, args.groups, args.num_per_group, args.seed)
    if args.max_images is not None:
        samples = samples.head(args.max_images)
    samples_path = out_dir / "selected_samples.csv"
    samples.to_csv(samples_path, index=False)
    print(f"Amostras selecionadas: {len(samples)} -> {samples_path}")
    img_dir = Path(args.data_dir) / ("test_images" if args.split == "test" else "train_images")
    gradcam = GradCAM(model, target_layer)
    records, generated_panels = [], []
    for idx, row in samples.iterrows():
        image_id = row["image_id"]
        img_path = img_dir / image_id
        original = read_rgb_image(str(img_path))
        transformed = transform(image=original)
        x = transformed["image"].unsqueeze(0).to(device)
        with torch.enable_grad():
            logits0 = model(x)
            probs0 = F.softmax(logits0, dim=1)[0]
            model_pred = int(probs0.argmax().item())
        if args.target_class == "true" and "label" in row and pd.notna(row["label"]):
            target_class = int(row["label"])
        else:
            target_class = model_pred
        cam, logits, probs, pred = gradcam(x, class_idx=target_class)
        prob_target = float(probs[0, target_class].item())
        prob_pred = float(probs[0, pred].item())
        overlay = make_overlay(original, cam, alpha=args.alpha)
        safe_id = Path(image_id).stem
        group = row.get("selection_group", "sample")
        out_panel = out_dir / f"{group}_{idx:03d}_{safe_id}_panel.png"
        out_overlay = out_dir / f"{group}_{idx:03d}_{safe_id}_overlay.png"
        out_heat = out_dir / f"{group}_{idx:03d}_{safe_id}_heatmap.png"
        title = format_title(row, pred, prob_pred, target_class)
        save_panel(original, overlay, cv2.resize(cam, (original.shape[1], original.shape[0])), title, out_panel, dpi=args.dpi)
        cv2.imwrite(str(out_overlay), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        heat = np.uint8(255 * cv2.resize(cam, (original.shape[1], original.shape[0])))
        cv2.imwrite(str(out_heat), cv2.applyColorMap(heat, cv2.COLORMAP_TURBO))
        generated_panels.append(out_panel)
        records.append({
            "image_id": image_id,
            "selection_group": group,
            "true_label": int(row["label"]) if "label" in row and pd.notna(row["label"]) else None,
            "model_pred": pred,
            "model_pred_prob": prob_pred,
            "cam_target_class": target_class,
            "cam_target_prob": prob_target,
            "panel_path": str(out_panel),
            "overlay_path": str(out_overlay),
            "heatmap_path": str(out_heat),
        })
        print(f"[{idx+1}/{len(samples)}] {image_id} -> pred={pred}, target={target_class}, salvo: {out_panel}")
    gradcam.remove()
    pd.DataFrame(records).to_csv(out_dir / "gradcam_results.csv", index=False)
    make_contact_sheet(generated_panels, out_dir / "gradcam_contact_sheet.png", cols=2, dpi=args.dpi)
    config = vars(args).copy()
    config.update({"model_name": model_name, "image_size": image_size, "segmentation": use_segmentation, "target_layer_resolved": layer_name, "device": str(device)})
    with open(out_dir / "gradcam_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print("Concluído.")
    print("Pasta de saída:", out_dir)


if __name__ == "__main__":
    main()
