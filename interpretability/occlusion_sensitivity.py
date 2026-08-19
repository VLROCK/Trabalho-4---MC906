#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gera mapas de sensibilidade por oclusão.
Mascara regiões da imagem e mede quanto cai a probabilidade da classe alvo.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
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
    parser.add_argument("--group", type=str, default="wrong_high_conf", choices=["correct_high_conf", "wrong_high_conf", "uncertain", "low_true_prob", "random"])
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--target_class", type=str, default="pred", choices=["pred", "true"])
    parser.add_argument("--patch_size", type=int, default=48)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--mask_value", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--segmentation", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def namespace_or_dict_to_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return {}


def load_checkpoint(checkpoint_path: str, device):
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


def resolve_config(args, ckpt_args):
    model_name = args.model or ckpt_args.get("model", ckpt_args.get("model_name", "resnet50"))
    image_size = args.image_size or int(ckpt_args.get("image_size", 224))
    drop_rate = float(ckpt_args.get("drop_rate", 0.0))
    if args.segmentation == "auto":
        segmentation = bool(ckpt_args.get("segmentation", False))
    elif args.segmentation == "on":
        segmentation = True
    else:
        segmentation = False
    return model_name, image_size, drop_rate, segmentation


def build_and_load_model(args, ckpt, ckpt_args, device):
    model_name, image_size, drop_rate, segmentation = resolve_config(args, ckpt_args)
    model = build_model(model_name, num_classes=5, pretrained=False, image_size=image_size, drop_rate=drop_rate)
    missing, unexpected = model.load_state_dict(get_state_dict(ckpt), strict=False)
    if missing:
        print(f"[Aviso] Chaves faltando: {len(missing)}", missing[:10])
    if unexpected:
        print(f"[Aviso] Chaves inesperadas: {len(unexpected)}", unexpected[:10])
    model.to(device)
    model.eval()
    return model, model_name, image_size, segmentation


def read_rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_prob_columns(df):
    cols = [c for c in df.columns if c.startswith("prob_")]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def add_prediction_columns(df):
    df = df.copy()
    prob_cols = get_prob_columns(df)
    if len(prob_cols) >= 5:
        probs = df[prob_cols].values.astype(float)
        df["pred_label"] = probs.argmax(axis=1)
        df["pred_confidence"] = probs.max(axis=1)
        df["entropy"] = -(probs * np.log(probs + 1e-12)).sum(axis=1)
        if "label" in df.columns:
            labels = df["label"].astype(int).values
            df["true_label_probability"] = probs[np.arange(len(df)), labels]
            df["is_correct"] = df["pred_label"].astype(int) == df["label"].astype(int)
    else:
        if "pred_label" in df.columns and "label" in df.columns:
            df["is_correct"] = df["pred_label"].astype(int) == df["label"].astype(int)
    return df


def load_split_dataframe(args):
    data_dir = Path(args.data_dir)
    if args.split == "test":
        sample_path = data_dir / "sample_submission.csv"
        if sample_path.exists():
            return pd.read_csv(sample_path)
        return pd.DataFrame({"image_id": sorted(os.listdir(data_dir / "test_images"))})
    if args.split_csv:
        return pd.read_csv(args.split_csv)
    ckpt_parent = Path(args.checkpoint).resolve().parent
    candidate = ckpt_parent / f"{args.split}_split.csv"
    if candidate.exists():
        return pd.read_csv(candidate)
    candidate = data_dir / f"{args.split}_split.csv"
    if candidate.exists():
        return pd.read_csv(candidate)
    return pd.read_csv(data_dir / "train.csv")


def merge_predictions(split_df, predictions_csv):
    if predictions_csv is None:
        return add_prediction_columns(split_df)
    pred_df = pd.read_csv(predictions_csv)
    df = split_df.merge(pred_df, on="image_id", how="inner", suffixes=("", "_pred"))
    if "label_pred" in df.columns and "label" not in df.columns:
        df["label"] = df["label_pred"]
    return add_prediction_columns(df)


def select_samples(df, group, n, seed):
    tmp = df.copy()
    if group == "correct_high_conf":
        if "is_correct" in tmp.columns:
            tmp = tmp[tmp["is_correct"] == True]
        return tmp.sort_values("pred_confidence", ascending=False, na_position="last").head(n)
    if group == "wrong_high_conf":
        if "is_correct" in tmp.columns:
            tmp = tmp[tmp["is_correct"] == False]
        return tmp.sort_values("pred_confidence", ascending=False, na_position="last").head(n)
    if group == "uncertain":
        if "entropy" in tmp.columns:
            return tmp.sort_values("entropy", ascending=False, na_position="last").head(n)
        return tmp.sample(min(n, len(tmp)), random_state=seed)
    if group == "low_true_prob":
        if "true_label_probability" in tmp.columns:
            return tmp.sort_values("true_label_probability", ascending=True, na_position="last").head(n)
        return tmp.sample(min(n, len(tmp)), random_state=seed)
    return tmp.sample(min(n, len(tmp)), random_state=seed)


@torch.no_grad()
def predict_prob(model, x, target_class):
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    pred = int(probs.argmax(dim=1).item())
    return float(probs[0, target_class].item()), pred, probs[0].detach().cpu().numpy()


@torch.no_grad()
def occlusion_map(model, x, target_class, patch_size, stride, mask_value, batch_size, device):
    _, c, h, w = x.shape
    base_prob, pred, probs = predict_prob(model, x, target_class)
    positions = []
    variants = []
    for y in range(0, h, stride):
        for x0 in range(0, w, stride):
            y1 = min(y + patch_size, h)
            x1 = min(x0 + patch_size, w)
            x_occ = x.clone()
            x_occ[:, :, y:y1, x0:x1] = mask_value
            variants.append(x_occ.cpu())
            positions.append((y, y1, x0, x1))
    drops = []
    for i in range(0, len(variants), batch_size):
        batch = torch.cat(variants[i:i + batch_size], dim=0).to(device)
        logits = model(batch)
        probs_batch = F.softmax(logits, dim=1)[:, target_class]
        drop = base_prob - probs_batch.detach().cpu().numpy()
        drops.extend(drop.tolist())
    heat = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    for (y, y1, x0, x1), drop in zip(positions, drops):
        heat[y:y1, x0:x1] += max(0.0, drop)
        count[y:y1, x0:x1] += 1.0
    heat = heat / np.maximum(count, 1e-6)
    if heat.max() > heat.min():
        heat = (heat - heat.min()) / (heat.max() - heat.min())
    else:
        heat = np.zeros_like(heat)
    return heat, base_prob, pred, probs


def make_overlay(original, heat, alpha=0.45):
    h, w = original.shape[:2]
    heat_resized = cv2.resize(heat, (w, h))
    heat_u8 = np.uint8(255 * heat_resized)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    overlay = (1 - alpha) * original.astype(np.float32) + alpha * color.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8), heat_resized


def save_panel(original, heat, overlay, title, out_path, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), dpi=dpi)
    axes[0].imshow(original); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(heat, cmap="turbo"); axes[1].set_title("Occlusion sensitivity"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title("Sobreposição"); axes[2].axis("off")
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt, ckpt_args = load_checkpoint(args.checkpoint, device)
    model, model_name, image_size, segmentation = build_and_load_model(args, ckpt, ckpt_args, device)
    print("Dispositivo:", device)
    print("Modelo:", model_name)
    print("Image size:", image_size)
    print("Segmentação:", segmentation)
    transform = build_transforms(train=False, image_size=image_size, augmentation="none", use_segmentation=segmentation, frequency_aug=False)
    split_df = load_split_dataframe(args)
    df = merge_predictions(split_df, args.predictions_csv)
    samples = select_samples(df, args.group, args.num_images, args.seed).reset_index(drop=True)
    samples.to_csv(out_dir / "selected_samples.csv", index=False)
    img_dir = Path(args.data_dir) / ("test_images" if args.split == "test" else "train_images")
    records = []
    for i, row in samples.iterrows():
        image_id = row["image_id"]
        original = read_rgb(img_dir / image_id)
        transformed = transform(image=original)
        x = transformed["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]
            pred = int(probs.argmax().item())
        if args.target_class == "true" and "label" in row and pd.notna(row["label"]):
            target = int(row["label"])
        else:
            target = pred
        heat, base_prob, pred2, full_probs = occlusion_map(model, x, target, args.patch_size, args.stride, args.mask_value, args.batch_size, device)
        overlay, heat_resized = make_overlay(original, heat, alpha=0.45)
        safe_id = Path(image_id).stem
        out_panel = out_dir / f"{args.group}_{i:03d}_{safe_id}_occlusion_panel.png"
        out_overlay = out_dir / f"{args.group}_{i:03d}_{safe_id}_occlusion_overlay.png"
        out_heat = out_dir / f"{args.group}_{i:03d}_{safe_id}_occlusion_heatmap.png"
        true_label = int(row["label"]) if "label" in row and pd.notna(row["label"]) else None
        title = f"{image_id}\ngroup={args.group} | true={true_label} | model_pred={pred} ({float(probs[pred]):.3f}) | target={target} ({base_prob:.3f})"
        save_panel(original, heat_resized, overlay, title, out_panel, args.dpi)
        cv2.imwrite(str(out_overlay), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        heat_u8 = np.uint8(255 * heat_resized)
        cv2.imwrite(str(out_heat), cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO))
        records.append({"image_id": image_id, "group": args.group, "true_label": true_label, "model_pred": pred, "target_class": target, "target_base_prob": base_prob, "panel_path": str(out_panel), "overlay_path": str(out_overlay), "heatmap_path": str(out_heat)})
        print(f"[{i+1}/{len(samples)}] {image_id} salvo em {out_panel}")
    pd.DataFrame(records).to_csv(out_dir / "occlusion_results.csv", index=False)
    with open(out_dir / "occlusion_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print("Concluído:", out_dir)


if __name__ == "__main__":
    main()
