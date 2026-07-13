#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Gera painéis de imagens originais a partir de um CSV de predições."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

CLASS_NAMES = {0: "CBB", 1: "CBSD", 2: "CGM", 3: "CMD", 4: "Healthy"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_csv", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "train", "test"])
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_per_group", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def get_prob_cols(df):
    cols = [c for c in df.columns if c.startswith("prob_")]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def enrich(df):
    df = df.copy()
    prob_cols = get_prob_cols(df)
    if len(prob_cols) >= 5:
        probs = df[prob_cols].values.astype(float)
        df["pred_label"] = probs.argmax(axis=1)
        df["pred_confidence"] = probs.max(axis=1)
        df["entropy"] = -(probs * np.log(probs + 1e-12)).sum(axis=1)
        if "label" in df.columns:
            labels = df["label"].astype(int).values
            df["true_label_probability"] = probs[np.arange(len(df)), labels]
            df["is_correct"] = df["pred_label"].astype(int) == df["label"].astype(int)
    return df


def read_rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def select_groups(df, n):
    groups = {}
    if "is_correct" in df.columns:
        groups["correct_high_conf"] = df[df["is_correct"]].sort_values("pred_confidence", ascending=False).head(n)
        groups["wrong_high_conf"] = df[~df["is_correct"]].sort_values("pred_confidence", ascending=False).head(n)
    if "entropy" in df.columns:
        groups["uncertain_high_entropy"] = df.sort_values("entropy", ascending=False).head(n)
    if "true_label_probability" in df.columns:
        groups["low_true_label_probability"] = df.sort_values("true_label_probability", ascending=True).head(n)
    if "label" in df.columns:
        parts = []
        per_class_n = max(1, n // 5)
        for _, part in df.groupby("label"):
            parts.append(part.sample(min(per_class_n, len(part)), random_state=42))
        groups["random_per_class"] = pd.concat(parts, ignore_index=True)
    return groups


def save_gallery(group_name, group_df, img_dir, out_path, dpi):
    if len(group_df) == 0:
        return
    cols = 4
    rows = math.ceil(len(group_df) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.5), dpi=dpi)
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (_, row) in zip(axes, group_df.iterrows()):
        img = read_rgb(img_dir / row["image_id"])
        ax.imshow(img)
        true_lab = row.get("label", None)
        pred_lab = row.get("pred_label", None)
        conf = row.get("pred_confidence", np.nan)
        ent = row.get("entropy", np.nan)
        title = f"{row['image_id']}\n"
        if pd.notna(true_lab):
            title += f"T={int(true_lab)} {CLASS_NAMES.get(int(true_lab),'')} | "
        if pd.notna(pred_lab):
            title += f"P={int(pred_lab)} {CLASS_NAMES.get(int(pred_lab),'')}"
        if pd.notna(conf):
            title += f"\nconf={float(conf):.3f}"
        if pd.notna(ent):
            title += f" ent={float(ent):.2f}"
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    fig.suptitle(group_name.replace("_", " "), fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions_csv)
    df = enrich(df)
    img_dir = Path(args.data_dir) / ("test_images" if args.split == "test" else "train_images")
    groups = select_groups(df, args.num_per_group)
    summary = {}
    for name, gdf in groups.items():
        gdf.to_csv(out_dir / f"{name}.csv", index=False)
        save_gallery(name, gdf, img_dir, out_dir / f"{name}.png", args.dpi)
        summary[name] = len(gdf)
    with open(out_dir / "gallery_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Galerias salvas em:", out_dir)
    print(summary)


if __name__ == "__main__":
    main()
