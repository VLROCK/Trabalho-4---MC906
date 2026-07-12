"""Utilidades compartilhadas: seed, config, split, métricas, embeddings, checkpoint."""

from __future__ import annotations

import json
import os
import random

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


# ----------------------------- básico -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stratified_split(df, val_size: float, seed: int):
    """Split treino/val estratificado por 'label' (mesma semente em todo o projeto)."""
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=seed, stratify=df["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ----------------------------- métricas -----------------------------

def compute_metrics(y_true, y_pred, class_names=None):
    labels = list(range(len(class_names))) if class_names else None
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return metrics, report, cm


def save_eval_outputs(out_dir, y_true, y_pred, pred_probs, image_ids, class_names, prefix):
    """Salva métricas, classification report, matriz de confusão e predições."""
    os.makedirs(out_dir, exist_ok=True)
    metrics, report, cm = compute_metrics(y_true, y_pred, class_names)

    with open(os.path.join(out_dir, f"{prefix}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, f"{prefix}_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        os.path.join(out_dir, f"{prefix}_confusion_matrix.csv")
    )

    pred_df = pd.DataFrame({"image_id": image_ids, "y_true": y_true, "y_pred": y_pred})
    for c, name in enumerate(class_names):
        pred_df[f"prob_{name}"] = pred_probs[:, c]
    pred_df.to_csv(os.path.join(out_dir, f"{prefix}_predictions.csv"), index=False)

    return metrics


# ----------------------------- checkpoint -----------------------------

def save_checkpoint(path, model, optimizer, epoch, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


# ----------------------------- embeddings -----------------------------

@torch.no_grad()
def extract_embeddings(backbone, loader, device):
    """Extrai embeddings (features do backbone) para um loader que retorna
    (image, label, image_id). Retorna (embeddings, labels, image_ids)."""
    backbone.eval()
    embs, labels, ids = [], [], []
    for images, y, names in loader:
        images = images.to(device, non_blocking=True)
        feats = backbone(images)
        embs.append(feats.cpu().numpy())
        labels.extend(y.numpy().tolist())
        ids.extend(list(names))
    return np.concatenate(embs, axis=0), np.array(labels), np.array(ids)
