"""Métricas e salvamento de resultados."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(y_true, y_pred, labels=None, target_names=None):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return metrics, report, cm


def save_eval_outputs(out_dir, y_true, y_pred, pred_probs=None, image_ids=None, class_names=None, prefix="val"):
    os.makedirs(out_dir, exist_ok=True)
    labels = list(range(len(class_names))) if class_names is not None else None

    metrics, report, cm = compute_metrics(y_true, y_pred, labels=labels, target_names=class_names)

    with open(os.path.join(out_dir, f"{prefix}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, f"{prefix}_classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names) if class_names else pd.DataFrame(cm)
    cm_df.to_csv(os.path.join(out_dir, f"{prefix}_confusion_matrix.csv"))

    pred_df = pd.DataFrame({
        "image_id": image_ids if image_ids is not None else np.arange(len(y_pred)),
        "y_true": y_true,
        "y_pred": y_pred,
    })

    if pred_probs is not None:
        for c in range(pred_probs.shape[1]):
            name = class_names[c] if class_names else str(c)
            pred_df[f"prob_{name}"] = pred_probs[:, c]

    pred_df.to_csv(os.path.join(out_dir, f"{prefix}_predictions.csv"), index=False)
    return metrics, report, cm
