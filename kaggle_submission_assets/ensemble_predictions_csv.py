"""Ensemble a partir de CSVs de probabilidades.

Serve para combinar:
- ensemble PyTorch já calculado por evaluate_ensemble.py
- probabilidades do CropNet geradas por predict_cropnet_tfhub.py
- qualquer outro arquivo com image_id e prob_0...prob_4

Para validação, os CSVs precisam ter label ou pelo menos o primeiro CSV precisa ter label.
Para teste, gera submission.csv.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score


PROB_COLS = [f"prob_{i}" for i in range(5)]
CLASS_NAMES = ["0", "1", "2", "3", "4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble de CSVs de probabilidades")
    parser.add_argument("--prediction_csvs", nargs="+", required=True)
    parser.add_argument("--weights", nargs="*", type=float, default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--normalize_input_probs", action="store_true",
                        help="Renormaliza prob_0..prob_4 de cada CSV antes de combinar.")
    return parser.parse_args()


def normalize_weights(weights: List[float] | None, n: int) -> np.ndarray:
    if weights is None or len(weights) == 0:
        arr = np.ones(n, dtype=np.float64)
    else:
        arr = np.array(weights, dtype=np.float64)
        if len(arr) != n:
            raise ValueError(f"--weights precisa ter {n} valores. Recebido: {len(arr)}")
    if np.any(arr < 0):
        raise ValueError("Pesos não podem ser negativos.")
    if arr.sum() <= 0:
        raise ValueError("Soma dos pesos precisa ser positiva.")
    return arr / arr.sum()


def read_prediction_csv(path: str, normalize: bool) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV não encontrado: {path}")
    df = pd.read_csv(path)
    missing = [c for c in ["image_id"] + PROB_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {path} sem colunas obrigatórias: {missing}")
    df = df.copy()
    if normalize:
        probs = df[PROB_COLS].values.astype(np.float64)
        denom = np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        df.loc[:, PROB_COLS] = probs / denom
    return df


def merge_predictions(dfs: List[pd.DataFrame], weights: np.ndarray) -> pd.DataFrame:
    base = dfs[0][["image_id"]].copy()

    if "label" in dfs[0].columns:
        base["label"] = dfs[0]["label"].values.astype(int)

    probs_final = np.zeros((len(base), 5), dtype=np.float64)

    for idx, (df, weight) in enumerate(zip(dfs, weights)):
        cols = ["image_id"] + PROB_COLS
        if "label" in df.columns:
            cols.insert(1, "label")

        merged = base[["image_id"]].merge(df[cols], on="image_id", how="left", validate="one_to_one")
        if merged[PROB_COLS].isna().any().any():
            bad = merged.loc[merged[PROB_COLS].isna().any(axis=1), "image_id"].head(10).tolist()
            raise ValueError(f"CSV {idx} não contém probabilidades para algumas imagens. Exemplos: {bad}")

        probs_final += weight * merged[PROB_COLS].values.astype(np.float64)

        if "label" in df.columns and "label" in base.columns:
            label_merged = base[["image_id", "label"]].merge(
                df[["image_id", "label"]], on="image_id", how="left", suffixes=("_base", "_csv")
            )
            if not np.all(label_merged["label_base"].values == label_merged["label_csv"].values):
                raise ValueError(f"Labels inconsistentes no CSV {idx}: {df}")

    out = base.copy()
    for i, col in enumerate(PROB_COLS):
        out[col] = probs_final[:, i]
    out["pred_label"] = probs_final.argmax(axis=1).astype(int)
    return out


def save_outputs(out: pd.DataFrame, args: argparse.Namespace, weights: np.ndarray) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    config = {
        "prediction_csvs": args.prediction_csvs,
        "weights": weights.tolist(),
        "split": args.split,
        "normalize_input_probs": bool(args.normalize_input_probs),
    }
    with open(os.path.join(args.out_dir, "csv_ensemble_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    if args.split == "val":
        if "label" not in out.columns:
            raise ValueError("Para split=val, pelo menos o primeiro CSV precisa conter label.")
        y_true = out["label"].values.astype(int)
        y_pred = out["pred_label"].values.astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        }
        out_path = os.path.join(args.out_dir, "ensemble_val_predictions.csv")
        out.to_csv(out_path, index=False)
        with open(os.path.join(args.out_dir, "ensemble_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            os.path.join(args.out_dir, "ensemble_confusion_matrix.csv")
        )
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(5)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).transpose().to_csv(os.path.join(args.out_dir, "ensemble_classification_report.csv"))
        print("Métricas do ensemble de CSVs:")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"Predições salvas em: {out_path}")
    else:
        prob_path = os.path.join(args.out_dir, "ensemble_test_probabilities.csv")
        out.to_csv(prob_path, index=False)
        sub = out[["image_id", "pred_label"]].rename(columns={"pred_label": "label"})
        sub_path = os.path.join(args.out_dir, "submission.csv")
        sub.to_csv(sub_path, index=False)
        print(f"Probabilidades salvas em: {prob_path}")
        print(f"Submission salva em: {sub_path}")


def main() -> None:
    args = parse_args()
    weights = normalize_weights(args.weights, len(args.prediction_csvs))
    dfs = [read_prediction_csv(path, normalize=args.normalize_input_probs) for path in args.prediction_csvs]
    out = merge_predictions(dfs, weights)
    save_outputs(out, args, weights)


if __name__ == "__main__":
    main()
