"""Confident Learning / deteccao de labels suspeitas.

Entrada esperada: CSV com image_id,label,pred_label,prob_0,...,prob_4.
Funciona com ensemble_val_predictions.csv do evaluate_ensemble.py ou com
predictions salvas pelo evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detecta possiveis labels erradas via probabilidades previstas.")

    parser.add_argument("--predictions_csv", type=str, required=True,
                        help="CSV com image_id, label e colunas prob_0...prob_4.")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--method", type=str, choices=["simple", "cleanlab", "both"], default="both")

    parser.add_argument("--only_misclassified", action="store_true",
                        help="Mantem apenas exemplos em que pred_label != label.")
    parser.add_argument("--max_self_confidence", type=float, default=None,
                        help="Mantem apenas exemplos com probabilidade da label original <= esse valor.")

    return parser.parse_args()


def get_prob_columns(df: pd.DataFrame) -> List[str]:
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if len(prob_cols) == 0:
        raise ValueError("Nao encontrei colunas prob_0, prob_1, ..., prob_K no CSV.")
    return sorted(prob_cols, key=lambda x: int(x.split("_")[1]))


def validate_predictions_df(df: pd.DataFrame, prob_cols: List[str]) -> None:
    required = {"image_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV precisa conter colunas {required}. Faltando: {missing}")

    labels = df["label"].values.astype(int)
    num_classes = len(prob_cols)
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(
            f"Labels fora do intervalo [0, {num_classes - 1}]. Labels unicas: {sorted(np.unique(labels).tolist())}"
        )


def add_common_scores(df: pd.DataFrame, prob_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    probs = out[prob_cols].values.astype(np.float64)
    labels = out["label"].values.astype(int)

    pred_labels = probs.argmax(axis=1)
    pred_confidence = probs.max(axis=1)
    self_confidence = probs[np.arange(len(out)), labels]

    probs_without_true = probs.copy()
    probs_without_true[np.arange(len(out)), labels] = -np.inf
    max_other_prob = probs_without_true.max(axis=1)

    normalized_margin = self_confidence - max_other_prob
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)

    out["pred_label"] = pred_labels.astype(int)
    out["pred_confidence"] = pred_confidence
    out["self_confidence"] = self_confidence
    out["true_label_probability"] = self_confidence
    out["max_other_probability"] = max_other_prob
    out["normalized_margin"] = normalized_margin
    out["entropy"] = entropy
    out["is_misclassified"] = pred_labels != labels

    return out


def apply_optional_filters(df: pd.DataFrame, only_misclassified: bool, max_self_confidence: float | None) -> pd.DataFrame:
    out = df.copy()
    if only_misclassified:
        out = out[out["is_misclassified"]].copy()
    if max_self_confidence is not None:
        out = out[out["self_confidence"] <= max_self_confidence].copy()
    return out


def simple_ranking(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.sort_values(
        by=["self_confidence", "pred_confidence", "normalized_margin"],
        ascending=[True, False, True],
    ).copy()
    ranked["simple_rank"] = np.arange(len(ranked))
    return ranked


def cleanlab_ranking(df: pd.DataFrame, prob_cols: List[str]) -> pd.DataFrame:
    try:
        from cleanlab.filter import find_label_issues
    except ImportError as exc:
        raise ImportError("cleanlab nao esta instalado. Instale com: pip install cleanlab") from exc

    probs = df[prob_cols].values.astype(np.float64)
    labels = df["label"].values.astype(int)

    ranked_indices = find_label_issues(
        labels=labels,
        pred_probs=probs,
        return_indices_ranked_by="self_confidence",
    )

    ranked = df.iloc[ranked_indices].copy()
    ranked["cleanlab_rank"] = np.arange(len(ranked))
    return ranked


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.predictions_csv)
    prob_cols = get_prob_columns(df)
    validate_predictions_df(df, prob_cols)

    scored = add_common_scores(df, prob_cols)
    scored = apply_optional_filters(
        scored,
        only_misclassified=args.only_misclassified,
        max_self_confidence=args.max_self_confidence,
    )

    scored_path = os.path.join(args.out_dir, "all_scored_predictions.csv")
    scored.to_csv(scored_path, index=False)

    summary = {
        "predictions_csv": args.predictions_csv,
        "num_examples_input": int(len(df)),
        "num_examples_after_filters": int(len(scored)),
        "prob_cols": prob_cols,
        "top_k": int(args.top_k),
        "method": args.method,
        "only_misclassified": bool(args.only_misclassified),
        "max_self_confidence": args.max_self_confidence,
    }

    if args.method in {"simple", "both"}:
        simple = simple_ranking(scored)
        simple_top = simple.head(args.top_k)
        simple_path = os.path.join(args.out_dir, "label_issues_simple.csv")
        simple_top.to_csv(simple_path, index=False)
        summary["simple_output"] = simple_path
        summary["simple_top_k"] = int(len(simple_top))

        print("\nTop suspeitas pelo metodo simples:")
        cols = [
            "image_id", "label", "pred_label", "pred_confidence",
            "self_confidence", "max_other_probability", "normalized_margin",
            "is_misclassified",
        ]
        print(simple_top[cols].head(20).to_string(index=False))

    if args.method in {"cleanlab", "both"}:
        try:
            cleanlab_ranked = cleanlab_ranking(scored, prob_cols)
            cleanlab_top = cleanlab_ranked.head(args.top_k)
            cleanlab_path = os.path.join(args.out_dir, "label_issues_cleanlab.csv")
            cleanlab_top.to_csv(cleanlab_path, index=False)
            summary["cleanlab_output"] = cleanlab_path
            summary["cleanlab_top_k"] = int(len(cleanlab_top))

            print("\nTop suspeitas pelo Cleanlab:")
            cols = [
                "image_id", "label", "pred_label", "pred_confidence",
                "self_confidence", "max_other_probability", "normalized_margin",
                "is_misclassified",
            ]
            print(cleanlab_top[cols].head(20).to_string(index=False))
        except ImportError as exc:
            summary["cleanlab_error"] = str(exc)
            print(f"\nAviso: {exc}")
            print("Continuando apenas com o metodo simples.")

    summary_path = os.path.join(args.out_dir, "confident_learning_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nScores completos salvos em: {scored_path}")
    print(f"Resumo salvo em: {summary_path}")


if __name__ == "__main__":
    main()
