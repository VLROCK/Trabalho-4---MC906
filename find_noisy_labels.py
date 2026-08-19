"""Detecção simples de labels suspeitas com Cleanlab.

Importante: idealmente, Confident Learning deve usar probabilidades out-of-fold.
Este script usa o split de validação do checkpoint, então serve como versão inicial
para análise de rótulos suspeitos, não como prova definitiva.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

try:
    from cleanlab.filter import find_label_issues
except ImportError as exc:
    raise ImportError("Instale cleanlab: pip install cleanlab") from exc

from evaluate import main as evaluate_main  # apenas para lembrar dependência; não usado diretamente


def parse_args():
    parser = argparse.ArgumentParser(description="Lista amostras suspeitas de label errado")
    parser.add_argument("--eval_predictions", type=str, required=True,
                        help="CSV gerado pelo evaluate.py, ex: val_predictions.csv")
    parser.add_argument("--out_csv", type=str, default="suspected_label_issues.csv")
    parser.add_argument("--top_k", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.eval_predictions)

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        raise ValueError("O CSV precisa conter colunas prob_*. Rode evaluate.py para gerar probabilidades.")

    labels = df["y_true"].values.astype(int)
    pred_probs = df[prob_cols].values

    ranked = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    issues = df.iloc[ranked[:args.top_k]].copy()
    issues["rank_suspeita"] = range(1, len(issues) + 1)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    issues.to_csv(args.out_csv, index=False)
    print(f"Salvo: {args.out_csv}")
    print("Inspecione manualmente as primeiras linhas antes de remover qualquer amostra.")


if __name__ == "__main__":
    main()
