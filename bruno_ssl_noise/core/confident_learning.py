"""Detector 1 — Confident Learning (cleanlab).

Compara os rótulos originais com as probabilidades preditas pelo modelo e ranqueia
as amostras mais prováveis de terem rótulo errado. É a abordagem mais direta:
usa diretamente a discordância entre "o que o rótulo diz" e "o que o modelo
confiante prevê".

Nota metodológica: o ideal é usar probabilidades *out-of-fold* (cross-validation)
para não vazar. Aqui usamos as probabilidades do split de validação, então isto
é uma versão inicial/honesta, não uma prova definitiva. Está documentado no
relatório.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect(pred_df: pd.DataFrame, top_k: int = 200) -> pd.DataFrame:
    """pred_df: precisa ter 'image_id', 'y_true' e colunas 'prob_*'.

    Retorna DataFrame com image_id, score_cl (quanto maior, mais suspeita) e rank.
    """
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores

    prob_cols = [c for c in pred_df.columns if c.startswith("prob_")]
    if not prob_cols:
        raise ValueError("pred_df precisa de colunas prob_*. Rode evaluate.py antes.")

    labels = pred_df["y_true"].values.astype(int)
    pred_probs = pred_df[prob_cols].values

    # quality score: 0 = muito suspeita, 1 = confiável. Invertemos para virar "score de suspeita".
    quality = get_label_quality_scores(labels=labels, pred_probs=pred_probs)
    ranked_idx = find_label_issues(
        labels=labels, pred_probs=pred_probs, return_indices_ranked_by="self_confidence"
    )

    out = pd.DataFrame({
        "image_id": pred_df["image_id"].values,
        "y_true": labels,
        "score_cl": 1.0 - quality,  # maior = mais suspeita
    })
    out["is_issue_cl"] = out["image_id"].isin(pred_df["image_id"].values[ranked_idx])

    # ranking: primeiro os marcados como issue (na ordem do cleanlab), depois por score
    order = pred_df["image_id"].values[ranked_idx].tolist()
    rank_map = {img: i + 1 for i, img in enumerate(order)}
    out["rank_cl"] = out["image_id"].map(rank_map)
    out = out.sort_values(["rank_cl", "score_cl"], ascending=[True, False], na_position="last")
    out["rank_cl"] = np.arange(1, len(out) + 1)
    return out.head(top_k).reset_index(drop=True) if top_k else out.reset_index(drop=True)
