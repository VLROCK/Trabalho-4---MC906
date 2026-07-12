"""Detector 3 — Loss-ranking.

Amostras cuja loss de treino permanece alta ao longo das épocas são candidatas
a rótulo errado: um modelo bem treinado aprende as amostras "limpas" e continua
"errando" as que têm rótulo inconsistente. Usamos a loss média das últimas
épocas (mais estável que uma época só).

Entrada: per_sample_loss.csv gerado pelo train.py (colunas image_id, epoch, loss).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect(per_sample_loss: pd.DataFrame, last_n_epochs: int = 3, top_k: int = 200) -> pd.DataFrame:
    """Ranqueia por loss média nas últimas `last_n_epochs` épocas."""
    epochs = sorted(per_sample_loss["epoch"].unique())
    keep = epochs[-last_n_epochs:] if len(epochs) >= last_n_epochs else epochs
    recent = per_sample_loss[per_sample_loss["epoch"].isin(keep)]

    agg = recent.groupby("image_id")["loss"].mean().reset_index()
    agg = agg.rename(columns={"loss": "score_loss"})
    agg = agg.sort_values("score_loss", ascending=False).reset_index(drop=True)
    agg["rank_loss"] = np.arange(1, len(agg) + 1)
    return agg.head(top_k).reset_index(drop=True) if top_k else agg
