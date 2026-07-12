"""Concordância entre os três detectores de ruído.

Cada detector devolve um conjunto top-K de image_ids suspeitos. Aqui medimos o
quanto eles concordam (interseções par a par via Jaccard, e quantos métodos
apontam cada imagem). Amostras sinalizadas pelos três métodos são as candidatas
mais fortes a rótulo errado.
"""

from __future__ import annotations

import itertools

import pandas as pd


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def summarize(sets: dict) -> dict:
    """sets: {nome_metodo: set(image_ids)}. Retorna Jaccard par a par."""
    pairs = {}
    for m1, m2 in itertools.combinations(sets, 2):
        pairs[f"{m1}__{m2}"] = jaccard(sets[m1], sets[m2])
    return pairs


def vote_table(sets: dict) -> pd.DataFrame:
    """Tabela: para cada image_id, quais métodos o apontaram e o total de votos."""
    all_ids = set().union(*sets.values()) if sets else set()
    rows = []
    for img in all_ids:
        flags = {m: (img in s) for m, s in sets.items()}
        rows.append({"image_id": img, **{f"in_{m}": flags[m] for m in sets}, "n_votes": sum(flags.values())})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("n_votes", ascending=False).reset_index(drop=True)
    return df
