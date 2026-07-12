"""Detector 2 — kNN-consistency no espaço de embeddings do SSL.

Ideia: se uma imagem rotulada como classe A está cercada, no espaço de features
do pré-treino self-supervised, por vizinhos majoritariamente de outra classe,
o rótulo dela é suspeito. Isto usa o espaço geométrico aprendido pelo Barlow
Twins — é onde o pré-treino do artigo "reaparece" na parte de detecção de erro.

score_knn = 1 - (fração de vizinhos que concordam com o rótulo da amostra).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def detect(embeddings: np.ndarray, labels: np.ndarray, image_ids: np.ndarray,
           k: int = 20, top_k: int = 200) -> pd.DataFrame:
    """embeddings: [N, D] do backbone SSL. labels: rótulos originais. image_ids: [N]."""
    # normaliza para usar distância por cosseno (mais estável em espaço de features)
    x = normalize(embeddings)
    k_eff = min(k + 1, len(x))  # +1 porque o vizinho mais próximo é a própria amostra

    nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine").fit(x)
    _, idx = nn.kneighbors(x)
    idx = idx[:, 1:]  # remove a própria amostra

    neighbor_labels = labels[idx]                      # [N, k]
    own = labels[:, None]                              # [N, 1]
    agree_frac = (neighbor_labels == own).mean(axis=1)  # fração que concorda

    # voto majoritário dos vizinhos (classe sugerida)
    suggested = np.array([np.bincount(row, minlength=int(labels.max()) + 1).argmax()
                          for row in neighbor_labels])

    out = pd.DataFrame({
        "image_id": image_ids,
        "y_true": labels,
        "knn_suggested": suggested,
        "score_knn": 1.0 - agree_frac,  # maior = mais suspeita
    })
    out = out.sort_values("score_knn", ascending=False).reset_index(drop=True)
    out["rank_knn"] = np.arange(1, len(out) + 1)
    return out.head(top_k).reset_index(drop=True) if top_k else out
