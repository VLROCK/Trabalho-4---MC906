"""Barlow Twins: projection head + loss.

É o análogo, para classificação, do pré-treino self-supervised que o artigo
aplica aos backbones antes do pipeline. A ideia do Barlow Twins: passar duas
views aumentadas da mesma imagem, projetá-las e forçar a matriz de
cross-correlação entre as duas projeções a ser próxima da identidade
(invariância na diagonal, redução de redundância fora dela).

Referência: Zbontar et al., "Barlow Twins", ICML 2021.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """MLP de projeção (3 camadas) usada no Barlow Twins."""

    def __init__(self, in_dim: int, proj_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class BarlowTwins(nn.Module):
    """Envolve um backbone (num_classes=0) com a projection head."""

    def __init__(self, backbone, feat_dim: int, proj_dim: int = 2048):
        super().__init__()
        self.backbone = backbone
        self.projector = ProjectionHead(feat_dim, proj_dim)

    def forward(self, view1, view2):
        z1 = self.projector(self.backbone(view1))
        z2 = self.projector(self.backbone(view2))
        return z1, z2


def barlow_twins_loss(z1, z2, lambda_bt: float = 0.005):
    """Loss do Barlow Twins.

    z1, z2: projeções [batch, proj_dim] das duas views.
    lambda_bt: peso do termo off-diagonal (redundância).
    """
    batch_size = z1.shape[0]

    # Normaliza cada dimensão pela média/desvio no batch.
    z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
    z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)

    # Matriz de cross-correlação [proj_dim, proj_dim].
    c = (z1_norm.T @ z2_norm) / batch_size

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = _off_diagonal(c).pow_(2).sum()
    return on_diag + lambda_bt * off_diag


def _off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
