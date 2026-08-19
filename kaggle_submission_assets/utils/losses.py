"""Losses reutilizáveis para os experimentos do Cassava.

Suporta:
- CrossEntropy normal
- Label smoothing
- Weighted CrossEntropy
- Focal Loss
- Weighted Focal Loss
- loss para MixUp/CutMix
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(labels, num_classes: int, device=None) -> torch.Tensor:
    """Calcula pesos inversamente proporcionais à frequência das classes.

    labels pode ser uma lista, numpy array ou tensor com rótulos inteiros.
    Retorna pesos normalizados com média próxima de 1.
    """
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels_tensor, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)

    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()

    if device is not None:
        weights = weights.to(device)
    return weights


class FocalLoss(nn.Module):
    """Focal Loss multiclasse.

    gamma > 0 reduz a contribuição de exemplos fáceis.
    alpha pode ser um tensor [num_classes] com pesos por classe.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        targets = targets.long()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        loss = -((1.0 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(
    loss_name: str,
    labels=None,
    num_classes: int = 5,
    device=None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
):
    """Cria a função de perda a partir de uma string.

    loss_name:
      - ce
      - label_smoothing
      - weighted_ce
      - weighted_label_smoothing
      - focal
      - weighted_focal
    """
    loss_name = loss_name.lower()
    class_weights = None

    if "weighted" in loss_name:
        if labels is None:
            raise ValueError("labels é necessário para perdas ponderadas.")
        class_weights = compute_class_weights(labels, num_classes=num_classes, device=device)

    if loss_name == "ce":
        return nn.CrossEntropyLoss()

    if loss_name == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if loss_name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights)

    if loss_name == "weighted_label_smoothing":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    if loss_name == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=None)

    if loss_name == "weighted_focal":
        return FocalLoss(gamma=focal_gamma, alpha=class_weights)

    raise ValueError(f"Loss desconhecida: {loss_name}")


def mix_criterion(criterion, logits, y_a, y_b, lam: float):
    """Loss para MixUp/CutMix usando labels misturados implicitamente."""
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
