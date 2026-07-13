"""MixUp e CutMix para classificação de imagens."""

from __future__ import annotations

import numpy as np
import torch


def rand_bbox(size, lam: float):
    """Gera bounding box para CutMix.

    size esperado: [B, C, H, W]
    """
    _, _, H, W = size
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2


def apply_mix(images, labels, mode: str = "none", alpha: float = 1.0, prob: float = 1.0):
    """Aplica MixUp ou CutMix em um batch.

    Retorna:
      images_mixed, y_a, y_b, lam, used_mix
    """
    mode = mode.lower()

    if mode == "none" or alpha <= 0 or np.random.rand() > prob:
        return images, labels, labels, 1.0, False

    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    y_a = labels
    y_b = labels[index]

    if mode == "mixup":
        mixed = lam * images + (1.0 - lam) * images[index]
        return mixed, y_a, y_b, float(lam), True

    if mode == "cutmix":
        mixed = images.clone()
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

        # Corrige lambda pela área realmente substituída.
        area = (x2 - x1) * (y2 - y1)
        lam = 1.0 - area / (images.size(-1) * images.size(-2))
        return mixed, y_a, y_b, float(lam), True

    raise ValueError(f"Modo de mix desconhecido: {mode}")
