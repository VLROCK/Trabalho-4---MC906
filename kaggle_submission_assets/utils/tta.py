"""TTA simples em tensores já normalizados."""

from __future__ import annotations

import torch


def _apply_tta_tensor(images: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "orig":
        return images
    if mode == "hflip":
        return torch.flip(images, dims=[3])
    if mode == "vflip":
        return torch.flip(images, dims=[2])
    if mode == "hvflip":
        return torch.flip(images, dims=[2, 3])
    raise ValueError(f"TTA desconhecido: {mode}")


@torch.no_grad()
def predict_logits(model, images: torch.Tensor, tta: bool = False) -> torch.Tensor:
    if not tta:
        return model(images)

    modes = ["orig", "hflip", "vflip", "hvflip"]
    logits_sum = None

    for mode in modes:
        aug_images = _apply_tta_tensor(images, mode)
        logits = model(aug_images)
        logits_sum = logits if logits_sum is None else logits_sum + logits

    return logits_sum / len(modes)
