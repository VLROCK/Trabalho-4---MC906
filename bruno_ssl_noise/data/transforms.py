"""Transforms Albumentations para classificação e para o SSL.

- build_transforms: treino/val supervisionado (mesma lógica do projeto original).
- build_ssl_transforms: augmentação forte para as duas views do Barlow Twins.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(train: bool, image_size: int = 224, augmentation: str = "medium"):
    """Transform supervisionado. augmentation: none | light | medium | strong."""
    t = []
    if train:
        if augmentation == "none":
            t += [A.Resize(image_size, image_size)]
        elif augmentation == "light":
            t += [
                A.Resize(image_size + 32, image_size + 32),
                A.RandomCrop(image_size, image_size),
                A.HorizontalFlip(p=0.5),
            ]
        elif augmentation == "medium":
            t += [
                A.Resize(image_size + 32, image_size + 32),
                A.RandomCrop(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Affine(scale=(0.9, 1.1), rotate=(-20, 20), p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.2),
            ]
        elif augmentation == "strong":
            t += [
                A.Resize(image_size + 48, image_size + 48),
                A.RandomCrop(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Affine(scale=(0.8, 1.2), rotate=(-35, 35), shear=(-8, 8), p=0.6),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
                A.HueSaturationValue(p=0.35),
                A.CoarseDropout(p=0.35),
            ]
        else:
            raise ValueError(f"augmentation inválido: {augmentation}")
    else:
        t += [
            A.Resize(image_size + 32, image_size + 32),
            A.CenterCrop(image_size, image_size),
        ]

    t += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(t)


def build_ssl_transforms(image_size: int = 224):
    """Augmentação forte para o Barlow Twins.

    Segue o espírito das views SSL: crop aleatório, flips, jitter de cor forte,
    grayscale e blur ocasionais. Cada chamada produz uma view diferente.
    """
    return A.Compose([
        A.RandomResizedCrop(image_size, image_size, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        A.ToGray(p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
