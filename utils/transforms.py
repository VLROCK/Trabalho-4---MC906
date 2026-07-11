"""Factory de transforms Albumentations para experimentos."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.frequency import FFTFilterAug
from utils.segmentation import HSVLeafCrop

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(
    train: bool,
    image_size: int = 224,
    augmentation: str = "medium",
    use_segmentation: bool = False,
    frequency_aug: bool = False,
):
    """Cria transform de treino ou validação/teste.

    augmentation:
      - none
      - light
      - medium
      - strong
    """
    transforms = []

    if use_segmentation:
        transforms.append(HSVLeafCrop(p=1.0))

    if train:
        if augmentation == "none":
            transforms += [
                A.Resize(image_size, image_size),
            ]
        elif augmentation == "light":
            transforms += [
                A.Resize(image_size + 32, image_size + 32),
                A.RandomCrop(image_size, image_size),
                A.HorizontalFlip(p=0.5),
            ]
        elif augmentation == "medium":
            transforms += [
                A.Resize(image_size + 32, image_size + 32),
                A.RandomCrop(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-20, 20), p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.2),
            ]
        elif augmentation == "strong":
            transforms += [
                A.Resize(image_size + 48, image_size + 48),
                A.RandomCrop(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Affine(scale=(0.8, 1.2), translate_percent=(-0.08, 0.08), rotate=(-35, 35), shear=(-8, 8), p=0.6),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
                A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=18, val_shift_limit=12, p=0.35),
                A.CLAHE(p=0.15),
                A.CoarseDropout(max_holes=8, max_height=max(8, image_size // 10), max_width=max(8, image_size // 10), p=0.35),
            ]
        else:
            raise ValueError(f"augmentation inválido: {augmentation}")

        if frequency_aug:
            transforms.append(FFTFilterAug(p=0.25))
    else:
        transforms += [
            A.Resize(image_size + 32, image_size + 32),
            A.CenterCrop(image_size, image_size),
        ]

    transforms += [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]

    return A.Compose(transforms)
