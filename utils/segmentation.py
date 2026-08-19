"""Pré-processamento simples para reduzir fundo por máscara HSV.

Não é segmentação perfeita. É um crop aproximado da região com folha, pensado como
experimento rápido para imagens amadoras com fundo variável.
"""

from __future__ import annotations

import cv2
import numpy as np
import albumentations as A


class HSVLeafCrop(A.ImageOnlyTransform):
    """Recorta a imagem ao redor de pixels verdes/amarelos/marrons prováveis da folha."""

    def __init__(self, margin: float = 0.08, min_area_ratio: float = 0.05, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.margin = margin
        self.min_area_ratio = min_area_ratio

    def apply(self, image, **params):
        if image is None or image.ndim != 3:
            return image

        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Faixas amplas: verde, amarelado e algumas regiões marrons de lesão.
        lower1 = np.array([15, 25, 20], dtype=np.uint8)
        upper1 = np.array([105, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower1, upper1)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        area = int((mask > 0).sum())
        if area < self.min_area_ratio * h * w:
            return image

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return image

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        mx = int(self.margin * (x2 - x1 + 1))
        my = int(self.margin * (y2 - y1 + 1))

        x1 = max(0, x1 - mx)
        x2 = min(w, x2 + mx)
        y1 = max(0, y1 - my)
        y2 = min(h, y2 + my)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return image
        return crop
