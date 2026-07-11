"""Transformações simples no domínio da frequência.

A ideia aqui não é substituir a CNN, mas criar uma augmentation que força o modelo
a ser robusto a mudanças de textura e iluminação. A saída continua RGB com 3 canais,
então funciona em qualquer backbone pré-treinado.
"""

from __future__ import annotations

import random
import cv2
import numpy as np
import albumentations as A


class FFTFilterAug(A.ImageOnlyTransform):
    """Aplica filtro passa-baixa ou passa-alta aleatório via FFT.

    mode="random": alterna entre lowpass/highpass.
    keep_ratio controla quanto do espectro central é mantido/removido.
    """

    def __init__(self, keep_ratio=(0.08, 0.25), mode="random", always_apply=False, p=0.25):
        super().__init__(always_apply=always_apply, p=p)
        self.keep_ratio = keep_ratio
        self.mode = mode

    def apply(self, image, **params):
        if image.ndim != 3:
            return image

        img = image.astype(np.float32)
        h, w, c = img.shape
        ratio = random.uniform(*self.keep_ratio)
        radius = int(min(h, w) * ratio)

        yy, xx = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

        chosen_mode = self.mode
        if chosen_mode == "random":
            chosen_mode = random.choice(["lowpass", "highpass"])

        if chosen_mode == "lowpass":
            mask = (dist <= radius).astype(np.float32)
        elif chosen_mode == "highpass":
            mask = (dist >= radius).astype(np.float32)
        else:
            raise ValueError(f"Modo FFT inválido: {self.mode}")

        out = np.empty_like(img)
        for ch in range(c):
            f = np.fft.fft2(img[:, :, ch])
            fshift = np.fft.fftshift(f)
            filtered = fshift * mask
            inv = np.fft.ifft2(np.fft.ifftshift(filtered))
            out[:, :, ch] = np.real(inv)

        out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
        return np.clip(out, 0, 255).astype(np.uint8)
