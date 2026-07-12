"""Datasets do Cassava.

Mantém a mesma ideia do dataset original do projeto (lê um CSV com colunas
`image_id` e `label` e carrega imagens com OpenCV), mas adiciona:

- CassavaDataset: aceita um DataFrame já filtrado (train/val split), e pode
  devolver o image_id junto (útil para rastrear amostras suspeitas).
- TwoViewDataset: gera duas views aumentadas da mesma imagem, para o
  pré-treino self-supervised (Barlow Twins).
"""

from __future__ import annotations

import os

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


def _load_rgb(img_path: str):
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Imagem não encontrada ou inválida: {img_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class CassavaDataset(Dataset):
    """Dataset supervisionado.

    df: DataFrame com colunas 'image_id' e (opcionalmente) 'label'.
    return_id: se True, retorna (image, label, image_id).
    """

    def __init__(self, df, img_dir, transform=None, return_id=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.return_id = return_id
        self.has_label = "label" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_name = row["image_id"]
        image = _load_rgb(os.path.join(self.img_dir, img_name))

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        label = int(row["label"]) if self.has_label else -1
        label = torch.tensor(label, dtype=torch.long)

        if self.return_id:
            return image, label, img_name
        return image, label


class TwoViewDataset(Dataset):
    """Gera duas views aumentadas da mesma imagem (para Barlow Twins).

    Não usa os rótulos: o pré-treino é totalmente self-supervised, exatamente
    como o artigo faz o pré-treino dos backbones antes do pipeline.
    """

    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image_id"]
        image = _load_rgb(os.path.join(self.img_dir, img_name))
        view1 = self.transform(image=image)["image"]
        view2 = self.transform(image=image)["image"]
        return view1, view2
