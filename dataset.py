import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset


class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image_id"]
        label = int(self.df.loc[idx, "label"])

        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"Imagem não encontrada ou inválida: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.long)