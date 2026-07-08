import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

class CassavaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Lê o CSV que contém as colunas 'image_id' e 'label'
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        # Diz ao PyTorch o tamanho total do dataset
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Pega o nome da imagem e a label da linha atual do CSV
        img_name = self.df.iloc[idx]['image_id']
        label = self.df.iloc[idx]['label']
        
        # 2. Monta o caminho completo e carrega a imagem
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        
        # O OpenCV lê em BGR, precisamos converter para RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 3. Aplica o Data Augmentation (Albumentations ou Torchvision)
        if self.transform:
            # Padrão de chamada do albumentations
            augmented = self.transform(image=image)
            image = augmented['image']
            
        # 4. Retorna a imagem e o tensor da classe
        return image, torch.tensor(label, dtype=torch.long)