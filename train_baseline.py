import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import CassavaDataset
from models.baseline import CassavaBaseline


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms():
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=20,
            p=0.5
        ),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])

    return train_transform, val_transform


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Treino", leave=True)

    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

        loop.set_postfix(
            loss=running_loss / total,
            acc=100.0 * correct / total
        )

    return running_loss / total, 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validação", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

    return running_loss / total, 100.0 * correct / total


def main():
    seed_everything(42)

    data_dir = "data/cassava_data"
    csv_path = os.path.join(data_dir, "train.csv")
    img_dir = os.path.join(data_dir, "train_images")

    batch_size = 32
    num_workers = 2
    epochs = 5
    lr = 1e-4
    weight_decay = 1e-2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando em: {device}")

    df = pd.read_csv(csv_path)

    train_df, val_df = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        stratify=df["label"]
    )

    train_transform, val_transform = get_transforms()

    train_dataset = CassavaDataset(
        df=train_df,
        img_dir=img_dir,
        transform=train_transform
    )

    val_dataset = CassavaDataset(
        df=val_df,
        img_dir=img_dir,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    model = CassavaBaseline(num_classes=5, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_resnet_cassava.pth")
            print(f"Novo melhor modelo salvo: {best_val_acc:.2f}%")

    print(f"\nMelhor Val Acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()