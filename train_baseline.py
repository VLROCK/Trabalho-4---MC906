import os
import random
import argparse

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
from models.resnet_dense import ResNet50DenseHead


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tuning ResNet50 com dense head para Cassava Leaf Disease"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/cassava_data",
        help="Diretório contendo train.csv e train_images/"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="pesos/resnet50_dense",
        help="Diretório onde os checkpoints serão salvos"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Número de épocas"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Tamanho do batch"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate da dense head. O backbone usa lr * 0.1"
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay do AdamW"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Número de workers do DataLoader"
    )

    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Proporção usada para validação"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed de reprodutibilidade"
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Dimensão da camada densa intermediária"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout da dense head"
    )

    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Se usado, congela o backbone e treina só a dense head"
    )

    return parser.parse_args()


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


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


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc=f"Epoch {epoch}/{epochs} - Treino", leave=True)

    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validação", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def build_optimizer(model, args):
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Na classe ResNet50DenseHead, a dense head fica dentro de model.model.fc
        if "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    print(f"Parâmetros backbone treináveis: {sum(p.numel() for p in backbone_params):,}")
    print(f"Parâmetros head treináveis: {sum(p.numel() for p in head_params):,}")

    if len(backbone_params) == 0:
        optimizer = optim.AdamW(
            head_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": args.lr * 0.1},
                {"params": head_params, "lr": args.lr},
            ],
            weight_decay=args.weight_decay
        )

    return optimizer


def main():
    args = parse_args()

    seed_everything(args.seed)

    os.makedirs(args.save_path, exist_ok=True)

    csv_path = os.path.join(args.data_dir, "train.csv")
    img_dir = os.path.join(args.data_dir, "train_images")

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Diretório data_dir não encontrado: {args.data_dir}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo train.csv não encontrado em: {csv_path}")

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Pasta train_images não encontrada em: {img_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Treinando em: {device}")

    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

    print("\nConfigurações:")
    print(f"data_dir: {args.data_dir}")
    print(f"save_path: {args.save_path}")
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"lr head: {args.lr}")
    print(f"lr backbone: {args.lr * 0.1}")
    print(f"weight_decay: {args.weight_decay}")
    print(f"num_workers: {args.num_workers}")
    print(f"val_size: {args.val_size}")
    print(f"seed: {args.seed}")
    print(f"hidden_dim: {args.hidden_dim}")
    print(f"dropout: {args.dropout}")
    print(f"freeze_backbone: {args.freeze_backbone}\n")

    df = pd.read_csv(csv_path)

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=df["label"]
    )

    print(f"Total de imagens: {len(df)}")
    print(f"Treino: {len(train_df)}")
    print(f"Validação: {len(val_df)}\n")

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

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    model = ResNet50DenseHead(
        num_classes=5,
        pretrained=True,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, args)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    best_val_acc = 0.0

    best_model_path = os.path.join(args.save_path, "best_resnet50_dense.pth")
    last_model_path = os.path.join(args.save_path, "last_resnet50_dense.pth")

    history_path = os.path.join(args.save_path, "history.csv")
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=args.epochs
        )

        val_loss, val_acc = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        scheduler.step()

        current_lr_backbone = optimizer.param_groups[0]["lr"]
        current_lr_head = optimizer.param_groups[-1]["lr"]

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"LR backbone: {current_lr_backbone:.2e} | "
            f"LR head: {current_lr_head:.2e}"
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr_backbone": current_lr_backbone,
            "lr_head": current_lr_head,
        }

        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "args": vars(args),
        }

        torch.save(checkpoint, last_model_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint["best_val_acc"] = best_val_acc

            torch.save(checkpoint, best_model_path)

            print(f"Novo melhor modelo salvo em: {best_model_path}")
            print(f"Melhor Val Acc até agora: {best_val_acc:.2f}%")

    print("\nTreino finalizado.")
    print(f"Melhor Val Acc: {best_val_acc:.2f}%")
    print(f"Melhor modelo salvo em: {best_model_path}")
    print(f"Último modelo salvo em: {last_model_path}")
    print(f"Histórico salvo em: {history_path}")


if __name__ == "__main__":
    main()