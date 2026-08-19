"""Treino modular para testar perdas, augmentations, CutMix/MixUp, segmentação,
frequência, TTA e diferentes arquiteturas.

Exemplo:
python train_experiments.py \
  --data_dir /content/cassava_data \
  --save_path /content/drive/MyDrive/experimentos \
  --experiment_name resnet50_focal_cutmix \
  --model resnet50 \
  --loss focal \
  --mix cutmix \
  --augmentation strong \
  --epochs 20
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import CassavaDataset
from models.model_factory import build_model
from utils.losses import build_criterion, compute_class_weights, mix_criterion
from utils.mix import apply_mix
from utils.transforms import build_transforms
from utils.tta import predict_logits

CLASS_NAMES = ["0", "1", "2", "3", "4"]


def parse_args():
    parser = argparse.ArgumentParser(description="Treino modular Cassava")

    parser.add_argument("--data_dir", type=str, default="data/cassava_data")
    parser.add_argument("--save_path", type=str, default="experiments")
    parser.add_argument("--experiment_name", type=str, default=None)

    parser.add_argument("--model", type=str, default="resnet50",
                        help="resnet50, efficientnet_b0, efficientnet_b3, convnext_tiny, swin_tiny, cnn_transformer_resnet50...")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--drop_rate", type=float, default=0.0)

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--loss", type=str, default="ce",
                        choices=["ce", "label_smoothing", "weighted_ce", "weighted_label_smoothing", "focal", "weighted_focal"])
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    parser.add_argument("--sampler", type=str, default="none", choices=["none", "weighted"])
    parser.add_argument("--augmentation", type=str, default="medium", choices=["none", "light", "medium", "strong"])
    parser.add_argument("--segmentation", action="store_true")
    parser.add_argument("--frequency_aug", action="store_true")

    parser.add_argument("--mix", type=str, default="none", choices=["none", "mixup", "cutmix"])
    parser.add_argument("--mix_alpha", type=float, default=1.0)
    parser.add_argument("--mix_prob", type=float, default=0.5)

    parser.add_argument("--amp", action="store_true", help="Usa mixed precision quando houver CUDA")
    parser.add_argument("--val_tta", action="store_true", help="Usa TTA na validação")

    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_experiment_dir(args):
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{timestamp}_{args.model}_{args.loss}_{args.mix}"
    out_dir = os.path.join(args.save_path, args.experiment_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def create_loaders(args, train_df, val_df, img_dir):
    train_tf = build_transforms(
        train=True,
        image_size=args.image_size,
        augmentation=args.augmentation,
        use_segmentation=args.segmentation,
        frequency_aug=args.frequency_aug,
    )
    val_tf = build_transforms(
        train=False,
        image_size=args.image_size,
        augmentation="none",
        use_segmentation=args.segmentation,
        frequency_aug=False,
    )

    train_ds = CassavaDataset(train_df, img_dir=img_dir, transform=train_tf)
    val_ds = CassavaDataset(val_df, img_dir=img_dir, transform=val_tf)

    pin = torch.cuda.is_available()

    sampler = None
    shuffle = True
    if args.sampler == "weighted":
        class_weights = compute_class_weights(train_df["label"].values, num_classes=5)
        sample_weights = class_weights[torch.as_tensor(train_df["label"].values, dtype=torch.long)]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, args):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    loop = tqdm(loader, desc="Treino", leave=True)
    use_amp = args.amp and device.type == "cuda"

    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        images_mixed, y_a, y_b, lam, used_mix = apply_mix(
            images, labels, mode=args.mix, alpha=args.mix_alpha, prob=args.mix_prob
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images_mixed)
            if used_mix:
                loss = mix_criterion(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)

        # Métrica de treino é aproximada quando há MixUp/CutMix.
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(labels.detach().cpu().numpy().tolist())

        loop.set_postfix(loss=running_loss / max(1, len(all_targets)))

    loss_avg = running_loss / len(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    return loss_avg, acc, macro_f1


@torch.no_grad()
def validate(model, loader, criterion, device, use_tta=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    for images, labels in tqdm(loader, desc="Validação", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = predict_logits(model, images, tta=use_tta)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        running_loss += loss.item() * labels.size(0)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())
        all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    loss_avg = running_loss / len(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    return loss_avg, acc, bal_acc, macro_f1, np.array(all_targets), np.array(all_preds), all_probs


def main():
    args = parse_args()
    seed_everything(args.seed)
    out_dir = make_experiment_dir(args)

    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(args.data_dir, "train.csv")
    img_dir = os.path.join(args.data_dir, "train_images")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"train.csv não encontrado: {csv_path}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"train_images não encontrado: {img_dir}")

    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(
        df,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=df["label"],
    )

    train_df.to_csv(os.path.join(out_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val_split.csv"), index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Experimento: {args.experiment_name}")
    print(f"Treino: {len(train_df)} | Val: {len(val_df)}")

    train_loader, val_loader = create_loaders(args, train_df, val_df, img_dir)

    model = build_model(
        args.model,
        num_classes=5,
        pretrained=args.pretrained,
        image_size=args.image_size,
        drop_rate=args.drop_rate,
    ).to(device)

    criterion = build_criterion(
        args.loss,
        labels=train_df["label"].values,
        num_classes=5,
        device=device,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_macro_f1 = -1.0
    best_acc = -1.0
    history = []

    best_path = os.path.join(out_dir, "best_model.pth")
    last_path = os.path.join(out_dir, "last_model.pth")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc, train_macro_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args
        )

        val_loss, val_acc, val_bal_acc, val_macro_f1, y_true, y_pred, probs = validate(
            model, val_loader, criterion, device, use_tta=args.val_tta
        )

        scheduler.step()

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_macro_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_balanced_acc": val_bal_acc,
            "val_macro_f1": val_macro_f1,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Val Macro-F1: {val_macro_f1*100:.2f}%"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_macro_f1": best_macro_f1,
            "best_acc": best_acc,
            "model_name": args.model,
            "num_classes": 5,
            "class_names": CLASS_NAMES,
            "args": vars(args),
        }
        torch.save(checkpoint, last_path)

        # Critério principal: macro-F1, melhor que accuracy em dataset desbalanceado.
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_acc = val_acc
            checkpoint["best_macro_f1"] = best_macro_f1
            checkpoint["best_acc"] = best_acc
            torch.save(checkpoint, best_path)
            print(f"Novo melhor salvo: {best_path}")

    print("\nTreino finalizado.")
    print(f"Melhor Val Acc: {best_acc*100:.2f}%")
    print(f"Melhor Val Macro-F1: {best_macro_f1*100:.2f}%")
    print(f"Arquivos em: {out_dir}")


if __name__ == "__main__":
    main()
