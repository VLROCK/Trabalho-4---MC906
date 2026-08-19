"""Avaliação genérica e geração de submissão.

Validação com labels:
python evaluate.py --checkpoint EXP/best_model.pth --data_dir /content/cassava_data --split val --out_dir EXP/eval --tta

Teste Kaggle sem labels:
python evaluate.py --checkpoint EXP/best_model.pth --data_dir /content/cassava_data --split test --out_dir EXP/test_pred --tta
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CassavaDataset
from dataset_ext import CassavaTestDataset
from models.model_factory import build_model
from utils.metrics import save_eval_outputs
from utils.transforms import build_transforms
from utils.tta import predict_logits


def parse_args():
    parser = argparse.ArgumentParser(description="Avaliação genérica Cassava")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--tta", action="store_true")

    # Permite sobrescrever configs do checkpoint.
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--segmentation", action="store_true")
    parser.add_argument("--val_size", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, device, args):
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})

    model_name = args.model or ckpt.get("model_name") or ckpt_args.get("model", "resnet50")
    image_size = args.image_size or ckpt_args.get("image_size", 224)
    drop_rate = ckpt_args.get("drop_rate", 0.0)
    num_classes = ckpt.get("num_classes", 5)

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        image_size=image_size,
        drop_rate=drop_rate,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, ckpt, ckpt_args


@torch.no_grad()
def predict_labeled(model, loader, device, use_tta=False):
    y_true, y_pred, probs_all, image_ids = [], [], [], []

    for batch in tqdm(loader, desc="Avaliando"):
        # CassavaDataset original retorna image,label. Não retorna image_id.
        images, labels = batch
        images = images.to(device, non_blocking=True)
        logits = predict_logits(model, images, tta=use_tta)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        probs_all.append(probs.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.concatenate(probs_all, axis=0)


@torch.no_grad()
def predict_test(model, loader, device, use_tta=False):
    image_ids, preds_all, probs_all = [], [], []

    for images, names in tqdm(loader, desc="Predizendo teste"):
        images = images.to(device, non_blocking=True)
        logits = predict_logits(model, images, tta=use_tta)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        image_ids.extend(list(names))
        preds_all.extend(preds.cpu().numpy().tolist())
        probs_all.append(probs.cpu().numpy())

    return image_ids, np.array(preds_all), np.concatenate(probs_all, axis=0)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    model, ckpt, ckpt_args = load_model_from_checkpoint(args.checkpoint, device, args)

    image_size = args.image_size or ckpt_args.get("image_size", 224)
    seed = args.seed if args.seed is not None else ckpt_args.get("seed", 42)
    val_size = args.val_size if args.val_size is not None else ckpt_args.get("val_size", 0.15)
    segmentation = args.segmentation or ckpt_args.get("segmentation", False)
    class_names = ckpt.get("class_names", ["0", "1", "2", "3", "4"])

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(args.checkpoint), f"eval_{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "eval_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    transform = build_transforms(
        train=False,
        image_size=image_size,
        augmentation="none",
        use_segmentation=segmentation,
        frequency_aug=False,
    )

    pin = torch.cuda.is_available()

    if args.split == "val":
        csv_path = os.path.join(args.data_dir, "train.csv")
        img_dir = os.path.join(args.data_dir, "train_images")
        df = pd.read_csv(csv_path)
        _, val_df = train_test_split(df, test_size=val_size, random_state=seed, stratify=df["label"])

        ds = CassavaDataset(val_df, img_dir=img_dir, transform=transform)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

        y_true, y_pred, probs = predict_labeled(model, loader, device, use_tta=args.tta)
        metrics, _, _ = save_eval_outputs(
            out_dir,
            y_true=y_true,
            y_pred=y_pred,
            pred_probs=probs,
            image_ids=val_df["image_id"].values,
            class_names=class_names,
            prefix="val_tta" if args.tta else "val",
        )
        print(metrics)

    else:
        test_dir = os.path.join(args.data_dir, "test_images")
        sample_path = os.path.join(args.data_dir, "sample_submission.csv")
        if os.path.exists(sample_path):
            test_df = pd.read_csv(sample_path)
        else:
            files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
            test_df = pd.DataFrame({"image_id": files})

        ds = CassavaTestDataset(test_df, img_dir=test_dir, transform=transform)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

        image_ids, preds, probs = predict_test(model, loader, device, use_tta=args.tta)
        sub = pd.DataFrame({"image_id": image_ids, "label": preds.astype(int)})
        sub_path = os.path.join(out_dir, "submission.csv")
        sub.to_csv(sub_path, index=False)

        prob_df = pd.DataFrame(probs, columns=[f"prob_{c}" for c in class_names])
        prob_df.insert(0, "image_id", image_ids)
        prob_df.to_csv(os.path.join(out_dir, "test_probabilities.csv"), index=False)
        print(f"Submission salva em: {sub_path}")
        print("Obs: o test set do Kaggle não tem labels públicas; métricas locais só existem no split de validação.")


if __name__ == "__main__":
    main()
