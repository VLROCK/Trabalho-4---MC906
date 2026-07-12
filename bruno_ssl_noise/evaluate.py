"""Etapa 3 — Avaliação por backbone individual E ensemble.

Recebe 1..N checkpoints. Para cada um, calcula acurácia, F1 por classe e matriz
de confusão no split de validação. Depois combina as probabilidades por média
(ensemble) e reporta as mesmas métricas. Os números individuais servem para
mostrar o ganho do ensemble — o argumento de "diversidade complementar" do
artigo (Seção V-E).

Também salva as probabilidades de validação de cada modelo e do ensemble, que
o detect_noise.py usa no Confident Learning.

Exemplo:
    python evaluate.py --data_dir $DATA --config configs/default.yaml \
        --checkpoints runs/resnet50/best.pth runs/densenet121/best.pth runs/effnet/best.pth \
        --out_dir runs/eval
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CassavaDataset
from data.transforms import build_transforms
from models.factory import build_classifier
from utils import get_device, load_config, save_eval_outputs, set_seed, stratified_split


def parse_args():
    ap = argparse.ArgumentParser(description="Avaliação individual + ensemble")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--checkpoints", type=str, nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, default="runs/eval")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y_true, probs_all, ids = [], [], []
    for images, labels, names in tqdm(loader, desc="eval"):
        images = images.to(device)
        probs = torch.softmax(model(images), dim=1)
        y_true.extend(labels.numpy().tolist())
        probs_all.append(probs.cpu().numpy())
        ids.extend(list(names))
    return np.array(y_true), np.concatenate(probs_all, axis=0), np.array(ids)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    class_names = cfg["class_names"]

    # Mesmo split de validação usado no treino.
    df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    img_dir = os.path.join(args.data_dir, "train_images")
    _, val_df = stratified_split(df, cfg["val_size"], cfg["seed"])

    val_tf = build_transforms(False, cfg["image_size"])
    val_ds = CassavaDataset(val_df, img_dir, transform=val_tf, return_id=True)
    bs = 8 if args.smoke else args.batch_size
    nw = 0 if args.smoke else cfg["train"]["num_workers"]
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)

    summary = []
    prob_stack = []
    y_true_ref, ids_ref = None, None

    for ckpt_path in args.checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device)
        model_name = ckpt.get("model_name", "resnet50")
        model = build_classifier(model_name, num_classes=cfg["num_classes"], pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        y_true, probs, ids = predict(model, val_loader, device)
        y_pred = probs.argmax(axis=1)
        metrics = save_eval_outputs(args.out_dir, y_true, y_pred, probs, ids, class_names, prefix=model_name)
        print(f"[eval] {model_name}: acc={metrics['accuracy']:.4f} macroF1={metrics['macro_f1']:.4f}")
        summary.append({"model": model_name, **metrics})

        prob_stack.append(probs)
        y_true_ref, ids_ref = y_true, ids

    # ---- ensemble: média das probabilidades ----
    if len(prob_stack) > 1:
        ens_probs = np.mean(prob_stack, axis=0)
        ens_pred = ens_probs.argmax(axis=1)
        metrics = save_eval_outputs(args.out_dir, y_true_ref, ens_pred, ens_probs, ids_ref,
                                    class_names, prefix="ensemble")
        print(f"[eval] ENSEMBLE: acc={metrics['accuracy']:.4f} macroF1={metrics['macro_f1']:.4f}")
        summary.append({"model": "ensemble", **metrics})

    pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    print(f"[eval] resumo salvo em {os.path.join(args.out_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
