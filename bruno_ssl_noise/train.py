"""Etapa 2 — Fine-tuning supervisionado de UM backbone.

Para o ensemble, basta rodar este script 2-3 vezes trocando --model. Cada run
gera seu próprio checkpoint `best.pth`. O ensemble é feito no evaluate.py.

Guarda, além do best.pth:
- checkpoints periódicos (a cada cfg.train.ckpt_every épocas), para SLURM.
- per_sample_loss.csv: loss de cada amostra de TREINO por época
  (usado depois pelo detector de ruído "loss-ranking").

Exemplo:
    python train.py --config configs/default.yaml --data_dir $DATA \
        --model resnet50 --ssl_backbone runs/ssl/backbone.pth --out_dir runs/resnet50
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CassavaDataset
from data.transforms import build_transforms
from models.factory import build_classifier
from utils import (
    get_device,
    load_config,
    save_checkpoint,
    save_eval_outputs,
    set_seed,
    stratified_split,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Fine-tuning supervisionado Cassava")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--model", type=str, default="resnet50")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--ssl_backbone", type=str, default=None,
                    help="caminho do backbone.pth do ssl_pretrain (opcional)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--no_pretrained", action="store_true",
                    help="não baixar pesos ImageNet (útil offline; o SSL já dá o ponto de partida)")
    ap.add_argument("--smoke", action="store_true", help="2 épocas, batch pequeno")
    return ap.parse_args()


def load_ssl_backbone(model, ssl_path, device, model_name):
    """Carrega pesos do backbone pré-treinado (SSL) no classificador.

    O SSL é feito para UMA arquitetura. Se o backbone salvo for de outra
    arquitetura (ex: SSL em resnet50, mas este run treina densenet121), o
    carregamento é ignorado e o modelo usa a init ImageNet — assim o pipeline
    do ensemble não quebra. O membro que casa com o SSL ganha o pré-treino; os
    demais entram com ImageNet.
    """
    ckpt = torch.load(ssl_path, map_location=device)
    ssl_model = ckpt.get("model_name")
    if ssl_model is not None and ssl_model.lower() != model_name.lower():
        print(f"[train] SSL backbone é de '{ssl_model}', mas este run é '{model_name}'. "
              f"Ignorando SSL e usando init ImageNet.")
        return
    missing, unexpected = model.backbone.load_state_dict(ckpt["backbone_state_dict"], strict=False)
    print(f"[train] SSL backbone carregado (missing={len(missing)}, unexpected={len(unexpected)})")


def build_criterion(cfg, labels, num_classes, device):
    name = cfg["train"]["loss"]
    if name == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])
    if name == "weighted_ce":
        counts = np.bincount(labels, minlength=num_classes).astype(float)
        counts = np.clip(counts, 1.0, None)
        w = counts.sum() / (num_classes * counts)
        w = torch.tensor(w / w.mean(), dtype=torch.float, device=device)
        return nn.CrossEntropyLoss(weight=w)
    return nn.CrossEntropyLoss()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    epochs = args.epochs or cfg["train"]["epochs"]
    batch_size = args.batch_size or cfg["train"]["batch_size"]
    lr = args.lr or cfg["train"]["lr"]
    if args.smoke:
        epochs, batch_size = 2, min(batch_size, 8)

    os.makedirs(args.out_dir, exist_ok=True)
    class_names = cfg["class_names"]
    num_classes = cfg["num_classes"]
    print(f"[train] model={args.model} epochs={epochs} batch={batch_size} lr={lr} device={device}")

    # Split estratificado (mesma semente do resto do projeto -> val fixo).
    df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    img_dir = os.path.join(args.data_dir, "train_images")
    train_df, val_df = stratified_split(df, cfg["val_size"], cfg["seed"])

    train_tf = build_transforms(True, cfg["image_size"], cfg["train"]["augmentation"])
    val_tf = build_transforms(False, cfg["image_size"])

    # return_id=True no treino para rastrear a loss por amostra (loss-ranking).
    train_ds = CassavaDataset(train_df, img_dir, transform=train_tf, return_id=True)
    val_ds = CassavaDataset(val_df, img_dir, transform=val_tf, return_id=True)

    pin = torch.cuda.is_available()
    nw = 0 if args.smoke else cfg["train"]["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin)

    model = build_classifier(args.model, num_classes=num_classes, pretrained=not args.no_pretrained).to(device)
    if args.ssl_backbone:
        load_ssl_backbone(model, args.ssl_backbone, device, args.model)

    criterion = build_criterion(cfg, train_df["label"].values, num_classes, device)
    criterion_noreduce = nn.CrossEntropyLoss(reduction="none")  # para loss por amostra
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = -1.0
    patience = cfg["train"]["early_stop_patience"]
    bad_epochs = 0
    per_sample_rows = []  # (image_id, epoch, loss) para o loss-ranking

    for epoch in range(1, epochs + 1):
        # ---- treino ----
        model.train()
        train_loss = 0.0
        for images, labels, ids in tqdm(train_loader, desc=f"train {epoch}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # loss por amostra (sem grad) para o detector loss-ranking
            with torch.no_grad():
                per = criterion_noreduce(logits, labels).cpu().numpy()
            for img_id, l in zip(ids, per):
                per_sample_rows.append({"image_id": img_id, "epoch": epoch, "loss": float(l)})

        scheduler.step()

        # ---- validação ----
        val_metrics, y_true, y_pred, probs, val_ids = evaluate_split(model, val_loader, device)
        print(f"[train] epoch {epoch}: train_loss={train_loss / len(train_loader):.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} val_macroF1={val_metrics['macro_f1']:.4f}")

        # checkpoint periódico
        if epoch % cfg["train"]["ckpt_every"] == 0:
            save_checkpoint(os.path.join(args.out_dir, f"epoch_{epoch}.pth"), model, optimizer, epoch,
                            extra={"model_name": args.model, "num_classes": num_classes, "class_names": class_names})

        # melhor modelo por macro-F1
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            bad_epochs = 0
            save_checkpoint(os.path.join(args.out_dir, "best.pth"), model, optimizer, epoch,
                            extra={"model_name": args.model, "num_classes": num_classes,
                                   "class_names": class_names, "val_metrics": val_metrics})
            # salva predições de validação do melhor modelo (para cleanlab/ensemble)
            save_eval_outputs(args.out_dir, y_true, y_pred, probs, val_ids, class_names, prefix="val")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[train] early stopping (sem melhora há {patience} épocas)")
                break

    # salva a loss por amostra ao longo do treino
    pd.DataFrame(per_sample_rows).to_csv(os.path.join(args.out_dir, "per_sample_loss.csv"), index=False)
    with open(os.path.join(args.out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "epochs": epochs, "lr": lr, "best_macro_f1": best_f1}, f, indent=2)
    print(f"[train] fim. best_macro_f1={best_f1:.4f}. Artefatos em {args.out_dir}")


@torch.no_grad()
def evaluate_split(model, loader, device):
    """Avalia um split que retorna (image, label, image_id)."""
    model.eval()
    y_true, y_pred, probs_all, ids = [], [], [], []
    for images, labels, names in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        probs_all.append(probs.cpu().numpy())
        ids.extend(list(names))

    from utils import compute_metrics
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    probs_all = np.concatenate(probs_all, axis=0)
    metrics, _, _ = compute_metrics(y_true, y_pred)
    return metrics, y_true, y_pred, probs_all, np.array(ids)


if __name__ == "__main__":
    main()
