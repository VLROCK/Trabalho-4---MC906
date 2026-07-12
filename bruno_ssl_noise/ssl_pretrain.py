"""Etapa 1 — Pré-treino self-supervised (Barlow Twins) no próprio Cassava.

Análogo ao pré-treino de backbone do artigo: aprende bons embeddings SEM usar
os rótulos, para (a) dar um bom ponto de partida ao fine-tuning supervisionado e
(b) fornecer um espaço de features para o kNN de detecção de ruído.

Salva apenas o backbone (state_dict) em <out_dir>/backbone.pth.

Exemplo (SLURM-friendly):
    python ssl_pretrain.py --config configs/default.yaml --data_dir $DATA \
        --out_dir runs/ssl
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import TwoViewDataset
from data.transforms import build_ssl_transforms
from models.barlow_twins import BarlowTwins, barlow_twins_loss
from models.factory import build_backbone
from utils import get_device, load_config, save_checkpoint, set_seed


def parse_args():
    ap = argparse.ArgumentParser(description="Barlow Twins pretrain no Cassava")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs/ssl")
    ap.add_argument("--model", type=str, default=None, help="sobrescreve ssl.backbone")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_pretrained", action="store_true", help="não usar pesos ImageNet")
    ap.add_argument("--smoke", action="store_true", help="1 época, para teste rápido")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    model_name = args.model or cfg["ssl"]["backbone"]
    epochs = args.epochs or cfg["ssl"]["epochs"]
    batch_size = args.batch_size or cfg["ssl"]["batch_size"]
    if args.smoke:
        epochs, batch_size = 1, min(batch_size, 8)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[SSL] backbone={model_name} epochs={epochs} batch={batch_size} device={device}")

    # Dados: usa todas as imagens de train.csv (sem rótulos).
    df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    img_dir = os.path.join(args.data_dir, "train_images")
    transform = build_ssl_transforms(cfg["image_size"])
    dataset = TwoViewDataset(df, img_dir=img_dir, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=True,
    )

    backbone, feat_dim = build_backbone(model_name, pretrained=not args.no_pretrained)
    model = BarlowTwins(backbone, feat_dim, proj_dim=cfg["ssl"]["proj_dim"]).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["ssl"]["lr"], weight_decay=cfg["ssl"]["weight_decay"]
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for view1, view2 in tqdm(loader, desc=f"SSL epoch {epoch}/{epochs}"):
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            z1, z2 = model(view1, view2)
            loss = barlow_twins_loss(z1, z2, lambda_bt=cfg["ssl"]["lambda_bt"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        print(f"[SSL] epoch {epoch}: loss={running / max(1, len(loader)):.4f}")

    # Salva só o backbone (é o que o fine-tuning e o kNN vão reutilizar).
    backbone_path = os.path.join(args.out_dir, "backbone.pth")
    torch.save({"model_name": model_name, "backbone_state_dict": model.backbone.state_dict()}, backbone_path)
    save_checkpoint(os.path.join(args.out_dir, "ssl_full.pth"), model, optimizer, epochs,
                    extra={"model_name": model_name})
    print(f"[SSL] backbone salvo em {backbone_path}")


if __name__ == "__main__":
    main()
