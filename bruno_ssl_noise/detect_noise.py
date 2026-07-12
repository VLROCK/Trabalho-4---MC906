"""Etapa 4 — Roda os três detectores de ruído e mede a concordância.

Entradas (geradas nas etapas anteriores):
- eval_dir: contém ensemble_predictions.csv (ou <model>_predictions.csv) do
  evaluate.py -> usado pelo Confident Learning.
- ssl_backbone: backbone.pth do ssl_pretrain -> espaço de embeddings do kNN.
- train_dir: contém per_sample_loss.csv de um train.py -> loss-ranking.

Saídas em out_dir:
- cl_suspects.csv, knn_suspects.csv, loss_suspects.csv (rankings por método)
- agreement.json (Jaccard par a par)
- votes.csv (quantos métodos apontam cada imagem)

Exemplo:
    python detect_noise.py --data_dir $DATA --config configs/default.yaml \
        --eval_dir runs/eval --pred_prefix ensemble \
        --ssl_backbone runs/ssl/backbone.pth --train_dir runs/resnet50 \
        --out_dir runs/noise
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from core import agreement, confident_learning, knn_consistency, loss_ranking
from data.dataset import CassavaDataset
from data.transforms import build_transforms
from models.factory import build_backbone
from utils import extract_embeddings, get_device, load_config, set_seed, stratified_split


def parse_args():
    ap = argparse.ArgumentParser(description="Detecção de rótulos suspeitos (3 métodos)")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--eval_dir", type=str, required=True)
    ap.add_argument("--pred_prefix", type=str, default="ensemble",
                    help="prefixo do *_predictions.csv usado no Confident Learning")
    ap.add_argument("--ssl_backbone", type=str, required=True)
    ap.add_argument("--train_dir", type=str, required=True,
                    help="pasta com per_sample_loss.csv (saída do train.py)")
    ap.add_argument("--out_dir", type=str, default="runs/noise")
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def load_ssl_embeddings(cfg, args, device):
    """Extrai embeddings SSL para o MESMO split de validação usado na avaliação."""
    df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    img_dir = os.path.join(args.data_dir, "train_images")
    _, val_df = stratified_split(df, cfg["val_size"], cfg["seed"])

    tf = build_transforms(False, cfg["image_size"])
    ds = CassavaDataset(val_df, img_dir, transform=tf, return_id=True)
    loader = DataLoader(ds, batch_size=8 if args.smoke else 64, shuffle=False,
                        num_workers=0 if args.smoke else cfg["train"]["num_workers"])

    ckpt = torch.load(args.ssl_backbone, map_location=device)
    backbone, _ = build_backbone(ckpt["model_name"], pretrained=False)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    backbone.to(device)
    return extract_embeddings(backbone, loader, device)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    top_k = args.top_k or cfg["noise"]["top_k"]

    # ---- Detector 1: Confident Learning ----
    pred_path = os.path.join(args.eval_dir, f"{args.pred_prefix}_predictions.csv")
    pred_df = pd.read_csv(pred_path)
    cl = confident_learning.detect(pred_df, top_k=top_k)
    cl.to_csv(os.path.join(args.out_dir, "cl_suspects.csv"), index=False)
    print(f"[noise] Confident Learning: {len(cl)} suspeitas (top_k={top_k})")

    # ---- Detector 2: kNN-consistency no espaço SSL ----
    embs, labels, ids = load_ssl_embeddings(cfg, args, device)
    knn = knn_consistency.detect(embs, labels, ids, k=cfg["noise"]["knn_k"], top_k=top_k)
    knn.to_csv(os.path.join(args.out_dir, "knn_suspects.csv"), index=False)
    print(f"[noise] kNN-consistency: {len(knn)} suspeitas")

    # ---- Detector 3: Loss-ranking ----
    psl = pd.read_csv(os.path.join(args.train_dir, "per_sample_loss.csv"))
    lr = loss_ranking.detect(psl, top_k=top_k)
    lr.to_csv(os.path.join(args.out_dir, "loss_suspects.csv"), index=False)
    print(f"[noise] Loss-ranking: {len(lr)} suspeitas")

    # ---- Concordância entre os três ----
    # Loss-ranking cobre o conjunto de TREINO; CL e kNN cobrem a VALIDAÇÃO.
    # A interseção honesta acontece nas imagens presentes em ambos os universos,
    # então documentamos os três conjuntos e comparamos onde há sobreposição.
    sets = {
        "confident_learning": set(cl["image_id"]),
        "knn_consistency": set(knn["image_id"]),
        "loss_ranking": set(lr["image_id"]),
    }
    pairs = agreement.summarize(sets)
    with open(os.path.join(args.out_dir, "agreement.json"), "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    votes = agreement.vote_table(sets)
    votes.to_csv(os.path.join(args.out_dir, "votes.csv"), index=False)

    print("[noise] Jaccard par a par:")
    for k, v in pairs.items():
        print(f"    {k}: {v:.3f}")
    n_multi = int((votes["n_votes"] >= 2).sum()) if not votes.empty else 0
    print(f"[noise] {n_multi} imagens apontadas por >=2 métodos. Artefatos em {args.out_dir}")


if __name__ == "__main__":
    main()
