"""Recorta uma amostra pequena e estratificada do Cassava REAL.

Serve para validar o fluxo com imagens reais (leitura de JPG, augmentations em
224px, backbones completos) sem processar as ~21k imagens. NÃO produz resultado
com significado estatístico — é teste de fluxo, mais fiel que o dummy sintético.

Uso:
    python data/make_sample.py --src /data/cassava --dst /data/cassava_sample \
        --per_class 60

Depois é só apontar o pipeline para --data_dir /data/cassava_sample:
    bash run_all.sh /data/cassava_sample runs_sample
(sem --smoke, mas com poucas épocas via configs/ ou --epochs, já valida real.)
"""

from __future__ import annotations

import argparse
import os
import shutil

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="pasta do Cassava real (com train.csv e train_images/)")
    ap.add_argument("--dst", type=str, required=True, help="pasta de saída da amostra")
    ap.add_argument("--per_class", type=int, default=60, help="imagens por classe")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src_csv = os.path.join(args.src, "train.csv")
    src_imgs = os.path.join(args.src, "train_images")
    if not os.path.exists(src_csv):
        raise FileNotFoundError(f"Não achei {src_csv}. Confira --src.")

    df = pd.read_csv(src_csv)
    # amostra estratificada: até per_class imagens de cada classe
    sample = (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), args.per_class), random_state=args.seed))
        .reset_index(drop=True)
    )

    dst_imgs = os.path.join(args.dst, "train_images")
    os.makedirs(dst_imgs, exist_ok=True)
    copied = 0
    for name in sample["image_id"]:
        src_path = os.path.join(src_imgs, name)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(dst_imgs, name))
            copied += 1

    sample.to_csv(os.path.join(args.dst, "train.csv"), index=False)
    print(f"Amostra criada em {args.dst}: {copied} imagens, "
          f"{sample['label'].nunique()} classes.")
    print("Distribuição por classe:")
    print(sample["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
