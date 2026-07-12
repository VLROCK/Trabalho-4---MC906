"""Gera um dataset Cassava sintético para smoke-test (sem baixar nada).

Cria imagens coloridas aleatórias e um train.csv com o mesmo formato do Kaggle
(colunas 'image_id' e 'label', 5 classes). Injeta propositalmente alguns
rótulos errados para que os detectores de ruído tenham o que encontrar.

Uso:
    python data/make_dummy.py --out_dir /tmp/cassava_dummy --n 60
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n", type=int, default=60, help="número de imagens")
    ap.add_argument("--num_classes", type=int, default=5)
    ap.add_argument("--noise_frac", type=float, default=0.1, help="fração de rótulos corrompidos")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    img_dir = os.path.join(args.out_dir, "train_images")
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    for i in range(args.n):
        true_label = int(rng.integers(0, args.num_classes))
        # Dá a cada classe uma cor de fundo dominante para haver sinal aprendível.
        base = np.zeros((64, 64, 3), dtype=np.uint8)
        base[..., true_label % 3] = int(120 + 20 * true_label)
        noise = rng.integers(0, 60, size=(64, 64, 3), dtype=np.uint8)
        img = np.clip(base + noise, 0, 255).astype(np.uint8)

        name = f"dummy_{i:04d}.jpg"
        Image.fromarray(img).save(os.path.join(img_dir, name))
        rows.append({"image_id": name, "label": true_label, "true_label": true_label})

    df = pd.DataFrame(rows)

    # Corrompe alguns rótulos e marca quais (para validar os detectores).
    n_noisy = max(1, int(args.noise_frac * len(df)))
    noisy_idx = rng.choice(len(df), size=n_noisy, replace=False)
    for idx in noisy_idx:
        wrong = int((df.loc[idx, "true_label"] + rng.integers(1, args.num_classes)) % args.num_classes)
        df.loc[idx, "label"] = wrong

    # train.csv tem o formato do Kaggle (só image_id + label).
    df[["image_id", "label"]].to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    # ground_truth.csv guarda o gabarito para checar os detectores no smoke-test.
    df.to_csv(os.path.join(args.out_dir, "ground_truth.csv"), index=False)

    print(f"Dataset dummy criado em {args.out_dir} ({len(df)} imagens, {n_noisy} rótulos corrompidos).")


if __name__ == "__main__":
    main()
