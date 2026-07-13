"""Gera um train.csv limpo removendo imagens suspeitas.

Exemplo:
python make_clean_train_csv.py \
  --train_csv "data/cassava_data/train.csv" \
  --issues_csv "pesos/experimentos/ensemble_top3_val/confident_learning/label_issues_cleanlab.csv" \
  --out_csv "data/cassava_data/train_clean_top200.csv" \
  --top_k 200
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove top-k possiveis labels ruidosas do train.csv.")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--issues_csv", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train_csv)
    issues_df = pd.read_csv(args.issues_csv).head(args.top_k)

    if "image_id" not in train_df.columns:
        raise ValueError("train_csv precisa ter coluna image_id.")
    if "image_id" not in issues_df.columns:
        raise ValueError("issues_csv precisa ter coluna image_id.")

    remove_ids = set(issues_df["image_id"].values.tolist())
    clean_df = train_df[~train_df["image_id"].isin(remove_ids)].reset_index(drop=True)

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    clean_df.to_csv(args.out_csv, index=False)

    print(f"Train original: {len(train_df)}")
    print(f"Removidos: {len(train_df) - len(clean_df)}")
    print(f"Train limpo: {len(clean_df)}")
    print(f"Arquivo salvo em: {args.out_csv}")


if __name__ == "__main__":
    main()
