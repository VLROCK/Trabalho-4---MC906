"""Plots genéricos de métricas.

Exemplos:
python plot_results.py --experiment_dirs /content/drive/MyDrive/experimentos/*
python plot_results.py --metrics_csv exp1/metrics.csv exp2/metrics.csv --labels resnet focal
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Plotagem de resultados")
    parser.add_argument("--experiment_dirs", nargs="*", default=None)
    parser.add_argument("--metrics_csv", nargs="*", default=None)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--out_dir", type=str, default="plots")
    return parser.parse_args()


def collect_metrics(args):
    paths = []

    if args.metrics_csv:
        paths.extend(args.metrics_csv)

    if args.experiment_dirs:
        for pattern in args.experiment_dirs:
            for d in glob.glob(pattern):
                candidate = os.path.join(d, "metrics.csv")
                if os.path.exists(candidate):
                    paths.append(candidate)

    if not paths:
        raise FileNotFoundError("Nenhum metrics.csv encontrado.")

    frames = []
    for i, path in enumerate(paths):
        df = pd.read_csv(path)
        if args.labels and i < len(args.labels):
            name = args.labels[i]
        else:
            name = os.path.basename(os.path.dirname(path)) or f"exp_{i}"
        df["experiment"] = name
        df["metrics_path"] = path
        frames.append(df)

    return pd.concat(frames, ignore_index=True), paths


def plot_metric(df, metric, out_dir):
    if metric not in df.columns:
        return

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="epoch", y=metric, hue="experiment", marker="o")
    plt.title(metric)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{metric}.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Salvo: {out_path}")


def plot_confusion_matrices(paths, out_dir):
    for path in paths:
        exp_dir = os.path.dirname(path)
        exp_name = os.path.basename(exp_dir)
        for cm_name in ["val_confusion_matrix.csv", "val_tta_confusion_matrix.csv"]:
            cm_path = os.path.join(exp_dir, "eval_val", cm_name)
            if not os.path.exists(cm_path):
                cm_path = os.path.join(exp_dir, cm_name)
            if not os.path.exists(cm_path):
                continue

            cm = pd.read_csv(cm_path, index_col=0)
            plt.figure(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt="g", cmap="Blues")
            plt.title(f"Matriz de confusão - {exp_name}")
            plt.ylabel("Verdadeiro")
            plt.xlabel("Predito")
            plt.tight_layout()
            out_path = os.path.join(out_dir, f"confusion_{exp_name}_{cm_name.replace('.csv','')}.png")
            plt.savefig(out_path, dpi=160)
            plt.close()
            print(f"Salvo: {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df, paths = collect_metrics(args)
    df.to_csv(os.path.join(args.out_dir, "all_metrics.csv"), index=False)

    for metric in [
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "train_macro_f1",
        "val_macro_f1",
        "val_balanced_acc",
    ]:
        plot_metric(df, metric, args.out_dir)

    # Tabela resumo com melhor época por macro-F1.
    summaries = []
    for exp, g in df.groupby("experiment"):
        if "val_macro_f1" in g.columns:
            best = g.sort_values("val_macro_f1", ascending=False).iloc[0]
        else:
            best = g.sort_values("val_acc", ascending=False).iloc[0]
        summaries.append(best)
    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(args.out_dir, "summary_best_epochs.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Resumo salvo: {summary_path}")

    plot_confusion_matrices(paths, args.out_dir)


if __name__ == "__main__":
    main()
