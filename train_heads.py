import os
import argparse
import random
import joblib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from torch.utils.data import DataLoader, TensorDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import CassavaDataset
from models.resnet_heads import ResNet50FeatureExtractor, MLPHead


def parse_args():
    parser = argparse.ArgumentParser(
        description="ResNet50 feature extractor + heads: SVM, MLP e XGBoost"
    )

    parser.add_argument("--data_dir", type=str, default="data/cassava_data")
    parser.add_argument("--save_path", type=str, default="checkpoints_heads")

    parser.add_argument("--head", type=str, default="all",
                        choices=["svm", "mlp", "xgb", "all"])

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mlp_epochs", type=int, default=30)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--mlp_dropout", type=float, default=0.3)

    parser.add_argument("--force_extract", action="store_true",
                        help="Força reextração das features mesmo se já existir cache.")

    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


def get_feature_transform():
    """
    Para extração de features, usamos transform determinística.
    Não usamos augmentation aqui porque queremos uma representação fixa da imagem.
    """
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])


def extract_features(model, loader, device):
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extraindo features"):
            images = images.to(device, non_blocking=True)

            features = model(images)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_features, all_labels


def prepare_features(args, device):
    os.makedirs(args.save_path, exist_ok=True)

    feature_cache_path = os.path.join(args.save_path, "resnet50_features.npz")

    if os.path.exists(feature_cache_path) and not args.force_extract:
        print(f"Carregando features salvas de: {feature_cache_path}")

        data = np.load(feature_cache_path)

        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]

        print(f"X_train: {X_train.shape}")
        print(f"X_val: {X_val.shape}")

        return X_train, y_train, X_val, y_val

    csv_path = os.path.join(args.data_dir, "train.csv")
    img_dir = os.path.join(args.data_dir, "train_images")

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"data_dir não encontrado: {args.data_dir}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"train.csv não encontrado em: {csv_path}")

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"train_images não encontrado em: {img_dir}")

    df = pd.read_csv(csv_path)

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=df["label"]
    )

    print(f"Total de imagens: {len(df)}")
    print(f"Treino: {len(train_df)}")
    print(f"Validação: {len(val_df)}")

    transform = get_feature_transform()

    train_dataset = CassavaDataset(
        df=train_df,
        img_dir=img_dir,
        transform=transform
    )

    val_dataset = CassavaDataset(
        df=val_df,
        img_dir=img_dir,
        transform=transform
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    feature_extractor = ResNet50FeatureExtractor(pretrained=True).to(device)

    print("\nExtraindo features de treino...")
    X_train, y_train = extract_features(feature_extractor, train_loader, device)

    print("\nExtraindo features de validação...")
    X_val, y_val = extract_features(feature_extractor, val_loader, device)

    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")

    np.savez_compressed(
        feature_cache_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )

    print(f"Features salvas em: {feature_cache_path}")

    return X_train, y_train, X_val, y_val


def train_svm(X_train, y_train, X_val, y_val, save_path):
    print("\n==============================")
    print("Treinando head SVM linear")
    print("==============================")

    model = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=1.0,
            max_iter=10000,
            class_weight="balanced"
        )
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)

    print(f"SVM Val Acc: {acc * 100:.2f}%")
    print(classification_report(y_val, preds))

    model_path = os.path.join(save_path, "svm_head.joblib")
    joblib.dump(model, model_path)

    print(f"SVM salvo em: {model_path}")

    return acc


def train_xgboost(X_train, y_train, X_val, y_val, save_path):
    print("\n==============================")
    print("Treinando head XGBoost")
    print("==============================")

    if not XGBOOST_AVAILABLE:
        raise ImportError(
            "XGBoost não está instalado. Adicione xgboost ao requirements.txt."
        )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=5,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)

    print(f"XGBoost Val Acc: {acc * 100:.2f}%")
    print(classification_report(y_val, preds))

    model_path = os.path.join(save_path, "xgboost_head.joblib")
    joblib.dump(model, model_path)

    print(f"XGBoost salvo em: {model_path}")

    return acc


def train_mlp(X_train, y_train, X_val, y_val, args, device):
    print("\n==============================")
    print("Treinando head MLP")
    print("==============================")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = MLPHead(
        input_dim=X_train.shape[1],
        hidden_dim=args.mlp_hidden_dim,
        num_classes=5,
        dropout=args.mlp_dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.mlp_lr,
        weight_decay=1e-3
    )

    best_val_acc = 0.0
    best_model_path = os.path.join(args.save_path, "mlp_head.pth")

    for epoch in range(args.mlp_epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"MLP Epoch {epoch + 1}/{args.mlp_epochs}")

        for features, labels in loop:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            loop.set_postfix(
                loss=running_loss / total,
                acc=100.0 * correct / total
            )

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        val_loss, val_acc = evaluate_mlp(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{args.mlp_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "input_dim": X_train.shape[1],
                "hidden_dim": args.mlp_hidden_dim,
                "num_classes": 5,
                "dropout": args.mlp_dropout,
                "args": vars(args)
            }

            torch.save(checkpoint, best_model_path)

            print(f"Novo melhor MLP salvo em: {best_model_path}")

    print(f"Melhor MLP Val Acc: {best_val_acc:.2f}%")

    return best_val_acc / 100.0


def evaluate_mlp(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

    return running_loss / total, 100.0 * correct / total


def main():
    args = parse_args()
    seed_everything(args.seed)

    os.makedirs(args.save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Dispositivo: {device}")

    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

    print("\nConfigurações:")
    print(f"data_dir: {args.data_dir}")
    print(f"save_path: {args.save_path}")
    print(f"head: {args.head}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_workers: {args.num_workers}")
    print(f"val_size: {args.val_size}")
    print(f"seed: {args.seed}")
    print(f"mlp_epochs: {args.mlp_epochs}")
    print()

    X_train, y_train, X_val, y_val = prepare_features(args, device)

    results = {}

    if args.head in ["svm", "all"]:
        svm_acc = train_svm(X_train, y_train, X_val, y_val, args.save_path)
        results["svm"] = svm_acc

    if args.head in ["xgb", "all"]:
        xgb_acc = train_xgboost(X_train, y_train, X_val, y_val, args.save_path)
        results["xgboost"] = xgb_acc

    if args.head in ["mlp", "all"]:
        mlp_acc = train_mlp(X_train, y_train, X_val, y_val, args, device)
        results["mlp"] = mlp_acc

    print("\n==============================")
    print("Resumo final")
    print("==============================")

    for name, acc in results.items():
        print(f"{name}: {acc * 100:.2f}%")

    results_path = os.path.join(args.save_path, "head_results.csv")

    results_df = pd.DataFrame([
        {"head": name, "val_acc": acc}
        for name, acc in results.items()
    ])

    results_df.to_csv(results_path, index=False)

    print(f"\nResultados salvos em: {results_path}")


if __name__ == "__main__":
    main()