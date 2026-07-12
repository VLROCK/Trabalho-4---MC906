import torch.nn as nn
from torchvision import models

try:
    import timm
except ImportError:
    timm = None

try:
    from models.cnn_transformer import CNNTransformerClassifier
except ImportError:
    CNNTransformerClassifier = None


def build_model(
    model_name: str,
    num_classes: int = 5,
    pretrained: bool = True,
    image_size: int = 224,
    drop_rate: float = 0.0,
):
    """
    Factory de modelos para o train_experiments.py.

    Usa torchvision para modelos principais, evitando Hugging Face/timm
    quando o cluster não tem internet dentro do salloc.
    """

    model_name = model_name.lower()

    # =========================
    # ResNet50 - torchvision
    # =========================
    if model_name == "resnet50":
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None

        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    # =========================
    # ResNet18 - torchvision
    # =========================
    if model_name == "resnet18":
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    # =========================
    # EfficientNet-B0 - torchvision
    # =========================
    if model_name == "efficientnet_b0":
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None

        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    # =========================
    # EfficientNet-B3 - torchvision
    # =========================
    if model_name == "efficientnet_b3":
        if pretrained:
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
        else:
            weights = None

        model = models.efficientnet_b3(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    # =========================
    # ConvNeXt-Tiny - torchvision
    # =========================
    if model_name == "convnext_tiny":
        if pretrained:
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        else:
            weights = None

        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    # =========================
    # CNN + Transformer custom
    # =========================
    if model_name == "cnn_transformer_resnet18":
        if CNNTransformerClassifier is None:
            raise ImportError(
                "CNNTransformerClassifier não encontrado em models/cnn_transformer.py"
            )

        return CNNTransformerClassifier(
            backbone_name="resnet18",
            num_classes=num_classes,
            pretrained=pretrained,
        )

    if model_name == "cnn_transformer_resnet50":
        if CNNTransformerClassifier is None:
            raise ImportError(
                "CNNTransformerClassifier não encontrado em models/cnn_transformer.py"
            )

        return CNNTransformerClassifier(
            backbone_name="resnet50",
            num_classes=num_classes,
            pretrained=pretrained,
        )

    # =========================
    # Fallback para timm
    # =========================
    if timm is None:
        raise ImportError(
            f"Modelo '{model_name}' não está implementado em torchvision "
            "e timm não está instalado."
        )

    print(
        f"Aviso: usando timm para o modelo '{model_name}'. "
        "Se pretrained=True, isso pode tentar baixar pesos da internet."
    )

    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )