"""Factory de modelos para facilitar experimentação.

Usa timm para ResNet/EfficientNet/ConvNeXt/Swin/ViT e um híbrido próprio
CNN+Transformer.
"""

from __future__ import annotations

import timm

from models.cnn_transformer import ResNetTransformerClassifier


def build_model(
    model_name: str,
    num_classes: int = 5,
    pretrained: bool = True,
    image_size: int = 224,
    drop_rate: float = 0.0,
):
    name = model_name.lower()

    aliases = {
        "resnet50": "resnet50",
        "resnet18": "resnet18",
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet_b3": "efficientnet_b3",
        "convnext_tiny": "convnext_tiny",
        "swin_tiny": "swin_tiny_patch4_window7_224",
        "vit_small": "vit_small_patch16_224",
    }

    if name == "cnn_transformer_resnet18":
        return ResNetTransformerClassifier(
            backbone_name="resnet18",
            num_classes=num_classes,
            pretrained=pretrained,
            image_size=image_size,
            embed_dim=384,
            num_heads=6,
            depth=2,
            dropout=drop_rate,
        )

    if name == "cnn_transformer_resnet50":
        return ResNetTransformerClassifier(
            backbone_name="resnet50",
            num_classes=num_classes,
            pretrained=pretrained,
            image_size=image_size,
            embed_dim=512,
            num_heads=8,
            depth=2,
            dropout=drop_rate,
        )

    timm_name = aliases.get(name, model_name)

    return timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
