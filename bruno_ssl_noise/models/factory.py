"""Factory de backbones e classificadores (usando timm).

O ensemble do artigo combina arquiteturas diferentes (ResNet50, DenseNet121,
OSNet) apostando na "diversidade complementar". Aqui usamos ResNet50,
DenseNet121 e EfficientNet-B0 (OSNet é específico de ReID). Cada backbone é
treinado separadamente; o ensemble é feito na avaliação (média de
probabilidades), então este arquivo só precisa saber construir um modelo.
"""

from __future__ import annotations

import timm
import torch.nn as nn

# Aliases amigáveis -> nomes do timm.
ALIASES = {
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "efficientnet_b0": "efficientnet_b0",
    "resnet18": "resnet18",  # leve, útil para smoke-test
}


def build_backbone(model_name: str, pretrained: bool = True):
    """Backbone SEM cabeça de classificação (num_classes=0).

    Retorna (backbone, feat_dim). O backbone devolve um vetor de features
    (pooled) que serve tanto para a projection head do SSL quanto para o kNN.
    """
    timm_name = ALIASES.get(model_name.lower(), model_name)
    backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool="avg")
    feat_dim = backbone.num_features
    return backbone, feat_dim


class Classifier(nn.Module):
    """Backbone + camada linear de classificação."""

    def __init__(self, model_name: str, num_classes: int = 5, pretrained: bool = True, drop_rate: float = 0.0):
        super().__init__()
        self.backbone, feat_dim = build_backbone(model_name, pretrained=pretrained)
        self.dropout = nn.Dropout(drop_rate)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(self.dropout(feats))

    def features(self, x):
        """Embeddings do backbone (sem a cabeça) — usado pelo kNN de ruído."""
        return self.backbone(x)


def build_classifier(model_name: str, num_classes: int = 5, pretrained: bool = True, drop_rate: float = 0.0):
    return Classifier(model_name, num_classes=num_classes, pretrained=pretrained, drop_rate=drop_rate)
