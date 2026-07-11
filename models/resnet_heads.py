import torch
import torch.nn as nn
from torchvision import models


class ResNet50FeatureExtractor(nn.Module):
    """
    ResNet50 pré-treinada no ImageNet usada apenas como extrator de features.

    Entrada:
        batch de imagens: [B, 3, 224, 224]

    Saída:
        features: [B, 2048]
    """

    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None

        resnet = models.resnet50(weights=weights)

        # Remove a última camada fully connected.
        # Mantém convoluções + avgpool.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Congela todos os parâmetros da ResNet.
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)          # [B, 2048, 1, 1]
        features = torch.flatten(features, 1)  # [B, 2048]
        return features


class MLPHead(nn.Module):
    """
    Head densa simples para classificar as features 2048D da ResNet50.
    """

    def __init__(self, input_dim=2048, hidden_dim=512, num_classes=5, dropout=0.3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)