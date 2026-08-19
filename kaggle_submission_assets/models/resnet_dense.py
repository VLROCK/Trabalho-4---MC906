import torch.nn as nn
from torchvision import models


class ResNet50DenseHead(nn.Module):
    def __init__(
        self,
        num_classes=5,
        pretrained=True,
        hidden_dim=512,
        dropout=0.4,
        freeze_backbone=False
    ):
        super().__init__()

        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None

        self.model = models.resnet50(weights=weights)

        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)