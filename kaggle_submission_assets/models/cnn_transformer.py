"""Modelo híbrido CNN + Transformer.

A CNN extrai um mapa espacial de características; o Transformer trata cada posição
espacial como token e modela relações globais entre regiões da folha.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResNetTransformerClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = 5,
        pretrained: bool = True,
        image_size: int = 224,
        embed_dim: int = 512,
        num_heads: int = 8,
        depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
            cnn_dim = 512
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = models.resnet50(weights=weights)
            cnn_dim = 2048
        else:
            raise ValueError("backbone_name deve ser 'resnet18' ou 'resnet50'.")

        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        self.proj = nn.Linear(cnn_dim, embed_dim)

        approx_grid = max(1, image_size // 32)
        self.pos_embed = nn.Parameter(torch.zeros(1, approx_grid * approx_grid, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _get_pos_embed(self, n_tokens: int):
        if self.pos_embed.size(1) == n_tokens:
            return self.pos_embed

        # Interpola embedding posicional quando image_size gera número de tokens diferente.
        old_tokens = self.pos_embed.size(1)
        old_grid = int(old_tokens ** 0.5)
        new_grid = int(n_tokens ** 0.5)

        pos = self.pos_embed.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
        pos = nn.functional.interpolate(pos, size=(new_grid, new_grid), mode="bilinear", align_corners=False)
        pos = pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)
        return pos

    def forward(self, x):
        fmap = self.cnn(x)  # [B, C, H, W]
        tokens = fmap.flatten(2).transpose(1, 2)  # [B, N, C]
        tokens = self.proj(tokens)
        tokens = tokens + self._get_pos_embed(tokens.size(1)).to(tokens.device)
        tokens = self.encoder(tokens)
        pooled = self.norm(tokens.mean(dim=1))
        return self.classifier(pooled)
