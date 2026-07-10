import torch
import torch.nn as nn
import timm

class CassavaBaseline(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, freeze_backbone=False):
        super().__init__()
        
        # 1. Carrega a ResNet50 com pesos da ImageNet
        self.backbone = timm.create_model('resnet50', pretrained=pretrained)
        
        # 2. Opcional: Congela os pesos da rede base
        # Se True, a rede apenas treina a última camada (Feature Extraction)
        # Se False, a rede inteira se ajusta levemente ao Cassava (Fine-Tuning)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # 3. Troca o cabeçote de classificação
        # O timm padroniza a extração do número de features da última camada
        in_features = self.backbone.get_classifier().in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)