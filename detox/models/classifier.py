"""ResNet-18 classifier for gender / age-group (paper LightCNN/ResNet substitute)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class FaceClassifier(nn.Module):
    def __init__(self, n_classes: int, emb_dim: int = 128, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(net.children())[:-1])
        self.fc = nn.Linear(512, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.classifier = nn.Linear(emb_dim, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(self.features(x), 1)
        return self.bn(self.fc(x))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embed(x)
        return self.classifier(emb), emb
