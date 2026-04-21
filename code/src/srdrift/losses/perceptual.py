from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize_to: int = 64):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1
        feats = models.vgg16(weights=weights).features[:16]
        for p in feats.parameters():
            p.requires_grad = False
        self.features = feats.eval()
        self.resize_to = resize_to

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.resize_to is not None:
            pred = torch.nn.functional.interpolate(pred, size=(self.resize_to, self.resize_to), mode="bilinear", align_corners=False)
            target = torch.nn.functional.interpolate(target, size=(self.resize_to, self.resize_to), mode="bilinear", align_corners=False)
        return torch.nn.functional.l1_loss(self.features(pred), self.features(target))
