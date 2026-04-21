from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torchvision import models


class FrozenVGGFeatureMaps(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg = models.vgg16(weights=weights).features
        self.blocks = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:16], vgg[16:23]])
        for p in self.blocks.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        h = x
        for block in self.blocks:
            h = block(h)
            outs.append(h)
        return outs
