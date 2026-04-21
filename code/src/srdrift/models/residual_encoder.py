from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def maybe_sn(module: nn.Module, use_sn: bool) -> nn.Module:
    return nn.utils.spectral_norm(module) if use_sn else module


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_sn: bool = False):
        super().__init__()
        self.conv1 = maybe_sn(nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1), use_sn)
        self.conv2 = maybe_sn(nn.Conv2d(out_ch, out_ch, 3, padding=1), use_sn)
        self.norm1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = maybe_sn(nn.Conv2d(in_ch, out_ch, 1, stride=stride), use_sn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.act(self.norm1(h))
        h = self.conv2(h)
        h = self.norm2(h)
        return self.act(h + self.skip(x))


class ResidualDiscriminatorEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 64, use_sn: bool = False):
        super().__init__()
        self.stem = maybe_sn(nn.Conv2d(in_ch, base_ch, 3, padding=1), use_sn)
        self.body = nn.Sequential(
            ResBlock(base_ch, base_ch, stride=1, use_sn=use_sn),
            ResBlock(base_ch, base_ch * 2, stride=2, use_sn=use_sn),
            ResBlock(base_ch * 2, base_ch * 4, stride=2, use_sn=use_sn),
            ResBlock(base_ch * 4, base_ch * 8, stride=2, use_sn=use_sn),
        )
        self.out_ch = base_ch * 8

    def forward(self, x: torch.Tensor):
        h = self.stem(x)
        h = self.body(h)
        pooled = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return h, pooled


class ResidualDiscClassifier(nn.Module):
    def __init__(self, encoder: ResidualDiscriminatorEncoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.out_ch, 1)

    def forward(self, x: torch.Tensor):
        fmap, pooled = self.encoder(x)
        logits = self.head(pooled)
        return logits, fmap, pooled
