from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from srdrift.image_ops import upsample_lr


class ZeroConv2d(nn.Conv2d):
    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class NoiseMLP(nn.Module):
    def __init__(self, noise_embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AdaGNResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, up: bool = False, down: bool = False):
        super().__init__()
        self.up = up
        self.down = down
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, 4 * out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        self.act = nn.SiLU()

    def _resample(self, x: torch.Tensor) -> torch.Tensor:
        if self.up:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if self.down:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self._resample(x)
        x_skip = self._resample(x)
        h = self.conv1(self.act(self.norm1(h)))
        scale1, shift1, scale2, shift2 = self.cond_proj(cond).chunk(4, dim=1)
        h = self.norm2(h)
        h = h * (1 + scale1[:, :, None, None]) + shift1[:, :, None, None]
        h = self.conv2(self.act(h))
        h = h * (1 + scale2[:, :, None, None]) + shift2[:, :, None, None]
        return h + self.skip(x_skip)


class NoiseConditionalResidualUNetSR(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        noise_image_channels: int = 3,
        base: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        num_blocks: int = 2,
        noise_embed_dim: int = 256,
        residual_out_scale: float = 1.0,
        scale: int = 4,
    ):
        super().__init__()
        self.noise_image_channels = noise_image_channels
        self.noise_embed_dim = noise_embed_dim
        self.residual_out_scale = residual_out_scale
        self.scale = scale

        self.cond_mlp = NoiseMLP(noise_embed_dim, base * 4)
        cond_dim = base * 4
        self.in_conv = nn.Conv2d(in_channels + noise_image_channels, base, 3, padding=1)

        dims = [base * m for m in channel_mult]
        self.downs = nn.ModuleList()
        cur_ch = base
        skip_channels = []
        for level, ch in enumerate(dims):
            if level == 0:
                for _ in range(num_blocks):
                    block = AdaGNResBlock(cur_ch, ch, cond_dim)
                    self.downs.append(block)
                    cur_ch = ch
                    skip_channels.append(cur_ch)
            else:
                block = AdaGNResBlock(cur_ch, ch, cond_dim, down=True)
                self.downs.append(block)
                cur_ch = ch
                skip_channels.append(cur_ch)
                for _ in range(num_blocks - 1):
                    block = AdaGNResBlock(cur_ch, ch, cond_dim)
                    self.downs.append(block)
                    cur_ch = ch
                    skip_channels.append(cur_ch)

        self.mid1 = AdaGNResBlock(cur_ch, cur_ch, cond_dim)
        self.mid2 = AdaGNResBlock(cur_ch, cur_ch, cond_dim)

        self.ups = nn.ModuleList()
        rev_dims = list(reversed(dims))
        rev_skips = list(reversed(skip_channels))
        for level, ch in enumerate(rev_dims):
            for block_idx in range(num_blocks):
                skip_ch = rev_skips.pop(0)
                in_ch = cur_ch + skip_ch
                up_flag = block_idx == 0 and level > 0
                self.ups.append(AdaGNResBlock(in_ch, ch, cond_dim, up=up_flag))
                cur_ch = ch

        self.out_norm = nn.GroupNorm(min(32, cur_ch), cur_ch)
        self.out_act = nn.SiLU()
        self.out_conv = ZeroConv2d(cur_ch, in_channels, 3, padding=1)

    def forward(
        self,
        lr: torch.Tensor,
        noise_img: Optional[torch.Tensor] = None,
        noise_vec: Optional[torch.Tensor] = None,
    ):
        b, _, h_lr, w_lr = lr.shape
        H, W = h_lr * self.scale, w_lr * self.scale
        x_up = upsample_lr(lr, out_hw=(H, W), scale=self.scale)
        if noise_img is None:
            noise_img = torch.randn(b, self.noise_image_channels, H, W, device=lr.device, dtype=lr.dtype)
        if noise_vec is None:
            noise_vec = torch.randn(b, self.noise_embed_dim, device=lr.device, dtype=lr.dtype)

        cond = self.cond_mlp(noise_vec)
        x = self.in_conv(torch.cat([x_up, noise_img], dim=1))
        skips = []
        for block in self.downs:
            x = block(x, cond)
            skips.append(x)
        x = self.mid1(x, cond)
        x = self.mid2(x, cond)
        for block in self.ups:
            skip = skips.pop()
            if block.up:
                x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            old_up = block.up
            block.up = False
            x = block(x, cond)
            block.up = old_up
        residual = self.out_conv(self.out_act(self.out_norm(x)))
        return x_up + self.residual_out_scale * residual, x_up
