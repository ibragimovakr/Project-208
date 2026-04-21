from __future__ import annotations

import torch
import torch.nn.functional as F


def rgb_to_y_channel_torch(x: torch.Tensor) -> torch.Tensor:
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def shave_tensor(x: torch.Tensor, border: int) -> torch.Tensor:
    if border <= 0:
        return x
    h, w = x.shape[-2:]
    if h <= 2 * border or w <= 2 * border:
        return x
    return x[..., border : h - border, border : w - border]


@torch.no_grad()
def calc_psnr_sr(x: torch.Tensor, y: torch.Tensor, shave: int = 4, use_y: bool = True) -> float:
    if use_y:
        x = rgb_to_y_channel_torch(x)
        y = rgb_to_y_channel_torch(y)
    x = shave_tensor(x, shave)
    y = shave_tensor(y, shave)
    mse = F.mse_loss(x, y)
    return (-10.0 * torch.log10(mse + 1e-12)).item()


@torch.no_grad()
def calc_lpips_sr(x: torch.Tensor, y: torch.Tensor, lpips_model, shave: int = 4) -> float:
    x = shave_tensor(x.clamp(0.0, 1.0), shave)
    y = shave_tensor(y.clamp(0.0, 1.0), shave)
    x = x * 2.0 - 1.0
    y = y * 2.0 - 1.0
    return float(lpips_model(x, y).mean().item())
