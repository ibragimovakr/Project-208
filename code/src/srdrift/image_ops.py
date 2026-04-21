from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def upsample_lr(lr: torch.Tensor, scale: int = 4, out_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    if out_hw is not None:
        return F.interpolate(lr, size=out_hw, mode="bicubic", align_corners=False)
    return F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)


def degrade_to_lr(x_hr: torch.Tensor, scale: int = 4) -> torch.Tensor:
    h, w = x_hr.shape[-2:]
    return F.interpolate(x_hr, size=(h // scale, w // scale), mode="bicubic", align_corners=False)


def gaussian_kernel(size: int = 5, sigma: float = 1.0, channels: int = 3, device: str = "cpu", dtype=torch.float32) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    kernel2d = torch.outer(g, g)
    kernel2d = kernel2d / kernel2d.sum()
    return kernel2d.view(1, 1, size, size).repeat(channels, 1, 1, 1)


def blur_image(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    kernel = gaussian_kernel(kernel_size, sigma, channels=x.shape[1], device=x.device, dtype=x.dtype)
    pad = kernel_size // 2
    return F.conv2d(x, kernel, padding=pad, groups=x.shape[1])


def unsharp_mask(x: torch.Tensor, amount: float = 0.05, sigma: float = 1.0) -> torch.Tensor:
    blur = blur_image(x, kernel_size=5, sigma=sigma)
    return (x + amount * (x - blur)).clamp(0.0, 1.0)


def mild_highpass_boost(x: torch.Tensor, amount: float = 0.03) -> torch.Tensor:
    low = blur_image(x, kernel_size=5, sigma=1.0)
    hp = x - low
    return (x + amount * hp).clamp(0.0, 1.0)


def enforce_lr_consistency(candidate: torch.Tensor, fallback: torch.Tensor, lr: torch.Tensor, scale: int, tolerance: float):
    lr_cand = degrade_to_lr(candidate, scale=scale)
    keep = (lr_cand - lr).abs().mean(dim=(1, 2, 3), keepdim=True) <= tolerance
    return torch.where(keep, candidate, fallback), keep


def build_positive_bank(hr: torch.Tensor, lr: torch.Tensor, cfg) -> Tuple[torch.Tensor, Dict[str, float]]:
    positives = [hr]
    accepted = 0
    total = 0
    builders = [
        lambda x: unsharp_mask(
            x,
            amount=random.uniform(*cfg.positive_sharpen_amount),
            sigma=random.uniform(*cfg.positive_sharpen_sigma),
        ),
        lambda x: mild_highpass_boost(
            x,
            amount=random.uniform(*cfg.positive_highpass_amount),
        ),
        lambda x: mild_highpass_boost(
            unsharp_mask(
                x,
                amount=random.uniform(*cfg.positive_sharpen_amount),
                sigma=random.uniform(*cfg.positive_sharpen_sigma),
            ),
            amount=random.uniform(*cfg.positive_combo_highpass_amount),
        ),
    ]
    while len(positives) < cfg.num_positive_views:
        builder = builders[(len(positives) - 1) % len(builders)]
        cand = builder(hr)
        cand, keep = enforce_lr_consistency(cand, hr, lr, scale=cfg.scale, tolerance=cfg.positive_lr_tolerance)
        keep_flat = keep.view(-1)
        accepted += int(keep_flat.sum().item())
        total += int(keep_flat.numel())
        positives.append(cand)
    stats = {"accepted": accepted, "total": total, "accept_rate": float(accepted / max(1, total))}
    return torch.stack(positives[: cfg.num_positive_views], dim=1), stats
