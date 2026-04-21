from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.amp import autocast

from srdrift.image_ops import degrade_to_lr
from srdrift.metrics import calc_lpips_sr, calc_psnr_sr


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state() -> Dict:
    out = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        out["torch_cuda_random"] = torch.cuda.get_rng_state_all()
    return out


def atomic_torch_save(obj, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_history_json(history: Dict, cfg, final_stats: Optional[Dict] = None, name: str = "history") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(cfg.output_root) / "histories" / f"{name}_{timestamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"config": cfg.to_dict(), "history": history, "final_stats": final_stats}, f, ensure_ascii=False, indent=2)
    return str(path)


def generate_multi_samples(model, lr: torch.Tensor, cfg, num_samples: Optional[int] = None):
    if num_samples is None:
        num_samples = cfg.num_samples_per_lr
    B, C, h_lr, w_lr = lr.shape
    H, W = h_lr * cfg.scale, w_lr * cfg.scale
    gcfg = cfg.generator
    lr_rep = lr[:, None].repeat(1, num_samples, 1, 1, 1).reshape(B * num_samples, C, h_lr, w_lr)
    noise_img = torch.randn(B * num_samples, gcfg.noise_image_channels, H, W, device=lr.device, dtype=lr.dtype)
    noise_vec = torch.randn(B * num_samples, gcfg.noise_embed_dim, device=lr.device, dtype=lr.dtype)
    x_hat_rep, x_up_rep = model(lr_rep, noise_img=noise_img, noise_vec=noise_vec)
    x_gen = x_hat_rep.view(B, num_samples, gcfg.in_channels, H, W)
    x_up = x_up_rep.view(B, num_samples, gcfg.in_channels, H, W)[:, 0]
    return x_gen, x_up


@torch.no_grad()
def sample_sr(model, lr: torch.Tensor, cfg, zero_noise: bool = True, return_up: bool = False):
    B = lr.shape[0]
    H = lr.shape[-2] * cfg.scale
    W = lr.shape[-1] * cfg.scale
    gcfg = cfg.generator
    if zero_noise:
        noise_img = torch.zeros(B, gcfg.noise_image_channels, H, W, device=lr.device, dtype=lr.dtype)
        noise_vec = torch.zeros(B, gcfg.noise_embed_dim, device=lr.device, dtype=lr.dtype)
    else:
        noise_img = torch.randn(B, gcfg.noise_image_channels, H, W, device=lr.device, dtype=lr.dtype)
        noise_vec = torch.randn(B, gcfg.noise_embed_dim, device=lr.device, dtype=lr.dtype)
    x_hat, x_up = model(lr, noise_img=noise_img, noise_vec=noise_vec)
    return (x_hat, x_up) if return_up else x_hat


@torch.no_grad()
def evaluate(model, loader, cfg, lpips_model=None, max_batches: int = 20, zero_noise: bool = True):
    model.eval()
    psnr_model, psnr_bic = [], []
    lpips_model_vals, lpips_bic_vals = [], []
    amp_device_type = "cuda" if "cuda" in cfg.device else "cpu"
    for i, (hr, lr) in enumerate(loader):
        if i >= max_batches:
            break
        hr = hr.to(cfg.device, non_blocking=True)
        lr = lr.to(cfg.device, non_blocking=True)
        with autocast(device_type=amp_device_type, enabled=(cfg.use_amp and "cuda" in cfg.device)):
            pred, up = sample_sr(model, lr, cfg, zero_noise=zero_noise, return_up=True)
            pred = pred.clamp(0.0, 1.0)
            up = up.clamp(0.0, 1.0)
        psnr_model.append(calc_psnr_sr(pred, hr, shave=cfg.shave_border, use_y=cfg.eval_on_y_channel))
        psnr_bic.append(calc_psnr_sr(up, hr, shave=cfg.shave_border, use_y=cfg.eval_on_y_channel))
        if lpips_model is not None:
            lpips_model_vals.append(calc_lpips_sr(pred, hr, lpips_model, shave=cfg.eval_lpips_shave_border))
            lpips_bic_vals.append(calc_lpips_sr(up, hr, lpips_model, shave=cfg.eval_lpips_shave_border))
    model.train()
    return {
        "psnr_model": float(np.mean(psnr_model)) if psnr_model else float("nan"),
        "psnr_bicubic": float(np.mean(psnr_bic)) if psnr_bic else float("nan"),
        "lpips_model": float(np.mean(lpips_model_vals)) if lpips_model_vals else float("nan"),
        "lpips_bicubic": float(np.mean(lpips_bic_vals)) if lpips_bic_vals else float("nan"),
    }
