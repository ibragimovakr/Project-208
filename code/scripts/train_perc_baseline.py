from __future__ import annotations

import argparse
import gc
import json
import os

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from srdrift.config import BaselineConfig
from srdrift.data import DIV2KPairDataset
from srdrift.image_ops import degrade_to_lr
from srdrift.losses.drifting import EMA
from srdrift.losses.perceptual import VGGPerceptualLoss
from srdrift.models.generator import NoiseConditionalResidualUNetSR
from srdrift.utils.common import atomic_torch_save, evaluate, get_rng_state, sample_sr, save_history_json, set_seed


def save_checkpoint(model, ema, optimizer, scheduler, scaler, epoch, global_step, best_psnr, best_lpips, history, cfg, tag: str):
    path = os.path.join(cfg.output_root, "checkpoints", f"{tag}.pt")
    atomic_torch_save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_psnr": best_psnr,
            "best_lpips": best_lpips,
            "model_state": model.state_dict(),
            "ema_state": ema.shadow.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "history": history,
            "config": cfg.to_dict(),
            "rng_state": get_rng_state(),
        },
        path,
    )
    print(f"[ckpt] saved: {path}")


def build_model(cfg: BaselineConfig):
    g = cfg.generator
    return NoiseConditionalResidualUNetSR(
        in_channels=g.in_channels,
        noise_image_channels=g.noise_image_channels,
        base=g.unet_base,
        channel_mult=g.unet_channel_mult,
        num_blocks=g.unet_num_blocks,
        noise_embed_dim=g.noise_embed_dim,
        residual_out_scale=g.residual_out_scale,
        scale=cfg.scale,
    )


def train(cfg: BaselineConfig):
    set_seed(cfg.seed)
    cfg.ensure_output_dirs()
    train_ds = DIV2KPairDataset(cfg.data.train_hr_dir, cfg.data.train_lr_dir, scale=cfg.scale, patch_size=cfg.patch_size, training=True)
    val_ds = DIV2KPairDataset(cfg.data.val_hr_dir, cfg.data.val_lr_dir, scale=cfg.scale, patch_size=None, training=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers > 0), prefetch_factor=2 if cfg.num_workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=cfg.val_batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers > 0), prefetch_factor=2 if cfg.num_workers > 0 else None)

    model = build_model(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
    amp_device_type = "cuda" if "cuda" in cfg.device else "cpu"
    scaler = GradScaler(device=amp_device_type, enabled=(cfg.use_amp and "cuda" in cfg.device))
    ema = EMA(model, decay=cfg.ema_decay)
    perceptual = VGGPerceptualLoss(resize_to=cfg.perceptual_train_size).to(cfg.device).eval()
    lpips_eval = lpips.LPIPS(net=cfg.eval_lpips_net).to(cfg.device).eval()
    for p in lpips_eval.parameters():
        p.requires_grad = False

    history = {"train_total": [], "train_pix": [], "train_perc": [], "train_lr_cons": [], "val_psnr": [], "val_psnr_bicubic": [], "val_lpips": [], "val_lpips_bicubic": []}
    best_psnr = -float("inf")
    best_lpips = float("inf")
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        running_total, running_pix, running_perc, running_lr_cons = [], [], [], []
        model.train()
        for hr, lr in train_loader:
            hr = hr.to(cfg.device, non_blocking=True)
            lr = lr.to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=amp_device_type, enabled=(cfg.use_amp and "cuda" in cfg.device)):
                pred, up = sample_sr(model, lr, cfg, zero_noise=cfg.zero_noise_train, return_up=True)
                pred = pred.clamp(0.0, 1.0)
                loss_pix = cfg.pixel_l1_w * F.l1_loss(pred, hr)
                loss_perc = cfg.perceptual_w * perceptual(pred, hr)
                pred_lr = degrade_to_lr(pred, scale=cfg.scale)
                loss_lr_cons = cfg.lr_consistency_w * F.l1_loss(pred_lr, lr)
                total_loss = loss_pix + loss_perc + loss_lr_cons
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            history["train_total"].append(float(total_loss.item()))
            history["train_pix"].append(float(loss_pix.item()))
            history["train_perc"].append(float(loss_perc.item()))
            history["train_lr_cons"].append(float(loss_lr_cons.item()))
            running_total.append(float(total_loss.item()))
            running_pix.append(float(loss_pix.item()))
            running_perc.append(float(loss_perc.item()))
            running_lr_cons.append(float(loss_lr_cons.item()))

            if global_step % cfg.log_every == 0:
                print(f"[train] ep {epoch} step {global_step} | total={total_loss.item():.5f} | pix={loss_pix.item():.5f} | perc={loss_perc.item():.5f} | lr_cons={loss_lr_cons.item():.5f}")
            global_step += 1

        scheduler.step()
        if (epoch % cfg.eval_every_epochs == 0) or (epoch == cfg.epochs):
            gc.collect()
            if "cuda" in cfg.device:
                torch.cuda.empty_cache()
            stats = evaluate(ema.shadow, val_loader, cfg, lpips_model=lpips_eval, max_batches=cfg.val_batches, zero_noise=cfg.zero_noise_eval)
            history["val_psnr"].append(stats["psnr_model"])
            history["val_psnr_bicubic"].append(stats["psnr_bicubic"])
            history["val_lpips"].append(stats["lpips_model"])
            history["val_lpips_bicubic"].append(stats["lpips_bicubic"])
            print(f"[val {epoch}] psnr={stats['psnr_model']:.3f} | bicubic_psnr={stats['psnr_bicubic']:.3f} | lpips={stats['lpips_model']:.4f} | bicubic_lpips={stats['lpips_bicubic']:.4f} | mean_total={np.mean(running_total):.5f} | mean_pix={np.mean(running_pix):.5f} | mean_perc={np.mean(running_perc):.5f} | mean_lr_cons={np.mean(running_lr_cons):.5f}")
            save_checkpoint(model, ema, optimizer, scheduler, scaler, epoch, global_step, best_psnr, best_lpips, history, cfg, "last")
            if stats["psnr_model"] > best_psnr:
                best_psnr = stats["psnr_model"]
                save_checkpoint(model, ema, optimizer, scheduler, scaler, epoch, global_step, best_psnr, best_lpips, history, cfg, "best_psnr")
            if stats["lpips_model"] < best_lpips:
                best_lpips = stats["lpips_model"]
                save_checkpoint(model, ema, optimizer, scheduler, scaler, epoch, global_step, best_psnr, best_lpips, history, cfg, "best_lpips")

    final_stats = evaluate(ema.shadow, val_loader, cfg, lpips_model=lpips_eval, max_batches=cfg.val_batches, zero_noise=cfg.zero_noise_eval)
    path = save_history_json(history, cfg, final_stats=final_stats, name="baseline_history")
    print("Final stats:", final_stats)
    print("Best PSNR:", best_psnr)
    print("Best LPIPS:", best_lpips)
    print("History saved to:", path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def load_cfg(path: str | None, stochastic: bool) -> BaselineConfig:
    cfg = BaselineConfig()
    if stochastic:
        cfg.zero_noise_train = False
        cfg.zero_noise_eval = False
        cfg.output_root = "./outputs/perc_stochastic"
    else:
        cfg.zero_noise_train = True
        cfg.zero_noise_eval = True
        cfg.output_root = "./outputs/perc_deterministic"
    if path is None:
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key, value in data.items():
        if key == "data":
            for dk, dv in value.items():
                setattr(cfg.data, dk, dv)
        elif key == "generator":
            for gk, gv in value.items():
                setattr(cfg.generator, gk, gv)
        else:
            setattr(cfg, key, value)
    return cfg


if __name__ == "__main__":
    args = parse_args()
    train(load_cfg(args.config, args.stochastic))
