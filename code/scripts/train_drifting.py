from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from srdrift.config import DriftConfig
from srdrift.data import DIV2KPairDataset
from srdrift.image_ops import build_positive_bank, degrade_to_lr
from srdrift.losses.drifting import EMA, SingleLevelConditionalDriftingLoss, get_lambda_drift, get_lambda_same_neg
from srdrift.models.feature_extractors import FrozenVGGFeatureMaps
from srdrift.models.generator import NoiseConditionalResidualUNetSR
from srdrift.utils.common import atomic_torch_save, evaluate, generate_multi_samples, get_rng_state, save_history_json, set_seed
from srdrift.utils.plotting import plot_drifting_curves


def save_checkpoint(model, ema, optimizer, scheduler, scaler, epoch, global_step, best_lpips, history, cfg, tag: str):
    path = os.path.join(cfg.output_root, "checkpoints", f"{tag}.pt")
    atomic_torch_save(
        {
            "epoch": epoch,
            "global_step": global_step,
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


def build_model(cfg: DriftConfig):
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


def train(cfg: DriftConfig):
    set_seed(cfg.seed)
    cfg.ensure_output_dirs()
    print("Device:", cfg.device)
    print("Train HR:", cfg.data.train_hr_dir)
    print("Train LR:", cfg.data.train_lr_dir)

    train_ds = DIV2KPairDataset(cfg.data.train_hr_dir, cfg.data.train_lr_dir, scale=cfg.scale, patch_size=cfg.patch_size, training=True)
    val_ds = DIV2KPairDataset(cfg.data.val_hr_dir, cfg.data.val_lr_dir, scale=cfg.scale, patch_size=None, training=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers > 0), prefetch_factor=2 if cfg.num_workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=cfg.val_batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers > 0), prefetch_factor=2 if cfg.num_workers > 0 else None)

    model = build_model(cfg).to(cfg.device)
    frozen_encoder = FrozenVGGFeatureMaps().to(cfg.device)
    drift_loss_fn = SingleLevelConditionalDriftingLoss(frozen_encoder, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
    amp_device_type = "cuda" if "cuda" in cfg.device else "cpu"
    scaler = GradScaler(device=amp_device_type, enabled=(cfg.use_amp and "cuda" in cfg.device))
    ema = EMA(model, decay=cfg.ema_decay)
    lpips_model = lpips.LPIPS(net=cfg.eval_lpips_net).to(cfg.device).eval()
    for p in lpips_model.parameters():
        p.requires_grad = False

    history = {
        "train_total": [],
        "train_pix": [],
        "train_lr_cons": [],
        "train_drift": [],
        "train_pos_accept_rate": [],
        "lambda_drift": [],
        "lambda_same_neg": [],
        "eval_epochs": [],
        "val_psnr": [],
        "val_psnr_bicubic": [],
        "val_lpips": [],
        "val_lpips_bicubic": [],
        "epoch_pos_accept_rate": [],
    }
    global_step = 0
    best_lpips = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        lambda_drift = get_lambda_drift(epoch, cfg)
        lambda_same_neg = get_lambda_same_neg(epoch, cfg)
        running_total, running_pix, running_lr, running_drift, running_pos_accept = [], [], [], [], []

        for hr, lr in train_loader:
            hr = hr.to(cfg.device, non_blocking=True)
            lr = lr.to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=amp_device_type, enabled=(cfg.use_amp and "cuda" in cfg.device)):
                x_gen, x_up = generate_multi_samples(model, lr, cfg, num_samples=cfg.num_samples_per_lr)
                positive_bank, positive_stats = build_positive_bank(hr, lr, cfg)
                anchor_idx = np.random.randint(x_gen.shape[1])
                x_anchor = x_gen[:, anchor_idx]
                loss_pix = cfg.pixel_l1_w * F.l1_loss(x_anchor, hr)
                B, N, C, H, W = x_gen.shape
                x_gen_lr = degrade_to_lr(x_gen.view(B * N, C, H, W), scale=cfg.scale)
                loss_lr = cfg.lr_consistency_w * F.l1_loss(x_gen_lr.view(B, N, C, H // cfg.scale, W // cfg.scale), lr[:, None].expand(-1, N, -1, -1, -1))
                if lambda_drift > 0.0:
                    drift_raw, drift_info = drift_loss_fn(x_gen=x_gen, x_pos=positive_bank, x_up=x_up, epoch=epoch)
                    loss_drift = lambda_drift * drift_raw
                else:
                    drift_raw = hr.new_tensor(0.0)
                    drift_info = {"scale": cfg.feature_scale_index, "A_pos_mean": float("nan"), "A_same_mean": float("nan"), "V_rms": float("nan"), "feat_scale": float("nan"), "drift_scale": float("nan"), "drift_norm_factor": float("nan")}
                    loss_drift = hr.new_tensor(0.0)
                total_loss = loss_pix + loss_lr + loss_drift

            if not torch.isfinite(total_loss):
                print(f"[warn] non-finite loss at epoch {epoch}, step {global_step}; skipped")
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if not torch.isfinite(grad_norm):
                print(f"[warn] non-finite grad norm at epoch {epoch}, step {global_step}; skipped")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            pos_accept_rate = float(positive_stats["accept_rate"])
            history["train_total"].append(float(total_loss.item()))
            history["train_pix"].append(float(loss_pix.item()))
            history["train_lr_cons"].append(float(loss_lr.item()))
            history["train_drift"].append(float(drift_raw.item()))
            history["train_pos_accept_rate"].append(pos_accept_rate)
            history["lambda_drift"].append(float(lambda_drift))
            history["lambda_same_neg"].append(float(lambda_same_neg))
            running_total.append(float(total_loss.item()))
            running_pix.append(float(loss_pix.item()))
            running_lr.append(float(loss_lr.item()))
            running_drift.append(float(drift_raw.item()))
            running_pos_accept.append(pos_accept_rate)

            if global_step % cfg.log_every == 0:
                print(
                    f"[train] ep {epoch} step {global_step}"
                    f" | total={total_loss.item():.5f}"
                    f" | pix={loss_pix.item():.5f}"
                    f" | lr={loss_lr.item():.5f}"
                    f" | drift_raw={drift_raw.item():.5f}"
                    f" | pos_ok={100.0 * pos_accept_rate:.1f}%"
                    f" | lambda_drift={lambda_drift:.3f}"
                    f" | lambda_same_neg={lambda_same_neg:.3f}"
                    f" | grad={float(grad_norm):.4f}"
                    f" | lr_now={optimizer.param_groups[0]['lr']:.3e}"
                    f" | s{drift_info['scale']}_Apos={drift_info['A_pos_mean']:.4f}"
                    f" | s{drift_info['scale']}_Asame={drift_info['A_same_mean']:.4f}"
                    f" | s{drift_info['scale']}_Vrms={drift_info['V_rms']:.4f}"
                )
            global_step += 1

        scheduler.step()
        epoch_pos_accept = float(np.mean(running_pos_accept)) if running_pos_accept else float("nan")
        history["epoch_pos_accept_rate"].append(epoch_pos_accept)

        if (epoch % cfg.eval_every_epochs == 0) or (epoch == cfg.epochs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            stats = evaluate(ema.shadow, val_loader, cfg, lpips_model=lpips_model, max_batches=cfg.val_batches, zero_noise=cfg.eval_use_zero_noise)
            history["eval_epochs"].append(epoch)
            history["val_psnr"].append(stats["psnr_model"])
            history["val_psnr_bicubic"].append(stats["psnr_bicubic"])
            history["val_lpips"].append(stats["lpips_model"])
            history["val_lpips_bicubic"].append(stats["lpips_bicubic"])
            print(
                f"[val {epoch}] psnr={stats['psnr_model']:.3f} | bicubic_psnr={stats['psnr_bicubic']:.3f}"
                f" | lpips={stats['lpips_model']:.4f} | bicubic_lpips={stats['lpips_bicubic']:.4f}"
                f" | mean_total={np.mean(running_total) if running_total else float('nan'):.5f}"
                f" | mean_pix={np.mean(running_pix) if running_pix else float('nan'):.5f}"
                f" | mean_lr={np.mean(running_lr) if running_lr else float('nan'):.5f}"
                f" | mean_drift={np.mean(running_drift) if running_drift else float('nan'):.5f}"
                f" | mean_pos_ok={100.0 * epoch_pos_accept:.1f}%"
            )
            if stats["lpips_model"] < best_lpips:
                best_lpips = stats["lpips_model"]
                save_checkpoint(model, ema, optimizer, scheduler, scaler, epoch, global_step, best_lpips, history, cfg, "best_lpips")
            save_checkpoint(model, ema, optimizer, scheduler, scaler, epoch, global_step, best_lpips, history, cfg, "last")
            if epoch >= cfg.save_every_start_epoch and (epoch - cfg.save_every_start_epoch) % cfg.save_every_n_epochs == 0:
                save_checkpoint(model, ema, optimizer, scheduler, scaler, epoch, global_step, best_lpips, history, cfg, f"epoch_{epoch:03d}")

    final_stats = evaluate(ema.shadow, val_loader, cfg, lpips_model=lpips_model, max_batches=cfg.val_batches, zero_noise=cfg.eval_use_zero_noise)
    history_path = save_history_json(history, cfg, final_stats=final_stats, name="drift_history")
    plot_paths = plot_drifting_curves(history, cfg, steps_per_epoch=len(train_loader))
    print("Final stats:", final_stats)
    print("Best LPIPS:", best_lpips)
    print("History saved to:", history_path)
    print("Plots:")
    for p in plot_paths:
        print(" -", p)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config override.")
    return parser.parse_args()


def load_cfg(path: str | None) -> DriftConfig:
    cfg = DriftConfig()
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
    train(load_cfg(args.config))
