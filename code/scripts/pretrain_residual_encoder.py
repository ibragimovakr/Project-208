from __future__ import annotations

import argparse
import gc
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from srdrift.config import EncoderPretrainConfig
from srdrift.data import DIV2KPairDataset
from srdrift.models.generator import NoiseConditionalResidualUNetSR
from srdrift.models.residual_encoder import ResidualDiscClassifier, ResidualDiscriminatorEncoder
from srdrift.utils.common import atomic_torch_save, get_rng_state, sample_sr, save_history_json, set_seed


def load_baseline_generator_from_ckpt(cfg: EncoderPretrainConfig):
    g = cfg.generator
    model = NoiseConditionalResidualUNetSR(
        in_channels=g.in_channels,
        noise_image_channels=g.noise_image_channels,
        base=g.unet_base,
        channel_mult=g.unet_channel_mult,
        num_blocks=g.unet_num_blocks,
        noise_embed_dim=g.noise_embed_dim,
        residual_out_scale=g.residual_out_scale,
        scale=cfg.scale,
    ).to(cfg.device)
    ckpt = torch.load(cfg.baseline_ckpt_path, map_location=cfg.device, weights_only=False)
    if "ema_state" in ckpt:
        state = ckpt["ema_state"]
    elif "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        raise KeyError("Checkpoint must contain 'ema_state' or 'model_state'.")
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def save_encoder_checkpoint(disc_model, optimizer, epoch, global_step, best_loss, history, cfg, tag: str):
    path = os.path.join(cfg.output_root, "checkpoints", f"{tag}.pt")
    atomic_torch_save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": best_loss,
            "encoder_state": disc_model.encoder.state_dict(),
            "disc_state": disc_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "config": cfg.to_dict(),
            "rng_state": get_rng_state(),
        },
        path,
    )
    print(f"[encoder ckpt] saved: {path}")


def train(cfg: EncoderPretrainConfig):
    set_seed(cfg.seed)
    cfg.ensure_output_dirs()
    train_ds = DIV2KPairDataset(cfg.data.train_hr_dir, cfg.data.train_lr_dir, scale=cfg.scale, patch_size=cfg.patch_size, training=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.encoder_train_batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers > 0), prefetch_factor=2 if cfg.num_workers > 0 else None)
    baseline_model = load_baseline_generator_from_ckpt(cfg)
    encoder = ResidualDiscriminatorEncoder(in_ch=3, base_ch=cfg.encoder_base_channels, use_sn=cfg.encoder_use_spectral_norm).to(cfg.device)
    disc_model = ResidualDiscClassifier(encoder).to(cfg.device)
    optimizer = torch.optim.AdamW(disc_model.parameters(), lr=cfg.encoder_pretrain_lr, weight_decay=cfg.encoder_pretrain_weight_decay)
    bce = nn.BCEWithLogitsLoss()
    amp_device_type = "cuda" if "cuda" in cfg.device else "cpu"
    scaler = GradScaler(device=amp_device_type, enabled=(cfg.use_amp and "cuda" in cfg.device))

    history = {"step_total": [], "step_acc_real": [], "step_acc_fake": [], "epoch_mean_total": [], "epoch_mean_acc_real": [], "epoch_mean_acc_fake": []}
    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, cfg.encoder_epochs + 1):
        disc_model.train()
        running_loss, running_acc_real, running_acc_fake = [], [], []
        for hr, lr in train_loader:
            hr = hr.to(cfg.device, non_blocking=True)
            lr = lr.to(cfg.device, non_blocking=True)
            with torch.no_grad():
                fake, _ = sample_sr(baseline_model, lr, cfg, zero_noise=False, return_up=True)
                fake = fake.clamp(0.0, 1.0)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=amp_device_type, enabled=(cfg.use_amp and "cuda" in cfg.device)):
                logits_real, _, _ = disc_model(hr)
                logits_fake, _, _ = disc_model(fake)
                target_real = torch.ones_like(logits_real)
                target_fake = torch.zeros_like(logits_fake)
                loss_real = bce(logits_real, target_real)
                loss_fake = bce(logits_fake, target_fake)
                total_loss = 0.5 * (loss_real + loss_fake)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc_real = ((logits_real > 0).float() == target_real).float().mean().item()
            acc_fake = ((logits_fake <= 0).float() == target_fake).float().mean().item()
            history["step_total"].append(float(total_loss.item()))
            history["step_acc_real"].append(float(acc_real))
            history["step_acc_fake"].append(float(acc_fake))
            running_loss.append(float(total_loss.item()))
            running_acc_real.append(float(acc_real))
            running_acc_fake.append(float(acc_fake))

            if global_step % cfg.encoder_log_every == 0:
                print(f"[encoder] ep {epoch} step {global_step} | total={total_loss.item():.5f} | acc_real={acc_real:.3f} | acc_fake={acc_fake:.3f}")
            global_step += 1

        epoch_mean_total = float(np.mean(running_loss)) if running_loss else float("nan")
        epoch_mean_acc_real = float(np.mean(running_acc_real)) if running_acc_real else float("nan")
        epoch_mean_acc_fake = float(np.mean(running_acc_fake)) if running_acc_fake else float("nan")
        history["epoch_mean_total"].append(epoch_mean_total)
        history["epoch_mean_acc_real"].append(epoch_mean_acc_real)
        history["epoch_mean_acc_fake"].append(epoch_mean_acc_fake)
        print(f"[encoder val {epoch}] mean_total={epoch_mean_total:.5f} | mean_acc_real={epoch_mean_acc_real:.3f} | mean_acc_fake={epoch_mean_acc_fake:.3f}")
        save_encoder_checkpoint(disc_model, optimizer, epoch, global_step, best_loss, history, cfg, "last")
        if np.isfinite(epoch_mean_total) and epoch_mean_total < best_loss:
            best_loss = epoch_mean_total
            save_encoder_checkpoint(disc_model, optimizer, epoch, global_step, best_loss, history, cfg, "best")
        if epoch >= cfg.encoder_save_every_start_epoch and (epoch - cfg.encoder_save_every_start_epoch) % cfg.encoder_save_every_n_epochs == 0:
            save_encoder_checkpoint(disc_model, optimizer, epoch, global_step, best_loss, history, cfg, f"epoch_{epoch:03d}")
        gc.collect()
        if "cuda" in cfg.device:
            torch.cuda.empty_cache()

    final_stats = {
        "best_loss": best_loss,
        "last_epoch_loss": history["epoch_mean_total"][-1] if history["epoch_mean_total"] else float("nan"),
        "last_epoch_acc_real": history["epoch_mean_acc_real"][-1] if history["epoch_mean_acc_real"] else float("nan"),
        "last_epoch_acc_fake": history["epoch_mean_acc_fake"][-1] if history["epoch_mean_acc_fake"] else float("nan"),
    }
    path = save_history_json(history, cfg, final_stats=final_stats, name="encoder_pretrain_history")
    print("Final encoder stats:", final_stats)
    print("History saved to:", path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def load_cfg(path: str | None) -> EncoderPretrainConfig:
    cfg = EncoderPretrainConfig()
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
