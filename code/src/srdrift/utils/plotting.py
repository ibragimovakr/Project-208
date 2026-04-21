from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _epoch_mean_from_steps(values: List[float], steps_per_epoch: int, n_eval_points: int) -> List[float]:
    if steps_per_epoch <= 0:
        return []
    out = []
    for start in range(0, len(values), steps_per_epoch):
        chunk = values[start : start + steps_per_epoch]
        if len(chunk) > 0:
            out.append(float(np.mean(chunk)))
    return out[:n_eval_points]


def plot_drifting_curves(history: Dict, cfg, steps_per_epoch: int):
    eval_epochs = history["eval_epochs"]
    n_eval = len(eval_epochs)
    train_total_epoch = _epoch_mean_from_steps(history["train_total"], steps_per_epoch, n_eval)
    train_pix_epoch = _epoch_mean_from_steps(history["train_pix"], steps_per_epoch, n_eval)
    train_lr_epoch = _epoch_mean_from_steps(history["train_lr_cons"], steps_per_epoch, n_eval)
    train_drift_epoch = _epoch_mean_from_steps(history["train_drift"], steps_per_epoch, n_eval)
    pos_accept_epoch = _epoch_mean_from_steps(history["train_pos_accept_rate"], steps_per_epoch, n_eval)
    saved_paths = []
    base = os.path.join(cfg.output_root, "plots")
    os.makedirs(base, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(eval_epochs, history["val_psnr"], marker="o", label="Val PSNR")
    plt.plot(eval_epochs, history["val_psnr_bicubic"], marker="o", label="Bicubic PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.title("Validation PSNR vs Epoch")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(base, "val_psnr_vs_epoch.png")
    plt.savefig(path, dpi=180)
    plt.close()
    saved_paths.append(path)

    plt.figure(figsize=(8, 5))
    plt.plot(eval_epochs, history["val_lpips"], marker="o", label="Val LPIPS")
    plt.plot(eval_epochs, history["val_lpips_bicubic"], marker="o", label="Bicubic LPIPS")
    plt.xlabel("Epoch")
    plt.ylabel("LPIPS")
    plt.title("Validation LPIPS vs Epoch")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(base, "val_lpips_vs_epoch.png")
    plt.savefig(path, dpi=180)
    plt.close()
    saved_paths.append(path)

    plt.figure(figsize=(8, 5))
    plt.plot(eval_epochs, train_total_epoch, marker="o", label="Train total")
    plt.plot(eval_epochs, train_pix_epoch, marker="o", label="Train pixel")
    plt.plot(eval_epochs, train_lr_epoch, marker="o", label="Train LR consistency")
    plt.plot(eval_epochs, train_drift_epoch, marker="o", label="Train drift raw")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses vs Epoch")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(base, "train_losses_vs_epoch.png")
    plt.savefig(path, dpi=180)
    plt.close()
    saved_paths.append(path)

    plt.figure(figsize=(8, 5))
    plt.plot(eval_epochs, pos_accept_epoch, marker="o", label="Positive accept rate")
    plt.xlabel("Epoch")
    plt.ylabel("Accept rate")
    plt.title("Positive acceptance rate vs Epoch")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(base, "positive_accept_rate_vs_epoch.png")
    plt.savefig(path, dpi=180)
    plt.close()
    saved_paths.append(path)
    return saved_paths
