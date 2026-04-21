from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import patches
from PIL import Image


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(arr).permute(2, 0, 1)


def tensor_to_np01(x: torch.Tensor) -> np.ndarray:
    x = x.detach().clamp(0.0, 1.0).cpu()
    if x.ndim == 4:
        x = x[0]
    return x.permute(1, 2, 0).numpy()


def np01_to_pil(x: np.ndarray) -> Image.Image:
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(x)


def upscale_np_for_display(x: np.ndarray, scale: int = 4) -> np.ndarray:
    pil = np01_to_pil(x)
    pil = pil.resize((pil.width * scale, pil.height * scale), Image.NEAREST)
    return np.asarray(pil).astype(np.float32) / 255.0


def resolve_crop_xy(spec: dict, H: int, W: int, default_size: int):
    size = int(spec.get("size", default_size))
    size = min(size, H, W)
    if "xy_abs" in spec:
        x, y = spec["xy_abs"]
    elif "xy_rel" in spec:
        rx, ry = spec["xy_rel"]
        x = int(round(rx * max(W - size, 0)))
        y = int(round(ry * max(H - size, 0)))
    else:
        raise ValueError("Crop spec must contain 'xy_abs' or 'xy_rel'.")
    x = max(0, min(int(x), W - size))
    y = max(0, min(int(y), H - size))
    return x, y, size


def crop_hr_tensor(x: torch.Tensor, x0: int, y0: int, size: int) -> torch.Tensor:
    return x[..., y0 : y0 + size, x0 : x0 + size]


def crop_corresponding_lr_tensor(lr: torch.Tensor, x_hr: int, y_hr: int, size_hr: int, scale: int) -> torch.Tensor:
    x0_lr = int(round(x_hr / scale))
    y0_lr = int(round(y_hr / scale))
    x1_lr = int(round((x_hr + size_hr) / scale))
    y1_lr = int(round((y_hr + size_hr) / scale))
    x0_lr = max(0, min(x0_lr, lr.shape[-1] - 1))
    y0_lr = max(0, min(y0_lr, lr.shape[-2] - 1))
    x1_lr = max(x0_lr + 1, min(x1_lr, lr.shape[-1]))
    y1_lr = max(y0_lr + 1, min(y1_lr, lr.shape[-2]))
    return lr[..., y0_lr:y1_lr, x0_lr:x1_lr]


def save_tensor_png(x: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np01_to_pil(tensor_to_np01(x)).save(path)


def make_marked_full_image_np(full_img_np: np.ndarray, crop_specs_resolved: list) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(full_img_np)
    ax.axis("off")
    for spec in crop_specs_resolved:
        rect = patches.Rectangle((spec["x"], spec["y"]), spec["size"], spec["size"], linewidth=2.8, edgecolor=spec["color"], facecolor="none")
        ax.add_patch(rect)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[..., :3]
    plt.close(fig)
    return img.astype(np.float32) / 255.0


def make_article_style_panel(image_name: str, full_marked_np: np.ndarray, crop_payloads: list, out_path: str, model_order: List[str], display_names: Dict[str, str], display_upscale: int = 4):
    n_crops = len(crop_payloads)
    n_cols = 1 + len(model_order)
    fig = plt.figure(figsize=(5.8 + 3.4 * len(model_order), 3.8 * n_crops))
    gs = fig.add_gridspec(nrows=n_crops, ncols=n_cols, width_ratios=[2.5] + [1.15] * len(model_order), wspace=0.03, hspace=0.02)
    ax_full = fig.add_subplot(gs[:, 0])
    ax_full.imshow(full_marked_np)
    ax_full.set_title(image_name, fontsize=16, pad=8)
    ax_full.axis("off")
    for r, crop_info in enumerate(crop_payloads):
        crop_color = crop_info["color"]
        for c, model_name in enumerate(model_order, start=1):
            ax = fig.add_subplot(gs[r, c])
            crop_np = crop_info["views"][model_name]["crop_np"]
            ax.imshow(upscale_np_for_display(crop_np, scale=display_upscale))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.5)
                spine.set_edgecolor(crop_color)
            if r == 0:
                ax.set_title(display_names.get(model_name, model_name), fontsize=12, pad=8)
            m = crop_info["views"][model_name]["metrics"]
            ax.set_xlabel("" if model_name == "HR" else f"{m['psnr']:.2f} / {m['lpips']:.3f}", fontsize=10, labelpad=4)
    plt.subplots_adjust(left=0.02, right=0.995, top=0.93, bottom=0.04)
    plt.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
