from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import lpips
import pandas as pd
import torch
from PIL import Image

from srdrift.config import CropVizConfig, DriftConfig
from srdrift.metrics import calc_lpips_sr, calc_psnr_sr
from srdrift.models.generator import NoiseConditionalResidualUNetSR
from srdrift.utils.common import sample_sr, set_seed
from srdrift.utils.crops import (
    crop_corresponding_lr_tensor,
    crop_hr_tensor,
    make_article_style_panel,
    make_marked_full_image_np,
    np01_to_pil,
    pil_to_tensor,
    resolve_crop_xy,
    save_tensor_png,
    tensor_to_np01,
)


def load_pair_by_name(image_name: str, hr_dir: str, lr_dir: str, scale: int):
    hr_path = os.path.join(hr_dir, image_name)
    stem = Path(image_name).stem
    lr_name = f"{stem}x{scale}.png"
    lr_path = os.path.join(lr_dir, lr_name)
    if not os.path.exists(hr_path):
        raise FileNotFoundError(f"HR image not found: {hr_path}")
    if not os.path.exists(lr_path):
        raise FileNotFoundError(f"LR image not found: {lr_path}")
    hr = pil_to_tensor(Image.open(hr_path).convert("RGB")).unsqueeze(0)
    lr = pil_to_tensor(Image.open(lr_path).convert("RGB")).unsqueeze(0)
    return hr, lr


def build_model_from_checkpoint(ckpt_path: str, cfg: DriftConfig):
    g = cfg.generator
    checkpoint = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
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
    if "ema_state" in checkpoint:
        model.load_state_dict(checkpoint["ema_state"])
    else:
        model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main(cfg_path: str, spec_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    drift_cfg = DriftConfig()
    for key, value in cfg_dict.items():
        if key == "data":
            for dk, dv in value.items():
                setattr(drift_cfg.data, dk, dv)
        elif key == "generator":
            for gk, gv in value.items():
                setattr(drift_cfg.generator, gk, gv)
        else:
            setattr(drift_cfg, key, value)

    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    crop_cfg = CropVizConfig(**spec["viz"])
    set_seed(crop_cfg.seed)
    os.makedirs(crop_cfg.output_root, exist_ok=True)
    checkpoints = spec["checkpoints"]
    image_names = spec["image_names"]
    crops_by_image = spec["crops_by_image"]
    model_order = spec["model_order"]
    display_names = spec["display_names"]

    models_loaded = {name: build_model_from_checkpoint(path, drift_cfg) for name, path in checkpoints.items()}
    lpips_model = lpips.LPIPS(net=drift_cfg.eval_lpips_net).to(drift_cfg.device).eval()
    all_rows, saved_panels = [], []

    for image_name in image_names:
        hr, lr = load_pair_by_name(image_name, crop_cfg.hr_dir, crop_cfg.lr_dir, crop_cfg.scale)
        hr = hr.to(drift_cfg.device)
        lr = lr.to(drift_cfg.device)
        outputs = {}
        first_model_key = list(models_loaded.keys())[0]
        sr_first, bicubic = sample_sr(models_loaded[first_model_key], lr, drift_cfg, zero_noise=crop_cfg.zero_noise, return_up=True)
        outputs["Bicubic"] = bicubic.clamp(0.0, 1.0)
        outputs["HR"] = hr
        outputs[first_model_key] = sr_first.clamp(0.0, 1.0)
        for ckpt_name, model in models_loaded.items():
            if ckpt_name == first_model_key:
                continue
            sr = sample_sr(model, lr, drift_cfg, zero_noise=crop_cfg.zero_noise, return_up=False)
            outputs[ckpt_name] = sr.clamp(0.0, 1.0)

        H, W = hr.shape[-2:]
        crop_specs_resolved = []
        for spec_crop in crops_by_image[image_name]:
            x, y, size = resolve_crop_xy(spec_crop, H, W, crop_cfg.default_crop_size)
            crop_specs_resolved.append({"color": spec_crop["color"], "x": x, "y": y, "size": size})

        hr_np = tensor_to_np01(hr)
        full_marked_np = make_marked_full_image_np(hr_np, crop_specs_resolved)
        image_out_dir = os.path.join(crop_cfg.output_root, Path(image_name).stem)
        os.makedirs(image_out_dir, exist_ok=True)
        full_marked_path = os.path.join(image_out_dir, f"{Path(image_name).stem}_full_marked.png")
        np01_to_pil(full_marked_np).save(full_marked_path)

        crop_payloads = []
        for crop_idx, crop_spec in enumerate(crop_specs_resolved):
            x0, y0, crop_size = crop_spec["x"], crop_spec["y"], crop_spec["size"]
            crop_dir = os.path.join(image_out_dir, f"crop_{crop_idx + 1}")
            os.makedirs(crop_dir, exist_ok=True)
            hr_crop = crop_hr_tensor(hr, x0, y0, crop_size)
            lr_crop = crop_corresponding_lr_tensor(lr, x0, y0, crop_size, crop_cfg.scale)
            views = {}
            for model_name in model_order:
                src = outputs[model_name]
                crop = crop_hr_tensor(src, x0, y0, crop_size)
                crop_path = os.path.join(crop_dir, f"{model_name}.png")
                save_tensor_png(crop, crop_path)
                if model_name == "HR":
                    metrics = {"psnr": float("nan"), "lpips": float("nan"), "lr_cons": float("nan")}
                else:
                    psnr_val = calc_psnr_sr(crop, hr_crop, shave=0, use_y=True)
                    lpips_val = calc_lpips_sr(crop, hr_crop, lpips_model=lpips_model, shave=0)
                    pred_down = torch.nn.functional.interpolate(crop, size=lr_crop.shape[-2:], mode="bicubic", align_corners=False)
                    lr_cons_val = float(torch.nn.functional.l1_loss(pred_down, lr_crop).item())
                    metrics = {"psnr": psnr_val, "lpips": lpips_val, "lr_cons": lr_cons_val}
                    all_rows.append({"image_name": image_name, "crop_idx": crop_idx + 1, "x": x0, "y": y0, "crop_size": crop_size, "model": model_name, **metrics, "crop_path": crop_path, "full_marked_path": full_marked_path})
                views[model_name] = {"crop_tensor": crop, "crop_np": tensor_to_np01(crop), "metrics": metrics, "crop_path": crop_path}
            crop_payloads.append({"color": crop_spec["color"], "x": x0, "y": y0, "size": crop_size, "views": views})

        panel_path = os.path.join(image_out_dir, f"{Path(image_name).stem}_article_style_panel.png")
        make_article_style_panel(image_name=image_name, full_marked_np=full_marked_np, crop_payloads=crop_payloads, out_path=panel_path, model_order=model_order, display_names=display_names, display_upscale=crop_cfg.display_upscale)
        saved_panels.append(panel_path)
        print(f"[saved] panel: {panel_path}")

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(crop_cfg.output_root, "crop_metrics_summary.csv")
    json_path = os.path.join(crop_cfg.output_root, "crop_metrics_summary.json")
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    print("Saved root:", crop_cfg.output_root)
    print("Metrics CSV:", csv_path)
    print("Metrics JSON:", json_path)
    for p in saved_panels:
        print(" -", p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Drifting config json.")
    parser.add_argument("--spec", required=True, help="Crop visualization spec json.")
    args = parser.parse_args()
    main(args.config, args.spec)
