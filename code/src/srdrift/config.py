from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json

import torch


@dataclass
class DataPaths:
    root: str = "./data"
    train_hr_dir: str = "./data/DIV2K_train_HR"
    train_lr_dir: str = "./data/DIV2K_train_LR_bicubic_X4/X4"
    val_hr_dir: str = "./data/DIV2K_valid_HR"
    val_lr_dir: str = "./data/DIV2K_valid_LR_bicubic_X4/X4"


@dataclass
class CommonConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    use_amp: bool = True
    scale: int = 4
    batch_size: int = 4
    val_batch_size: int = 1
    num_workers: int = 2
    epochs: int = 40
    lr: float = 2e-4
    min_lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    data: DataPaths = field(default_factory=DataPaths)
    output_root: str = "./outputs/default"

    def ensure_output_dirs(self) -> Dict[str, str]:
        base = Path(self.output_root)
        dirs = {
            "base": str(base),
            "checkpoints": str(base / "checkpoints"),
            "histories": str(base / "histories"),
            "plots": str(base / "plots"),
            "visuals": str(base / "visuals"),
        }
        for value in dirs.values():
            Path(value).mkdir(parents=True, exist_ok=True)
        return dirs

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratorConfig:
    in_channels: int = 3
    noise_image_channels: int = 3
    noise_embed_dim: int = 256
    unet_base: int = 64
    unet_channel_mult: Tuple[int, ...] = (1, 2, 4)
    unet_num_blocks: int = 2
    residual_out_scale: float = 1.0


@dataclass
class DriftConfig(CommonConfig):
    patch_size: int = 128
    save_every_n_epochs: int = 2
    save_every_start_epoch: int = 20
    num_samples_per_lr: int = 6
    num_positive_views: int = 6
    temperature: float = 0.10
    drift_step_size: float = 0.03
    drift_loss_weight: float = 0.10
    drift_loss_weight_max: float = 0.25
    warmup_epochs: int = 10
    drift_ramp_epochs: int = 8
    lambda_pos: float = 1.0
    lambda_same_neg_start: float = 0.40
    lambda_same_neg_end: float = 0.60
    feature_max_positions: int = 48
    feature_scale_index: int = 2
    norm_eps: float = 1e-6
    feat_scale_min: float = 1e-4
    drift_fp32: bool = True
    drift_scale_center: float = 1.0
    drift_scale_max: float = 2.0
    pixel_l1_w: float = 0.1
    lr_consistency_w: float = 1.0
    positive_sharpen_amount: Tuple[float, float] = (0.03, 0.07)
    positive_sharpen_sigma: Tuple[float, float] = (0.7, 1.2)
    positive_highpass_amount: Tuple[float, float] = (0.02, 0.05)
    positive_combo_highpass_amount: Tuple[float, float] = (0.015, 0.035)
    positive_lr_tolerance: float = 0.03
    log_every: int = 50
    eval_every_epochs: int = 1
    val_batches: int = 20
    eval_use_zero_noise: bool = True
    eval_on_y_channel: bool = True
    shave_border: int = 4
    eval_lpips_net: str = "vgg"
    eval_lpips_shave_border: int = 4
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    output_root: str = "./outputs/drift_sr"
    encoder_ckpt_path: Optional[str] = None


@dataclass
class BaselineConfig(CommonConfig):
    patch_size: int = 96
    val_hr_crop_size: Optional[int] = 256
    epochs: int = 50
    min_lr: float = 5e-5
    num_samples_per_lr: int = 1
    pixel_l1_w: float = 1.0
    perceptual_w: float = 0.05
    lr_consistency_w: float = 1.0
    perceptual_train_net: str = "vgg"
    perceptual_train_size: int = 64
    log_every: int = 100
    eval_every_epochs: int = 3
    val_batches: int = 10
    eval_on_y_channel: bool = True
    shave_border: int = 4
    eval_lpips_net: str = "vgg"
    eval_lpips_shave_border: int = 4
    zero_noise_train: bool = True
    zero_noise_eval: bool = True
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    output_root: str = "./outputs/perc_baseline"


@dataclass
class EncoderPretrainConfig(CommonConfig):
    patch_size: int = 128
    encoder_train_batch_size: int = 4
    encoder_pretrain_lr: float = 2e-4
    encoder_pretrain_weight_decay: float = 1e-4
    encoder_epochs: int = 20
    encoder_log_every: int = 50
    encoder_save_every_n_epochs: int = 2
    encoder_save_every_start_epoch: int = 10
    encoder_base_channels: int = 64
    encoder_use_spectral_norm: bool = False
    baseline_ckpt_path: str = "./outputs/perc_stochastic/checkpoints/best_lpips.pt"
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    output_root: str = "./outputs/residual_encoder_pretrain"


@dataclass
class CropVizConfig:
    hr_dir: str = "./data/DIV2K_valid_HR"
    lr_dir: str = "./data/DIV2K_valid_LR_bicubic_X4/X4"
    output_root: str = "./outputs/article_style_sr_crops"
    scale: int = 4
    seed: int = 42
    default_crop_size: int = 126
    display_upscale: int = 4
    zero_noise: bool = True


def save_config_json(cfg: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
