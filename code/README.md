# Drifting for Single-Image Super-Resolution

PyTorch repository for experiments on **stochastic single-image super-resolution (SISR)** with a **drifting-based training objective**, perceptual-loss baselines, residual feature encoder pretraining, and article-style crop visualizations.


## Overview

This repository contains four main experiment groups:

- **Drifting SR** — the main stochastic super-resolution model trained with LR consistency, pixel loss, and a single-level conditional drifting loss.
- **Perceptual baselines** — deterministic and stochastic baselines trained without drifting.
- **Residual encoder pretraining** — optional pretraining of a task-specific residual feature encoder.
- **Crop visualization** — generation of article-style qualitative comparisons with marked crop boxes and per-crop metrics.

## Repository Structure

```text
sr_drifting_repo/
├── configs/
│   ├── drift_local.json
│   ├── perc_deterministic_local.json
│   ├── perc_stochastic_local.json
│   ├── encoder_pretrain_local.json
│   └── crops_example.json
├── scripts/
│   ├── train_drifting.py
│   ├── train_perc_baseline.py
│   ├── train_perc_deterministic.py
│   ├── train_perc_stochastic.py
│   ├── pretrain_residual_encoder.py
│   └── make_article_crops.py
├── src/
│   └── srdrift/
│       ├── config.py
│       ├── data.py
│       ├── image_ops.py
│       ├── metrics.py
│       ├── models/
│       │   ├── generator.py
│       │   ├── feature_extractors.py
│       │   └── residual_encoder.py
│       ├── losses/
│       │   ├── drifting.py
│       │   └── perceptual.py
│       └── utils/
│           ├── common.py
│           ├── plotting.py
│           └── crops.py
├── data/
├── outputs/
├── requirements.txt
└── README.md
```

## Dataset Layout

The code expects DIV2K to be placed locally with the following structure:

```text
sr_drifting_repo/
├── data/
│   ├── DIV2K_train_HR/
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   ├── DIV2K_train_LR_bicubic_X4/
│   │   └── X4/
│   │       ├── 0001x4.png
│   │       ├── 0002x4.png
│   │       └── ...
│   ├── DIV2K_valid_HR/
│   │   ├── 0801.png
│   │   ├── 0802.png
│   │   └── ...
│   └── DIV2K_valid_LR_bicubic_X4/
│       └── X4/
│           ├── 0801x4.png
│           ├── 0802x4.png
│           └── ...
```

LR images are resolved automatically from HR filenames using the rule:

```text
0802.png -> 0802x4.png
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=./src
```

## Configs

All experiment paths and hyperparameters are controlled through JSON config files in `configs/`.

Default local paths already assume a non-Kaggle setup:

- `./data/DIV2K_train_HR`
- `./data/DIV2K_train_LR_bicubic_X4/X4`
- `./data/DIV2K_valid_HR`
- `./data/DIV2K_valid_LR_bicubic_X4/X4`

Before running experiments, update the config files if your local directory layout differs.

## Training

### 1. Drifting SR

Main experiment with stochastic SR samples, LR consistency, pixel reconstruction, and drifting loss.

```bash
python scripts/train_drifting.py --config configs/drift_local.json
```

Outputs are written to the directory specified by `output_root`, typically:

```text
outputs/drift_sr/
├── checkpoints/
├── histories/
├── plots/
└── visuals/
```

### 2. Deterministic perceptual baseline

Perceptual-loss baseline with zero noise during training and evaluation.

```bash
python scripts/train_perc_deterministic.py --config configs/perc_deterministic_local.json
```

### 3. Stochastic perceptual baseline

Perceptual-loss baseline with stochastic noise input.

```bash
python scripts/train_perc_stochastic.py --config configs/perc_stochastic_local.json
```

### 4. Generic perceptual baseline entrypoint

```bash
python scripts/train_perc_baseline.py --config configs/perc_stochastic_local.json
```

## Residual Encoder Pretraining

Optional pretraining stage for a task-specific residual feature encoder.

Before running this step, make sure `baseline_ckpt_path` in `configs/encoder_pretrain_local.json` points to an existing baseline checkpoint.

```bash
python scripts/pretrain_residual_encoder.py --config configs/encoder_pretrain_local.json
```

## Article-Style Crop Visualization

This script builds qualitative comparison panels with:

- a full HR image with colored crop boxes,
- enlarged crop views for multiple models,
- crop-level PSNR / LPIPS values,
- saved per-crop assets and summary tables.

First, update checkpoint paths and crop definitions in `configs/crops_example.json`, then run:

```bash
python scripts/make_article_crops.py \
  --config configs/drift_local.json \
  --spec configs/crops_example.json
```

Outputs are typically saved under:

```text
outputs/article_style_sr_crops/
```

## Outputs

Each experiment writes artifacts into its own `output_root`. Depending on the script, outputs may include:

- `checkpoints/` — saved model checkpoints (`best_lpips`, `last`, periodic checkpoints)
- `histories/` — training logs and serialized history
- `plots/` — PSNR / LPIPS / loss curves
- `visuals/` — generated figures and qualitative outputs
- crop summary tables in CSV / JSON format

## Main Components

### `src/srdrift/data.py`
DIV2K paired dataset loader with crop extraction, augmentation, and HR/LR filename matching.

### `src/srdrift/image_ops.py`
Bicubic upsampling/downsampling, blur, sharpen, high-pass helpers, positive-view generation, and LR-consistency utilities.

### `src/srdrift/models/generator.py`
Noise-conditioned residual U-Net generator for stochastic SR prediction.

### `src/srdrift/models/feature_extractors.py`
Frozen VGG feature extractor used by drifting and perceptual objectives.

### `src/srdrift/models/residual_encoder.py`
Residual encoder and discriminator-style components for encoder pretraining experiments.

### `src/srdrift/losses/drifting.py`
Single-level conditional drifting loss, drift scheduling, and EMA utilities.

### `src/srdrift/losses/perceptual.py`
Perceptual-loss baseline objective.

### `src/srdrift/utils/crops.py`
Utilities for marked full-image rendering and article-style crop panels.
