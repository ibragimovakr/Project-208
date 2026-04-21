from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DIV2KPairDataset(Dataset):
    def __init__(self, hr_dir: str, lr_dir: str, scale: int = 4, patch_size: Optional[int] = None, training: bool = True):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.scale = scale
        self.patch_size = patch_size
        self.training = training
        self.to_tensor = transforms.ToTensor()
        self.hr_files = sorted(self.hr_dir.glob("*.png"))
        if len(self.hr_files) == 0:
            raise RuntimeError(f"No HR png files found in: {self.hr_dir}")
        if self.training and (self.patch_size is None or self.patch_size % self.scale != 0):
            raise ValueError("patch_size must be set and divisible by scale for training")

    def __len__(self):
        return len(self.hr_files)

    def _lr_path_from_hr(self, hr_path: Path) -> Path:
        return self.lr_dir / f"{hr_path.stem}x{self.scale}.png"

    def _random_crop_pair(self, hr: Image.Image, lr: Image.Image) -> Tuple[Image.Image, Image.Image]:
        hr_ps = self.patch_size
        lr_ps = hr_ps // self.scale
        lr_w, lr_h = lr.size
        x_lr = random.randint(0, lr_w - lr_ps)
        y_lr = random.randint(0, lr_h - lr_ps)
        x_hr = x_lr * self.scale
        y_hr = y_lr * self.scale
        return (
            hr.crop((x_hr, y_hr, x_hr + hr_ps, y_hr + hr_ps)),
            lr.crop((x_lr, y_lr, x_lr + lr_ps, y_lr + lr_ps)),
        )

    def _center_crop_pair(self, hr: Image.Image, lr: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if self.patch_size is None:
            return hr, lr
        hr_ps = self.patch_size
        lr_ps = hr_ps // self.scale
        lr_w, lr_h = lr.size
        hr_w, hr_h = hr.size
        lr_ps = min(lr_ps, lr_w, hr_w // self.scale, lr_h, hr_h // self.scale)
        hr_ps = lr_ps * self.scale
        x_lr = max(0, (lr_w - lr_ps) // 2)
        y_lr = max(0, (lr_h - lr_ps) // 2)
        x_hr = x_lr * self.scale
        y_hr = y_lr * self.scale
        return (
            hr.crop((x_hr, y_hr, x_hr + hr_ps, y_hr + hr_ps)),
            lr.crop((x_lr, y_lr, x_lr + lr_ps, y_lr + lr_ps)),
        )

    def _augment_pair(self, hr: Image.Image, lr: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
            lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
            lr = lr.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            hr = hr.transpose(Image.ROTATE_90)
            lr = lr.transpose(Image.ROTATE_90)
        return hr, lr

    def _modcrop_pair(self, hr: Image.Image, lr: Image.Image) -> Tuple[Image.Image, Image.Image]:
        lr_w, lr_h = lr.size
        hr_w, hr_h = hr.size
        lr_w = min(lr_w, hr_w // self.scale)
        lr_h = min(lr_h, hr_h // self.scale)
        hr_w = lr_w * self.scale
        hr_h = lr_h * self.scale
        return hr.crop((0, 0, hr_w, hr_h)), lr.crop((0, 0, lr_w, lr_h))

    def __getitem__(self, idx: int):
        hr_path = self.hr_files[idx]
        lr_path = self._lr_path_from_hr(hr_path)
        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")
        if self.training:
            hr, lr = self._random_crop_pair(hr, lr)
            hr, lr = self._augment_pair(hr, lr)
        else:
            hr, lr = self._modcrop_pair(hr, lr)
        return self.to_tensor(hr), self.to_tensor(lr)
