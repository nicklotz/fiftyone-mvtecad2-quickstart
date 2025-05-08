from __future__ import annotations
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule

from .util import RESIZE_CROP, Tag
from .config import Paths


class MVTecAd2Sample(Dataset):
    """Single sample wrapper for MVTec AD 2 images (optionally with masks)."""

    def __init__(self, image_paths: List[Path], masks_present: bool) -> None:
        self.image_paths = image_paths
        self.masks_present = masks_present

    def __len__(self) -> int:  # noqa: D401
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = RESIZE_CROP(Image.open(path).convert("RGB"))
        sample = {
            "image": image,
            "label": int("/good/" not in str(path).lower()),
            "image_path": str(path),
        }

        if self.masks_present:
            mask_path = path.parent / f"{path.stem}_mask.png"
            if mask_path.exists():
                mask = Image.open(mask_path)
            else:
                mask = Image.new("L", image.shape[1:][::-1], 0)
            sample["mask"] = torch.from_numpy(np.array(mask)).unsqueeze(0) / 255

        return sample


class MVTecAd2DataModule(LightningDataModule):
    def __init__(self, paths: Paths, batch_size: int = 8) -> None:
        super().__init__()
        self.paths = paths
        self.batch_size = batch_size

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = MVTecAd2Sample(list(self.paths.train.glob("*")), masks_present=False)
        self.val_ds = MVTecAd2Sample(list(self.paths.validation.rglob("*.*")), masks_present=True)
        self.test_ds = MVTecAd2Sample(list(self.paths.test.rglob("*.*")), masks_present=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, self.batch_size, shuffle=False, num_workers=0)

    test_dataloader = val_dataloader
