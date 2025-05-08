"""Inference helpers for mask prediction and writing outputs."""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import fiftyone as fo

from .util import RESIZE_CROP, normalise, Tag


class PredictionWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_mask(self, sample: fo.Sample) -> None:
        mask = sample["pred_mask"].mask.squeeze().astype("uint8")
        (self.output_dir / f"{Path(sample.filepath).stem}_mask.png").write_bytes(
            Image.fromarray(mask).tobytes()
        )


def predict_masks(model, dataset: fo.Dataset, thresh: float) -> None:
    model = model.cpu().eval()

    def amap(path: str):
        with torch.no_grad():
            return normalise(
                model(RESIZE_CROP(Image.open(path).convert("RGB")).unsqueeze(0))[
                    "anomaly_map"
                ].squeeze().cpu()
            ).numpy()

    for s in dataset.match_tags(str(Tag.TEST)):
        seg = (amap(s.filepath) > thresh).astype("uint8") * 255
        s["pred_mask"] = fo.Segmentation(mask=seg)
        s.save()
