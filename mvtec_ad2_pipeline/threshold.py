"""Automatic threshold search on validation split."""
from typing import Iterable
import numpy as np
import torch
from PIL import Image
import fiftyone as fo

from .util import RESIZE_CROP, normalise, Tag


class ThresholdFinder:
    def __init__(self, model, dataset: fo.Dataset):
        self.model = model.cpu().eval()
        self.dataset = dataset

    def _anomaly_map(self, image: Image.Image) -> np.ndarray:
        tensor = RESIZE_CROP(image).unsqueeze(0)
        with torch.no_grad():
            amap = self.model(tensor)["anomaly_map"].squeeze().cpu()
        return normalise(amap).numpy()

    def _val_samples(self):
        return self.dataset.match_tags(str(Tag.VAL))

    def best_threshold(self, sweep: Iterable[float]) -> float:
        best_t, best_f1 = 0.6, 0.0

        for t in sweep:
            for s in self._val_samples():
                amap = self._anomaly_map(Image.open(s.filepath))
                s["tmp_mask"] = fo.Segmentation(mask=(amap > t).astype("uint8") * 255)
                s.save()

            f1_score = (
                self.dataset.evaluate_segmentations("tmp_mask", "ground_truth", metric="f1")
                .metrics()
                .get("overall", {})
                .get("f1", 0.0)
            )

            if f1_score > best_f1:
                best_t, best_f1 = t, f1_score

            for s in self._val_samples():
                s.clear_field("tmp_mask")
                s.save()

        print(f"auto-thresh: τ={best_t:.2f} (val F1={best_f1:.3f})")
        return best_t
