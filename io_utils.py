"""Dataset construction and FiftyOne app utilities."""
import glob
from pathlib import Path
from datetime import datetime

import fiftyone as fo

from .util import Tag
from .config import Paths


def build_fiftyone_dataset(paths: Paths) -> fo.Dataset:
    ds_name = f"ad2_{paths.category.lower()}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    ds = fo.Dataset(ds_name)
    ds.persistent = True
    samples = []

    def add_split(root: Path, tag: Tag, gt_root: Path | None = None) -> None:
        for file in glob.glob(str(root / '**' / '*.*'), recursive=True):
            if Path(file).suffix.lower() not in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}:
                continue

            s = fo.Sample(filepath=file, tags=[str(tag)])

            if gt_root:
                rel = Path(file).relative_to(root)
                mask_path = gt_root / rel.parent / f"{rel.stem}_mask.png"
                if mask_path.exists():
                    s['ground_truth'] = fo.Segmentation(mask_path=str(mask_path))
            samples.append(s)

    add_split(paths.train, Tag.TRAIN)
    add_split(paths.validation, Tag.VAL, paths.validation / 'ground_truth')
    add_split(paths.test, Tag.TEST, paths.test / 'ground_truth')

    ds.add_samples(samples)
    print(f"{ds_name}: {len(ds)} samples")
    return ds


def launch_fiftyone(ds: fo.Dataset) -> None:
    fo.close_app()
    fo.launch_app(view=ds.select_fields(['ground_truth', 'pred_mask'])).wait()
