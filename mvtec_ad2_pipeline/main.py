"""Command-line entry point for the pipeline."""
import numpy as np
from pathlib import Path

from .config import CliArgs, Paths
from .data import MVTecAd2DataModule
from .io_utils import build_fiftyone_dataset, launch_fiftyone
from .model import train_patchcore
from .threshold import ThresholdFinder
from .inference import predict_masks, PredictionWriter
from .util import Tag


def run() -> None:
    args = CliArgs.parse()
    paths = Paths(args.root, args.category)

    data = MVTecAd2DataModule(paths)
    ds = build_fiftyone_dataset(paths)
    model = train_patchcore(data)

    if args.auto_threshold:
        finder = ThresholdFinder(model, ds)
        tau = finder.best_threshold(np.linspace(0.30, 0.90, 25))
    else:
        tau = args.threshold
    print(f"inference: τ={tau:.2f}")

    predict_masks(model, ds, tau)
    ds.evaluate_segmentations("pred_mask", "ground_truth", metric="f1").print_report()

    writer = PredictionWriter(Path("ad2_submission/ad2_preds"))
    for s in ds.match_tags(str(Tag.TEST)):
        writer.write_mask(s)
    print("✓ masks written to ad2_submission/ad2_preds")

    launch_fiftyone(ds)


if __name__ == "__main__":
    run()
