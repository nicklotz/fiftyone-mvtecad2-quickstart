# fiftyone-mvtecad2-quickstart

This program runs a pre-trained Patchcore model against one cateogry of the MVTecAD2 dataset, evaluates pixel-level F1, then provides the option to visualize the ground truth vs. predictions in FiftyOne.

Tested on MacOS 15.4.1, Apple M4 Pro

### Usage

```
python -m mvtec_ad2_pipeline.main \
  --root /path/to/MVTecAD2 \
  --category vial  # or can, bottle, etc. \
  --auto-thresh
```
