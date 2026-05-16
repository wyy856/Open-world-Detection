# WOOD to COCO Open-World Detection Adapter

This directory is a migration scaffold for turning the original WOOD image-classification OOD idea into a COCO open-world object detection experiment.

The original repository trains a DenseNet classifier and computes a Wasserstein/OOD score on image-level class probabilities. For COCO open-world detection, the right unit is no longer an image. The unit is a detector proposal or predicted box. This scaffold therefore keeps MMDetection responsible for detection and moves the WOOD idea into the ROI classification branch.

## Target Stack

- MMDetection 3.x
- Faster R-CNN R50-FPN as the first baseline
- COCO 2017 annotations
- Known/unknown COCO split

MMDetection 3.x expects custom model parts to be registered with `mmdet.registry.MODELS`, and custom dataset class metadata is configured with `metainfo=dict(classes=...)`.

## Directory Layout

```text
mmdet_owod/
  configs/
    faster-rcnn_r50_fpn_coco_owod_t1.py
  owod/
    __init__.py
    owod_bbox_head.py
    owod_metric.py
    wood_loss.py
  tools/
    make_coco_owod_split.py
```

## Step 1: Prepare COCO

Expected COCO layout:

```text
data/coco/
  annotations/
    instances_train2017.json
    instances_val2017.json
  train2017/
  val2017/
```

Generate task-1 open-world annotations:

```bash
python mmdet_owod/tools/make_coco_owod_split.py \
  --train-json data/coco/annotations/instances_train2017.json \
  --val-json data/coco/annotations/instances_val2017.json \
  --out-dir data/coco/annotations/owod_t1 \
  --known-ids 1-60 \
  --train-unknown-mode unknown
```

This creates:

- `train_mmdet.json`: known classes are normal targets; by default unknown boxes are mapped to one extra `unknown` class.
- `val_mmdet.json`: known classes are normal targets; unknown class boxes are mapped to one extra `unknown` class.
- `metadata.json`: split metadata for configs and reporting.

Use `--train-unknown-mode ignore` for a stricter open-world setting where unknown training boxes are ignored. Do not simply remove unknown boxes: the detector will then learn to classify those regions as background.

## Step 2: Install MMDetection

Install MMDetection in a separate environment. Keep this original WOOD repository as a patch/scaffold source.

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmdet>=3.0.0" geomloss pycocotools
```

Then copy or symlink `mmdet_owod/owod` into your MMDetection working directory, or add this repository root to `PYTHONPATH`.

## Step 3: Train A Baseline

From an MMDetection checkout or environment where `tools/train.py` is available:

```bash
PYTHONPATH=/path/to/WOOD-master:$PYTHONPATH \
python tools/train.py /path/to/WOOD-master/mmdet_owod/configs/faster-rcnn_r50_fpn_coco_owod_t1.py
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH="C:\path\to\WOOD-master;$env:PYTHONPATH"
python tools/train.py C:\path\to\WOOD-master\mmdet_owod\configs\faster-rcnn_r50_fpn_coco_owod_t1.py
```

## Step 4: WOOD Unknown Scoring

The config uses `OWODShared2FCBBoxHead`, which subclasses MMDetection's `Shared2FCBBoxHead` and registers through `mmdet.registry.MODELS`.

The file `owod/wood_loss.py` registers `WOODProposalLoss`. It is detector-head agnostic:

- input: proposal/ROI classification logits shaped `(num_rois, num_known_or_open_classes)`
- known labels: `0..num_known_classes-1`
- unknown label: `num_known_classes`
- background label: usually `num_classes` in MMDetection bbox heads

The bbox head integration is in `owod/owod_bbox_head.py`. It:

- keeps standard Faster R-CNN classification and bbox regression losses;
- adds `loss_wood` for labels mapped to the explicit `unknown` class;
- boosts the unknown logit at inference when a proposal has high objectness, low known-class confidence, and high WOOD distance.

The config also uses `OWODMetric` for quick validation diagnostics:

- `unknown_recall`
- `unknown_precision`
- `known_recall`
- `absolute_open_set_error`
- `wilderness_impact`

## What Still Needs Full Detector Integration

This scaffold does not pretend the migration is one file. These are the remaining detector-specific tasks:

1. Add open-world metrics such as unknown recall, wilderness impact, and absolute open-set error.
2. Tune `unknown_score_thr`, `known_score_thr`, `objectness_thr`, and `unknown_logit_boost` on validation.
3. For a stricter OWOD paper-style benchmark, regenerate annotations with `--train-unknown-mode ignore` and add pseudo-unknown mining.

That is the real conversion point from image-level WOOD to object-level open-world detection.

## References

- MMDetection 3.x custom dataset docs: https://mmdetection.readthedocs.io/en/v3.0.0/advanced_guides/customize_dataset.html
- MMDetection 3.x custom model docs: https://mmdetection.readthedocs.io/en/latest/advanced_guides/customize_models.html
