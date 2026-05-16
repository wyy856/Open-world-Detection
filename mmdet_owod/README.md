# WOOD to COCO Open-World Detection Adapter

This directory adapts WOOD image-level OOD detection into COCO open-world object detection with MMDetection 3.x.

The detector keeps Faster R-CNN responsible for localization. WOOD is moved to proposal/ROI classification scores, and the incremental stage adds a CH-HNN inspired ANN-gated SNN branch for anti-forgetting.

## Target Stack

- MMDetection 3.x
- Faster R-CNN R50-FPN
- COCO 2017 annotations
- WOOD proposal scoring for unknown detection
- CH-HNN style SNN branch for incremental learning

## Directory Layout

```text
mmdet_owod/
  configs/
    faster-rcnn_r50_fpn_coco_owod_t1.py
    faster-rcnn_r50_fpn_coco_owod_t2_chhnn.py
  owod/
    __init__.py
    chhnn_incremental.py
    owod_bbox_head.py
    owod_metric.py
    wood_loss.py
  tools/
    make_coco_owod_split.py
```

## Task 1 Flow

```text
Image
  -> ResNet-50 + FPN
  -> RPN proposals
  -> RoIAlign
  -> OWODShared2FCBBoxHead
       ├─ cls head: known + unknown + background
       ├─ bbox regression head
       └─ WOOD proposal loss / unknown scoring
  -> NMS
  -> bbox + score + label
```

Generate task-1 annotations:

```bash
python mmdet_owod/tools/make_coco_owod_split.py \
  --train-json data/coco/annotations/instances_train2017.json \
  --val-json data/coco/annotations/instances_val2017.json \
  --out-dir data/coco/annotations/owod_t1 \
  --known-ids 1-60 \
  --train-unknown-mode unknown
```

Train task 1:

```bash
PYTHONPATH=/path/to/Open-world-Detection:$PYTHONPATH \
python tools/train.py /path/to/Open-world-Detection/mmdet_owod/configs/faster-rcnn_r50_fpn_coco_owod_t1.py
```

## CH-HNN Incremental Flow

The incremental branch follows the CH-HNN idea and adapts it to ROI features:

```text
RoI feature
  ├─ standard Faster R-CNN cls/bbox branch
  ├─ WOOD unknown scoring branch
  └─ CH-HNN incremental branch
       ├─ ANN modulation gate
       ├─ LIF SNN cell over modulated ROI features
       ├─ SNN class predictor for new classes
       ├─ old-class distillation
       └─ metaplastic consolidation penalty
```

Generate task-2 annotations. COCO category ids are not continuous, so `1-79` corresponds to 70 actual categories:

```bash
python mmdet_owod/tools/make_coco_owod_split.py \
  --train-json data/coco/annotations/instances_train2017.json \
  --val-json data/coco/annotations/instances_val2017.json \
  --out-dir data/coco/annotations/owod_t2 \
  --known-ids 1-79 \
  --train-unknown-mode unknown
```

Train task 2 from task-1 checkpoint:

```bash
PYTHONPATH=/path/to/Open-world-Detection:$PYTHONPATH \
python tools/train.py /path/to/Open-world-Detection/mmdet_owod/configs/faster-rcnn_r50_fpn_coco_owod_t2_chhnn.py
```

## Data Layout

```text
data/coco/
  annotations/
    instances_train2017.json
    instances_val2017.json
    owod_t1/
    owod_t2/
  train2017/
  val2017/
```

## Notes

- `--train-unknown-mode unknown` is easiest for first runs because unknown boxes supervise the explicit unknown class.
- `--train-unknown-mode ignore` is stricter and closer to open-world benchmarking, but needs pseudo-unknown mining for good results.
- `OWODMetric` gives quick diagnostics: unknown recall, unknown precision, known recall, absolute open-set error, and wilderness impact.

## References

- CH-HNN reference project: https://github.com/qqish/CH-HNN
- MMDetection custom dataset docs: https://mmdetection.readthedocs.io/en/v3.0.0/advanced_guides/customize_dataset.html
- MMDetection custom model docs: https://mmdetection.readthedocs.io/en/latest/advanced_guides/customize_models.html
