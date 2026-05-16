"""Task-2 CH-HNN incremental config for COCO OWOD.

This config expands the known set to 70 COCO classes and enables a CH-HNN
style ANN-gated SNN auxiliary classifier in the ROI bbox head.
"""

_base_ = "./faster-rcnn_r50_fpn_coco_owod_t1.py"

# Task 2 treats COCO category ids up to 79 as known. Because COCO category ids
# are not continuous, this corresponds to 70 actual categories. Generate
# annotations with:
#
# python mmdet_owod/tools/make_coco_owod_split.py \
#   --train-json data/coco/annotations/instances_train2017.json \
#   --val-json data/coco/annotations/instances_val2017.json \
#   --out-dir data/coco/annotations/owod_t2 \
#   --known-ids 1-79 \
#   --train-unknown-mode unknown

known_classes = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven",
)
classes = (*known_classes, "unknown")
num_known_classes = len(known_classes)
num_classes = len(classes)

train_dataloader = dict(
    dataset=dict(
        ann_file="annotations/owod_t2/train_mmdet.json",
        metainfo=dict(classes=classes),
    )
)

val_dataloader = dict(
    dataset=dict(
        ann_file="annotations/owod_t2/val_mmdet.json",
        metainfo=dict(classes=classes),
    )
)

test_dataloader = val_dataloader
val_evaluator = dict(type="OWODMetric", unknown_label=num_known_classes, iou_thr=0.5, score_thr=0.05)
test_evaluator = val_evaluator

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=num_classes,
            num_known_classes=num_known_classes,
            old_classes=55,
            chhnn_incremental=dict(
                type="CHHNNIncrementalModule",
                in_features=1024,
                hidden_features=512,
                num_classes=num_classes,
                old_classes=55,
                time_steps=4,
                loss_weight=0.2,
                distill_weight=0.5,
                consolidate_weight=1e-4,
                tau=0.5,
                threshold=1.0,
            ),
            loss_wood=dict(
                num_known_classes=num_known_classes,
                unknown_label=num_known_classes,
                background_label=num_classes,
            ),
        )
    )
)

load_from = "./work_dirs/faster-rcnn_r50_fpn_coco_owod_t1/latest.pth"
work_dir = "./work_dirs/faster-rcnn_r50_fpn_coco_owod_t2_chhnn"
