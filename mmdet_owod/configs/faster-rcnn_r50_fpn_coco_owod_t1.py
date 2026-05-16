"""Faster R-CNN baseline config for COCO open-world task 1.

Use this config from an MMDetection 3.x checkout:

    python tools/train.py /path/to/WOOD-master/mmdet_owod/configs/faster-rcnn_r50_fpn_coco_owod_t1.py

Before training, generate the annotation files with:

    python mmdet_owod/tools/make_coco_owod_split.py \
      --train-json data/coco/annotations/instances_train2017.json \
      --val-json data/coco/annotations/instances_val2017.json \
      --out-dir data/coco/annotations/owod_t1 \
      --known-ids 1-60
"""

custom_imports = dict(imports=["mmdet_owod.owod"], allow_failed_imports=False)

_base_ = "mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"

data_root = "data/coco/"

known_classes = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
)
classes = (*known_classes, "unknown")
num_known_classes = len(known_classes)
num_classes = len(classes)

train_dataloader = dict(
    dataset=dict(
        type="BaseDetDataset",
        data_root=data_root,
        ann_file="annotations/owod_t1/train_mmdet.json",
        data_prefix=dict(img=""),
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
    )
)

val_dataloader = dict(
    dataset=dict(
        type="BaseDetDataset",
        data_root=data_root,
        ann_file="annotations/owod_t1/val_mmdet.json",
        data_prefix=dict(img=""),
        metainfo=dict(classes=classes),
        test_mode=True,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type="OWODMetric", unknown_label=num_known_classes, iou_thr=0.5, score_thr=0.05)
test_evaluator = val_evaluator

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type="OWODShared2FCBBoxHead",
            num_classes=num_classes,
            num_known_classes=num_known_classes,
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            loss_wood=dict(
                type="WOODProposalLoss",
                num_known_classes=num_known_classes,
                beta=0.1,
                unknown_label=num_known_classes,
                background_label=num_classes,
                include_known_ce=False,
                use_dynamic_cost=True,
                blur=1.0,
            ),
            unknown_score_thr=0.08,
            known_score_thr=0.25,
            objectness_thr=0.2,
            unknown_logit_boost=3.0,
            enable_unknown_inference=True,
        )
    )
)

train_cfg = dict(max_epochs=12)

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3),
)

work_dir = "./work_dirs/faster-rcnn_r50_fpn_coco_owod_t1"
