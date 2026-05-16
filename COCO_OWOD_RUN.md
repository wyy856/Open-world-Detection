# COCO 开放世界目标检测运行说明

这个仓库原始代码是 WOOD 图像分类 OOD 检测。新增的 `mmdet_owod/` 是把 WOOD 思想迁移到 COCO 开放世界目标检测的 MMDetection 3.x 适配层。

## 1. 数据集放在哪里

把 COCO 2017 放在 MMDetection 工作目录或本仓库下的 `data/coco/`：

```text
Open-world-Detection/
  data/
    coco/
      annotations/
        instances_train2017.json
        instances_val2017.json
      train2017/
      val2017/
```

如果你是在独立的 MMDetection 仓库里运行，也使用同样结构：

```text
mmdetection/
  data/
    coco/
      annotations/
      train2017/
      val2017/
```

## 2. 安装环境

建议新建 conda 环境后安装：

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmdet>=3.0.0" geomloss pycocotools
```

然后把本仓库加入 `PYTHONPATH`。

```bash
export PYTHONPATH=/path/to/Open-world-Detection:$PYTHONPATH
```

Windows PowerShell：

```powershell
$env:PYTHONPATH="C:\path\to\Open-world-Detection;$env:PYTHONPATH"
```

## 3. Task 1：生成 COCO 开放世界划分

Task 1 默认把 COCO category id `1-60` 当 known。COCO id 不连续，所以这对应 55 个真实 known 类，其余类别当 unknown。

```bash
python mmdet_owod/tools/make_coco_owod_split.py \
  --train-json data/coco/annotations/instances_train2017.json \
  --val-json data/coco/annotations/instances_val2017.json \
  --out-dir data/coco/annotations/owod_t1 \
  --known-ids 1-60 \
  --train-unknown-mode unknown
```

会生成：

```text
data/coco/annotations/owod_t1/
  train_mmdet.json
  val_mmdet.json
  metadata.json
```

`--train-unknown-mode unknown` 用 unknown 框监督 unknown 类，适合先跑通完整训练。

`--train-unknown-mode ignore` 更接近严格开放世界设定，训练时 unknown 框只作为 ignore 区域，不直接监督 unknown 类。

## 4. Task 1：训练 WOOD-OWOD 检测器

从 MMDetection 根目录运行：

```bash
PYTHONPATH=/path/to/Open-world-Detection:$PYTHONPATH \
python tools/train.py /path/to/Open-world-Detection/mmdet_owod/configs/faster-rcnn_r50_fpn_coco_owod_t1.py
```

输出会放到：

```text
work_dirs/faster-rcnn_r50_fpn_coco_owod_t1/
```

## 5. 当前检测部分实现了什么

- `mmdet_owod/tools/make_coco_owod_split.py`：COCO known/unknown 划分。
- `mmdet_owod/owod/wood_loss.py`：proposal/ROI 级别的 WOOD 损失。
- `mmdet_owod/owod/owod_bbox_head.py`：带 unknown 推理规则的 Faster R-CNN bbox head。
- `mmdet_owod/owod/owod_metric.py`：轻量开放世界验证指标。
- `mmdet_owod/configs/faster-rcnn_r50_fpn_coco_owod_t1.py`：Task 1 训练配置。

## 6. 增量部分：CH-HNN + SNN 抗遗忘

增量阶段参考 `qqish/CH-HNN` 和你仓库里的 CHHNN 思路：不是重新训练一个普通检测器，而是在 RoI bbox head 里额外启用 CH-HNN 风格分支。

```text
Image
  -> Backbone + FPN
  -> RPN proposals
  -> RoIAlign 得到 RoI feature
       ├─ 原 Faster R-CNN cls/bbox head：known / unknown / background + bbox regression
       ├─ WOOD proposal score：unknown 判别
       └─ CH-HNN 增量分支：
            ANN gate 生成调制信号
            SNN LIF cell 处理调制后的 RoI 特征
            SNN classifier 学新类
            distillation + metaplastic consolidation 缓解旧类遗忘
```

新增文件：

- `mmdet_owod/owod/chhnn_incremental.py`：ANN 调制 + SNN 增量分支。
- `mmdet_owod/configs/faster-rcnn_r50_fpn_coco_owod_t2_chhnn.py`：Task 2 增量配置。

## 7. Task 2：生成增量划分

Task 2 使用 COCO category id `1-79` 作为 known。由于 COCO id 不连续，这对应 70 个真实类别；Task 1 的 55 类作为 old classes，新增类别作为 new classes，剩余类别继续作为 unknown。

```bash
python mmdet_owod/tools/make_coco_owod_split.py \
  --train-json data/coco/annotations/instances_train2017.json \
  --val-json data/coco/annotations/instances_val2017.json \
  --out-dir data/coco/annotations/owod_t2 \
  --known-ids 1-79 \
  --train-unknown-mode unknown
```

## 8. Task 2：训练 CH-HNN 增量检测器

先确保 Task 1 checkpoint 在：

```text
work_dirs/faster-rcnn_r50_fpn_coco_owod_t1/latest.pth
```

然后运行：

```bash
PYTHONPATH=/path/to/Open-world-Detection:$PYTHONPATH \
python tools/train.py /path/to/Open-world-Detection/mmdet_owod/configs/faster-rcnn_r50_fpn_coco_owod_t2_chhnn.py
```

Task 2 的输出目录：

```text
work_dirs/faster-rcnn_r50_fpn_coco_owod_t2_chhnn/
```

## 9. 后续实验建议

先用 `--train-unknown-mode unknown` 确认训练、推理、unknown 类输出都正常。之后再切到 `ignore`，加入 pseudo-unknown mining，并补充更完整的开放世界指标：

- unknown recall
- wilderness impact
- absolute open-set error
- known-class mAP
- old-class retention / forgetting rate
