# COCO 开放世界目标检测运行说明

这个仓库原始代码是 WOOD 图像分类 OOD 检测。新增的 `mmdet_owod/` 是把 WOOD 思想迁移到 COCO 开放世界目标检测的 MMDetection 3.x 适配层。

## 1. 数据集放在哪里

把 COCO 2017 放在 MMDetection 工作目录或本仓库下的 `data/coco/`：

```text
WOOD-master/
  data/
    coco/
      annotations/
        instances_train2017.json
        instances_val2017.json
      train2017/
        000000000009.jpg
        ...
      val2017/
        000000000139.jpg
        ...
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

Linux/macOS:

```bash
export PYTHONPATH=/path/to/WOOD-master:$PYTHONPATH
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="C:\path\to\WOOD-master;$env:PYTHONPATH"
```

## 3. 生成 COCO 开放世界划分

默认把 COCO 类别 id `1-60` 当 known，剩余类别当 unknown。

在 `WOOD-master` 目录下运行：

```bash
python mmdet_owod/tools/make_coco_owod_split.py \
  --train-json data/coco/annotations/instances_train2017.json \
  --val-json data/coco/annotations/instances_val2017.json \
  --out-dir data/coco/annotations/owod_t1 \
  --known-ids 1-60 \
  --train-unknown-mode unknown
```

Windows PowerShell:

```powershell
python mmdet_owod\tools\make_coco_owod_split.py `
  --train-json data\coco\annotations\instances_train2017.json `
  --val-json data\coco\annotations\instances_val2017.json `
  --out-dir data\coco\annotations\owod_t1 `
  --known-ids 1-60 `
  --train-unknown-mode unknown
```

会生成：

```text
data/coco/annotations/owod_t1/
  train_mmdet.json
  val_mmdet.json
  metadata.json
```

`--train-unknown-mode unknown` 用 COCO 后 20 类作为 unknown 监督，适合先跑通完整训练。

`--train-unknown-mode ignore` 更接近严格开放世界设定，训练时 unknown 框只作为 ignore 区域，不直接监督 unknown 类。

## 4. 训练

如果你已经安装了 MMDetection，并且当前目录能访问 `tools/train.py`：

```bash
python tools/train.py /path/to/WOOD-master/mmdet_owod/configs/faster-rcnn_r50_fpn_coco_owod_t1.py
```

如果你在 `WOOD-master` 里运行，但没有 MMDetection 源码目录，也可以用 MMDetection 安装后的命令入口，具体取决于你的安装方式。最稳的是下载 MMDetection 源码，然后从 MMDetection 根目录运行上面的 `tools/train.py` 命令。

## 5. 当前实现了什么

新增代码包括：

- `mmdet_owod/tools/make_coco_owod_split.py`：COCO known/unknown 划分。
- `mmdet_owod/owod/wood_loss.py`：proposal/ROI 级别的 WOOD 损失。
- `mmdet_owod/owod/owod_bbox_head.py`：带 unknown 推理规则的 Faster R-CNN bbox head。
- `mmdet_owod/owod/owod_metric.py`：轻量开放世界验证指标。
- `mmdet_owod/configs/faster-rcnn_r50_fpn_coco_owod_t1.py`：训练配置。

## 6. 后续实验建议

先用 `--train-unknown-mode unknown` 确认训练、推理、unknown 类输出都正常。之后再切到 `ignore`，加入 pseudo-unknown mining，并补充开放世界指标：

- unknown recall
- wilderness impact
- absolute open-set error
- known-class mAP
