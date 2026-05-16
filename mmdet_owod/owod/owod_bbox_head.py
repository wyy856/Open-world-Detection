"""MMDetection bbox head with WOOD unknown regularization."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.layers import multiclass_nms
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead
from mmdet.models.utils import empty_instances
from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor, scale_boxes


@MODELS.register_module()
class OWODShared2FCBBoxHead(Shared2FCBBoxHead):
    """Shared2FC bbox head with an explicit unknown class and WOOD scoring."""

    def __init__(
        self,
        num_known_classes: int,
        loss_wood: Optional[dict] = None,
        unknown_score_thr: float = 0.08,
        known_score_thr: float = 0.25,
        objectness_thr: float = 0.2,
        unknown_logit_boost: float = 3.0,
        enable_unknown_inference: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_known_classes = num_known_classes
        self.unknown_label = num_known_classes
        self.background_label = self.num_classes
        self.unknown_score_thr = unknown_score_thr
        self.known_score_thr = known_score_thr
        self.objectness_thr = objectness_thr
        self.unknown_logit_boost = unknown_logit_boost
        self.enable_unknown_inference = enable_unknown_inference

        if loss_wood is None:
            self.loss_wood = None
        else:
            loss_wood = loss_wood.copy()
            loss_wood.setdefault("num_known_classes", num_known_classes)
            loss_wood.setdefault("unknown_label", self.unknown_label)
            loss_wood.setdefault("background_label", self.background_label)
            self.loss_wood = MODELS.build(loss_wood)

    def loss(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        rois: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        bbox_weights: Tensor,
        reduction_override: Optional[str] = None,
    ) -> dict:
        losses = super().loss(
            cls_score,
            bbox_pred,
            rois,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            reduction_override=reduction_override,
        )
        if self.loss_wood is not None and cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.0)
            losses["loss_wood"] = self.loss_wood(
                cls_score,
                labels,
                label_weights=label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override,
            )
        return losses

    def _boost_unknown_logits(self, cls_score: Tensor) -> Tensor:
        if not self.enable_unknown_inference or self.loss_wood is None or cls_score.numel() == 0:
            return cls_score

        scores = F.softmax(cls_score, dim=-1)
        known_scores = scores[:, : self.num_known_classes]
        background_scores = scores[:, self.background_label]
        max_known_scores = known_scores.max(dim=1).values
        objectness_scores = 1.0 - background_scores
        wood_scores = self.loss_wood.score(cls_score)

        unknown_like = (
            (wood_scores >= self.unknown_score_thr)
            & (max_known_scores <= self.known_score_thr)
            & (objectness_scores >= self.objectness_thr)
        )
        if not unknown_like.any():
            return cls_score

        boosted = cls_score.clone()
        known_anchor = cls_score[:, : self.num_known_classes].max(dim=1).values
        boosted_unknown = known_anchor + self.unknown_logit_boost
        boosted[unknown_like, self.unknown_label] = torch.maximum(
            boosted[unknown_like, self.unknown_label], boosted_unknown[unknown_like]
        )
        return boosted

    def _predict_by_feat_single(
        self,
        roi: Tensor,
        cls_score: Tensor,
        bbox_pred: Tensor,
        img_meta: dict,
        rescale: bool = False,
        rcnn_test_cfg: Optional[ConfigDict] = None,
    ) -> InstanceData:
        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances(
                [img_meta],
                roi.device,
                task_type="bbox",
                instance_results=[results],
                box_type=self.predict_box_type,
                use_box_type=False,
                num_classes=self.num_classes,
                score_per_cls=rcnn_test_cfg is None,
            )[0]

        cls_score = self._boost_unknown_logits(cls_score)
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(cls_score, dim=-1) if cls_score is not None else None

        img_shape = img_meta["img_shape"]
        num_rois = roi.size(0)
        if bbox_pred is not None:
            num_classes = 1 if self.reg_class_agnostic else self.num_classes
            roi = roi.repeat_interleave(num_classes, dim=0)
            bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
            bboxes = self.bbox_coder.decode(roi[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = roi[:, 1:].clone()
            if img_shape is not None and bboxes.size(-1) == 4:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            assert img_meta.get("scale_factor") is not None
            scale_factor = [1 / s for s in img_meta["scale_factor"]]
            bboxes = scale_boxes(bboxes, scale_factor)

        bboxes = get_box_tensor(bboxes)
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)

        if rcnn_test_cfg is None:
            results.bboxes = bboxes
            results.scores = scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                box_dim=box_dim,
            )
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels
        return results
