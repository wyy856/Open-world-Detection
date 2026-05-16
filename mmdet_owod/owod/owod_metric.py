"""Lightweight open-world detection metrics for smoke testing."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric

from mmdet.registry import METRICS


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _bbox_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], a_min=0, a_max=None) * np.clip(
        boxes1[:, 3] - boxes1[:, 1], a_min=0, a_max=None
    )
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], a_min=0, a_max=None) * np.clip(
        boxes2[:, 3] - boxes2[:, 1], a_min=0, a_max=None
    )
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, a_min=1e-6, a_max=None)


@METRICS.register_module()
class OWODMetric(BaseMetric):
    """Compute simple known/unknown detection diagnostics."""

    default_prefix = "owod"

    def __init__(
        self,
        unknown_label: int,
        iou_thr: float = 0.5,
        score_thr: float = 0.05,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.unknown_label = unknown_label
        self.iou_thr = iou_thr
        self.score_thr = score_thr

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            pred = sample.pred_instances
            gt = sample.gt_instances
            pred_scores = _to_numpy(pred.scores)
            keep = pred_scores >= self.score_thr
            pred_bboxes = _to_numpy(pred.bboxes)[keep]
            pred_labels = _to_numpy(pred.labels)[keep]
            gt_bboxes = _to_numpy(gt.bboxes)
            gt_labels = _to_numpy(gt.labels)
            self.results.append(
                {
                    "pred_bboxes": pred_bboxes,
                    "pred_labels": pred_labels,
                    "gt_bboxes": gt_bboxes,
                    "gt_labels": gt_labels,
                }
            )

    def compute_metrics(self, results: list[dict]) -> dict:
        unknown_gt_total = 0
        unknown_gt_hit = 0
        unknown_pred_total = 0
        unknown_pred_hit = 0
        known_gt_total = 0
        known_gt_hit = 0
        absolute_open_set_error = 0

        for result in results:
            pred_bboxes = result["pred_bboxes"]
            pred_labels = result["pred_labels"]
            gt_bboxes = result["gt_bboxes"]
            gt_labels = result["gt_labels"]

            unknown_gt = gt_labels == self.unknown_label
            known_gt = (gt_labels >= 0) & (gt_labels != self.unknown_label)
            unknown_pred = pred_labels == self.unknown_label

            unknown_gt_boxes = gt_bboxes[unknown_gt]
            known_gt_boxes = gt_bboxes[known_gt]
            unknown_pred_boxes = pred_bboxes[unknown_pred]
            known_pred_boxes = pred_bboxes[~unknown_pred]

            unknown_gt_total += len(unknown_gt_boxes)
            unknown_pred_total += len(unknown_pred_boxes)
            known_gt_total += len(known_gt_boxes)

            if len(unknown_gt_boxes) and len(unknown_pred_boxes):
                ious = _bbox_iou(unknown_gt_boxes, unknown_pred_boxes)
                unknown_gt_hit += int((ious.max(axis=1) >= self.iou_thr).sum())
                unknown_pred_hit += int((ious.max(axis=0) >= self.iou_thr).sum())
            if len(known_gt_boxes) and len(known_pred_boxes):
                ious = _bbox_iou(known_gt_boxes, known_pred_boxes)
                known_gt_hit += int((ious.max(axis=1) >= self.iou_thr).sum())
            if len(known_gt_boxes) and len(unknown_pred_boxes):
                ious = _bbox_iou(known_gt_boxes, unknown_pred_boxes)
                absolute_open_set_error += int((ious.max(axis=1) >= self.iou_thr).sum())

        unknown_recall = unknown_gt_hit / max(unknown_gt_total, 1)
        unknown_precision = unknown_pred_hit / max(unknown_pred_total, 1)
        known_recall = known_gt_hit / max(known_gt_total, 1)
        wilderness_impact = absolute_open_set_error / max(known_gt_hit, 1)
        return {
            "unknown_recall": unknown_recall,
            "unknown_precision": unknown_precision,
            "known_recall": known_recall,
            "absolute_open_set_error": absolute_open_set_error,
            "wilderness_impact": wilderness_impact,
            "unknown_gt": unknown_gt_total,
            "unknown_predictions": unknown_pred_total,
        }
