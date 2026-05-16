"""WOOD-style proposal loss for open-world object detection."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from geomloss import SamplesLoss
except ImportError:
    SamplesLoss = None

try:
    from mmdet.registry import MODELS
except ImportError:
    MODELS = None


def _register_module(cls):
    if MODELS is not None:
        return MODELS.register_module()(cls)
    return cls


def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    labels = labels.long().view(-1, 1)
    out = labels.new_zeros((labels.shape[0], num_classes), dtype=torch.float32)
    return out.scatter_(1, labels, 1.0)


def _class_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        n = x.shape[0]
        m = y.shape[0]
        return 1.0 - torch.eye(n, m, device=x.device, dtype=x.dtype)
    if x.ndim == 3:
        batch = x.shape[0]
        n = x.shape[1]
        m = y.shape[1]
        return (1.0 - torch.eye(n, m, device=x.device, dtype=x.dtype)).unsqueeze(0).repeat(batch, 1, 1)
    raise ValueError(f"Unsupported tensor shape for class cost: {tuple(x.shape)}")


@_register_module
class WOODProposalLoss(nn.Module):
    """WOOD regularizer for ROI/proposal classification logits."""

    def __init__(
        self,
        num_known_classes: int,
        beta: float = 0.1,
        unknown_label: Optional[int] = None,
        background_label: Optional[int] = None,
        include_known_ce: bool = False,
        use_dynamic_cost: bool = True,
        blur: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_known_classes = num_known_classes
        self.beta = beta
        self.unknown_label = num_known_classes if unknown_label is None else unknown_label
        self.background_label = num_known_classes + 1 if background_label is None else background_label
        self.include_known_ce = include_known_ce
        self.use_dynamic_cost = use_dynamic_cost
        self.blur = blur
        self.eps = eps

    def forward(
        self,
        cls_score: torch.Tensor,
        labels: torch.Tensor,
        label_weights: Optional[torch.Tensor] = None,
        avg_factor: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        if SamplesLoss is None:
            raise ImportError("WOODProposalLoss requires geomloss. Install it with `pip install geomloss`.")
        if cls_score.ndim != 2:
            raise ValueError(f"cls_score must be 2-D, got {tuple(cls_score.shape)}")

        known_logits = cls_score[:, : self.num_known_classes]
        known_probs = F.softmax(known_logits, dim=1).clamp_min(self.eps)
        loss = cls_score.sum() * 0.0

        known_mask = (labels >= 0) & (labels < self.num_known_classes)
        if self.include_known_ce and known_mask.any():
            ce = F.cross_entropy(known_logits[known_mask], labels[known_mask], reduction="none")
            if label_weights is not None:
                ce = ce * label_weights[known_mask]
            denom = avg_factor if avg_factor is not None else max(int(known_mask.sum()), 1)
            loss = loss + ce.sum() / denom

        unknown_mask = labels == self.unknown_label
        if unknown_mask.any():
            unknown_probs = known_probs[unknown_mask]
            wood_score = self._min_distance_to_known_onehot(unknown_probs)
            if label_weights is not None:
                wood_score = wood_score * label_weights[unknown_mask]
            denom = avg_factor if avg_factor is not None else max(int(unknown_mask.sum()), 1)
            loss = loss - self.beta * wood_score.sum() / denom
        return loss

    def score(self, cls_score: torch.Tensor) -> torch.Tensor:
        known_logits = cls_score[:, : self.num_known_classes]
        known_probs = F.softmax(known_logits, dim=1).clamp_min(self.eps)
        return self._min_distance_to_known_onehot(known_probs)

    def _min_distance_to_known_onehot(self, probs: torch.Tensor) -> torch.Tensor:
        all_labels = torch.arange(self.num_known_classes, device=probs.device)
        targets = _one_hot(all_labels, self.num_known_classes).unsqueeze(-1)
        inputs = probs.unsqueeze(-1)
        loss_fn = SamplesLoss(
            "sinkhorn",
            p=2,
            blur=self.blur,
            cost=_class_cost if self.use_dynamic_cost else None,
        )
        values = []
        for item in inputs:
            repeated = item.unsqueeze(0).repeat(self.num_known_classes, 1, 1)
            dist = loss_fn(repeated[:, :, 0], repeated, targets[:, :, 0], targets)
            values.append(dist.min())
        return torch.stack(values, dim=0)
