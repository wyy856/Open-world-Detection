"""CH-HNN inspired incremental module for ROI-level detection features.

This is not a direct copy of CH-HNN. It adapts the same design idea to object
detection: an ANN gate modulates ROI features, an SNN classifier learns
incremental classes, and metaplastic consolidation discourages forgetting.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from mmdet.registry import MODELS
except ImportError:
    MODELS = None


def _register_module(cls):
    if MODELS is not None:
        return MODELS.register_module()(cls)
    return cls


class SurrogateSpike(torch.autograd.Function):
    """Binary spike with a smooth surrogate gradient."""

    @staticmethod
    def forward(ctx, membrane: Tensor, threshold: float, lens: float) -> Tensor:
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        ctx.lens = lens
        return (membrane >= threshold).to(membrane.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (membrane,) = ctx.saved_tensors
        distance = (membrane - ctx.threshold).abs()
        grad = (distance < ctx.lens).to(membrane.dtype) / (2.0 * ctx.lens)
        return grad_output * grad, None, None


class LIFLinearCell(nn.Module):
    """Linear leaky-integrate-and-fire cell."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau: float = 0.5,
        threshold: float = 1.0,
        lens: float = 0.5,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.tau = tau
        self.threshold = threshold
        self.lens = lens

    def forward(self, x: Tensor, membrane: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        current = self.fc(x)
        if membrane is None:
            membrane = torch.zeros_like(current)
        membrane = membrane * self.tau + current
        spike = SurrogateSpike.apply(membrane, self.threshold, self.lens)
        membrane = membrane * (1.0 - spike.detach())
        return spike, membrane


class AnnModulationGate(nn.Module):
    """ANN gate that generates task/class modulation for ROI features."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features),
            nn.Sigmoid(),
        )

    def forward(self, roi_features: Tensor) -> Tensor:
        return self.net(roi_features)


@_register_module
class CHHNNIncrementalModule(nn.Module):
    """ANN-modulated SNN auxiliary classifier with consolidation support."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_classes: int,
        time_steps: int = 4,
        loss_weight: float = 0.2,
        distill_weight: float = 0.5,
        consolidate_weight: float = 1e-4,
        old_classes: int = 0,
        tau: float = 0.5,
        threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.old_classes = old_classes
        self.time_steps = time_steps
        self.loss_weight = loss_weight
        self.distill_weight = distill_weight
        self.consolidate_weight = consolidate_weight

        self.gate = AnnModulationGate(in_features, hidden_features, in_features)
        self.encoder = nn.Linear(in_features, hidden_features)
        self.snn_cell = LIFLinearCell(hidden_features, hidden_features, tau=tau, threshold=threshold)
        self.classifier = nn.Linear(hidden_features, num_classes)

        self.register_buffer("_has_anchor", torch.zeros((), dtype=torch.bool), persistent=False)
        self._anchor_params: dict[str, Tensor] = {}
        self._importance: dict[str, Tensor] = {}

    def forward(self, roi_features: Tensor) -> Tensor:
        gate = self.gate(roi_features)
        x = self.encoder(roi_features * gate)
        membrane = None
        spikes = []
        for _ in range(self.time_steps):
            spike, membrane = self.snn_cell(x, membrane)
            spikes.append(spike)
        spike_rate = torch.stack(spikes, dim=0).mean(dim=0)
        return self.classifier(spike_rate)

    def loss(
        self,
        roi_features: Tensor,
        labels: Tensor,
        label_weights: Optional[Tensor] = None,
        teacher_logits: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        logits = self(roi_features)
        valid = (labels >= 0) & (labels < self.num_classes)
        losses: dict[str, Tensor] = {}
        zero = logits.sum() * 0.0

        if valid.any():
            cls_loss = F.cross_entropy(logits[valid], labels[valid], reduction="none")
            if label_weights is not None:
                cls_loss = cls_loss * label_weights[valid]
            losses["loss_chhnn_cls"] = cls_loss.mean() * self.loss_weight
        else:
            losses["loss_chhnn_cls"] = zero

        if teacher_logits is not None and self.old_classes > 0 and valid.any():
            old_logits = logits[valid, : self.old_classes]
            teacher_old = teacher_logits[valid, : self.old_classes].detach()
            distill = F.kl_div(
                F.log_softmax(old_logits, dim=1),
                F.softmax(teacher_old, dim=1),
                reduction="batchmean",
            )
            losses["loss_chhnn_distill"] = distill * self.distill_weight
        else:
            losses["loss_chhnn_distill"] = zero

        losses["loss_chhnn_consolidate"] = self.consolidation_loss(logits.device)
        return losses

    @torch.no_grad()
    def snapshot_for_next_task(self) -> None:
        """Store current parameters as metaplastic anchors for the next task."""
        self._anchor_params = {}
        self._importance = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            self._anchor_params[name] = param.detach().clone()
            self._importance[name] = param.detach().abs().clone().add_(1e-3)
        self._has_anchor.fill_(True)

    def consolidation_loss(self, device: torch.device) -> Tensor:
        if not bool(self._has_anchor):
            return torch.zeros((), device=device)
        loss = torch.zeros((), device=device)
        for name, param in self.named_parameters():
            if name not in self._anchor_params:
                continue
            anchor = self._anchor_params[name].to(device)
            importance = self._importance[name].to(device)
            loss = loss + (importance * (param - anchor).pow(2)).sum()
        return loss * self.consolidate_weight
