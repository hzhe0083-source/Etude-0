"""Masked per-example losses; only data-provided validity controls coverage."""

from dataclasses import dataclass
import math
from typing import Literal, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MaskedLoss:
    per_sample: Tensor                     # [B], already normalized within each sample
    valid: Tensor                          # bool [B]

    def __post_init__(self) -> None:
        if self.per_sample.ndim != 1 or not self.per_sample.numel() or not self.per_sample.is_floating_point():
            raise ValueError("per_sample must be a nonempty floating point [B] tensor")
        if self.valid.shape != self.per_sample.shape or self.valid.dtype != torch.bool:
            raise ValueError("valid must be bool [B]")
        if self.valid.device != self.per_sample.device:
            raise ValueError("loss and validity must share a device")
        if not torch.isfinite(self.per_sample[self.valid]).all():
            raise ValueError("valid losses must be finite")

    @property
    def valid_count(self) -> Tensor:
        return self.valid.sum()

    @property
    def coverage(self) -> Tensor:
        return self.valid.float().mean()

    @property
    def loss(self) -> Tensor:
        values = torch.where(self.valid, self.per_sample, 0.0)
        return values.sum() / self.valid_count.clamp_min(1)


def _effective_mask(values: Tensor, mask: Tensor, weights: Tensor | None) -> tuple[Tensor, Tensor]:
    if not values.is_floating_point() or not values.ndim or not values.shape[0]:
        raise ValueError("values must be floating point with a nonempty batch axis")
    if mask.dtype != torch.bool or mask.shape != values.shape or mask.device != values.device:
        raise ValueError("mask must be bool and exactly match values, including device")
    if weights is None:
        weights = torch.ones_like(values, dtype=torch.float32)
    else:
        if weights.requires_grad or weights.device != values.device:
            raise ValueError("weights must be fixed data-side weights on the values device")
        try:
            weights = torch.broadcast_to(weights.float(), values.shape)
        except RuntimeError as error:
            raise ValueError("weights must broadcast to values without expanding them") from error
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("weights must be finite and nonnegative")
    return mask & (weights > 0), weights


def masked_mean(values: Tensor, mask: Tensor, weights: Tensor | None = None) -> MaskedLoss:
    """Normalize within each example; NaN/Inf at excluded positions are ignored."""
    valid, weights = _effective_mask(values, mask, weights)
    if not torch.isfinite(values[valid]).all():
        raise ValueError("included values must be finite")
    safe = torch.where(valid, values.float(), 0.0)
    weighted_mask = torch.where(valid, weights, 0.0)
    numerator = (safe * weighted_mask).reshape(values.shape[0], -1).sum(1)
    denominator = weighted_mask.reshape(values.shape[0], -1).sum(1)
    present = denominator > 0
    return MaskedLoss(numerator / torch.where(present, denominator, 1.0), present)


def _safe_pair(prediction: Tensor, target: Tensor, mask: Tensor, weights: Tensor | None):
    if prediction.shape != target.shape or prediction.device != target.device or not target.is_floating_point():
        raise ValueError("prediction and target must have the same floating point shape and device")
    valid, _ = _effective_mask(prediction, mask, weights)
    if not torch.isfinite(prediction[valid]).all() or not torch.isfinite(target[valid]).all():
        raise ValueError("included predictions and targets must be finite")
    # Sanitize before arithmetic: masking an already computed NaN can poison backward.
    return torch.where(valid, prediction.float(), 0.0), torch.where(valid, target.float(), 0.0)


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor, weights: Tensor | None = None) -> MaskedLoss:
    prediction, target = _safe_pair(prediction, target, mask, weights)
    return masked_mean((prediction - target).square(), mask, weights)


def masked_bce(logits: Tensor, target: Tensor, mask: Tensor, weights: Tensor | None = None) -> MaskedLoss:
    logits, target = _safe_pair(logits, target, mask, weights)
    if ((target < 0) | (target > 1)).any():
        raise ValueError("included BCE targets must lie in [0, 1]")
    return masked_mean(F.binary_cross_entropy_with_logits(logits, target, reduction="none"), mask, weights)


def merge_category_mass(probabilities: Tensor, category_groups: Tensor) -> Tensor:
    """Merge annotated equivalence classes, retaining every category's mass.

    ``category_groups`` is an int64 map broadcastable to ``probabilities``.
    Illegal categories still need nonnegative group IDs; -1/drop is forbidden.
    """
    if probabilities.ndim < 2 or not probabilities.is_floating_point():
        raise ValueError("probabilities must have batch and category axes")
    if category_groups.dtype != torch.int64 or category_groups.device != probabilities.device:
        raise ValueError("category_groups must be int64 on the probability device")
    try:
        groups = torch.broadcast_to(category_groups, probabilities.shape)
    except RuntimeError as error:
        raise ValueError("category_groups must broadcast to probabilities") from error
    if not groups.numel() or (groups < 0).any():
        raise ValueError("every category must map to a retained nonnegative group")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    if not torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities[..., 0]), atol=1e-5, rtol=1e-5):
        raise ValueError("category probabilities must sum to one")
    count = int(groups.max().item()) + 1
    if count > probabilities.shape[-1]:
        raise ValueError("group IDs cannot create more groups than source categories")
    return probabilities.new_zeros((*probabilities.shape[:-1], count)).scatter_add(-1, groups, probabilities)


def paired_js(
    logits_a: Tensor,
    logits_b: Tensor,
    valid_a: Tensor,
    valid_b: Tensor,
    *,
    kind: Literal["categorical", "bernoulli"],
    weights: Tensor | None = None,
    category_groups: Tensor | None = None,
) -> MaskedLoss:
    """Natural-log FP32 JS, preserving gradients into both paired branches."""
    if logits_a.shape != logits_b.shape or logits_a.device != logits_b.device:
        raise ValueError("paired logits must have identical shapes and devices")
    if not logits_a.is_floating_point() or not logits_b.is_floating_point():
        raise ValueError("logits must be floating point")
    if kind not in ("categorical", "bernoulli"):
        raise ValueError("kind must be categorical or bernoulli")
    if kind == "categorical" and (logits_a.ndim < 2 or not logits_a.shape[-1]):
        raise ValueError("categorical logits need a nonempty final class axis")
    field = logits_a[..., 0] if kind == "categorical" else logits_a
    _effective_mask(field, valid_a, weights)
    _effective_mask(field, valid_b, weights)
    common = valid_a & valid_b
    effective, _ = _effective_mask(field, common, weights)
    select = effective.unsqueeze(-1) if kind == "categorical" else effective
    select = select.expand_as(logits_a)
    if not torch.isfinite(logits_a[select]).all() or not torch.isfinite(logits_b[select]).all():
        raise ValueError("included logits must be finite")
    first = torch.where(select, logits_a.float(), 0.0)
    second = torch.where(select, logits_b.float(), 0.0)
    if kind == "categorical":
        log_p, log_q = first.log_softmax(-1), second.log_softmax(-1)
    else:
        if category_groups is not None:
            raise ValueError("category merging applies only to categorical fields")
        log_p = torch.stack((F.logsigmoid(-first), F.logsigmoid(first)), -1)
        log_q = torch.stack((F.logsigmoid(-second), F.logsigmoid(second)), -1)
    p, q = log_p.exp(), log_q.exp()
    if category_groups is not None:
        p = merge_category_mass(p, category_groups)
        q = merge_category_mass(q, category_groups)
        tiny = torch.finfo(torch.float32).tiny
        log_p, log_q = p.clamp_min(tiny).log(), q.clamp_min(tiny).log()
    log_midpoint = torch.logaddexp(log_p, log_q) - math.log(2.0)
    divergence = 0.5 * (p * (log_p - log_midpoint) + q * (log_q - log_midpoint)).sum(-1)
    return masked_mean(divergence.clamp_min(0.0), common, weights)


def aggregate_fields(fields: Mapping[str, MaskedLoss], weights: Mapping[str, float] | None = None) -> MaskedLoss:
    """Average available field means with fixed field weights, then examples."""
    if not fields:
        raise ValueError("at least one field is needed, even when its mask is empty")
    weights = {} if weights is None else weights
    if set(weights) - set(fields):
        raise ValueError("weights name unknown fields")
    first = next(iter(fields.values()))
    numerator = torch.zeros_like(first.per_sample)
    denominator = torch.zeros_like(first.per_sample)
    for name, result in fields.items():
        if result.per_sample.shape != first.per_sample.shape or result.per_sample.device != first.per_sample.device:
            raise ValueError("all field losses must share the batch shape and device")
        weight = float(weights.get(name, 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("field weights must be finite and nonnegative")
        numerator = numerator + weight * torch.where(result.valid, result.per_sample, 0.0)
        denominator = denominator + weight * result.valid
    present = denominator > 0
    return MaskedLoss(numerator / torch.where(present, denominator, 1.0), present)


def pair_supervised_mean(first: MaskedLoss | Tensor, second: MaskedLoss | Tensor) -> MaskedLoss | Tensor:
    """The arithmetic two-view mean; never double supervision for adding a view."""
    if isinstance(first, MaskedLoss) and isinstance(second, MaskedLoss):
        if first.per_sample.shape != second.per_sample.shape or first.per_sample.device != second.per_sample.device:
            raise ValueError("paired supervision must share batch shape and device")
        a = torch.where(first.valid, first.per_sample, 0.0)
        b = torch.where(second.valid, second.per_sample, 0.0)
        return MaskedLoss((a + b) / 2, first.valid | second.valid)
    if isinstance(first, Tensor) and isinstance(second, Tensor) and first.shape == second.shape and first.device == second.device:
        return (first + second) / 2
    raise ValueError("paired supervision must be two matching tensors or two MaskedLoss values")
