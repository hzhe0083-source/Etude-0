"""Small learned effect interfaces; physical predictions never accept a task."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import EffectRequirement, PhysicalOutcome


def _finite(name: str, value: Tensor, ndim: int | None = None) -> None:
    if not isinstance(value, Tensor) or (ndim is not None and value.ndim != ndim):
        raise ValueError(f"{name} must be a rank-{ndim} tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _offsets(value: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
    _finite("step_offsets", value, 1)
    if not value.numel() or (value < 0).any() or (value[1:] <= value[:-1]).any():
        raise ValueError("step_offsets must be nonempty, nonnegative and increasing")
    return value.to(device=device, dtype=dtype)


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    if mask.dtype != torch.bool or mask.shape != value.shape:
        raise ValueError("supervision masks must be boolean and match the loss")
    return torch.where(mask, value, 0).sum() / mask.sum().clamp_min(1)


def _pair_features(tokens: Tensor) -> Tensor:
    """Ordered pairs of the penultimate (entity or role) dimension."""
    count = tokens.shape[-2]
    first = tokens.unsqueeze(-2).expand(*tokens.shape[:-2], count, count, tokens.shape[-1])
    second = tokens.unsqueeze(-3).expand_as(first)
    return torch.cat((first, second), dim=-1)


@dataclass
class GoalTokens:
    current: Tensor
    remaining: Tensor


@dataclass
class DecodedRequirement:
    entity_ids: Tensor
    step_offsets: Tensor
    binding_logits: Tensor
    geometry: Tensor
    relation_logits: Tensor
    event_logits: Tensor
    requirement_mask_logits: dict[str, Tensor]

    def probabilities(self) -> dict[str, Tensor]:
        return {"binding": self.binding_logits.softmax(-1),
                "relations": self.relation_logits.sigmoid(),
                "events": self.event_logits.sigmoid()}

    def materialize(self, threshold: float = 0.5) -> EffectRequirement:
        """Decode a runtime requirement; model masks never gate training loss.

        All predictions are available for comparison, not asserted true labels.
        Empty predicted requirements remain empty, so ``effect_cost`` rejects
        them when the task is active. Completion must be checked externally.
        """
        if not 0 < threshold < 1:
            raise ValueError("mask threshold must be in (0,1)")
        values = {"geometry": self.geometry, "relations": self.relation_logits.sigmoid(),
                  "events": self.event_logits.sigmoid()}
        # Requirements are Boolean desired relations, whereas physical outputs
        # remain probabilities. Do not use these thresholded values as labels.
        values["relations"] = (values["relations"] >= threshold).to(self.geometry.dtype)
        values["events"] = (values["events"] >= threshold).to(self.geometry.dtype)
        binding = self.binding_logits.argmax(-1)
        masks = {name: logits.sigmoid() >= threshold
                 for name, logits in self.requirement_mask_logits.items()}
        valid = {name: torch.ones_like(value, dtype=torch.bool) for name, value in values.items()}
        valid["binding"] = torch.ones_like(binding, dtype=torch.bool)
        return EffectRequirement(self.entity_ids, self.step_offsets, binding,
                                 **values, requirement_mask=masks, label_valid=valid).validate()


class RequirementCodec(nn.Module):
    """G/Q: role tokens refer to observed entity features, never numeric IDs.

    Token layout is time-major [B,T*R,D]. ``current`` and ``remaining`` use
    this same codec. A geometry control has identical token/parameter capacity,
    but zeros all relationship/event values, masks and validity inputs.
    """

    def __init__(self, entity_dim: int, geometry_dim: int, relation_dim: int,
                 event_dim: int, roles: int, token_dim: int = 64,
                 interface: str = "full"):
        super().__init__()
        if interface not in {"full", "geometry"}:
            raise ValueError("interface must be full or geometry")
        if min(entity_dim, geometry_dim, relation_dim, event_dim, roles, token_dim) <= 0:
            raise ValueError("codec dimensions must be positive")
        self.entity_dim, self.roles, self.token_dim = entity_dim, roles, token_dim
        self.geometry_dim, self.relation_dim, self.event_dim = geometry_dim, relation_dim, event_dim
        self.interface = interface
        width = entity_dim + 3 * (geometry_dim + roles * (relation_dim + event_dim))
        self.encoder = nn.Sequential(nn.Linear(width, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim))
        self.role = nn.Parameter(torch.randn(roles, token_dim) / math.sqrt(token_dim))
        self.time = nn.Linear(1, token_dim)
        self.entity_key = nn.Linear(entity_dim, token_dim, bias=False)
        self.binding_query = nn.Linear(token_dim, token_dim, bias=False)
        self.geometry = nn.Linear(token_dim, geometry_dim)
        self.relations = nn.Sequential(nn.Linear(2 * token_dim, token_dim), nn.SiLU(), nn.Linear(token_dim, relation_dim))
        self.events = nn.Sequential(nn.Linear(2 * token_dim, token_dim), nn.SiLU(), nn.Linear(token_dim, event_dim))
        self.geometry_mask = nn.Linear(token_dim, geometry_dim)
        self.relation_mask = nn.Linear(2 * token_dim, relation_dim)
        self.event_mask = nn.Linear(2 * token_dim, event_dim)

    def _entities(self, entity_features: Tensor, entity_ids: Tensor) -> None:
        _finite("entity_features", entity_features, 3)
        if entity_features.shape[:2] != entity_ids.shape or entity_features.shape[-1] != self.entity_dim:
            raise ValueError("entity feature shape does not match entity table/codec")
        if entity_ids.dtype != torch.int64 or entity_ids.device != entity_features.device:
            raise ValueError("entity table must be int64 on the feature device")
        if not (entity_ids >= 0).any(-1).all():
            raise ValueError("every sample needs at least one observed entity")

    def encode(self, requirement: EffectRequirement, entity_features: Tensor) -> Tensor:
        requirement.validate()
        self._entities(entity_features, requirement.entity_ids)
        batch, steps, roles, geometry_dim = requirement.geometry.shape
        if (roles, geometry_dim, requirement.relations.shape[-1], requirement.events.shape[-1]) != (
                self.roles, self.geometry_dim, self.relation_dim, self.event_dim):
            raise ValueError("requirement field dimensions do not match codec")
        selected = entity_features.gather(1, requirement.binding.clamp_min(0)[..., None].expand(-1, -1, self.entity_dim))
        selected = selected * (requirement.binding >= 0)[..., None]
        pieces = [selected[:, None].expand(-1, steps, -1, -1)]
        for name in ("geometry", "relations", "events"):
            value = getattr(requirement, name)
            mask, valid = requirement.requirement_mask[name], requirement.label_valid[name]
            clean = torch.where(mask & valid, value, 0)
            field = torch.cat((clean, mask.to(value.dtype), valid.to(value.dtype)), -1)
            if self.interface == "geometry" and name != "geometry":
                field = torch.zeros_like(field)
            pieces.append(field.reshape(batch, steps, roles, -1))
        times = _offsets(requirement.step_offsets, entity_features.device, entity_features.dtype)
        tokens = self.encoder(torch.cat(pieces, -1))
        tokens = tokens + self.role[None, None] + self.time(torch.log1p(times[:, None]))[None, :, None]
        return tokens.reshape(batch, steps * roles, self.token_dim)

    def decode(self, tokens: Tensor, entity_features: Tensor, step_offsets: Tensor,
               entity_ids: Tensor) -> DecodedRequirement:
        self._entities(entity_features, entity_ids)
        _finite("tokens", tokens, 3)
        offsets = _offsets(step_offsets, tokens.device, tokens.dtype)
        batch = entity_features.shape[0]
        if tokens.shape != (batch, len(offsets) * self.roles, self.token_dim):
            raise ValueError("token shape must be [B,len(step_offsets)*roles,token_dim]")
        state = tokens.reshape(batch, len(offsets), self.roles, self.token_dim)
        logits = torch.einsum("brd,bnd->brn", self.binding_query(state.mean(1)), self.entity_key(entity_features))
        logits = logits / math.sqrt(self.token_dim)
        logits = logits.masked_fill(entity_ids[:, None] < 0, torch.finfo(logits.dtype).min)
        pairs = _pair_features(state)
        return DecodedRequirement(entity_ids, step_offsets, logits, self.geometry(state),
                                  self.relations(pairs), self.events(pairs),
                                  {"geometry": self.geometry_mask(state),
                                   "relations": self.relation_mask(pairs), "events": self.event_mask(pairs)})

    def forward(self, requirement: EffectRequirement, entity_features: Tensor) -> DecodedRequirement:
        return self.decode(self.encode(requirement, entity_features), entity_features,
                           requirement.step_offsets, requirement.entity_ids)


def decoded_requirement_loss(prediction: DecodedRequirement, requirement: EffectRequirement,
                             interface: str = "full") -> Tensor:
    """Data validity gates label losses; predicted masks never gate losses."""
    requirement.validate()
    if interface not in {"full", "geometry"}:
        raise ValueError("interface must be full or geometry")
    if (not torch.equal(prediction.entity_ids, requirement.entity_ids)
            or not torch.equal(prediction.step_offsets, requirement.step_offsets)):
        raise ValueError("prediction and target must use the same entity/time indices")
    if prediction.binding_logits.shape[:2] != requirement.binding.shape:
        raise ValueError("binding output shape differs from target")
    binding_loss = F.cross_entropy(prediction.binding_logits.transpose(1, 2),
                                   requirement.binding.clamp_min(0), reduction="none")
    terms = [_masked_mean(binding_loss, requirement.label_valid["binding"])]
    output = {"geometry": prediction.geometry, "relations": prediction.relation_logits,
              "events": prediction.event_logits}
    for name in (("geometry",) if interface == "geometry" else ("geometry", "relations", "events")):
        target = getattr(requirement, name)
        valid = requirement.label_valid[name] & requirement.requirement_mask[name]
        safe_target = torch.where(valid, target, 0)
        if output[name].shape != target.shape:
            raise ValueError(f"{name} output shape differs from target")
        loss = ((output[name] - safe_target).square() if name == "geometry" else
                F.binary_cross_entropy_with_logits(output[name], safe_target, reduction="none"))
        terms.append(_masked_mean(loss, valid))
        mask_logits = prediction.requirement_mask_logits[name]
        if mask_logits.shape != target.shape:
            raise ValueError(f"{name} requirement mask shape differs from target")
        terms.append(F.binary_cross_entropy_with_logits(mask_logits,
                                                        requirement.requirement_mask[name].to(target.dtype)))
    return torch.stack(terms).mean()


class EffectReader(nn.Module):
    """Read supplied demonstration tokens; reuse upstream visual encoders.

    Scene state is a set, while observation history is ordered. Learned role and
    current/remaining queries can attend to different task details. No new
    image backbone, task language bypass or camera classifier is introduced.
    """

    def __init__(self, demo_dim: int, entity_dim: int, proprio_dim: int,
                 embodiment_dim: int, roles: int, token_dim: int = 64):
        super().__init__()
        self.roles, self.token_dim = roles, token_dim
        self.entity_dim, self.proprio_dim = entity_dim, proprio_dim
        self.demo_dim, self.embodiment_dim = demo_dim, embodiment_dim
        self.demo = nn.Linear(demo_dim, token_dim)
        self.entity = nn.Linear(entity_dim, token_dim)
        self.robot = nn.GRU(entity_dim + proprio_dim, token_dim, batch_first=True)
        self.embodiment = nn.Linear(embodiment_dim, token_dim)
        self.query = nn.Parameter(torch.randn(2, roles, token_dim) / math.sqrt(token_dim))
        self.time = nn.Linear(1, token_dim)
        self.output = nn.Sequential(nn.Linear(2 * token_dim, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim))

    def forward(self, demo_tokens: Tensor, entity_history: Tensor,
                proprio_history: Tensor, embodiment: Tensor,
                current_offsets: Tensor, remaining_offsets: Tensor,
                entity_present: Tensor | None = None) -> GoalTokens:
        for name, value, rank in (("demo_tokens", demo_tokens, 3), ("entity_history", entity_history, 4),
                                  ("proprio_history", proprio_history, 3), ("embodiment", embodiment, 2)):
            _finite(name, value, rank)
        batch, length, entities, width = entity_history.shape
        if (width != self.entity_dim or proprio_history.shape != (batch, length, self.proprio_dim)
                or embodiment.shape != (batch, self.embodiment_dim)
                or demo_tokens.shape[0] != batch or demo_tokens.shape[-1] != self.demo_dim
                or min(length, entities, demo_tokens.shape[1]) <= 0):
            raise ValueError("reader input dimensions do not match")
        if entity_present is None:
            scene = entity_history.mean(2)
        else:
            if entity_present.shape != (batch, entities) or entity_present.dtype != torch.bool:
                raise ValueError("entity_present must be boolean [B,N]")
            if not entity_present.any(-1).all():
                raise ValueError("reader needs an observed entity in every sample")
            mask = entity_present[:, None, :, None]
            scene = torch.where(mask, entity_history, 0).sum(2) / entity_present.sum(-1)[:, None, None]
        _, robot = self.robot(torch.cat((scene, proprio_history), -1))
        context = robot[-1] + self.embodiment(embodiment)
        demo = self.demo(demo_tokens)
        # Demonstrations are ordered video tokens, not a set: otherwise a
        # reordered "hold, release" / "release, hold" prompt is indistinguishable.
        positions = torch.arange(demo.shape[1], device=demo.device, dtype=torch.float32)[:, None]
        frequencies = torch.exp(torch.arange(0, self.token_dim, 2, device=demo.device,
                                              dtype=torch.float32) * (-math.log(10000) / self.token_dim))
        angles = positions * frequencies[None]
        positional = torch.zeros(demo.shape[1], self.token_dim, device=demo.device, dtype=torch.float32)
        positional[:, 0::2] = angles.sin()
        positional[:, 1::2] = angles[:, :self.token_dim // 2].cos()
        demo = demo + positional.to(demo.dtype)[None]
        scene_tokens = self.entity(entity_history[:, -1])
        outputs = []
        for kind, offsets in enumerate((current_offsets, remaining_offsets)):
            times = _offsets(offsets, demo.device, demo.dtype)
            query = self.query[kind][None, None] + self.time(torch.log1p(times[:, None]))[None, :, None]
            query = query + context[:, None, None]
            attention = torch.einsum("btrd,bsd->btrs", query, demo).div(math.sqrt(self.token_dim)).softmax(-1)
            attended = torch.einsum("btrs,bsd->btrd", attention, demo)
            binding_scores = torch.einsum("btrd,bnd->btrn", query + attended, scene_tokens) / math.sqrt(self.token_dim)
            if entity_present is not None:
                binding_scores = binding_scores.masked_fill(~entity_present[:, None, None], torch.finfo(binding_scores.dtype).min)
            scene_attended = torch.einsum("btrn,bnd->btrd", binding_scores.softmax(-1), scene_tokens)
            outputs.append(self.output(torch.cat((query + scene_attended, attended), -1)).reshape(batch, -1, self.token_dim))
        return GoalTokens(*outputs)


@dataclass
class PhysicalPrediction:
    geometry: Tensor                 # B,H,N,D; physical scene entity order
    relation_logits: Tensor          # B,H,N,N,C; independent Bernoulli fields
    event_logits: Tensor             # B,H,N,N,E
    prediction_uncertainty: Tensor   # B,H,N; never a supervision mask


class CausalEffectPredictor(nn.Module):
    """Shared per-entity recurrent predictor with ordered pair relation readouts.

    There is intentionally no demonstration, goal, text or WAM-cache argument.
    History features must be produced before task conditioning. Motion outputs
    are relative to the current observed state, not relative to the previous
    predicted frame. All entities use the same parameters.
    """

    def __init__(self, observation_dim: int, proprio_dim: int, action_dim: int,
                 embodiment_dim: int, geometry_dim: int, relation_dim: int,
                 event_dim: int, hidden_dim: int = 64):
        super().__init__()
        dimensions = (observation_dim, proprio_dim, action_dim, embodiment_dim,
                      geometry_dim, relation_dim, event_dim, hidden_dim)
        if min(dimensions) <= 0:
            raise ValueError("all predictor dimensions must be positive")
        self.observation_dim, self.proprio_dim = observation_dim, proprio_dim
        self.action_dim, self.embodiment_dim = action_dim, embodiment_dim
        self.history = nn.GRU(observation_dim + proprio_dim, hidden_dim, batch_first=True)
        self.future = nn.GRU(action_dim + embodiment_dim + hidden_dim,
                             hidden_dim, batch_first=True)
        self.geometry = nn.Linear(hidden_dim, geometry_dim)
        self.relations = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, relation_dim))
        self.events = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, event_dim))
        self.uncertainty = nn.Linear(hidden_dim, 1)

    def forward(self, history: Tensor, proprio: Tensor, actions: Tensor,
                embodiment: Tensor, *, entity_present: Tensor | None = None) -> PhysicalPrediction:
        for name, value, rank in (("history", history, 4), ("proprio", proprio, 3),
                                  ("actions", actions, 3), ("embodiment", embodiment, 2)):
            _finite(name, value, rank)
        batch, length, entities, width = history.shape
        horizon = actions.shape[1]
        if min(batch, length, entities, horizon) <= 0:
            raise ValueError("predictor axes must not be empty")
        if (width != self.observation_dim or proprio.shape != (batch, length, self.proprio_dim)
                or actions.shape != (batch, horizon, self.action_dim)
                or embodiment.shape != (batch, self.embodiment_dim)):
            raise ValueError("predictor feature dimensions or batches do not match")
        state = torch.cat((history, proprio[:, :, None].expand(-1, -1, entities, -1)), -1)
        state = state.transpose(1, 2).reshape(batch * entities, length, -1)
        _, encoded = self.history(state)
        # Shared scene context is invariant to entity permutation; it supplies
        # other-object evidence without using task-dependent attention caches.
        entity_state = encoded[0].reshape(batch, entities, -1)
        if entity_present is None:
            scene = entity_state.mean(1)
        else:
            if (entity_present.shape != (batch, entities) or entity_present.dtype != torch.bool
                    or not entity_present.any(-1).all()):
                raise ValueError("entity_present must be boolean [B,N] with a present entity per sample")
            scene = torch.where(entity_present[..., None], entity_state, 0).sum(1)
            scene = scene / entity_present.sum(-1, keepdim=True)
        inputs = torch.cat((actions, embodiment[:, None].expand(-1, horizon, -1),
                            scene[:, None].expand(-1, horizon, -1)), -1)
        inputs = inputs[:, None].expand(-1, entities, -1, -1).reshape(batch * entities, horizon, -1)
        hidden, _ = self.future(inputs, encoded)
        hidden = hidden.reshape(batch, entities, horizon, -1).transpose(1, 2)
        pairs = _pair_features(hidden)
        return PhysicalPrediction(self.geometry(hidden), self.relations(pairs),
                                  self.events(pairs), F.softplus(self.uncertainty(hidden)).squeeze(-1))


class TemporalInteractionHead(nn.Module):
    """One shared time-query head; Phi must retain aligned entity features.

    Phi is [B,N,D] or [B,N,S,D] (pool S only). A caller using video patch tokens
    must first align/pool them to scene entities using observations, not goals.
    This head never receives demonstration, requirement or candidate actions.
    """

    def __init__(self, feature_dim: int, relation_dim: int, event_dim: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.state = nn.Linear(feature_dim, hidden_dim)
        self.time = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.relations = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, relation_dim))
        self.events = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, event_dim))

    def forward(self, phi: Tensor, offsets: Tensor) -> dict[str, Tensor]:
        _finite("phi", phi)
        if phi.ndim == 4:
            phi = phi.mean(-2)
        if phi.ndim != 3 or phi.shape[-1] != self.feature_dim or phi.shape[1] == 0:
            raise ValueError("phi must be entity-aligned [B,N,D] or [B,N,S,D]")
        times = _offsets(offsets, phi.device, phi.dtype)
        hidden = self.state(phi)[:, None] + self.time(torch.log1p(times[:, None]))[None, :, None]
        pairs = _pair_features(hidden)
        return {"relation_logits": self.relations(pairs), "event_logits": self.events(pairs)}


def _prediction_fields(prediction: PhysicalPrediction) -> dict[str, Tensor]:
    values = {"geometry": prediction.geometry, "relations": prediction.relation_logits,
              "events": prediction.event_logits}
    for name, value in values.items():
        _finite(name, value, 4 if name == "geometry" else 5)
    batch, horizon, entities, _ = prediction.geometry.shape
    for name in ("relations", "events"):
        if values[name].shape[:4] != (batch, horizon, entities, entities):
            raise ValueError(f"{name} prediction is not aligned with geometry")
    uncertainty = prediction.prediction_uncertainty
    _finite("prediction_uncertainty", uncertainty, 3)
    if uncertainty.shape != (batch, horizon, entities) or (uncertainty < 0).any():
        raise ValueError("uncertainty must be nonnegative [B,H,N]")
    return values


def physical_prediction_loss(prediction: PhysicalPrediction, outcome: PhysicalOutcome) -> Tensor:
    """Supervise actual executed steps only; no prediction can disable a label.

    The uncertainty head estimates detached per-entity residual magnitude. It
    is not a calibrated success probability; validation calibration is still
    required before using a rejection threshold.
    """
    outcome.validate()
    values = _prediction_fields(prediction)
    if outcome.step_offsets[-1] > prediction.geometry.shape[1]:
        raise ValueError("outcome extends beyond the predicted/executed horizon")
    indices = outcome.step_offsets - 1
    batch, steps, entities, _ = outcome.geometry.shape
    residual = prediction.geometry.new_zeros(batch, steps, entities)
    counts = torch.zeros_like(residual)
    terms = []
    for name, value in values.items():
        selected = value[:, indices]
        target = getattr(outcome, name)
        if selected.shape != target.shape:
            raise ValueError(f"{name} labels do not match prediction")
        valid = outcome.label_valid[name]
        safe = torch.where(valid, target, 0)
        if name == "geometry":
            point_loss = (selected - safe).square()
            error = point_loss
        else:
            point_loss = F.binary_cross_entropy_with_logits(selected, safe, reduction="none")
            error = (selected.sigmoid() - safe).square()
        terms.append(_masked_mean(point_loss, valid))
        observed = torch.where(valid, error.detach(), 0)
        if name == "geometry":
            residual = residual + observed.sum(-1)
            counts = counts + valid.sum(-1)
        else:
            residual = residual + observed.sum((-1, -2)) + observed.sum((-1, -3))
            counts = counts + valid.sum((-1, -2)) + valid.sum((-1, -3))
    target_uncertainty = residual / counts.clamp_min(1)
    unc_loss = (prediction.prediction_uncertainty[:, indices] - target_uncertainty).square()
    terms.append(_masked_mean(unc_loss, counts > 0))
    return torch.stack(terms).mean()


def effect_cost(prediction: PhysicalPrediction, requirement: EffectRequirement,
                entity_visible: Tensor | None = None, prefix_steps: int | None = None,
                *, field_weights: Mapping[str, float] | None = None,
                uncertainty_weight: float = 1.0, active: Tensor | None = None) -> Tensor:
    """Compare a complete action window against time-indexed requirements.

    ``step_offsets`` are one-based action steps. A terminal requirement stays
    at its terminal offset even when deployment executes a short prefix. The
    caller must pass the same entity ordering used to construct F's history.
    ``prefix_steps`` adds prefix uncertainty for involved entities; it never
    shifts completion requirements into the prefix. Scores are matching costs,
    not probabilities or hardware safety certificates. Inf means reject.
    """
    requirement.validate()
    values = _prediction_fields(prediction)
    batch, horizon, entities, geometry_dim = prediction.geometry.shape
    if requirement.entity_ids.shape != (batch, entities):
        raise ValueError("physical prediction and requirement entity tables differ")
    if requirement.geometry.shape[-1] != geometry_dim:
        raise ValueError("geometry dimensions differ")
    if prefix_steps is None:
        prefix_steps = max(1, horizon // 4)
    if not isinstance(prefix_steps, int) or not 1 <= prefix_steps <= horizon:
        raise ValueError("prefix_steps must be in [1,H]")
    if not math.isfinite(uncertainty_weight) or uncertainty_weight < 0:
        raise ValueError("uncertainty weight must be finite and nonnegative")
    weights = {"geometry": 1.0, "relations": 1.0, "events": 1.0}
    if field_weights is not None:
        if set(field_weights) != set(weights):
            raise ValueError("field_weights must name geometry, relations and events")
        weights.update(field_weights)
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights.values()):
        raise ValueError("field weights must be finite and positive")
    if entity_visible is None:
        entity_visible = requirement.entity_ids >= 0
    if entity_visible.shape != (batch, entities) or entity_visible.dtype != torch.bool:
        raise ValueError("entity_visible must be boolean [B,N]")
    if active is None:
        active = torch.ones(batch, dtype=torch.bool, device=prediction.geometry.device)
    if active.shape != (batch,) or active.dtype != torch.bool:
        raise ValueError("active must be boolean [B]")
    if (requirement.step_offsets > horizon).any():
        # A caller asking about an unpredicted future must extend its planning
        # horizon or use an explicit progress target, not silently drop labels.
        raise ValueError("requirement extends beyond the candidate horizon")
    indices = requirement.step_offsets - 1
    binding = requirement.binding
    roles = binding.shape[1]
    bound = binding.clamp_min(0)
    used_roles = torch.zeros_like(binding, dtype=torch.bool)
    reject = active & ~requirement.has_requirement
    score = prediction.geometry.new_zeros(batch)
    denominator = torch.zeros_like(score)
    for name, value in values.items():
        value = value[:, indices]
        target = getattr(requirement, name)
        needed = requirement.requirement_mask[name]
        reject |= (needed & ~requirement.label_valid[name]).flatten(1).any(-1)
        if name == "geometry":
            used_roles |= needed.any((1, 3))
            selected = value.gather(2, bound[:, None, :, None].expand(-1, len(indices), -1, value.shape[-1]))
            error = (selected - torch.where(needed, target, 0)).square()
        else:
            used_roles |= needed.any((1, 3, 4)) | needed.any((1, 2, 4))
            selected = value.gather(2, bound[:, None, :, None, None].expand(-1, len(indices), -1, entities, value.shape[-1]))
            selected = selected.gather(3, bound[:, None, None, :, None].expand(-1, len(indices), roles, -1, value.shape[-1]))
            if selected.shape != target.shape:
                raise ValueError(f"{name} requirement channels do not match prediction")
            error = (selected.sigmoid() - torch.where(needed, target, 0)).square()
        count = needed.flatten(1).sum(-1)
        term = torch.where(needed, error, 0).flatten(1).sum(-1) / count.clamp_min(1)
        present = count > 0
        score = score + weights[name] * term
        denominator = denominator + weights[name] * present
    visible_roles = entity_visible.gather(1, bound)
    reject |= (used_roles & (~visible_roles | (binding < 0) | ~requirement.label_valid["binding"])).any(-1)
    role_uncertainty = prediction.prediction_uncertainty.gather(2, bound[:, None].expand(-1, horizon, -1))
    uncertainty_mask = used_roles[:, None].expand(-1, horizon, -1)
    uncertainty = torch.where(uncertainty_mask, role_uncertainty, 0).sum((1, 2))
    uncertainty = uncertainty / uncertainty_mask.sum((1, 2)).clamp_min(1)
    prefix_mask = uncertainty_mask[:, :prefix_steps]
    prefix_uncertainty = torch.where(prefix_mask, role_uncertainty[:, :prefix_steps], 0).sum((1, 2))
    prefix_uncertainty = prefix_uncertainty / prefix_mask.sum((1, 2)).clamp_min(1)
    score = score / denominator.clamp_min(1) + uncertainty_weight * (uncertainty + prefix_uncertainty) / 2
    # An explicitly inactive task is separately verified by the caller. It
    # should not invent a no-op requirement, nor count as an active zero-cost win.
    score = torch.where(active, score, torch.zeros_like(score))
    return score.masked_fill(reject & active, float("inf"))
