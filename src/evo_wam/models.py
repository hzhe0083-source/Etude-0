"""Small learned effect interfaces; physical predictions never accept a task."""

from __future__ import annotations

from dataclasses import dataclass
from graphlib import TopologicalSorter
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import (EffectRequirement, PhysicalOutcome, BINDING_UNUSED,
                        BINDING_UNMATCHED, BINDING_UNCERTAIN)


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


def _event_descriptors(step_offsets: Tensor, roles: int, events: int, dtype: torch.dtype) -> Tensor:
    """Structured, directed event-slot coordinates, independent of scene IDs."""
    nodes = torch.arange(step_offsets.numel() * roles * roles * events, device=step_offsets.device)
    event = nodes % events
    target = (nodes // events) % roles
    source = (nodes // (events * roles)) % roles
    time = nodes // (events * roles * roles)
    return torch.cat((F.one_hot(source, roles), F.one_hot(target, roles),
                      F.one_hot(event, events), torch.log1p(step_offsets[time].float())[:, None]), -1).to(dtype)


def _edge_targets(requirement: EffectRequirement, capacity: int,
                  validity: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Canonical edge slots; annotation order is not task semantics."""
    edges, valid = requirement.event_precedence, requirement.label_valid["event_precedence"]
    if validity is not None:
        valid = valid & validity
    if edges.shape[1] > capacity:
        raise ValueError(f"precedence exceeds the configured {capacity} edge slots")
    batch, count, _ = edges.shape
    padded = edges.new_full((batch, capacity, 2), -1)
    known = torch.ones(batch, capacity, dtype=torch.bool, device=edges.device)
    nodes = requirement.events[0].numel()
    # Invisible endpoint values must not choose the slot of a visible edge.
    # Known edges come first, known empty slots next, and unknown slots last.
    safe_edges = torch.where(valid[..., None], edges, -1)
    keys = safe_edges[..., 0] * nodes + safe_edges[..., 1]
    keys = torch.where(safe_edges[..., 0] >= 0, keys, nodes * nodes)
    keys = torch.where(valid, keys, nodes * nodes + 1)
    order = keys.argsort(dim=1, stable=True)
    padded[:, :count] = safe_edges.gather(1, order[..., None].expand(-1, -1, 2))
    known[:, :count] = valid.gather(1, order)
    return padded, known


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
    geometry_tolerance: Tensor
    event_windows: Tensor
    precedence_presence_logits: Tensor
    precedence_source_logits: Tensor
    precedence_target_logits: Tensor

    def probabilities(self) -> dict[str, Tensor]:
        return {"binding": self.binding_logits.softmax(-1),
                "relations": self.relation_logits.sigmoid(),
                "events": self.event_logits.sigmoid()}

    def materialize(self, threshold: float = 0.5, *, interface: str = "full",
                    min_binding_confidence: float = 0.5,
                    min_binding_margin: float = 0.1) -> EffectRequirement:
        """Decode a runtime requirement; model masks never gate training loss.

        All predictions are available for comparison, not asserted true labels.
        Empty predicted requirements remain empty, so ``effect_cost`` rejects
        them when the task is active. Completion must be checked externally.
        """
        if not 0 < threshold < 1:
            raise ValueError("mask threshold must be in (0,1)")
        if interface not in {"geometry", "full"}:
            raise ValueError("interface must be geometry or full")
        if (not math.isfinite(min_binding_confidence) or not 0 < min_binding_confidence < 1
                or not math.isfinite(min_binding_margin) or not 0 < min_binding_margin < 1):
            raise ValueError("binding confidence and margin must be finite and in (0,1)")
        values = {"geometry": self.geometry, "relations": self.relation_logits.sigmoid(),
                  "events": self.event_logits.sigmoid()}
        # Requirements are Boolean desired relations, whereas physical outputs
        # remain probabilities. Do not use these thresholded values as labels.
        values["relations"] = (values["relations"] >= threshold).to(self.geometry.dtype)
        values["events"] = (values["events"] >= threshold).to(self.geometry.dtype)
        entities = self.entity_ids.shape[1]
        if self.binding_logits.shape[-1] != entities + 3:
            raise ValueError("binding output must include UNUSED, UNMATCHED and UNCERTAIN classes")
        probabilities = self.binding_logits.float().softmax(-1)
        best, indices = probabilities.topk(2, dim=-1)
        winner = indices[..., 0]
        binding = torch.where(winner < entities, winner, -(winner - entities + 1))
        confident = ((best[..., 0] >= min_binding_confidence)
                     & ((best[..., 0] - best[..., 1]) >= min_binding_margin))
        binding = torch.where(confident, binding, BINDING_UNCERTAIN)
        masks = {name: logits.sigmoid() >= threshold
                 for name, logits in self.requirement_mask_logits.items()}
        if interface == "geometry":
            for name in ("relations", "events"):
                masks[name] = torch.zeros_like(masks[name])
        used = masks["geometry"].any((1, 3))
        for name in ("relations", "events"):
            used |= masks[name].any((1, 3, 4)) | masks[name].any((1, 2, 4))
        # A contradictory UNUSED prediction is not permission to delete a goal.
        binding = torch.where(used & (binding == BINDING_UNUSED), BINDING_UNCERTAIN, binding)
        valid = {name: torch.ones_like(value, dtype=torch.bool) for name, value in values.items()}
        valid["binding"] = torch.ones_like(binding, dtype=torch.bool)
        windows = self.event_windows.round().to(torch.int64)
        windows[..., 0] = windows[..., 0].clamp_min(1)
        windows[..., 1] = torch.maximum(windows[..., 1], windows[..., 0])
        edges = torch.stack((self.precedence_source_logits.argmax(-1),
                             self.precedence_target_logits.argmax(-1)), -1)
        present = self.precedence_presence_logits.sigmoid() >= threshold
        if interface == "geometry":
            present = torch.zeros_like(present)
        edges = torch.where(present[..., None], edges, -1)
        # Invalid predicted ordering is observable failure, not silently repaired
        # or dropped. Contract validation below rejects cycles/self/inactive edges.
        valid.update(geometry_tolerance=torch.ones_like(self.geometry_tolerance, dtype=torch.bool),
                     event_windows=torch.ones_like(values["events"], dtype=torch.bool),
                     event_precedence=torch.ones_like(present, dtype=torch.bool))
        return EffectRequirement(self.entity_ids, self.step_offsets, binding,
                                 **values, requirement_mask=masks, label_valid=valid,
                                 geometry_tolerance=self.geometry_tolerance,
                                 event_windows=windows, event_precedence=edges).validate()


class RequirementCodec(nn.Module):
    """G/Q: role tokens refer to observed entity features, never numeric IDs.

    Token layout is time-major [B,T*R,D]. ``current`` and ``remaining`` use
    this same codec. A geometry control has identical token/parameter capacity,
    but zeros all relationship/event values and requirement masks.
    """

    def __init__(self, entity_dim: int, geometry_dim: int, relation_dim: int,
                 event_dim: int, roles: int, token_dim: int = 64,
                 interface: str = "full", max_precedence_edges: int = 4):
        super().__init__()
        if interface not in {"full", "geometry"}:
            raise ValueError("interface must be full or geometry")
        if min(entity_dim, geometry_dim, relation_dim, event_dim, roles, token_dim) <= 0:
            raise ValueError("codec dimensions must be positive")
        if type(max_precedence_edges) is not int or max_precedence_edges < 1:
            raise ValueError("max_precedence_edges must be positive")
        self.entity_dim, self.roles, self.token_dim = entity_dim, roles, token_dim
        self.geometry_dim, self.relation_dim, self.event_dim = geometry_dim, relation_dim, event_dim
        self.interface = interface
        self.max_precedence_edges = max_precedence_edges
        width = entity_dim + 3 * geometry_dim + 2 * roles * relation_dim + 4 * roles * event_dim
        self.encoder = nn.Sequential(nn.Linear(width, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim))
        self.binding_status = nn.Parameter(torch.randn(3, entity_dim) / math.sqrt(entity_dim))
        self.role = nn.Parameter(torch.randn(roles, token_dim) / math.sqrt(token_dim))
        self.time = nn.Linear(1, token_dim)
        self.entity_key = nn.Linear(entity_dim, token_dim, bias=False)
        self.binding_query = nn.Linear(token_dim, token_dim, bias=False)
        self.binding_status_logits = nn.Linear(token_dim, 3)
        self.geometry = nn.Linear(token_dim, geometry_dim)
        self.tolerance = nn.Linear(token_dim, geometry_dim)
        self.relations = nn.Sequential(nn.Linear(2 * token_dim, token_dim), nn.SiLU(), nn.Linear(token_dim, relation_dim))
        self.events = nn.Sequential(nn.Linear(2 * token_dim, token_dim), nn.SiLU(), nn.Linear(token_dim, event_dim))
        self.geometry_mask = nn.Linear(token_dim, geometry_dim)
        self.relation_mask = nn.Linear(2 * token_dim, relation_dim)
        self.event_mask = nn.Linear(2 * token_dim, event_dim)
        self.windows = nn.Linear(2 * token_dim, 2 * event_dim)
        self.edge_encoder = nn.Sequential(nn.Linear(2 * (2 * roles + event_dim + 1), token_dim),
                                          nn.SiLU(), nn.Linear(token_dim, token_dim))
        self.event_type = nn.Parameter(torch.randn(event_dim, token_dim) / math.sqrt(token_dim))
        self.event_node = nn.Linear(2 * token_dim, token_dim)
        self.edge_query = nn.Parameter(torch.randn(max_precedence_edges, token_dim) / math.sqrt(token_dim))
        self.edge_source = nn.Linear(token_dim, token_dim)
        self.edge_target = nn.Linear(token_dim, token_dim)
        self.edge_presence = nn.Linear(token_dim, 1)

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
        if ((requirement.binding != BINDING_UNUSED) & ~requirement.label_valid["binding"]).any():
            raise ValueError("goal binding states must be annotated, including unmatched/uncertain states")
        batch, steps, roles, geometry_dim = requirement.geometry.shape
        if (roles, geometry_dim, requirement.relations.shape[-1], requirement.events.shape[-1]) != (
                self.roles, self.geometry_dim, self.relation_dim, self.event_dim):
            raise ValueError("requirement field dimensions do not match codec")
        selected = entity_features.gather(1, requirement.binding.clamp_min(0)[..., None].expand(-1, -1, self.entity_dim))
        status = self.binding_status[(-requirement.binding - 1).clamp(0, 2)]
        selected = torch.where((requirement.binding >= 0)[..., None], selected, status)
        pieces = [selected[:, None].expand(-1, steps, -1, -1)]
        for name in ("geometry", "relations", "events"):
            value = getattr(requirement, name)
            mask, valid = requirement.requirement_mask[name], requirement.label_valid[name]
            if self.interface == "full" or name == "geometry":
                if (mask & ~valid).any():
                    raise ValueError("encoding an executable goal requires known required values; unknown outcome labels remain allowed")
            clean = torch.where(mask & valid, value, 0)
            # Annotation coverage is not task semantics. Never ask the reader
            # to reproduce a dataset annotator's label-validity choices.
            field = torch.cat((clean, mask.to(value.dtype)), -1)
            if self.interface == "geometry" and name != "geometry":
                field = torch.zeros_like(field)
            pieces.append(field.reshape(batch, steps, roles, -1))
        tolerance_mask = requirement.requirement_mask["geometry"]
        if (tolerance_mask & ~requirement.label_valid["geometry_tolerance"]).any():
            raise ValueError("required geometry tolerance must be annotated before encoding")
        pieces.append(torch.where(tolerance_mask, requirement.geometry_tolerance, 0))
        window_mask = requirement.requirement_mask["events"]
        if self.interface == "full":
            if (window_mask & ~requirement.label_valid["event_windows"]).any():
                raise ValueError("required event windows must be annotated before encoding")
            if not requirement.label_valid["event_precedence"].all():
                raise ValueError("event ordering must be annotated before encoding")
            windows = torch.where(window_mask[..., None], requirement.event_windows, 0)
            windows = torch.log1p(windows.to(entity_features.dtype))
        else:
            windows = entity_features.new_zeros(*requirement.events.shape, 2)
        pieces.append(windows.reshape(batch, steps, roles, -1))
        times = _offsets(requirement.step_offsets, entity_features.device, entity_features.dtype)
        tokens = self.encoder(torch.cat(pieces, -1))
        tokens = tokens + self.role[None, None] + self.time(torch.log1p(times[:, None]))[None, :, None]
        if self.interface == "full":
            edges, known = _edge_targets(requirement, self.max_precedence_edges)
            descriptors = _event_descriptors(requirement.step_offsets, roles, self.event_dim, tokens.dtype)
            endpoints = descriptors[edges.clamp_min(0)].flatten(-2)
            edge_values = self.edge_encoder(endpoints)
            edge_values = edge_values * (known & (edges[..., 0] >= 0))[..., None]
            tokens = tokens + edge_values.sum(1)[:, None, None]
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
        logits = torch.cat((logits, self.binding_status_logits(state.mean(1))), -1)
        pairs = _pair_features(state)
        raw_windows = self.windows(pairs).reshape(batch, len(offsets), self.roles, self.roles, self.event_dim, 2)
        starts = 1 + F.softplus(raw_windows[..., 0])
        windows = torch.stack((starts, starts + F.softplus(raw_windows[..., 1])), -1)
        nodes = self.event_node(pairs)[..., None, :] + self.event_type
        nodes = nodes.reshape(batch, -1, self.token_dim)
        queries = self.edge_query[None] + state.mean((1, 2))[:, None]
        source = torch.einsum("bpd,bnd->bpn", self.edge_source(queries), nodes) / math.sqrt(self.token_dim)
        target = torch.einsum("bpd,bnd->bpn", self.edge_target(queries), nodes) / math.sqrt(self.token_dim)
        return DecodedRequirement(entity_ids, step_offsets, logits, self.geometry(state),
                                  self.relations(pairs), self.events(pairs),
                                  {"geometry": self.geometry_mask(state),
                                   "relations": self.relation_mask(pairs), "events": self.event_mask(pairs)},
                                  F.softplus(self.tolerance(state)), windows,
                                  self.edge_presence(queries).squeeze(-1), source, target)

    def forward(self, requirement: EffectRequirement, entity_features: Tensor) -> DecodedRequirement:
        return self.decode(self.encode(requirement, entity_features), entity_features,
                           requirement.step_offsets, requirement.entity_ids)


def decoded_requirement_loss(prediction: DecodedRequirement, requirement: EffectRequirement,
                             interface: str = "full", *,
                             evidence_valid: Mapping[str, Tensor] | None = None) -> Tensor:
    """Data validity gates label losses; predicted masks never gate losses."""
    requirement.validate()
    if interface not in {"full", "geometry"}:
        raise ValueError("interface must be full or geometry")
    if (not torch.equal(prediction.entity_ids, requirement.entity_ids)
            or not torch.equal(prediction.step_offsets, requirement.step_offsets)):
        raise ValueError("prediction and target must use the same entity/time indices")
    if prediction.binding_logits.shape != (*requirement.binding.shape, requirement.entity_ids.shape[1] + 3):
        raise ValueError("binding output shape differs from target")
    evidence = {name: torch.ones_like(valid) for name, valid in requirement.label_valid.items()}
    if evidence_valid is not None:
        if set(evidence_valid) - set(evidence):
            raise ValueError("unknown requirement evidence field")
        for name, valid in evidence_valid.items():
            if (valid.dtype != torch.bool or valid.shape != evidence[name].shape
                    or valid.device != evidence[name].device):
                raise ValueError(f"{name} evidence must match the data-owned validity shape/device")
            evidence[name] = valid
    binding_target = torch.where(~evidence["binding"] & (requirement.binding != BINDING_UNUSED),
                                 BINDING_UNCERTAIN, requirement.binding)
    binding_target = torch.where(binding_target >= 0, binding_target,
                                 requirement.entity_ids.shape[1] - binding_target - 1)
    binding_loss = F.cross_entropy(prediction.binding_logits.transpose(1, 2),
                                   binding_target, reduction="none")
    terms = [_masked_mean(binding_loss, requirement.label_valid["binding"])]
    output = {"geometry": prediction.geometry, "relations": prediction.relation_logits,
              "events": prediction.event_logits}
    for name in (("geometry",) if interface == "geometry" else ("geometry", "relations", "events")):
        target = getattr(requirement, name)
        valid = requirement.label_valid[name] & requirement.requirement_mask[name] & evidence[name]
        safe_target = torch.where(valid, target, 0)
        if output[name].shape != target.shape:
            raise ValueError(f"{name} output shape differs from target")
        loss = ((output[name] - safe_target).square() if name == "geometry" else
                F.binary_cross_entropy_with_logits(output[name], safe_target, reduction="none"))
        terms.append(_masked_mean(loss, valid))
        mask_logits = prediction.requirement_mask_logits[name]
        if mask_logits.shape != target.shape:
            raise ValueError(f"{name} requirement mask shape differs from target")
        mask_loss = F.binary_cross_entropy_with_logits(mask_logits,
                                                       requirement.requirement_mask[name].to(target.dtype), reduction="none")
        terms.append(_masked_mean(mask_loss, evidence[name]))
    tolerance_valid = (requirement.requirement_mask["geometry"]
                       & requirement.label_valid["geometry_tolerance"] & evidence["geometry_tolerance"])
    tolerance_target = torch.where(tolerance_valid, requirement.geometry_tolerance, 0)
    terms.append(_masked_mean((prediction.geometry_tolerance - tolerance_target).square(), tolerance_valid))
    if interface == "full":
        window_valid = (requirement.requirement_mask["events"]
                        & requirement.label_valid["event_windows"] & evidence["event_windows"])
        window_target = torch.where(window_valid[..., None], requirement.event_windows, 0).to(prediction.event_windows.dtype)
        window_error = (torch.log1p(prediction.event_windows) - torch.log1p(window_target)).square().mean(-1)
        terms.append(_masked_mean(window_error, window_valid))
        edges, known = _edge_targets(requirement, prediction.precedence_presence_logits.shape[1],
                                     evidence["event_precedence"])
        present = edges[..., 0] >= 0
        presence_loss = F.binary_cross_entropy_with_logits(prediction.precedence_presence_logits,
                                                           present.to(prediction.geometry.dtype), reduction="none")
        terms.append(_masked_mean(presence_loss, known))
        for end, logits in enumerate((prediction.precedence_source_logits, prediction.precedence_target_logits)):
            endpoint_loss = F.cross_entropy(logits.transpose(1, 2), edges[..., end].clamp_min(0), reduction="none")
            terms.append(_masked_mean(endpoint_loss, known & present))
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


def _windowed_event_error(probabilities: Tensor, requirement: EffectRequirement,
                          event_threshold: float) -> tuple[Tensor, Tensor]:
    """Windowed scores and one globally consistent occurrence assignment.

    For strict before-only DAGs, assigning every node its earliest feasible
    observed occurrence is exact for feasibility: an earlier predecessor never
    reduces a successor's choices. This does not optimize robot actions.
    """
    batch, horizon, _, _, channels = probabilities.shape
    _, slots, roles, _, _ = requirement.events.shape
    errors = probabilities.new_zeros(requirement.events.shape)
    impossible = torch.zeros(batch, dtype=torch.bool, device=probabilities.device)
    for sample in range(batch):
        if not requirement.semantics_known[sample]:
            continue  # effect_cost rejects the whole sample; never drops a term.
        needed = requirement.requirement_mask["events"][sample].flatten()
        targets = requirement.events[sample].flatten()
        windows = requirement.event_windows[sample].reshape(-1, 2)
        nodes = needed.nonzero(as_tuple=True)[0].tolist()
        likelihoods = {}
        for node in nodes:
            event = node % channels
            other = (node // channels) % roles
            role = (node // (channels * roles)) % roles
            first = int(requirement.binding[sample, role])
            second = int(requirement.binding[sample, other])
            start, end = windows[node].tolist()
            if end > horizon:
                raise ValueError("event window extends beyond the candidate horizon")
            likelihoods[node] = probabilities[sample, :, first, second, event]
            observed = likelihoods[node][start - 1:end].max()
            errors[sample].view(-1)[node] = (observed - targets[node]).square()
        graph: dict[int, set[int]] = {}
        for source, target in requirement.event_precedence[sample].tolist():
            if source >= 0:
                graph.setdefault(source, set())
                graph.setdefault(target, set()).add(source)
        earliest = {}
        for node in TopologicalSorter(graph).static_order():
            start, end = windows[node].tolist()
            parents = graph[node]
            if any(earliest[parent] is None for parent in parents):
                earliest[node] = None
            else:
                start = max([start] + [earliest[parent] + 1 for parent in parents])
                matches = (likelihoods[node][start - 1:end] >= event_threshold).nonzero(as_tuple=True)[0]
                earliest[node] = start + int(matches[0]) if matches.numel() else None
            if earliest[node] is None:
                impossible[sample] = True
    return errors, impossible


def effect_cost(prediction: PhysicalPrediction, requirement: EffectRequirement,
                entity_visible: Tensor | None = None, prefix_steps: int | None = None,
                *, field_weights: Mapping[str, float] | None = None,
                uncertainty_weight: float = 1.0, active: Tensor | None = None,
                event_threshold: float = 0.5) -> Tensor:
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
    if not math.isfinite(event_threshold) or not 0 < event_threshold < 1:
        raise ValueError("event threshold must be finite and in (0,1)")
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
    time_needed = requirement.requirement_mask["geometry"].flatten(2).any(-1)
    time_needed |= requirement.requirement_mask["relations"].flatten(2).any(-1)
    if (time_needed & (requirement.step_offsets[None] > horizon)).any():
        # A caller asking about an unpredicted future must extend its planning
        # horizon or use an explicit progress target, not silently drop labels.
        raise ValueError("requirement extends beyond the candidate horizon")
    indices = (requirement.step_offsets - 1).clamp_max(horizon - 1)
    binding = requirement.binding
    roles = binding.shape[1]
    bound = binding.clamp_min(0)
    used_roles = torch.zeros_like(binding, dtype=torch.bool)
    reject = active & (~requirement.has_requirement | ~requirement.resolved | ~requirement.semantics_known)
    score = prediction.geometry.new_zeros(batch)
    denominator = torch.zeros_like(score)
    for name, value in values.items():
        if name != "events":
            value = value[:, indices]
        target = getattr(requirement, name)
        needed = requirement.requirement_mask[name]
        reject |= (needed & ~requirement.label_valid[name]).flatten(1).any(-1)
        if name == "geometry":
            used_roles |= needed.any((1, 3))
            selected = value.gather(2, bound[:, None, :, None].expand(-1, len(indices), -1, value.shape[-1]))
            safe_target = torch.where(needed & requirement.label_valid[name], target, 0)
            safe_tolerance = torch.where(needed & requirement.label_valid["geometry_tolerance"],
                                         requirement.geometry_tolerance, 0)
            error = F.relu((selected - safe_target).abs() - safe_tolerance).square()
        else:
            used_roles |= needed.any((1, 3, 4)) | needed.any((1, 2, 4))
            if name == "events":
                if target.shape[-1] != value.shape[-1]:
                    raise ValueError("event requirement channels do not match prediction")
                error, impossible = _windowed_event_error(value.sigmoid(), requirement, event_threshold)
                reject |= impossible
            else:
                selected = value.gather(2, bound[:, None, :, None, None].expand(-1, len(indices), -1, entities, value.shape[-1]))
                selected = selected.gather(3, bound[:, None, None, :, None].expand(-1, len(indices), roles, -1, value.shape[-1]))
                if selected.shape != target.shape:
                    raise ValueError(f"{name} requirement channels do not match prediction")
                safe_target = torch.where(needed & requirement.label_valid[name], target, 0)
                error = (selected.sigmoid() - safe_target).square()
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
