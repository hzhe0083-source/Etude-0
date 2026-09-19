"""Task requirements and task-independent, scene-indexed outcome labels.

Geometry uses the caller's fixed robot frame and the current time as its origin.
Relations are sampled at ``step_offsets``; events describe the intervals ending
at those offsets. Event absence is valid only for fully observed intervals.
"""

from dataclasses import dataclass, replace
from graphlib import CycleError, TopologicalSorter

import torch
from torch import Tensor


FIELDS = ("geometry", "relations", "events")
REQUIREMENT_SEMANTICS = ("geometry_tolerance", "event_windows", "event_precedence")
BINDING_UNUSED = -1
BINDING_UNMATCHED = -2
BINDING_UNCERTAIN = -3


def _scene(entity_ids: Tensor, step_offsets: Tensor) -> tuple[int, int, int]:
    if entity_ids.dtype != torch.int64 or entity_ids.ndim != 2:
        raise ValueError("entity_ids must be int64 [B, N]")
    if step_offsets.dtype != torch.int64 or step_offsets.ndim != 1:
        raise ValueError("step_offsets must be int64 [T]")
    if entity_ids.device != step_offsets.device:
        raise ValueError("scene tensors must share a device")
    batch, entities = entity_ids.shape
    if not batch or not entities or not step_offsets.numel():
        raise ValueError("batch, entity table and time grid must be nonempty")
    if (entity_ids < -1).any() or (step_offsets <= 0).any():
        raise ValueError("entity IDs must be nonnegative or -1; time offsets must be positive")
    if (step_offsets[1:] <= step_offsets[:-1]).any():
        raise ValueError("step_offsets must be strictly increasing")
    for row in entity_ids:
        present = row[row >= 0]
        if present.unique().numel() != present.numel():
            raise ValueError("present entity IDs must be unique within each sample")
    return batch, entities, step_offsets.numel()


def _mask(mask: Tensor, shape: tuple, device: torch.device, name: str) -> None:
    if mask.dtype != torch.bool or tuple(mask.shape) != tuple(shape) or mask.device != device:
        raise ValueError(f"{name} must be bool with shape {tuple(shape)} on {device}")


def _fields(owner, slots: int, batch: int, steps: int) -> None:
    expected_keys = set(FIELDS)
    if hasattr(owner, "binding"):
        expected_keys |= {"binding", *REQUIREMENT_SEMANTICS}
    if set(owner.label_valid) != expected_keys:
        raise ValueError(f"label_valid keys must be {sorted(expected_keys)}")
    for name in FIELDS:
        value = getattr(owner, name)
        prefix = (batch, steps, slots) if name == "geometry" else (batch, steps, slots, slots)
        if not value.is_floating_point() or value.ndim != len(prefix) + 1:
            raise ValueError(f"{name} must be floating point with shape {prefix} + [features]")
        if tuple(value.shape[:-1]) != prefix or value.shape[-1] == 0:
            raise ValueError(f"invalid {name} shape")
        if value.device != owner.entity_ids.device:
            raise ValueError("all label tensors must share the scene device")
        valid = owner.label_valid[name]
        _mask(valid, value.shape, value.device, f"label_valid[{name}]")
        observed = value[valid]
        if not torch.isfinite(observed).all():
            raise ValueError(f"valid {name} labels must be finite")
        if name != "geometry" and ((observed != 0) & (observed != 1)).any():
            raise ValueError(f"valid {name} labels must be binary")


def _active_check(active: Tensor | None, has_requirement: Tensor) -> None:
    if active is not None:
        _mask(active, has_requirement.shape, has_requirement.device, "active")
        if (active & ~has_requirement).any():
            raise ValueError("active samples cannot have empty requirements")


def _order(entity_ids: Tensor, order: Tensor) -> Tensor:
    batch, entities = entity_ids.shape
    if order.dtype != torch.int64 or order.device != entity_ids.device:
        raise ValueError("entity permutation must be int64 on the scene device")
    if order.shape == (entities,):
        order = order.unsqueeze(0).expand(batch, -1)
    if order.shape != entity_ids.shape:
        raise ValueError("entity permutation must be [N] or [B, N]")
    expected = torch.arange(entities, device=order.device).expand(batch, -1)
    if not torch.equal(order.sort(dim=1).values, expected):
        raise ValueError("each permutation row must contain every slot exactly once")
    return order


def _gather(value: Tensor, order: Tensor, axis: int) -> Tensor:
    shape = [order.shape[0]] + [1] * (value.ndim - 1)
    shape[axis] = order.shape[1]
    return value.gather(axis, order.reshape(shape).expand_as(value))


@dataclass
class PhysicalOutcome:
    entity_ids: Tensor                     # int64 [B, N], padding=-1
    step_offsets: Tensor                   # int64 [T], action-step offsets
    geometry: Tensor                       # float [B, T, N, Dg]
    relations: Tensor                      # float [B, T, N, N, C], directed
    events: Tensor                         # float [B, T, N, N, E], directed
    label_valid: dict[str, Tensor]         # bool, exactly the field shapes

    def validate(self) -> "PhysicalOutcome":
        batch, entities, steps = _scene(self.entity_ids, self.step_offsets)
        _fields(self, entities, batch, steps)
        present = self.entity_ids >= 0
        for name in FIELDS:
            allowed = present[:, None, :, None]
            if name != "geometry":
                allowed = present[:, None, :, None, None] & present[:, None, None, :, None]
            if (self.label_valid[name] & ~allowed).any():
                raise ValueError(f"padding entities cannot have valid {name} labels")
        return self

    def permute_entities(self, order: Tensor) -> "PhysicalOutcome":
        """Return new slots in ``order[new_slot] = old_slot`` order."""
        self.validate()
        order = _order(self.entity_ids, order)
        values, masks = {}, {}
        for name in FIELDS:
            values[name] = _gather(getattr(self, name), order, 2)
            masks[name] = _gather(self.label_valid[name], order, 2)
            if name != "geometry":
                values[name] = _gather(values[name], order, 3)
                masks[name] = _gather(masks[name], order, 3)
        return replace(self, entity_ids=self.entity_ids.gather(1, order), label_valid=masks, **values)


@dataclass
class EffectRequirement:
    entity_ids: Tensor                     # same scene table as PhysicalOutcome
    step_offsets: Tensor                   # int64 [T]
    binding: Tensor                        # int64 [B, R], slot or UNUSED/UNMATCHED/UNCERTAIN
    geometry: Tensor                       # float [B, T, R, Dg]
    relations: Tensor                      # float [B, T, R, R, C]
    events: Tensor                         # float [B, T, R, R, E]
    requirement_mask: dict[str, Tensor]    # bool, geometry/relations/events
    label_valid: dict[str, Tensor]         # also binding and REQUIREMENT_SEMANTICS
    geometry_tolerance: Tensor | None = None   # float [B,T,R,Dg], interval half-width
    event_windows: Tensor | None = None        # int64 [B,T,R,R,E,2], inclusive action steps
    event_precedence: Tensor | None = None     # int64 [B,P,2], flattened event node indices

    def __post_init__(self) -> None:
        """Legacy Python defaults preserve exact, fixed-time requirements only.

        JSON v2 callers must supply the three semantic fields and their validity
        explicitly. These defaults are not annotations inferred from a video.
        """
        self.label_valid = dict(self.label_valid)
        if self.geometry_tolerance is None:
            self.geometry_tolerance = torch.zeros_like(self.geometry)
            self.label_valid.setdefault("geometry_tolerance", torch.ones_like(self.geometry, dtype=torch.bool))
        if self.event_windows is None and self.events.ndim == 5 and self.step_offsets.ndim == 1:
            if self.events.shape[1] == self.step_offsets.numel():
                times = self.step_offsets.reshape(1, -1, 1, 1, 1, 1)
                self.event_windows = times.expand(*self.events.shape, 2).clone()
                self.label_valid.setdefault("event_windows", torch.ones_like(self.events, dtype=torch.bool))
        if self.event_precedence is None:
            self.event_precedence = self.entity_ids.new_empty((self.entity_ids.shape[0], 0, 2))
            self.label_valid.setdefault("event_precedence", torch.ones_like(self.event_precedence[..., 0], dtype=torch.bool))

    @property
    def has_requirement(self) -> Tensor:
        return torch.stack([self.requirement_mask[name].flatten(1).any(1) for name in FIELDS]).any(0)

    @property
    def required_roles(self) -> Tensor:
        """[B,R] roles participating in any required positive or negative condition."""
        roles = self.requirement_mask["geometry"].any(dim=(1, 3))
        for name in ("relations", "events"):
            mask = self.requirement_mask[name]
            roles = roles | mask.any(dim=(1, 3, 4)) | mask.any(dim=(1, 2, 4))
        return roles

    @property
    def resolved(self) -> Tensor:
        """[B] all necessary role bindings are known and refer to present entities.

        This is not a task-completion or confidence prediction. Call validate()
        first; unmatched/uncertain roles keep their requirements and yield False.
        """
        present = self.entity_ids.gather(1, self.binding.clamp_min(0)) >= 0
        known = (self.binding >= 0) & self.label_valid["binding"] & present
        return (~self.required_roles | known).all(-1)

    @property
    def geometry_bounds(self) -> tuple[Tensor, Tensor]:
        return self.geometry - self.geometry_tolerance, self.geometry + self.geometry_tolerance

    @property
    def semantics_known(self) -> Tensor:
        """[B] data availability for every required value and temporal constraint.

        A padded precedence slot whose validity is False is unknown, not proof
        that no order is required; consequently it prevents complete supervision.
        """
        known = self.resolved
        for name in FIELDS:
            known = known & ~(self.requirement_mask[name] & ~self.label_valid[name]).flatten(1).any(-1)
        for name, source in (("geometry_tolerance", "geometry"), ("event_windows", "events")):
            known = known & ~(self.requirement_mask[source] & ~self.label_valid[name]).flatten(1).any(-1)
        return known & self.label_valid["event_precedence"].all(-1)

    def _validate_semantics(self) -> None:
        tolerance = self.geometry_tolerance
        if (not isinstance(tolerance, Tensor) or not tolerance.is_floating_point()
                or tolerance.shape != self.geometry.shape or tolerance.device != self.geometry.device):
            raise ValueError("geometry_tolerance must match geometry's shape and device")
        valid = self.label_valid["geometry_tolerance"]
        _mask(valid, self.geometry.shape, self.geometry.device, "geometry_tolerance label_valid")
        if not torch.isfinite(tolerance[valid]).all() or (tolerance[valid] < 0).any():
            raise ValueError("known geometry tolerances must be finite and nonnegative")
        bounds_valid = valid & self.label_valid["geometry"]
        if any(not torch.isfinite(bound[bounds_valid]).all() for bound in self.geometry_bounds):
            raise ValueError("known geometry intervals must have finite bounds")

        windows = self.event_windows
        if (not isinstance(windows, Tensor) or windows.dtype != torch.int64
                or windows.shape != (*self.events.shape, 2) or windows.device != self.events.device):
            raise ValueError("event_windows must be int64 [B,T,R,R,E,2] on the event device")
        valid_windows = self.label_valid["event_windows"]
        _mask(valid_windows, self.events.shape, self.events.device, "event_windows label_valid")
        known_windows = windows[valid_windows]
        if (known_windows < 1).any() or (known_windows[:, 0] > known_windows[:, 1]).any():
            raise ValueError("known event windows must be nonempty inclusive positive step intervals")

        edges = self.event_precedence
        batch = self.entity_ids.shape[0]
        if (not isinstance(edges, Tensor) or edges.dtype != torch.int64 or edges.ndim != 3
                or edges.shape[0] != batch or edges.shape[-1] != 2 or edges.device != self.events.device):
            raise ValueError("event_precedence must be int64 [B,P,2] on the event device")
        valid_edges = self.label_valid["event_precedence"]
        _mask(valid_edges, edges.shape[:2], edges.device, "event_precedence label_valid")
        nodes = self.events[0].numel()
        if ((edges < -1) | (edges >= nodes)).any() or ((edges[..., 0] < 0) != (edges[..., 1] < 0)).any():
            raise ValueError("precedence slots must be event-index pairs or (-1,-1) padding")
        for sample in range(batch):
            present = valid_edges[sample] & (edges[sample, :, 0] >= 0)
            pairs = [tuple(pair) for pair in edges[sample, present].tolist()]
            if len(set(pairs)) != len(pairs) or any(first == second for first, second in pairs):
                raise ValueError("known precedence edges cannot be repeated or self-referential")
            graph: dict[int, set[int]] = {}
            needed = self.requirement_mask["events"][sample].flatten()
            positive = self.events[sample].flatten() == 1
            for first, second in pairs:
                if not (needed[first] & needed[second] & positive[first] & positive[second]):
                    raise ValueError("known precedence edges must reference required positive events")
                graph.setdefault(first, set())
                graph.setdefault(second, set()).add(first)
            try:
                ordered = tuple(TopologicalSorter(graph).static_order())
            except CycleError as error:
                raise ValueError("known event precedence must form a DAG") from error
            # Earlier feasible predecessors cannot delay any valid successor;
            # this checks temporal consistency without planning robot actions.
            flat_windows = windows[sample].reshape(nodes, 2)
            flat_valid = valid_windows[sample].flatten()
            earliest: dict[int, int] = {}
            for node in ordered:
                if not flat_valid[node] or any(parent not in earliest for parent in graph[node]):
                    continue
                start, end = flat_windows[node].tolist()
                earliest[node] = max([start] + [earliest[parent] + 1 for parent in graph[node]])
                if earliest[node] > end:
                    raise ValueError("known event windows contradict strict event precedence")

    def validate(self, active: Tensor | None = None) -> "EffectRequirement":
        batch, entities, steps = _scene(self.entity_ids, self.step_offsets)
        if self.binding.dtype != torch.int64 or self.binding.ndim != 2 or self.binding.shape[0] != batch:
            raise ValueError("binding must be int64 [B, R]")
        if self.binding.device != self.entity_ids.device or not self.binding.shape[1]:
            raise ValueError("binding must have roles and share the scene device")
        if ((self.binding < BINDING_UNCERTAIN) | (self.binding >= entities)).any():
            raise ValueError("binding must reference a scene slot or UNUSED/UNMATCHED/UNCERTAIN")
        roles = self.binding.shape[1]
        _fields(self, roles, batch, steps)
        _mask(self.label_valid["binding"], self.binding.shape, self.binding.device, "binding label_valid")
        bound = self.binding >= 0
        present = self.entity_ids.gather(1, self.binding.clamp_min(0)) >= 0
        if (bound & ~present).any():
            raise ValueError("bindings cannot reference padding entities")
        if set(self.requirement_mask) != set(FIELDS):
            raise ValueError(f"requirement_mask keys must be {FIELDS}")
        for name in FIELDS:
            mask = self.requirement_mask[name]
            _mask(mask, getattr(self, name).shape, self.binding.device, f"requirement_mask[{name}]")
        if (self.required_roles & (self.binding == BINDING_UNUSED)).any():
            raise ValueError("UNUSED roles cannot participate in required conditions")
        self._validate_semantics()
        # Requirement existence and annotation availability are deliberately independent.
        _active_check(active, self.has_requirement)
        return self

    def permute_entities(self, order: Tensor) -> "EffectRequirement":
        """Permute scene slots and inverse-map bindings; role tensors stay fixed."""
        self.validate()
        order = _order(self.entity_ids, order)
        inverse = order.argsort(dim=1)
        mapped = inverse.gather(1, self.binding.clamp_min(0))
        binding = torch.where(self.binding >= 0, mapped, self.binding)
        return replace(self, entity_ids=self.entity_ids.gather(1, order), binding=binding)


@dataclass
class TaskRequirement:
    current: EffectRequirement
    remaining: EffectRequirement

    @property
    def resolved(self) -> Tensor:
        return self.current.resolved & self.remaining.resolved

    @property
    def semantics_known(self) -> Tensor:
        return self.current.semantics_known & self.remaining.semantics_known

    def validate(self, active: Tensor | None = None) -> "TaskRequirement":
        self.current.validate()
        self.remaining.validate()
        if self.current.entity_ids.device != self.remaining.entity_ids.device or not torch.equal(self.current.entity_ids, self.remaining.entity_ids):
            raise ValueError("current and remaining must share one scene entity table and device")
        _active_check(active, self.current.has_requirement | self.remaining.has_requirement)
        return self

    def permute_entities(self, order: Tensor) -> "TaskRequirement":
        self.validate()
        return TaskRequirement(self.current.permute_entities(order), self.remaining.permute_entities(order))
