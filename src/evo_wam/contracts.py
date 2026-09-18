"""Task requirements and task-independent, scene-indexed outcome labels.

Geometry uses the caller's fixed robot frame and the current time as its origin.
Relations are sampled at ``step_offsets``; events describe the intervals ending
at those offsets. Event absence is valid only for fully observed intervals.
"""

from dataclasses import dataclass, replace

import torch
from torch import Tensor


FIELDS = ("geometry", "relations", "events")


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
    expected_keys = set(FIELDS) | ({"binding"} if hasattr(owner, "binding") else set())
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
    binding: Tensor                        # int64 [B, R], scene slot or -1
    geometry: Tensor                       # float [B, T, R, Dg]
    relations: Tensor                      # float [B, T, R, R, C]
    events: Tensor                         # float [B, T, R, R, E]
    requirement_mask: dict[str, Tensor]    # bool, geometry/relations/events
    label_valid: dict[str, Tensor]         # also includes binding [B, R]

    @property
    def has_requirement(self) -> Tensor:
        return torch.stack([self.requirement_mask[name].flatten(1).any(1) for name in FIELDS]).any(0)

    def validate(self, active: Tensor | None = None) -> "EffectRequirement":
        batch, entities, steps = _scene(self.entity_ids, self.step_offsets)
        if self.binding.dtype != torch.int64 or self.binding.ndim != 2 or self.binding.shape[0] != batch:
            raise ValueError("binding must be int64 [B, R]")
        if self.binding.device != self.entity_ids.device or not self.binding.shape[1]:
            raise ValueError("binding must have roles and share the scene device")
        if ((self.binding < -1) | (self.binding >= entities)).any():
            raise ValueError("binding must reference an existing slot or use -1")
        roles = self.binding.shape[1]
        _fields(self, roles, batch, steps)
        _mask(self.label_valid["binding"], self.binding.shape, self.binding.device, "binding label_valid")
        bound = self.binding >= 0
        present = self.entity_ids.gather(1, self.binding.clamp_min(0)) >= 0
        if (bound & ~present).any() or (self.label_valid["binding"] & ~bound).any():
            raise ValueError("bindings cannot reference padding; valid binding labels must be bound")
        if set(self.requirement_mask) != set(FIELDS):
            raise ValueError(f"requirement_mask keys must be {FIELDS}")
        used_roles = torch.zeros_like(bound)
        for name in FIELDS:
            mask = self.requirement_mask[name]
            _mask(mask, getattr(self, name).shape, self.binding.device, f"requirement_mask[{name}]")
            if name == "geometry":
                used_roles |= mask.any(dim=(1, 3))
            else:
                used_roles |= mask.any(dim=(1, 3, 4)) | mask.any(dim=(1, 2, 4))
        if (used_roles & ~bound).any():
            raise ValueError("required roles must be bound to present scene entities")
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
