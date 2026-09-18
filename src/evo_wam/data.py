"""Validated paired data, shared denoising inputs, and leakage-free splits.

Identifiers stay in metadata. Only explicit tensor fields reach model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .contracts import EffectRequirement, FIELDS, PhysicalOutcome, TaskRequirement
from .losses import paired_dropout_disabled


@dataclass(frozen=True)
class DenoisingInput:
    clean: Tensor
    noisy: Tensor
    noise: Tensor
    tau: Tensor
    query_step: int

    @property
    def flow_target(self) -> Tensor:
        return self.noise - self.clean


def shared_denoising_inputs(
    targets: Mapping[str, Tensor],
    sigma_tables: Mapping[str, Tensor],
    query_steps: Mapping[str, int],
    *,
    generator: torch.Generator,
    time_dim: int | None = None,
) -> dict[str, DenoisingInput]:
    """Sample each target separately; both view branches reuse these objects.

    Pass sigma tables from the pinned upstream scheduler, preserving its shift
    and discrete time distribution. ``time_dim=2`` supports per-frame video
    times; ``None`` draws one time per batch item. No scheduler is reimplemented.
    """
    if not targets or targets.keys() != sigma_tables.keys() or targets.keys() != query_steps.keys():
        raise ValueError("targets, sigma_tables, and query_steps must have identical nonempty keys")
    result = {}
    batch_size = None
    for name, clean in targets.items():
        if clean.ndim < 2 or not clean.is_floating_point() or not torch.isfinite(clean).all():
            raise ValueError(f"{name}: expected finite floating batched target")
        batch_size = clean.shape[0] if batch_size is None else batch_size
        if clean.shape[0] != batch_size or batch_size == 0:
            raise ValueError("all targets must share a nonempty batch")
        table = sigma_tables[name]
        if table.ndim != 1 or not table.numel() or not torch.isfinite(table).all() or not ((table >= 0) & (table <= 1)).all():
            raise ValueError(f"{name}: sigma table must contain finite values in [0, 1]")
        if type(query_steps[name]) is not int or query_steps[name] < 1:
            raise ValueError("query steps must be positive integers")
        draw_shape = (batch_size,)
        broadcast_shape = [batch_size] + [1] * (clean.ndim - 1)
        if time_dim is not None:
            if not 1 <= time_dim < clean.ndim:
                raise ValueError("time_dim must be a non-batch target dimension")
            draw_shape = (batch_size, clean.shape[time_dim])
            broadcast_shape[time_dim] = clean.shape[time_dim]
        # Draw with the generator's device, then transfer once to the target.
        indices = torch.randint(table.numel(), draw_shape, generator=generator, device=generator.device)
        tau = table.to(generator.device)[indices].to(clean.device, torch.float32)
        noise = torch.randn(clean.shape, generator=generator, device=generator.device, dtype=clean.dtype).to(clean.device)
        weights = tau.reshape(broadcast_shape).to(clean.dtype)
        noisy = (1 - weights) * clean + weights * noise
        result[name] = DenoisingInput(clean, noisy, noise, tau, query_steps[name])
    return result


@dataclass(frozen=True)
class TaskCondition:
    demonstration: Tensor | None = None
    current: Tensor | None = None
    remaining: Tensor | None = None
    text: str = ""
    task_cache: object | None = None


@dataclass(frozen=True)
class BranchInput:
    robot_history: Tensor
    denoising: Mapping[str, DenoisingInput]
    condition: TaskCondition


@dataclass(frozen=True)
class PairedTrainingInput:
    branches: tuple[BranchInput, ...]
    conditional: bool
    loss_enabled: Mapping[str, bool]


def prepare_training_input(
    robot_history: Tensor,
    target_frames: Mapping[str, Tensor],
    conditions: tuple[TaskCondition, TaskCondition],
    sigma_tables: Mapping[str, Tensor],
    query_steps: Mapping[str, int],
    *,
    history_encoder: Callable[[Tensor], Tensor],
    target_encoder: Callable[[Tensor], Tensor],
    generator: torch.Generator,
    unconditional_probability: float = 0.1,
    enable_ifp: bool = True,
    enable_interaction: bool = True,
    enable_cv: bool = True,
    time_dim: int | None = None,
) -> PairedTrainingInput:
    """Encode robot inputs once and produce either two views or one null branch.

    The condition readers are called by the training loop only for conditional
    samples. A null branch deliberately contains no stored true requirement.
    """
    if not 0 <= unconditional_probability <= 1:
        raise ValueError("unconditional_probability must be in [0, 1]")
    if len(conditions) != 2:
        raise ValueError("a conditional pair requires exactly two views")
    if any((condition.current is None) != (condition.remaining is None) for condition in conditions):
        raise ValueError("current and remaining task tokens must both be present or both await the reader")
    if any(condition.text for condition in conditions):
        raise ValueError("main ICL experiments require empty task text")
    conditional = bool(torch.rand((), generator=generator, device=generator.device) >= unconditional_probability)
    if conditional and any(condition.demonstration is None for condition in conditions):
        raise ValueError("both conditional branches must retain demonstration evidence")
    history = history_encoder(robot_history)
    targets = {name: target_encoder(frames) for name, frames in target_frames.items()}
    denoising = shared_denoising_inputs(targets, sigma_tables, query_steps, generator=generator, time_dim=time_dim)
    selected = conditions if conditional else (TaskCondition(),)
    branches = tuple(BranchInput(history, denoising, condition) for condition in selected)
    enabled = {
        "next_video": True,
        "ifp": enable_ifp,
        "requirement": conditional,
        "execution": conditional,
        "interaction": conditional and enable_interaction,
        "cv": conditional and enable_cv,
    }
    return PairedTrainingInput(branches, conditional, enabled)



def executed_prefix_valid(
    label_valid: Tensor, executed_steps: int, *, time_dim: int = 0, step_offsets: Tensor | None = None
) -> Tensor:
    """Mask unexecuted outcomes, never assigning continuation labels to a plan."""
    if label_valid.dtype != torch.bool:
        raise ValueError("label_valid must be Boolean")
    if not 0 <= time_dim < label_valid.ndim:
        raise ValueError("invalid time dimension")
    horizon = label_valid.shape[time_dim]
    if step_offsets is None:
        step_offsets = torch.arange(1, horizon + 1, device=label_valid.device)
    if step_offsets.dtype != torch.int64 or step_offsets.shape != (horizon,):
        raise ValueError("step_offsets must be int64 with one offset per label time")
    if (step_offsets <= 0).any() or (step_offsets[1:] <= step_offsets[:-1]).any():
        raise ValueError("step_offsets must be positive and strictly increasing")
    if type(executed_steps) is not int or executed_steps < 0:
        raise ValueError("executed_steps must be a nonnegative integer")
    shape = [1] * label_valid.ndim
    shape[time_dim] = horizon
    prefix = step_offsets.to(label_valid.device).reshape(shape) <= executed_steps
    return label_valid & prefix


@dataclass(frozen=True)
class LoadedSample:
    metadata: Mapping
    robot_history: Tensor
    proprio_history: Tensor
    embodiment: Tensor
    entity_patch_weights: Tensor
    robot_latent: Tensor
    actions: Tensor
    demonstrations: tuple[Tensor, Tensor]
    outcome: PhysicalOutcome
    requirement: TaskRequirement
    common_valid: Mapping[str, Tensor]
    native_inputs: Mapping


def load_sample(manifest_path: str | Path) -> LoadedSample:
    """Read one audited JSON + NPZ sample on CPU, never pickle or object arrays.

    Arrays omit the batch dimension on disk. Metadata, file names, task IDs,
    source identifiers, and free text are not included in model-facing tensors.
    Unknown NPZ keys fail closed to catch misspelled labels or accidental inputs.
    """
    path = Path(manifest_path)
    with path.open(encoding="utf-8") as stream:
        meta = json.load(stream)
    if not isinstance(meta, dict) or type(meta.get("format_version")) is not int or meta["format_version"] != 1:
        raise ValueError("manifest must be a version-1 JSON object")
    for name in ("source_id", "trajectory_id", "history_id", "coordinate_frame", "task_annotation"):
        if not isinstance(meta.get(name), str) or not meta[name]:
            raise ValueError(f"manifest requires nonempty {name}")
    for name in ("window_start", "executed_steps"):
        if type(meta.get(name)) is not int or meta[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not isinstance(meta.get("control_dt"), (float, int)) or isinstance(meta["control_dt"], bool) or not np.isfinite(meta["control_dt"]) or meta["control_dt"] <= 0:
        raise ValueError("control_dt must be a finite positive number")
    if not isinstance(meta.get("view_ids"), list) or len(meta["view_ids"]) != 2 or any(not isinstance(v, str) or not v for v in meta["view_ids"]):
        raise ValueError("view_ids must identify exactly two synchronized views")
    array_name = meta.get("arrays")
    if not isinstance(array_name, str):
        raise ValueError("arrays must name a relative NPZ file")
    array_path = (path.parent / array_name).resolve()
    if Path(array_name).is_absolute() or not array_path.is_relative_to(path.parent.resolve()) or array_path.suffix != ".npz":
        raise ValueError("array file must be an NPZ within the manifest directory")
    required = {"entity_ids", "step_offsets", "robot_history", "proprio_history", "embodiment", "entity_patch_weights", "robot_latent", "actions", "demo_view_0", "demo_view_1"}
    for field in FIELDS:
        required |= {f"outcome_{field}", f"outcome_{field}_valid"}
        for part in ("current", "remaining"):
            required |= {f"{part}_{field}", f"{part}_{field}_valid", f"{part}_{field}_required"}
    for part in ("current", "remaining"):
        required |= {f"{part}_binding", f"{part}_binding_valid", f"{part}_step_offsets"}
    # Visibility/evidence validity is data-owned and separate for each view.
    cv_fields = ("current_binding", "remaining_binding", "relations", "events")
    required |= {f"view{view}_{field}_valid" for view in (0, 1) for field in cv_fields}
    native_map = meta.get("native_arrays", {})
    if not isinstance(native_map, dict) or set(native_map) - {"latent_dict", "action_dict", "mcp_latent_dicts"}:
        raise ValueError("native_arrays accepts only latent/action/MCP streams, never task conditions")
    native_fields = {"noisy_latents", "latent", "timesteps", "cond_timesteps", "grid_id", "targets", "valid_mask", "actions_mask"}
    native_streams = []
    for name, streams in native_map.items():
        if name == "mcp_latent_dicts":
            if not isinstance(streams, list):
                raise ValueError("mcp_latent_dicts must be a list of stream maps")
        else:
            streams = [streams]
        for stream in streams:
            if not isinstance(stream, dict) or not stream or set(stream) - native_fields:
                raise ValueError("invalid native stream fields")
            if any(not isinstance(v, str) or not v.startswith("native_") for v in stream.values()):
                raise ValueError("native stream arrays must have the native_ prefix")
            required.update(stream.values())
            native_streams.append(stream)
    with np.load(array_path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError(f"NPZ keys mismatch; missing={sorted(required - set(archive.files))}, extra={sorted(set(archive.files) - required)}")
        arrays = {}
        for name in required:
            array = archive[name]
            if array.dtype.kind not in "bifu" or array.dtype.hasobject:
                raise ValueError(f"{name}: only numeric and Boolean arrays are supported")
            arrays[name] = torch.from_numpy(array.copy())

    entity_ids = arrays["entity_ids"].unsqueeze(0)
    offsets = arrays["step_offsets"]
    values = {field: arrays[f"outcome_{field}"].unsqueeze(0) for field in FIELDS}
    validity = {field: arrays[f"outcome_{field}_valid"].unsqueeze(0) for field in FIELDS}
    outcome = PhysicalOutcome(entity_ids, offsets, label_valid=validity, **values).validate()
    actions = arrays["actions"]
    if actions.ndim != 2 or not actions.shape[0] or not actions.shape[1] or not actions.is_floating_point() or not torch.isfinite(actions).all():
        raise ValueError("actions must be finite floating [planned_steps, action_dim]")
    if meta["executed_steps"] > actions.shape[0] or offsets[-1] > actions.shape[0]:
        raise ValueError("executed steps and outcome offsets cannot exceed the planned action window")
    outcome.label_valid = {
        field: executed_prefix_valid(mask, meta["executed_steps"], time_dim=1, step_offsets=offsets)
        for field, mask in validity.items()
    }

    parts = {}
    for part in ("current", "remaining"):
        parts[part] = EffectRequirement(
            entity_ids, arrays[f"{part}_step_offsets"], arrays[f"{part}_binding"].unsqueeze(0),
            **{field: arrays[f"{part}_{field}"].unsqueeze(0) for field in FIELDS},
            requirement_mask={field: arrays[f"{part}_{field}_required"].unsqueeze(0) for field in FIELDS},
            label_valid={field: arrays[f"{part}_{field}_valid"].unsqueeze(0) for field in (*FIELDS, "binding")},
        ).validate()
    requirement = TaskRequirement(**parts).validate()
    expected_cv_shapes = {
        "current_binding": parts["current"].binding.shape,
        "remaining_binding": parts["remaining"].binding.shape,
        "relations": outcome.relations.shape,
        "events": outcome.events.shape,
    }
    common_valid = {}
    for field, shape in expected_cv_shapes.items():
        masks = [arrays[f"view{view}_{field}_valid"].unsqueeze(0) for view in (0, 1)]
        if any(mask.dtype != torch.bool or mask.shape != shape for mask in masks):
            raise ValueError(f"{field}: per-view evidence masks must be bool with shape {shape}")
        base = parts[field.split("_")[0]].label_valid["binding"] if field.endswith("_binding") else outcome.label_valid[field]
        common_valid[field] = masks[0] & masks[1] & base
    for name in ("robot_history", "robot_latent", "demo_view_0", "demo_view_1"):
        tensor = arrays[name]
        if tensor.ndim < 2 or not tensor.numel() or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"{name}: expected nonempty finite floating array with at least two axes")
    history = arrays["robot_history"]
    if history.ndim != 3 or history.shape[1] != entity_ids.shape[1]:
        raise ValueError("robot_history must be [history_steps, scene_entities, entity_features]")
    proprio = arrays["proprio_history"]
    embodiment = arrays["embodiment"]
    patches = arrays["entity_patch_weights"]
    for name in ("proprio_history", "embodiment", "entity_patch_weights"):
        value = arrays[name]
        if not value.numel() or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be nonempty finite floating data")
    if proprio.ndim != 2 or proprio.shape[0] != history.shape[0]:
        raise ValueError("proprio_history must be [history_steps, proprio_features]")
    if embodiment.ndim != 1:
        raise ValueError("embodiment must be [embodiment_features]")
    if patches.ndim != 2 or patches.shape[0] != entity_ids.shape[1] or (patches < 0).any():
        raise ValueError("entity_patch_weights must be nonnegative [scene_entities, video_patches]")
    if (patches[entity_ids[0] < 0] != 0).any() or (patches[entity_ids[0] >= 0].sum(-1) <= 0).any():
        raise ValueError("present entities need observation patch mass; padded entities need zero mass")
    native_inputs = {}
    for name, streams in native_map.items():
        if name == "mcp_latent_dicts":
            native_inputs[name] = [{key: arrays[value] for key, value in stream.items()} for stream in streams]
        else:
            native_inputs[name] = {key: arrays[value] for key, value in streams.items()}
    for stream in native_streams:
        if any(not torch.isfinite(arrays[name]).all() for name in stream.values()):
            raise ValueError("native input arrays must be finite")
    scalars = meta.get("native_scalars", {})
    if not isinstance(scalars, dict) or set(scalars) - {"chunk_size", "max_frame_chunk_size", "window_size"}:
        raise ValueError("invalid native scalar keys")
    if any(type(value) is not int or value < 0 for value in scalars.values()):
        raise ValueError("native chunk/window scalars must be nonnegative integers")
    native_inputs.update(scalars)
    return LoadedSample(
        meta, history.unsqueeze(0), proprio.unsqueeze(0), embodiment.unsqueeze(0), patches.unsqueeze(0),
        arrays["robot_latent"].unsqueeze(0), actions.unsqueeze(0),
        (arrays["demo_view_0"].unsqueeze(0), arrays["demo_view_1"].unsqueeze(0)), outcome, requirement, common_valid, native_inputs,
    )


def connected_components(records: Sequence[Mapping]) -> list[tuple[int, ...]]:
    """Join original human sources and robot trajectories, including transitive links."""
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    keys = []
    for record in records:
        for field in ("source_id", "trajectory_id"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise ValueError(f"nonempty {field} required for split grouping")
        source = ("human", record["source_id"])
        robot = ("robot", record["trajectory_id"])
        parent[find(robot)] = find(source)
        keys.append(source)
    groups: dict[tuple[str, str], list[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(find(key), []).append(i)
    return [tuple(group) for group in groups.values()]


def assign_splits(
    records: Sequence[Mapping], *, seed: int = 0, fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> list[str]:
    """Deterministic component-level split; assignment ignores tasks and labels.

    Fractions apply to components in expectation, not samples. Inspect achieved
    counts; a giant connected component cannot be subdivided without leakage.
    """
    if len(fractions) != 3 or any(not np.isfinite(v) or v < 0 for v in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError("three nonnegative split fractions must sum to one")
    result = [""] * len(records)
    for group in connected_components(records):
        nodes = sorted({(kind, records[i][field]) for i in group for kind, field in (("human", "source_id"), ("robot", "trajectory_id"))})
        key = json.dumps([seed, nodes], separators=(",", ":"), ensure_ascii=False).encode()
        draw = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
        split = "train" if draw < fractions[0] else "validation" if draw < sum(fractions[:2]) else "test"
        for i in group:
            result[i] = split
    return result


def validate_splits(records: Sequence[Mapping]) -> None:
    for record in records:
        if record.get("split") not in {"train", "validation", "test"}:
            raise ValueError("every record must have train, validation, or test split")
    for group in connected_components(records):
        if len({records[i]["split"] for i in group}) != 1:
            raise ValueError("human source / robot trajectory component crosses splits")


def load_experiment(path: str | Path, *, for_test: bool = False) -> dict:
    """Load an experiment without silently relaxing its frozen comparison rules."""
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    from .zerowam import ZERO_WAM_COMMIT
    if config.get("upstream_commit") != ZERO_WAM_COMMIT:
        raise ValueError("experiment must use the pinned Zero-WAM source revision")
    if config.get("schema_version") != 1 or config.get("interface") not in {"geometry", "full"}:
        raise ValueError("unsupported experiment schema or interface")
    condition = config["conditioning"]
    if condition["unconditional_probability"] != 0.1 or condition["drop_icl"] != 0 or condition["droptext_target"] != 0 or condition["task_text"] != "" or condition["pair_dropout"] != 0:
        raise ValueError("experiment violates the paired condition-drop protocol")
    if config["pair_supervision_reduction"] != "mean" or config["candidate_count"] != 1 or config["f_ranking"]:
        raise ValueError("representation experiments require averaged pairs and one unsorted candidate")
    if not np.isfinite(config["lambda_cv"]) or config["lambda_cv"] < 0:
        raise ValueError("lambda_cv must be finite and nonnegative")
    for part in ("current", "remaining"):
        offsets = config[f"{part}_offsets"]
        if not offsets or any(type(v) is not int or v <= 0 for v in offsets) or offsets != sorted(set(offsets)):
            raise ValueError("requirement offsets must be positive and strictly increasing")
        if config["tokens"][part] != config["dimensions"]["roles"] * len(offsets):
            raise ValueError("token budget must equal role count times query times")
    if for_test and not config.get("validation_locked", False):
        raise ValueError("lock the common configuration on validation data before test evaluation")
    return config


@dataclass(frozen=True)
class Observation:
    """Deployment inputs: no future outcomes, true requirements or actions."""
    entity_ids: Tensor
    robot_history: Tensor
    proprio_history: Tensor
    embodiment: Tensor
    robot_latent: Tensor
    demonstrations: tuple[Tensor, ...]
    chunk_size: int
    actions_per_frame: int
    action_space: dict


def load_observation(manifest_path: str | Path) -> Observation:
    path = Path(manifest_path)
    meta = json.loads(path.read_text())
    if meta.get("format_version") != 1 or meta.get("kind") != "observation":
        raise ValueError("inference requires a version-1 observation manifest, not a labeled training sample")
    for key in ("chunk_size", "actions_per_frame"):
        if type(meta.get(key)) is not int or meta[key] < 1:
            raise ValueError(f"observation requires positive {key}")
    views = meta.get("view_ids", [])
    if not isinstance(views, list) or not views or any(not isinstance(v, str) or not v for v in views):
        raise ValueError("observation must identify its demonstration views")
    name = meta.get("arrays")
    if not isinstance(name, str):
        raise ValueError("observation arrays must name a relative NPZ")
    arrays_path = (path.parent / name).resolve()
    if Path(name).is_absolute() or not arrays_path.is_relative_to(path.parent.resolve()) or arrays_path.suffix != ".npz":
        raise ValueError("observation arrays must remain inside the manifest directory")
    required = {"entity_ids", "robot_history", "proprio_history", "embodiment", "robot_latent"}
    required |= {f"demo_view_{i}" for i in range(len(views))}
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if set(arrays.files) != required:
            raise ValueError("observation NPZ must contain only current/history and demonstration fields")
        values = {key: torch.from_numpy(arrays[key].copy()) for key in required}
    ids = values["entity_ids"]
    if ids.dtype != torch.int64 or ids.ndim != 1 or not (ids >= 0).any() or (ids < -1).any() or ids[ids >= 0].unique().numel() != (ids >= 0).sum():
        raise ValueError("observation entity IDs must be unique nonnegative integers or -1 padding")
    for key, value in values.items():
        if key != "entity_ids" and (not value.is_floating_point() or not torch.isfinite(value).all() or not value.numel()):
            raise ValueError("observation features must be nonempty finite floating tensors")
    history, proprio = values["robot_history"], values["proprio_history"]
    if history.ndim != 3 or history.shape[1] != len(ids) or proprio.ndim != 2 or proprio.shape[0] != history.shape[0]:
        raise ValueError("observation entity/history/state axes differ")
    if values["embodiment"].ndim != 1 or values["robot_latent"].ndim != 4:
        raise ValueError("embodiment must be [E], observed video latent [C,F,H,W]")
    if any(values[f"demo_view_{i}"].ndim != 2 for i in range(len(views))):
        raise ValueError("demonstrations must be ordered [tokens,features]")
    return Observation(ids[None], history[None], proprio[None], values["embodiment"][None],
                       values["robot_latent"][None], tuple(values[f"demo_view_{i}"][None] for i in range(len(views))),
                       meta["chunk_size"], meta["actions_per_frame"], meta.get("action_space", {}))
