"""Validated paired data, shared denoising inputs, and leakage-free splits.

Identifiers stay in metadata. Only explicit tensor fields reach model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .contracts import EffectRequirement, FIELDS, REQUIREMENT_SEMANTICS, PhysicalOutcome, TaskRequirement
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


def _pair_kind(kind: str, views: int) -> str:
    if views not in (1, 2):
        raise ValueError("robot-supervised examples require one or two demonstration views")
    if kind not in ("none", "synchronized_views"):
        raise ValueError("pair_kind must be none or synchronized_views")
    if kind == "synchronized_views" and views != 2:
        raise ValueError("synchronized_views requires two explicitly recorded views")
    return kind


def validate_demo_encoding(meta: Mapping, width: int) -> dict:
    """Identify raw features versus learned effect tokens without guessing by shape."""
    if type(width) is not int or width < 1:
        raise ValueError("demonstration feature width must be a positive integer")
    encoding = meta.get("demonstration_encoding", {"kind": "raw_features"})
    if not isinstance(encoding, dict):
        raise ValueError("demonstration_encoding must be an explicit encoding object")
    if encoding.get("kind") == "raw_features":
        if set(encoding) != {"kind"}:
            raise ValueError("raw_features encoding accepts only its kind")
        return {"kind": "raw_features"}
    if encoding.get("kind") == "video_effect_tokens" and (
            type(encoding.get("encoder_version")) is not int or encoding["encoder_version"] != 2):
        raise ValueError("video_effect_tokens require encoder_version 2; re-pretrain and re-export legacy tokens")
    fields = {"kind", "encoder_version", "encoder_sha256", "feature_space_id", "token_dim", "window_frames", "num_tokens"}
    if encoding.get("kind") != "video_effect_tokens" or set(encoding) != fields:
        raise ValueError("video_effect_tokens requires exact encoder identity, feature space and token configuration")
    digest = encoding["encoder_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
        raise ValueError("encoder_sha256 must be a 64-character hexadecimal digest")
    feature_space = encoding["feature_space_id"]
    if not isinstance(feature_space, str) or not feature_space.strip():
        raise ValueError("encoded demonstrations require a nonempty feature_space_id")
    if type(encoding["token_dim"]) is not int or encoding["token_dim"] != width:
        raise ValueError("encoded token_dim must match the actual demonstration feature width")
    if type(encoding["window_frames"]) is not int or encoding["window_frames"] < 3:
        raise ValueError("effect-token windows must contain at least three frames")
    if type(encoding["num_tokens"]) is not int or encoding["num_tokens"] < 1:
        raise ValueError("effect-token num_tokens must be a positive integer")
    return {**encoding, "encoder_sha256": digest.lower()}


def prepare_training_input(
    robot_history: Tensor,
    target_frames: Mapping[str, Tensor],
    conditions: tuple[TaskCondition, ...],
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
    pair_kind: str = "none",
) -> PairedTrainingInput:
    """Encode robot inputs once for one/two demonstrations or one null branch.

    The condition readers are called by the training loop only for conditional
    samples. CV needs an explicit synchronized pair; a single view is never
    duplicated. A null branch deliberately contains no stored true requirement.
    """
    if not 0 <= unconditional_probability <= 1:
        raise ValueError("unconditional_probability must be in [0, 1]")
    pair_kind = _pair_kind(pair_kind, len(conditions))
    if any((condition.current is None) != (condition.remaining is None) for condition in conditions):
        raise ValueError("current and remaining task tokens must both be present or both await the reader")
    if any(condition.text for condition in conditions):
        raise ValueError("main ICL experiments require empty task text")
    conditional = bool(torch.rand((), generator=generator, device=generator.device) >= unconditional_probability)
    if conditional and any(condition.demonstration is None for condition in conditions):
        raise ValueError("every conditional branch must retain demonstration evidence")
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
        "cv": conditional and enable_cv and pair_kind == "synchronized_views",
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
class ObservedActionHistory:
    """Only commands confirmed executed before the current observation."""
    commands: Tensor                         # [1,K,A], K may explicitly be zero
    step_offsets: Tensor                     # int64 [K], command end offsets <= 0
    action_space: Mapping
    observation_step: int
    control_dt: float


def _action_space(space, dimension: int) -> dict:
    if (not isinstance(space, dict) or space.get("representation") != "zero-wam-normalized"
            or not isinstance(space.get("normalization_id"), str) or not space["normalization_id"]
            or type(space.get("dimension")) is not int or space["dimension"] != dimension):
        raise ValueError("observed action space requires normalized format, normalization ID and matching dimension")
    valid = space.get("valid_channels")
    if not isinstance(valid, list) or len(valid) != dimension or any(type(value) is not bool for value in valid) or not any(valid):
        raise ValueError("action valid_channels must explicitly identify every channel")
    return space


def _past_offsets(value: Tensor, count: int, name: str, observation_step: int) -> None:
    if value.dtype != torch.int64 or value.shape != (count,):
        raise ValueError(f"{name} must be int64 with one end offset per observed item")
    if (value > 0).any() or (value < -observation_step).any() or (value[1:] <= value[:-1]).any():
        raise ValueError(f"{name} must be increasing executed/observed offsets, never future or before episode start")


def _history_metadata(meta: Mapping, values: Mapping[str, Tensor]):
    step = meta.get("observation_step")
    if type(step) is not int or step < 0:
        raise ValueError("observation_step must identify the current absolute control step")
    dt = meta.get("control_dt")
    if type(dt) not in (int, float) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("control_dt must be finite and positive")
    per_frame = meta.get("actions_per_frame")
    if type(per_frame) is not int or per_frame < 1:
        raise ValueError("actions_per_frame must be a positive integer")
    commands = values["observed_action_history"]
    if commands.ndim != 2 or not commands.shape[1] or not commands.is_floating_point() or not torch.isfinite(commands).all():
        raise ValueError("observed_action_history must be finite floating [executed_steps,A]")
    space = _action_space(meta.get("observed_action_space"), commands.shape[1])
    if space != _action_space(meta.get("action_space"), commands.shape[1]):
        raise ValueError("observed history and current action space / normalization must agree")
    active = torch.tensor(space["valid_channels"], device=commands.device)
    if (commands[:, ~active] != 0).any():
        raise ValueError("unused observed action channels must be zero")
    offsets = values["observed_action_step_offsets"]
    video = values["robot_latent"]
    if video.ndim != 4 or not video.numel() or not video.is_floating_point() or not torch.isfinite(video).all():
        raise ValueError("observed robot_latent must be finite [C,F,H,W]")
    video_offsets = values["observed_video_step_offsets"]
    _past_offsets(offsets, len(commands), "observed_action_step_offsets", step)
    _past_offsets(video_offsets, video.shape[1], "observed_video_step_offsets", step)
    descriptors = meta.get("history_chunks")
    if not isinstance(descriptors, list) or not descriptors:
        raise ValueError("history_chunks must explicitly describe the observed streams")
    coverage = {"video": 0, "action": 0}
    lengths = {"video": video.shape[1], "action": len(commands)}
    previous_id = -1
    rope_ends = {"video": 0, "action": 0}
    chunk_rope = {}
    for chunk in descriptors:
        if not isinstance(chunk, dict) or set(chunk) != {"mode", "slice", "frame_id", "rope_offset"}:
            raise ValueError("history descriptors must contain only mode/slice/frame_id/rope_offset")
        mode = chunk["mode"]
        bounds = chunk["slice"]
        if mode not in coverage or not isinstance(bounds, list) or len(bounds) != 2 or any(type(v) is not int for v in bounds):
            raise ValueError("invalid observed history mode or slice")
        start, end = bounds
        if start != coverage[mode] or not start < end <= lengths[mode]:
            raise ValueError("history slices must cover each stored observed item once, in order")
        frame_id, rope = chunk["frame_id"], chunk["rope_offset"]
        if type(frame_id) is not int or frame_id <= previous_id or frame_id % 2 != int(mode == "action"):
            raise ValueError("history frame IDs must increase, with video even and action odd")
        if type(rope) is not int or rope < rope_ends[mode]:
            raise ValueError("history RoPE offsets must be nonnegative and nonoverlapping within each stream")
        chunk_index = frame_id // 2
        if chunk_index in chunk_rope and chunk_rope[chunk_index] != rope:
            raise ValueError("aligned video/action chunks must share a RoPE origin")
        chunk_rope[chunk_index] = rope
        count = end - start
        rope_ends[mode] = rope + (count if mode == "video" else (count + per_frame - 1) // per_frame)
        previous_id, coverage[mode] = frame_id, end
    if coverage != lengths:
        raise ValueError("history descriptors must include all stored observed video and executed commands")
    observed = ObservedActionHistory(commands[None], offsets, dict(space), step, float(dt))
    return observed, video_offsets, tuple(dict(chunk) for chunk in descriptors), per_frame


def _validate_sample_history(sample) -> None:
    """The in-memory provider path must enforce the same rules as file loading."""
    history = sample.observed_action_history
    video = sample.robot_latent
    if video.ndim != 5 or video.shape[0] != 1:
        raise ValueError("observed robot video must have native batch size one")
    if history.commands.ndim != 3 or history.commands.shape[0] != 1:
        raise ValueError("observed commands must have explicit batch size one")
    if hasattr(sample, "metadata"):
        space = sample.metadata.get("action_space")
        if sample.metadata.get("observation_step") != history.observation_step:
            raise ValueError("history timestamps differ from the training observation")
    else:
        space = sample.action_space
    meta = {"observation_step": history.observation_step, "control_dt": history.control_dt,
            "actions_per_frame": sample.actions_per_frame, "action_space": space,
            "observed_action_space": history.action_space, "history_chunks": list(sample.history_chunks)}
    _history_metadata(meta, {"robot_latent": video[0], "observed_video_step_offsets": sample.observed_video_step_offsets,
                             "observed_action_history": history.commands[0], "observed_action_step_offsets": history.step_offsets})


def native_history(sample, dtype: torch.dtype | None = None) -> tuple:
    """Pack only explicit past observations, preserving partial action prefixes.

    Native frame IDs are computational order, not physical timestamps. RoPE
    offsets remain independently recorded. Unexecuted rectangle padding has
    token_valid=False; it never becomes a command or readable history token.
    """
    from .zerowam import NativeHistoryChunk
    _validate_sample_history(sample)
    result = []
    for chunk in sample.history_chunks:
        start, end = chunk["slice"]
        if chunk["mode"] == "video":
            latent = sample.robot_latent[:, :, start:end]
            valid = None
        else:
            commands = sample.observed_action_history.commands[:, start:end]
            batch, count, channels = commands.shape
            per_frame = sample.actions_per_frame
            frames = (count + per_frame - 1) // per_frame
            padded = commands.new_zeros(batch, frames * per_frame, channels)
            padded[:, :count] = commands
            latent = padded.reshape(batch, frames, per_frame, channels).permute(0, 3, 1, 2).unsqueeze(-1)
            valid = torch.arange(frames * per_frame, device=commands.device) < count
        if dtype is not None:
            latent = latent.to(dtype=dtype)
        result.append(NativeHistoryChunk(chunk["mode"], latent, chunk["frame_id"], chunk["rope_offset"], valid))
    return tuple(result)


def sampling_position(sample) -> dict[str, int]:
    """Next native video position after explicitly recorded computational chunks."""
    _validate_sample_history(sample)
    rope_ends = []
    for chunk in sample.history_chunks:
        count = chunk["slice"][1] - chunk["slice"][0]
        frames = count if chunk["mode"] == "video" else (count + sample.actions_per_frame - 1) // sample.actions_per_frame
        rope_ends.append(chunk["rope_offset"] + frames)
    return {"frame_id": 2 * (max(chunk["frame_id"] for chunk in sample.history_chunks) // 2 + 1),
            "rope_offset": max(rope_ends)}


@dataclass(frozen=True)
class LoadedSample:
    metadata: Mapping
    robot_history: Tensor
    proprio_history: Tensor
    embodiment: Tensor
    entity_patch_weights: Tensor
    robot_latent: Tensor
    actions: Tensor
    demonstrations: tuple[Tensor, ...]
    outcome: PhysicalOutcome
    requirement: TaskRequirement
    per_view_valid: tuple[Mapping[str, Tensor], ...]
    native_inputs: Mapping
    observed_action_history: ObservedActionHistory
    observed_video_step_offsets: Tensor
    history_chunks: tuple[Mapping, ...]
    actions_per_frame: int
    pair_kind: str = "none"

    @property
    def demonstration_encoding(self) -> dict:
        widths = {view.shape[-1] for view in self.demonstrations}
        if len(widths) != 1:
            raise ValueError("all demonstration views must have the same feature width")
        return validate_demo_encoding(self.metadata, widths.pop())

    def native_history(self, dtype: torch.dtype | None = None) -> tuple:
        return native_history(self, dtype)

    def sampling_position(self) -> dict[str, int]:
        return sampling_position(self)


def load_sample(manifest_path: str | Path) -> LoadedSample:
    """Read one audited JSON + NPZ sample on CPU, never pickle or object arrays.

    Arrays omit the batch dimension on disk. Metadata, file names, task IDs,
    source identifiers, and free text are not included in model-facing tensors.
    Unknown NPZ keys fail closed to catch misspelled labels or accidental inputs.
    """
    path = Path(manifest_path)
    with path.open(encoding="utf-8") as stream:
        meta = json.load(stream)
    if not isinstance(meta, dict) or type(meta.get("format_version")) is not int or meta["format_version"] != 2 or meta.get("kind") != "training_sample":
        raise ValueError("training requires an explicit version-2 training_sample manifest; v1 history/evidence cannot be inferred")
    for name in ("source_id", "trajectory_id", "history_id", "coordinate_frame", "task_annotation"):
        if not isinstance(meta.get(name), str) or not meta[name]:
            raise ValueError(f"manifest requires nonempty {name}")
    for name in ("window_start", "executed_steps"):
        if type(meta.get(name)) is not int or meta[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if meta.get("observation_step") != meta["window_start"]:
        raise ValueError("training observation_step must equal the future window's start")
    if not isinstance(meta.get("control_dt"), (float, int)) or isinstance(meta["control_dt"], bool) or not np.isfinite(meta["control_dt"]) or meta["control_dt"] <= 0:
        raise ValueError("control_dt must be a finite positive number")
    views = meta.get("view_ids")
    if not isinstance(views, list) or any(not isinstance(v, str) or not v for v in views) or len(set(views)) != len(views):
        raise ValueError("view_ids must name the recorded demonstration views without duplicates")
    pair_kind = _pair_kind(meta.get("pair_kind", "none"), len(views))
    demo_keys = tuple(f"demo_view_{view}" for view in range(len(views)))
    array_name = meta.get("arrays")
    if not isinstance(array_name, str):
        raise ValueError("arrays must name a relative NPZ file")
    array_path = (path.parent / array_name).resolve()
    if Path(array_name).is_absolute() or not array_path.is_relative_to(path.parent.resolve()) or array_path.suffix != ".npz":
        raise ValueError("array file must be an NPZ within the manifest directory")
    required = {"entity_ids", "step_offsets", "robot_history", "proprio_history", "embodiment", "entity_patch_weights", "robot_latent", "actions", "observed_action_history", "observed_action_step_offsets", "observed_video_step_offsets", *demo_keys}
    for field in FIELDS:
        required |= {f"outcome_{field}", f"outcome_{field}_valid"}
        for part in ("current", "remaining"):
            required |= {f"{part}_{field}", f"{part}_{field}_valid", f"{part}_{field}_required"}
    for part in ("current", "remaining"):
        required |= {f"{part}_binding", f"{part}_binding_valid", f"{part}_step_offsets"}
        required |= {f"{part}_{field}{suffix}" for field in REQUIREMENT_SEMANTICS for suffix in ("", "_valid")}
    # Visibility/evidence validity is data-owned and separate for each view.
    cv_fields = tuple(f"{part}_{field}" for part in ("current", "remaining") for field in ("binding", *FIELDS, *REQUIREMENT_SEMANTICS)) + ("relations", "events")
    required |= {f"view{view}_{field}_valid" for view in range(len(views)) for field in cv_fields}
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
            label_valid={field: arrays[f"{part}_{field}_valid"].unsqueeze(0) for field in (*FIELDS, "binding", *REQUIREMENT_SEMANTICS)},
            **{field: arrays[f"{part}_{field}"].unsqueeze(0) for field in REQUIREMENT_SEMANTICS},
        ).validate()
    requirement = TaskRequirement(**parts).validate()
    per_view_valid = []
    for view in range(len(views)):
        evidence = {}
        for part, target in parts.items():
            for field, label_mask in target.label_valid.items():
                mask = arrays[f"view{view}_{part}_{field}_valid"].unsqueeze(0)
                if mask.dtype != torch.bool or mask.shape != label_mask.shape:
                    raise ValueError(f"{part}.{field}: view evidence must match label mask shape and Boolean dtype")
                evidence[f"{part}.{field}"] = mask
        for field in ("relations", "events"):
            mask = arrays[f"view{view}_{field}_valid"].unsqueeze(0)
            if mask.dtype != torch.bool or mask.shape != outcome.label_valid[field].shape:
                raise ValueError(f"{field}: view evidence must match physical label mask shape and Boolean dtype")
            evidence[field] = mask
        per_view_valid.append(evidence)
    for name in ("robot_history", "robot_latent", *demo_keys):
        tensor = arrays[name]
        if tensor.ndim < 2 or not tensor.numel() or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"{name}: expected nonempty finite floating array with at least two axes")
    widths = {arrays[name].shape[-1] for name in demo_keys}
    if len(widths) != 1:
        raise ValueError("all demonstration views must have the same feature width")
    validate_demo_encoding(meta, widths.pop())
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
    observed_actions, observed_video_offsets, history_chunks, per_frame = _history_metadata(meta, arrays)
    if observed_actions.commands.shape[-1] != actions.shape[-1]:
        raise ValueError("observed commands and future action labels must use one action dimension")
    return LoadedSample(
        meta, history.unsqueeze(0), proprio.unsqueeze(0), embodiment.unsqueeze(0), patches.unsqueeze(0),
        arrays["robot_latent"].unsqueeze(0), actions.unsqueeze(0),
        tuple(arrays[name].unsqueeze(0) for name in demo_keys), outcome, requirement,
        tuple(per_view_valid), native_inputs, observed_actions, observed_video_offsets, history_chunks, per_frame, pair_kind,
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
    if config.get("schema_version") != 2 or config.get("interface") not in {"geometry", "full"}:
        raise ValueError("unsupported experiment schema or interface")
    condition = config["conditioning"]
    if condition["unconditional_probability"] != 0.1 or condition["drop_icl"] != 0 or condition["droptext_target"] != 0 or condition["task_text"] != "" or condition["pair_dropout"] != 0:
        raise ValueError("experiment violates the paired condition-drop protocol")
    if config["pair_supervision_reduction"] != "mean" or config["candidate_count"] != 1 or config["f_ranking"]:
        raise ValueError("representation experiments require averaged pairs and one unsorted candidate")
    if not np.isfinite(config["lambda_cv"]) or config["lambda_cv"] < 0:
        raise ValueError("lambda_cv must be finite and nonnegative")
    weights = config["training"].get("loss_weights")
    expected_weights = {"requirement", "latent", "execution", "next_video", "native_action", "ifp", "interaction", "cv", "physical"}
    if not isinstance(weights, dict) or set(weights) != expected_weights:
        raise ValueError("training.loss_weights must explicitly configure every objective")
    if any(type(value) not in (int, float) or not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("loss weights must be finite and nonnegative")
    if weights["cv"] != config["lambda_cv"]:
        raise ValueError("loss_weights.cv and the registered lambda_cv must agree")
    start = config["training"].get("exec_start_step")
    if type(start) is not int or not 0 <= start <= config["training"]["max_steps"]:
        raise ValueError("exec_start_step must be an explicit warmup boundary within the training budget")
    capacity = config["dimensions"].get("max_precedence_edges")
    if type(capacity) is not int or capacity < 1:
        raise ValueError("max_precedence_edges must be an explicit positive capacity")
    policy = config.get("binding_policy")
    if not isinstance(policy, dict) or set(policy) != {"confidence_threshold", "margin_threshold", "validation_locked"}:
        raise ValueError("binding_policy must specify confidence, margin and validation lock")
    for name in ("confidence_threshold", "margin_threshold"):
        if type(policy[name]) not in (int, float) or not np.isfinite(policy[name]) or not 0 < policy[name] < 1:
            raise ValueError("binding thresholds must be finite in (0,1)")
    if type(policy["validation_locked"]) is not bool:
        raise ValueError("binding policy validation lock must be Boolean")
    for part in ("current", "remaining"):
        offsets = config[f"{part}_offsets"]
        if not offsets or any(type(v) is not int or v <= 0 for v in offsets) or offsets != sorted(set(offsets)):
            raise ValueError("requirement offsets must be positive and strictly increasing")
        if config["tokens"][part] != config["dimensions"]["roles"] * len(offsets):
            raise ValueError("token budget must equal role count times query times")
    if for_test and not config.get("validation_locked", False):
        raise ValueError("lock the common configuration on validation data before test evaluation")
    if for_test and not policy["validation_locked"]:
        raise ValueError("lock binding refusal thresholds on validation data before test evaluation")
    return config


@dataclass(frozen=True)
class Observation:
    """Deployment input: actual past actions allowed; future actions/labels forbidden."""
    entity_ids: Tensor
    robot_history: Tensor
    proprio_history: Tensor
    embodiment: Tensor
    robot_latent: Tensor
    demonstrations: tuple[Tensor, ...]
    chunk_size: int
    actions_per_frame: int
    action_space: dict
    observed_action_history: ObservedActionHistory
    observed_video_step_offsets: Tensor
    history_chunks: tuple[Mapping, ...]
    demonstration_encoding: dict = field(default_factory=lambda: {"kind": "raw_features"})

    def native_history(self, dtype: torch.dtype | None = None) -> tuple:
        return native_history(self, dtype)

    def sampling_position(self) -> dict[str, int]:
        return sampling_position(self)


def load_observation(manifest_path: str | Path) -> Observation:
    path = Path(manifest_path)
    meta = json.loads(path.read_text())
    if not isinstance(meta, dict) or meta.get("format_version") != 2 or meta.get("kind") != "observation":
        raise ValueError("inference requires an explicit version-2 observation manifest, never future labels or inferred v1 history")
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
    required = {"entity_ids", "robot_history", "proprio_history", "embodiment", "robot_latent", "observed_action_history", "observed_action_step_offsets", "observed_video_step_offsets"}
    required |= {f"demo_view_{i}" for i in range(len(views))}
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if set(arrays.files) != required:
            raise ValueError("observation NPZ must contain only current/history and demonstration fields")
        values = {key: torch.from_numpy(arrays[key].copy()) for key in required}
    ids = values["entity_ids"]
    if ids.dtype != torch.int64 or ids.ndim != 1 or not (ids >= 0).any() or (ids < -1).any() or ids[ids >= 0].unique().numel() != (ids >= 0).sum():
        raise ValueError("observation entity IDs must be unique nonnegative integers or -1 padding")
    for key, value in values.items():
        if key not in {"entity_ids", "observed_action_history", "observed_action_step_offsets", "observed_video_step_offsets"} and (not value.is_floating_point() or not torch.isfinite(value).all() or not value.numel()):
            raise ValueError("observation features must be nonempty finite floating tensors")
    history, proprio = values["robot_history"], values["proprio_history"]
    if history.ndim != 3 or history.shape[1] != len(ids) or proprio.ndim != 2 or proprio.shape[0] != history.shape[0]:
        raise ValueError("observation entity/history/state axes differ")
    if values["embodiment"].ndim != 1 or values["robot_latent"].ndim != 4:
        raise ValueError("embodiment must be [E], observed video latent [C,F,H,W]")
    if any(values[f"demo_view_{i}"].ndim != 2 for i in range(len(views))):
        raise ValueError("demonstrations must be ordered [tokens,features]")
    widths = {values[f"demo_view_{i}"].shape[-1] for i in range(len(views))}
    if len(widths) != 1:
        raise ValueError("all demonstration views must have the same feature width")
    encoding = validate_demo_encoding(meta, widths.pop())
    observed_actions, video_offsets, history_chunks, per_frame = _history_metadata(meta, values)
    return Observation(ids[None], history[None], proprio[None], values["embodiment"][None],
                       values["robot_latent"][None], tuple(values[f"demo_view_{i}"][None] for i in range(len(views))),
                       meta["chunk_size"], per_frame, dict(meta["action_space"]), observed_actions, video_offsets, history_chunks, encoding)
