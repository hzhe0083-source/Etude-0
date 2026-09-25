"""Explicit robot SE(3) goal blocks, independent of optional video supervision."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor

from .goal_language import load_goal_language
from .icl_data import (LATENT_NORMALIZATION, NativeICLSample, _IDENTITY_FIELDS, _arrays as _icl_arrays,
                       _identity, _metadata as _icl_metadata, load_icl_sample)
from .video_data import SPLITS, _local_path, _masked_values, _source_components, _text


@dataclass(frozen=True)
class GoalSample:
    metadata: Mapping
    state: Tensor                  # [1,S], current measured robot state
    goal_poses: Tensor             # [1,E,4,4], explicit goal-frame transforms
    goal_gripper: Tensor           # [1,E], declared closed=0/open=1 scale
    language: Tensor               # [1,L,D], frozen independent text features
    language_identity: Mapping
    actions: Tensor                # [1,A,F,N,1], one future action block only
    actions_mask: Tensor
    visual: NativeICLSample | None = None


@dataclass(frozen=True)
class GoalObservation:
    metadata: Mapping
    state: Tensor                  # [1,S], current measured robot state
    history: Tensor                # [1,C,F,H,W], observed frames only
    history_times: Tensor          # [F], seconds
    demonstration: Tensor          # [1,C,Fd,Hd,Wd], complete context video
    demonstration_times: Tensor    # [Fd], seconds
    language: Tensor               # [1,L,D], frozen independent text features
    language_identity: Mapping


def _interface_metadata(metadata: Mapping) -> None:
    from .cli import action_space

    for name in ("state_space_id", "coordinate_frame"):
        _text(metadata, name)
    space = metadata["action_space"]
    action_space(metadata, space.get("dimension") if isinstance(space, dict) else None)
    if metadata["pose_units"] != "m":
        raise ValueError("goal pose translation units must be m")
    effectors = metadata["end_effectors"]
    if (not isinstance(effectors, list) or not effectors
            or any(not isinstance(name, str) or not name.strip() for name in effectors)
            or len(set(effectors)) != len(effectors)):
        raise ValueError("end_effectors must be an ordered nonempty list of unique nonempty names")
    if metadata["pose_representation"] != "absolute_robot_base_tool":
        raise ValueError("pose_representation must be absolute_robot_base_tool")
    tools = metadata["tool_frames"]
    if (not isinstance(tools, list) or len(tools) != len(effectors)
            or any(not isinstance(name, str) or not name.strip() for name in tools)
            or len(set(tools)) != len(tools)):
        raise ValueError("tool_frames must name one unique tool frame per ordered end effector")
    gripper = metadata["gripper_space"]
    if not isinstance(gripper, dict) or set(gripper) != {"normalization_id", "closed", "open", "units"}:
        raise ValueError("gripper_space must declare normalization_id, closed, open and units")
    for name in ("normalization_id", "units"):
        _text(gripper, name)
    for name in ("closed", "open"):
        if (not isinstance(gripper[name], list) or len(gripper[name]) != len(effectors)
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in gripper[name])):
            raise ValueError("gripper_space closed/open must contain one finite physical value per end effector")
    if any(closed == opened for closed, opened in zip(gripper["closed"], gripper["open"])):
        raise ValueError("gripper_space closed and open values must differ for every end effector")
    for name in ("current_time", "control_dt"):
        if type(metadata[name]) not in (int, float) or not math.isfinite(metadata[name]):
            raise ValueError(f"{name} must be a finite number in seconds")
    if metadata["control_dt"] <= 0:
        raise ValueError("control_dt must be positive")
    if "provenance" in metadata and not isinstance(metadata["provenance"], dict):
        raise ValueError("provenance must be an object, never an additional model input")


def _validate_state(state: Tensor) -> None:
    if (state.ndim != 1 or state.numel() < 1 or not state.is_floating_point()
            or not torch.isfinite(state).all()):
        raise ValueError("state must be finite floating nonempty [S]")


def _metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    required = {"format_version", "kind", "sample_id", "arrays", "robot_source", "action_space",
                "state_space_id", "coordinate_frame", "pose_units", "goal_source", "end_effectors",
                "current_time", "goal_time", "control_dt", "pose_representation", "tool_frames",
                "gripper_space", "language"}
    if (not isinstance(metadata, dict) or required - set(metadata)
            or set(metadata) - required - {"visual_pair", "provenance"}
            or type(metadata.get("format_version")) is not int or metadata["format_version"] != 2
            or metadata.get("kind") != "se3_goal_sample"):
        raise ValueError("expected an explicit version-2 se3_goal_sample schema")
    _interface_metadata(metadata)
    for name in ("sample_id", "arrays"):
        _text(metadata, name)
    _local_path(path.parent, metadata["arrays"], ".npz")
    _local_path(path.parent, _text(metadata, "language"), ".json")
    source = metadata["robot_source"]
    _identity(source)
    if set(source) != _IDENTITY_FIELDS or source["domain"] != "robot":
        raise ValueError("robot_source needs exactly robot source_id, source_group, domain and trajectory_id")
    if metadata["goal_source"] not in {"measured_endpoint", "controller_target"}:
        raise ValueError("goal_source must explicitly distinguish measured_endpoint from controller_target")
    if type(metadata["goal_time"]) not in (int, float) or not math.isfinite(metadata["goal_time"]):
        raise ValueError("goal_time must be a finite number in seconds")
    if metadata["goal_time"] <= metadata["current_time"]:
        raise ValueError("goal_time must follow current_time")
    if "visual_pair" in metadata:
        _local_path(path.parent, _text(metadata, "visual_pair"), ".json")
    return metadata


def _matching_visual_metadata(metadata: Mapping, visual: Mapping) -> None:
    target = visual["target"]
    if any(target.get(name) != value for name, value in metadata["robot_source"].items()):
        raise ValueError("visual target must match the goal block robot_source identity")
    if target.get("action_space") != metadata["action_space"]:
        raise ValueError("visual target action_space must match the goal block")
    demo = visual["demonstration"]
    if demo["domain"] == "robot" and (
            demo["source_id"] == target["source_id"] or demo["source_group"] == target["source_group"]
            or ("trajectory_id" in demo and demo["trajectory_id"] == target.get("trajectory_id"))):
        raise ValueError("robot demonstration must be an independent recording/trajectory from the target")


def load_goal_sample(manifest_path: str | Path, *, visual: bool = False) -> GoalSample:
    """Stage 1 reads no videos; Stage 2 additionally checks the supplied visual pair."""
    from .cli import action_space
    from .goal_interface import validate_se3

    path = Path(manifest_path)
    metadata = _metadata(path)
    expected = {"state", "goal_poses", "goal_gripper", "actions", "actions_mask"}
    with np.load(_local_path(path.parent, metadata["arrays"], ".npz"), allow_pickle=False) as archive:
        if len(archive.files) != len(expected) or set(archive.files) != expected:
            raise ValueError("goal block NPZ needs exactly state, goal_poses, goal_gripper, actions and actions_mask")
        arrays = {name: torch.from_numpy(archive[name].copy()) for name in expected}
    state, poses, actions, mask = (arrays[name] for name in ("state", "goal_poses", "actions", "actions_mask"))
    _validate_state(state)
    if poses.shape != (len(metadata["end_effectors"]), 4, 4):
        raise ValueError("goal_poses must be [E,4,4] in the declared end_effectors order")
    validate_se3(poses)
    gripper = arrays["goal_gripper"]
    if (gripper.shape != (len(metadata["end_effectors"]),) or not gripper.is_floating_point()
            or not torch.isfinite(gripper).all() or ((gripper < 0) | (gripper > 1)).any()):
        raise ValueError("goal_gripper must be finite floating [E] normalized to [0,1] with closed=0 and open=1")
    if actions.ndim != 4 or min(actions.shape) < 1 or actions.shape[-1] != 1:
        raise ValueError("goal actions must be nonempty [A,F,N,1] for one future block")
    space = action_space(metadata, actions.shape[0])
    if mask.dtype != torch.bool or mask.shape != actions.shape:
        raise ValueError("actions_mask must be Boolean with exactly the actions shape")
    mask = mask & torch.tensor(space["valid_channels"])[:, None, None, None]
    actions = _masked_values(actions, mask, "actions")
    if not mask.any():
        raise ValueError("goal blocks require valid future action supervision")
    duration = actions.shape[1] * actions.shape[2] * metadata["control_dt"]
    if not math.isclose(metadata["goal_time"] - metadata["current_time"], duration, abs_tol=1e-6, rel_tol=0):
        raise ValueError("goal_time - current_time must equal F*N*control_dt for this future action block")
    pair = None
    if visual:
        if "visual_pair" not in metadata:
            raise ValueError("visual Stage 2 requires an explicit visual_pair")
        pair = load_icl_sample(_local_path(path.parent, metadata["visual_pair"], ".json"))
        _matching_visual_metadata(metadata, pair.metadata)
        history = pair.history_frames
        if pair.target.shape[2] - history != actions.shape[1]:
            raise ValueError("visual target future frame count must match the goal action block")
        future, future_mask = pair.actions[0, :, history:], pair.actions_mask[0, :, history:]
        if (future.shape != actions.shape or not torch.equal(future_mask, mask)
                or not torch.equal(future[mask], actions[mask])):
            raise ValueError("visual target future actions and validity masks must match the goal block")
        if (not math.isclose(metadata["current_time"], pair.target_times[history - 1].item(), abs_tol=1e-6, rel_tol=0)
                or not math.isclose(metadata["goal_time"], pair.target_times[-1].item(), abs_tol=1e-6, rel_tol=0)):
            raise ValueError("goal current_time/goal_time must match the visual target history endpoint/final time")
        future_times = metadata["current_time"] + torch.arange(1, actions.shape[1] + 1, dtype=torch.float64) * actions.shape[2] * metadata["control_dt"]
        if not torch.allclose(pair.target_times[history:].double(), future_times, atol=1e-6, rtol=0):
            raise ValueError("visual target future times must follow the action block cadence N*control_dt")
    language, language_identity = load_goal_language(_local_path(path.parent, metadata["language"], ".json"))
    return GoalSample(metadata, state.unsqueeze(0), poses.unsqueeze(0), gripper.unsqueeze(0),
                      language, language_identity, actions.unsqueeze(0), mask.unsqueeze(0), pair)


def load_goal_observation(manifest_path: str | Path) -> GoalObservation:
    """Load inference inputs with no slots for future video, action or goal labels."""
    path = Path(manifest_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    required = {"format_version", "kind", "arrays", "demonstration", "feature_space_id", "latent_normalization",
                "action_space", "current_time", "control_dt", "actions_per_frame", "state_space_id",
                "coordinate_frame", "pose_units", "end_effectors", "pose_representation", "tool_frames",
                "gripper_space", "language"}
    if (not isinstance(metadata, dict) or required - set(metadata) or set(metadata) - required - {"provenance"}
            or type(metadata.get("format_version")) is not int or metadata["format_version"] != 2
            or metadata.get("kind") != "se3_goal_observation"):
        raise ValueError("expected an explicit version-2 se3_goal_observation schema without supervision fields")
    _interface_metadata(metadata)
    _text(metadata, "feature_space_id")
    if metadata["latent_normalization"] != LATENT_NORMALIZATION:
        raise ValueError(f"latent_normalization must be {LATENT_NORMALIZATION!r}")
    if type(metadata["actions_per_frame"]) is not int or metadata["actions_per_frame"] < 1:
        raise ValueError("actions_per_frame must be a positive integer")
    record = metadata["demonstration"]
    _identity(record)
    if set(record) - _IDENTITY_FIELDS - {"arrays"}:
        raise ValueError("demonstration accepts only video source identity and arrays")
    _local_path(path.parent, _text(record, "arrays"), ".npz")
    expected = {"state", "history_latent", "history_times"}
    with np.load(_local_path(path.parent, _text(metadata, "arrays"), ".npz"), allow_pickle=False) as archive:
        if len(archive.files) != len(expected) or set(archive.files) != expected:
            raise ValueError("observation NPZ needs exactly state, history_latent and history_times, without supervision")
        arrays = {name: torch.from_numpy(archive[name].copy()) for name in expected}
    state, history, times = (arrays[name] for name in ("state", "history_latent", "history_times"))
    _validate_state(state)
    if (history.ndim != 4 or min(history.shape) < 1 or not history.is_floating_point()
            or not torch.isfinite(history).all()):
        raise ValueError("history_latent must be finite floating nonempty [C,F,H,W]")
    if (not times.is_floating_point() or times.shape != (history.shape[1],)
            or not torch.isfinite(times).all() or (times[1:] <= times[:-1]).any()):
        raise ValueError("history_times must be finite floating [F] strictly increasing seconds")
    if not math.isclose(metadata["current_time"], times[-1].item(), abs_tol=1e-6, rel_tol=0):
        raise ValueError("current_time must match the final observed history time")
    demo = _icl_arrays(path, record, robot_target=False)
    if demo["latent"].shape[0] != history.shape[0]:
        raise ValueError("demonstration and history must share the same latent channel dimension")
    language, language_identity = load_goal_language(_local_path(path.parent, _text(metadata, "language"), ".json"))
    return GoalObservation(metadata, state.unsqueeze(0), history.unsqueeze(0), times,
                           demo["latent"].unsqueeze(0), demo["frame_times"], language, language_identity)


def load_goal_index(index_path: str | Path, split: str = "train") -> tuple[list[Path], list[dict]]:
    """Audit all robot, visual and alias/bridge identities across splits without arrays."""
    if split not in SPLITS:
        raise ValueError("split must be train, validation or test")
    path = Path(index_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(document, dict) or type(document.get("format_version")) is not int
            or document["format_version"] != 2 or document.get("kind") != "se3_goal_index"
            or set(document) - {"format_version", "kind", "samples", "source_aliases", "bridge_sources"}):
        raise ValueError("expected a version-2 se3_goal_index")
    entries, aliases, bridges = (document.get("samples"), document.get("source_aliases", []),
                                  document.get("bridge_sources", []))
    if not isinstance(entries, list) or not entries or not isinstance(aliases, list) or not isinstance(bridges, list):
        raise ValueError("goal index needs nonempty samples and optional source_aliases/bridge_sources lists")
    selected, records, sample_ids = [], [], set()
    goal_source = None
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"manifest", "split"} or entry["split"] not in SPLITS:
            raise ValueError("goal index samples require exactly manifest and a valid split")
        manifest = _local_path(path.parent, entry["manifest"], ".json")
        metadata = _metadata(manifest)
        if goal_source is not None and metadata["goal_source"] != goal_source:
            raise ValueError("goal index must not mix measured_endpoint and controller_target labels")
        goal_source = metadata["goal_source"]
        if metadata["sample_id"] in sample_ids:
            raise ValueError("sample_id must be unique across the goal index")
        sample_ids.add(metadata["sample_id"])
        sources = [metadata["robot_source"]]
        if "visual_pair" in metadata:
            pair = _icl_metadata(_local_path(manifest.parent, metadata["visual_pair"], ".json"))
            _matching_visual_metadata(metadata, pair)
            sources.extend(pair[role] for role in ("demonstration", "target"))
        for source in sources:
            record = {key: source[key] for key in _IDENTITY_FIELDS if key in source}
            records.append({**record, "sample_id": metadata["sample_id"], "record_kind": "video", "split": entry["split"]})
        if entry["split"] == split:
            selected.append(manifest)
    for alias in aliases:
        if not isinstance(alias, dict) or set(alias) - _IDENTITY_FIELDS - {"split"}:
            raise ValueError("source_aliases contain only video source identities and split")
        _identity(alias)
        records.append({**alias, "record_kind": "video"})
    for bridge in bridges:
        if not isinstance(bridge, dict) or set(bridge) - {"source_id", "source_group", "trajectory_id", "split"}:
            raise ValueError("bridge_sources contain only source/trajectory identities and split")
        records.append({**bridge, "record_kind": "bridge"})
    _source_components(records)
    if not selected:
        raise ValueError(f"goal dataset has no {split} samples")
    return selected, records
