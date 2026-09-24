"""Task-local robot hindsight goals and causal gripper events for G and pi."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor

from .goal_data import _interface_metadata
from .goal_interface import validate_gripper, validate_se3
from .goal_language import load_goal_language
from .icl_data import LATENT_NORMALIZATION, _IDENTITY_FIELDS, _arrays, _identity
from .video_data import SPLITS, _local_path, _masked_values, _source_components, _text


@dataclass(frozen=True)
class EventRules:
    signal_source: str = "measured"
    close_threshold: float = .25
    open_threshold: float = .75
    debounce_steps: int = 2

    def __post_init__(self):
        if self.signal_source != "measured":
            raise ValueError("G/pi events must consistently use measured gripper values")
        if (any(type(x) not in (int, float) or not math.isfinite(x)
                for x in (self.close_threshold, self.open_threshold))
                or not 0 <= self.close_threshold < self.open_threshold <= 1):
            raise ValueError("event thresholds must satisfy 0 <= close_threshold < open_threshold <= 1")
        if type(self.debounce_steps) is not int or self.debounce_steps < 1:
            raise ValueError("debounce_steps must be a positive integer")

    @classmethod
    def from_metadata(cls, value: Mapping) -> EventRules:
        if not isinstance(value, dict) or set(value) != set(asdict(cls())):
            raise ValueError("event_rules requires signal_source, close_threshold, open_threshold and debounce_steps")
        return cls(**value)


@dataclass(frozen=True)
class GripperEvent:
    time: float
    effector: int
    kind: str


@dataclass(frozen=True)
class EventState:
    stable: tuple[int | None, ...]
    candidate: tuple[int | None, ...]
    counts: tuple[int, ...]
    last_time: float | None = None


def initial_event_state(effectors: int) -> EventState:
    if type(effectors) is not int or effectors < 1:
        raise ValueError("event detector effectors must be a positive integer")
    return EventState((None,) * effectors, (None,) * effectors, (0,) * effectors)


def gripper_event_step(state: EventState, values: Tensor, time: float,
                       rules: EventRules) -> tuple[EventState, tuple[GripperEvent, ...]]:
    """Confirm transitions at the current step, never backdate them to onset."""
    validate_gripper(values, (len(state.stable),), "measured gripper")
    if (type(time) not in (int, float) or not math.isfinite(time)
            or (state.last_time is not None and time <= state.last_time)):
        raise ValueError("event time must be finite and strictly increase")
    stable, candidates, counts = list(state.stable), list(state.candidate), list(state.counts)
    events = []
    for index, value in enumerate(values.detach().cpu().tolist()):
        target = 0 if value <= rules.close_threshold else 1 if value >= rules.open_threshold else None
        if target is None or target == stable[index]:
            candidates[index], counts[index] = None, 0
            continue
        counts[index] = counts[index] + 1 if candidates[index] == target else 1
        candidates[index] = target
        if counts[index] == rules.debounce_steps:
            if stable[index] is not None:
                events.append(GripperEvent(float(time), index, "open" if target else "close"))
            stable[index], candidates[index], counts[index] = target, None, 0
    return EventState(tuple(stable), tuple(candidates), tuple(counts), float(time)), tuple(events)


class GripperEventDetector:
    def __init__(self, effectors: int, rules: EventRules = EventRules()):
        self.rules = rules
        self.state = initial_event_state(effectors)

    def update(self, values: Tensor, time: float) -> tuple[GripperEvent, ...]:
        self.state, events = gripper_event_step(self.state, values, time, self.rules)
        return events

    def reset(self) -> None:
        self.state = initial_event_state(len(self.state.stable))


def detect_gripper_events(gripper: Tensor, times: Tensor,
                          rules: EventRules = EventRules()) -> tuple[GripperEvent, ...]:
    if gripper.ndim != 2 or min(gripper.shape) < 1:
        raise ValueError("event gripper must be nonempty [T,E]")
    validate_gripper(gripper, tuple(gripper.shape), "measured gripper")
    _times(times, gripper.shape[0], "control times")
    detector = GripperEventDetector(gripper.shape[1], rules)
    return tuple(event for values, time in zip(gripper, times.tolist())
                 for event in detector.update(values, time))


def next_subgoal_time(current_time: float, terminal_time: float,
                      events: tuple[GripperEvent, ...]) -> float:
    if (any(type(value) not in (int, float) or not math.isfinite(value)
            for value in (current_time, terminal_time)) or current_time < 0 or current_time >= terminal_time):
        raise ValueError("current_time must be task-local, nonnegative and strictly before terminal")
    return min((event.time for event in events if current_time < event.time <= terminal_time),
               default=float(terminal_time))


@dataclass(frozen=True)
class PiGoalSample:
    metadata: Mapping
    state: Tensor
    history: Tensor                # [1,C,F,H,W], physically sliced task-local history
    history_times: Tensor
    target_frame: Tensor           # [1,C,1,H,W], independently passed to frozen E
    goal_poses: Tensor
    goal_gripper: Tensor
    language: Tensor
    language_identity: Mapping
    actions: Tensor                # [1,A,F,N,1], padded only beyond terminal
    actions_mask: Tensor
    subgoal_time: float
    terminal_time: float
    events: tuple[GripperEvent, ...]


@dataclass(frozen=True)
class GTranslatorSample(PiGoalSample):
    demonstration: Tensor
    demonstration_times: Tensor


def _times(times: Tensor, frames: int, name: str) -> None:
    if (not times.is_floating_point() or times.shape != (frames,)
            or not torch.isfinite(times).all() or (times[1:] <= times[:-1]).any()):
        raise ValueError(f"{name} must be finite floating [{frames}] strictly increasing seconds")


def _metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    required = {"format_version", "kind", "sample_id", "arrays", "robot_source", "action_space",
                "state_space_id", "coordinate_frame", "pose_units", "goal_source", "end_effectors",
                "control_dt", "pose_representation", "tool_frames", "gripper_space", "language",
                "feature_space_id", "latent_normalization", "action_frames", "actions_per_frame",
                "task_start_time", "success", "event_rules", "frame_stride", "temporal_down_rate",
                "alignment", "subgoal_encoding"}
    if isinstance(metadata, dict) and metadata.get("format_version") == 1:
        raise ValueError("g_pi_task version 1 is unsupported; rebuild as version 2 with separate control and latent grids")
    if (not isinstance(metadata, dict) or required - set(metadata)
            or set(metadata) - required - {"demonstration", "compatibility", "provenance"}
            or type(metadata.get("format_version")) is not int or metadata["format_version"] != 2
            or metadata.get("kind") != "g_pi_task"):
        raise ValueError("expected an explicit version-2 g_pi_task schema")
    _interface_metadata({**metadata, "current_time": 0.})
    for name in ("sample_id", "arrays", "language", "feature_space_id"):
        _text(metadata, name)
    for name, suffix in (("arrays", ".npz"), ("language", ".json")):
        _local_path(path.parent, metadata[name], suffix)
    _identity(metadata["robot_source"])
    if set(metadata["robot_source"]) != _IDENTITY_FIELDS or metadata["robot_source"]["domain"] != "robot":
        raise ValueError("robot_source needs exactly robot source_id, source_group, domain and trajectory_id")
    if metadata["goal_source"] != "measured_endpoint":
        raise ValueError("G/pi goal_source must be measured_endpoint")
    if metadata["latent_normalization"] != LATENT_NORMALIZATION:
        raise ValueError(f"latent_normalization must be {LATENT_NORMALIZATION!r}")
    if type(metadata["task_start_time"]) not in (int, float) or metadata["task_start_time"] != 0:
        raise ValueError("task_start_time must equal 0 at the recording start")
    if metadata["success"] is not True:
        raise ValueError("G/pi terminal supervision requires an explicitly successful recording")
    for name in ("action_frames", "actions_per_frame", "frame_stride"):
        if type(metadata[name]) is not int or metadata[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(metadata["temporal_down_rate"]) is not int or metadata["temporal_down_rate"] != 4:
        raise ValueError("temporal_down_rate must equal the native Wan VAE rate 4")
    if metadata["actions_per_frame"] != metadata["temporal_down_rate"] * metadata["frame_stride"]:
        raise ValueError("actions_per_frame must equal temporal_down_rate * frame_stride = 4 * frame_stride")
    if metadata["alignment"] != "zerowam_causal_first_then_four":
        raise ValueError("alignment must be zerowam_causal_first_then_four")
    if metadata["subgoal_encoding"] != "wan_vae_single_frame":
        raise ValueError("subgoal_encoding must be wan_vae_single_frame, independently encoded from sequence history")
    EventRules.from_metadata(metadata["event_rules"])
    if ("demonstration" in metadata) != ("compatibility" in metadata):
        raise ValueError("paired demonstrations require task compatibility evidence")
    if "demonstration" in metadata:
        demo = metadata["demonstration"]
        _identity(demo)
        if demo["domain"] != "human" or set(demo) != {"source_id", "source_group", "domain", "arrays"}:
            raise ValueError("G demonstration must be a human video with only source identity and arrays")
        _local_path(path.parent, _text(demo, "arrays"), ".npz")
        pairing = metadata["compatibility"]
        if (not isinstance(pairing, dict) or set(pairing) != {"kind", "evidence"}
                or pairing["kind"] != "audited_semantic_task"):
            raise ValueError("compatibility requires audited_semantic_task and task-level evidence")
        _text(pairing, "evidence")
    return metadata


def validate_latent_grid(metadata: Mapping, control_times: Tensor, latent_available_times: Tensor) -> Tensor:
    """Return covered control indices for native first-frame-then-four causal latents."""
    frame_stride, temporal_down_rate = metadata.get("frame_stride"), metadata.get("temporal_down_rate")
    if type(frame_stride) is not int or frame_stride < 1:
        raise ValueError("frame_stride must be a positive integer")
    if type(temporal_down_rate) is not int or temporal_down_rate != 4:
        raise ValueError("temporal_down_rate must equal the native Wan VAE rate 4")
    if (type(metadata.get("actions_per_frame")) is not int
            or metadata["actions_per_frame"] != temporal_down_rate * frame_stride):
        raise ValueError("actions_per_frame must equal temporal_down_rate * frame_stride = 4 * frame_stride")
    if metadata.get("alignment") != "zerowam_causal_first_then_four":
        raise ValueError("alignment must be zerowam_causal_first_then_four")
    if (type(metadata.get("control_dt")) not in (int, float)
            or not math.isfinite(metadata["control_dt"]) or metadata["control_dt"] <= 0):
        raise ValueError("control_dt must be finite and positive")
    if control_times.ndim != 1 or control_times.numel() < 1:
        raise ValueError("control_times must be a nonempty control grid")
    _times(control_times, control_times.numel(), "control_times")
    expected = torch.arange(control_times.numel(), dtype=torch.float64) * metadata["control_dt"]
    tolerance = min(1e-6, metadata["control_dt"] * 1e-4)
    if not torch.allclose(control_times.double(), expected, atol=tolerance, rtol=0):
        raise ValueError("control_times must start at task time 0 and include every control step at control_dt")
    latent_times = latent_available_times
    _times(latent_times, latent_times.numel(), "latent_available_times")
    indices = torch.arange(0, control_times.numel(), temporal_down_rate * frame_stride)
    if (latent_times.shape != indices.shape
            or not torch.allclose(latent_times.double(), control_times[indices].double(), atol=tolerance, rtol=0)):
        raise ValueError("latent_available_times must cover the complete causal grid: first frame at 0, "
                         "then every 4 * frame_stride control steps at the last covered raw frame")
    return indices


def subgoal_control_indices(gripper: Tensor, times: Tensor,
                            rules: EventRules = EventRules()) -> Tensor:
    """All confirmed events, deduplicated at simultaneous times, then terminal."""
    events = detect_gripper_events(gripper, times, rules)
    subgoals = sorted({event.time for event in events} | {times[-1].item()})
    return torch.tensor([(times.double() == time).nonzero()[0].item() for time in subgoals])


def load_g_pi_sample(manifest_path: str | Path, *, route: str = "pi_goal",
                     current_time: float | None = None,
                     generator: torch.Generator | None = None) -> PiGoalSample:
    """Sample a newly available causal latent; pi never opens the human archive.

    Unpadded action column k is the command executed on (control_times[k],
    control_times[k + 1]]. Native Zero-WAM's initial N zero history slots are
    absent here: latent availability and future action windows are separate.
    """
    from .cli import action_space

    if route not in {"g_translator", "pi_goal"}:
        raise ValueError("G/pi route must be g_translator or pi_goal")
    path = Path(manifest_path)
    metadata = _metadata(path)
    expected = {"latent", "latent_available_times", "control_times", "states", "poses", "gripper",
                "actions", "actions_mask", "subgoal_times", "subgoal_latents"}
    with np.load(_local_path(path.parent, metadata["arrays"], ".npz"), allow_pickle=False) as archive:
        if len(archive.files) != len(expected) or set(archive.files) != expected:
            raise ValueError("G/pi version-2 task NPZ needs exactly latent, latent_available_times, control_times, "
                             "states, poses, gripper, actions, actions_mask, subgoal_times and subgoal_latents")
        arrays = {name: torch.from_numpy(archive[name].copy()) for name in expected}
    latent, latent_times, times, states, poses, gripper, actions, mask, subgoal_times, subgoal_latents = (
        arrays[name] for name in ("latent", "latent_available_times", "control_times", "states", "poses",
                                  "gripper", "actions", "actions_mask", "subgoal_times", "subgoal_latents"))
    if (latent.ndim != 4 or min(latent.shape) < 1
            or not latent.is_floating_point() or not torch.isfinite(latent).all()):
        raise ValueError("task latent must be finite floating nonempty [C,Tl,H,W]")
    if times.ndim != 1 or times.numel() < 2:
        raise ValueError("control_times must contain at least two recorded control steps")
    frames, effectors = times.numel(), len(metadata["end_effectors"])
    _times(latent_times, latent.shape[1], "latent_available_times")
    actions_per_frame = metadata["actions_per_frame"]
    latent_indices = validate_latent_grid(metadata, times, latent_times)
    if (states.ndim != 2 or states.shape[0] != frames or states.shape[1] < 1
            or not states.is_floating_point() or not torch.isfinite(states).all()):
        raise ValueError("states must be finite floating [Tc,S] measured robot states on control_times")
    if poses.shape != (frames, effectors, 4, 4):
        raise ValueError("poses must be [Tc,E,4,4] on control_times in the declared end_effectors order")
    validate_se3(poses)
    validate_gripper(gripper, (frames, effectors), "measured gripper")
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != frames:
        raise ValueError("task actions must be unpadded [A,Tc] commands on the control_times grid")
    space = action_space(metadata, actions.shape[0])
    if mask.dtype != torch.bool or mask.shape != actions.shape:
        raise ValueError("actions_mask must be Boolean with exactly the actions shape")
    mask = mask & torch.tensor(space["valid_channels"])[:, None]
    actions = _masked_values(actions, mask, "actions")
    rules = EventRules.from_metadata(metadata["event_rules"])
    events = detect_gripper_events(gripper, times, rules)
    terminal = times[-1].item()
    goal_indices = subgoal_control_indices(gripper, times, rules)
    expected_subgoals = times[goal_indices]
    _times(subgoal_times, expected_subgoals.numel(), "subgoal_times")
    if not torch.equal(subgoal_times.double(), expected_subgoals.double()):
        raise ValueError("subgoal_times must exactly match all recomputed measured event times plus terminal, "
                         "with simultaneous events deduplicated")
    if (subgoal_latents.shape != (subgoal_times.numel(), latent.shape[0], 1, *latent.shape[2:])
            or not subgoal_latents.is_floating_point() or not torch.isfinite(subgoal_latents).all()):
        raise ValueError("subgoal_latents must be finite floating [K,C,1,H,W] independently encoded single frames")
    if current_time is None:
        candidates = latent_indices[latent_indices < frames - 1]
        candidates = candidates[mask[:, candidates].any(0)]
        if not candidates.numel():
            raise ValueError("task requires valid action supervision at a latent availability strictly before terminal")
        index = candidates[torch.randint(candidates.numel(), (), generator=generator)].item()
        current_time = times[index].item()
    else:
        next_subgoal_time(current_time, terminal, events)
        tolerance = min(1e-6, metadata["control_dt"] * 1e-4)
        match = torch.isclose(latent_times.double(), torch.tensor(current_time, dtype=torch.float64),
                              atol=tolerance, rtol=0)
        if not match.any():
            raise ValueError("current_time must identify a latent availability on the control grid")
        index = latent_indices[match.nonzero()[0]].item()
        current_time = times[index].item()
    subgoal = next_subgoal_time(current_time, terminal, events)
    target_index = (expected_subgoals.double() == subgoal).nonzero()[0].item()
    goal_index = goal_indices[target_index].item()
    count = metadata["action_frames"] * actions_per_frame
    block = actions.new_zeros(actions.shape[0], count)
    block_mask = torch.zeros_like(block, dtype=torch.bool)
    available = min(count, frames - index)
    block[:, :available] = actions[:, index:index + available]
    block_mask[:, :available] = mask[:, index:index + available]
    # The command at the confirmed event time belongs to the next target.
    block_mask[:, goal_index - index:] = False
    block = _masked_values(block, block_mask, "actions")
    if not block_mask.any():
        raise ValueError("sample requires valid action supervision before subgoal_time")
    language, language_identity = load_goal_language(_local_path(path.parent, metadata["language"], ".json"))
    sample_metadata = {**metadata, "current_time": current_time, "subgoal_time": subgoal,
                       "terminal_time": terminal, "events": [asdict(event) for event in events],
                       "action_alignment": "control_step_start_unpadded"}
    if route == "pi_goal":
        sample_metadata.pop("demonstration", None)
        sample_metadata.pop("compatibility", None)
    shape = (1, actions.shape[0], metadata["action_frames"], actions_per_frame, 1)
    history_frames = index // actions_per_frame + 1
    values = dict(metadata=sample_metadata, state=states[index:index + 1].clone(),
                  history=latent[:, :history_frames].clone().unsqueeze(0),
                  history_times=latent_times[:history_frames].clone(),
                  target_frame=subgoal_latents[target_index:target_index + 1].clone(),
                  goal_poses=poses[goal_index:goal_index + 1].clone(),
                  goal_gripper=gripper[goal_index:goal_index + 1].clone(),
                  language=language, language_identity=language_identity,
                  actions=block.reshape(shape), actions_mask=block_mask.reshape(shape),
                  subgoal_time=subgoal, terminal_time=terminal, events=events)
    if route == "pi_goal":
        return PiGoalSample(**values)
    if "demonstration" not in metadata:
        raise ValueError("g_translator requires a task-paired human demonstration")
    demonstration = _arrays(path, metadata["demonstration"], robot_target=False)
    if demonstration["latent"].shape[0] != latent.shape[0]:
        raise ValueError("demonstration and robot history must share the latent channel dimension")
    return GTranslatorSample(**values, demonstration=demonstration["latent"].unsqueeze(0),
                              demonstration_times=demonstration["frame_times"])


def load_g_pi_index(index_path: str | Path, split: str = "train") -> tuple[list[Path], list[dict]]:
    """Audit task-level robot/human identities across splits without opening NPZs."""
    if split not in SPLITS:
        raise ValueError("split must be train, validation or test")
    path = Path(index_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(document, dict) or type(document.get("format_version")) is not int
            or document["format_version"] != 1 or document.get("kind") != "g_pi_index"
            or set(document) - {"format_version", "kind", "samples", "source_aliases", "bridge_sources"}):
        raise ValueError("expected a version-1 g_pi_index")
    entries = document.get("samples")
    aliases, bridges = document.get("source_aliases", []), document.get("bridge_sources", [])
    if not isinstance(entries, list) or not entries or not isinstance(aliases, list) or not isinstance(bridges, list):
        raise ValueError("G/pi index needs nonempty samples and optional source_aliases/bridge_sources lists")
    selected, records, identifiers = [], [], set()
    feature_space, rules = None, None
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"manifest", "split"} or entry["split"] not in SPLITS:
            raise ValueError("G/pi index samples require exactly manifest and a valid split")
        manifest = _local_path(path.parent, entry["manifest"], ".json")
        metadata = _metadata(manifest)
        if metadata["sample_id"] in identifiers:
            raise ValueError("sample_id must be unique across the G/pi index")
        identifiers.add(metadata["sample_id"])
        if feature_space is not None and metadata["feature_space_id"] != feature_space:
            raise ValueError("G/pi index requires a common feature_space_id")
        if rules is not None and metadata["event_rules"] != rules:
            raise ValueError("G/pi index requires identical measured event_rules")
        feature_space, rules = metadata["feature_space_id"], metadata["event_rules"]
        sources = [metadata["robot_source"]]
        if "demonstration" in metadata:
            sources.append(metadata["demonstration"])
        for source in sources:
            record = {name: source[name] for name in _IDENTITY_FIELDS if name in source}
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
        raise ValueError(f"G/pi dataset has no {split} samples")
    return selected, records


def g_pi_sample_files(manifest_path: str | Path, sample: PiGoalSample) -> list[Path]:
    """List actual sample dependencies for content-hash resume identities."""
    from .goal_language import _language_metadata

    path = Path(manifest_path)
    language = _local_path(path.parent, sample.metadata["language"], ".json")
    language_metadata = _language_metadata(language)
    paths = [path, _local_path(path.parent, sample.metadata["arrays"], ".npz"), language,
             _local_path(language.parent, language_metadata["arrays"], ".npz")]
    if isinstance(sample, GTranslatorSample):
        paths.append(_local_path(path.parent, sample.metadata["demonstration"]["arrays"], ".npz"))
    return paths
