"""Auditable offline visual/proprioceptive subgoal boundaries; never model inputs."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor


SOURCES = {"gripper", "sim_relation", "pedal", "candidate_match"}
RELATION_REGISTRY = "visual_relations_v1"
RELATIONS = {"in_hand", "placed", "opened", "closed"}
OFFLINE_ARRAYS = {"object_positions", "object_extents", "object_joint_fractions", "pedal_times"}
SIM_THRESHOLDS = {"hand_distance": .1, "min_lift": .02, "grasp_width_max": .6,
                  "release_min": .75, "position_tolerance": .01, "speed_max": .02,
                  "open_fraction": .95, "closed_fraction": .05}
CANDIDATE_THRESHOLDS = {"blocked_width_min": .05, "blocked_width_max": .6,
                        "width_stability": .01, "speed_max": .02,
                        "min_duration": .1, "max_duration": 60.}
VERSIONS = {"gripper": "measured_gripper_v1", "sim_relation": "visual_geometry_v1",
            "pedal": "pedal_clock_v1", "candidate_match": "proprio_candidates_v1"}


def _number(value, name, minimum=0.):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


def _string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _indices(value, frames):
    if (not isinstance(value, list) or not value or any(type(i) is not int or i < 0 or i >= frames for i in value)
            or value != sorted(set(value)) or value[-1] != frames - 1):
        raise ValueError("subgoal control_indices must strictly increase and include terminal")
    return value


def _thresholds(value, defaults):
    if not isinstance(value, dict) or set(value) != set(defaults):
        raise ValueError(f"subgoal thresholds must explicitly contain {sorted(defaults)}")
    return {key: _number(number, f"threshold {key}") for key, number in value.items()}


def validate_relations(value, effectors):
    if not isinstance(value, list) or not value:
        raise ValueError("relations must contain a nonempty ordered offline task program")
    counts = {}
    for relation in value:
        if not isinstance(relation, dict):
            raise ValueError("each relation must declare predicate, object_role, effector and occurrence")
        required = {"predicate", "object_role", "effector", "occurrence"}
        if relation.get("predicate") == "placed":
            required |= {"relation", "target_role"}
        if (set(relation) != required or not isinstance(relation.get("predicate"), str)
                or relation["predicate"] not in RELATIONS):
            raise ValueError("visual_relations_v1 supports only in_hand, placed, opened and closed")
        _string(relation["object_role"], "object_role")
        if type(relation["effector"]) is not int or not 0 <= relation["effector"] < effectors:
            raise ValueError("relation effector must identify an ordered end effector")
        if relation["predicate"] == "placed":
            if relation["relation"] not in {"on", "in"}:
                raise ValueError("placed relation must be on or in for visual_relations_v1")
            _string(relation["target_role"], "target_role")
            if relation["object_role"] == relation["target_role"]:
                raise ValueError("placed object and target roles must differ")
        identity = tuple((name, relation[name]) for name in sorted(required - {"occurrence"}))
        counts[identity] = counts.get(identity, 0) + 1
        if type(relation["occurrence"]) is not int or relation["occurrence"] != counts[identity]:
            raise ValueError("relation occurrence must count repeated identical role instances from 1")
    return value


def _finite(value, shape, name):
    if (not isinstance(value, Tensor) or tuple(value.shape) != tuple(shape)
            or not value.is_floating_point() or not torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite floating {list(shape)}")
    return value.double()


def _inputs(arrays):
    from .g_pi_data import _times
    from .goal_interface import validate_gripper, validate_se3

    times = arrays.get("control_times")
    if not isinstance(times, Tensor) or times.ndim != 1 or len(times) < 2:
        raise ValueError("subgoal evidence requires control_times with at least two control steps")
    _times(times, len(times), "control_times")
    poses, grip = arrays.get("poses"), arrays.get("gripper")
    if not isinstance(grip, Tensor) or grip.ndim != 2 or grip.shape[0] != len(times):
        raise ValueError("subgoal evidence requires recorded gripper [Tc,E]")
    validate_gripper(grip, tuple(grip.shape), "recorded gripper")
    if not isinstance(poses, Tensor) or poses.shape != (len(times), grip.shape[1], 4, 4):
        raise ValueError("subgoal evidence requires poses [Tc,E,4,4]")
    validate_se3(poses)
    return times.double(), poses.double(), grip.double()


def _confirmed(signals: Tensor, stable_steps: int) -> list[list[int]]:
    """Rising runs confirm now; a new occurrence requires the predicate to reset."""
    output = [[] for _ in range(signals.shape[1])]
    for column in range(signals.shape[1]):
        count = 0
        for index, active in enumerate(signals[:, column].tolist()):
            count = count + 1 if active else 0
            if count == stable_steps:
                output[column].append(index)
    return output


def _sim_indices(annotation, arrays, times, poses, grip):
    thresholds = _thresholds(annotation["thresholds"], SIM_THRESHOLDS)
    if (not 0 <= thresholds["closed_fraction"] < thresholds["open_fraction"] <= 1
            or not 0 < thresholds["grasp_width_max"] < thresholds["release_min"] <= 1):
        raise ValueError("visual relation opening/gripper thresholds must be ordered in [0,1]")
    roles = annotation.get("object_roles")
    if (not isinstance(roles, list) or not roles or any(not isinstance(x, str) or not x.strip() for x in roles)
            or len(set(roles)) != len(roles)):
        raise ValueError("sim_relation object_roles must give a unique ordered role list")
    if annotation.get("geometry") != "axis_aligned_robot_base_boxes":
        raise ValueError("sim_relation geometry must be axis_aligned_robot_base_boxes")
    positions = _finite(arrays.get("object_positions"), (len(times), len(roles), 3), "object_positions")
    extents = _finite(arrays.get("object_extents"), (len(roles), 3), "object_extents")
    if (extents <= 0).any():
        raise ValueError("object_extents must be positive axis-aligned half extents in metres")
    speed = torch.zeros_like(positions[..., 0])
    speed[1:] = (positions[1:] - positions[:-1]).norm(dim=-1) / (times[1:] - times[:-1])[:, None]
    joints = None
    if any(item["predicate"] in {"opened", "closed"} for item in annotation["relations"]):
        joints = _finite(arrays.get("object_joint_fractions"), (len(times), len(roles)), "object_joint_fractions")
        if ((joints < 0) | (joints > 1)).any():
            raise ValueError("object_joint_fractions must normalize visible closed/open limits into [0,1]")
    signals = []
    for relation in annotation["relations"]:
        if relation["object_role"] not in roles or (relation.get("target_role", roles[0]) not in roles):
            raise ValueError("relation references an unknown object_role or target_role")
        index, effector = roles.index(relation["object_role"]), relation["effector"]
        pos, predicate = positions[:, index], relation["predicate"]
        if predicate == "in_hand":
            signal = ((pos - poses[:, effector, :3, 3]).norm(dim=-1) <= thresholds["hand_distance"])
            signal &= grip[:, effector] <= thresholds["grasp_width_max"]
            signal &= pos[:, 2] - pos[0, 2] >= thresholds["min_lift"]
        elif predicate == "placed":
            target = roles.index(relation["target_role"])
            delta = pos - positions[:, target]
            tolerance = thresholds["position_tolerance"]
            if relation["relation"] == "on":
                signal = (delta[:, :2].abs() + extents[index, :2] <= extents[target, :2] + tolerance).all(-1)
                signal &= (delta[:, 2] - extents[index, 2] - extents[target, 2]).abs() <= tolerance
            else:
                signal = (delta.abs() + extents[index] <= extents[target] + tolerance).all(-1)
            signal &= (grip[:, effector] >= thresholds["release_min"]) & (speed[:, index] <= thresholds["speed_max"])
        else:
            signal = (joints[:, index] >= thresholds["open_fraction"] if predicate == "opened"
                      else joints[:, index] <= thresholds["closed_fraction"])
        signals.append(signal)
    confirmations = _confirmed(torch.stack(signals, dim=1), annotation["stable_steps"])
    indices, previous = [], -1
    for relation, candidates in zip(annotation["relations"], confirmations):
        upcoming = [index for index in candidates if index > previous]
        if not upcoming:
            raise ValueError("sim_relation has a missing or out-of-order stable relation occurrence")
        index = upcoming[0]
        indices.append(index)
        previous = index
    return indices


def generate_candidates(arrays: Mapping[str, Tensor], event_rules, thresholds=None, stable_steps=2):
    """Confirm measured open/close, blocked closure and speed minima causally."""
    from .g_pi_data import EventRules, detect_gripper_events

    rules = EventRules.from_metadata(event_rules) if isinstance(event_rules, dict) else event_rules
    if not isinstance(rules, EventRules) or rules.signal_source != "measured":
        raise ValueError("candidate generation requires measured event_rules")
    if type(stable_steps) is not int or stable_steps < 1:
        raise ValueError("candidate stable_steps must be a positive integer")
    config = _thresholds(dict(CANDIDATE_THRESHOLDS) if thresholds is None else thresholds, CANDIDATE_THRESHOLDS)
    if (not 0 <= config["blocked_width_min"] < config["blocked_width_max"] < rules.open_threshold
            or config["min_duration"] > config["max_duration"]):
        raise ValueError("candidate blocked widths and duration thresholds must be ordered")
    times, poses, grip = _inputs(arrays)
    candidates = [{"control_index": int((times == event.time).nonzero()[0]), "effector": event.effector,
                   "kind": f"gripper_{event.kind}"}
                  for event in detect_gripper_events(grip, times, rules)]
    for effector in range(grip.shape[1]):
        armed, closing, stable = False, False, 0
        held, released = False, 0
        speed = torch.zeros(len(times), dtype=torch.float64)
        speed[1:] = (poses[1:, effector, :3, 3] - poses[:-1, effector, :3, 3]).norm(dim=-1) / (times[1:] - times[:-1])
        moving, stopped = False, 0
        for index in range(len(times)):
            width = grip[index, effector].item()
            if held:
                released = released + 1 if width >= rules.open_threshold else 0
                if released == rules.debounce_steps:
                    candidates.append({"control_index": index, "effector": effector, "kind": "gripper_open"})
                    held, released = False, 0
            if width >= rules.open_threshold:
                armed, closing, stable = True, False, 0
            if index and armed:
                change = width - grip[index - 1, effector].item()
                closing |= change < -config["width_stability"]
                valid = (closing and config["blocked_width_min"] <= width <= config["blocked_width_max"]
                         and abs(change) <= config["width_stability"])
                stable = stable + 1 if valid else 0
                if stable == stable_steps:
                    candidates.append({"control_index": index, "effector": effector, "kind": "blocked_close"})
                    armed, held = False, True
            if speed[index] > config["speed_max"]:
                moving, stopped = True, 0
            elif moving:
                stopped += 1
                if stopped == stable_steps:
                    candidates.append({"control_index": index, "effector": effector, "kind": "speed_minimum"})
                    moving = False
            # A local minimum is only observable at its following rising sample.
            if (index >= 3 and speed[index - 2] > speed[index - 1] < speed[index]
                    and speed[index - 1] <= config["speed_max"]):
                candidates.append({"control_index": index, "effector": effector, "kind": "speed_minimum"})
    unique = {(item["control_index"], item["effector"], item["kind"]): item for item in candidates}
    return [unique[key] for key in sorted(unique)]


def _candidate_indices(annotation, arrays, metadata, times):
    thresholds = _thresholds(annotation["thresholds"], CANDIDATE_THRESHOLDS)
    candidates = generate_candidates(arrays, metadata["event_rules"], thresholds, annotation["stable_steps"])
    if "candidates" not in annotation or annotation["candidates"] != candidates:
        raise ValueError("candidate_match candidates must exactly match regenerated causal evidence")
    if annotation.get("weak_label") is not True:
        raise ValueError("candidate_match must explicitly declare weak_label=true")
    _string(annotation.get("template_version"), "candidate template_version")
    kinds = {"in_hand": {"blocked_close"}, "placed": {"gripper_open"},
             "opened": {"speed_minimum"}, "closed": {"speed_minimum"}}
    indices, previous, cursor = [], 0, 0
    relations = annotation["relations"]
    useful = set().union(*(kinds[item["predicate"]] for item in relations))
    for relation in relations:
        selected = None
        while cursor < len(candidates):
            candidate = candidates[cursor]
            cursor += 1
            if (candidate["control_index"] <= previous or candidate["effector"] != relation["effector"]
                    or candidate["kind"] not in useful):
                continue
            if candidate["kind"] not in kinds[relation["predicate"]]:
                raise ValueError("candidate_match evidence contradicts relation order")
            selected = candidate["control_index"]
            break
        if selected is None:
            raise ValueError("candidate_match is missing a required relation segment")
        duration = (times[selected] - times[previous]).item()
        if not thresholds["min_duration"] <= duration <= thresholds["max_duration"]:
            raise ValueError("candidate_match segment duration is outside declared bounds")
        indices.append(selected)
        previous = selected
    for candidate in candidates[cursor:]:
        if candidate["control_index"] > previous and candidate["kind"] in useful:
            raise ValueError("candidate_match has extra relation candidates after the declared program")
    tail = (times[-1] - times[previous]).item()
    if tail > thresholds["max_duration"]:
        raise ValueError("candidate_match terminal segment exceeds maximum duration")
    return indices


def resolve_subgoal_indices(metadata: Mapping, arrays: Mapping[str, Tensor], *, check_indices=True):
    """Return exact control indices and audit metadata without exposing labels to models."""
    from .g_pi_data import EventRules, subgoal_control_indices

    source = metadata.get("subgoal_source", "gripper")
    if not isinstance(source, str) or source not in SOURCES:
        raise ValueError("subgoal_source must be gripper, sim_relation, pedal or candidate_match")
    times, poses, grip = _inputs(arrays)
    annotation = metadata.get("subgoal_annotation")
    rules = EventRules.from_metadata(metadata["event_rules"])
    if rules.signal_source == "command":
        if metadata.get("gripper_signal_source") != "command":
            raise ValueError("command events require explicit gripper_signal_source=command")
        _string(metadata.get("gripper_source_evidence"), "gripper_source_evidence")
        if source != "gripper":
            raise ValueError("command gripper cannot establish measured candidate or visual relation evidence")
    if source == "gripper":
        indices = subgoal_control_indices(grip, times, rules).tolist()
        version = VERSIONS[source] if rules.signal_source == "measured" else "command_gripper_v1"
        audit = {"detector_version": version, "thresholds": asdict(rules),
                 "stable_steps": rules.debounce_steps, "control_indices": indices,
                 "evidence": f"gripper:{rules.signal_source}", "weak_label": True}
        if rules.signal_source == "command":
            audit["source_evidence"] = metadata["gripper_source_evidence"]
        if annotation is not None and annotation != audit:
            raise ValueError("gripper subgoal_annotation must match recomputed measured event rules and indices")
        return torch.tensor(indices, dtype=torch.long), audit
    required = {"detector_version", "thresholds", "stable_steps", "evidence", "weak_label", "relations", "relation_registry"}
    if not isinstance(annotation, dict) or required - set(annotation):
        raise ValueError("subgoal_annotation requires version, thresholds, stable_steps, evidence, weak_label and relations")
    optional = {"control_indices", "relation_control_indices"}
    extras = {"sim_relation": {"object_roles", "geometry"},
              "pedal": {"annotation_version", "clock_map", "confirmation_delay"},
              "candidate_match": {"template_version", "candidates"}}[source]
    if set(annotation) - required - optional - extras:
        raise ValueError("subgoal_annotation contains unsupported fields; only visual/proprioceptive evidence is allowed")
    if annotation["detector_version"] != VERSIONS[source]:
        raise ValueError(f"{source} requires detector_version={VERSIONS[source]}")
    if annotation["relation_registry"] != RELATION_REGISTRY:
        raise ValueError("relation_registry must equal visual_relations_v1; extensions need a new fixed registry")
    _string(annotation["evidence"], "subgoal evidence")
    if type(annotation["stable_steps"]) is not int or annotation["stable_steps"] < 1:
        raise ValueError("subgoal stable_steps must be a positive integer")
    if type(annotation["weak_label"]) is not bool:
        raise ValueError("subgoal weak_label must be explicit Boolean")
    validate_relations(annotation["relations"], grip.shape[1])
    if source == "sim_relation":
        indices = _sim_indices(annotation, arrays, times, poses, grip)
    elif source == "candidate_match":
        indices = _candidate_indices(annotation, arrays, metadata, times)
    else:
        if annotation["stable_steps"] != 1 or annotation["thresholds"] != {}:
            raise ValueError("pedal uses raw confirmed presses: stable_steps=1 and thresholds={}")
        _string(annotation.get("annotation_version"), "pedal annotation_version")
        clock = annotation.get("clock_map")
        if not isinstance(clock, dict) or set(clock) != {"scale", "offset"}:
            raise ValueError("pedal clock_map must declare scale and offset")
        scale = _number(clock["scale"], "pedal clock scale")
        if scale == 0:
            raise ValueError("pedal clock scale must be positive")
        offset = clock["offset"]
        if type(offset) not in (int, float) or not math.isfinite(offset):
            raise ValueError("pedal clock offset must be finite")
        _number(annotation.get("confirmation_delay"), "pedal confirmation_delay")
        raw = _finite(arrays.get("pedal_times"), (len(annotation["relations"]),), "pedal_times")
        if (raw[1:] <= raw[:-1]).any():
            raise ValueError("pedal raw event log times must strictly increase")
        mapped = raw * scale + offset
        if (mapped < times[0]).any() or (mapped > times[-1]).any():
            raise ValueError("mapped pedal times must lie inside the successful recording")
        indices = torch.searchsorted(times, mapped).tolist()
        if len(indices) != len(set(indices)):
            raise ValueError("distinct pedal events must map to distinct control steps")
    relation_indices = list(indices)
    if "relation_control_indices" in annotation and annotation["relation_control_indices"] != relation_indices:
        raise ValueError("relation_control_indices do not match recomputed relation evidence")
    indices = sorted(set(indices) | {len(times) - 1})
    if check_indices:
        recorded = _indices(annotation.get("control_indices"), len(times))
        if recorded != indices:
            raise ValueError("subgoal control_indices do not match recomputed source evidence")
    return torch.tensor(indices, dtype=torch.long), {**annotation, "control_indices": indices,
                                                    "relation_control_indices": relation_indices}


def generate_candidate_annotation(spec_path, output):
    """Build a weak-label annotation from a local task template and robot evidence."""
    from .video_data import _local_path

    path, output = Path(spec_path), Path(output)
    spec = json.loads(path.read_text())
    required = {"format_version", "kind", "arrays", "event_rules", "relations", "thresholds",
                "stable_steps", "evidence", "template_version"}
    if (not isinstance(spec, dict) or set(spec) != required or type(spec["format_version"]) is not int
            or spec["format_version"] != 1
            or spec["kind"] != "g_pi_candidate_spec"):
        raise ValueError("expected version-1 g_pi_candidate_spec with arrays and an offline relation template")
    with np.load(_local_path(path.parent, spec["arrays"], ".npz"), allow_pickle=False) as archive:
        if not {"control_times", "poses", "gripper"} <= set(archive.files):
            raise ValueError("candidate evidence NPZ requires control_times, poses and measured gripper")
        arrays = {name: torch.from_numpy(archive[name].copy()) for name in ("control_times", "poses", "gripper")}
    annotation = {name: spec[name] for name in ("relations", "thresholds", "stable_steps", "evidence", "template_version")}
    annotation.update(detector_version=VERSIONS["candidate_match"], relation_registry=RELATION_REGISTRY, weak_label=True,
                      candidates=generate_candidates(arrays, spec["event_rules"], spec["thresholds"], spec["stable_steps"]))
    metadata = {"subgoal_source": "candidate_match", "subgoal_annotation": annotation, "event_rules": spec["event_rules"]}
    indices, audit = resolve_subgoal_indices(metadata, arrays, check_indices=False)
    result = {"subgoal_source": "candidate_match", "subgoal_annotation": audit,
              "subgoal_times": arrays["control_times"][indices].tolist()}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result
