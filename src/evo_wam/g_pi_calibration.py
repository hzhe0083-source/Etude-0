"""Offline, validation-only calibration of the four conjunctive stop tests."""

from __future__ import annotations

from itertools import combinations
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .g_pi_context import load_target_cache
from .g_pi_controller import GoalThresholds, goal_distances


DISTANCES = ("z", "position_m", "rotation_deg", "gripper")


def _json(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError) as exc:
        raise ValueError("calibration metadata must contain finite JSON values") from exc


def _identity(value):
    if (not isinstance(value, dict) or value.get("pooling") != "adaptive_avg_pool2d_spatial"
            or value.get("normalization") != "l2_last_dim" or value.get("token_order") != "camera_then_row_major"):
        raise ValueError("calibration requires the current two-dimensional E identity")
    layout = value.get("camera_layout")
    if (not isinstance(layout, list) or not layout or type(value.get("num_views")) is not int
            or value["num_views"] != len(layout) or any(not isinstance(view, dict)
                or set(view) != {"name", "token_width"} or not isinstance(view["name"], str)
                or not view["name"].strip() or type(view["token_width"]) is not int or view["token_width"] < 1
                for view in layout) or len({view["name"] for view in layout}) != len(layout)):
        raise ValueError("calibration E identity requires an ordered camera_layout and matching num_views")
    grid = value.get("grid_size")
    if (not isinstance(grid, list) or len(grid) != 2
            or any(type(x) is not int or x < 1 for x in grid)
            or value.get("k_z") != len(layout) * grid[0] * grid[1]
            or type(value.get("d_z")) is not int or value["d_z"] < 1
            or not isinstance(value.get("base_sha256"), str) or len(value["base_sha256"]) != 64):
        raise ValueError("calibration E identity requires grid_size, matching k_z/d_z and base checksum")
    return _json(value)


def _distances(value):
    if (not isinstance(value, dict) or set(value) != set(DISTANCES)
            or any(type(value[key]) not in (int, float) or not math.isfinite(value[key])
                   or value[key] < 0 for key in DISTANCES)):
        raise ValueError("calibration distances require finite nonnegative z, position_m, rotation_deg and gripper")
    return {key: float(value[key]) for key in DISTANCES}


def _upper(rows):
    return {key: max((row[key] for row in rows), default=0.) for key in DISTANCES}


def _accepted(distance, thresholds):
    return all(distance[key] <= thresholds[key] for key in DISTANCES)


def _thresholds(positive, negative, margin_fraction):
    if type(margin_fraction) not in (int, float) or not 0 < margin_fraction < 1:
        raise ValueError("calibration margin_fraction must lie strictly between zero and one")
    lower, upper = _upper(positive), _upper(negative)
    if any(_accepted(row, lower) for row in negative):
        raise ValueError("no feasible stopping thresholds: successful-target spread or arrival error "
                         "overlaps adjacent subgoals; inspect layout heterogeneity and boundary quality")
    span = {key: max(0., upper[key] - lower[key]) for key in DISTANCES}
    # Stop is an AND test. Each adjacent pair needs one separating coordinate;
    # requiring every coordinate to separate would reject z-only transitions.
    limit = min(max((row[key] - lower[key]) / span[key] if span[key] else 0.
                    for key in DISTANCES) for row in negative)
    thresholds = {key: lower[key] + margin_fraction * limit * span[key] for key in DISTANCES}
    if (not all(_accepted(row, thresholds) for row in positive)
            or any(_accepted(row, thresholds) for row in negative)):
        raise ValueError("no representable strict separation between calibration positives and adjacent goals")
    return thresholds, lower


def calibrate_goal_thresholds(recordings, *, encoder_identity, registry, arrivals=(),
                              margin_fraction=.5):
    """Calibrate on robot goals; group identities never enter a model call.

    Each recording supplies sample_id, intent_group, source_group and goals.
    Success goals are compared at equal robot subgoal ordinals across distinct
    source groups. Their spread is a conservative tolerance bound, not an
    assumption that differences between layouts are sensor noise.
    """
    identity = _identity(encoder_identity)
    if not isinstance(registry, dict) or not registry:
        raise ValueError("calibration requires a nonempty robot registry")
    registry = _json(registry)
    if not isinstance(recordings, (list, tuple)) or not recordings:
        raise ValueError("calibration requires validation recordings")
    groups, sample_ids, adjacent = {}, set(), []
    counts = []
    for record in recordings:
        required = {"sample_id", "intent_group", "source_group", "goals"}
        if (not isinstance(record, Mapping) or required - set(record)
                or set(record) - required - {"subgoal_keys"}):
            raise ValueError("calibration recordings require sample_id, intent_group, source_group and goals")
        for key in ("sample_id", "intent_group", "source_group"):
            if not isinstance(record[key], str) or not record[key].strip():
                raise ValueError(f"calibration {key} must be a nonempty offline identity")
        if record["sample_id"] in sample_ids:
            raise ValueError("calibration sample_id must be unique")
        sample_ids.add(record["sample_id"])
        goals = record["goals"]
        if not isinstance(goals, (list, tuple)) or not goals:
            raise ValueError("each calibration recording requires terminal-inclusive goals")
        keys = record.get("subgoal_keys", [f"ordinal:{i}" for i in range(len(goals) - 1)] + ["terminal"])
        if (not isinstance(keys, list) or len(keys) != len(goals)
                or any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys)):
            raise ValueError("calibration subgoal_keys must identify every robot ordinal or relation instance uniquely")
        record = {**record, "subgoal_keys": keys}
        for goal in goals:
            goal_distances(goal, goal)
            if tuple(goal["z"].shape) != (1, identity["k_z"], identity["d_z"]):
                raise ValueError("calibration goal z must match E identity and contain one robot")
            if not torch.allclose(goal["z"].double().norm(dim=-1),
                                  torch.ones_like(goal["z"][..., 0]).double(), atol=.01, rtol=0):
                raise ValueError("calibration goal z tokens must be unit normalized")
        groups.setdefault(record["intent_group"], []).append(record)
        adjacent.extend(goal_distances(left, right) for left, right in zip(goals, goals[1:]))
    spread = []
    for group, entries in groups.items():
        sources = {entry["source_group"] for entry in entries}
        if len(sources) < 2:
            raise ValueError("each calibration intent requires at least two independent robot source groups")
        if len({len(entry["goals"]) for entry in entries}) != 1:
            raise ValueError("same-intent calibration recordings require matching robot subgoal counts")
        if any(entry["subgoal_keys"] != entries[0]["subgoal_keys"] for entry in entries):
            raise ValueError("same-intent calibration requires the same ordered robot subgoal instances")
        group_pairs = 0
        for left, right in combinations(entries, 2):
            if left["source_group"] == right["source_group"]:
                continue
            spread.extend(goal_distances(a, b) for a, b in zip(left["goals"], right["goals"]))
            group_pairs += 1
        counts.append({"intent_group": group, "recordings": len(entries), "source_groups": len(sources),
                       "independent_pairs": group_pairs, "subgoals": len(entries[0]["goals"]),
                       "subgoal_keys": entries[0]["subgoal_keys"]})
    if not adjacent:
        raise ValueError("calibration requires adjacent subgoals to establish a stopping separation bound")
    arrival_errors = []
    for arrival in arrivals:
        if not isinstance(arrival, Mapping) or set(arrival) != {"reference", "achieved"}:
            raise ValueError("oracle-goal arrivals require reference and achieved goals")
        distance = goal_distances(arrival["reference"], arrival["achieved"])
        for name in ("reference", "achieved"):
            if tuple(arrival[name]["z"].shape) != (1, identity["k_z"], identity["d_z"]):
                raise ValueError("oracle-goal arrival z must match E identity")
            if not torch.allclose(arrival[name]["z"].double().norm(dim=-1),
                                  torch.ones_like(arrival[name]["z"][..., 0]).double(), atol=.01, rtol=0):
                raise ValueError("oracle-goal arrival z tokens must be unit normalized")
        arrival_errors.append(distance)
    thresholds, lower = _thresholds(spread + arrival_errors, adjacent, margin_fraction)
    return {"format_version": 1, "kind": "g_pi_stop_calibration", "encoder_identity": identity,
            "registry": registry, "thresholds": thresholds,
            "evidence": {"method": "max_positive_joint_and_separation", "margin_fraction": margin_fraction,
                         "positive_bounds": lower, "successful_target_spread": spread,
                         "oracle_arrival_errors": arrival_errors, "adjacent_distances": adjacent,
                         "groups": counts, "recordings": len(recordings)},
            "validation_files": {}}


def load_calibration_artifact(path, expected_identity, expected_registry=None):
    """Validate identities and recheck every recorded bound before deployment."""
    if isinstance(path, Mapping):
        artifact = _json(dict(path))
    else:
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = {"format_version", "kind", "encoder_identity", "registry", "thresholds", "evidence", "validation_files"}
    if (not isinstance(artifact, dict) or set(artifact) != fields or artifact.get("format_version") != 1
            or artifact.get("kind") != "g_pi_stop_calibration"):
        raise ValueError("expected a version-1 g_pi_stop_calibration artifact")
    if _identity(artifact["encoder_identity"]) != _identity(expected_identity):
        raise ValueError("E identity mismatch in stopping calibration")
    if expected_registry is not None and artifact["registry"] != _json(expected_registry):
        raise ValueError("robot registry mismatch in stopping calibration")
    thresholds = _distances(artifact["thresholds"])
    GoalThresholds(**thresholds)
    evidence = artifact["evidence"]
    if (not isinstance(evidence, dict) or evidence.get("method") != "max_positive_joint_and_separation"
            or not evidence.get("successful_target_spread") or not evidence.get("adjacent_distances")
            or not isinstance(evidence.get("oracle_arrival_errors"), list)):
        raise ValueError("stopping calibration is missing validation evidence")
    positive = [_distances(row) for row in evidence["successful_target_spread"] + evidence["oracle_arrival_errors"]]
    negative = [_distances(row) for row in evidence["adjacent_distances"]]
    calculated, lower = _thresholds(positive, negative, evidence.get("margin_fraction"))
    if thresholds != calculated or evidence.get("positive_bounds") != lower:
        raise ValueError("stopping calibration thresholds do not match their validation evidence")
    groups = evidence.get("groups")
    if not isinstance(groups, list) or not groups or any(
            not isinstance(group, dict) or group.get("source_groups", 0) < 2
            or group.get("independent_pairs", 0) < 1 for group in groups):
        raise ValueError("stopping calibration needs independent source-group evidence")
    files = artifact["validation_files"]
    if not isinstance(files, dict) or any(not isinstance(path, str) or not path
            or not isinstance(digest, str) or len(digest) != 64 for path, digest in files.items()):
        raise ValueError("stopping calibration requires validation file hashes")
    return artifact


def load_calibration_policy(path, expected_identity, expected_registry=None):
    artifact = load_calibration_artifact(path, expected_identity, expected_registry)
    return GoalThresholds(**artifact["thresholds"])


def _subgoal_keys(sample, indices, times):
    source = sample.metadata.get("subgoal_source", "gripper")
    annotation = sample.metadata.get("subgoal_annotation", {})
    relations = annotation.get("relations", [])
    relation_indices = annotation.get("relation_control_indices", [])
    if source != "gripper" and len(relations) != len(relation_indices):
        raise ValueError("calibration requires recomputed relation-to-control-index evidence")
    keys = []
    for index in indices.tolist():
        if source == "gripper":
            labels = [{"effector": event.effector, "kind": event.kind}
                      for event in sample.events if event.time == times[index].item()]
        else:
            labels = [relation for control_index, relation in zip(relation_indices, relations) if control_index == index]
        if index == times.numel() - 1:
            labels.append("terminal")
        keys.append(f"{source}:{len(keys)}:" + json.dumps(labels, sort_keys=True))
    return keys


def calibrate_validation(manifest_path, output_path, *, encoder=None, expected_identity=None,
                         expected_registry=None):
    """Read validated robot tasks and identity-checked E caches, or run E alone."""
    from .cli import file_sha256, write_json
    from .g_pi_data import g_pi_sample_files, load_g_pi_sample
    from .g_pi_deployment import load_goal_prediction
    from .goal_training import goal_registry
    from .video_data import _local_path

    path, output = Path(manifest_path), Path(output_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    required = {"format_version", "kind", "split", "encoder_identity", "registry", "recordings"}
    if (not isinstance(document, dict) or required - set(document)
            or set(document) - required - {"oracle_goal_arrivals", "margin_fraction"}
            or document["format_version"] != 1 or document["kind"] != "g_pi_validation"
            or document["split"] != "validation" or not isinstance(document["recordings"], list)):
        raise ValueError("expected a version-1 g_pi_validation manifest with split='validation'")
    identity = _identity(document["encoder_identity"])
    if expected_identity is not None and identity != _identity(expected_identity):
        raise ValueError("E identity mismatch in validation manifest")
    if encoder is not None and encoder.identity != identity:
        raise ValueError("E identity mismatch in calibration encoder")
    registry = document["registry"]
    if expected_registry is not None and registry != _json(expected_registry):
        raise ValueError("robot registry mismatch in validation manifest")
    files = {str(path.resolve()): file_sha256(path)}
    recordings, arrivals, sources = [], [], {}
    for record in document["recordings"]:
        if (not isinstance(record, dict) or {"task", "intent_group"} - set(record)
                or set(record) - {"task", "intent_group", "target_cache", "subgoal_keys"}):
            raise ValueError("validation recording requires task, intent_group and optional target_cache")
        task = _local_path(path.parent, record["task"], ".json")
        sample = load_g_pi_sample(task, route="pi_goal", generator=torch.Generator().manual_seed(0),
                                  read_language=registry.get("language_identity") is not None)
        if goal_registry(sample) != registry:
            raise ValueError("robot registry mismatch in validation recording")
        metadata = sample.metadata
        group = metadata["robot_source"]["source_group"]
        for field in ("source_id", "trajectory_id"):
            identifier = (field, metadata["robot_source"][field])
            if identifier in sources and sources[identifier] != group:
                raise ValueError("reused robot source or trajectory cannot count as independent source groups")
            sources[identifier] = group
        arrays_path = _local_path(task.parent, metadata["arrays"], ".npz")
        for dependency in g_pi_sample_files(task, sample):
            files[str(dependency.resolve())] = file_sha256(dependency)
        with np.load(arrays_path, allow_pickle=False) as archive:
            times = torch.from_numpy(archive["control_times"].copy())
            subgoal_times = torch.from_numpy(archive["subgoal_times"].copy())
            poses = torch.from_numpy(archive["poses"].copy())
            gripper = torch.from_numpy(archive["gripper"].copy())
            latents = torch.from_numpy(archive["subgoal_latents"].copy())
        indices = torch.searchsorted(times, subgoal_times)
        if "target_cache" in record:
            cache = _local_path(path.parent, record["target_cache"], ".npz")
            z = load_target_cache(cache, identity)
            files[str(cache.resolve())] = file_sha256(cache)
        else:
            if encoder is None:
                raise ValueError("validation requires target_cache or an explicit frozen E policy")
            parameter = next(encoder.native.parameters())
            with torch.no_grad():
                z = torch.cat([encoder(frame[None].to(parameter)).detach().float().cpu() for frame in latents])
        if z.shape[0] != subgoal_times.numel():
            raise ValueError("target cache must contain every subgoal in stored chronological order")
        goals = [{"z": z[i:i + 1], "goal_poses": poses[index:index + 1],
                  "goal_gripper": gripper[index:index + 1]} for i, index in enumerate(indices.tolist())]
        keys = _subgoal_keys(sample, indices, times)
        if "subgoal_keys" in record and record["subgoal_keys"] != keys:
            raise ValueError("declared calibration subgoal_keys differ from recomputed robot boundary instances")
        result = {"sample_id": metadata["sample_id"], "source_group": group,
                  "intent_group": record["intent_group"], "goals": goals, "subgoal_keys": keys}
        recordings.append(result)
    for record in document.get("oracle_goal_arrivals", []):
        if not isinstance(record, dict) or set(record) != {"reference", "achieved"}:
            raise ValueError("oracle_goal_arrivals require reference and achieved goal sidecars")
        arrival = {}
        for name in ("reference", "achieved"):
            goal_path = _local_path(path.parent, record[name], ".json")
            arrival[name] = load_goal_prediction(goal_path, encoder_identity=identity, registry=registry)
            metadata = json.loads(goal_path.read_text(encoding="utf-8"))
            goal_arrays = _local_path(goal_path.parent, metadata["arrays"], ".npz")
            for dependency in (goal_path, goal_arrays):
                files[str(dependency.resolve())] = file_sha256(dependency)
        arrivals.append(arrival)
    artifact = calibrate_goal_thresholds(recordings, encoder_identity=identity, registry=registry,
        arrivals=arrivals, margin_fraction=document.get("margin_fraction", .5))
    artifact["validation_files"] = files
    load_calibration_artifact(artifact, identity, registry)
    if output.exists():
        raise ValueError("calibration output must be a new artifact path")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, artifact)
    return artifact


def calibrate_g_pi_thresholds(args):
    encoder, identity, registry = None, None, None
    if getattr(args, "artifact", None) is not None:
        from .g_pi_training import load_g_pi_encoder

        encoder, payload = load_g_pi_encoder(args.artifact, device=getattr(args, "device", "cuda"),
                                            checkpoint=getattr(args, "checkpoint", None))
        identity, registry = payload["encoder_identity"], payload["registry"]
    elif getattr(args, "policy", None) is not None:
        from .g_pi_training import load_g_pi_policy

        _, _, encoder, payload = load_g_pi_policy(args.policy, device=getattr(args, "device", "cuda"),
                                                 checkpoint=getattr(args, "checkpoint", None))
        identity, registry = payload["encoder_identity"], payload["registry"]
    artifact = calibrate_validation(args.manifest, args.output, encoder=encoder,
                                    expected_identity=identity, expected_registry=registry)
    return {"calibration": str(Path(args.output).resolve()), "thresholds": artifact["thresholds"],
            "recordings": artifact["evidence"]["recordings"], "validation_only": True}
