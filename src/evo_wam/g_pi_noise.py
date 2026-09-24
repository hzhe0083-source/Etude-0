"""Bounded hindsight-goal corruption and validation-only residual calibration."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F

from .goal_interface import validate_goal_poses, validate_gripper


SCALES = ("z_std", "translation_std", "rotation_std", "gripper_std")
BOUNDS = ("translation_max_m", "rotation_max_deg")


def _nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def validate_goal_noise(settings, *, candidate_separation_m=None):
    """Keep Gaussian scales explicit; a positional radius cannot reach a rival role."""
    if not isinstance(settings, dict) or settings.keys() - set(SCALES + BOUNDS):
        raise ValueError("goal_noise permits only z_std, translation_std, rotation_std, gripper_std, "
                         "translation_max_m and rotation_max_deg")
    values = {name: _nonnegative(settings.get(name, 0.), name) for name in SCALES}
    for name in BOUNDS:
        if name in settings:
            values[name] = _nonnegative(settings[name], name)
    if candidate_separation_m is not None:
        separation = _nonnegative(candidate_separation_m, "candidate_separation_m")
        if separation == 0:
            raise ValueError("candidate_separation_m must be a strictly positive audited minimum")
    if values["translation_std"]:
        radius = values.get("translation_max_m")
        if candidate_separation_m is None or radius is None or radius <= 0:
            raise ValueError("positive translation_std requires candidate_separation_m and positive translation_max_m")
        if radius >= float(candidate_separation_m) / 2:
            raise ValueError("translation_max_m must be strictly smaller than half candidate_separation_m")
    elif "translation_max_m" in values and candidate_separation_m is not None:
        if values["translation_max_m"] >= float(candidate_separation_m) / 2:
            raise ValueError("translation_max_m must be strictly smaller than half candidate_separation_m")
    angle = values.get("rotation_max_deg")
    if angle is not None and angle > 180:
        raise ValueError("rotation_max_deg must not exceed 180 degrees")
    if values["rotation_std"] and (angle is None or angle <= 0):
        raise ValueError("positive rotation_std (radians) requires positive rotation_max_deg")
    return values


def _bounded_vectors(shape, std, radius, generator):
    """Sample N(0,std² I) conditioned on ||x|| <= radius, without boundary atoms.

    A ball proposal avoids near-zero Gaussian rejection probabilities when the
    allowed radius is much smaller than std. Both proposals use the CPU RNG.
    """
    count = math.prod(shape[:-1])
    result = torch.empty((count, 3), dtype=torch.float64)
    pending = torch.arange(count)
    while pending.numel():
        proposal = torch.randn((pending.numel(), 3), dtype=torch.float64, generator=generator)
        if radius < std:
            norms = proposal.norm(dim=-1, keepdim=True)
            unit = proposal / norms.clamp_min(torch.finfo(torch.float64).tiny)
            distance = radius * torch.rand((pending.numel(), 1), dtype=torch.float64, generator=generator).pow(1. / 3.)
            proposal = unit * distance
            probability = torch.exp(-.5 * (proposal / std).square().sum(-1))
            accepted = (norms[:, 0] > 0) & (torch.rand(pending.numel(), dtype=torch.float64, generator=generator) < probability)
        else:
            proposal *= std
            accepted = proposal.norm(dim=-1) <= radius
        result[pending[accepted]] = proposal[accepted]
        pending = pending[~accepted]
    return result.reshape(shape)


def perturb_goal_bounded(goal: dict, *, z_std=0., translation_std=0., rotation_std=0., gripper_std=0.,
                         translation_max_m=None, rotation_max_deg=None, candidate_separation_m=None,
                         generator: torch.Generator | None = None) -> dict[str, Tensor]:
    """Independent detached goal noise; translation metres, rotation std radians.

    Rotation is a bounded axis-angle draw applied on the left in the goal frame.
    z is normalized before additive noise, as in the existing goal interface.
    Gripper noise is clipped to [0,1].
    """
    settings = dict(zip(SCALES, (z_std, translation_std, rotation_std, gripper_std)))
    settings.update({key: value for key, value in (("translation_max_m", translation_max_m),
                                                   ("rotation_max_deg", rotation_max_deg)) if value is not None})
    noise = validate_goal_noise(settings, candidate_separation_m=candidate_separation_m)
    if not isinstance(goal, dict) or not {"z", "goal_poses", "goal_gripper"} <= goal.keys():
        raise ValueError("goal must contain z, goal_poses and goal_gripper")
    validate_goal_poses(goal["goal_poses"])
    validate_gripper(goal["goal_gripper"], goal["goal_poses"].shape[:2])
    z = goal["z"]
    if (not isinstance(z, Tensor) or z.ndim != 3 or min(z.shape) < 1 or not z.is_floating_point()
            or not torch.isfinite(z).all() or z.shape[0] != goal["goal_poses"].shape[0]):
        raise ValueError("goal z must be finite floating [B,K,D] sharing the pose batch")
    if generator is not None and (not isinstance(generator, torch.Generator) or generator.device.type != "cpu"):
        raise ValueError("goal noise requires a CPU torch.Generator")
    z = F.normalize(z.detach().float(), dim=-1)
    poses = goal["goal_poses"].detach().clone()
    gripper = goal["goal_gripper"].detach().clone()
    if noise["z_std"]:
        z = z + noise["z_std"] * torch.randn(z.shape, generator=generator, device="cpu").to(z)
    if noise["translation_std"]:
        if noise["translation_max_m"] > torch.finfo(poses.dtype).max / 8:
            raise ValueError("translation_max_m exceeds the representable pose range")
        position = poses[..., :3, 3].clone().reshape(-1, 3)
        pending = torch.arange(len(position), device=poses.device)
        moved = position.clone()
        while pending.numel():
            offset = _bounded_vectors((pending.numel(), 3), noise["translation_std"], noise["translation_max_m"], generator)
            proposal = position[pending] + offset.to(poses)
            # Rounding after adding to a nonzero pose can otherwise cross the bound.
            accepted = (proposal.double() - position[pending].double()).norm(dim=-1) <= noise["translation_max_m"]
            moved[pending[accepted]] = proposal[accepted]
            pending = pending[~accepted]
        poses[..., :3, 3] = moved.reshape(poses[..., :3, 3].shape)
    if noise["rotation_std"]:
        vector = _bounded_vectors(poses[..., :3, 3].shape, noise["rotation_std"], math.radians(noise["rotation_max_deg"]), generator)
        x, y, z_axis = vector.unbind(-1)
        zero = torch.zeros_like(x)
        skew = torch.stack((zero, -z_axis, y, z_axis, zero, -x, -y, x, zero), -1).unflatten(-1, (3, 3))
        perturbation = torch.matrix_exp(skew).to(device=poses.device)
        rotation = perturbation @ poses[..., :3, :3].double()
        poses = poses.float()
        poses[..., :3, :3] = rotation.to(poses)
    if noise["gripper_std"]:
        offset = torch.randn(gripper.shape, generator=generator, device="cpu").to(gripper)
        gripper = (gripper + noise["gripper_std"] * offset).clamp(0, 1)
    if not torch.isfinite(z).all():
        raise ValueError("goal z noise exceeds the representable feature range")
    validate_goal_poses(poses)
    validate_gripper(gripper, poses.shape[:2])
    return {"z": z, "goal_poses": poses, "goal_gripper": gripper}


def _weighted_quantile(values, weights, quantile):
    ordered = sorted(zip(values, weights))
    total, threshold = 0., quantile * sum(weights)
    for value, weight in ordered:
        total += weight
        if total >= threshold:
            return float(value)
    return float(ordered[-1][0])


def calibrate_noise(manifest_path, output_path):
    """Turn held-out G residuals into explicit corruption settings, without a model.

    Each source group has equal mass, each pair equal mass within its group.
    q95 of worst token/effector residual divided by sqrt(dimension) defines a
    component scale, not a Gaussian fit or a calibrated confidence interval.
    Hard radii cover all observed residuals; unsafe positional residuals fail.
    """
    from .g_pi_calibration import _identity
    from .g_pi_controller import goal_distances
    from .g_pi_deployment import load_goal_prediction
    from .video_data import _local_path
    from .vision import sha256

    path, output = Path(manifest_path).resolve(), Path(output_path).resolve()
    if output.exists():
        raise ValueError("noise calibration output must be a new file")
    manifest = json.loads(path.read_text())
    required = {"format_version", "kind", "split", "encoder_identity", "registry", "records"}
    if (not isinstance(manifest, dict) or required - set(manifest)
            or set(manifest) - required - {"candidate_separation_m", "scale_quantile"}
            or type(manifest["format_version"]) is not int or manifest["format_version"] != 1
            or manifest["kind"] != "g_pi_noise_validation" or manifest["split"] != "validation"):
        raise ValueError("expected a version-1 validation-only g_pi_noise_validation manifest")
    identity = _identity(manifest["encoder_identity"])
    registry = manifest["registry"]
    required_registry = {"coordinate_frame", "pose_units", "pose_representation", "tool_frames", "end_effectors", "gripper_space"}
    if not isinstance(registry, dict) or required_registry - set(registry) or registry.get("pose_units") != "m":
        raise ValueError("noise calibration registry must declare metre-valued goal poses and all pose/gripper conventions")
    effectors = registry["end_effectors"]
    if (not isinstance(effectors, list) or not effectors
            or any(not isinstance(value, str) or not value.strip() for value in effectors)
            or len(set(effectors)) != len(effectors)):
        raise ValueError("noise calibration end_effectors must be ordered unique nonempty names")
    quantile = manifest.get("scale_quantile", .95)
    if type(quantile) not in (int, float) or not math.isfinite(quantile) or not 0 < quantile <= 1:
        raise ValueError("scale_quantile must lie in (0,1]")
    records = manifest["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("noise calibration requires held-out G prediction/reference pairs")
    rows, identifiers, pair_ids, group_counts = [], set(), set(), {}
    dependencies = {str(path): sha256(path)}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"sample_id", "source_group", "prediction", "reference"}:
            raise ValueError("noise records require sample_id, source_group, prediction and reference sidecars")
        for key, value in record.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"noise record {key} must be a nonempty string")
        if record["sample_id"] in identifiers:
            raise ValueError("noise calibration sample_id must be unique")
        identifiers.add(record["sample_id"])
        goals, pair = [], []
        for name in ("prediction", "reference"):
            goal_path = _local_path(path.parent, record[name], ".json")
            goal = load_goal_prediction(goal_path, encoder_identity=identity, registry=registry)
            goals.append(goal)
            meta = json.loads(goal_path.read_text())
            array_path = _local_path(goal_path.parent, meta["arrays"], ".npz")
            for dependency in (goal_path, array_path):
                dependencies[str(dependency)] = sha256(dependency)
            pair.append((str(goal_path), sha256(array_path)))
        pair_key = tuple(item[0] for item in pair)
        if pair_key in pair_ids:
            raise ValueError("repeated prediction/reference files cannot count as independent calibration pairs")
        pair_ids.add(pair_key)
        for goal in goals:
            norms = goal["z"].double().norm(dim=-1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=.01, rtol=0):
                raise ValueError("noise calibration z must already use normalized E/goal tokens")
        residual = goal_distances(*goals)
        rows.append({"sample_id": record["sample_id"], "source_group": record["source_group"], **residual})
        group_counts[record["source_group"]] = group_counts.get(record["source_group"], 0) + 1
    if len(group_counts) < 2:
        raise ValueError("noise calibration requires at least two independent validation source groups")
    weights = [1. / group_counts[row["source_group"]] for row in rows]
    scale = lambda key, divisor=1.: _weighted_quantile([row[key] / divisor for row in rows], weights, quantile)
    settings = {"z_std": scale("z", math.sqrt(identity["d_z"])),
                "translation_std": scale("position_m", math.sqrt(3.)),
                "rotation_std": math.radians(scale("rotation_deg", math.sqrt(3.))),
                "gripper_std": scale("gripper"),
                "translation_max_m": max(row["position_m"] for row in rows),
                "rotation_max_deg": max(row["rotation_deg"] for row in rows)}
    separation = manifest.get("candidate_separation_m")
    if settings["translation_max_m"]:
        if separation is None or settings["translation_max_m"] >= _nonnegative(separation, "candidate_separation_m") / 2:
            raise ValueError("validation G translation residual reaches half the candidate separation; "
                             "inspect wrong-object errors instead of silently clipping calibration")
    settings = validate_goal_noise(settings, candidate_separation_m=separation)
    config = {"goal_noise": settings}
    if separation is not None:
        config["candidate_separation_m"] = separation
    artifact = {"format_version": 1, "kind": "g_pi_noise_calibration", "split": "validation",
                "encoder_identity": identity, "registry": registry, "config": config,
                "units": {"z_std": "unit_token_coordinate", "translation_std": "m", "rotation_std": "rad",
                          "gripper_std": "closed_0_open_1", "translation_max_m": "m", "rotation_max_deg": "deg"},
                "method": {"name": "source_balanced_quantile_component_scale", "scale_quantile": quantile,
                           "hard_radius": "maximum_validation_residual", "translation_dimension": 3,
                           "rotation_dimension": 3, "z_dimension": identity["d_z"],
                           "confidence_calibration": False},
                "evidence": {"source_group_counts": group_counts, "residuals": rows},
                "validation_files": dependencies}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    return artifact


def calibrate_g_pi_noise(args):
    return calibrate_noise(args.manifest, args.output)
