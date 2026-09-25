"""Audited local HumanGen RoboTwin conversion; no downloads or inferred labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from .g_pi_context import validate_camera_layout
from .g_pi_data import EventRules, load_g_pi_index, load_g_pi_sample, validate_latent_grid
from .g_pi_subgoals import resolve_subgoal_indices
from .goal_interface import validate_se3
from .icl_data import LATENT_NORMALIZATION
from .robotwin import RobotwinActionTransform, USED_CHANNELS
from .video_data import SPLITS, _source_components
from .vision import sha256


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty audited string")
    return value


def _schema(value, required, optional, name):
    if not isinstance(value, dict) or required - set(value) or set(value) - required - optional:
        raise ValueError(f"{name} requires {sorted(required)} and only optional {sorted(optional)}")


def _path(parent, name, suffix=None):
    _text(name, "local input path")
    if "://" in name:
        raise ValueError("HumanGen conversion reads local inputs only; download separately with a size limit")
    path = (parent / name).resolve()
    if not path.is_file() or (suffix is not None and path.suffix != suffix):
        raise ValueError(f"missing local {suffix or 'input'} file: {path}")
    return path


def _hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError(f"{name} must be a SHA256 digest")
    return value


def _arrays(path):
    with np.load(path, allow_pickle=False) as data:
        if len(data.files) != len(set(data.files)):
            raise ValueError("NPZ input has repeated array names")
        return {name: data[name].copy() for name in data.files}


def read_humangen_episode(path, fields):
    """Read published LeRobot columns, or a local numeric NPZ with those columns."""
    _schema(fields, {"state", "action", "timestamp"}, set(), "raw fields")
    for key, value in fields.items():
        _text(value, key)
    path = Path(path)
    if path.suffix == ".npz":
        values = _arrays(path)
    elif path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ValueError("reading HumanGen Parquet requires already-installed pyarrow; "
                             "supply an audited numeric NPZ instead (no dependency is installed)") from error
        table = pq.read_table(path, columns=list(fields.values()))
        values = {name: np.asarray(table[name].to_pylist()) for name in fields.values()}
    else:
        raise ValueError("raw episode must be a local .npz or .parquet file")
    if set(fields.values()) - set(values):
        raise ValueError("raw episode is missing declared state/action/timestamp columns")
    arrays = {name: np.asarray(values[field]) for name, field in fields.items()}
    count = len(arrays["timestamp"])
    for name, shape in (("state", (count, 16)), ("action", (count, 16)), ("timestamp", (count,))):
        value = arrays[name]
        if (value.shape != shape or not np.issubdtype(value.dtype, np.floating)
                or not np.isfinite(value).all() or count < 2):
            raise ValueError(f"raw {name} must be finite floating {shape} with at least two control rows")
    return arrays


def read_humangen_latent(path):
    """Unpack published model-normalized (f h w)c latents, discarding all text."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"latent", "latent_num_frames", "latent_height", "latent_width", "frame_ids", "ori_fps"}
    if not isinstance(payload, dict) or required - set(payload):
        raise ValueError("published HumanGen latent needs spatial/time dimensions, frame_ids and ori_fps")
    dimensions = [payload[key] for key in ("latent_num_frames", "latent_height", "latent_width")]
    if any(type(value) is not int or value < 1 for value in dimensions):
        raise ValueError("published latent dimensions must be positive integers")
    frames, height, width = dimensions
    latent = payload["latent"]
    if (not isinstance(latent, torch.Tensor) or not latent.is_floating_point()
            or not torch.isfinite(latent).all() or latent.numel() == 0):
        raise ValueError("published latent must be a nonempty finite floating tensor")
    if latent.ndim == 2 and latent.shape[0] == frames * height * width:
        latent = latent.reshape(frames, height, width, -1).permute(3, 0, 1, 2)
    if latent.ndim != 4 or tuple(latent.shape[1:]) != tuple(dimensions):
        raise ValueError("published latent shape must match [C,F,H,W] or [(F H W),C]")
    frame_ids = payload["frame_ids"]
    if (not isinstance(frame_ids, list) or not frame_ids or any(type(i) is not int or i < 0 for i in frame_ids)
            or frame_ids != sorted(set(frame_ids)) or (len(frame_ids) - 1) // 4 + 1 != frames):
        raise ValueError("published frame_ids must be ordered raw indices on the first-then-four causal grid")
    fps = payload["ori_fps"]
    if type(fps) not in (int, float) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("published ori_fps must be finite and positive")
    times = np.asarray(frame_ids[::4], dtype=np.float64) / fps
    return {"latent": latent.contiguous().float().numpy(), "frame_times": times}, {
        "source_sha256": sha256(path), "frame_ids": frame_ids, "ori_fps": fps,
        "text_fields_discarded": sorted(set(payload) & {"text", "text_emb", "task", "task_emb", "local_instruction", "local_instruction_emb"}),
        "latent_normalization": LATENT_NORMALIZATION}


def load_humangen_robot_latents(paths, control_times, *, frame_stride):
    """Keep caller's camera order; prove each cached frame was already available."""
    if not isinstance(paths, (list, tuple)) or not paths:
        raise ValueError("robot latents require one published .pth per ordered camera")
    times = np.asarray(control_times)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("robot control_times must contain every finite control step")
    dt = float(times[1] - times[0])
    if type(frame_stride) is not int or frame_stride < 1:
        raise ValueError("frame_stride must be a positive integer")
    videos, audits = [], []
    for path in paths:
        arrays, audit = read_humangen_latent(path)
        metadata = {"control_dt": dt, "frame_stride": frame_stride, "temporal_down_rate": 4,
                    "actions_per_frame": 4 * frame_stride, "alignment": "zerowam_causal_first_then_four"}
        validate_latent_grid(metadata, torch.from_numpy(times), torch.from_numpy(arrays["frame_times"]))
        ids = np.asarray(audit["frame_ids"])
        if ids[0] != 0 or np.any(np.diff(ids) != frame_stride) or ids[-1] >= len(times):
            raise ValueError("published robot frame_ids must start at zero and sample the control grid at frame_stride")
        if videos and (arrays["latent"].shape[:3] != videos[0].shape[:3]
                       or audit["frame_ids"] != audits[0]["frame_ids"]):
            raise ValueError("all robot cameras must have exactly matching causal coverage, channels and height")
        videos.append(arrays["latent"])
        audits.append(audit)
    return {"latent": np.concatenate(videos, axis=-1), "latent_available_times": arrays["frame_times"]}, audits


def audit_humangen(manifest_path, output_path):
    """Check real local published inputs without pretending missing labels are known."""
    path, output = Path(manifest_path).resolve(), Path(output_path).resolve()
    spec = json.loads(path.read_text())
    required = {"format_version", "kind", "fields", "control_dt", "frame_stride", "episodes"}
    _schema(spec, required, {"info"}, "humangen_robotwin_audit")
    if spec["format_version"] != 1 or spec["kind"] != "humangen_robotwin_audit":
        raise ValueError("expected version-1 humangen_robotwin_audit")
    if type(spec["control_dt"]) not in (int, float) or not np.isfinite(spec["control_dt"]) or spec["control_dt"] <= 0:
        raise ValueError("audit control_dt must be finite and positive")
    if not isinstance(spec["episodes"], list) or not spec["episodes"]:
        raise ValueError("audit requires at least one local episode")
    info_identity = None
    if "info" in spec:
        info_path = _path(path.parent, spec["info"], ".json")
        info = json.loads(info_path.read_text())
        fps = info.get("fps")
        if (type(fps) not in (int, float) or not np.isfinite(fps) or fps <= 0
                or not np.isclose(1. / fps, spec["control_dt"], rtol=0, atol=1e-9)):
            raise ValueError("published info fps differs from the audited control period")
        for key in ("state", "action"):
            feature = info.get("features", {}).get(spec["fields"][key], {})
            if feature.get("shape") != [16]:
                raise ValueError("published info must declare the audited 16D state and action fields")
        info_identity = {"path": str(info_path), "sha256": sha256(info_path), "fps": fps}
    reports = []
    for entry in spec["episodes"]:
        _schema(entry, {"episode_id", "raw", "robot_latents", "human_latent"}, {"original_parquet", "raw_extraction_evidence"}, "audit episode")
        raw_path = _path(path.parent, entry["raw"])
        raw = read_humangen_episode(raw_path, spec["fields"])
        times = np.arange(len(raw["timestamp"]), dtype=np.float64) * spec["control_dt"]
        if not np.allclose(raw["timestamp"], times, atol=min(1e-6, spec["control_dt"] * 1e-4), rtol=0):
            raise ValueError("raw timestamp grid differs from the declared control period")
        robot_paths = [_path(path.parent, filename, ".pth") for filename in entry["robot_latents"]]
        history, camera_audits = load_humangen_robot_latents(robot_paths, times, frame_stride=spec["frame_stride"])
        demo, demo_audit = read_humangen_latent(_path(path.parent, entry["human_latent"], ".pth"))
        if history["latent"].shape[0] != demo["latent"].shape[0]:
            raise ValueError("published robot/human latent channel counts differ")
        exact_shift = np.array_equal(raw["action"][:-1], raw["state"][1:])
        report = {"episode_id": entry["episode_id"], "raw_sha256": sha256(raw_path), "control_rows": len(times),
                  "robot_history_shape": list(history["latent"].shape), "human_shape": list(demo["latent"].shape),
                  "latent_available_times": history["latent_available_times"].tolist(), "terminal_time": float(times[-1]),
                  "action_equals_next_recorded_state": exact_shift,
                  "gripper_ranges": [[float(raw["state"][:, i].min()), float(raw["state"][:, i].max())] for i in (7, 15)],
                  "camera_audits": camera_audits, "human_audit": demo_audit}
        if "original_parquet" in entry:
            _text(entry.get("raw_extraction_evidence"), "raw_extraction_evidence")
            original = _path(path.parent, entry["original_parquet"], ".parquet")
            report.update(original_parquet_sha256=sha256(original), raw_extraction_evidence=entry["raw_extraction_evidence"])
        reports.append(report)
    result = {"format_version": 1, "kind": "humangen_robotwin_preflight", "episodes": reports,
              "training_ready": False, "model_loaded": False,
              "info_identity": info_identity, "gripper_signal_source": "unknown",
              "gripper_source_evidence": "The audited schema and numeric columns do not establish sensor versus command provenance; collector evidence is required.",
              "requires_audit": ["measured versus command gripper source (field names and shifted arrays do not prove it)",
                                 "endpoint measurement, quaternion order and source-to-robot-base calibration",
                                 "per-episode success and task-level human pairing/generation ancestry",
                                 "matching action normalization and VAE identities"],
              "requires_preprocessing": ["independently encode every confirmed event and terminal RGB image with the pinned VAE",
                                         "supply audited goal_language caches for pi training"],
              "note": "History latents cannot substitute for single-image subgoal latents; no g_pi_task was written."}
    if output.exists():
        raise ValueError("audit output must be a new file")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def _canonical_pose(values, convention):
    """Canonicalize published left xyz-quat-grip/right xyz-quat-grip vectors."""
    from scipy.spatial.transform import Rotation

    result = values.astype(np.float64).copy()
    for start in (0, 8):
        result[:, start:start + 3] *= .001 if convention["translation_units"] == "mm" else 1.
        quaternion = result[:, start + 3:start + 7]
        if convention["quaternion_order"] == "wxyz":
            quaternion = quaternion[:, [1, 2, 3, 0]]
        norm = np.linalg.norm(quaternion, axis=1)
        if np.any(np.abs(norm - 1.) > .01):
            raise ValueError("raw quaternions must be nonzero and unit within 0.01; verify quaternion fields/order")
        result[:, start + 3:start + 7] = Rotation.from_quat(quaternion).as_quat()
    return result


def _pose_labels(state, convention):
    from scipy.spatial.transform import Rotation

    poses = np.tile(np.eye(4), (len(state), 2, 1, 1))
    base = np.asarray(convention["base_from_source"], dtype=np.float64)
    validate_se3(torch.from_numpy(base))
    for effector, start in enumerate((0, 8)):
        poses[:, effector, :3, :3] = Rotation.from_quat(state[:, start + 3:start + 7]).as_matrix()
        poses[:, effector, :3, 3] = state[:, start:start + 3]
    poses = base @ poses
    validate_se3(torch.from_numpy(poses))
    return poses.astype(np.float32)


def _validate_spec(spec):
    required = {"format_version", "kind", "repository", "fields", "pose_convention", "gripper_signal",
                "normalization", "robot", "event_rules", "episodes"}
    _schema(spec, required, set(), "humangen_robotwin_conversion")
    if type(spec["format_version"]) is not int or spec["format_version"] != 1 or spec["kind"] != "humangen_robotwin_conversion":
        raise ValueError("expected version-1 humangen_robotwin_conversion")
    _schema(spec["repository"], {"id", "revision", "evidence"}, set(), "repository")
    for key, value in spec["repository"].items():
        _text(value, f"repository {key}")
    pose = spec["pose_convention"]
    _schema(pose, {"quaternion_order", "translation_units", "source_frame", "coordinate_frame", "base_from_source",
                   "tool_frames", "pose_source", "evidence"}, set(), "pose_convention")
    if pose["quaternion_order"] not in {"xyzw", "wxyz"} or pose["translation_units"] not in {"m", "mm"}:
        raise ValueError("pose_convention must explicitly declare xyzw/wxyz and m/mm")
    if pose["pose_source"] != "measured_endpoint":
        raise ValueError("HumanGen goal poses require audited measured_endpoint; controller targets cannot substitute")
    for key in ("source_frame", "coordinate_frame", "evidence"):
        _text(pose[key], key)
    if (not isinstance(pose["tool_frames"], list) or len(pose["tool_frames"]) != 2
            or len(set(pose["tool_frames"])) != 2):
        raise ValueError("pose_convention tool_frames must identify left and right endpoint frames")
    for frame in pose["tool_frames"]:
        _text(frame, "tool frame")
    base = np.asarray(pose["base_from_source"], dtype=np.float64)
    if base.shape != (4, 4):
        raise ValueError("base_from_source must be a calibrated [4,4] rigid transform")
    validate_se3(torch.from_numpy(base))
    grip = spec["gripper_signal"]
    _schema(grip, {"source", "field", "indices", "closed", "open", "units", "normalization_id", "evidence"}, set(), "gripper_signal")
    if grip["source"] not in {"measured", "command"} or grip["field"] != spec["fields"]["state"] or grip["indices"] != [7, 15]:
        raise ValueError("gripper_signal must audit measured/command provenance of state columns [7,15]")
    for key in ("units", "normalization_id", "evidence"):
        _text(grip[key], key)
    for key in ("closed", "open"):
        value = grip[key]
        if not isinstance(value, list) or len(value) != 2 or any(type(x) not in (int, float) or not np.isfinite(x) for x in value):
            raise ValueError("gripper closed/open must contain two finite raw endpoint values")
    if any(a == b for a, b in zip(grip["closed"], grip["open"])):
        raise ValueError("gripper closed/open calibration endpoints must differ")
    rules = EventRules.from_metadata(spec["event_rules"])
    if rules.signal_source != grip["source"]:
        raise ValueError("event_rules and gripper_signal must use the same measured/command source")
    _schema(spec["normalization"], {"path", "sha256"}, set(), "normalization")
    _hash(spec["normalization"]["sha256"], "normalization sha256")
    robot = spec["robot"]
    _schema(robot, {"control_dt", "frame_stride", "action_frames", "state_space_id", "feature_space_id", "camera_layout", "patch_size"}, set(), "robot")
    for key in ("state_space_id", "feature_space_id"):
        _text(robot[key], key)
    for key in ("frame_stride", "action_frames"):
        if type(robot[key]) is not int or robot[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(robot["control_dt"]) not in (int, float) or not np.isfinite(robot["control_dt"]) or robot["control_dt"] <= 0:
        raise ValueError("control_dt must be a positive finite control period")
    validate_camera_layout(robot["camera_layout"])
    if (not isinstance(robot["patch_size"], list) or len(robot["patch_size"]) != 2
            or any(type(x) is not int or x < 1 for x in robot["patch_size"])):
        raise ValueError("patch_size must explicitly declare positive native spatial patch height/width")
    if not isinstance(spec["episodes"], list) or not spec["episodes"]:
        raise ValueError("conversion requires at least one explicitly successful episode")


def _visual_cache(path, raw_path, robot, indices):
    cache = json.loads(path.read_text())
    required = {"format_version", "kind", "arrays", "arrays_sha256", "raw_sha256", "feature_space_id",
                "latent_normalization", "frame_stride", "camera_layout", "patch_size", "subgoal_encoding",
                "subgoal_control_indices", "vae_identity", "evidence"}
    _schema(cache, required, set(), "humangen_robotwin_visuals")
    if type(cache["format_version"]) is not int or cache["format_version"] != 1 or cache["kind"] != "humangen_robotwin_visuals":
        raise ValueError("expected version-1 humangen_robotwin_visuals cache")
    _text(cache["evidence"], "visual cache evidence")
    if cache["raw_sha256"] != sha256(raw_path):
        raise ValueError("visual cache raw episode identity mismatch")
    for key in ("feature_space_id", "frame_stride", "camera_layout", "patch_size"):
        if cache[key] != robot[key]:
            raise ValueError(f"visual cache {key} mismatch")
    if cache["latent_normalization"] != LATENT_NORMALIZATION or cache["subgoal_encoding"] != "wan_vae_single_frame":
        raise ValueError("visual cache must use normalized Wan sequence latents and independent single-frame targets")
    if cache["subgoal_control_indices"] != indices.tolist():
        raise ValueError("visual cache subgoal_control_indices mismatch; independently re-encode the confirmed boundaries")
    if not isinstance(cache["vae_identity"], dict) or not cache["vae_identity"]:
        raise ValueError("visual cache must declare the VAE identity")
    for key, value in cache["vae_identity"].items():
        _text(key, "VAE identity file")
        _hash(value, "VAE identity hash")
    archive = _path(path.parent, cache["arrays"], ".npz")
    if sha256(archive) != _hash(cache["arrays_sha256"], "visual arrays hash"):
        raise ValueError("visual cache arrays identity mismatch")
    arrays = _arrays(archive)
    if set(arrays) != {"latent", "latent_available_times", "subgoal_latents", "subgoal_times"}:
        raise ValueError("visual cache requires exactly latent, latent_available_times, subgoal_latents and subgoal_times")
    latent = arrays["latent"]
    if (latent.ndim != 4 or min(latent.shape) < 1
            or latent.shape[-1] != sum(view["token_width"] for view in robot["camera_layout"]) * robot["patch_size"][1]
            or latent.shape[-2] % robot["patch_size"][0]):
        raise ValueError("visual cache spatial grid does not match the declared camera boundaries/native patches")
    return arrays, cache


def _episode(spec, entry, parent, folder, transform):
    required = {"sample_id", "split", "robot_source_id", "trajectory_id", "raw", "success", "success_evidence", "visual_cache", "human"}
    _schema(entry, required, {"language", "subgoal_source", "subgoal_annotation"}, "episode")
    for name in ("sample_id", "robot_source_id", "trajectory_id", "success_evidence"):
        _text(entry[name], name)
    if entry["split"] not in SPLITS or entry["success"] is not True:
        raise ValueError("each episode requires a train/validation/test split and explicitly verified success")
    if entry.get("subgoal_source", "gripper") not in {"gripper", "candidate_match"}:
        raise ValueError("HumanGen permits only weak gripper or candidate_match boundaries; no object-state labels")
    grip, robot = spec["gripper_signal"], spec["robot"]
    if entry.get("subgoal_source") == "candidate_match" and grip["source"] != "measured":
        raise ValueError("candidate_match requires measured gripper widths; commands cannot establish blocked closure")
    raw_path = _path(parent, entry["raw"])
    raw = read_humangen_episode(raw_path, spec["fields"])
    count = len(raw["timestamp"])
    times = np.arange(count, dtype=np.float64) * robot["control_dt"]
    if not np.allclose(raw["timestamp"], times, atol=min(1e-6, robot["control_dt"] * 1e-4), rtol=0):
        raise ValueError("raw timestamps must start at zero and cover every control row; gaps/reordering cannot be resampled")
    state = _canonical_pose(raw["state"], spec["pose_convention"])
    action = _canonical_pose(raw["action"], spec["pose_convention"])
    gripper = (state[:, [7, 15]] - np.asarray(grip["closed"])) / (np.asarray(grip["open"]) - np.asarray(grip["closed"]))
    if not np.isfinite(gripper).all() or np.any(gripper < 0) or np.any(gripper > 1):
        raise ValueError("raw gripper values fall outside audited closed/open calibration; do not silently clip")
    arrays = {"control_times": times, "states": state.astype(np.float32),
              "poses": _pose_labels(state, spec["pose_convention"]), "gripper": gripper.astype(np.float32)}
    # Same relative pose, 30-channel map, quantiles and [-2,2] training clipping as Zero-WAM.
    actions = np.stack([transform.from_absolute(row, state[0]) for row in action]).clip(-2., 2.)
    arrays.update(actions=actions.T.astype(np.float32), actions_mask=np.broadcast_to(
        np.isin(np.arange(30), USED_CHANNELS)[:, None], (30, count)).copy())
    source = entry.get("subgoal_source", "gripper")
    subgoal_metadata = {"event_rules": spec["event_rules"], "subgoal_source": source,
                        "gripper_signal_source": grip["source"], "gripper_source_evidence": grip["evidence"]}
    if "subgoal_annotation" in entry:
        subgoal_metadata["subgoal_annotation"] = entry["subgoal_annotation"]
    indices, audit = resolve_subgoal_indices(subgoal_metadata, {name: torch.from_numpy(value) for name, value in arrays.items()})
    if audit["weak_label"] is not True:
        raise ValueError("HumanGen subgoal annotations must remain explicitly weak labels")
    visual_path = _path(parent, entry["visual_cache"], ".json")
    visual, visual_meta = _visual_cache(visual_path, raw_path, robot, indices)
    arrays.update(visual)
    alignment = {"frame_stride": robot["frame_stride"], "temporal_down_rate": 4,
                 "actions_per_frame": 4 * robot["frame_stride"], "control_dt": robot["control_dt"],
                 "alignment": "zerowam_causal_first_then_four", "subgoal_encoding": "wan_vae_single_frame"}
    validate_latent_grid(alignment, torch.from_numpy(times), torch.from_numpy(visual["latent_available_times"]))
    human = entry["human"]
    _schema(human, {"source_id", "source_group", "video_id", "latent", "pairing_evidence", "origin_evidence"},
            {"origin_robot_trajectory_id", "video_sha256"}, "human source")
    for key in set(human) - {"video_sha256", "latent"}:
        _text(human[key], f"human {key}")
    if "video_sha256" in human:
        _hash(human["video_sha256"], "human video SHA256")
    human_path = _path(parent, human["latent"])
    if human_path.suffix == ".pth":
        demo, _ = read_humangen_latent(human_path)
    elif human_path.suffix == ".npz":
        demo = _arrays(human_path)
    else:
        raise ValueError("human latent must be a local .npz or published .pth")
    if set(demo) != {"latent", "frame_times"}:
        raise ValueError("human latent archive must contain only latent and frame_times; no labels or text")
    folder.mkdir()
    np.savez_compressed(folder / "robot.npz", **arrays)
    np.savez_compressed(folder / "human.npz", **demo)
    meta = {"format_version": 3, "kind": "g_pi_task", "sample_id": entry["sample_id"], "arrays": "robot.npz",
            "robot_source": {"source_id": entry["robot_source_id"], "source_group": entry["robot_source_id"],
                             "domain": "robot", "trajectory_id": entry["trajectory_id"]},
            "action_space": {"representation": "zero-wam-normalized", "normalization_id": transform.normalization_id,
                             "dimension": 30, "valid_channels": np.isin(np.arange(30), USED_CHANNELS).tolist()},
            "state_space_id": robot["state_space_id"], "coordinate_frame": spec["pose_convention"]["coordinate_frame"],
            "pose_units": "m", "goal_source": "measured_endpoint", "end_effectors": ["left", "right"],
            "pose_representation": "absolute_robot_base_tool", "tool_frames": spec["pose_convention"]["tool_frames"],
            "gripper_space": {key: grip[key] for key in ("normalization_id", "closed", "open", "units")},
            "gripper_signal_source": grip["source"], "gripper_source_evidence": grip["evidence"],
            "feature_space_id": robot["feature_space_id"], "latent_normalization": LATENT_NORMALIZATION,
            "action_frames": robot["action_frames"], "task_start_time": 0., "success": True,
            "event_rules": spec["event_rules"], "subgoal_source": source, "subgoal_annotation": audit, **alignment,
            "demonstration": {"source_id": human["source_id"], "source_group": human["source_group"],
                              "domain": "human", "arrays": "human.npz"},
            "compatibility": {"kind": "audited_semantic_task", "evidence": human["pairing_evidence"]},
            "provenance": {"converter": "humangen_robotwin_v1", "repository": spec["repository"],
                           "raw_sha256": sha256(raw_path), "raw_columns": spec["fields"],
                           "pose_convention": spec["pose_convention"], "gripper_signal": grip,
                           "success_evidence": entry["success_evidence"], "human_source": {k: v for k, v in human.items() if k != "latent"},
                           "human_latent_sha256": sha256(human_path),
                           "visual_cache_sha256": sha256(visual_path), "vae_identity": visual_meta["vae_identity"],
                           "camera_layout": robot["camera_layout"], "patch_size": robot["patch_size"],
                           "action_reference": "episode_initial_source_frame_xyzw",
                           "action_clipping": [-2., 2.], "terminal_action_masked_by_loader": True}}
    if "language" in entry:
        from .goal_language import _language_metadata, load_goal_language

        language = _path(parent, entry["language"], ".json")
        load_goal_language(language)
        language_meta = _language_metadata(language)
        archive = _path(language.parent, language_meta["arrays"], ".npz")
        shutil.copyfile(archive, folder / "language.npz")
        language_meta["arrays"] = "language.npz"
        (folder / "language.json").write_text(json.dumps(language_meta, indent=2) + "\n")
        meta["language"] = "language.json"
    manifest = folder / "task.json"
    manifest.write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
    load_g_pi_sample(manifest, route="g_translator", generator=torch.Generator().manual_seed(0))
    if "language" in meta:
        load_g_pi_sample(manifest, route="pi_goal", generator=torch.Generator().manual_seed(0))
    # Raw video identity and parent trajectories close aliases that file names cannot catch.
    aliases = [{"source_id": human["source_id"], "source_group": f"humangen-video:{human['video_id']}", "domain": "human", "split": entry["split"]},
               {"source_id": entry["robot_source_id"], "source_group": f"sha256:robot-raw:{sha256(raw_path)}", "domain": "robot", "trajectory_id": entry["trajectory_id"], "split": entry["split"]}]
    if "video_sha256" in human:
        aliases.append({"source_id": human["source_id"], "source_group": f"sha256:human-video:{human['video_sha256']}", "domain": "human", "split": entry["split"]})
    digest = hashlib.sha256()
    for key in ("latent", "frame_times"):
        value = np.ascontiguousarray(demo[key])
        digest.update(str((key, value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    aliases.append({"source_id": human["source_id"], "source_group": f"sha256:human-latent:{digest.hexdigest()}", "domain": "human", "split": entry["split"]})
    bridges = []
    if "origin_robot_trajectory_id" in human:
        bridges.append({"source_id": human["source_id"], "trajectory_id": human["origin_robot_trajectory_id"], "split": entry["split"]})
    return meta, aliases, bridges


def convert_humangen(manifest_path, output_path):
    """Convert audited successful episodes atomically, retaining source split closure."""
    path, output = Path(manifest_path).resolve(), Path(output_path).resolve()
    spec = json.loads(path.read_text())
    _validate_spec(spec)
    if output.exists():
        raise ValueError("HumanGen conversion output must be a new directory")
    stats = _path(path.parent, spec["normalization"]["path"], ".json")
    transform = RobotwinActionTransform.from_stats(stats, expected_sha256=spec["normalization"]["sha256"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".humangen-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        index = {"format_version": 1, "kind": "g_pi_index", "samples": [], "source_aliases": [], "bridge_sources": []}
        records, identifiers, language_missing = [], set(), []
        for number, entry in enumerate(spec["episodes"]):
            folder = staging / f"episode_{number:06d}"
            meta, aliases, bridges = _episode(spec, entry, path.parent, folder, transform)
            if meta["sample_id"] in identifiers:
                raise ValueError("conversion sample_id must be unique")
            identifiers.add(meta["sample_id"])
            index["samples"].append({"manifest": f"{folder.name}/task.json", "split": entry["split"]})
            index["source_aliases"].extend(aliases)
            index["bridge_sources"].extend(bridges)
            records.extend({**source, "record_kind": "video", "split": entry["split"]}
                           for source in (meta["robot_source"], meta["demonstration"]))
            records.extend({**alias, "record_kind": "video"} for alias in aliases)
            records.extend({**bridge, "record_kind": "bridge"} for bridge in bridges)
            if "language" not in meta:
                language_missing.append(meta["sample_id"])
        _source_components(records)
        index_path = staging / "index.json"
        index_path.write_text(json.dumps(index, indent=2) + "\n")
        for split in {entry["split"] for entry in spec["episodes"]}:
            load_g_pi_index(index_path, split)
        report = {"format_version": 1, "kind": "humangen_conversion_report", "episodes": len(identifiers),
                  "manifest_sha256": sha256(path), "normalization_sha256": sha256(stats),
                  "gripper_signal_source": spec["gripper_signal"]["source"], "weak_labels": True,
                  "missing_pi_language": language_missing, "source_components": len(_source_components(records)),
                  "model_loaded": False}
        (staging / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(output)
    return {**report, "index": str(output / "index.json")}


def convert_humangen_g_pi(args):
    return convert_humangen(args.manifest, args.output)


def audit_humangen_g_pi(args):
    return audit_humangen(args.manifest, args.output)
