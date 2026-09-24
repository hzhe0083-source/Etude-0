"""Observed-only offline inputs and separately exported G/π predictions."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch

from .cli import file_sha256, write_json
from .goal_data import _interface_metadata, _validate_state
from .goal_interface import validate_goal_poses, validate_gripper
from .goal_language import load_goal_language
from .icl_data import LATENT_NORMALIZATION, _arrays, _identity, _IDENTITY_FIELDS
from .video_data import _local_path, _text


@dataclass(frozen=True)
class PiObservation:
    metadata: dict
    state: torch.Tensor
    history: torch.Tensor
    history_times: torch.Tensor
    language: torch.Tensor | None
    language_identity: dict | None
    goal: dict


@dataclass(frozen=True)
class GObservation:
    metadata: dict
    state: torch.Tensor
    history: torch.Tensor
    history_times: torch.Tensor
    demonstration: torch.Tensor


def _goal_registry(registry):
    keys = ("coordinate_frame", "pose_units", "pose_representation", "tool_frames",
            "end_effectors", "gripper_space")
    return {key: registry[key] for key in keys}


def save_goal_prediction(path, goal, encoder_identity, registry):
    """Write G's goal and a checked identity sidecar accepted by π."""
    path = Path(path)
    sidecar = path.with_suffix(".json")
    if path.suffix != ".npz" or path.exists() or sidecar.exists():
        raise ValueError("goal prediction requires fresh .npz and .json paths")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{name: value.detach().float().cpu().numpy() for name, value in goal.items()})
    write_json(sidecar, {"format_version": 1, "kind": "g_pi_goal", "arrays": path.name,
        "arrays_sha256": file_sha256(path), "encoder_identity": encoder_identity,
        "registry": _goal_registry(registry)})
    return sidecar


def load_goal_prediction(path, *, encoder_identity, registry):
    path = Path(path)
    metadata = json.loads(path.read_text())
    required = {"format_version", "kind", "arrays", "arrays_sha256", "encoder_identity", "registry"}
    if (not isinstance(metadata, dict) or set(metadata) != required or metadata["format_version"] != 1
            or metadata["kind"] != "g_pi_goal"):
        raise ValueError("expected a version-1 g_pi_goal identity sidecar")
    if metadata["encoder_identity"] != encoder_identity:
        raise ValueError("goal E identity differs from the policy encoder identity")
    if metadata["registry"] != _goal_registry(registry):
        raise ValueError("goal pose/gripper coordinate conventions differ from the policy")
    arrays = _local_path(path.parent, metadata["arrays"], ".npz")
    if file_sha256(arrays) != metadata["arrays_sha256"]:
        raise ValueError("goal arrays checksum mismatch")
    with np.load(arrays, allow_pickle=False) as archive:
        if len(archive.files) != 3 or set(archive.files) != {"z", "goal_poses", "goal_gripper"}:
            raise ValueError("goal NPZ needs exactly z, goal_poses and goal_gripper")
        goal = {key: torch.from_numpy(archive[key].copy()) for key in archive.files}
    z = goal["z"]
    if (z.shape != (1, encoder_identity["k_z"], encoder_identity["d_z"])
            or not z.is_floating_point() or not torch.isfinite(z).all()):
        raise ValueError("goal z must be finite floating [1,K_z,d_z]")
    validate_goal_poses(goal["goal_poses"])
    if goal["goal_poses"].shape[:2] != (1, len(registry["end_effectors"])):
        raise ValueError("goal poses must follow the policy end_effectors order")
    validate_gripper(goal["goal_gripper"], goal["goal_poses"].shape[:2])
    return goal


def load_g_pi_observation(path, payload):
    """Reject supervision fields; π never loads or returns a demonstration."""
    from .g_pi_data import validate_latent_grid

    path = Path(path)
    metadata = json.loads(path.read_text())
    if isinstance(metadata, dict) and metadata.get("format_version") in (1, 2):
        raise ValueError(f"g_pi_observation version {metadata['format_version']} is unsupported; rebuild version 3 for the 2D goal grid")
    route, registry = payload["config"]["interface_type"], payload["registry"]
    common = {"format_version", "kind", "arrays", "feature_space_id", "latent_normalization",
        "action_space", "current_time", "control_dt", "actions_per_frame", "state_space_id",
        "coordinate_frame", "pose_units", "end_effectors", "pose_representation", "tool_frames",
        "gripper_space", "frame_stride", "temporal_down_rate", "alignment", "subgoal_encoding"}
    extra = {"demonstration"} if route == "g_translator" else {"goal"}
    optional = {"provenance"} | ({"language"} if route == "pi_goal" else set())
    if (route not in {"g_translator", "pi_goal"} or not isinstance(metadata, dict)
            or (common | extra) - metadata.keys() or metadata.keys() - common - extra - optional
            or type(metadata.get("format_version")) is not int or metadata["format_version"] != 3
            or metadata["kind"] != "g_pi_observation"):
        raise ValueError("expected route-specific version-3 g_pi_observation without supervision")
    _interface_metadata(metadata)
    if (type(metadata["actions_per_frame"]) is not int or metadata["actions_per_frame"] < 1
            or metadata["latent_normalization"] != LATENT_NORMALIZATION):
        raise ValueError("observation needs positive actions_per_frame and native latent_normalization")
    for name in registry.keys() - {"goal_source", "language_identity", "event_rules", "subgoal_source"}:
        if metadata.get(name) != registry[name]:
            raise ValueError(f"observed {name} differs from the trained interface")
    if _text(metadata, "feature_space_id") != payload["visual_feature_space"]:
        raise ValueError("observed visual encoder differs from the trained interface")
    with np.load(_local_path(path.parent, _text(metadata, "arrays"), ".npz"), allow_pickle=False) as archive:
        if len(archive.files) != 3 or set(archive.files) != {"state", "history_latent", "latent_available_times"}:
            raise ValueError("observation NPZ needs exactly state, history_latent and latent_available_times")
        state, history, times = (torch.from_numpy(archive[key].copy())
                                 for key in ("state", "history_latent", "latent_available_times"))
    _validate_state(state)
    if (history.ndim != 4 or min(history.shape) < 1 or not history.is_floating_point()
            or not torch.isfinite(history).all()):
        raise ValueError("history_latent must be finite floating [C,T,H,W]")
    layout = payload["encoder_identity"]["camera_layout"]
    patch_width = payload["native_config"]["patch_size"][2]
    if history.shape[-1] != sum(view["token_width"] for view in layout) * patch_width:
        raise ValueError("observed canvas width must match the policy camera_layout in native patch tokens")
    current_time, dt = metadata["current_time"], metadata["control_dt"]
    tolerance = min(1e-6, dt * 1e-4)
    control_index = round(current_time / dt)
    if current_time < 0 or not math.isclose(control_index * dt, current_time, abs_tol=tolerance, rel_tol=0):
        raise ValueError("current_time must lie on the task-local control grid")
    if (times.shape != (history.shape[1],) or not times.is_floating_point()
            or not torch.isfinite(times).all() or (times > current_time + tolerance).any()):
        raise ValueError("latent availability times must match history and never follow current_time")
    controls = torch.arange(control_index + 1, dtype=torch.float64) * dt
    validate_latent_grid(metadata, controls, times)
    if route == "g_translator":
        record = metadata["demonstration"]
        _identity(record)
        if record["domain"] != "human" or set(record) - _IDENTITY_FIELDS - {"arrays"}:
            raise ValueError("G observation requires a human demonstration video identity")
        demonstration = _arrays(path, record, robot_target=False)["latent"]
        if demonstration.shape[0] != history.shape[0]:
            raise ValueError("demonstration and robot latent channels must match")
        return GObservation(metadata, state[None], history[None], times, demonstration[None])
    language, identity = None, None
    if "language" in metadata:
        language, identity = load_goal_language(_local_path(path.parent, _text(metadata, "language"), ".json"))
        if identity != registry["language_identity"]:
            raise ValueError("language encoder differs from the trained interface")
    goal = load_goal_prediction(_local_path(path.parent, _text(metadata, "goal"), ".json"),
                               encoder_identity=payload["encoder_identity"], registry=registry)
    return PiObservation(metadata, state[None], history[None], times, language, identity, goal)


@torch.no_grad()
def predict_g_pi_cli(args):
    from .g_pi_interface import GTranslator, PiGoalPolicy
    from .g_pi_training import load_g_pi_policy

    output = Path(args.output)
    if output.suffix != ".npz" or output.exists() or output.with_suffix(".json").exists():
        raise ValueError("prediction requires fresh .npz and .json output paths")
    native, interface, encoder, payload = load_g_pi_policy(args.policy, device=args.device,
                                                         checkpoint=getattr(args, "checkpoint", None))
    observation = load_g_pi_observation(args.observation, payload)
    config, registry = payload["config"], payload["registry"]
    if config["interface_type"] == "g_translator":
        model = GTranslator(native, interface, config, feature_layer=config["goal_encoder"]["layer"])
        result = model.predict(observation.demonstration, observation.history, observation.state)
        sidecar = save_goal_prediction(output, result, encoder.identity, registry)
    else:
        shape = (1, native.config.action_dim, config["chunk_size"], registry["actions_per_frame"], 1)
        mask = torch.tensor(registry["action_space"]["valid_channels"], dtype=torch.bool)[None, :, None, None, None]
        model = PiGoalPolicy(native, interface, config, action_shape=shape, actions_mask=mask,
                             seed=args.seed, video_native=encoder.native)
        result = {"actions": model.predict(observation.history, observation.state, observation.language,
                                           observation.goal, frame_times=observation.history_times)}
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **{name: value.float().cpu().numpy() for name, value in result.items()})
        sidecar = output.with_suffix(".json")
        write_json(sidecar, {"format_version": 1, "kind": "g_pi_actions", "arrays": output.name,
            "arrays_sha256": file_sha256(output), "encoder_identity": encoder.identity, "registry": registry})
    return {"output": str(output.resolve()), "manifest": str(sidecar.resolve()),
            "interface_type": config["interface_type"], "test_time_updates": False,
            "video_generation": False, "commands_sent": 0}
