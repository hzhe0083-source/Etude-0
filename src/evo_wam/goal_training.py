"""Two-stage SE(3)-supervised interface with no raw-video action shortcut."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .cli import file_sha256, source_module, write_json
from .goal_action import goal_action_forward, goal_action_sample, install_action_lora
from .goal_data import GoalObservation, load_goal_index, load_goal_observation, load_goal_sample
from .goal_future import generated_robot_features
from .goal_interface import GoalInterface, goal_pose_loss
from .icl_training import build_icl_model, native_icl_loss, prepare_icl_inputs, validate_icl_config, _model_state
from .video_data import patch_grid_coordinates, _source_components
from .zerowam import VideoLoRA, ZERO_WAM_COMMIT


STAGES = {"goal", "visual"}


def load_goal_config(path):
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict) or config.get("kind") != "se3_goal_experiment":
        raise ValueError("expected se3_goal_experiment, not a raw/demo-compression ICL config")
    validate_icl_config({**config, "kind": "native_icl_experiment"})
    if "demo_bottleneck" in config or config["domain_schedule"] != ["robot"]:
        raise ValueError("the SE(3) action interface uses robot-supervised targets and no demo bottleneck")
    interface = config.get("goal_interface")
    required = {"state_dim", "dim", "num_tokens", "num_heads", "translation_scale"}
    if not isinstance(interface, dict) or set(interface) != required:
        raise ValueError("goal_interface must declare state/capacity dimensions and translation_scale")
    for key in required - {"translation_scale"}:
        if type(interface[key]) is not int or interface[key] < 1:
            raise ValueError("goal interface dimensions must be positive integers")
    if interface["dim"] % interface["num_heads"]:
        raise ValueError("goal interface dim must be divisible by num_heads")
    for value in (interface["translation_scale"], config["pose_weight"], config["video_weight"]):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("pose/video weights and translation scale must be positive and finite")
    for name in ("sampling_steps", "action_sampling_steps"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError("sampling steps must be positive integers")
    return config


def goal_registry(sample):
    keys = ("state_space_id", "coordinate_frame", "pose_units", "end_effectors", "goal_source", "action_space", "control_dt")
    return {**{key: sample.metadata[key] for key in keys}, "actions_per_frame": sample.actions.shape[3]}


def build_goal_system(config, registry, *, stage, checkpoint=None, tiny_native=False, device="cuda"):
    if stage not in STAGES:
        raise ValueError("SE(3) stage must be goal or visual")
    native, null, identity = build_icl_model(config, checkpoint=checkpoint, tiny_native=tiny_native, device=device)
    install_action_lora(native, **config["lora"])
    layers = list(native.mcp_hidden_collect_layers)
    if not layers or len(set(layers)) != len(layers) or max(layers) >= len(native.blocks):
        raise ValueError("native feature-collection layers must identify actual video blocks")
    if registry["action_space"]["dimension"] != native.config.action_dim:
        raise ValueError("goal action dimension does not match the native checkpoint")
    interface = GoalInterface(native_dim=native.inner_dim, feature_dim=len(layers) * native.inner_dim,
        effectors=len(registry["end_effectors"]), **config["goal_interface"]).to(native.patch_embedding_mlp.weight)
    native.requires_grad_(False)
    for name, module in native.named_modules():
        if isinstance(module, VideoLoRA):
            module.enable(("action_" in name) == (stage == "goal"))
    for name, parameter in interface.named_parameters():
        enabled = (name.startswith(("goal_encoder.", "state_encoder.", "condition_adapter.")) if stage == "goal"
                   else name.startswith(("visual_", "pose_decoder.")))
        parameter.requires_grad_(enabled)
    native.eval()
    interface.eval()
    return native, interface, null, identity, layers


def _check_sample(sample, config, registry):
    if goal_registry(sample) != registry:
        raise ValueError("goal frame, source type, state/action normalization or control timing changed")
    if sample.state.shape[-1] != config["goal_interface"]["state_dim"]:
        raise ValueError("state width differs from the configured observed robot state")
    if sample.actions.shape[2] != config["chunk_size"]:
        raise ValueError("one SE(3) target must supervise exactly one configured future action chunk")


def _action_noise(actions, mask, generator, device):
    scheduler = source_module("utils/scheduler.py").FlowMatchScheduler(shift=1., sigma_min=0., extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    clean, mask = actions.detach().cpu().float(), mask.cpu()
    noise = torch.randn(clean.shape, generator=generator)
    times = scheduler.timesteps[torch.randint(1000, (clean.shape[2],), generator=generator)]
    noisy = torch.where(mask, scheduler.add_noise(clean, noise, times, t_dim=2), 0)
    target = scheduler.training_target(clean, noise, times)
    return noisy.to(device), times[None].to(device), target.to(device), mask.to(device)


def visual_goal_tokens(native, interface, demonstration, history, state, null, config, generator,
                       *, feature_layers, current_time, control_dt, actions_per_frame, demo_times=None):
    """This API deliberately has no ground-truth future, pose or action argument."""
    generated, features = generated_robot_features(native, demonstration, history, null,
        {**config, "feature_layers": feature_layers}, generator, demo_times=demo_times)
    frames = generated.shape[2]
    times = torch.arange(1, frames + 1, device=features.device, dtype=torch.float64)
    times = times * actions_per_frame * control_dt + current_time
    coordinates = patch_grid_coordinates([generated.shape[-2] // native.patch_size[1],
                                         generated.shape[-1] // native.patch_size[2]]).to(features.device)
    return interface.read_future(features, state, times, coordinates), generated


def goal_training_loss(native, interface, null, sample, config, generators, *, stage, feature_layers):
    device = native.action_embedder.weight.device
    state = sample.state.to(device)
    if stage == "goal":
        tokens = interface.encode_goal(sample.goal_poses.to(device))
    elif stage == "visual":
        if sample.visual is None:
            raise ValueError("visual training requires a compatible reference and observed robot history")
        pair = sample.visual
        tokens, _ = visual_goal_tokens(native, interface, pair.demonstration,
            pair.target[:, :, :pair.history_frames], state, null, config, generators["future"],
            feature_layers=feature_layers, current_time=sample.metadata["current_time"],
            control_dt=sample.metadata["control_dt"], actions_per_frame=sample.actions.shape[3],
            demo_times=pair.demonstration_times)
    else:
        raise ValueError("stage must be goal or visual")
    condition = interface.condition(tokens, state)
    noisy, times, target, mask = _action_noise(sample.actions, sample.actions_mask, generators["action"], device)
    predicted = goal_action_forward(native, noisy, times, condition).float()
    losses = {"action": torch.where(mask, (predicted - target.detach()).square(), 0).sum() / mask.sum().clamp_min(1)}
    losses["total"] = losses["action"]
    if stage == "visual":
        pose = goal_pose_loss(interface.decode_goal(tokens), sample.goal_poses.to(device), interface.translation_scale)
        losses["pose_translation"], losses["pose_rotation"] = pose["translation"], pose["rotation"]
        # Teacher-forced video supervision is separate from the generated-only
        # goal/action path. No raw-video action expert is called for this loss.
        video_inputs = prepare_icl_inputs(sample.visual, config, native, null, generators["video"], include_actions=False)
        video = native_icl_loss(native, video_inputs, config, human=False, video_only=True)
        losses["video"], losses["ifp"] = video["video"], video["ifp"]
        losses["total"] = losses["total"] + config["pose_weight"] * pose["total"] + config["video_weight"] * video["total"]
    return losses


def _system_state(native, interface, tiny):
    return {"native": _model_state(native, tiny),
            "interface": {key: value.detach().cpu() for key, value in interface.state_dict().items()}}


def _restore_system(native, interface, model, tiny):
    expected = _system_state(native, interface, tiny)
    if set(model) != set(expected) or any(model[key].keys() != expected[key].keys() for key in expected):
        raise ValueError("SE(3) artifact must contain the full action/video adapters and goal interface")
    native.load_state_dict(model["native"], strict=tiny)
    interface.load_state_dict(model["interface"], strict=True)


def read_goal_artifact(path, kind="se3_goal_training"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or payload.get("format_version") != 1
            or payload.get("kind") != kind or payload.get("upstream_commit") != ZERO_WAM_COMMIT):
        raise ValueError(f"expected a pinned {kind} artifact; raw ICL/demo bottleneck artifacts are different models")
    return payload


def train_goal_interface(args):
    config = load_goal_config(args.config)
    stage = args.stage
    if stage not in STAGES or args.resume and args.initialize:
        raise ValueError("choose one SE(3) stage and either resume or initialize")
    paths, sources = load_goal_index(args.index)
    if type(args.steps) is not int or args.steps < 1:
        raise ValueError("steps must be a positive integer")
    first = load_goal_sample(paths[0], visual=stage == "visual")
    registry = goal_registry(first)
    _check_sample(first, config, registry)
    previous = read_goal_artifact(args.resume or args.initialize) if args.resume or args.initialize else None
    if stage == "visual" and previous is None:
        raise ValueError("Stage 2 requires a successfully trained goal/action interface from Stage 1")
    if args.initialize and (stage != "visual" or previous["stage"] != "goal" or previous["updates"] < 1):
        raise ValueError("initialize is only Stage 1 goal -> Stage 2 visual")
    if previous and (previous["config"] != config or previous["registry"] != registry or previous["tiny_native"] != args.tiny_native):
        raise ValueError("stage transition/resume requires identical interface, coordinate/state/action conventions and base mode")
    identities = {"index": file_sha256(args.index)}
    for path in paths:
        identities[str(path)] = file_sha256(path)
        if stage == "visual":
            meta = json.loads(path.read_text())
            pair_path = path.parent / meta["visual_pair"]
            identities[str(pair_path.resolve())] = file_sha256(pair_path)
    start = previous["attempted_steps"] if args.resume else 0
    if start + args.steps > config["training"]["max_steps"]:
        raise ValueError("cumulative SE(3) stage budget exceeded")
    if args.resume and (previous["stage"] != stage or previous["data_identity"] != identities or previous["seed"] != args.seed):
        raise ValueError("resume requires the same stage, index and seed")
    if previous:
        _source_components([*previous["source_records"], *sources])
    visited = dict(previous["visited_arrays"]) if args.resume else {}
    for name, digest in visited.items():
        if file_sha256(name) != digest:
            raise ValueError("consumed SE(3)/video arrays changed before resume")
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not args.resume:
        raise ValueError("SE(3) training requires a fresh output directory or explicit resume")
    torch.manual_seed(args.seed)
    native, interface, null, identity, layers = build_goal_system(config, registry, stage=stage,
        checkpoint=args.checkpoint, tiny_native=args.tiny_native, device=args.device)
    if previous:
        if previous["base_identity"] != identity or previous["feature_layers"] != layers:
            raise ValueError("frozen base/feature layers differ from the learned goal interface")
        _restore_system(native, interface, previous["model"], args.tiny_native)
    parameters = [p for module in (native, interface) for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["training"]["learning_rate"], weight_decay=0.)
    generators = {name: torch.Generator().manual_seed(args.seed + offset)
                  for offset, name in enumerate(("action", "future", "video"))}
    updates = 0
    if args.resume:
        optimizer.load_state_dict(previous["optimizer"])
        updates = previous["updates"]
        for name, generator in generators.items():
            generator.set_state(previous["rng"][name])
        torch.set_rng_state(previous["torch_rng"])
        if torch.cuda.is_available() and previous["cuda_rng"]:
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
    visual_space = previous.get("visual_feature_space") if args.resume else None
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    started = time.monotonic()
    with (output / "metrics.jsonl").open("a") as log:
        for step in range(start, start + args.steps):
            path = paths[step % len(paths)]
            sample = load_goal_sample(path, visual=stage == "visual")
            _check_sample(sample, config, registry)
            arrays = [path.parent / sample.metadata["arrays"]]
            if sample.visual is not None:
                space = sample.visual.metadata["feature_space_id"]
                if visual_space is not None and visual_space != space:
                    raise ValueError("visual feature space must remain identical across paired robot samples")
                visual_space = space
                pair_path = path.parent / sample.metadata["visual_pair"]
                arrays += [pair_path.parent / sample.visual.metadata[role]["arrays"] for role in ("demonstration", "target")]
            for array in arrays:
                key = str(array.resolve())
                if key not in visited:
                    visited[key] = file_sha256(array)
            optimizer.zero_grad(set_to_none=True)
            losses = goal_training_loss(native, interface, null, sample, config, generators,
                                        stage=stage, feature_layers=layers)
            if not torch.isfinite(losses["total"]):
                raise ValueError("nonfinite SE(3) objective; optimizer not updated")
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, config["training"]["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            log.write(json.dumps({**{key: float(value.detach()) for key, value in losses.items()},
                "stage": stage, "step": step, "updated": True, "gradient_norm": float(norm),
                "goal_source": registry["goal_source"], "goal_input": "true_goal" if stage == "goal" else "generated_future_only",
                "raw_video_to_action": False}, allow_nan=False) + "\n")
            log.flush()
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    goal_updates = updates if stage == "goal" else previous["goal_stage_updates"]
    payload = {"format_version": 1, "kind": "se3_goal_training", "upstream_commit": ZERO_WAM_COMMIT,
        "config": config, "stage": stage, "model": _system_state(native, interface, args.tiny_native),
        "base_identity": identity, "tiny_native": args.tiny_native, "feature_layers": layers,
        "registry": registry, "visual_feature_space": visual_space, "updates": updates, "goal_stage_updates": goal_updates,
        "attempted_steps": start + args.steps, "seed": args.seed, "optimizer": optimizer.state_dict(),
        "rng": {key: gen.get_state() for key, gen in generators.items()}, "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "data_identity": identities, "visited_arrays": visited,
        "source_records": previous["source_records"] if args.resume else [*(previous["source_records"] if previous else []), *sources]}
    temp = output / "goal_interface.pt.tmp"
    torch.save(payload, temp)
    temp.replace(output / "goal_interface.pt")
    report = {"artifact": str((output / "goal_interface.pt").resolve()), "stage": stage,
        "updates": updates, "goal_stage_updates": goal_updates, "elapsed_seconds": time.monotonic() - started,
        "action_adapters_updated": stage == "goal", "raw_video_to_action": False,
        "ground_truth_future_for_goal": False, "robot_execution_evaluated": False}
    write_json(output / "run.json", report)
    return report


def export_goal_policy(args):
    payload = read_goal_artifact(args.artifact)
    if payload["stage"] != "visual" or payload["updates"] < 1 or payload["goal_stage_updates"] < 1:
        raise ValueError("export requires both trained goal/action and visual-interface stages")
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("SE(3) policy export requires a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    excluded = {"optimizer", "rng", "torch_rng", "cuda_rng", "visited_arrays", "data_identity"}
    policy = {key: value for key, value in payload.items() if key not in excluded}
    policy["kind"] = "se3_goal_policy"
    torch.save(policy, output / "policy.pt")
    write_json(output / "policy.json", {"kind": "se3_goal_policy", "format_version": 1,
        "policy_sha256": file_sha256(output / "policy.pt"), "source_artifact_sha256": file_sha256(args.artifact)})
    return {"policy": str(output.resolve()), "test_time_updates": False, "raw_video_to_action": False}


def load_goal_policy(path, *, checkpoint=None, device="cuda"):
    folder = Path(path)
    manifest = json.loads((folder / "policy.json").read_text())
    if (manifest.get("kind") != "se3_goal_policy" or manifest.get("format_version") != 1
            or file_sha256(folder / "policy.pt") != manifest.get("policy_sha256")):
        raise ValueError("SE(3) policy manifest/checksum mismatch")
    payload = read_goal_artifact(folder / "policy.pt", kind="se3_goal_policy")
    if payload["stage"] != "visual" or min(payload["updates"], payload["goal_stage_updates"]) < 1:
        raise ValueError("both interface stages must have successful updates")
    native, interface, null, identity, layers = build_goal_system(payload["config"], payload["registry"],
        stage="visual", checkpoint=checkpoint, tiny_native=payload["tiny_native"], device=device)
    if identity != payload["base_identity"] or layers != payload["feature_layers"]:
        raise ValueError("policy requires the same immutable native checkpoint and feature layers")
    _restore_system(native, interface, payload["model"], payload["tiny_native"])
    native.eval().requires_grad_(False)
    interface.eval().requires_grad_(False)
    return native, interface, null, payload


@torch.no_grad()
def predict_goal_actions(native, interface, null, payload, observation, *, seed=0):
    if not isinstance(observation, GoalObservation):
        raise ValueError("prediction accepts observed-only GoalObservation, never a supervised training sample")
    if any(p.requires_grad for module in (native, interface) for p in module.parameters()):
        raise ValueError("freeze the complete SE(3) policy before inference")
    registry, config = payload["registry"], payload["config"]
    for name in registry.keys() - {"goal_source"}:
        if observation.metadata.get(name) != registry[name]:
            raise ValueError(f"observed {name} differs from the trained robot interface")
    if observation.metadata["feature_space_id"] != payload["visual_feature_space"]:
        raise ValueError("observation/demo visual encoder differs from the trained interface")
    state = observation.state.to(native.action_embedder.weight.device)
    tokens, future = visual_goal_tokens(native, interface, observation.demonstration, observation.history, state,
        null, config, torch.Generator().manual_seed(seed + 1), feature_layers=payload["feature_layers"],
        current_time=observation.metadata["current_time"], control_dt=registry["control_dt"],
        actions_per_frame=registry["actions_per_frame"], demo_times=observation.demonstration_times)
    condition = interface.condition(tokens, state)
    shape = (1, native.config.action_dim, config["chunk_size"], registry["actions_per_frame"], 1)
    mask = torch.tensor(registry["action_space"]["valid_channels"], dtype=torch.bool)[None, :, None, None, None]
    actions = goal_action_sample(native, condition, shape, mask, torch.Generator().manual_seed(seed),
                                 steps=config["action_sampling_steps"])
    return {"actions": actions, "goal_poses": interface.decode_goal(tokens), "generated_future": future}


def predict_goal_cli(args):
    output = Path(args.output)
    if output.exists() or output.suffix != ".npz":
        raise ValueError("SE(3) prediction requires a fresh .npz output path")
    native, interface, null, payload = load_goal_policy(args.policy, checkpoint=args.checkpoint, device=args.device)
    observation = load_goal_observation(args.observation)
    result = predict_goal_actions(native, interface, null, payload, observation, seed=args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{name: value.float().cpu().numpy() for name, value in result.items()})
    return {"output": str(output.resolve()), "coordinate_frame": payload["registry"]["coordinate_frame"],
        "goal_source": payload["registry"]["goal_source"], "test_time_updates": False, "commands_sent": 0}
