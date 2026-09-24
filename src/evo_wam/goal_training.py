"""Goal-supervised action training, with legacy and observed-context routes."""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .cli import file_sha256, source_module, write_json
from .goal_action import action_named_parameters, goal_action_forward, goal_action_sample, install_action_interface
from .goal_data import GoalObservation, load_goal_index, load_goal_observation, load_goal_sample
from .goal_future import generated_robot_features
from .goal_context import observed_context_features
from .goal_interface import GoalInterface, goal_pose_loss
from .goal_observed_interface import ObservedGoalInterface
from .icl_training import build_icl_model, native_icl_loss, prepare_icl_inputs, validate_icl_config
from .video_data import patch_grid_coordinates, _source_components, _local_path
from .zerowam import ZERO_WAM_COMMIT, load_native_class


STAGES = {"goal", "visual", "joint", "g", "pi"}
G_PI_ROUTES = {"g_translator", "pi_goal"}
ARCHITECTURE = "recurrent_goal_full_v2"
OBSERVED_ARCHITECTURE = "observed_goal_dual_v1"


def goal_architecture(config):
    if config.get("interface_type") in G_PI_ROUTES:
        from .g_pi_training import g_pi_architecture
        return g_pi_architecture(config)
    return OBSERVED_ARCHITECTURE if config.get("interface_type") == "observed_dual" else ARCHITECTURE


def _check_stage(config, stage):
    if config["interface_type"] in G_PI_ROUTES:
        expected = "g" if config["interface_type"] == "g_translator" else "pi"
        if stage != expected:
            raise ValueError(f"interface_type {config['interface_type']} requires stage {expected}")
        return
    allowed = {"joint"} if config["interface_type"] == "observed_dual" else {"goal", "visual"}
    if stage not in allowed:
        raise ValueError(f"interface_type {config['interface_type']} requires stage {sorted(allowed)}")


def _has_visual_input(stage):
    return stage in {"visual", "joint"}


def _raw_visual_condition(config, stage):
    return _has_visual_input(stage) and config["interface_type"] in {"direct_features", "observed_dual"}


def _require_trained_policy(payload):
    config = payload["config"]
    _check_stage(config, payload["stage"])
    if config["interface_type"] == "observed_dual":
        if payload["updates"] < 1:
            raise ValueError("observed-context export requires successful joint training updates")
    elif payload["stage"] != "visual" or min(payload["updates"], payload["goal_stage_updates"]) < 1:
        raise ValueError("policy requires both successfully trained stages")


def autocast_for(native):
    device = native.action_embedder.weight.device.type
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()


def load_goal_config(path):
    return validate_goal_config(json.loads(Path(path).read_text()))


def validate_goal_config(config):
    if isinstance(config, dict) and config.get("interface_type") in G_PI_ROUTES:
        from .g_pi_training import validate_g_pi_config
        return validate_g_pi_config(config)
    if (not isinstance(config, dict) or config.get("kind") != "se3_goal_experiment"
            or config.get("schema_version") != 2 or "lora" in config):
        raise ValueError("expected version-2 full-parameter se3_goal_experiment without LoRA")
    validate_icl_config({**config, "kind": "native_icl_experiment", "schema_version": 1}, adaptation=False)
    if ("demo_bottleneck" in config or config["domain_schedule"] != ["robot"]
            or config["human_context"] != "cross_video"):
        raise ValueError("goal training requires cross-video robot targets without demo compression")
    if config.get("interface_type") not in {"latent", "direct_features", "observed_dual"}:
        raise ValueError("interface_type must explicitly select latent, direct_features or observed_dual")
    observed = config["interface_type"] == "observed_dual"
    interface = config.get("goal_interface")
    required = {"state_dim", "dim", "num_heads", "translation_scale"}
    if not observed:
        required |= {"num_tokens", "num_layer_groups", "num_pose_tokens"}
    if not isinstance(interface, dict) or set(interface) != required:
        raise ValueError("goal_interface fields must match the selected interface dimensions and translation_scale")
    if any(type(interface[key]) is not int or interface[key] < 1 for key in required - {"translation_scale"}):
        raise ValueError("goal interface dimensions must be positive integers")
    if interface["dim"] % interface["num_heads"] or (not observed and interface["num_pose_tokens"] > interface["num_tokens"]):
        raise ValueError("invalid attention heads or pose-token count")
    weights = [interface["translation_scale"], config["pose_weight"], config["training"]["backbone_learning_rate"]]
    if observed:
        if (type(config.get("video_weight")) not in (int, float) or config["video_weight"] != 0
                or config["ifp"]["enabled"] or any(config["ifp"]["loss_weights"])):
            raise ValueError("observed_dual requires video_weight=0 and disabled, zero-weight IFP; no video targets")
        if "sampling_steps" in config:
            raise ValueError("observed_dual has no video sampling_steps; use action_sampling_steps only")
    else:
        weights.append(config["video_weight"])
    for value in weights:
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("loss weights, translation scale and learning rates must be positive and finite")
    for name in (("action_sampling_steps",) if observed else ("sampling_steps", "action_sampling_steps")):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError("sampling steps must be positive integers")
    return config


def goal_registry(sample):
    keys = ("state_space_id", "coordinate_frame", "pose_units", "pose_representation", "tool_frames",
            "end_effectors", "gripper_space", "goal_source", "action_space", "control_dt")
    registry = {**{key: sample.metadata[key] for key in keys}, "actions_per_frame": sample.actions.shape[3],
                "language_identity": sample.language_identity}
    if "event_rules" in sample.metadata:
        registry["event_rules"] = sample.metadata["event_rules"]
    return registry


def _interface(native, config, registry):
    if registry["action_space"]["dimension"] != native.config.action_dim:
        raise ValueError("goal action dimension differs from native checkpoint")
    if registry["language_identity"]["text_dim"] != native.config.text_dim:
        raise ValueError("language encoder width differs from native checkpoint")
    if config["interface_type"] == "observed_dual":
        return ObservedGoalInterface(native_dim=native.inner_dim, effectors=len(registry["end_effectors"]),
            **config["goal_interface"]).to(native.action_embedder.weight)
    return GoalInterface(native_dim=native.inner_dim, feature_dim=native.inner_dim,
        num_layers=len(native.blocks), effectors=len(registry["end_effectors"]),
        **config["goal_interface"]).to(native.action_embedder.weight)


def _set_training(native, interface, stage, interface_type):
    if interface_type == "observed_dual":
        if stage != "joint":
            raise ValueError("observed_dual trains jointly from observed context")
        native.requires_grad_(True).train()
        interface.requires_grad_(True).train()
        return
    native.requires_grad_(stage == "visual")
    if stage == "goal":
        for _, parameter in action_named_parameters(native):
            parameter.requires_grad_(True)
    for name, parameter in interface.named_parameters():
        if stage == "goal":
            enabled = name.startswith(("goal_encoder.", "state_encoder.", "condition_adapter."))
        elif interface_type == "direct_features":
            enabled = name.startswith("state_encoder.")
        else:
            enabled = not name.startswith("goal_encoder.")
        parameter.requires_grad_(enabled)
    native.train()
    interface.train()


def build_goal_system(config, registry, *, stage, checkpoint=None, tiny_native=False, device="cuda"):
    if config.get("interface_type") in G_PI_ROUTES:
        from .g_pi_training import build_g_pi_system
        return build_g_pi_system(config, registry, stage=stage, checkpoint=checkpoint,
                                 tiny_native=tiny_native, device=device)
    _check_stage(config, stage)
    native, _, identity = build_icl_model(config, checkpoint=checkpoint, tiny_native=tiny_native,
                                        device=device, adaptation=False, dtype=torch.float32)
    install_action_interface(native)
    interface = _interface(native, config, registry)
    _set_training(native, interface, stage, config["interface_type"])
    return native, interface, None, identity, list(range(len(native.blocks)))


def _check_sample(sample, config, registry):
    if goal_registry(sample) != registry:
        raise ValueError("goal frame/source, gripper, language or robot conventions changed")
    if sample.state.shape[-1] != config["goal_interface"]["state_dim"]:
        raise ValueError("state width differs from configuration")
    if sample.actions.shape[2] != config["chunk_size"]:
        raise ValueError("one query must supervise exactly one configured action block")


def _action_noise(actions, mask, generator, device):
    scheduler = source_module("utils/scheduler.py").FlowMatchScheduler(shift=1., sigma_min=0., extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    clean, mask = actions.detach().cpu().float(), mask.cpu()
    noise = torch.randn(clean.shape, generator=generator)
    times = scheduler.timesteps[torch.randint(1000, (clean.shape[2],), generator=generator)]
    noisy = torch.where(mask, scheduler.add_noise(clean, noise, times, t_dim=2), 0)
    target = scheduler.training_target(clean, noise, times)
    return noisy.to(device), times[None].to(device), target.to(device), mask.to(device)


def masked_action_loss(prediction, target, mask):
    """Loss-only function: callers hold noisy inputs fixed when intervening on targets."""
    if prediction.shape != target.shape or mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError("action prediction, target and Boolean mask must have identical shapes")
    difference = torch.where(mask, prediction.float() - target.detach().float(), 0)
    return difference.square().sum() / mask.sum().clamp_min(1)


def visual_goal_tokens(native, interface, demonstration, history, state, language, config, generator,
                       *, feature_layers, current_time, control_dt, actions_per_frame, demo_times=None):
    """Observed-only path; return per-layer conditions, final latents and sampled future."""
    if config["interface_type"] not in {"latent", "direct_features"}:
        raise ValueError("unknown visual interface; no implicit raw-feature fallback")
    if feature_layers != list(range(len(native.blocks))):
        raise ValueError("the interface requires every native coupling layer in depth order")
    weight = native.action_embedder.weight
    language = language.to(weight)
    language_hidden = native.condition_embedder_action.text_embedder(language)
    semantic = interface.semantic(language_hidden, state)
    tokens = interface.initial_queries(state.shape[0]) if config["interface_type"] == "latent" else None
    times = torch.arange(1, config["chunk_size"] + 1, device=weight.device, dtype=torch.float64)
    times = times * actions_per_frame * control_dt + current_time
    coordinates = patch_grid_coordinates([history.shape[-2] // native.patch_size[1],
                                         history.shape[-1] // native.patch_size[2]]).to(weight.device)
    conditions = []

    def read_layer(index, features):
        nonlocal tokens
        if index != len(conditions):
            raise ValueError("native feature callbacks must follow coupling-layer order")
        if tokens is None:
            condition = interface.direct_condition(features, state, language_hidden)
        else:
            tokens = interface.read_layer(tokens, features, semantic, times, coordinates, index)
            condition = interface.condition(tokens, state, language_hidden)
        conditions.append(condition)

    generated, _ = generated_robot_features(native, demonstration, history, language,
        {**config, "feature_layers": feature_layers}, generator, demo_times=demo_times, on_layer=read_layer)
    if len(conditions) != len(native.blocks):
        raise ValueError("not every action layer received its visual condition")
    return conditions, tokens, generated


def observed_goal_conditions(native, interface, demonstration, history, state, language, config, *,
                             feature_layers, demo_times=None):
    """One clean context pass: Wan and pose-decoder features condition actions.

    No future video, goal labels, action labels, or sampler enter this path.
    The final decoder supplies SE(3)/gripper outputs; its hidden features and
    the corresponding Wan features both condition every action layer.
    """
    if config["interface_type"] != "observed_dual":
        raise ValueError("observed context requires the observed_dual interface")
    if feature_layers != list(range(len(native.blocks))):
        raise ValueError("the observed interface requires every native layer in order")
    weight = native.action_embedder.weight
    language = language.to(weight)
    state = state.to(weight)
    language_hidden = native.condition_embedder_action.text_embedder(language)
    conditions, predictions = [], []

    def read_layer(index, features):
        if index != len(conditions):
            raise ValueError("observed feature callbacks must follow native layer order")
        condition, prediction = interface.condition_from_features(features, state, language_hidden)
        conditions.append(condition)
        predictions[:] = [prediction]

    observed_context_features(native, demonstration, history, language,
        {**config, "feature_layers": feature_layers}, demo_times=demo_times, on_layer=read_layer)
    if len(conditions) != len(native.blocks) or not predictions:
        raise ValueError("not every action layer received observed-context features")
    return conditions, predictions[0]


def _clear_video_cache(native):
    names = {"pos"} | {name for block in native.blocks for name in block.attn1.attn_caches}
    for name in names:
        native.clear_cache(name)


def goal_training_loss(native, interface, unused, sample, config, generators, *, stage, feature_layers):
    if config.get("interface_type") in G_PI_ROUTES:
        from .g_pi_training import g_pi_training_loss
        return g_pi_training_loss(native, interface, unused, sample, config, generators,
                                  stage=stage, feature_layers=feature_layers)
    del unused
    _check_stage(config, stage)
    device = native.action_embedder.weight.device
    state, language = sample.state.to(device), sample.language.to(device)
    with autocast_for(native):
        decoded = None
        if stage == "goal":
            tokens = interface.encode_goal(sample.goal_poses.to(device), sample.goal_gripper.to(device))
            language_hidden = native.condition_embedder_action.text_embedder(language)
            condition = interface.condition(tokens, state, language_hidden)
            conditions = [condition] * len(native.blocks)
        elif _has_visual_input(stage):
            if sample.visual is None:
                raise ValueError("visual training requires an operation-compatible reference and observed history")
            pair = sample.visual
            if stage == "joint":
                conditions, decoded = observed_goal_conditions(native, interface, pair.demonstration,
                    pair.target[:, :, :pair.history_frames], state, language, config,
                    feature_layers=feature_layers, demo_times=pair.demonstration_times)
                tokens = None
            else:
                conditions, tokens, _ = visual_goal_tokens(native, interface, pair.demonstration,
                    pair.target[:, :, :pair.history_frames], state, language, config, generators["future"],
                    feature_layers=feature_layers, current_time=sample.metadata["current_time"],
                    control_dt=sample.metadata["control_dt"], actions_per_frame=sample.actions.shape[3],
                    demo_times=pair.demonstration_times)
        else:
            raise ValueError("stage must be goal, visual or joint")
        noisy, times, target, mask = _action_noise(sample.actions, sample.actions_mask, generators["action"], device)
        predicted = goal_action_forward(native, noisy, times, conditions)
        losses = {"action": masked_action_loss(predicted, target, mask)}
        losses["total"] = losses["action"]
        if _has_visual_input(stage):
            if tokens is not None:
                decoded = interface.decode_goal(tokens)
            if decoded is not None:
                pose = goal_pose_loss(decoded["goal_poses"], sample.goal_poses.to(device), interface.translation_scale,
                    gripper_prediction=decoded["goal_gripper"], gripper_target=sample.goal_gripper.to(device))
                for key, value in pose.items():
                    if key != "total":
                        losses[f"pose_{key}"] = value
                losses["total"] = losses["total"] + config["pose_weight"] * pose["total"]
            if stage == "joint":
                # Deliberately do not construct video/IFP targets or forwards.
                return losses
            # Teacher-forced video/IFP execution owns no cache from the deployment path.
            _clear_video_cache(native)
            try:
                inputs = prepare_icl_inputs(sample.visual, config, native, language, generators["video"], include_actions=False)
                video = native_icl_loss(native, inputs, config, human=False, video_only=True)
            finally:
                _clear_video_cache(native)
            losses["video"], losses["ifp"] = video["video"], video["ifp"]
            losses["total"] = losses["total"] + config["video_weight"] * video["total"]
    return losses


def _system_state(native, interface, tiny=None):
    return {name: {key: value.detach().cpu() for key, value in module.state_dict().items()}
            for name, module in (("native", native), ("interface", interface))}


def _restore_system(native, interface, model, tiny=None):
    if set(model) != {"native", "interface"}:
        raise ValueError("artifact must contain full native and interface weights")
    for name, module in (("native", native), ("interface", interface)):
        expected = module.state_dict()
        if model[name].keys() != expected.keys() or any(
                model[name][key].shape != value.shape or model[name][key].dtype != value.dtype
                for key, value in expected.items()):
            raise ValueError("complete model keys, shapes and precision must match the independent action topology")
        module.load_state_dict(model[name], strict=True)


def read_goal_artifact(path, kind="se3_goal_training"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and payload.get("kind") == "g_pi_training":
        from .g_pi_training import read_g_pi_artifact
        return read_g_pi_artifact(path, payload=payload)
    if (not isinstance(payload, dict) or payload.get("format_version") != 2
            or payload.get("kind") != kind or not isinstance(payload.get("config"), dict)
            or payload.get("architecture") != goal_architecture(payload["config"])
            or payload.get("upstream_commit") != ZERO_WAM_COMMIT or payload.get("precision") != "float32"):
        raise ValueError("expected a version-2 FP32 full training artifact; deployment and old adapters cannot resume")
    validate_goal_config(payload["config"])
    _check_stage(payload["config"], payload["stage"])
    return payload


def _optimizer(native, interface, config, stage):
    action_ids = {id(p) for _, p in action_named_parameters(native)}
    video = [p for p in native.parameters() if p.requires_grad and id(p) not in action_ids]
    action = [p for p in native.parameters() if p.requires_grad and id(p) in action_ids]
    groups = [{"params": action + [p for p in interface.parameters() if p.requires_grad],
               "lr": config["training"]["learning_rate"], "name": "action_interface"}]
    if video:
        groups.append({"params": video, "lr": config["training"]["backbone_learning_rate"], "name": "video"})
    return torch.optim.AdamW(groups, weight_decay=0.)


def _sample_files(path, sample):
    files = [path, path.parent / sample.metadata["arrays"]]
    language = path.parent / sample.metadata["language"]
    files += [language, language.parent / json.loads(language.read_text())["arrays"]]
    if sample.visual is not None:
        pair = path.parent / sample.metadata["visual_pair"]
        files.append(pair)
        files += [pair.parent / sample.visual.metadata[role]["arrays"] for role in ("demonstration", "target")]
    return files


def train_goal_interface(args):
    config, stage = load_goal_config(args.config), args.stage
    if config["interface_type"] in G_PI_ROUTES:
        from .g_pi_training import train_g_pi_interface
        return train_g_pi_interface(args)
    if stage not in STAGES or args.resume and args.initialize:
        raise ValueError("choose a stage and either resume or initialize")
    _check_stage(config, stage)
    paths, sources = load_goal_index(args.index)
    if type(args.steps) is not int or args.steps < 1:
        raise ValueError("steps must be positive")
    first = load_goal_sample(paths[0], visual=_has_visual_input(stage))
    registry = goal_registry(first)
    _check_sample(first, config, registry)
    previous = read_goal_artifact(args.resume or args.initialize) if args.resume or args.initialize else None
    if stage == "visual" and previous is None:
        raise ValueError("Stage 2 requires a successfully trained Stage 1 artifact")
    if args.initialize and (stage != "visual" or previous["stage"] != "goal" or previous["updates"] < 1):
        raise ValueError("initialize is only Stage 1 goal -> Stage 2 visual")
    comparable = lambda c: {key: value for key, value in c.items() if key != "interface_type"}
    if previous and (comparable(previous["config"]) != comparable(config)
                     or previous["registry"] != registry or previous["tiny_native"] != args.tiny_native):
        raise ValueError("stage transition/resume requires matching model, data conventions and training settings")
    identities = {"index": file_sha256(args.index), **{str(p.resolve()): file_sha256(p) for p in paths}}
    start = previous["attempted_steps"] if args.resume else 0
    if start + args.steps > config["training"]["max_steps"]:
        raise ValueError("cumulative stage budget exceeded")
    if args.resume and (previous["stage"] != stage or previous["data_identity"] != identities
                        or previous["seed"] != args.seed or previous["config"] != config):
        raise ValueError("resume requires the same stage, interface, index and seed")
    if previous:
        _source_components([*previous["source_records"], *sources])
    visited = dict(previous["visited_arrays"]) if args.resume else {}
    if any(file_sha256(path) != digest for path, digest in visited.items()):
        raise ValueError("consumed training inputs changed before resume")
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not args.resume:
        raise ValueError("training requires a fresh output directory or explicit resume")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    native, interface, unused, identity, layers = build_goal_system(config, registry, stage=stage,
        checkpoint=args.checkpoint, tiny_native=args.tiny_native, device=args.device)
    if previous:
        if previous["base_identity"] != identity or previous["feature_layers"] != layers:
            raise ValueError("initial checkpoint or coupling layers changed")
        _restore_system(native, interface, previous["model"])
    optimizer = _optimizer(native, interface, config, stage)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    generators = {name: torch.Generator().manual_seed(args.seed + offset)
                  for offset, name in enumerate(("action", "future", "video"))}
    updates = 0
    if args.resume:
        if previous["scheduler"] != {"kind": "constant", "state": None} or previous["data_cursor"] != start % len(paths):
            raise ValueError("scheduler or data cursor differs from this trainer")
        optimizer.load_state_dict(previous["optimizer"])
        updates = previous["updates"]
        for name, generator in generators.items():
            generator.set_state(previous["rng"][name])
        torch.set_rng_state(previous["torch_rng"])
        random.setstate(previous["python_rng"])
        n = previous["numpy_rng"]
        np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
        if torch.cuda.is_available() and previous["cuda_rng"]:
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
    visual_space = previous.get("visual_feature_space") if args.resume else None
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with (output / "metrics.jsonl").open("a") as log:
        for step in range(start, start + args.steps):
            path = paths[step % len(paths)]
            sample = load_goal_sample(path, visual=_has_visual_input(stage))
            _check_sample(sample, config, registry)
            if sample.visual is not None:
                space = sample.visual.metadata["feature_space_id"]
                if visual_space is not None and visual_space != space:
                    raise ValueError("visual encoder changed across samples")
                visual_space = space
            for file in _sample_files(path, sample):
                visited.setdefault(str(file.resolve()), file_sha256(file))
            optimizer.zero_grad(set_to_none=True)
            losses = goal_training_loss(native, interface, unused, sample, config, generators,
                                        stage=stage, feature_layers=layers)
            if not torch.isfinite(losses["total"]):
                raise ValueError("nonfinite objective; optimizer not updated")
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, config["training"]["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            log.write(json.dumps({**{key: float(value.detach()) for key, value in losses.items()},
                "stage": stage, "step": step, "updated": True, "gradient_norm": float(norm),
                "goal_source": registry["goal_source"], "goal_input": ("true_goal" if stage == "goal" else
                    "observed_context" if stage == "joint" else "generated_future_only"),
                "interface_type": config["interface_type"], "raw_video_to_action": _raw_visual_condition(config, stage),
                "video_generation": stage == "visual", "video_supervision": stage == "visual"},
                allow_nan=False) + "\n")
            log.flush()
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    numpy_state = np.random.get_state()
    goal_updates = updates if stage == "goal" else previous["goal_stage_updates"] if previous else 0
    payload = {"format_version": 2, "kind": "se3_goal_training", "architecture": goal_architecture(config),
        "precision": "float32", "upstream_commit": ZERO_WAM_COMMIT, "config": config, "stage": stage,
        "model": _system_state(native, interface), "native_config": {k: v for k, v in dict(native.config).items() if not k.startswith("_")},
        "base_identity": identity, "tiny_native": args.tiny_native, "feature_layers": layers,
        "registry": registry, "visual_feature_space": visual_space, "updates": updates, "goal_stage_updates": goal_updates,
        "attempted_steps": start + args.steps, "data_cursor": (start + args.steps) % len(paths),
        "seed": args.seed, "optimizer": optimizer.state_dict(), "scheduler": {"kind": "constant", "state": None},
        "rng": {key: gen.get_state() for key, gen in generators.items()}, "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], "python_rng": random.getstate(),
        "numpy_rng": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "data_identity": identities, "visited_arrays": visited,
        "stage1_artifact_sha256": (file_sha256(args.initialize) if args.initialize else previous.get("stage1_artifact_sha256") if previous else None),
        "source_records": previous["source_records"] if args.resume else [*(previous["source_records"] if previous else []), *sources]}
    temp = output / "goal_interface.pt.tmp"
    torch.save(payload, temp)
    temp.replace(output / "goal_interface.pt")
    report = {"artifact": str((output / "goal_interface.pt").resolve()), "stage": stage,
        "updates": updates, "goal_stage_updates": goal_updates, "elapsed_seconds": elapsed,
        "trainable_parameters": sum(p.numel() for p in parameters), "precision": "float32", "lora": False,
        "optimizer_groups": [{"name": g["name"], "lr": g["lr"], "parameters": sum(p.numel() for p in g["params"])} for g in optimizer.param_groups],
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if str(args.device).startswith("cuda") else None,
        "interface_type": config["interface_type"], "raw_video_to_action": _raw_visual_condition(config, stage),
        "video_generation": stage == "visual", "video_supervision": stage == "visual",
        "ground_truth_future_for_goal": False, "robot_execution_evaluated": False}
    write_json(output / "run.json", report)
    return report


def export_goal_policy(args):
    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file

    payload = read_goal_artifact(args.artifact)
    if payload["config"]["interface_type"] in G_PI_ROUTES:
        from .g_pi_training import export_g_pi_policy
        return export_g_pi_policy(args, payload=payload)
    _require_trained_policy(payload)
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("export requires a fresh directory")
    precision = getattr(args, "dtype", "float32")
    if precision not in {"float32", "bfloat16"}:
        raise ValueError("deployment dtype must be float32 or bfloat16")
    dtype = getattr(torch, precision)
    state = {f"{module}.{key}": value.to(dtype=dtype) if value.is_floating_point() else value
             for module, values in payload["model"].items() for key, value in values.items()}
    split = split_torch_state_dict_into_shards(state, filename_pattern="model{suffix}.safetensors",
                                              max_shard_size=getattr(args, "max_shard_size", "2GB"))
    output.mkdir(parents=True, exist_ok=True)
    files, weight_map = {}, {}
    for filename, keys in split.filename_to_tensors.items():
        # Cloning this shard breaks upstream state-dict aliases without retaining another full model.
        save_file({key: state[key].contiguous().clone() for key in keys}, str(output / filename))
        files[filename] = file_sha256(output / filename)
        weight_map.update({key: filename for key in keys})
    excluded = {"model", "optimizer", "scheduler", "rng", "torch_rng", "cuda_rng", "python_rng", "numpy_rng",
                "visited_arrays", "data_identity", "data_cursor"}
    policy = {key: value for key, value in payload.items() if key not in excluded}
    policy.update(kind="se3_goal_policy", precision=precision, shards=files, weight_map=weight_map,
                  source_artifact_sha256=file_sha256(args.artifact))
    write_json(output / "policy.json", policy)
    return {"policy": str(output.resolve()), "test_time_updates": False,
            "raw_video_to_action": _raw_visual_condition(payload["config"], payload["stage"]), "precision": precision,
            "video_generation": payload["stage"] == "visual"}


def load_goal_policy(path, *, device="cuda"):
    from safetensors.torch import load_file

    folder = Path(path)
    payload = json.loads((folder / "policy.json").read_text())
    if payload.get("kind") == "g_pi_policy":
        from .g_pi_training import load_g_pi_policy
        return load_g_pi_policy(path, device=device)
    if (payload.get("kind") != "se3_goal_policy" or payload.get("format_version") != 2
            or not isinstance(payload.get("config"), dict)
            or payload.get("architecture") != goal_architecture(payload["config"])
            or payload.get("upstream_commit") != ZERO_WAM_COMMIT
            or payload.get("precision") not in {"float32", "bfloat16"}):
        raise ValueError("expected a version-2 complete goal policy")
    validate_goal_config(payload["config"])
    _require_trained_policy(payload)
    dtype = getattr(torch, payload["precision"])
    native = load_native_class()(**payload["native_config"]).to(device=device, dtype=dtype)
    install_action_interface(native)
    interface = _interface(native, payload["config"], payload["registry"])
    state = {}
    for filename, digest in payload["shards"].items():
        shard = _local_path(folder, filename, ".safetensors")
        if file_sha256(shard) != digest:
            raise ValueError("policy shard checksum mismatch")
        tensors = load_file(str(shard))
        if any(key in state or payload["weight_map"].get(key) != filename for key in tensors):
            raise ValueError("policy shard registry mismatch")
        state.update(tensors)
    if state.keys() != payload["weight_map"].keys():
        raise ValueError("incomplete policy shards")
    model = {module: {key[len(module) + 1:]: value for key, value in state.items() if key.startswith(module + ".")}
             for module in ("native", "interface")}
    if sum(map(len, model.values())) != len(state):
        raise ValueError("unknown policy module")
    _restore_system(native, interface, model)
    if payload["feature_layers"] != list(range(len(native.blocks))):
        raise ValueError("policy coupling layers differ from model")
    native.eval().requires_grad_(False)
    interface.eval().requires_grad_(False)
    return native, interface, None, payload


@torch.no_grad()
def predict_goal_actions(native, interface, unused, payload, observation, *, seed=0):
    del unused
    if not isinstance(observation, GoalObservation):
        raise ValueError("prediction accepts observed-only GoalObservation, not supervised samples")
    if any(p.requires_grad for m in (native, interface) for p in m.parameters()):
        raise ValueError("freeze the complete policy before inference")
    registry, config = payload["registry"], payload["config"]
    for name in registry.keys() - {"goal_source", "language_identity"}:
        if observation.metadata.get(name) != registry[name]:
            raise ValueError(f"observed {name} differs from trained interface")
    if observation.language_identity != registry["language_identity"]:
        raise ValueError("language encoder differs from trained interface")
    if observation.metadata["feature_space_id"] != payload["visual_feature_space"]:
        raise ValueError("visual encoder differs from trained interface")
    weight = native.action_embedder.weight
    with autocast_for(native):
        decoded = None
        if config["interface_type"] == "observed_dual":
            conditions, decoded = observed_goal_conditions(native, interface, observation.demonstration,
                observation.history, observation.state.to(weight), observation.language.to(weight), config,
                feature_layers=payload["feature_layers"], demo_times=observation.demonstration_times)
            tokens, future = None, None
        else:
            conditions, tokens, future = visual_goal_tokens(native, interface, observation.demonstration,
                observation.history, observation.state.to(weight), observation.language.to(weight), config,
                torch.Generator().manual_seed(seed + 1), feature_layers=payload["feature_layers"],
                current_time=observation.metadata["current_time"], control_dt=registry["control_dt"],
                actions_per_frame=registry["actions_per_frame"], demo_times=observation.demonstration_times)
        shape = (1, native.config.action_dim, config["chunk_size"], registry["actions_per_frame"], 1)
        mask = torch.tensor(registry["action_space"]["valid_channels"], dtype=torch.bool)[None, :, None, None, None]
        actions = goal_action_sample(native, conditions, shape, mask, torch.Generator().manual_seed(seed),
                                     steps=config["action_sampling_steps"])
        result = {"actions": actions}
        if future is not None:
            result["generated_future"] = future
        if tokens is not None:
            decoded = interface.decode_goal(tokens)
        if decoded is not None:
            result.update(decoded)
    return result


def predict_goal_cli(args):
    manifest = json.loads((Path(args.policy) / "policy.json").read_text())
    if manifest.get("kind") == "g_pi_policy":
        from .g_pi_deployment import predict_g_pi_cli
        return predict_g_pi_cli(args)
    output = Path(args.output)
    if output.exists() or output.suffix != ".npz":
        raise ValueError("prediction requires a fresh .npz output path")
    native, interface, unused, payload = load_goal_policy(args.policy, device=args.device)
    result = predict_goal_actions(native, interface, unused, payload, load_goal_observation(args.observation), seed=args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{name: value.float().cpu().numpy() for name, value in result.items()})
    return {"output": str(output.resolve()), "coordinate_frame": payload["registry"]["coordinate_frame"],
        "goal_source": payload["registry"]["goal_source"], "test_time_updates": False, "commands_sent": 0,
        "interface_type": payload["config"]["interface_type"],
        "video_generation": payload["config"]["interface_type"] != "observed_dual"}
