"""Independent translator and hindsight goal-policy training and deployment."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .cli import file_sha256, write_json
from .goal_action import action_named_parameters, goal_action_forward, install_action_interface
from .icl_training import build_icl_model, validate_icl_config
from .video_data import _local_path, _source_components, patch_grid_coordinates
from .zerowam import DEFAULT_SOURCE, ZERO_WAM_COMMIT


ROUTES = {"g_translator": "g", "pi_goal": "pi"}


def g_pi_architecture(config):
    return {"g_translator": "g_translator_shared_v2", "pi_goal": "pi_goal_shared_v2"}[config["interface_type"]]


def _check_stage(config, stage):
    if stage != ROUTES.get(config.get("interface_type")):
        raise ValueError("g_translator requires stage g; pi_goal requires independent stage pi")


def validate_g_pi_config(config):
    from .g_pi_data import EventRules

    if (not isinstance(config, dict) or config.get("kind") != "se3_goal_experiment"
            or config.get("schema_version") != 2 or config.get("interface_type") not in ROUTES
            or "lora" in config or "demo_bottleneck" in config):
        raise ValueError("expected version-2 g_translator or pi_goal configuration without LoRA or demo bottleneck")
    validate_icl_config({**config, "kind": "native_icl_experiment", "schema_version": 1}, adaptation=False)
    if (config["domain_schedule"] != ["robot"] or config["human_context"] != "cross_video"
            or config.get("video_weight") != 0 or config["ifp"]["enabled"]
            or any(config["ifp"]["loss_weights"]) or "sampling_steps" in config):
        raise ValueError("G/pi require robot-only targets, no video generation or IFP, and video_weight=0")
    settings = config.get("goal_encoder")
    if (not isinstance(settings, dict) or set(settings) - {"layer", "k_z"}
            or type(settings.get("layer")) is not int or settings["layer"] < 0
            or type(settings.get("k_z", 8)) is not int or settings.get("k_z", 8) < 1):
        raise ValueError("goal_encoder requires a nonnegative layer and positive k_z (default 8)")
    interface = config.get("goal_interface")
    required = {"state_dim", "dim", "num_heads", "translation_scale"}
    optional = set()
    if config["interface_type"] == "g_translator":
        required |= {"num_layers"}
        optional = {"use_state"}
    else:
        required |= {"num_tokens", "num_layer_groups", "num_pose_tokens"}
    if not isinstance(interface, dict) or not required <= interface.keys() or interface.keys() - required - optional:
        raise ValueError("goal_interface fields must match the selected G/pi decoder")
    if any(type(interface[key]) is not int or interface[key] < 1 for key in required - {"translation_scale"}):
        raise ValueError("goal interface dimensions must be positive integers")
    if interface["dim"] % interface["num_heads"]:
        raise ValueError("goal interface dim must be divisible by num_heads")
    if "use_state" in interface and type(interface["use_state"]) is not bool:
        raise ValueError("goal_interface.use_state must be Boolean")
    if config["interface_type"] == "pi_goal" and interface["num_pose_tokens"] > interface["num_tokens"]:
        raise ValueError("num_pose_tokens must not exceed num_tokens")
    for value in (interface["translation_scale"], config.get("pose_weight")):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("translation_scale and pose_weight must be positive and finite")
    if type(config.get("action_sampling_steps")) is not int or config["action_sampling_steps"] < 1:
        raise ValueError("action_sampling_steps must be a positive integer")
    noise = config.get("goal_noise", {})
    if (not isinstance(noise, dict) or noise.keys() - {"z_std", "translation_std", "rotation_std", "gripper_std"}
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in noise.values())):
        raise ValueError("goal_noise scales must be finite nonnegative z/translation/rotation/gripper standard deviations")
    if config["interface_type"] == "g_translator" and any(noise.values()):
        raise ValueError("goal noise is only supported for hindsight pi training")
    EventRules.from_metadata(config.get("event_rules"))
    if type(config.get("base_seed", 0)) is not int or not 0 <= config.get("base_seed", 0) < 2 ** 63:
        raise ValueError("base_seed must be a nonnegative integer smaller than 2**63")
    for name in ("empty_text_emb_path", "empty_emb_path"):
        if name in config and (not isinstance(config[name], str) or not config[name].strip()):
            raise ValueError(f"{name} must be a nonempty local path")
    if ("empty_text_emb_path" in config and "empty_emb_path" in config
            and Path(config["empty_text_emb_path"]).resolve() != Path(config["empty_emb_path"]).resolve()):
        raise ValueError("empty_text_emb_path and empty_emb_path specify different embeddings")
    return config


def _interface(native, config, registry):
    from .g_pi_interface import GGoalDecoder, PiGoalInterface

    if registry["action_space"]["dimension"] != native.config.action_dim:
        raise ValueError("action dimension differs from the native checkpoint")
    if registry["language_identity"]["text_dim"] != native.config.text_dim:
        raise ValueError("language width differs from the native checkpoint")
    common = dict(native_dim=native.inner_dim, d_z=native.inner_dim,
                  effectors=len(registry["end_effectors"]), k_z=config["goal_encoder"].get("k_z", 8))
    if config["interface_type"] == "g_translator":
        interface = GGoalDecoder(**common, **config["goal_interface"])
    else:
        interface = PiGoalInterface(**common, feature_dim=native.inner_dim,
                                   num_layers=len(native.blocks), **config["goal_interface"])
        interface.control_dt = registry["control_dt"]
        interface.latent_frame_dt = registry["actions_per_frame"] * registry["control_dt"]
    return interface.to(device=native.action_embedder.weight.device, dtype=torch.float32)


def _set_precision(native, *, video_precision, route):
    """Convert frozen storage without rounding the FP32 action master weights."""
    if video_precision not in {"float32", "bfloat16"} or route not in ROUTES:
        raise ValueError("G/pi require float32 or bfloat16 video and an explicit route")
    action = {id(parameter) for _, parameter in action_named_parameters(native)} if route == "pi_goal" else set()
    dtype = getattr(torch, video_precision)
    for parameter in native.parameters():
        parameter.data = parameter.data.to(dtype=torch.float32 if id(parameter) in action else dtype)
    for module in native.modules():
        for name, buffer in module.named_buffers(recurse=False):
            if buffer.is_floating_point():
                module._buffers[name] = buffer.to(dtype=dtype)
    return native


def _set_training(native, interface, encoder, config):
    native.eval().requires_grad_(False)
    encoder.eval().requires_grad_(False)
    interface.train().requires_grad_(True)
    if config["interface_type"] == "pi_goal":
        for _, parameter in action_named_parameters(native):
            parameter.requires_grad_(True)


def build_g_pi_system(config, registry, *, stage, checkpoint=None, tiny_native=False, device="cuda",
                      video_precision=None):
    from .g_pi_context import FrozenGoalEncoder, install_empty_text

    validate_g_pi_config(config)
    _check_stage(config, stage)
    source = config.get("empty_text_emb_path", config.get("empty_emb_path"))
    if source is not None and not Path(source).is_file():
        raise ValueError("configured empty text embedding must be an existing local file")
    video_precision = video_precision or ("bfloat16" if str(device).startswith("cuda") else "float32")
    # Build once on CPU. Fixed tiny-base randomness is independent of training
    # RNG, and mixed storage is established before the sole device transfer.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.get("base_seed", 0))
        native, null, identity = build_icl_model(config, checkpoint=checkpoint, tiny_native=tiny_native,
            device="cpu", adaptation=False, dtype=torch.float32, empty_text_path=source)
    identity = json.loads(json.dumps(identity))
    if tiny_native:
        identity["base_seed"] = config.get("base_seed", 0)
    install_action_interface(native)
    _set_precision(native, video_precision=video_precision, route=config["interface_type"])
    if source is not None:
        path = Path(source).resolve()
        if tiny_native:
            null = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(null, torch.Tensor) and null.ndim == 2:
                null = null[None]
        text_identity = {"kind": "zerowam-empty-text", "path": str(path), "sha256": file_sha256(path)}
    elif tiny_native:
        text_identity = {"kind": "tiny-native-empty-text-fixture"}
    else:
        path = DEFAULT_SOURCE / "wan_va/assets/empty_text_emb.pt"
        text_identity = {"kind": "zerowam-empty-text", "path": str(path.resolve()), "sha256": file_sha256(path)}
    install_empty_text(native, null, text_identity)
    native.eval().requires_grad_(False)
    native.to(device=device)
    encoder = FrozenGoalEncoder(native, layer=config["goal_encoder"]["layer"],
        k_z=config["goal_encoder"].get("k_z", 8), base_id=identity)
    interface = _interface(native, config, registry)
    _set_training(native, interface, encoder, config)
    return native, interface, encoder, identity, list(range(len(native.blocks)))


def _check_sample(sample, config, registry):
    from .goal_training import goal_registry

    if goal_registry(sample) != registry:
        raise ValueError("G/pi robot, language, coordinate or event conventions changed")
    if sample.state.shape[-1] != config["goal_interface"]["state_dim"]:
        raise ValueError("state width differs from configuration")
    if sample.actions.shape[2] != config["chunk_size"]:
        raise ValueError("actions must contain exactly one configured chunk")
    if sample.metadata["event_rules"] != config["event_rules"]:
        raise ValueError("event_rules differ between configuration and task data")


def noisy_goal(goal, generator, settings):
    from .g_pi_interface import perturb_goal

    return perturb_goal(goal, generator=generator, **settings)


def g_pi_training_loss(native, interface, encoder, sample, config, generators, *, stage, feature_layers):
    from .g_pi_context import g_context_features, pi_context_features
    from .g_pi_interface import g_goal_loss
    from .goal_training import _action_noise, autocast_for, masked_action_loss

    _check_stage(config, stage)
    if feature_layers != list(range(len(native.blocks))):
        raise ValueError("G/pi feature layers must follow every native block")
    weight = native.action_embedder.weight
    state = sample.state.to(device=weight.device, dtype=torch.float32)
    # E shares the immutable video branch and owns its fixed compute precision.
    with torch.no_grad(), torch.autocast(device_type=weight.device.type, enabled=False):
        goal = {"z": encoder(sample.target_frame), "goal_poses": sample.goal_poses.to(weight.device),
                "goal_gripper": sample.goal_gripper.to(weight.device)}
    with autocast_for(native):
        if stage == "g":
            features = g_context_features(native, sample.demonstration, sample.history, config)
            prediction = interface(features[config["goal_encoder"]["layer"]], state)
            return g_goal_loss(prediction, goal, translation_scale=interface.translation_scale,
                               pose_weight=config["pose_weight"])
        features = pi_context_features(encoder.native, sample.history, config)
        goal = noisy_goal(goal, generators["goal"], config.get("goal_noise", {}))
        language = native.condition_embedder_action.text_embedder(sample.language.to(weight))
        coordinates = patch_grid_coordinates([sample.history.shape[-2] // native.patch_size[1],
                                               sample.history.shape[-1] // native.patch_size[2]]).to(weight.device)
        conditions = interface.conditions(features, state, language, goal, sample.history_times, coordinates)
        noisy, times, target, mask = _action_noise(sample.actions, sample.actions_mask, generators["action"], weight.device)
        prediction = goal_action_forward(native, noisy, times, conditions)
        action = masked_action_loss(prediction, target, mask)
        return {"action": action, "total": action}


def _optimizer(native, interface, encoder, config):
    parameters = [p for p in interface.parameters() if p.requires_grad]
    if config["interface_type"] == "pi_goal":
        parameters += [p for _, p in action_named_parameters(native) if p.requires_grad]
    expected = {id(p) for module in (native, interface, encoder) for p in module.parameters() if p.requires_grad}
    if {id(p) for p in parameters} != expected or len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("optimizer ownership must contain only the selected goal decoder or pi action interface")
    if any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("the frozen target encoder cannot enter an optimizer")
    return torch.optim.AdamW([{"params": parameters, "lr": config["training"]["learning_rate"],
                              "name": ROUTES[config["interface_type"]]}], weight_decay=0.)


def _action_state(native, interface):
    from .g_pi_interface import PiGoalInterface

    return dict(action_named_parameters(native)) if isinstance(interface, PiGoalInterface) else {}


def _system_state(native, interface, encoder):
    if encoder.state_dict():
        raise ValueError("E must share the base without registering or serializing its weights")
    modules = {"interface": interface.state_dict(), "action": _action_state(native, interface)}
    return {name: {key: value.detach().cpu() for key, value in values.items()} for name, values in modules.items()}


def _restore_system(native, interface, encoder, model):
    if encoder.state_dict() or not isinstance(model, dict) or set(model) != {"interface", "action"}:
        raise ValueError("G/pi artifacts contain only trainable interface/action weights, never frozen native or E weights")
    for name, expected in (("interface", interface.state_dict()), ("action", _action_state(native, interface))):
        if (model[name].keys() != expected.keys() or any(model[name][key].shape != value.shape
                or model[name][key].dtype != value.dtype for key, value in expected.items())):
            raise ValueError("G/pi trainable keys, shapes and FP32 precision must match")
        with torch.no_grad():
            for key, value in expected.items():
                value.copy_(model[name][key])


def _frozen_checksums(native, encoder, config):
    from .g_pi_context import assert_frozen_base, frozen_base_checksum

    assert_frozen_base(encoder.native)
    action = {id(p) for _, p in action_named_parameters(native)} if config["interface_type"] == "pi_goal" else set()
    if any(p.requires_grad for p in native.parameters() if id(p) not in action):
        raise ValueError("video backbone parameters must all be frozen")
    if encoder.native is not native:
        raise ValueError("E and G/pi must share the same frozen video base")
    checksum = frozen_base_checksum(native)
    return {"encoder": checksum, "native_video": checksum}


def _base_reference(config, checkpoint, tiny_native, identity, encoder):
    return {"kind": "tiny-native" if tiny_native else "local-checkpoint",
            "checkpoint": None if tiny_native else str(Path(checkpoint).resolve()),
            "base_seed": config.get("base_seed", 0), "identity": identity,
            "empty_text_identity": encoder.identity["empty_text_identity"],
            "encoder_identity": encoder.identity,
            "video_precision": str(encoder.native.patch_embedding_mlp.weight.dtype).removeprefix("torch.")}


def _base_location(reference, checkpoint=None):
    if reference["kind"] == "tiny-native":
        if checkpoint is not None:
            raise ValueError("tiny-native artifacts rebuild their fixed base_seed and cannot use another checkpoint")
        return None
    path = Path(checkpoint or reference["checkpoint"])
    folder = path / "transformer" if (path / "transformer").is_dir() else path
    expected = reference["identity"].get("sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("real base reference must record native checkpoint file hashes")
    paths = [folder / name for name in expected]
    actual_names = {p.name for p in folder.glob("*.safetensors")} | {"config.json"} | {
        p.name for p in folder.glob("*.safetensors.index.json")}
    if (actual_names != set(expected) or any(not p.is_file() or file_sha256(p) != expected[p.name] for p in paths)):
        raise ValueError("referenced base checkpoint is missing or its file hashes changed")
    return str(path)


def validate_g_pi_artifact(payload, *, kind="g_pi_training", expected_encoder_identity=None):
    if (not isinstance(payload, dict) or payload.get("kind") != kind or payload.get("format_version") != 2
            or not isinstance(payload.get("config"), dict) or payload.get("upstream_commit") != ZERO_WAM_COMMIT
            or payload.get("precision") != "float32" or payload.get("encoder_precision") not in {"float32", "bfloat16"}):
        raise ValueError("expected a version-2 lightweight G/pi artifact with FP32 trainables and a shared frozen base reference")
    config = validate_g_pi_config(payload["config"])
    _check_stage(config, payload.get("stage"))
    identity = payload.get("encoder_identity")
    if (payload.get("architecture") != g_pi_architecture(config)
            or payload.get("interface_type") != config["interface_type"]
            or not isinstance(identity, dict) or identity.get("layer") != config["goal_encoder"]["layer"]
            or identity.get("k_z") != config["goal_encoder"].get("k_z", 8)
            or payload.get("k_z") != identity.get("k_z") or payload.get("d_z") != identity.get("d_z")
            or type(identity.get("d_z")) is not int or identity["d_z"] < 1
            or identity.get("timestep") != 0 or identity.get("normalization") != "l2_last_dim"
            or identity.get("pooling") != "adaptive_avg_pool1d_spatial"
            or identity.get("base_id") != payload.get("base_identity")
            or not identity.get("empty_text_identity")
            or identity.get("empty_text_identity") != payload.get("empty_text_identity")
            or not identity.get("base_sha256") or payload.get("event_rules") != config["event_rules"]
            or payload.get("registry", {}).get("event_rules") != config["event_rules"]):
        raise ValueError("G/pi route, E identity, dimensions or event rules do not match the artifact")
    if expected_encoder_identity is not None and identity != expected_encoder_identity:
        raise ValueError("E identity mismatch")
    reference = payload.get("base_reference")
    if (not isinstance(reference, dict) or reference.get("kind") not in {"tiny-native", "local-checkpoint"}
            or reference.get("kind") != ("tiny-native" if payload.get("tiny_native") else "local-checkpoint")
            or reference.get("identity") != payload["base_identity"]
            or reference.get("encoder_identity") != identity
            or reference.get("empty_text_identity") != payload["empty_text_identity"]
            or reference.get("video_precision") != payload["encoder_precision"]
            or reference.get("base_seed") != config.get("base_seed", 0)
            or (reference["kind"] == "local-checkpoint" and not isinstance(reference.get("checkpoint"), str))):
        raise ValueError("base reference, shared E identity, precision or base_seed mismatch")
    if kind == "g_pi_training":
        model = payload.get("model")
        if (not isinstance(model, dict) or set(model) != {"interface", "action"}
                or not model["interface"] or (config["interface_type"] == "g_translator" and model["action"])
                or any(not isinstance(v, torch.Tensor) or v.dtype != torch.float32
                       for values in model.values() for v in values.values())):
            raise ValueError("lightweight artifacts contain FP32 interface/action trainables only; frozen weights are forbidden")
    return payload


def read_g_pi_artifact(path, *, payload=None, expected_encoder_identity=None):
    if payload is None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    return validate_g_pi_artifact(payload, expected_encoder_identity=expected_encoder_identity)


def train_g_pi_interface(args):
    from .g_pi_data import g_pi_sample_files, load_g_pi_index, load_g_pi_sample
    from .goal_training import goal_registry

    config = validate_g_pi_config(json.loads(Path(args.config).read_text()))
    stage = args.stage
    _check_stage(config, stage)
    if getattr(args, "initialize", None):
        raise ValueError("G and pi train independently; stage initialization or joint training is not supported")
    if type(args.steps) is not int or args.steps < 1:
        raise ValueError("steps must be positive")
    paths, sources = load_g_pi_index(args.index)
    first = load_g_pi_sample(paths[0], route=config["interface_type"], generator=torch.Generator().manual_seed(args.seed))
    registry = goal_registry(first)
    _check_sample(first, config, registry)
    previous = read_g_pi_artifact(args.resume) if args.resume else None
    identities = {"index": file_sha256(args.index), **{str(p.resolve()): file_sha256(p) for p in paths}}
    start = previous["attempted_steps"] if previous else 0
    if start + args.steps > config["training"]["max_steps"]:
        raise ValueError("cumulative independent stage budget exceeded")
    if previous and (previous["config"] != config or previous["stage"] != stage
            or previous["registry"] != registry or previous["tiny_native"] != args.tiny_native
            or previous["data_identity"] != identities or previous["seed"] != args.seed):
        raise ValueError("resume requires the same route, config, data, robot conventions and seed")
    if previous:
        _source_components([*previous["source_records"], *sources])
    visited = dict(previous["visited_arrays"]) if previous else {}
    if any(file_sha256(path) != digest for path, digest in visited.items()):
        raise ValueError("consumed training inputs changed before resume")
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not previous:
        raise ValueError("training requires a fresh output directory or explicit resume")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    checkpoint = _base_location(previous["base_reference"], args.checkpoint) if previous else args.checkpoint
    native, interface, encoder, identity, layers = build_g_pi_system(config, registry, stage=stage,
        checkpoint=checkpoint, tiny_native=args.tiny_native, device=args.device,
        video_precision=previous["encoder_precision"] if previous else None)
    if previous:
        if previous["base_identity"] != identity or previous["feature_layers"] != layers:
            raise ValueError("base checkpoint or feature layers changed")
        encoder.validate_identity(previous["encoder_identity"])
        _restore_system(native, interface, encoder, previous["model"])
    frozen = _frozen_checksums(native, encoder, config)
    if frozen["encoder"] != encoder.identity["base_sha256"]:
        raise ValueError("E weights differ from the recorded frozen encoder identity")
    if previous and frozen != previous["frozen_checksums"]:
        raise ValueError("frozen backbone checksum changed in the resumed artifact")
    optimizer = _optimizer(native, interface, encoder, config)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    generators = {name: torch.Generator().manual_seed(args.seed + offset)
                  for offset, name in enumerate(("sample", "action", "goal"))}
    updates = 0
    if previous:
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
    visual_space = previous.get("visual_feature_space") if previous else None
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    started = time.monotonic()
    with (output / "metrics.jsonl").open("a") as log:
        for step in range(start, start + args.steps):
            path = paths[step % len(paths)]
            sample = load_g_pi_sample(path, route=config["interface_type"], generator=generators["sample"])
            _check_sample(sample, config, registry)
            space = sample.metadata["feature_space_id"]
            if visual_space is not None and visual_space != space:
                raise ValueError("visual feature space changed across tasks")
            visual_space = space
            for file in g_pi_sample_files(path, sample):
                name, digest = str(file.resolve()), file_sha256(file)
                if name in visited and visited[name] != digest:
                    raise ValueError("training input changed after it was consumed")
                visited[name] = digest
            optimizer.zero_grad(set_to_none=True)
            losses = g_pi_training_loss(native, interface, encoder, sample, config, generators,
                                       stage=stage, feature_layers=layers)
            if not torch.isfinite(losses["total"]):
                raise ValueError("nonfinite objective; optimizer not updated")
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, config["training"]["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            log.write(json.dumps({**{key: float(value.detach()) for key, value in losses.items()},
                "step": step, "stage": stage, "interface_type": config["interface_type"],
                "gradient_norm": float(norm), "current_time": sample.metadata["current_time"],
                "subgoal_time": sample.subgoal_time, "terminal_time": sample.terminal_time,
                "video_generation": False, "video_supervision": False, "updated": True}, allow_nan=False) + "\n")
            log.flush()
    if _frozen_checksums(native, encoder, config) != frozen:
        raise ValueError("frozen video backbone or E changed during training")
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    n = np.random.get_state()
    reference = _base_reference(config, checkpoint, args.tiny_native, identity, encoder)
    payload = {"format_version": 2, "kind": "g_pi_training", "architecture": g_pi_architecture(config),
        "precision": "float32", "encoder_precision": reference["video_precision"], "upstream_commit": ZERO_WAM_COMMIT,
        "config": config, "stage": stage, "interface_type": config["interface_type"],
        "model": _system_state(native, interface, encoder),
        "native_config": {k: v for k, v in dict(native.config).items() if not k.startswith("_")},
        "base_identity": identity, "base_reference": reference, "encoder_identity": encoder.identity,
        "empty_text_identity": encoder.identity["empty_text_identity"],
        "k_z": encoder.identity["k_z"], "d_z": encoder.identity["d_z"], "event_rules": config["event_rules"],
        "tiny_native": args.tiny_native, "feature_layers": layers, "registry": registry,
        "frozen_checksums": frozen, "visual_feature_space": visual_space, "updates": updates,
        "attempted_steps": start + args.steps, "data_cursor": (start + args.steps) % len(paths), "seed": args.seed,
        "optimizer": optimizer.state_dict(), "scheduler": {"kind": "constant", "state": None},
        "rng": {key: gen.get_state() for key, gen in generators.items()}, "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(), "numpy_rng": [n[0], n[1].tolist(), *n[2:]],
        "data_identity": identities, "visited_arrays": visited,
        "source_records": previous["source_records"] if previous else sources}
    temp = output / "goal_interface.pt.tmp"
    torch.save(payload, temp)
    temp.replace(output / "goal_interface.pt")
    report = {"artifact": str((output / "goal_interface.pt").resolve()), "stage": stage, "updates": updates,
        "interface_type": config["interface_type"], "elapsed_seconds": elapsed,
        "trainable_parameters": sum(p.numel() for p in parameters), "precision": "float32", "lora": False,
        "encoder_precision": reference["video_precision"], "base_reference": reference,
        "shared_video_base": encoder.native is native,
        "encoder_identity": encoder.identity, "frozen_checksums": frozen,
        "optimizer_groups": [{"name": g["name"], "lr": g["lr"],
                              "parameters": sum(p.numel() for p in g["params"])} for g in optimizer.param_groups],
        "video_generation": False, "video_supervision": False, "robot_execution_evaluated": False}
    write_json(output / "run.json", report)
    return report


def export_g_pi_policy(args, *, payload=None):
    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file

    payload = read_g_pi_artifact(args.artifact, payload=payload)
    if payload["updates"] < 1:
        raise ValueError("G/pi export requires a successfully trained independent route")
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("export requires a fresh directory")
    requested = getattr(args, "dtype", "float32")
    if requested not in {"float32", "bfloat16"}:
        raise ValueError("deployment dtype must be float32 or bfloat16")
    # The legacy dtype flag is accepted, but v2 never quantizes trainables or
    # embeds a second copy of frozen video weights into an exported policy.
    state = {f"{module}.{key}": value
             for module, values in payload["model"].items() for key, value in values.items()}
    split = split_torch_state_dict_into_shards(state, filename_pattern="model{suffix}.safetensors",
                                              max_shard_size=getattr(args, "max_shard_size", "2GB"))
    output.mkdir(parents=True, exist_ok=True)
    files, weight_map = {}, {}
    for filename, keys in split.filename_to_tensors.items():
        save_file({key: state[key].contiguous().clone() for key in keys}, str(output / filename))
        files[filename] = file_sha256(output / filename)
        weight_map.update({key: filename for key in keys})
    excluded = {"model", "optimizer", "scheduler", "rng", "torch_rng", "cuda_rng", "python_rng", "numpy_rng",
                "visited_arrays", "data_identity", "data_cursor"}
    policy = {key: value for key, value in payload.items() if key not in excluded}
    policy.update(kind="g_pi_policy", precision="float32", requested_dtype=requested, shards=files, weight_map=weight_map,
                  source_artifact_sha256=file_sha256(args.artifact))
    write_json(output / "policy.json", policy)
    return {"policy": str(output.resolve()), "interface_type": payload["interface_type"],
            "precision": "float32", "requested_dtype": requested,
            "encoder_precision": payload["encoder_precision"], "requires_base_checkpoint": not payload["tiny_native"],
            "test_time_updates": False,
            "video_generation": False}


def load_g_pi_policy(path, *, device="cuda", checkpoint=None, expected_encoder_identity=None):
    from safetensors.torch import load_file
    from .g_pi_context import frozen_base_checksum

    folder = Path(path)
    payload = validate_g_pi_artifact(json.loads((folder / "policy.json").read_text()), kind="g_pi_policy",
                                    expected_encoder_identity=expected_encoder_identity)
    if payload["updates"] < 1:
        raise ValueError("G/pi policy must contain successful training updates")
    state = {}
    for filename, digest in payload["shards"].items():
        shard = _local_path(folder, filename, ".safetensors")
        if file_sha256(shard) != digest:
            raise ValueError("G/pi policy shard checksum mismatch")
        tensors = load_file(str(shard))
        if any(key in state or payload["weight_map"].get(key) != filename for key in tensors):
            raise ValueError("G/pi policy shard registry mismatch")
        state.update(tensors)
    if state.keys() != payload["weight_map"].keys():
        raise ValueError("incomplete G/pi policy shards")
    model = {module: {key[len(module) + 1:]: value for key, value in state.items() if key.startswith(module + ".")}
             for module in ("interface", "action")}
    if sum(map(len, model.values())) != len(state):
        raise ValueError("unknown G/pi policy module")
    checkpoint = _base_location(payload["base_reference"], checkpoint)
    native, interface, encoder, identity, layers = build_g_pi_system(payload["config"], payload["registry"],
        stage=payload["stage"], checkpoint=checkpoint, tiny_native=payload["tiny_native"], device=device,
        video_precision=payload["encoder_precision"])
    if identity != payload["base_identity"]:
        raise ValueError("base checkpoint identity changed before loading the trainables")
    encoder.validate_identity(payload["encoder_identity"])
    _restore_system(native, interface, encoder, model)
    if frozen_base_checksum(encoder.native) != payload["encoder_identity"]["base_sha256"]:
        raise ValueError("E weights differ from the recorded identity")
    if payload["feature_layers"] != layers:
        raise ValueError("policy feature layers differ from native blocks")
    for module in (native, interface, encoder):
        module.eval().requires_grad_(False)
    return native, interface, encoder, payload
