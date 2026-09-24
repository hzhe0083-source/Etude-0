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
    return {"g_translator": "g_translator_intent_v4", "pi_goal": "pi_goal_endpoint_v4"}[config["interface_type"]]


def g_pi_artifact_version(config):
    return 4


def intent_training_settings(config):
    settings = config.get("intent_training", {})
    mode = config.get("goal_interface", {}).get("intent_mode", "connected")
    return {"manifest": None, "data_version": None, "groups_per_batch": 2, "samples_per_group": 2,
            "contrastive_weight": 0. if mode == "regression_only" or not settings.get("manifest") else 1.,
            "temperature": .1, **settings}


def conditioning_mode(config):
    return {"g_translator": "demo_robot_state_internal_empty",
            "pi_goal": "language_state_goal"}[config["interface_type"]]


def language_drop_probability(config):
    return config.get("p_drop", 0.)


def pi_training_settings(config):
    values = config.get("pi_training", {})
    if (not isinstance(values, dict) or values.keys() - {"pose_weight", "ablations", "goal_source"}):
        raise ValueError("pi_training requires pose_weight, ablations and goal_source only")
    ablations = values.get("ablations", {})
    if (not isinstance(ablations, dict) or ablations.keys() - {"no_stage1", "no_lit_pose", "exact_goal"}
            or any(type(value) is not bool for value in ablations.values())):
        raise ValueError("pi ablations require Boolean no_stage1, no_lit_pose and exact_goal")
    result = {"pose_weight": .3, "goal_source": "hindsight", **values,
              "ablations": {"no_stage1": False, "no_lit_pose": False, "exact_goal": False, **ablations}}
    weight = result["pose_weight"]
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError("pi pose_weight must be finite and nonnegative")
    if result["ablations"]["no_lit_pose"]:
        result["pose_weight"] = 0.
    elif weight == 0:
        raise ValueError("zero pi pose_weight requires explicit no_lit_pose ablation")
    if result["goal_source"] != "hindsight":
        raise ValueError("pi goal_source supports hindsight only; cross-fitted G predictions are reserved and unsupported")
    return result


def _check_stage(config, stage):
    valid = {"g_translator": {"g"}, "pi_goal": {"pi_prior", "pi"}}
    if stage not in valid.get(config.get("interface_type"), set()):
        raise ValueError("g_translator requires stage g; pi_goal requires stage pi_prior or pi")
    if stage == "pi" and not pi_training_settings(config)["ablations"]["exact_goal"]:
        if not any(config.get("goal_noise", {}).get(key, 0.) for key in
                   ("z_std", "translation_std", "rotation_std", "gripper_std")):
            raise ValueError("pi stage 2 requires goal noise or the explicit exact_goal ablation")
    if stage == "pi_prior":
        from .g_pi_distributed import distributed_settings

        if distributed_settings(config)["enabled"]:
            raise ValueError("pi_prior currently supports single-GPU training only, without FSDP")
        if config.get("target_cache_index"):
            raise ValueError("pi_prior never reads image targets or an E target cache")
        if pi_training_settings(config)["ablations"]["no_stage1"]:
            raise ValueError("pi_prior cannot run with the no_stage1 ablation")


def validate_g_pi_config(config):
    from .g_pi_data import EventRules
    from .g_pi_context import validate_camera_layout

    if (not isinstance(config, dict) or config.get("kind") != "se3_goal_experiment"
            or config.get("schema_version") != 3 or config.get("interface_type") not in ROUTES
            or "lora" in config or "demo_bottleneck" in config):
        raise ValueError("expected version-3 g_translator or pi_goal configuration without LoRA or demo bottleneck")
    validate_icl_config({**config, "kind": "native_icl_experiment", "schema_version": 1}, adaptation=False)
    if (config["domain_schedule"] != ["robot"] or config["human_context"] != "cross_video"
            or config.get("video_weight") != 0 or config["ifp"]["enabled"]
            or any(config["ifp"]["loss_weights"]) or "sampling_steps" in config):
        raise ValueError("G/pi require robot-only targets, no video generation or IFP, and video_weight=0")
    settings = config.get("goal_encoder")
    if (not isinstance(settings, dict) or set(settings) - {"layer", "grid_size", "camera_layout"}
            or type(settings.get("layer")) is not int or settings["layer"] < 0):
        raise ValueError("goal_encoder requires a nonnegative layer and 2D grid_size; legacy k_z is unsupported")
    validate_camera_layout(settings.get("camera_layout"))
    grid = settings.get("grid_size", [4, 4])
    if (not isinstance(grid, (list, tuple)) or len(grid) != 2
            or any(type(value) is not int or value < 1 for value in grid)):
        raise ValueError("goal_encoder.grid_size requires positive integer [height,width] (default [4,4])")
    if config.get("conditioning_mode") != conditioning_mode(config):
        raise ValueError("conditioning_mode must match the selected route")
    probability = language_drop_probability(config)
    if (type(probability) not in (int, float) or not math.isfinite(probability)
            or not 0 <= probability <= 1 or (config["interface_type"] == "g_translator" and probability != 0)):
        raise ValueError("p_drop must lie in [0,1] for pi; G never reads language and requires p_drop=0")
    interface = config.get("goal_interface")
    required = {"state_dim", "dim", "num_heads", "translation_scale"}
    optional = set()
    if config["interface_type"] == "g_translator":
        if config.get("demo_route", "one_way") not in {"one_way", "via_u_only"}:
            raise ValueError("demo_route must be one_way or via_u_only")
        required |= {"num_layers"}
        optional = {"use_state", "intent_mode", "num_intent_tokens", "num_intent_layers"}
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
    if config["interface_type"] == "g_translator":
        if interface.get("intent_mode", "connected") not in {"regression_only", "independent", "connected"}:
            raise ValueError("intent_mode must be regression_only, independent or connected")
        if config.get("demo_route", "one_way") == "via_u_only" and interface.get("intent_mode", "connected") != "connected":
            raise ValueError("via_u_only requires connected intent_mode so raw demonstrations cannot bypass u")
        for name in ("num_intent_tokens", "num_intent_layers"):
            if name in interface and (type(interface[name]) is not int or interface[name] < 1):
                raise ValueError("intent query count and layer count must be positive integers")
        settings = config.get("intent_training", {})
        if not isinstance(settings, dict) or settings.keys() - intent_training_settings({}).keys():
            raise ValueError("intent_training requires manifest, group/sample counts, contrastive_weight and temperature")
        values = intent_training_settings(config)
        if values["data_version"] not in {None, "v1", "v2"}:
            raise ValueError("intent data_version must be v1 or v2")
        if values["manifest"] is not None and (not isinstance(values["manifest"], str) or not values["manifest"].strip()):
            raise ValueError("intent manifest must be a nonempty local path")
        if any(type(values[name]) is not int or values[name] < 2 for name in ("groups_per_batch", "samples_per_group")):
            raise ValueError("contrastive batches require at least two groups and two independent samples per group")
        if (type(values["contrastive_weight"]) not in (int, float) or not math.isfinite(values["contrastive_weight"])
                or values["contrastive_weight"] < 0 or type(values["temperature"]) not in (int, float)
                or not math.isfinite(values["temperature"]) or values["temperature"] <= 0):
            raise ValueError("contrastive_weight must be finite nonnegative and temperature finite positive")
        if values["contrastive_weight"] and (values["manifest"] is None or interface.get("intent_mode") == "regression_only"):
            raise ValueError("positive contrastive_weight requires an intent manifest and an enabled intent head")
    elif "intent_training" in config:
        raise ValueError("pi never sees demonstrations or offline intent groups")
    if config["interface_type"] == "pi_goal" and interface["num_pose_tokens"] > interface["num_tokens"]:
        raise ValueError("num_pose_tokens must not exceed num_tokens")
    for value in (interface["translation_scale"], config.get("pose_weight")):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("translation_scale and pose_weight must be positive and finite")
    if type(config.get("action_sampling_steps")) is not int or config["action_sampling_steps"] < 1:
        raise ValueError("action_sampling_steps must be a positive integer")
    noise = config.get("goal_noise", {})
    if config["interface_type"] == "pi_goal":
        from .g_pi_noise import validate_goal_noise

        pi_training_settings(config)
        validate_goal_noise(noise, candidate_separation_m=config.get("candidate_separation_m"))
    else:
        if "pi_training" in config or "candidate_separation_m" in config:
            raise ValueError("G does not use pi stage, endpoint or goal-noise training settings")
        if (not isinstance(noise, dict) or noise.keys() - {"z_std", "translation_std", "rotation_std", "gripper_std"}
                or any(type(v) not in (int, float) or not math.isfinite(v) or v != 0 for v in noise.values())):
            raise ValueError("goal noise is only supported for hindsight pi training")
    from .g_pi_distributed import distributed_settings

    distributed_settings(config)
    if "target_cache_index" in config and (not isinstance(config["target_cache_index"], str) or not config["target_cache_index"].strip()):
        raise ValueError("target_cache_index must be a nonempty local index path")
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
    if config["interface_type"] == "pi_goal" and registry["language_identity"]["text_dim"] != native.config.text_dim:
        raise ValueError("language width differs from the native checkpoint")
    common = dict(native_dim=native.inner_dim, d_z=native.inner_dim,
                  effectors=len(registry["end_effectors"]), k_z=len(config["goal_encoder"]["camera_layout"]) * math.prod(config["goal_encoder"].get("grid_size", [4, 4])))
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


def _interface_state(interface, *, stage=None):
    from .g_pi_interface import PiGoalInterface

    values = interface.state_dict()
    if not isinstance(interface, PiGoalInterface):
        return values
    stage = stage or getattr(interface, "_g_pi_stage", "pi")
    if stage == "pi_prior":
        return {name: value for name, value in values.items()
                if name.startswith(("endpoint_encoder.", "state_encoder.", "condition_adapter."))}
    if stage == "pi":
        return {name: value for name, value in values.items() if not name.startswith("endpoint_encoder.")}
    raise ValueError("pi interface checkpoint ownership requires pi_prior or pi stage")


def _set_training(native, interface, encoder, config, stage=None):
    stage = stage or ROUTES[config["interface_type"]]
    _check_stage(config, stage)
    native.eval().requires_grad_(False)
    encoder.eval().requires_grad_(False)
    interface._g_pi_stage = stage
    interface.train().requires_grad_(True)
    if config["interface_type"] == "pi_goal":
        if stage == "pi_prior":
            interface.requires_grad_(False)
            for name in ("endpoint_encoder", "state_encoder", "condition_adapter"):
                getattr(interface, name).requires_grad_(True)
        else:
            interface.endpoint_encoder.requires_grad_(False)
        for _, parameter in action_named_parameters(native):
            parameter.requires_grad_(True)


def build_g_pi_encoder(config, *, stage, checkpoint=None, tiny_native=False, device="cuda",
                       video_precision=None):
    """Build fixed E before training, without a goal decoder or task language."""
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
        grid_size=config["goal_encoder"].get("grid_size", [4, 4]),
        camera_layout=config["goal_encoder"]["camera_layout"], base_id=identity)
    return native, encoder, identity, list(range(len(native.blocks)))


def build_g_pi_system(config, registry, *, stage, checkpoint=None, tiny_native=False, device="cuda",
                      video_precision=None):
    native, encoder, identity, layers = build_g_pi_encoder(config, stage=stage, checkpoint=checkpoint,
        tiny_native=tiny_native, device=device, video_precision=video_precision)
    interface = _interface(native, config, registry)
    _set_training(native, interface, encoder, config, stage)
    return native, interface, encoder, identity, layers


def _check_sample(sample, config, registry):
    from .goal_training import goal_registry

    if goal_registry(sample) != registry:
        raise ValueError("G/pi robot, language, coordinate or subgoal conventions changed")
    if sample.state.shape[-1] != config["goal_interface"]["state_dim"]:
        raise ValueError("state width differs from configuration")
    if sample.actions.shape[2] != config["chunk_size"]:
        raise ValueError("actions must contain exactly one configured chunk")
    if sample.metadata["event_rules"] != config["event_rules"]:
        raise ValueError("event_rules differ between configuration and task data")


def noisy_goal(goal, generator, settings, *, candidate_separation_m=None):
    from .g_pi_noise import perturb_goal_bounded

    return perturb_goal_bounded(goal, generator=generator, candidate_separation_m=candidate_separation_m, **settings)


def _training_language(native, language, generator, probability):
    """Sample once before projection so LIT and every action layer agree."""
    if (type(probability) not in (int, float) or not math.isfinite(probability)
            or not 0 <= probability <= 1):
        raise ValueError("language dropout probability must lie in [0,1]")
    if probability == 0:
        return language
    if generator is None:
        raise ValueError("language dropout requires its resumable training RNG")
    if torch.rand((), generator=generator).item() < probability:
        if not hasattr(native, "g_pi_empty_text"):
            raise ValueError("language dropout requires the pretrained empty prompt embedding")
        return native.g_pi_empty_text.detach()
    return language


def _endpoint_losses(prediction, sample, interface):
    from .goal_interface import goal_pose_loss

    anchor = (prediction["goal_poses"].sum() + prediction["goal_gripper"].sum()) * 0
    valid = sample.block_end_valid
    if valid:
        loss = goal_pose_loss(prediction["goal_poses"], sample.block_end_poses,
            translation_scale=interface.translation_scale, gripper_prediction=prediction["goal_gripper"],
            gripper_target=sample.block_end_gripper)
    else:
        loss = {name: anchor for name in ("total", "position_error_m", "orientation_error_deg", "gripper_error")}
    reaches = valid and sample.reaches_subgoal
    before = valid and not sample.reaches_subgoal
    return {"lit_pose": loss["total"],
        "lit_pose_before": loss["total"] if before else anchor,
        "lit_pose_reaches": loss["total"] if reaches else anchor,
        "lit_pose_before_count": anchor.detach() + int(before),
        "lit_pose_reaches_count": anchor.detach() + int(reaches),
        "endpoint_valid_count": anchor.detach() + int(valid),
        "endpoint_position_error_m": loss["position_error_m"],
        "endpoint_orientation_error_deg": loss["orientation_error_deg"],
        "endpoint_gripper_error": loss["gripper_error"]}


def g_pi_training_loss(native, interface, encoder, sample, config, generators, *, stage, feature_layers, cached_z=None):
    from .g_pi_context import split_g_context_features, pi_context_features
    from .g_pi_interface import g_goal_loss
    from .goal_training import _action_noise, autocast_for, masked_action_loss

    _check_stage(config, stage)
    if feature_layers != list(range(len(native.blocks))):
        raise ValueError("G/pi feature layers must follow every native block")
    weight = native.action_embedder.weight
    state = sample.state.to(device=weight.device, dtype=torch.float32)
    if stage == "pi_prior":
        if not sample.block_end_valid:
            raise ValueError("pi_prior requires a valid recorded endpoint after the last supervised action")
        if cached_z is not None:
            raise ValueError("pi_prior never consumes E targets or a hindsight subgoal")
        with autocast_for(native):
            language = _training_language(native, sample.language, generators.get("language"),
                                          language_drop_probability(config))
            language = native.condition_embedder_action.text_embedder(language.to(weight))
            conditions = interface.endpoint_conditions(sample.block_end_poses, sample.block_end_gripper,
                                                        state, language)
            noisy, times, target, mask = _action_noise(sample.actions, sample.actions_mask, generators["action"], weight.device)
            prediction = goal_action_forward(native, noisy, times, conditions)
            action = masked_action_loss(prediction, target, mask)
        return {"action": action, "total": action}
    # E shares the immutable video branch and owns its fixed compute precision.
    with torch.no_grad(), torch.autocast(device_type=weight.device.type, enabled=False):
        goal = {"z": encoder(sample.target_frame) if cached_z is None else cached_z.to(weight.device),
                "goal_poses": sample.goal_poses.to(weight.device), "goal_gripper": sample.goal_gripper.to(weight.device)}
    with autocast_for(native):
        if stage == "g":
            demo, robot = split_g_context_features(native, sample.demonstration, sample.history, config)
            layer = config["goal_encoder"]["layer"]
            prediction = interface(demo[layer], robot[layer], state)
            return g_goal_loss(prediction, goal, translation_scale=interface.translation_scale,
                               pose_weight=config["pose_weight"])
        features = pi_context_features(encoder.native, sample.history, config)
        settings = pi_training_settings(config)
        if not settings["ablations"]["exact_goal"]:
            goal = noisy_goal(goal, generators["goal"], config.get("goal_noise", {}),
                              candidate_separation_m=config.get("candidate_separation_m"))
        language = _training_language(native, sample.language, generators.get("language"),
                                      language_drop_probability(config))
        language = native.condition_embedder_action.text_embedder(language.to(weight))
        coordinates = patch_grid_coordinates([sample.history.shape[-2] // native.patch_size[1],
                                               sample.history.shape[-1] // native.patch_size[2]]).to(weight.device)
        conditions, endpoint = interface.stage2_conditions(features, state, language, goal,
                                                           sample.history_times, coordinates)
        noisy, times, target, mask = _action_noise(sample.actions, sample.actions_mask, generators["action"], weight.device)
        prediction = goal_action_forward(native, noisy, times, conditions)
        action = masked_action_loss(prediction, target, mask)
        endpoint_losses = _endpoint_losses(endpoint, sample, interface)
        return {"action": action, **endpoint_losses,
                "total": action + settings["pose_weight"] * endpoint_losses["lit_pose"]}


def g_intent_training_loss(native, interface, encoder, records, entries, config, *, uncertain_pairs=(), cached_targets=None):
    """Each record is (complete demo, paired robot sample or None)."""
    from .g_pi_context import demo_context_features, split_g_context_features
    from .g_pi_interface import g_goal_loss
    from .g_pi_intent import intent_contrastive_loss
    from .goal_training import autocast_for

    if config["interface_type"] != "g_translator" or not records or len(records) != len(entries):
        raise ValueError("G intent training requires matching nonempty demo records and offline entries")
    weight = native.action_embedder.weight
    layer = config["goal_encoder"]["layer"]
    paired, representations = [], []
    with autocast_for(native):
        for record_index, (demonstration, sample) in enumerate(records):
            if sample is None:
                if interface.intent_mode != "regression_only":
                    features = demo_context_features(native, demonstration, config)
                    representations.append(interface.encode_intent(features[layer]))
                continue
            demo, robot = split_g_context_features(native, demonstration, sample.history, config)
            prediction = interface(demo[layer], robot[layer], sample.state.to(device=weight.device, dtype=torch.float32))
            if prediction["u"] is not None:
                representations.append(prediction["u"])
            with torch.no_grad(), torch.autocast(device_type=weight.device.type, enabled=False):
                goal = {"z": encoder(sample.target_frame) if cached_targets is None else cached_targets[record_index].to(weight.device), "goal_poses": sample.goal_poses.to(weight.device),
                        "goal_gripper": sample.goal_gripper.to(weight.device)}
            paired.append(g_goal_loss(prediction, goal, translation_scale=interface.translation_scale,
                                      pose_weight=config["pose_weight"]))
        anchor = next(interface.parameters()).sum() * 0
        # Human-only examples never enter the regression denominator.
        losses = {key: torch.stack([item[key] for item in paired]).mean() for key in paired[0]} if paired else {"total": anchor}
        regression = losses.pop("total")
        settings = intent_training_settings(config)
        contrastive = (intent_contrastive_loss(torch.cat(representations), entries,
                          temperature=settings["temperature"], uncertain_pairs=uncertain_pairs)
                       if settings["contrastive_weight"] else anchor)
        return {**losses, "regression": regression, "contrastive": contrastive,
                "total": regression + settings["contrastive_weight"] * contrastive}


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
    modules = {"interface": _interface_state(interface), "action": _action_state(native, interface)}
    return {name: {key: value.detach().cpu() for key, value in values.items()} for name, values in modules.items()}


def _restore_system(native, interface, encoder, model):
    if encoder.state_dict() or not isinstance(model, dict) or set(model) != {"interface", "action"}:
        raise ValueError("G/pi artifacts contain only trainable interface/action weights, never frozen native or E weights")
    for name, expected in (("interface", _interface_state(interface)), ("action", _action_state(native, interface))):
        if (model[name].keys() != expected.keys() or any(model[name][key].shape != value.shape
                or model[name][key].dtype != value.dtype for key, value in expected.items())):
            raise ValueError("G/pi trainable keys, shapes and FP32 precision must match")
        with torch.no_grad():
            for key, value in expected.items():
                value.copy_(model[name][key])


def pi_stage_contract(config, stage, stage1_sha256=None):
    from .g_pi_noise import validate_goal_noise

    settings = pi_training_settings(config)
    return {"stage": stage, "pose_weight": settings["pose_weight"] if stage == "pi" else 0.,
            "ablations": settings["ablations"], "goal_source": settings["goal_source"],
            "goal_noise": validate_goal_noise(config.get("goal_noise", {}),
                                               candidate_separation_m=config.get("candidate_separation_m")),
            "candidate_separation_m": config.get("candidate_separation_m"),
            "goal_noise_enabled": stage == "pi" and not settings["ablations"]["exact_goal"],
            "stage1_artifact_sha256": stage1_sha256}


def _initialization(args, config, previous):
    source = getattr(args, "initialize", None)
    if source and previous:
        raise ValueError("choose either resume or stage-1 initialization")
    if source and (config["interface_type"] != "pi_goal" or args.stage != "pi"):
        raise ValueError("initialization is only pi_prior -> pi stage 2")
    if config["interface_type"] != "pi_goal" or args.stage != "pi":
        return None
    if pi_training_settings(config)["ablations"]["no_stage1"]:
        if source:
            raise ValueError("no_stage1 ablation cannot initialize from a prior checkpoint")
        return None
    if previous:
        return None
    if not source:
        raise ValueError("pi stage 2 requires a successful pi_prior --initialize artifact or explicit no_stage1 ablation")
    prior = read_g_pi_artifact(source)
    if prior["interface_type"] != "pi_goal" or prior["stage"] != "pi_prior" or prior["updates"] < 1:
        raise ValueError("pi stage 2 requires a successfully trained pi_prior artifact")
    return prior


def _initialize_pi(native, interface, encoder, config, registry, prior, identity, layers):
    keys = ("goal_interface", "goal_encoder", "base_seed", "chunk_size", "max_frame_chunk_size", "window_size", "icl_rope_h")
    if (any(prior["config"].get(key) != config.get(key) for key in keys)
            or prior["registry"] != registry or prior["base_identity"] != identity or prior["feature_layers"] != layers):
        raise ValueError("pi stage transition requires identical model, base and robot/language conventions")
    encoder.validate_identity(prior["encoder_identity"])
    wanted = {name: value for name, value in _interface_state(interface).items()
              if name.startswith(("state_encoder.", "condition_adapter."))}
    source = {name: value for name, value in prior["model"]["interface"].items() if name in wanted}
    for expected, values in ((wanted, source), (_action_state(native, interface), prior["model"]["action"])):
        if (expected.keys() != values.keys() or any(value.shape != values[name].shape or value.dtype != values[name].dtype
                                                  for name, value in expected.items())):
            raise ValueError("pi stage-1 shared projection or action expert keys, shapes or FP32 precision changed")
        with torch.no_grad():
            for name, value in expected.items():
                value.copy_(values[name])


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
    if (not isinstance(payload, dict) or payload.get("kind") != kind or payload.get("format_version") != g_pi_artifact_version(payload.get("config"))
            or not isinstance(payload.get("config"), dict) or payload.get("upstream_commit") != ZERO_WAM_COMMIT
            or payload.get("precision") != "float32" or payload.get("encoder_precision") not in {"float32", "bfloat16"}):
        raise ValueError("expected a version-4 intent G or endpoint pi lightweight artifact (old versions cannot resume) with FP32 trainables and a shared frozen base reference")
    config = validate_g_pi_config(payload["config"])
    from .g_pi_distributed import validate_distributed_artifact

    validate_distributed_artifact(payload, kind=kind)
    _check_stage(config, payload.get("stage"))
    if config["interface_type"] == "pi_goal":
        stage1_sha = payload.get("stage1_artifact_sha256")
        if payload.get("pi_training") != pi_stage_contract(config, payload["stage"], stage1_sha):
            raise ValueError("pi checkpoint stage, endpoint loss, noise or separation metadata differ from configuration")
        needs_prior = payload["stage"] == "pi" and not pi_training_settings(config)["ablations"]["no_stage1"]
        if needs_prior and (not isinstance(stage1_sha, str) or len(stage1_sha) != 64
                            or any(char not in "0123456789abcdef" for char in stage1_sha)):
            raise ValueError("pi stage 2 requires recorded stage-1 artifact provenance")
        if not needs_prior and stage1_sha is not None:
            raise ValueError("pi_prior or no_stage1 ablation cannot record stage-1 initialization")
    if config["interface_type"] == "g_translator" and payload.get("demo_route") != config.get("demo_route", "one_way"):
        raise ValueError("G demo_route differs from the recorded one_way/via_u_only architecture")
    registry = payload.get("registry")
    if not isinstance(registry, dict):
        raise ValueError("G/pi registry must contain robot conventions")
    identity = payload.get("encoder_identity")
    if (payload.get("architecture") != g_pi_architecture(config)
            or payload.get("interface_type") != config["interface_type"]
            or not isinstance(identity, dict) or identity.get("layer") != config["goal_encoder"]["layer"]
            or payload.get("conditioning_mode") != conditioning_mode(config)
            or payload.get("p_drop") != language_drop_probability(config)
            or identity.get("grid_size") != list(config["goal_encoder"].get("grid_size", [4, 4]))
            or identity.get("token_order") != "camera_then_row_major"
            or identity.get("camera_layout") != config["goal_encoder"]["camera_layout"]
            or identity.get("num_views") != len(config["goal_encoder"]["camera_layout"])
            or identity.get("k_z") != len(config["goal_encoder"]["camera_layout"]) * math.prod(config["goal_encoder"].get("grid_size", [4, 4]))
            or payload.get("k_z") != identity.get("k_z") or payload.get("d_z") != identity.get("d_z")
            or type(identity.get("d_z")) is not int or identity["d_z"] < 1
            or identity.get("timestep") != 0 or identity.get("normalization") != "l2_last_dim"
            or identity.get("pooling") != "adaptive_avg_pool2d_spatial"
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
    else:
        thresholds = _validate_stop_thresholds(payload.get("stop_thresholds"))
        calibration = payload.get("stopping_calibration")
        if not isinstance(calibration, dict):
            raise ValueError("policy requires explicit threshold provenance or recorded calibration")
        if calibration != {"kind": "explicit_thresholds"}:
            from .g_pi_calibration import load_calibration_artifact

            calibration = load_calibration_artifact(calibration, expected_identity=identity,
                                                     expected_registry=registry)
            if thresholds != calibration["thresholds"]:
                raise ValueError("policy stop thresholds differ from the recorded calibration")
    return payload


def read_g_pi_artifact(path, *, payload=None, expected_encoder_identity=None):
    if payload is None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    return validate_g_pi_artifact(payload, expected_encoder_identity=expected_encoder_identity)


def load_g_pi_encoder(path, *, device="cuda", checkpoint=None):
    """Rebuild only fixed E's base from a training artifact for calibration."""
    payload = read_g_pi_artifact(path)
    location = _base_location(payload["base_reference"], checkpoint)
    native, _, encoder, identity, _ = build_g_pi_system(payload["config"], payload["registry"],
        stage=payload["stage"], checkpoint=location, tiny_native=payload["tiny_native"], device=device,
        video_precision=payload["encoder_precision"])
    if identity != payload["base_identity"]:
        raise ValueError("base checkpoint identity changed before calibration")
    encoder.validate_identity(payload["encoder_identity"])
    native.eval().requires_grad_(False)
    return encoder, payload


def _validate_stop_thresholds(value):
    from .g_pi_controller import GoalThresholds

    if not isinstance(value, dict) or set(value) != {"z", "position_m", "rotation_deg", "gripper"}:
        raise ValueError("policy requires explicit stop thresholds z, position_m, rotation_deg and gripper")
    GoalThresholds(**value)
    return dict(value)


def _export_stopping(args, payload):
    calibration = getattr(args, "calibration", None)
    thresholds = getattr(args, "stop_thresholds", None)
    if (calibration is None) == (thresholds is None):
        raise ValueError("G/pi export requires exactly one explicit stop_thresholds file or calibration artifact")
    if calibration is not None:
        from .g_pi_calibration import load_calibration_artifact

        artifact = load_calibration_artifact(calibration, expected_identity=payload["encoder_identity"],
                                             expected_registry=payload["registry"])
        return _validate_stop_thresholds(artifact["thresholds"]), artifact
    if not isinstance(thresholds, dict):
        thresholds = json.loads(Path(thresholds).read_text())
    return _validate_stop_thresholds(thresholds), {"kind": "explicit_thresholds"}


def train_g_pi_interface(args):
    from .g_pi_data import g_pi_sample_files, load_g_pi_index, load_g_pi_sample
    from .goal_training import goal_registry

    config = json.loads(Path(args.config).read_text())
    if getattr(args, "intent_groups", None) is not None:
        if config.get("interface_type") != "g_translator":
            raise ValueError("intent groups are only allowed for G, never pi")
        config["intent_training"] = {**config.get("intent_training", {}),
                                     "manifest": str(Path(args.intent_groups).resolve())}
    config = validate_g_pi_config(config)
    from .g_pi_distributed import distributed_settings, train_distributed_pi

    _check_stage(config, args.stage)
    if distributed_settings(config)["enabled"]:
        return train_distributed_pi(args, config)
    stage = args.stage
    _check_stage(config, stage)
    if type(args.steps) is not int or args.steps < 1:
        raise ValueError("steps must be positive")
    paths, sources = load_g_pi_index(args.index)
    table = None
    intent = intent_training_settings(config)
    if stage == "g" and intent["manifest"] is not None:
        from .g_pi_intent import load_intent_table, intent_table_files, intent_source_records

        table_path = Path(intent["manifest"])
        if not table_path.is_absolute():
            table_path = Path(args.config).resolve().parent / table_path
        table = load_intent_table(table_path)
        sources = [*sources, *intent_source_records(table)]
        _source_components(sources)
        if intent["data_version"] is not None and table.metadata["data_version"] != intent["data_version"]:
            raise ValueError("intent data_version differs from the configured v1/v2 data contract")
        indexed = {path.resolve() for path in paths}
        paired = {entry.paired_task.resolve() for entry in table.entries
                  if entry.split == "train" and entry.paired_task is not None}
        if not paired or not paired <= indexed:
            raise ValueError("intent training requires paired train tasks belonging to the robot training index")
    sample_route = "pi_prior" if stage == "pi_prior" else config["interface_type"]
    first = load_g_pi_sample(paths[0], route=sample_route, generator=torch.Generator().manual_seed(args.seed))
    registry = goal_registry(first)
    _check_sample(first, config, registry)
    if table is not None and (table.metadata["feature_space_id"] != first.metadata["feature_space_id"]
            or table.metadata["latent_normalization"] != first.metadata["latent_normalization"]):
        raise ValueError("intent demos and paired robot samples must share latent feature space and normalization")
    previous = read_g_pi_artifact(args.resume) if args.resume else None
    prior = _initialization(args, config, previous)
    rng_names = ("sample", "action", "goal", "language") + (("intent",) if table is not None else ())
    if previous and set(previous.get("rng", {})) != set(rng_names):
        raise ValueError("resume requires matching sample, action, goal, language and optional intent RNG states")
    identities = {"index": file_sha256(args.index), **{str(p.resolve()): file_sha256(p) for p in paths}}
    if table is not None:
        identities.update({str(path.resolve()): file_sha256(path) for path in intent_table_files(table)})
    start = previous["attempted_steps"] if previous else 0
    if start + args.steps > config["training"]["max_steps"]:
        raise ValueError("cumulative independent stage budget exceeded")
    if previous and (previous["config"] != config or previous["stage"] != stage
            or previous["registry"] != registry or previous["tiny_native"] != args.tiny_native
            or previous["data_identity"] != identities or previous["seed"] != args.seed):
        raise ValueError("resume requires the same route, config, data, robot conventions and seed")
    if previous or prior:
        _source_components([*(previous or prior)["source_records"], *sources])
    visited = dict(previous["visited_arrays"]) if previous else {}
    if any(file_sha256(path) != digest for path, digest in visited.items()):
        raise ValueError("consumed training inputs changed before resume")
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not previous:
        raise ValueError("training requires a fresh output directory or explicit resume")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    origin = previous or prior
    checkpoint = _base_location(origin["base_reference"], args.checkpoint) if origin else args.checkpoint
    native, interface, encoder, identity, layers = build_g_pi_system(config, registry, stage=stage,
        checkpoint=checkpoint, tiny_native=args.tiny_native, device=args.device,
        video_precision=origin["encoder_precision"] if origin else None)
    if prior:
        _initialize_pi(native, interface, encoder, config, registry, prior, identity, layers)
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
    target_cache = None
    if config.get("target_cache_index"):
        from .g_pi_targets import load_target_cache_index

        cache_path = Path(config["target_cache_index"])
        if not cache_path.is_absolute():
            cache_path = Path(args.config).resolve().parent / cache_path
        target_cache = load_target_cache_index(cache_path, encoder.identity, task_paths=paths)
        for file in target_cache.files:
            name, digest = str(file.resolve()), file_sha256(file)
            if name in visited and visited[name] != digest:
                raise ValueError("consumed target cache changed before resume")
            visited[name] = digest
    optimizer = _optimizer(native, interface, encoder, config)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    generators = {name: torch.Generator().manual_seed(args.seed + offset)
                  for offset, name in enumerate(rng_names)}
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
            selected = None
            records, paired_samples = [], []
            if table is not None:
                from .g_pi_intent import load_intent_demo, sample_intent_batch

                selected = sample_intent_batch(table, generators["intent"],
                    groups_per_batch=intent["groups_per_batch"], samples_per_group=intent["samples_per_group"])
                for entry in selected:
                    demonstration = load_intent_demo(entry).demonstration
                    sample = (load_g_pi_sample(entry.paired_task, route=config["interface_type"],
                              generator=generators["sample"]) if entry.paired_task is not None else None)
                    records.append((demonstration, sample))
                    if sample is not None:
                        paired_samples.append((entry.paired_task, sample))
            else:
                path = paths[step % len(paths)]
                sample = load_g_pi_sample(path, route=sample_route, generator=generators["sample"])
                paired_samples = [(path, sample)]
            for path, sample in paired_samples:
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
            losses = (g_intent_training_loss(native, interface, encoder, records, selected, config,
                          uncertain_pairs=table.uncertain_pairs,
                          cached_targets=([target_cache.goal(record[1], entry.paired_task) if record[1] is not None else None
                                           for record, entry in zip(records, selected)] if target_cache is not None else None)) if table is not None else
                      g_pi_training_loss(native, interface, encoder, sample, config, generators,
                                         stage=stage, feature_layers=layers,
                                         cached_z=target_cache.goal(sample, path) if target_cache is not None else None))
            if not torch.isfinite(losses["total"]):
                raise ValueError("nonfinite objective; optimizer not updated")
            supervised = bool(paired_samples) or (table is not None and intent["contrastive_weight"] > 0)
            norm = 0.
            if supervised:
                losses["total"].backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, config["training"]["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                updates += 1
            log.write(json.dumps({**{key: float(value.detach()) for key, value in losses.items()},
                "step": step, "stage": stage, "interface_type": config["interface_type"],
                "gradient_norm": float(norm), "paired_samples": len(paired_samples),
                "human_only_samples": len(records) - len(paired_samples) if table is not None else 0,
                "current_time": [s.metadata["current_time"] for _, s in paired_samples] if table is not None else sample.metadata["current_time"],
                "subgoal_time": [s.subgoal_time for _, s in paired_samples] if table is not None else sample.subgoal_time,
                "terminal_time": [s.terminal_time for _, s in paired_samples] if table is not None else sample.terminal_time,
                "video_generation": False, "video_supervision": False, "updated": supervised}, allow_nan=False) + "\n")
            log.flush()
    if _frozen_checksums(native, encoder, config) != frozen:
        raise ValueError("frozen video backbone or E changed during training")
    if table is not None and any(file_sha256(path) != identities[str(path.resolve())] for path in intent_table_files(table)):
        raise ValueError("intent training inputs changed after their fingerprints were recorded")
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    n = np.random.get_state()
    reference = _base_reference(config, checkpoint, args.tiny_native, identity, encoder)
    payload = {"format_version": g_pi_artifact_version(config), "kind": "g_pi_training", "architecture": g_pi_architecture(config),
        "conditioning_mode": conditioning_mode(config),
        "p_drop": language_drop_probability(config),
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
        "source_records": previous["source_records"] if previous else [*(prior["source_records"] if prior else []), *sources]}
    if config["interface_type"] == "pi_goal":
        source_sha = file_sha256(args.initialize) if prior else previous.get("stage1_artifact_sha256") if previous else None
        payload["stage1_artifact_sha256"] = source_sha
        payload["pi_training"] = pi_stage_contract(config, stage, source_sha)
    if stage == "g":
        payload["demo_route"] = config.get("demo_route", "one_way")
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
    if payload["stage"] == "pi_prior":
        raise ValueError("pi_prior checkpoints are only for resume or stage-2 initialization, not goal-policy deployment")
    thresholds, calibration = _export_stopping(args, payload)
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("export requires a fresh directory")
    requested = getattr(args, "dtype", "float32")
    if requested not in {"float32", "bfloat16"}:
        raise ValueError("deployment dtype must be float32 or bfloat16")
    # The legacy dtype flag is accepted, but v3 never quantizes trainables or
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
                "visited_arrays", "data_identity", "data_cursor", "rank_states"}
    policy = {key: value for key, value in payload.items() if key not in excluded}
    policy.update(kind="g_pi_policy", precision="float32", requested_dtype=requested, shards=files, weight_map=weight_map,
                  source_artifact_sha256=file_sha256(args.artifact), stop_thresholds=thresholds,
                  stopping_calibration=calibration)
    write_json(output / "policy.json", policy)
    return {"policy": str(output.resolve()), "interface_type": payload["interface_type"],
            "precision": "float32", "requested_dtype": requested,
            "encoder_precision": payload["encoder_precision"], "requires_base_checkpoint": not payload["tiny_native"],
            "test_time_updates": False,
            "video_generation": False}


def load_g_pi_policy(path, *, device="cuda", checkpoint=None, expected_encoder_identity=None,
                     shared_base=None):
    from safetensors.torch import load_file
    from .g_pi_context import assert_frozen_base, frozen_base_checksum

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
    if shared_base is None:
        native, interface, encoder, identity, layers = build_g_pi_system(payload["config"], payload["registry"],
            stage=payload["stage"], checkpoint=checkpoint, tiny_native=payload["tiny_native"], device=device,
            video_precision=payload["encoder_precision"])
    else:
        if (payload["interface_type"] != "g_translator" or not isinstance(shared_base, tuple)
                or len(shared_base) != 3):
            raise ValueError("shared_base must be (native, encoder, pi_policy_metadata) when loading G")
        native, encoder, existing = shared_base
        validate_g_pi_artifact(existing, kind="g_pi_policy")
        if (existing["interface_type"] != "pi_goal" or encoder.native is not native
                or existing["base_identity"] != payload["base_identity"]
                or existing["encoder_identity"] != payload["encoder_identity"]
                or existing["empty_text_identity"] != payload["empty_text_identity"]
                or existing["encoder_precision"] != payload["encoder_precision"]
                or existing["feature_layers"] != payload["feature_layers"]):
            raise ValueError("shared base requires identical pi/G base, E, empty prompt, precision and layers")
        requested = torch.device(device)
        actual = native.patch_embedding_mlp.weight
        requested_index = (torch.cuda.current_device() if requested.type == "cuda" and requested.index is None
                           else requested.index)
        if (actual.device.type != requested.type or actual.device.index != requested_index
                or actual.dtype != getattr(torch, payload["encoder_precision"])):
            raise ValueError("shared base device or frozen video precision differs from the requested G policy")
        assert_frozen_base(native)
        encoder.validate_identity(payload["encoder_identity"])
        if frozen_base_checksum(native) != payload["encoder_identity"]["base_sha256"]:
            raise ValueError("shared base checksum differs from the recorded frozen E identity")
        identity, layers = existing["base_identity"], list(range(len(native.blocks)))
        # Only G's decoder is constructed/restored. In particular no precision,
        # training-mode or state changes touch pi's trained action parameters.
        interface = _interface(native, payload["config"], payload["registry"])
    if identity != payload["base_identity"]:
        raise ValueError("base checkpoint identity changed before loading the trainables")
    encoder.validate_identity(payload["encoder_identity"])
    _restore_system(native, interface, encoder, model)
    if frozen_base_checksum(encoder.native) != payload["encoder_identity"]["base_sha256"]:
        raise ValueError("E weights differ from the recorded identity")
    if payload["feature_layers"] != layers:
        raise ValueError("policy feature layers differ from native blocks")
    for module in ((native, interface, encoder) if shared_base is None else (interface,)):
        module.eval().requires_grad_(False)
    return native, interface, encoder, payload
