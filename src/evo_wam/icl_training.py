"""Mixed native ICL adaptation: robot video/actions plus human video continuation."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .cli import file_sha256, move, source_module, write_json
from .icl_data import load_icl_index, load_icl_sample
from .native_icl import forward_video_only, install_icl_lora, merge_icl_lora
from .zerowam import DEFAULT_SOURCE, ZERO_WAM_COMMIT, load_native_class, unpack_velocity


def load_icl_config(path):
    config = json.loads(Path(path).read_text())
    return validate_icl_config(config)


def validate_icl_config(config):
    """Shared validation for callers which already loaded a native configuration."""
    if config.get("schema_version") != 1 or config.get("kind") != "native_icl_experiment":
        raise ValueError("expected a native_icl_experiment configuration")
    if type(config.get("synthetic_dimensions_only")) is not bool:
        raise ValueError("explicit synthetic_dimensions_only flag required")
    schedule = config["domain_schedule"]
    if not isinstance(schedule, list) or not schedule or set(schedule) - {"human", "robot"}:
        raise ValueError("domain_schedule must contain human/robot slots")
    if config["human_context"] not in {"none", "cross_video"}:
        raise ValueError("human_context must be none or cross_video")
    for value in (config["lambda_human"], config["video_snr_shift"], config["mcp_snr_shift"],
                  config["lora"]["alpha"], config["training"]["learning_rate"], config["training"]["gradient_clip"]):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("loss, scheduler and optimizer settings must be positive and finite")
    for value in (config["chunk_size"], config["max_frame_chunk_size"], config["lora"]["rank"],
                  config["training"]["max_steps"], config["ifp"]["future_chunk_stride"]):
        if type(value) is not int or value < 1:
            raise ValueError("chunk, rank, step and stride settings must be positive integers")
    if config["chunk_size"] > config["max_frame_chunk_size"]:
        raise ValueError("chunk_size exceeds max_frame_chunk_size")
    if type(config["window_size"]) is not int or config["window_size"] < 1:
        raise ValueError("window_size must be positive")
    if type(config["icl_rope_h"]) is not int or config["icl_rope_h"] < 1:
        raise ValueError("icl_rope_h must explicitly separate demonstration coordinates")
    if type(config["ifp"]["enabled"]) is not bool:
        raise ValueError("ifp.enabled must be Boolean")
    weights = config["ifp"]["loss_weights"]
    if not isinstance(weights, list) or not weights or any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("IFP loss_weights must be a nonempty list of finite nonnegative weights")
    if "demo_bottleneck" in config:
        bottleneck = config["demo_bottleneck"]
        sizes = {"dim", "num_heads", "group_frames", "tokens_per_group", "layers"}
        if not isinstance(bottleneck, dict) or set(bottleneck) != sizes | {"consistency_weight"}:
            raise ValueError("demo_bottleneck requires explicit dimensions, temporal grouping and consistency_weight")
        if any(type(bottleneck[key]) is not int or bottleneck[key] < 1 for key in sizes):
            raise ValueError("bottleneck sizes must be positive integers")
        if bottleneck["dim"] % bottleneck["num_heads"]:
            raise ValueError("bottleneck dim must be divisible by num_heads")
        weight = bottleneck["consistency_weight"]
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
            raise ValueError("consistency_weight must be finite and nonnegative")
        if config["human_context"] != "cross_video":
            raise ValueError("bottleneck comparisons require the cross_video context path")
    return config


def build_icl_model(config, *, checkpoint=None, tiny_native=False, device="cuda"):
    cls = load_native_class()
    if tiny_native:
        native_config = dict(patch_size=(1, 1, 1), num_attention_heads=2, attention_head_dim=18,
            in_channels=4, out_channels=4, action_dim=3, text_dim=8, freq_dim=4, ffn_dim=16,
            num_layers=2, rope_max_seq_len=128, action_inner_dim=36, action_ffn_dim=16,
            attn_window=config["window_size"], enable_mcp=True,
            num_mcp_modules=len(config["ifp"]["loss_weights"]), mcp_blocks_per_group=1,
            mcp_hidden_collect_layers=(0, 1))
        native = cls(**native_config).to(device=device, dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32)
        null = torch.zeros(1, 2, 8, device=device)
        # Diffusers' private default-field list has process-dependent ordering.
        # Identity uses the explicit constructor settings and pinned source.
        identity = {"kind": "tiny-native-full-state", "config": native_config}
    else:
        if config["synthetic_dimensions_only"] or not checkpoint:
            raise ValueError("real training requires an audited configuration and a local Zero-WAM checkpoint")
        folder = Path(checkpoint).resolve()
        folder = folder / "transformer" if (folder / "transformer").is_dir() else folder
        weights = sorted(folder.glob("*.safetensors"))
        if not weights or not (folder / "config.json").is_file():
            raise ValueError("provide a local native safetensors checkpoint; no download or random fallback")
        identity = {"kind": "released-safetensors", "sha256": {p.name: file_sha256(p)
            for p in [folder / "config.json", *weights, *sorted(folder.glob("*.safetensors.index.json"))]}}
        native = cls.from_pretrained(str(folder), local_files_only=True, use_safetensors=True,
                                    torch_dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32).to(device)
        null = torch.load(DEFAULT_SOURCE / "wan_va/assets/empty_text_emb.pt", weights_only=True, map_location=device)[None]
    if config["ifp"]["enabled"] and (not native.enable_mcp or len(native.mcp_blocks) != len(config["ifp"]["loss_weights"])):
        raise ValueError("IFP weights must match the native checkpoint's MCP groups")
    if null.ndim != 3 or null.shape[-1] != native.config.text_dim or not torch.isfinite(null).all():
        raise ValueError("native empty-text embedding does not match the checkpoint")
    install_icl_lora(native, **config["lora"])
    if "demo_bottleneck" in config:
        from .demo_context import install_demo_interface
        install_demo_interface(native, {key: value for key, value in config["demo_bottleneck"].items()
                                       if key != "consistency_weight"})
        native.demo_icl_rope_h = config["icl_rope_h"]
        native.eval()
    return native, null, identity


def prepare_icl_inputs(sample, config, native, null, generator, *, include_actions=True):
    """Use original scheduler/grid conventions; human actions are absent, not zero labels."""
    from wan_va.utils import get_mesh_id

    device = next(native.parameters()).device
    patch = tuple(native.patch_size)
    robot = sample.metadata["target"]["domain"] == "robot"
    if patch[0] != 1:
        raise ValueError("this native ICL data path requires temporal patch size 1")
    if sample.history_frames % config["chunk_size"]:
        raise ValueError("history_frames must end at a native chunk boundary")
    for video in (sample.target, sample.demonstration):
        if video.shape[1] != native.config.in_channels or any(size % unit for size, unit in zip(video.shape[-3:], patch)):
            raise ValueError("video channels/grid must match the native checkpoint patch embedding")
    if sample.target.shape[1] != native.config.out_channels:
        raise ValueError("target channels differ from the native output head")
    if config["icl_rope_h"] < sample.target.shape[-2] // patch[1]:
        raise ValueError("icl_rope_h overlaps the target spatial grid")
    scheduler_cls = source_module("utils/scheduler.py").FlowMatchScheduler

    def stream(clean, shift, *, action=False, frame_shift=0, mask=None):
        clean = clean.detach().cpu().float()
        if mask is not None:
            mask = mask.cpu()
            clean = torch.where(mask, clean, 0)
        scheduler = scheduler_cls(shift=shift, sigma_min=0., extra_one_step=True)
        scheduler.set_timesteps(1000, training=True)
        times = scheduler.timesteps[torch.randint(1000, (clean.shape[2],), generator=generator)]
        noise = torch.randn(clean.shape, generator=generator)
        noisy = scheduler.add_noise(clean, noise, times, t_dim=2)
        target = scheduler.training_target(clean, noise, times)
        # Upstream noising retains random values in inactive action channels;
        # only their supervision is masked. Human examples have no such stream.
        pf, ph, pw = (1, 1, 1) if action else patch
        grid = get_mesh_id(clean.shape[2] // pf, clean.shape[3] // ph, clean.shape[4] // pw,
                           int(action), f_shift=frame_shift, action=False)
        valid = torch.ones_like(clean, dtype=torch.bool) if mask is None else mask.clone()
        if not robot:
            valid[:, :, :sample.history_frames] = False
        return move({"latent": clean, "noisy_latents": noisy, "targets": target,
                     "timesteps": times[None], "cond_timesteps": torch.zeros_like(times)[None],
                     "grid_id": grid[None], "valid_mask": valid,
                     "training_weight": torch.ones_like(times)[None] if action else scheduler.training_weight(times)[None]}, device)

    inputs = {"latent_dict": stream(sample.target, config["video_snr_shift"]),
              "chunk_size": config["chunk_size"], "max_frame_chunk_size": config["max_frame_chunk_size"],
              "window_size": config["window_size"], "text_emb": null,
              "encoder_seq_ids": torch.zeros(null.shape[1], dtype=torch.int, device=device)}
    if robot:
        if sample.actions is None or sample.actions.shape[1] != native.config.action_dim:
            raise ValueError("robot actions must match the checkpoint action dimension")
        if include_actions:
            inputs["action_dict"] = stream(sample.actions, 1., action=True, mask=sample.actions_mask)
    elif sample.actions is not None or sample.actions_mask is not None:
        raise ValueError("human examples cannot supply robot actions")
    if robot or config["human_context"] == "cross_video":
        demonstration = sample.demonstration.detach().to(device)
        f, h, w = demonstration.shape[-3:]
        inputs["icl_latent_dict"] = {"latent": demonstration,
            "timesteps": torch.zeros(1, f, device=device),
            "grid_id": get_mesh_id(f // patch[0], h // patch[1], w // patch[2], 0,
                                    h_shift=config["icl_rope_h"])[None].to(device)}
        if "demo_bottleneck" in config:
            from .demo_context import prepare_demo_context
            raw_context = {**inputs["icl_latent_dict"], "frame_times": sample.demonstration_times.to(device)}
            inputs["icl_latent_dict"] = prepare_demo_context(native, raw_context)
            if config["demo_bottleneck"]["consistency_weight"] > 0:
                if sample.appearance_demonstration is None:
                    raise ValueError("positive consistency_weight requires an audited appearance_variant for every demonstration")
                variant = prepare_demo_context(native, {**raw_context,
                    "latent": sample.appearance_demonstration.detach().to(device)})
                # Only the compressed original reaches WAM; the variant is used
                # solely by the auxiliary representation-consistency objective.
                inputs["appearance_tokens"] = variant["latent"].tokens
        inputs["text_emb"] = torch.cat((null, null), dim=1)
        inputs["encoder_seq_ids"] = torch.cat((inputs["encoder_seq_ids"],
            torch.ones(null.shape[1], dtype=torch.int, device=device)))
    if config["ifp"]["enabled"]:
        shift_latents = source_module("mcp.py").shift_latents_for_mcp
        inputs["mcp_latent_dicts"] = []
        for index in range(len(config["ifp"]["loss_weights"])):
            offset = (1 + index * config["ifp"]["future_chunk_stride"]) * config["chunk_size"]
            shifted, valid = shift_latents(sample.target, offset)
            future = stream(shifted, config["mcp_snr_shift"], frame_shift=offset)
            future["valid_mask"] &= valid.to(device)
            inputs["mcp_latent_dicts"].append(future)
    return inputs


def native_icl_loss(native, inputs, config, *, human, video_only=False):
    if human or video_only:
        video, future = forward_video_only(native, inputs)
        action = None
    else:
        output = native(inputs, train_mode=True)
        video, action = output[:2]
        future = output[2] if len(output) == 3 else []

    def mse(prediction, stream, patch_size, *, valid_mean=True):
        prediction = unpack_velocity(prediction, stream["targets"].shape, patch_size).float()
        target, mask = stream["targets"].detach().float(), stream["valid_mask"]
        raw = (prediction - target).square() * stream["training_weight"][:, None, :, None, None]
        masked = torch.where(mask, raw, 0)
        # Native robot action loss averages over the full tensor after masking;
        # changing this denominator would silently reweight sparse action labels.
        return masked.sum() / mask.sum().clamp_min(1) if valid_mean else masked.mean()

    losses = {"video": mse(video, inputs["latent_dict"], native.patch_size, valid_mean=human)}
    if not human and not video_only:
        losses["action"] = mse(action, inputs["action_dict"], (1, 1, 1), valid_mean=False)
    streams = inputs.get("mcp_latent_dicts", [])
    if len(future) != len(streams):
        raise ValueError("native IFP output count differs from targets")
    losses["ifp"] = sum((weight * mse(prediction, stream, native.patch_size)
        for weight, prediction, stream in zip(config["ifp"]["loss_weights"], future, streams)), losses["video"].new_zeros(()))
    losses["total"] = sum(losses.values()) * (config["lambda_human"] if human else 1.)
    weight = config.get("demo_bottleneck", {}).get("consistency_weight", 0.)
    if weight > 0:
        if "appearance_tokens" not in inputs:
            raise ValueError("appearance consistency requires the audited variant representation")
        first = inputs["icl_latent_dict"]["latent"].tokens.float()
        second = inputs["appearance_tokens"].float()
        if first.shape != second.shape or not torch.isfinite(second).all():
            raise ValueError("appearance tokens must match the original demonstration's temporal groups")
        consistency = (F.normalize(first, dim=-1) - F.normalize(second, dim=-1)).square().mean()
        losses["appearance_consistency"] = consistency
        losses["total"] = losses["total"] + weight * consistency * (config["lambda_human"] if human else 1.)
    return losses


def _model_state(native, tiny):
    return {key: value.detach().cpu() for key, value in native.state_dict().items()
            if tiny or ".down." in key or ".up." in key or key.startswith("demo_bottleneck.")}


def _read_icl_artifact(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or payload.get("format_version") != 1
            or payload.get("kind") != "native_icl_adaptation" or payload.get("upstream_commit") != ZERO_WAM_COMMIT):
        raise ValueError("expected a matching native_icl_adaptation artifact")
    return payload


def train_native_icl(args):
    config = load_icl_config(args.config)
    paths, sources = load_icl_index(args.index)
    if type(args.steps) is not int or args.steps < 1:
        raise ValueError("steps must be a positive integer")
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not args.resume:
        raise ValueError("native ICL training requires a fresh output directory or explicit resume")
    grouped = {domain: [] for domain in set(config["domain_schedule"])}
    identities = {"index": file_sha256(args.index)}
    human_sources, human_targets = set(), set()
    for path in paths:
        meta = json.loads(path.read_text())
        domain = meta["target"]["domain"]
        if domain in grouped:
            grouped[domain].append(path)
            identities[str(path)] = file_sha256(path)
            if domain == "human":
                human_sources.add((meta["demonstration"]["source_id"], str((path.parent / meta["demonstration"]["arrays"]).resolve())))
                human_targets.add((meta["target"]["source_id"], str((path.parent / meta["target"]["arrays"]).resolve())))
    if any(not values for values in grouped.values()):
        raise ValueError("every scheduled domain requires training samples")
    if human_sources - human_targets:
        raise ValueError("matched human-video controls require every demo video cache to also occur as a target (e.g. reciprocal pairs)")
    previous = _read_icl_artifact(args.resume) if args.resume else None
    start = previous["attempted_steps"] if previous else 0
    if start + args.steps > config["training"]["max_steps"]:
        raise ValueError("cumulative native ICL budget exceeded")
    if previous and (previous["config"] != config or previous["data_identity"] != identities or previous["seed"] != args.seed
                     or previous["tiny_native"] != args.tiny_native):
        raise ValueError("resume requires identical config, data index, seed and base mode")
    visited = dict(previous["visited_arrays"]) if previous else {}
    for path, digest in visited.items():
        if file_sha256(path) != digest:
            raise ValueError("previously consumed video/action arrays changed before resume")
    torch.manual_seed(args.seed)
    native, null, identity = build_icl_model(config, checkpoint=args.checkpoint, tiny_native=args.tiny_native, device=args.device)
    parameters = [p for p in native.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["training"]["learning_rate"], weight_decay=0.)
    generators = {domain: torch.Generator().manual_seed(args.seed + (domain == "human")) for domain in grouped}
    counts = {domain: 0 for domain in grouped}
    updates = dict(counts)
    if previous:
        if identity != previous["base_identity"]:
            raise ValueError("frozen native checkpoint changed before resume")
        expected = _model_state(native, args.tiny_native)
        if expected.keys() != previous["model"].keys():
            raise ValueError("native adapter parameter registry changed")
        native.load_state_dict(previous["model"], strict=args.tiny_native)
        optimizer.load_state_dict(previous["optimizer"])
        counts, updates = previous["domain_samples"], previous["domain_updates"]
        for domain, generator in generators.items():
            generator.set_state(previous["domain_rng"][domain])
        torch.set_rng_state(previous["torch_rng"])
        if torch.cuda.is_available() and previous["cuda_rng"]:
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
        for step in range(start, start + args.steps):
            domain = config["domain_schedule"][step % len(config["domain_schedule"])]
            path = grouped[domain][counts[domain] % len(grouped[domain])]
            sample = load_icl_sample(path)
            for role in ("demonstration", "target"):
                array = str((path.parent / sample.metadata[role]["arrays"]).resolve())
                if array not in visited:
                    visited[array] = file_sha256(array)
            if config.get("demo_bottleneck", {}).get("consistency_weight", 0.) > 0:
                if "appearance_variant" not in sample.metadata:
                    raise ValueError("appearance-consistency training requires an audited variant")
                array = str((path.parent / sample.metadata["appearance_variant"]["arrays"]).resolve())
                if array not in visited:
                    visited[array] = file_sha256(array)
            inputs = prepare_icl_inputs(sample, config, native, null, generators[domain])
            optimizer.zero_grad(set_to_none=True)
            losses = native_icl_loss(native, inputs, config, human=domain == "human")
            if not torch.isfinite(losses["total"]):
                raise ValueError("nonfinite native ICL loss; no optimizer update")
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, config["training"]["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            counts[domain] += 1
            updates[domain] += 1
            row = {key: float(value.detach()) for key, value in losses.items()}
            row.update(step=step, domain=domain, updated=True, gradient_norm=float(norm),
                       sample_id=sample.metadata["sample_id"], source_id=sample.metadata["demonstration"]["source_id"],
                       target_id=sample.metadata["target"]["source_id"], used_demonstration="icl_latent_dict" in inputs,
                       video_frames=sample.target.shape[2], demonstration_frames=sample.demonstration.shape[2],
                       ifp_valid_values=[int(stream["valid_mask"].sum()) for stream in inputs.get("mcp_latent_dicts", [])])
            if "demo_bottleneck" in config:
                row["context_tokens"] = inputs["icl_latent_dict"]["latent"].hidden.shape[1]
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    payload = {"format_version": 1, "kind": "native_icl_adaptation", "upstream_commit": ZERO_WAM_COMMIT,
        "config": config, "native_config": dict(native.config), "model": _model_state(native, args.tiny_native),
        "base_identity": identity, "tiny_native": args.tiny_native, "optimizer": optimizer.state_dict(),
        "attempted_steps": start + args.steps, "domain_samples": counts, "domain_updates": updates,
        "domain_rng": {domain: gen.get_state() for domain, gen in generators.items()}, "seed": args.seed,
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "data_identity": identities, "visited_arrays": visited, "source_records": sources}
    temporary = output / "native_icl.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(output / "native_icl.pt")
    report = {"artifact": str((output / "native_icl.pt").resolve()), "domain_samples": counts,
        "domain_updates": updates, "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if str(args.device).startswith("cuda") else None,
        "wam_video_adapters_updated": True, "action_parameters_updated": False,
        "demo_bottleneck_updated": "demo_bottleneck" in config,
        "test_time_updates": False, "robot_execution_evaluated": False}
    write_json(output / "run.json", report)
    return report


@torch.no_grad()
def export_native_icl(args):
    payload = _read_icl_artifact(args.artifact)
    if sum(payload["domain_updates"].values()) < 1:
        raise ValueError("cannot export an untrained adaptation")
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("native export requires a fresh output directory")
    native, _, identity = build_icl_model(payload["config"], checkpoint=args.checkpoint,
        tiny_native=payload["tiny_native"], device=args.device)
    if identity != payload["base_identity"]:
        raise ValueError("native export must use the identical frozen checkpoint")
    expected = _model_state(native, payload["tiny_native"])
    if expected.keys() != payload["model"].keys():
        raise ValueError("native adapter registry changed")
    native.load_state_dict(payload["model"], strict=payload["tiny_native"])
    merge_icl_lora(native).eval().requires_grad_(False)
    # Native action/text projections have shared parameter aliases. Preserve
    # every original key, copying one shard at a time for safetensors storage.
    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file

    bottleneck = payload["config"].get("demo_bottleneck")
    folder = output / ("backbone" if bottleneck else "transformer")
    native.save_config(folder)
    state = {key: value for key, value in native.state_dict().items() if not key.startswith("demo_bottleneck.")}
    shards = split_torch_state_dict_into_shards(state, max_shard_size="5GB",
        filename_pattern="diffusion_pytorch_model{suffix}.safetensors")
    for filename, names in shards.filename_to_tensors.items():
        tensors = {name: state[name].detach().to(device="cpu", copy=True).contiguous() for name in names}
        save_file(tensors, str(folder / filename), metadata={"format": "pt"})
        del tensors
    if shards.is_sharded:
        write_json(folder / "diffusion_pytorch_model.safetensors.index.json",
                   {"metadata": shards.metadata, "weight_map": shards.tensor_to_filename})
    if bottleneck:
        context_path = output / "demo_bottleneck.pt"
        torch.save({key: value.detach().cpu() for key, value in native.demo_bottleneck.state_dict().items()}, context_path)
        files = [context_path, *sorted(folder.iterdir())]
        spaces = {record["feature_space_id"] for record in payload["source_records"] if "feature_space_id" in record}
        if len(spaces) != 1:
            raise ValueError("bottleneck deployment requires one audited visual feature space")
        manifest = {"format_version": 1, "kind": "native_icl_temporal_bottleneck", "upstream_commit": ZERO_WAM_COMMIT,
            "demo_bottleneck": {key: value for key, value in bottleneck.items() if key != "consistency_weight"},
            "icl_rope_h": payload["config"]["icl_rope_h"], "feature_space_id": next(iter(spaces)),
            "tiny_native": payload["tiny_native"],
            "files": {str(path.relative_to(output)): file_sha256(path) for path in files}}
        write_json(output / "bottleneck.json", manifest)
        return {"deployment": str(output.resolve()), "context_interface": "temporal_bottleneck",
                "artifact_sha256": file_sha256(args.artifact), "tiny_native": payload["tiny_native"],
                "test_time_updates": False, "commands_sent": 0}
    report = {"transformer": str((output / "transformer").resolve()), "artifact_sha256": file_sha256(args.artifact),
              "tiny_native": payload["tiny_native"], "test_time_updates": False, "commands_sent": 0}
    write_json(output / "adaptation.json", report)
    return report
