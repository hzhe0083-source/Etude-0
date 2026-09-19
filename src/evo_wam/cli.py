"""Auditable local commands. Native training never falls back to a toy model."""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass, replace
import importlib.metadata
import importlib.util
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def action_space(metadata, dimension):
    space = metadata.get("action_space", {})
    if not isinstance(space, dict):
        raise ValueError("action_space must be an object")
    channels = space.get("valid_channels", [])
    if (space.get("representation") != "zero-wam-normalized"
            or not isinstance(space.get("normalization_id"), str) or not space["normalization_id"]
            or type(space.get("dimension")) is not int or space["dimension"] != dimension
            or not isinstance(channels, list) or len(channels) != dimension
            or any(type(value) is not bool for value in channels) or not any(channels)):
        raise ValueError("action_space must identify native normalized units, normalization ID, dimension and active channels")
    return space


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32 if value.is_floating_point() else value.dtype)
    if is_dataclass(value):
        return replace(value, **{f.name: move(getattr(value, f.name), device) for f in fields(value)})
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(move(v, device) for v in value)
    return value


def source_module(filename):
    """Load a pure upstream utility without importing its GPU model package."""
    from .zerowam import verify_source
    path = verify_source() / "wan_va" / filename
    spec = importlib.util.spec_from_file_location("evo_upstream_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def doctor():
    from .zerowam import verify_source, ZERO_WAM_COMMIT
    report = {"python": sys.version.split()[0], "torch": torch.__version__,
              "cuda_available": torch.cuda.is_available(), "upstream_commit": ZERO_WAM_COMMIT,
              "source": str(verify_source()), "full_checkpoint_verified": False,
              "simulator_verified": False, "real_robot_verified": False}
    for package in ("diffusers", "transformers", "flash-attn", "torchvision"):
        try:
            report[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report[package] = None
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name()
        report["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    return report


def make_fixture(directory):
    """Numeric integration input, not a collected trajectory or robot result."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("fixture output must be an empty directory")
    rng = np.random.default_rng(0)
    f, n, roles, horizon = 18, 3, 2, 36
    arrays = {
        "entity_ids": np.array([11, 22, 33], dtype=np.int64),
        "step_offsets": np.arange(1, horizon + 1, dtype=np.int64),
        "robot_history": rng.normal(size=(2, n, 4)).astype("float32"),
        "proprio_history": rng.normal(size=(2, 3)).astype("float32"),
        "embodiment": np.zeros(2, dtype="float32"),
        "entity_patch_weights": np.zeros((n, f * 2), dtype="float32"),
        "robot_latent": rng.normal(size=(4, 1, 1, 2)).astype("float32"),
        "observed_action_history": np.empty((0, 3), dtype="float32"),
        "observed_action_step_offsets": np.empty((0,), dtype="int64"),
        "observed_video_step_offsets": np.array([0], dtype="int64"),
        "actions": rng.normal(size=(horizon, 3)).astype("float32"),
        "demo_view_0": rng.normal(size=(6, 4)).astype("float32"),
        "demo_view_1": rng.normal(size=(6, 4)).astype("float32"),
        "native_video_clean": rng.normal(size=(1, 4, f, 1, 2)).astype("float32"),
    }
    arrays["native_action_clean"] = arrays["actions"].reshape(1, f, 2, 1, 3).transpose(0, 4, 1, 2, 3).copy()
    arrays["entity_patch_weights"][:, :4] = 0.25
    # This grid is fixture data. Runtime training uses the audited input grids.
    for name, h, w, stream_id in (("video", 1, 2, 0), ("action", 2, 1, 1)):
        coords = np.stack(np.meshgrid(np.arange(f), np.arange(h), np.arange(w), indexing="ij")).reshape(3, -1)
        arrays[f"native_{name}_grid"] = np.concatenate((coords, np.full((1, coords.shape[1]), stream_id)), axis=0)[None]
    for field, shape in (("geometry", (horizon, n, 3)), ("relations", (horizon, n, n, 3)), ("events", (horizon, n, n, 2))):
        arrays[f"outcome_{field}"] = np.zeros(shape, dtype="float32")
        arrays[f"outcome_{field}_valid"] = np.ones(shape, dtype="bool")
    for part, offsets in (("current", [1, 2, 3, 4]), ("remaining", [5, 6, 7, 8])):
        arrays[f"{part}_step_offsets"] = np.asarray(offsets, dtype="int64")
        arrays[f"{part}_binding"] = np.array([0, 1] if part == "current" else [0, 2], dtype="int64")
        arrays[f"{part}_binding_valid"] = np.ones(roles, dtype="bool")
        for field, shape in (("geometry", (4, roles, 3)), ("relations", (4, roles, roles, 3)), ("events", (4, roles, roles, 2))):
            arrays[f"{part}_{field}"] = np.zeros(shape, dtype="float32")
            arrays[f"{part}_{field}_valid"] = np.ones(shape, dtype="bool")
            arrays[f"{part}_{field}_required"] = np.ones(shape, dtype="bool")
        arrays[f"{part}_geometry_tolerance"] = np.full_like(arrays[f"{part}_geometry"], 0.01)
        arrays[f"{part}_geometry_tolerance_valid"] = np.ones_like(arrays[f"{part}_geometry"], dtype="bool")
        event_shape = arrays[f"{part}_events"].shape
        arrays[f"{part}_event_windows"] = np.broadcast_to(np.asarray(offsets)[:, None, None, None, None], (*event_shape, 2)).copy()
        arrays[f"{part}_event_windows_valid"] = np.ones(event_shape, dtype="bool")
        arrays[f"{part}_event_precedence"] = np.full((4, 2), -1, dtype="int64")
        arrays[f"{part}_event_precedence_valid"] = np.ones(4, dtype="bool")
    for view in (0, 1):
        for part in ("current", "remaining"):
            for field in ("binding", "geometry", "relations", "events", "geometry_tolerance", "event_windows", "event_precedence"):
                arrays[f"view{view}_{part}_{field}_valid"] = arrays[f"{part}_{field}_valid"].copy()
        for field in ("relations", "events"):
            arrays[f"view{view}_{field}_valid"] = np.ones_like(arrays[f"outcome_{field}"], dtype="bool")
    meta = {"format_version": 2, "kind": "training_sample", "arrays": "arrays.npz", "source_id": "synthetic-human-0",
            "trajectory_id": "synthetic-robot-0", "history_id": "synthetic-state-0",
            "coordinate_frame": "synthetic_robot_frame", "task_annotation": "synthetic-contract-fixture",
            "window_start": 0, "observation_step": 0, "executed_steps": horizon, "control_dt": 0.05,
            "actions_per_frame": 2,
            "history_chunks": [{"mode": "video", "slice": [0, 1], "frame_id": 0, "rope_offset": 0}],
            "view_ids": ["synthetic-front", "synthetic-side"], "pair_kind": "synchronized_views", "provenance": "synthetic",
            "native_arrays": {name + "_dict": {"latent": f"native_{name}_clean", "grid_id": f"native_{name}_grid"}
                              for name in ("latent", "action")},
            "native_scalars": {"chunk_size": 2, "max_frame_chunk_size": 4, "window_size": 32}}
    meta["action_space"] = {"representation": "zero-wam-normalized", "normalization_id": "synthetic-v1",
                            "dimension": 3, "valid_channels": [True, True, True]}
    meta["observed_action_space"] = dict(meta["action_space"])
    meta["native_arrays"]["latent_dict"] = {"latent": "native_video_clean", "grid_id": "native_video_grid"}
    np.savez_compressed(directory / "arrays.npz", **arrays)
    write_json(directory / "sample.json", meta)
    write_json(directory / "index.json", {"samples": [{"manifest": "sample.json", "split": "train"}]})
    observation_keys = ["entity_ids", "robot_history", "proprio_history", "embodiment", "robot_latent", "demo_view_0", "demo_view_1",
                        "observed_action_history", "observed_action_step_offsets", "observed_video_step_offsets"]
    np.savez_compressed(directory / "observation.npz", **{key: arrays[key] for key in observation_keys})
    write_json(directory / "observation.json", {"format_version": 2, "kind": "observation", "arrays": "observation.npz",
        "view_ids": meta["view_ids"], "chunk_size": 2, "actions_per_frame": 2,
        "action_space": meta["action_space"], "observed_action_space": meta["observed_action_space"],
        "observation_step": 0, "control_dt": meta["control_dt"], "history_chunks": meta["history_chunks"],
        "provenance": "synthetic"})
    return {"manifest": str(directory / "sample.json"), "index": str(directory / "index.json"),
            "provenance": "synthetic; validates data/compute paths only"}


def load_index(path, split="train"):
    from .data import validate_splits
    path = Path(path)
    document = json.loads(path.read_text())
    if not isinstance(document.get("samples"), list) or not document["samples"]:
        raise ValueError("dataset index must have a nonempty samples list with manifest/split entries")
    records, selected = [], []
    for entry in document["samples"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("manifest"), str):
            raise ValueError("each index entry needs a relative manifest and split")
        manifest = (path.parent / entry["manifest"]).resolve()
        if Path(entry["manifest"]).is_absolute() or not manifest.is_relative_to(path.parent.resolve()):
            raise ValueError("sample manifests must stay within the dataset index directory")
        metadata = json.loads(manifest.read_text())
        record = {**metadata, "split": entry.get("split")}
        records.append(record)
        if entry.get("split") == split:
            selected.append(manifest)
    validate_splits(records)
    if not selected:
        raise ValueError(f"dataset has no {split} samples")
    return selected, records


def fresh_native(sample, config, generator, device, dtype):
    """Once-per-pair noising using the pinned upstream schedulers and MCP shift."""
    from .data import shared_denoising_inputs
    scheduler_class = source_module("utils/scheduler.py").FlowMatchScheduler
    shifts = {"video": config["video_snr_shift"], "action": 1.0, "mcp": config["ifp"]["mcp_snr_shift"]}
    schedulers = {}
    for name, shift in shifts.items():
        scheduler = scheduler_class(shift=shift, sigma_min=0.0, extra_one_step=True)
        scheduler.set_timesteps(1000, training=True)
        schedulers[name] = scheduler
    native = sample.native_inputs
    if sample.metadata["executed_steps"] != sample.actions.shape[1]:
        raise ValueError("native video/action training needs a fully executed paired window; use calibrate-f for prefix-only candidate data")
    if not {"latent_dict", "action_dict", "chunk_size", "max_frame_chunk_size", "window_size"} <= native.keys():
        raise ValueError("native training requires clean video/action streams, grids, and chunk/window scalars")
    space = action_space(sample.metadata, config["dimensions"]["action_dim"])
    recorded = native["action_dict"]["latent"].float()
    normalized = recorded.permute(0, 2, 3, 4, 1).reshape(1, -1, recorded.shape[1])
    if normalized.shape != sample.actions.shape or not torch.allclose(normalized.to(sample.actions), sample.actions, atol=1e-5, rtol=1e-5):
        raise ValueError("F actions must equal the native normalized action labels, including layout and units")
    channels = torch.tensor(space["valid_channels"], device=normalized.device)
    if (normalized[..., ~channels] != 0).any():
        raise ValueError("inactive native action channels must be zero")
    targets, grids, masks, schedule_keys = {}, {}, {}, {}
    for name, key in (("video", "latent_dict"), ("action", "action_dict")):
        stream = native[key]
        targets[name] = stream["latent"].to(device=device, dtype=dtype)
        if targets[name].ndim != 5 or targets[name].shape[0] != 1:
            raise ValueError("native clean streams must explicitly include batch size 1")
        grids[name] = stream["grid_id"].to(device)
        masks[name] = {k: v.to(device) for k, v in stream.items() if k in {"valid_mask", "actions_mask"}}
        if name == "action":
            channel_mask = channels.to(device)[None, :, None, None, None]
            masks[name]["actions_mask"] = channel_mask & masks[name].get("actions_mask", torch.ones_like(channel_mask)).bool()
        schedule_keys[name] = name
    if config["ifp"]["enabled"]:
        shift_latents = source_module("mcp.py").shift_latents_for_mcp
        for k in range(config["ifp"]["num_mcp_modules"]):
            name = f"mcp{k}"
            shift = (1 + k * config["ifp"]["future_chunk_stride"]) * native["chunk_size"]
            targets[name], valid = shift_latents(targets["video"], shift)
            grids[name] = grids["video"].clone()
            grids[name][:, 0] += shift
            masks[name], schedule_keys[name] = {"valid_mask": valid}, "mcp"
    draws = shared_denoising_inputs(targets, {k: schedulers[schedule_keys[k]].sigmas for k in targets},
                                   {k: 1 for k in targets}, generator=generator, time_dim=2)
    streams = {}
    for name, draw in draws.items():
        scheduler = schedulers[schedule_keys[name]]
        # tau is sigma, while the native time embedding uses sigma*1000.
        times = draw.tau.float() * scheduler.num_train_timesteps
        streams[name] = {"latent": draw.clean, "noisy_latents": draw.noisy, "targets": draw.flow_target,
                         "timesteps": times, "cond_timesteps": torch.zeros_like(times),
                         "grid_id": grids[name], "training_weight": scheduler.training_weight(times[0])[None],
                         **masks[name]}
        if "actions_mask" in masks[name]:
            for key in ("latent", "noisy_latents", "targets"):
                streams[name][key] = streams[name][key] * masks[name]["actions_mask"].to(dtype)
    return {"latent_dict": streams["video"], "action_dict": streams["action"],
            "mcp_latent_dicts": [streams[f"mcp{k}"] for k in range(config["ifp"]["num_mcp_modules"])]
            if config["ifp"]["enabled"] else [],
            **{k: native[k] for k in ("chunk_size", "max_frame_chunk_size", "window_size")}}


def build_trainer(config, *, checkpoint=None, tiny_native=False, device="cuda", stage="interface"):
    from .models import RequirementCodec, EffectReader, CausalEffectPredictor, TemporalInteractionHead
    from .training import EvoTrainer, LossWeights
    from .zerowam import ZeroWAMAdapter, DEFAULT_SOURCE
    dims, lora = config["dimensions"], config["lora"]
    options = dict(current_tokens=config["tokens"]["current"], remaining_tokens=config["tokens"]["remaining"],
                   lora_rank=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"])
    if tiny_native:
        native_config = dict(patch_size=(1, 1, 1), num_attention_heads=2, attention_head_dim=18,
                             in_channels=4, out_channels=4, action_dim=3, text_dim=8, freq_dim=4,
                             ffn_dim=16, num_layers=1, rope_max_seq_len=128, action_inner_dim=36,
                             action_ffn_dim=16, attn_window=32, enable_mcp=True,
                             num_mcp_modules=config["ifp"]["num_mcp_modules"], mcp_hidden_collect_layers=(0,))
        adapter = ZeroWAMAdapter.from_config(native_config, dims["token_dim"], device=device, **options)
        null = torch.zeros(1, 2, 8, device=device)
    else:
        if config.get("synthetic_dimensions_only", True):
            raise ValueError("replace synthetic pilot dimensions with audited data dimensions before checkpoint training")
        if not checkpoint:
            raise ValueError("a local released Zero-WAM checkpoint is required; no random fallback")
        header_path = Path(checkpoint)
        header_path = header_path / "transformer" if (header_path / "transformer").is_dir() else header_path
        header = json.loads((header_path / "config.json").read_text())
        if header.get("action_dim") != dims["action_dim"]:
            raise ValueError("audited action_dim differs from checkpoint header; refusing expensive weight loading")
        adapter = ZeroWAMAdapter.from_checkpoint(checkpoint, dims["token_dim"], device=device, **options)
        null = torch.load(DEFAULT_SOURCE / "wan_va/assets/empty_text_emb.pt", map_location="cpu", weights_only=True)[None].to(device)
    if adapter.native.config.action_dim != dims["action_dim"]:
        raise ValueError("declared action dimensions do not match the actual native checkpoint")
    if not tiny_native:
        for name in ("num_mcp_modules", "mcp_blocks_per_group", "mcp_hidden_collect_layers"):
            actual, expected = getattr(adapter.native.config, name), config["ifp"][name]
            if isinstance(expected, list):
                actual = list(actual)
            if actual != expected:
                raise ValueError(f"native checkpoint {name} differs from the registered configuration")
    codec = RequirementCodec(dims["entity_dim"], dims["geometry_dim"], dims["relation_dim"], dims["event_dim"],
                             dims["roles"], dims["token_dim"], config["interface"],
                             max_precedence_edges=dims["max_precedence_edges"]).to(device)
    reader = EffectReader(dims["demo_dim"], dims["entity_dim"], dims["proprio_dim"], dims["embodiment_dim"],
                          dims["roles"], dims["token_dim"]).to(device)
    physical = CausalEffectPredictor(dims["entity_dim"], dims["proprio_dim"], dims["action_dim"], dims["embodiment_dim"],
                                     dims["geometry_dim"], dims["relation_dim"], dims["event_dim"]).to(device)
    interaction = TemporalInteractionHead(adapter.native.inner_dim, dims["relation_dim"], dims["event_dim"]).to(device)
    trainer = EvoTrainer(adapter, codec, reader, physical, interaction, stage=stage,
                         weights=LossWeights(**config["training"]["loss_weights"]),
                         exec_start_step=config["training"]["exec_start_step"], enable_ifp=config["ifp"]["enabled"],
                         enable_interaction=config["interaction_supervision"],
                         ifp_weights=tuple(config["ifp"]["mcp_loss_weights"]),
                         learning_rate=config["training"]["learning_rate"])
    trainer.tiny_native = tiny_native
    trainer.native_runtime_config = dict(adapter.native.config)
    if tiny_native:
        trainer.base_identity = {"kind": "tiny-native-full-state", "config": native_config}
    else:
        folder = Path(checkpoint).resolve()
        folder = folder / "transformer" if (folder / "transformer").is_dir() else folder
        weight_files = sorted(folder.glob("*.safetensors"))
        if not weight_files:
            raise ValueError("released checkpoint must use inspectable safetensors weights")
        # Read once per run; resume must not silently change the frozen base.
        identities = {}
        for path in [folder / "config.json", *weight_files]:
            identities[path.name] = file_sha256(path)
        trainer.base_identity = {"kind": "released-safetensors", "sha256": identities}
    return trainer, null


def build_batch(sample, config, trainer, null, generator, *, conditional=None):
    from .training import TrainingBatch
    device = next(trainer.codec.parameters()).device
    sample = move(sample, device)
    native = fresh_native(sample, config, generator, device, next(trainer.adapter.native.parameters()).dtype)
    for part in ("current", "remaining"):
        if getattr(sample.requirement, part).step_offsets.tolist() != config[f"{part}_offsets"]:
            raise ValueError(f"{part} offsets differ from the frozen experiment config")
    if conditional is None:
        conditional = bool(torch.rand((), generator=generator) >= config["conditioning"]["unconditional_probability"])
    chunk = native["chunk_size"]
    video_shape = list(native["latent_dict"]["latent"].shape)
    video_shape[2] = chunk
    native_dtype = next(trainer.adapter.native.parameters()).dtype
    sample_noise = torch.randn(video_shape, generator=generator).to(device=device, dtype=native_dtype)
    action = native["action_dict"]
    action_times = action["timesteps"][:, :chunk].reshape(-1)
    history = sample.native_history(dtype=native_dtype)
    execution_valid = torch.ones_like(action["noisy_latents"][:, :, :chunk], dtype=torch.bool)
    for key in ("actions_mask", "valid_mask"):
        if key in action:
            execution_valid &= torch.broadcast_to(action[key].bool(), action["noisy_latents"].shape)[:, :, :chunk]
    return TrainingBatch(
        entity_features=sample.robot_history[:, -1], entity_history=sample.robot_history,
        proprio_history=sample.proprio_history, embodiment=sample.embodiment,
        null_text=null, native_inputs=native, conditional=conditional,
        requirements=sample.requirement if conditional else None,
        demonstrations=sample.demonstrations if conditional else (), outcome=sample.outcome,
        physical_actions=sample.actions, entity_patch_weights=sample.entity_patch_weights,
        sample_noise=sample_noise, noisy_actions=action["noisy_latents"][:, :, :chunk],
        action_timestep=action_times, execution_valid=execution_valid,
        sample_kwargs={"history": history, "steps": 4, "shift": config["video_snr_shift"],
                       **sample.sampling_position()},
        per_view_valid=sample.per_view_valid if conditional else None,
        pair_kind=sample.pair_kind if conditional else "none")


def adapter_state(trainer):
    """Save learned artifacts without duplicating the immutable 10.8B base."""
    return {key: value.detach().cpu() for key, value in trainer.state_dict().items()
            if getattr(trainer, "tiny_native", False) or not key.startswith("adapter.native.") or ".down." in key or ".up." in key
            or key.startswith("adapter.native.mcp_")}


def save_run(path, trainer, config, generator, attempted_steps, tiny_native):
    from .zerowam import ZERO_WAM_COMMIT
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    payload = {"format_version": 2, "upstream_commit": ZERO_WAM_COMMIT, "config": config,
               "stage": trainer.stage, "model": adapter_state(trainer), "optimizer": trainer.optimizer.state_dict(),
               "updates": trainer.updates, "attempted_steps": attempted_steps, "rng": generator.get_state(),
               "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               "tiny_native": tiny_native, "base_identity": trainer.base_identity,
               "action_spaces": getattr(trainer, "action_spaces", {}),
               "demonstration_encoding": getattr(trainer, "demonstration_encoding", {"kind": "raw_features"}),
               "video_pretraining_sources": getattr(trainer, "video_pretraining_sources", [])}
    torch.save(payload, temporary)
    temporary.replace(path)


def load_run(path, *, mmap=False):
    from .zerowam import ZERO_WAM_COMMIT
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    if (not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 2
            or checkpoint.get("upstream_commit") != ZERO_WAM_COMMIT):
        raise ValueError("artifact source/schema differs; v2 requires retraining the effect interface")
    return checkpoint


def restore_run(path, trainer, config, generator, *, resume):
    checkpoint = load_run(path)
    if checkpoint["tiny_native"] != trainer.tiny_native or checkpoint["base_identity"] != trainer.base_identity:
        raise ValueError("artifact frozen base identity or tiny/native mode differs")
    if resume and (checkpoint["stage"] != trainer.stage or checkpoint["config"] != config):
        raise ValueError("resume requires the same stage and full config; use initialize for stage transitions")
    if getattr(trainer, "action_spaces", {}) and checkpoint.get("action_spaces", {}) != trainer.action_spaces:
        raise ValueError("action normalization registry differs from the learned interface")
    encoding = checkpoint.get("demonstration_encoding", {"kind": "raw_features"})
    if hasattr(trainer, "demonstration_encoding") and trainer.demonstration_encoding != encoding:
        raise ValueError("demonstration encoding identity differs from the trained reader")
    for key in ("dimensions", "interface", "tokens", "current_offsets", "remaining_offsets", "lora", "binding_policy"):
        if checkpoint["config"][key] != config[key]:
            raise ValueError(f"artifact {key} differs; do not mix interface controls or token contracts")
    expected = adapter_state(trainer)
    if set(checkpoint["model"]) != set(expected):
        raise ValueError("artifact parameter keys do not match the instantiated adapter")
    if any(checkpoint["model"][key].shape != value.shape for key, value in expected.items()):
        raise ValueError("artifact tensor shapes differ from the current interface architecture")
    trainer.load_state_dict(checkpoint["model"], strict=False)
    trainer.action_spaces = checkpoint.get("action_spaces", {})
    trainer.demonstration_encoding = encoding
    trainer.video_pretraining_sources = checkpoint.get("video_pretraining_sources", [])
    if resume:
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.updates = checkpoint["updates"]
        generator.set_state(checkpoint["rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if torch.cuda.is_available() and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        return checkpoint["attempted_steps"]
    return 0


def train(args):
    from .data import load_experiment, load_sample, validate_demo_encoding
    config = load_experiment(args.config)
    samples, records = load_index(args.index, "train")
    encodings = [validate_demo_encoding(record, config["dimensions"]["demo_dim"]) for record in records]
    if any(encoding != encodings[0] for encoding in encodings):
        raise ValueError("all dataset splits must use the same frozen demonstration representation")
    encoding = encodings[0]
    pretraining_sources = []
    if encoding["kind"] == "video_effect_tokens":
        from .video_cli import load_video_encoder, encoding_metadata
        from .video_data import validate_video_sources
        if not args.demo_encoder:
            raise ValueError("encoded demonstrations require --demo-encoder for identity and source-split auditing")
        _, video_payload = load_video_encoder(args.demo_encoder)
        if encoding_metadata(video_payload, file_sha256(args.demo_encoder)) != encoding:
            raise ValueError("demonstration tokens were produced by a different video encoder or window policy")
        pretraining_sources = video_payload.get("training_source_records", video_payload["source_records"])
        validate_video_sources([*pretraining_sources, *[dict(record, record_kind="bridge") for record in records]])
    elif args.demo_encoder:
        raise ValueError("raw features cannot claim a video encoder; encode demonstrations first")
    if args.stage != "interface" and not (args.resume or args.initialize):
        raise ValueError("reader/joint stages require the validated preceding stage artifact")
    if args.resume or args.initialize:
        header = load_run(args.resume or args.initialize, mmap=True)
        if header.get("physical_calibration"):
            raise ValueError("candidate-calibrated policy is locked; continue training from its pre-calibration artifact")
        del header
    if config["training"]["batch_size"] != 1 or config["training"]["gradient_accumulation"] != 1:
        raise ValueError("this entry point supports the pinned native single-sample rank and no accumulation")
    if args.steps < 1 or args.steps > config["training"]["max_steps"]:
        raise ValueError("steps must fit the registered experiment budget")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("use a fresh output directory, or explicitly resume the same run")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    trainer, null = build_trainer(config, checkpoint=args.checkpoint, tiny_native=args.tiny_native,
                                  device=args.device, stage=args.stage)
    trainer.demonstration_encoding = encoding
    trainer.video_pretraining_sources = pretraining_sources
    trainer.action_spaces = {}
    for record in records:
        if record["split"] == "train":
            space = action_space(record, config["dimensions"]["action_dim"])
            key = space["normalization_id"]
            if key in trainer.action_spaces and trainer.action_spaces[key] != space:
                raise ValueError("one action normalization ID cannot describe different formats")
            trainer.action_spaces[key] = space
    start = 0
    if args.resume or args.initialize:
        start = restore_run(args.resume or args.initialize, trainer, config, generator, resume=bool(args.resume))
    if start + args.steps > config["training"]["max_steps"]:
        raise ValueError("cumulative attempted steps exceed the registered training budget")
    write_json(output / "config.json", config)
    write_json(output / "native_config.json", trainer.native_runtime_config)
    revision = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True)
    write_json(output / "run.json", {"stage": args.stage, "seed": args.seed,
        "requested_steps": args.steps, "starting_attempted_step": start,
        "index_sha256": file_sha256(args.index), "evo_revision": revision.stdout.strip(),
        "worktree_dirty": bool(dirty.stdout.strip()), "torch": torch.__version__, "device": args.device,
        "tiny_native": args.tiny_native, "base_identity": trainer.base_identity,
        "action_spaces": trainer.action_spaces, "demonstration_encoding": trainer.demonstration_encoding,
        "video_pretraining_source_records": len(trainer.video_pretraining_sources)})
    started = time.monotonic()
    metrics_path = output / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as log:
        for step in range(start, start + args.steps):
            sample = load_sample(samples[step % len(samples)])
            batch = build_batch(sample, config, trainer, null, generator)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            before = time.monotonic()
            metrics = trainer.train_step(batch)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            metrics.update(step=step, conditional=batch.conditional, stage=args.stage,
                           view_count=len(batch.demonstrations), pair_kind=batch.pair_kind,
                           compute_seconds=time.monotonic() - before,
                           provenance="tiny-native-random-weights" if args.tiny_native else "checkpoint-training")
            log.write(json.dumps(metrics, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps(metrics, allow_nan=False), flush=True)
            if (step + 1) % args.save_every == 0:
                save_run(output / "adapter.pt", trainer, config, generator, step + 1, args.tiny_native)
    save_run(output / "adapter.pt", trainer, config, generator, start + args.steps, args.tiny_native)
    return {"output": str(output), "updates": trainer.updates, "elapsed_seconds": time.monotonic() - started,
            "robot_evaluation_performed": False}


def calibrate_physical(args):
    """One bounded F-only pass on recorded training candidates, including prefixes."""
    from .data import load_sample
    from .models import CausalEffectPredictor, physical_prediction_loss
    from .zerowam import ZERO_WAM_COMMIT
    artifact = load_run(args.artifact)
    if artifact["upstream_commit"] != ZERO_WAM_COMMIT or artifact.get("physical_calibration"):
        raise ValueError("use a matching artifact that has not already had its F calibration pass")
    if artifact["stage"] not in {"reader", "joint"}:
        raise ValueError("calibrate current policy candidates only after reader training")
    config, dims = artifact["config"], artifact["config"]["dimensions"]
    if not 1 <= args.steps <= config["training"]["max_steps"]:
        raise ValueError("physical calibration must fit the registered finite budget")
    paths, _ = load_index(args.index, "train")
    output = Path(args.output)
    if output.exists():
        raise ValueError("physical calibration writes a new artifact; output already exists")
    torch.manual_seed(args.seed)
    predictor = CausalEffectPredictor(dims["entity_dim"], dims["proprio_dim"], dims["action_dim"],
        dims["embodiment_dim"], dims["geometry_dim"], dims["relation_dim"], dims["event_dim"]).to(args.device)
    predictor.load_state_dict({key.removeprefix("physical."): value for key, value in artifact["model"].items()
                               if key.startswith("physical.")}, strict=True)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=config["training"]["learning_rate"], weight_decay=0.0)
    policy_digest = file_sha256(args.artifact)
    losses, interactions, provenance = [], {}, set()
    for step in range(args.steps):
        sample = move(load_sample(paths[step % len(paths)]), args.device)
        space = action_space(sample.metadata, dims["action_dim"])
        if artifact.get("action_spaces", {}).get(space["normalization_id"]) != space:
            raise ValueError("candidate calibration uses an unknown action normalization")
        channel_mask = torch.tensor(space["valid_channels"], device=args.device)
        if (sample.actions[..., ~channel_mask] != 0).any():
            raise ValueError("candidate inactive action channels must be zero")
        if sample.metadata.get("policy_artifact_sha256") != policy_digest:
            raise ValueError("candidate data must identify the exact frozen policy artifact that generated it")
        if not any(bool(mask.any()) for mask in sample.outcome.label_valid.values()):
            raise ValueError("candidate calibration needs observed outcome labels")
        optimizer.zero_grad(set_to_none=True)
        prediction = predictor(sample.robot_history, sample.proprio_history, sample.actions, sample.embodiment,
                               entity_present=sample.outcome.entity_ids >= 0)
        loss = physical_prediction_loss(prediction, sample.outcome)
        if not torch.isfinite(loss):
            raise ValueError("non-finite calibration loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        key = (sample.metadata["trajectory_id"], sample.metadata["window_start"])
        interactions[key] = sample.metadata["executed_steps"]
        provenance.add(str(sample.metadata.get("provenance", "provided-recording")))
    for key, value in predictor.state_dict().items():
        artifact["model"]["physical." + key] = value.detach().cpu()
    report = {"optimizer_steps": args.steps, "unique_recorded_windows": len(interactions),
              "recorded_action_steps": sum(interactions.values()), "loss_first": losses[0], "loss_last": losses[-1],
              "thresholds_calibrated": False, "success_probability_claim": False,
              "data_provenance": sorted(provenance), "tiny_native": artifact["tiny_native"],
              "index_sha256": hashlib.sha256(Path(args.index).read_bytes()).hexdigest()}
    artifact["physical_calibration"] = report
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(output)
    write_json(output.with_suffix(".json"), report)
    return {**report, "artifact": str(output)}


@torch.no_grad()
def predict(args):
    """Generate normalized candidate arrays; never issue live robot commands."""
    from .data import load_observation
    from .models import effect_cost
    from .zerowam import TaskConditions
    artifact = load_run(args.artifact)
    config = artifact["config"]
    if artifact["stage"] not in {"reader", "joint"}:
        raise ValueError("demonstration inference requires a trained reader artifact")
    if not args.diagnostic and not config.get("validation_locked", False):
        raise ValueError("formal prediction/evaluation needs a validation-locked config; use diagnostic for interface checks")
    if args.candidates == 4 and not artifact.get("physical_calibration"):
        raise ValueError("four-candidate F ranking requires the bounded candidate-distribution calibration")
    if not args.diagnostic and not config["binding_policy"]["validation_locked"]:
        raise ValueError("formal inference requires validation-locked binding thresholds")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("prediction output must be a fresh directory")
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    trainer, null = build_trainer(config, checkpoint=args.checkpoint, tiny_native=artifact["tiny_native"],
                                  device=args.device, stage="reader")
    restore_run(args.artifact, trainer, config, generator, resume=False)
    trainer.eval().requires_grad_(False)
    sample = move(load_observation(args.manifest), args.device)
    space = action_space({"action_space": sample.action_space}, config["dimensions"]["action_dim"])
    if trainer.action_spaces.get(space["normalization_id"]) != space:
        raise ValueError("observation action normalization differs from the trained interface")
    from .inference import NativePolicy, RequirementRejected
    policy = NativePolicy(trainer, null, config, sampling_steps=args.sampling_steps, view=args.view,
                          diagnostic=args.diagnostic)
    scoring = {}
    if args.candidates == 4 and not args.diagnostic:
        if not args.scoring_config:
            raise ValueError("formal ranking requires a separately validation-locked scoring config")
        scoring = json.loads(Path(args.scoring_config).read_text())
        digest = file_sha256(args.artifact)
        if scoring.get("validation_locked") is not True or scoring.get("policy_artifact_sha256") != digest:
            raise ValueError("scoring parameters must be locked for this exact calibrated artifact")
        if args.max_cost is not None:
            raise ValueError("formal ranking cannot override its locked cost threshold")
        if not isinstance(scoring.get("max_cost"), (int, float)) or not np.isfinite(scoring["max_cost"]) or scoring["max_cost"] < 0:
            raise ValueError("locked max_cost must be finite and nonnegative")
        weights = scoring.get("field_weights", {})
        if set(weights) != {"geometry", "relations", "events"} or any(not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0 for v in weights.values()):
            raise ValueError("locked scoring must explicitly state every finite positive field weight")
        event_threshold = scoring.get("event_threshold")
        if not isinstance(event_threshold, (int, float)) or not np.isfinite(event_threshold) or not 0 < event_threshold < 1:
            raise ValueError("locked scoring must state its event-occurrence threshold in (0,1)")
        uncertainty = scoring.get("uncertainty_weight")
        if not isinstance(uncertainty, (int, float)) or not np.isfinite(uncertainty) or uncertainty < 0:
            raise ValueError("locked scoring must explicitly state its nonnegative uncertainty weight")
    try:
        candidates, requirements = policy.candidates(sample, None, random.Random(args.seed), count=args.candidates)
    except RequirementRejected as error:
        output.mkdir(parents=True, exist_ok=True)
        report = {"status": "rejected", "reason": str(error), "selected_index": None,
                  "commands_sent": 0, "diagnostic": args.diagnostic}
        write_json(output / "prediction.json", report)
        return report
    requirement = requirements.current
    actions = [np.asarray(candidate.actions, dtype=np.float32) for candidate in candidates]
    costs = []
    if args.candidates == 4:
        for array in actions:
            physical_action = torch.from_numpy(array).to(args.device)[None]
            outcome = trainer.physical(sample.robot_history, sample.proprio_history, physical_action,
                                        sample.embodiment, entity_present=sample.entity_ids >= 0)
            costs.append(float(effect_cost(outcome, requirement, field_weights=scoring.get("field_weights"),
                                           uncertainty_weight=scoring.get("uncertainty_weight", 1.0),
                                           event_threshold=scoring.get("event_threshold", 0.5))[0]))
    from .evaluation import Candidate, rank_candidates
    chosen, status = 0, "selected"
    if costs:
        threshold = scoring.get("max_cost", float("inf") if args.max_cost is None else args.max_cost)
        selection = rank_candidates([Candidate(str(i), array.tolist()) for i, array in enumerate(actions)], costs, max_cost=threshold)
        chosen = int(selection.candidate_id) if selection.candidate_id is not None else None
        status = selection.status
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "candidates.npz", **{f"candidate_{i}": array for i, array in enumerate(actions)})
    report = {"status": status, "selected_index": chosen, "candidate_count": len(actions),
              "costs": [value if np.isfinite(value) else None for value in costs],
              "current_binding_slots": requirement.binding[0].tolist(), "entity_ids": sample.entity_ids[0].tolist(),
              "prefix_steps": max(1, actions[0].shape[0] // 4), "normalized_actions": True,
              "action_space": space,
              "commands_sent": 0, "diagnostic": args.diagnostic,
              "provenance": "tiny-native-random-base" if artifact["tiny_native"] else "released-base-with-adapters"}
    write_json(output / "prediction.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    commands.add_parser("check-native")
    commands.add_parser("check")
    visual = commands.add_parser("preprocess-visual")
    visual.add_argument("--manifest", required=True)
    visual.add_argument("--output", required=True)
    visual.add_argument("--device", default="cuda")
    video_visual = commands.add_parser("preprocess-video")
    video_visual.add_argument("--manifest", required=True)
    video_visual.add_argument("--output", required=True)
    video_visual.add_argument("--device", default="cuda")
    video_train = commands.add_parser("pretrain-video")
    video_train.add_argument("--config", required=True)
    video_train.add_argument("--index", required=True)
    video_train.add_argument("--output", required=True)
    video_train.add_argument("--resume")
    video_train.add_argument("--steps", type=int, default=1)
    video_train.add_argument("--seed", type=int, default=0)
    video_train.add_argument("--device", default="cpu")
    video_eval = commands.add_parser("evaluate-video")
    video_eval.add_argument("--artifact", required=True)
    video_eval.add_argument("--index", required=True)
    video_eval.add_argument("--split", choices=("validation", "test"), default="validation")
    video_eval.add_argument("--max-samples", type=int, default=100)
    video_eval.add_argument("--device", default="cpu")
    video_eval.add_argument("--output", required=True)
    encode_demo = commands.add_parser("encode-demonstrations")
    encode_demo.add_argument("--artifact", required=True)
    encode_demo.add_argument("--manifest", required=True)
    encode_demo.add_argument("--output", required=True)
    encode_demo.add_argument("--device", default="cpu")
    fixture = commands.add_parser("make-fixture")
    fixture.add_argument("--output", required=True)
    validate = commands.add_parser("validate-data")
    validate.add_argument("--index", required=True)
    run = commands.add_parser("train")
    run.add_argument("--config", required=True)
    run.add_argument("--index", required=True)
    run.add_argument("--stage", choices=("interface", "reader", "joint"), required=True)
    run.add_argument("--checkpoint")
    run.add_argument("--demo-encoder", help="local encoder artifact matching precomputed effect tokens")
    run.add_argument("--tiny-native", action="store_true")
    run.add_argument("--device", default="cuda")
    run.add_argument("--steps", type=int, default=1)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--save-every", type=int, default=100)
    run.add_argument("--output", required=True)
    restore = run.add_mutually_exclusive_group()
    restore.add_argument("--resume")
    restore.add_argument("--initialize")
    calibration = commands.add_parser("calibrate-f")
    calibration.add_argument("--artifact", required=True)
    calibration.add_argument("--index", required=True)
    calibration.add_argument("--steps", type=int, default=10)
    calibration.add_argument("--seed", type=int, default=0)
    calibration.add_argument("--device", default="cpu")
    calibration.add_argument("--output", required=True)
    inference = commands.add_parser("predict")
    inference.add_argument("--artifact", required=True)
    inference.add_argument("--manifest", required=True)
    inference.add_argument("--checkpoint")
    inference.add_argument("--view", type=int, choices=(0, 1), default=0)
    inference.add_argument("--candidates", type=int, choices=(1, 4), default=1)
    inference.add_argument("--sampling-steps", type=int, default=4)
    inference.add_argument("--max-cost", type=float)
    inference.add_argument("--scoring-config")
    inference.add_argument("--seed", type=int, default=0)
    inference.add_argument("--device", default="cuda")
    inference.add_argument("--diagnostic", action="store_true")
    inference.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor()
        elif args.command == "make-fixture":
            result = make_fixture(args.output)
        elif args.command == "preprocess-visual":
            from .vision import preprocess
            result = preprocess(args.manifest, args.output, device=args.device)
        elif args.command == "preprocess-video":
            from .vision import preprocess_video
            result = preprocess_video(args.manifest, args.output, device=args.device)
        elif args.command == "pretrain-video":
            from .video_cli import train_video
            result = train_video(args)
        elif args.command == "evaluate-video":
            from .video_cli import evaluate_video
            result = evaluate_video(args)
        elif args.command == "encode-demonstrations":
            from .video_cli import encode_demonstrations
            result = encode_demonstrations(args)
        elif args.command == "validate-data":
            from .data import load_sample
            samples, records = load_index(args.index)
            for path in samples:
                load_sample(path)
            result = {"train_samples": len(samples), "total_samples": len(records), "split_integrity": "passed"}
        elif args.command == "check-native":
            from .zerowam import tiny_native_smoke
            result = tiny_native_smoke()
        elif args.command == "check":
            completed = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"])
            return completed.returncode
        elif args.command == "calibrate-f":
            result = calibrate_physical(args)
        elif args.command == "predict":
            result = predict(args)
        else:
            if args.save_every < 1:
                raise ValueError("save-every must be positive")
            result = train(args)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
