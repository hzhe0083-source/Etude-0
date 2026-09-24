"""FSDP2 training for the complete pi action branch over a shared frozen base."""

from contextlib import contextmanager
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import time
from types import MethodType

import numpy as np
import torch
from torch import distributed as dist, nn


def distributed_settings(config):
    settings = config.get("distributed", {})
    if (not isinstance(settings, dict) or settings.keys() - {"enabled", "activation_checkpointing"}
            or any(type(value) is not bool for value in settings.values())):
        raise ValueError("distributed requires Boolean enabled and activation_checkpointing only")
    result = {"enabled": False, "activation_checkpointing": True, **settings}
    if result["enabled"] and config.get("interface_type") != "pi_goal":
        raise ValueError("FSDP training is supported only for the independent pi route")
    return result


def validate_distributed_artifact(payload, *, kind):
    settings = distributed_settings(payload["config"])
    if not settings["enabled"]:
        if "distributed" in payload or "rank_states" in payload:
            raise ValueError("disabled FSDP configuration cannot contain distributed training state")
        return
    metadata = payload.get("distributed")
    if (not isinstance(metadata, dict) or metadata.keys() != {"kind", "world_size", "activation_checkpointing", "batch_layout", "optimizer_format"}
            or metadata["kind"] != "fsdp2" or type(metadata["world_size"]) is not int or metadata["world_size"] < 1
            or metadata["activation_checkpointing"] != settings["activation_checkpointing"]
            or metadata["batch_layout"] != "one_sample_per_rank_round_robin" or metadata["optimizer_format"] != "full_named_v1"):
        raise ValueError("FSDP artifact requires matching sharding, world size, batch and optimizer metadata")
    if kind == "g_pi_training":
        states = payload.get("rank_states")
        if (not isinstance(states, list) or len(states) != metadata["world_size"]
                or any(not isinstance(state, dict) or set(state) != {"rng", "torch_rng", "cuda_rng", "python_rng", "numpy_rng", "visited_arrays"}
                       or set(state["rng"]) != {"sample", "action", "goal", "language"} for state in states)):
            raise ValueError("FSDP checkpoint requires one complete RNG and data state per rank")


def canonical_name(name):
    return name.replace("_checkpoint_wrapped_module.", "")


def full_tensor(value):
    from torch.distributed.tensor import DTensor

    value = value.detach()
    return (value.full_tensor() if isinstance(value, DTensor) else value).cpu()


def _masked_forward(self, *args, _g_pi_masks=None, **kwargs):
    if _g_pi_masks is None:
        return self._g_pi_original_forward(*args, **kwargs)
    previous = self.attn1.self_block_mask, self.attn2.cross_block_mask
    self.attn1.self_block_mask, self.attn2.cross_block_mask = _g_pi_masks
    try:
        return self._g_pi_original_forward(*args, **kwargs)
    finally:
        self.attn1.self_block_mask, self.attn2.cross_block_mask = previous


def _checkpoint_forward(self, *args, **kwargs):
    block = self._checkpoint_wrapped_module
    kwargs["_g_pi_masks"] = block.attn1.self_block_mask, block.attn2.cross_block_mask
    return self._g_pi_checkpoint_forward(*args, **kwargs)


def shard_pi(native, interface, config):
    """Use upstream sharding; register the actual manually invoked forward paths."""
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, register_fsdp_forward_method
    from wan_va.distributed.fsdp import apply_ac, shard_model

    settings = distributed_settings(config)
    if settings["activation_checkpointing"]:
        for block in native.blocks:
            block._g_pi_original_forward = block.forward
            block.forward = MethodType(_masked_forward, block)
        apply_ac(native)
        for block in native.blocks:
            # The action runner restores masks before backward. Save the actual
            # masks as checkpoint inputs so recomputation never sees stale ones.
            block._g_pi_checkpoint_forward = block.forward
            block.forward = MethodType(_checkpoint_forward, block)
    # Preserve FP32 action master weights and BF16 immutable video storage.
    # Casting the whole native to BF16 would round action masters and can also
    # change E. Existing CUDA autocast supplies BF16 compute for trainables.
    shard_model(native, param_dtype=None, reduce_dtype=torch.float32)
    fully_shard(interface, mp_policy=MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32,
                                                        cast_forward_inputs=False))
    register_fsdp_forward_method(interface, "conditions")

    def loss(self, encoder, sample, generators, layers, cached_z):
        from .g_pi_training import g_pi_training_loss

        return g_pi_training_loss(self, interface, encoder, sample, config, generators,
                                  stage="pi", feature_layers=layers, cached_z=cached_z)

    def encode(self, encoder, frame):
        return encoder(frame)

    native.g_pi_encode = MethodType(encode, native)
    register_fsdp_forward_method(native, "g_pi_encode")
    native.g_pi_loss = MethodType(loss, native)
    register_fsdp_forward_method(native, "g_pi_loss")
    return nn.ModuleDict({"native": native, "interface": interface})


def distributed_checksum(native):
    from .g_pi_context import assert_frozen_base, frozen_base_named_parameters, _tensor_hash

    assert_frozen_base(native)
    digest = hashlib.sha256()
    for name, value in list(frozen_base_named_parameters(native)) + list(native.named_buffers()):
        digest.update(canonical_name(name).encode())
        digest.update(_tensor_hash(full_tensor(value)).encode())
    return digest.hexdigest()


def trainable_state(native, interface):
    from .goal_action import action_named_parameters

    # Every rank must participate in full_tensor's all-gathers, including when
    # only rank zero writes the portable (unsharded) artifact.
    return {"interface": {canonical_name(k): full_tensor(v) for k, v in interface.state_dict().items()},
            "action": {canonical_name(k): full_tensor(v) for k, v in action_named_parameters(native)}}


def rank_rng_state(generators, visited):
    n = np.random.get_state()
    return {"rng": {k: g.get_state() for k, g in generators.items()}, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(), "python_rng": random.getstate(),
            "numpy_rng": [n[0], n[1].tolist(), *n[2:]], "visited_arrays": visited}


def restore_rank_rng(state, generators):
    for name, generator in generators.items():
        generator.set_state(state["rng"][name])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    random.setstate(state["python_rng"])
    n = state["numpy_rng"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))


@contextmanager
def process_group(device):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("FSDP pi requires CUDA and torchrun's RANK/WORLD_SIZE/LOCAL_RANK environment")
    if not {"RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"} <= os.environ.keys():
        raise ValueError("FSDP pi must be launched with torchrun, including the single-GPU path")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    owned = not dist.is_initialized()
    if owned:
        dist.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=torch.device("cuda", local_rank))
    try:
        if dist.get_backend() != "nccl":
            raise ValueError("FSDP pi requires an NCCL process group")
        yield dist.get_rank(), dist.get_world_size(), f"cuda:{local_rank}"
    finally:
        if owned:
            dist.destroy_process_group()


def rank_consensus(value):
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, value)
    if any(record != records[0] for record in records[1:]):
        raise ValueError("FSDP ranks disagree on configuration, robot data or frozen base identity")


def merge_input_hashes(visited, current):
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, current)
    merged = dict(visited)
    for record in records:
        for name, digest in record.items():
            if name in merged and merged[name] != digest:
                raise ValueError("consumed training inputs differ across FSDP ranks or changed during training")
            merged[name] = digest
    return merged


def train_distributed_pi(args, config):
    with process_group(args.device) as (rank, world, device):
        return _train(args, config, rank, world, device)


def _train(args, config, rank, world, device):
    from torch.distributed.checkpoint.state_dict import (get_optimizer_state_dict, set_optimizer_state_dict,
                                                       StateDictOptions)
    from .cli import file_sha256, write_json
    from .g_pi_data import g_pi_sample_files, load_g_pi_index, load_g_pi_sample
    from .g_pi_training import (build_g_pi_system, _base_location, _base_reference, _check_sample,
        _check_stage, _frozen_checksums, _optimizer, _restore_system, conditioning_mode,
        g_pi_architecture, g_pi_artifact_version, language_drop_probability, read_g_pi_artifact)
    from .goal_training import goal_registry
    from .video_data import _source_components
    from .zerowam import ZERO_WAM_COMMIT

    _check_stage(config, args.stage)
    if getattr(args, "initialize", None):
        raise ValueError("G and pi train independently; stage initialization is unsupported")
    if type(args.steps) is not int or args.steps < 1:
        raise ValueError("steps must be positive")
    paths, sources = load_g_pi_index(args.index)
    # One local trajectory sample per rank; an update is a global world-size
    # batch. Wrapping is explicit and allows tiny datasets for smoke tests.
    first = load_g_pi_sample(paths[0], route="pi_goal", generator=torch.Generator().manual_seed(args.seed))
    registry = goal_registry(first)
    _check_sample(first, config, registry)
    previous = read_g_pi_artifact(args.resume) if args.resume else None
    distribution = {"kind": "fsdp2", "world_size": world,
                    "activation_checkpointing": distributed_settings(config)["activation_checkpointing"],
                    "batch_layout": "one_sample_per_rank_round_robin", "optimizer_format": "full_named_v1"}
    identities = {"index": file_sha256(args.index), **{str(p.resolve()): file_sha256(p) for p in paths}}
    start = previous["attempted_steps"] if previous else 0
    if start + args.steps > config["training"]["max_steps"]:
        raise ValueError("cumulative independent stage budget exceeded")
    if previous and (previous["config"] != config or previous["stage"] != "pi"
            or previous["registry"] != registry or previous["tiny_native"] != args.tiny_native
            or previous["data_identity"] != identities or previous["seed"] != args.seed
            or previous.get("distributed") != distribution or len(previous.get("rank_states", [])) != world):
        raise ValueError("exact FSDP resume requires identical world size, route, config, data, conventions and seed")
    if previous:
        _source_components([*previous["source_records"], *sources])
    visited = dict(previous["rank_states"][rank]["visited_arrays"]) if previous else {}
    if any(file_sha256(path) != digest for path, digest in visited.items()):
        raise ValueError("consumed training inputs changed before resume")
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not previous:
        raise ValueError("training requires a fresh output directory or explicit resume")
    # Initialization identical across ranks; local RNG streams begin only after
    # model construction. A fixed world size is required for exact continuation.
    torch.manual_seed(args.seed)
    checkpoint = _base_location(previous["base_reference"], args.checkpoint) if previous else args.checkpoint
    native, interface, encoder, identity, layers = build_g_pi_system(config, registry, stage="pi",
        checkpoint=checkpoint, tiny_native=args.tiny_native, device=device,
        video_precision=previous["encoder_precision"] if previous else None)
    if previous:
        if previous["base_identity"] != identity or previous["feature_layers"] != layers:
            raise ValueError("base checkpoint or feature layers changed")
        encoder.validate_identity(previous["encoder_identity"])
        _restore_system(native, interface, encoder, previous["model"])
    frozen = _frozen_checksums(native, encoder, config)
    if frozen["encoder"] != encoder.identity["base_sha256"] or (previous and frozen != previous["frozen_checksums"]):
        raise ValueError("frozen base checksum differs from E or resumed artifact")
    reference = _base_reference(config, checkpoint, args.tiny_native, identity, encoder)
    cache = None
    if config.get("target_cache_index"):
        from .g_pi_targets import load_target_cache_index

        path = Path(config["target_cache_index"])
        if not path.is_absolute():
            path = Path(args.config).resolve().parent / path
        cache = load_target_cache_index(path, encoder.identity, task_paths=paths)
        for file in cache.files:
            name, digest = str(file.resolve()), file_sha256(file)
            if name in visited and visited[name] != digest:
                raise ValueError("consumed target cache changed before resume")
            visited[name] = digest
    rank_consensus({"config": config, "registry": registry, "data_identity": identities,
                    "base_reference": reference, "encoder_identity": encoder.identity})
    visited = merge_input_hashes({}, visited)
    model = shard_pi(native, interface, config)
    optimizer = _optimizer(native, interface, encoder, config)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    rank_seed = args.seed + 104729 * rank
    torch.manual_seed(rank_seed)
    random.seed(rank_seed)
    np.random.seed(rank_seed % 2**32)
    generators = {name: torch.Generator().manual_seed(rank_seed + offset)
                  for offset, name in enumerate(("sample", "action", "goal", "language"))}
    updates = previous["updates"] if previous else 0
    if previous:
        if (previous["scheduler"] != {"kind": "constant", "state": None}
                or previous["data_cursor"] != (start * world) % len(paths)):
            raise ValueError("FSDP scheduler or global data cursor differs from this trainer")
        set_optimizer_state_dict(model, optimizer, previous["optimizer"],
                                 options=StateDictOptions(full_state_dict=True, strict=True))
        restore_rank_rng(previous["rank_states"][rank], generators)
    visual_space = previous.get("visual_feature_space") if previous else None
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "config.json", config)
    dist.barrier()
    started = time.monotonic()
    for step in range(start, start + args.steps):
        path = paths[(step * world + rank) % len(paths)]
        sample = load_g_pi_sample(path, route="pi_goal", generator=generators["sample"])
        _check_sample(sample, config, registry)
        space = sample.metadata["feature_space_id"]
        if visual_space is not None and space != visual_space:
            raise ValueError("visual feature space changed across tasks")
        visual_space = space
        current = {str(file.resolve()): file_sha256(file) for file in g_pi_sample_files(path, sample)}
        visited = merge_input_hashes(visited, current)
        optimizer.zero_grad(set_to_none=True)
        losses = native.g_pi_loss(encoder, sample, generators, layers, cache.goal(sample, path) if cache else None)
        finite = torch.isfinite(losses["total"]).to(dtype=torch.int)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise ValueError("nonfinite objective on an FSDP rank; optimizer not updated")
        losses["total"].backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, config["training"]["gradient_clip"],
                                             error_if_nonfinite=True, foreach=False)
        optimizer.step()
        updates += 1
        metrics = torch.stack([losses["action"].detach().float(), norm.detach().float().full_tensor()]).to(device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= world
        if rank == 0:
            with (output / "metrics.jsonl").open("a") as log:
                log.write(json.dumps({"step": step, "stage": "pi", "action": metrics[0].item(),
                    "total": metrics[0].item(), "gradient_norm": metrics[1].item(), "paired_samples": world,
                    "video_generation": False, "updated": True}) + "\n")
    checksum = distributed_checksum(native)
    if {"encoder": checksum, "native_video": checksum} != frozen:
        raise ValueError("frozen video backbone or E changed during FSDP training")
    weights = trainable_state(native, interface)
    optim = get_optimizer_state_dict(model, optimizer,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True))
    states = [None] * world
    dist.all_gather_object(states, rank_rng_state(generators, visited))
    spaces = [None] * world
    dist.all_gather_object(spaces, visual_space)
    if len(set(spaces)) != 1:
        raise ValueError("visual feature space differs across FSDP ranks")
    torch.cuda.synchronize()
    report = {"artifact": str((output / "goal_interface.pt").resolve()), "stage": "pi", "updates": updates,
        "interface_type": "pi_goal", "elapsed_seconds": time.monotonic() - started,
        "trainable_parameters": sum(p.numel() for p in parameters), "precision": "float32", "lora": False,
        "encoder_precision": reference["video_precision"], "base_reference": reference, "distributed": distribution,
        "shared_video_base": encoder.native is native, "encoder_identity": encoder.identity, "frozen_checksums": frozen,
        "video_generation": False, "video_supervision": False, "robot_execution_evaluated": False}
    if rank == 0:
        state = states[0]
        payload = {"format_version": g_pi_artifact_version(config), "kind": "g_pi_training",
            "architecture": g_pi_architecture(config), "conditioning_mode": conditioning_mode(config),
            "p_drop": language_drop_probability(config), "precision": "float32",
            "encoder_precision": reference["video_precision"], "upstream_commit": ZERO_WAM_COMMIT,
            "config": config, "stage": "pi", "interface_type": "pi_goal", "model": weights,
            "native_config": {k: v for k, v in dict(native.config).items() if not k.startswith("_")},
            "base_identity": identity, "base_reference": reference, "encoder_identity": encoder.identity,
            "empty_text_identity": encoder.identity["empty_text_identity"], "k_z": encoder.identity["k_z"],
            "d_z": encoder.identity["d_z"], "event_rules": config["event_rules"], "tiny_native": args.tiny_native,
            "feature_layers": layers, "registry": registry, "frozen_checksums": frozen,
            "visual_feature_space": visual_space, "updates": updates, "attempted_steps": start + args.steps,
            "data_cursor": ((start + args.steps) * world) % len(paths), "seed": args.seed, "optimizer": optim,
            "scheduler": {"kind": "constant", "state": None}, "rank_states": states, "distributed": distribution,
            **{key: value for key, value in state.items() if key != "cuda_rng"}, "cuda_rng": [state["cuda_rng"]],
            "data_identity": identities, "source_records": previous["source_records"] if previous else sources}
        temp = output / "goal_interface.pt.tmp"
        torch.save(payload, temp)
        temp.replace(output / "goal_interface.pt")
        write_json(output / "run.json", report)
    dist.barrier()
    return report
