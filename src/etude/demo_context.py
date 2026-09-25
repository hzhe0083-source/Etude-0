"""One compressed demonstration interface for native training and KV caching.

Compressed RoPE coordinates mean (time group, demo namespace, query slot),
not image patches. Target observations and action streams stay native.
"""

from dataclasses import dataclass
from types import MethodType

import torch
from torch import Tensor

from .demo_bottleneck import TemporalDemoBottleneck


@dataclass(frozen=True)
class PreparedDemoContext:
    hidden: Tensor
    temb: Tensor
    timestep_proj: Tensor
    grid_id: Tensor
    tokens: Tensor
    group_times: Tensor
    owner: object


def _check_prepared(native, context):
    if context.owner is not native._demo_context_owner:
        raise ValueError("prepared demonstration belongs to another model")
    if not isinstance(context.hidden, Tensor) or context.hidden.ndim != 3:
        raise ValueError("malformed prepared demonstration hidden")
    count = context.hidden.shape[1]
    width = native.inner_dim
    groups = context.group_times.numel() if isinstance(context.group_times, Tensor) else 0
    if (count < 1 or groups < 1 or count != groups * native.demo_bottleneck.tokens_per_group
            or context.group_times.shape != (groups,)):
        raise ValueError("malformed prepared demonstration time groups")
    for name, shape in (("hidden", (1, count, width)), ("temb", (1, count, width)),
                        ("timestep_proj", (1, count, 6, width)), ("grid_id", (1, 4, count)),
                        ("tokens", (1, count, native.demo_bottleneck.dim))):
        value = getattr(context, name)
        if (not isinstance(value, Tensor) or value.shape != shape
                or value.device != native.patch_embedding_mlp.weight.device):
            raise ValueError(f"malformed prepared demonstration {name}")
        if name != "grid_id" and value.dtype != native.patch_embedding_mlp.weight.dtype:
            raise ValueError(f"prepared demonstration {name} dtype differs from model")
    namespace = getattr(native, "demo_icl_rope_h", None)
    if namespace is not None and torch.any(context.grid_id[:, 1] != namespace):
        raise ValueError("demonstration RoPE namespace differs from the trained model")
    return context.hidden, context.temb, context.timestep_proj


def prepare_demo_context(native, icl):
    """Return a fresh native-compatible ICL dictionary; never retain raw video.

With the interface disabled this returns the original dictionary unchanged.
Prepared contexts can be reused within the same forward graph without running
the compressor twice. They are model-local, not serializable input artifacts.
"""
    if icl is None or not hasattr(native, "demo_bottleneck"):
        return icl
    if not isinstance(icl, dict):
        raise ValueError("demonstration input must be an ICL dictionary")
    latent = icl.get("latent")
    if isinstance(latent, PreparedDemoContext):
        _check_prepared(native, latent)
        if (not isinstance(icl.get("grid_id"), Tensor)
                or not torch.equal(icl["grid_id"], latent.grid_id)
                or not isinstance(icl.get("timesteps"), Tensor)
                or icl["timesteps"].shape != latent.hidden.shape[:2]
                or torch.any(icl["timesteps"] != 0)):
            raise ValueError("prepared demonstration metadata does not match context")
        return icl
    if not isinstance(latent, Tensor) or latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError("raw demonstration latent must have shape [1,C,F,H,W]")
    if not latent.is_floating_point() or not torch.isfinite(latent).all():
        raise ValueError("demonstration latent must be finite floating point")
    pf, ph, pw = native.patch_size
    _, channels, frames, height, width = latent.shape
    if (pf != 1 or channels != native.config.in_channels or min(frames, height, width) < 1
            or height % ph or width % pw):
        raise ValueError("demonstration must match native channels/patch grid with temporal patch size 1")
    times = icl.get("frame_times")
    if (not isinstance(times, Tensor) or times.shape != (frames,)
            or not torch.isfinite(times).all() or torch.any(times[1:] <= times[:-1])):
        raise ValueError("demonstration requires strictly increasing frame_times for every latent frame")
    timesteps = icl.get("timesteps")
    if (not isinstance(timesteps, Tensor) or timesteps.shape != (1, frames)
            or torch.any(timesteps != 0)):
        raise ValueError("demonstration requires clean zero timesteps with shape [1,F]")
    grid = icl.get("grid_id")
    gh, gw = height // ph, width // pw
    if not isinstance(grid, Tensor) or grid.shape != (1, 4, frames * gh * gw):
        raise ValueError("demonstration grid must match its actual patch layout")
    from wan_va.utils import get_mesh_id

    shift = grid[0, 1].min().item()
    if shift < 1 or int(shift) != shift:
        raise ValueError("demonstration requires a positive integer RoPE namespace")
    namespace = getattr(native, "demo_icl_rope_h", None)
    if namespace is not None and shift != namespace:
        raise ValueError("demonstration RoPE namespace differs from the trained model")
    expected = get_mesh_id(frames, gh, gw, 0, h_shift=int(shift)).to(grid)
    if not torch.equal(grid[0], expected):
        raise ValueError("demonstration grid must describe the actual ordered patch layout")
    device = native.patch_embedding_mlp.weight.device
    # The saved native method preserves upstream patch embedding exactly.
    features, _, _ = native._demo_original_training_embed(
        latent.to(device), timesteps.to(device), "video")
    features = features.reshape(1, frames, gh * gw, native.inner_dim)
    yy, xx = torch.meshgrid(torch.arange(gh, device=device), torch.arange(gw, device=device), indexing="ij")
    coords = torch.stack((2 * (xx + .5) / gw - 1, 2 * (yy + .5) / gh - 1), dim=-1).reshape(-1, 2)
    tokens, group_times = native.demo_bottleneck(features, times.to(device), coords)
    hidden = native.demo_bottleneck.adapter(tokens)
    count = hidden.shape[1]
    temb, projection = native.condition_embedder(hidden.new_zeros(1, count), dtype=hidden.dtype)
    groups = group_times.numel()
    slots = count // groups
    compressed_grid = torch.stack((
        torch.arange(groups, device=device).repeat_interleave(slots),
        torch.full((count,), int(shift), device=device),
        torch.arange(slots, device=device).repeat(groups),
        torch.zeros(count, device=device, dtype=torch.long)))[None]
    context = PreparedDemoContext(hidden, temb, projection.unflatten(2, (6, -1)),
                                  compressed_grid, tokens, group_times, native._demo_context_owner)
    return {"latent": context, "timesteps": hidden.new_zeros(1, count),
            "grid_id": compressed_grid, "frame_times": times}


def _training_embed(native, latents, timesteps, mode):
    if isinstance(latents, PreparedDemoContext):
        if mode != "video":
            raise ValueError("compressed demonstrations cannot enter the action stream")
        return _check_prepared(native, latents)
    return native._demo_original_training_embed(latents, timesteps, mode)


def _embed_stream(native, stream, mode):
    context = stream.get("noisy_latents")
    if isinstance(context, PreparedDemoContext):
        if mode != "video":
            raise ValueError("compressed demonstrations cannot enter the action stream")
        return _check_prepared(native, context)
    return native._demo_original_embed_stream(stream, mode)


def _forward_train(native, input_dict):
    icl = input_dict.get("icl_latent_dict")
    if icl is not None:
        input_dict = {**input_dict, "icl_latent_dict": prepare_demo_context(native, icl)}
    return native._demo_original_forward_train(input_dict)


def _forward_stream(native, input_dict, mode, update_cache, cache_name, clean_window_cache):
    stream_key = "latent_res_lst" if mode == "video" else "action_res_lst"
    grid_key = "latent_grid_id" if mode == "video" else "action_grid_id"
    stream = input_dict[stream_key]
    cache_types = stream.get("cache_type_ids")
    is_demo = int(input_dict.get("cache_type_id", 1)) == 2
    if cache_types is not None:
        if not isinstance(cache_types, Tensor) or cache_types.ndim != 1 or not cache_types.numel():
            raise ValueError("cache_type_ids must be a nonempty vector")
        is_demo = is_demo or bool(torch.any(cache_types == 2))
        if is_demo and not torch.all(cache_types == 2):
            raise ValueError("demonstration and target cache types cannot be mixed")
    if isinstance(stream.get("noisy_latents"), PreparedDemoContext) and not is_demo:
        raise ValueError("prepared demonstration requires ICL cache type 2")
    if is_demo:
        if mode != "video":
            raise ValueError("ICL context must use the video stream, never actions")
        if native.type_ids_cache is not None and native.type_ids_cache.numel():
            raise ValueError("demonstration initialization requires an empty cache; reset the episode before replacing context")
        raw = stream.get("noisy_latents")
        if isinstance(raw, PreparedDemoContext):
            _check_prepared(native, raw)
        raw_count = (raw.hidden.shape[1] if isinstance(raw, PreparedDemoContext) else
                     None if not isinstance(raw, Tensor) or raw.ndim != 5 else
                     raw.shape[2] * (raw.shape[3] // native.patch_size[1]) * (raw.shape[4] // native.patch_size[2]))
        if cache_types is not None and cache_types.numel() != raw_count:
            raise ValueError("demonstration cache metadata length does not match context")
        for name in ("current_seq_ids", "current_frame_ids"):
            value = input_dict.get(name)
            if value is not None and (not isinstance(value, Tensor) or value.shape != (raw_count,)
                                      or torch.any(value != 0)):
                raise ValueError(f"demonstration {name} must describe one bidirectional sequence")
        times = stream.get("timesteps")
        if isinstance(times, Tensor) and times.ndim == 1:
            times = times[None]
        grid = input_dict.get(grid_key, stream.get("grid_id"))
        if isinstance(grid, Tensor) and grid.ndim == 2:
            grid = grid[None]
        prepared = prepare_demo_context(native, {"latent": stream.get("noisy_latents"),
            "timesteps": times, "grid_id": grid, "frame_times": stream.get("frame_times")})
        context = prepared["latent"]
        count, device = context.hidden.shape[1], context.hidden.device
        # All demo tokens form one bidirectional context block; RoPE retains order.
        input_dict = {**input_dict,
            stream_key: {"noisy_latents": context, "timesteps": prepared["timesteps"],
                         "cache_type_ids": torch.full((count,), 2, device=device, dtype=torch.int)},
            grid_key: context.grid_id[0],
            "current_seq_ids": torch.zeros(count, device=device, dtype=torch.int),
            "current_frame_ids": torch.zeros(count, device=device, dtype=torch.int)}
    return native._demo_original_forward_stream(input_dict, mode, update_cache, cache_name, clean_window_cache)


def install_demo_interface(native, config):
    """Install only on this model instance; leave the pinned upstream source intact."""
    if config is None or config.get("enabled") is False:
        return native
    if hasattr(native, "demo_bottleneck"):
        raise ValueError("demonstration interface is already installed")
    # Previously cached raw demo K/V must not survive enabling the interface.
    cache_names = {"pos"} | {name for block in native.blocks for name in block.attn1.attn_caches}
    for name in cache_names:
        native.clear_cache(name)
    options = {key: value for key, value in config.items() if key != "enabled"}
    native.demo_bottleneck = TemporalDemoBottleneck(input_dim=native.inner_dim, **options)
    native.demo_bottleneck.to(native.patch_embedding_mlp.weight)
    native._demo_context_owner = object()
    for name, replacement in (("_training_embed", _training_embed), ("_embed_stream", _embed_stream),
                               ("forward_train", _forward_train), ("_forward_stream", _forward_stream)):
        setattr(native, "_demo_original" + (name if name.startswith("_") else "_" + name), getattr(native, name))
        setattr(native, name, MethodType(replacement, native))
    return native


@torch.inference_mode()
def cache_demo_context(native, latent, frame_times, null, icl_rope_h=None):
    """Cache a demo through the same compressor used for both training domains."""
    if not hasattr(native, "demo_bottleneck"):
        raise ValueError("cache_demo_context requires an installed demonstration bottleneck")
    namespace = getattr(native, "demo_icl_rope_h", None)
    if icl_rope_h is None:
        icl_rope_h = 24 if namespace is None else namespace
    if namespace is not None and icl_rope_h != namespace:
        raise ValueError("demonstration RoPE namespace differs from the trained model")
    from wan_va.utils import get_mesh_id

    weight = native.patch_embedding_mlp.weight
    latent = latent.to(weight)
    pf, ph, pw = native.patch_size
    frames, height, width = latent.shape[-3:]
    grid = get_mesh_id(frames // pf, height // ph, width // pw, 0, h_shift=icl_rope_h).to(weight.device)
    native({"latent_res_lst": {"noisy_latents": latent,
            "timesteps": torch.zeros(frames, device=weight.device),
            "frame_times": frame_times.to(weight.device),
            "cache_type_ids": torch.full((grid.shape[1],), 2, device=weight.device, dtype=torch.int)},
        "latent_grid_id": grid, "text_emb": null.to(weight),
        "encoder_seq_ids": torch.zeros(null.shape[1], device=weight.device, dtype=torch.int)},
        update_cache=1, mode="forward_latent_only")
    return native.cache_counts()
