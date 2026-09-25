"""Frozen clean-video features and visual-goal rules over one shared base."""

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


def assert_frozen_base(native):
    if any(parameter.requires_grad for _, parameter in frozen_base_named_parameters(native)):
        raise ValueError("G/pi video base must have every non-action parameter frozen")


def _action_ids(native):
    if not getattr(native, "_goal_action_interface", False):
        return set()
    from .goal_action import action_named_parameters

    # Installed action K/V must remain independent of the frozen video branch.
    for block in native.blocks:
        for attention in (block.attn1, block.attn2):
            for name in ("to_q", "to_k", "to_v", "to_out", "norm_q", "norm_k"):
                video_ids = {id(value) for value in getattr(attention, name).parameters()}
                if any(id(value) in video_ids for value in getattr(attention, f"action_{name}").parameters()):
                    raise ValueError("shared G/pi base requires independent action attention parameters")
    return {id(parameter) for _, parameter in action_named_parameters(native)}


def frozen_base_named_parameters(native):
    """Reuse the installed action selector; aliases are filtered by identity."""
    action_ids = _action_ids(native)
    for name, parameter in native.named_parameters():
        if id(parameter) not in action_ids:
            yield name, parameter


def _tensor_hash(value):
    value = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str((tuple(value.shape), value.dtype)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _json_identity(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError) as exc:
        raise ValueError("identity must contain finite JSON serializable values") from exc


def frozen_base_checksum(native):
    """Hash the frozen base once per tensor, excluding isolated action experts."""
    digest = hashlib.sha256()
    for name, value in list(frozen_base_named_parameters(native)) + list(native.named_buffers()):
        digest.update(name.encode())
        digest.update(_tensor_hash(value).encode())
    return digest.hexdigest()


def install_empty_text(native, embedding, identity):
    """Attach the pretrained empty-prompt embedding and its source provenance."""
    if (not isinstance(embedding, torch.Tensor) or embedding.ndim != 3
            or embedding.shape[0] != 1 or embedding.shape[1] < 1
            or embedding.shape[2] != native.config.text_dim or not embedding.is_floating_point()
            or not torch.isfinite(embedding).all()):
        raise ValueError("empty text embedding must be finite [1,L,text_dim]")
    if (not isinstance(identity, (str, dict)) or not identity
            or (isinstance(identity, str) and not identity.strip())):
        raise ValueError("empty text embedding requires a nonempty source identity")
    identity = _json_identity(identity)
    value = embedding.detach().clone().to(native.patch_embedding_mlp.weight)
    native.register_buffer("g_pi_empty_text", value, persistent=True)
    native.g_pi_empty_text_identity = {"source": identity, "sha256": _tensor_hash(value)}
    return native


def save_target_cache(path, z, encoder_identity):
    """Write portable float32 target tokens with their complete E identity."""
    if not isinstance(encoder_identity, dict):
        raise ValueError("target cache requires E identity metadata")
    encoder_identity = _json_identity(encoder_identity)
    if (not isinstance(z, torch.Tensor) or z.ndim != 3 or z.shape[0] < 1
            or tuple(z.shape[1:]) != (encoder_identity.get("k_z"), encoder_identity.get("d_z"))
            or not z.is_floating_point() or not torch.isfinite(z).all()):
        raise ValueError("target z must be finite [B,K_z,d_z] matching E identity")
    with open(path, "wb") as stream:
        np.savez(stream, z=z.detach().float().cpu().numpy(),
                 encoder_identity=np.asarray(json.dumps(encoder_identity, sort_keys=True)))


def load_target_cache(path, expected_identity):
    if not isinstance(expected_identity, dict):
        raise ValueError("target cache requires an expected E identity dictionary")
    expected_identity = _json_identity(expected_identity)
    with np.load(path, allow_pickle=False) as data:
        if "encoder_identity" not in data or "z" not in data:
            raise ValueError("target cache must contain z and E identity")
        identity = json.loads(str(data["encoder_identity"].item()))
        if not isinstance(identity, dict) or identity != expected_identity:
            raise ValueError("E identity mismatch in target cache")
        z = torch.from_numpy(np.array(data["z"], copy=True))
    if (z.ndim != 3 or z.shape[0] < 1 or tuple(z.shape[1:]) !=
            (identity.get("k_z"), identity.get("d_z")) or not z.is_floating_point()
            or not torch.isfinite(z).all()):
        raise ValueError("cached target z must be finite [B,K_z,d_z] matching E identity")
    return z


def truncate_robot_history(frames, current_index=None):
    """Physically remove future frames before any embedding or validation."""
    if not isinstance(frames, torch.Tensor) or frames.ndim != 5 or frames.shape[2] < 1:
        raise ValueError("robot history must be nonempty [1,C,T,H,W]")
    if current_index is None:
        return frames
    if type(current_index) is not int or not 0 <= current_index < frames.shape[2]:
        raise ValueError("current_index must select an available robot history frame")
    return frames[:, :, :current_index + 1].contiguous()


def _positive_int(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_camera_layout(layout):
    """Ordered view names and widths in native spatial patch tokens."""
    if not isinstance(layout, (list, tuple)) or not layout:
        raise ValueError("camera_layout must be a nonempty ordered list of named token widths")
    result, names = [], set()
    for view in layout:
        if (not isinstance(view, dict) or set(view) != {"name", "token_width"}
                or not isinstance(view["name"], str) or not view["name"].strip()
                or view["name"] in names):
            raise ValueError("camera_layout requires unique nonempty names and token_width only")
        _positive_int("camera_layout token_width", view["token_width"])
        result.append({"name": view["name"], "token_width": view["token_width"]})
        names.add(view["name"])
    return result


def _validate_camera_canvas(native, video, layout):
    layout = validate_camera_layout(layout)
    width = video.shape[4] // native.patch_size[2]
    if video.shape[4] % native.patch_size[2] or sum(view["token_width"] for view in layout) != width:
        raise ValueError("camera_layout token widths must exactly cover the native spatial patch width")


def g_attention_mask(demo_tokens, robot_frames, spatial_tokens, chunk_size, *, device="cpu"):
    """Dense reference for [demo, robot], useful without native CUDA kernels."""
    if type(demo_tokens) is not int or demo_tokens < 0:
        raise ValueError("demo_tokens must be a nonnegative integer")
    for name, value in (("robot_frames", robot_frames), ("spatial_tokens", spatial_tokens),
                        ("chunk_size", chunk_size)):
        _positive_int(name, value)
    count = demo_tokens + robot_frames * spatial_tokens
    query = torch.arange(count, device=device)[:, None]
    key = torch.arange(count, device=device)[None, :]
    return _visible_pair(query, key, demo_tokens, count, spatial_tokens * chunk_size)


def _visible_pair(query, key, demo_tokens, count, chunk_tokens):
    demo_pair = (query < demo_tokens) & (key < demo_tokens)
    robot_pair = (query >= demo_tokens) & ((key < demo_tokens) | (
        (key - demo_tokens) // chunk_tokens <= (query - demo_tokens) // chunk_tokens))
    return (query < count) & (key < count) & (demo_pair | robot_pair)


def _validate_config(native, config):
    for key in ("chunk_size", "max_frame_chunk_size", "icl_rope_h", "window_size"):
        _positive_int(key, config.get(key))
    if config["chunk_size"] > config["max_frame_chunk_size"]:
        raise ValueError("chunk_size exceeds max_frame_chunk_size")
    if native.attn_window != config["window_size"]:
        raise ValueError("window_size must match the native attention window")
    if native.patch_size[0] != 1:
        raise ValueError("G/pi context requires temporal patch size 1")
    if hasattr(native, "demo_bottleneck"):
        raise ValueError("G/pi context requires the complete native demonstration")


def _validate_video(native, value, name):
    _, ph, pw = native.patch_size
    if (not isinstance(value, torch.Tensor) or value.ndim != 5 or value.shape[0] != 1
            or value.shape[1] != native.config.in_channels or min(value.shape[-3:]) < 1
            or value.shape[3] % ph or value.shape[4] % pw or not value.is_floating_point()
            or not torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite [1,C,F,H,W] matching the native patch grid")


def _versions(native):
    return tuple((id(value), value._version, str(value.device), str(value.dtype))
                 for _, value in list(frozen_base_named_parameters(native)) + list(native.named_buffers()))


@dataclass(frozen=True)
class DemoCache:
    """Task-local demo features and actual rotary-transformed per-layer K/V."""

    native_id: int
    versions: tuple
    demo_hash: str
    height_shift: int
    tokens: int
    features: dict
    keys_values: tuple


@dataclass
class RobotPrefixCache:
    """Task-local closed-chunk K/V; G and pi never exchange this state."""

    owner: str
    identity: tuple | None = None
    observed_frames: int = 0
    history_hash: str | None = None
    closed_frames: int = 0
    features: dict = field(default_factory=dict)
    keys_values: tuple = ()
    demo_cache: DemoCache | None = None
    processed_tokens: int = 0
    last_processed_tokens: int = 0

    def __post_init__(self):
        if self.owner not in ("g", "pi"):
            raise ValueError("robot prefix cache owner must be g or pi")

    def clear(self):
        self.identity, self.history_hash = None, None
        self.observed_frames, self.closed_frames = 0, 0
        self.features, self.keys_values, self.demo_cache = {}, (), None
        self.processed_tokens, self.last_processed_tokens = 0, 0


def _prefix_identity(native, demonstration, history, config, owner):
    return (owner, id(native), _versions(native),
            json.dumps(_json_identity(config), sort_keys=True),
            json.dumps(_json_identity(getattr(native, "g_pi_empty_text_identity", None)), sort_keys=True),
            None if demonstration is None else _tensor_hash(demonstration),
            tuple(history.shape[:2]) + tuple(history.shape[3:]),
            str(history.dtype), str(history.device))


def _prefix_context(native, demonstration, history, config, cache, *, owner,
                    demo_cache=None, conditioned=True):
    """Only unseen/unfinished chunks enter native; validation precedes reuse."""
    if not isinstance(cache, RobotPrefixCache) or cache.owner != owner:
        raise ValueError("robot prefix cache owner mismatch; G and pi require separate caches")
    assert_frozen_base(native)
    _validate_config(native, config)
    _validate_video(native, history, "history")
    if "goal_encoder" in config:
        _validate_camera_canvas(native, history, config["goal_encoder"].get("camera_layout"))
    if demonstration is not None:
        _validate_video(native, demonstration, "demonstration")
    identity = _prefix_identity(native, demonstration, history, config, owner)
    if cache.identity is not None and cache.identity != identity:
        raise ValueError("robot prefix cache identity mismatch; clear it after changing base, demo, route, or config")
    if (cache.observed_frames > history.shape[2] or (cache.observed_frames and
            _tensor_hash(history[:, :, :cache.observed_frames]) != cache.history_hash)):
        raise ValueError("robot prefix cache history mismatch; clear it when resetting or changing a task prefix")
    actual_demo = demonstration if conditioned else None
    if demonstration is not None:
        if demo_cache is None:
            demo_cache = cache.demo_cache or build_demo_cache(native, demonstration, config)
        # Validate even when all queried robot frames are already cached.
        demo_context_features(native, demonstration, config, demo_cache=demo_cache)
    elif demo_cache is not None:
        raise ValueError("robot-only prefix context cannot read a demonstration cache")
    context_demo_cache = demo_cache if conditioned else None
    spatial = (history.shape[3] // native.patch_size[1]) * (history.shape[4] // native.patch_size[2])
    processed = (history.shape[2] - cache.closed_frames) * spatial
    if processed:
        features, keys_values = _context(native, actual_demo, history, config,
            demo_cache=context_demo_cache, robot_prefix=cache, cache_robot=True)
    else:
        features = {layer: (value if context_demo_cache is None else
                    torch.cat((context_demo_cache.features[layer], value), dim=1))
                    for layer, value in cache.features.items()}
        keys_values = cache.keys_values
    closed_frames = history.shape[2] // config["chunk_size"] * config["chunk_size"]
    demo_tokens = 0 if context_demo_cache is None else context_demo_cache.tokens
    if closed_frames > cache.closed_frames:
        closed_tokens = closed_frames * spatial
        cache.features = {layer: value[:, demo_tokens:demo_tokens + closed_tokens].clone()
                          for layer, value in features.items()}
        cache.keys_values = tuple((key[:, demo_tokens:demo_tokens + closed_tokens].clone(),
                                   value[:, demo_tokens:demo_tokens + closed_tokens].clone())
                                  for key, value in keys_values)
    cache.identity, cache.demo_cache = identity, demo_cache
    cache.closed_frames, cache.observed_frames = closed_frames, history.shape[2]
    cache.history_hash = _tensor_hash(history)
    cache.last_processed_tokens = processed
    cache.processed_tokens += processed
    return features


@torch.no_grad()
def _context(native, demonstration, history, config, *, demo_cache=None, cache_demo=False,
             robot_prefix=None, cache_robot=False):
    # Native FlexAttention produces bf16 values even with fp32 checkpoint weights.
    device_type = native.patch_embedding_mlp.weight.device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
        return _context_impl(native, demonstration, history, config,
                             demo_cache=demo_cache, cache_demo=cache_demo,
                             robot_prefix=robot_prefix, cache_robot=cache_robot)


def _context_impl(native, demonstration, history, config, *, demo_cache=None, cache_demo=False,
                  robot_prefix=None, cache_robot=False):
    assert_frozen_base(native)
    _validate_config(native, config)
    if not hasattr(native, "g_pi_empty_text") or not hasattr(native, "g_pi_empty_text_identity"):
        raise ValueError("G/pi context requires the pretrained empty text embedding and source identity")
    for name, value in (("demonstration", demonstration), ("history", history)):
        if value is not None:
            _validate_video(native, value, name)
    if history is not None and "goal_encoder" in config:
        _validate_camera_canvas(native, history, config["goal_encoder"].get("camera_layout"))
    if demonstration is None and history is None:
        raise ValueError("context requires demonstration or robot history")
    if history is not None and config["icl_rope_h"] < history.shape[3] // native.patch_size[1]:
        raise ValueError("icl_rope_h overlaps the robot history spatial grid")
    if demo_cache is not None:
        if (not isinstance(demo_cache, DemoCache) or demonstration is None
                or demo_cache.native_id != id(native) or demo_cache.versions != _versions(native)
                or demo_cache.height_shift != config["icl_rope_h"]
                or demo_cache.demo_hash != _tensor_hash(demonstration)):
            raise ValueError("demo cache does not match the frozen base, demonstration, or RoPE namespace")
        if history is None:
            raise ValueError("cached G context requires robot history")

    from torch.nn.attention.flex_attention import create_block_mask
    from wan_va.modules.icl_model import ICLAttentionBackend
    from wan_va.utils import get_mesh_id

    weight = native.patch_embedding_mlp.weight
    _, ph, pw = native.patch_size
    demo_tokens = (0 if demonstration is None else
                   demonstration.shape[2] * (demonstration.shape[3] // ph) * (demonstration.shape[4] // pw))
    spatial = 1 if history is None else (history.shape[3] // ph) * (history.shape[4] // pw)
    closed_frames = 0 if robot_prefix is None else robot_prefix.closed_frames
    offset = (demo_tokens if demo_cache is not None else 0) + closed_frames * spatial
    streams = []
    if demonstration is not None and demo_cache is None:
        streams.append((demonstration, config["icl_rope_h"], 0))
    if history is not None:
        streams.append((history[:, :, closed_frames:], 0, closed_frames))
    modes = [(module, module.training) for module in native.modules()]
    previous = [(block.attn1.self_block_mask, block.attn2.cross_block_mask,
                 block.attn1.attn_caches.get("g_pi_context")) for block in native.blocks]
    backend_masks = ICLAttentionBackend.self_mask, ICLAttentionBackend.cross_mask
    features, keys_values = {}, []
    try:
        native.eval()
        hidden_parts, projection_parts, grids = [], [], []
        for video, shift, frame_offset in streams:
            video = video.detach().to(weight)
            frames, height, width = video.shape[-3:]
            timestep = torch.zeros(1, frames, device=weight.device, dtype=torch.float32)
            hidden, _, projection = native._training_embed(video, timestep, "video")
            hidden_parts.append(hidden)
            projection_parts.append(projection)
            grid = get_mesh_id(frames, height // ph, width // pw, 0,
                               h_shift=shift).to(weight.device)
            grid[0] += frame_offset
            grids.append(grid)
        hidden = torch.cat(hidden_parts, dim=1)
        projection = torch.cat(projection_parts, dim=1)
        count = hidden.shape[1]
        padding = 128 - count % 128
        pad = hidden.new_zeros(1, padding, native.inner_dim)
        rotary = F.pad(native.rope(torch.cat(grids, dim=1)[None])[:, :, None],
                       (0, 0, 0, 0, 0, padding))
        # Use the pretrained empty prompt; task language/state never enter Wan.
        text = native.g_pi_empty_text.to(native.condition_embedder.text_embedder.linear_1.weight)
        text_hidden = native.condition_embedder.text_embedder(text)

        # Native dynamic FlexAttention captures tensors, not arithmetic on
        # symbolic closure scalars. Pad metadata for its partial KV blocks.
        query_positions = torch.arange(count + padding, device=weight.device) + offset
        key_length = offset + count + padding
        key_positions = torch.arange(((key_length + 127) // 128) * 128, device=weight.device)
        query_valid, key_valid = query_positions < count + offset, key_positions < count + offset
        query_demo, key_demo = query_positions < demo_tokens, key_positions < demo_tokens
        chunk_tokens = spatial * config["chunk_size"]
        query_chunks = (query_positions - demo_tokens) // chunk_tokens
        key_chunks = (key_positions - demo_tokens) // chunk_tokens
        text_valid = torch.arange(((text.shape[1] + 127) // 128) * 128,
                                  device=weight.device) < text.shape[1]

        def visible(batch, head, query, key):
            return query_valid[query] & key_valid[key] & (
                key_demo[key] | (~query_demo[query] & (key_chunks[key] <= query_chunks[query])))

        def valid_text(batch, head, query, key):
            return query_valid[query] & text_valid[key]

        self_mask = create_block_mask(visible, 1, 1, count + padding,
                                      offset + count + padding, device=weight.device)
        cross_mask = create_block_mask(valid_text, 1, 1, count + padding, text.shape[1], device=weight.device)
        ICLAttentionBackend.self_mask, ICLAttentionBackend.cross_mask = self_mask, cross_mask
        for index, block in enumerate(native.blocks):
            block.attn1.clear_cache("g_pi_context")
            prefix_parts = []
            if demo_cache is not None:
                prefix_parts.append(demo_cache.keys_values[index])
            if closed_frames:
                prefix_parts.append(robot_prefix.keys_values[index])
            if prefix_parts:
                key, value = (torch.cat([pair[axis] for pair in prefix_parts], dim=1)
                              for axis in range(2))
                block.attn1.attn_caches["g_pi_context"] = {"k": key, "v": value}
            block.attn1.self_block_mask = self_mask
            block.attn2.cross_block_mask = cross_mask
            hidden, _ = block(hidden, None, pad, text_hidden, projection, None, rotary,
                              update_cache=int(cache_demo or cache_robot), cache_name="g_pi_context")
            if (hidden.shape != (1, count, native.inner_dim) or not torch.isfinite(hidden).all()):
                raise ValueError("native block must return finite packed G/pi context features")
            feature_parts = []
            if demo_cache is not None:
                feature_parts.append(demo_cache.features[index])
            if closed_frames:
                feature_parts.append(robot_prefix.features[index])
            features[index] = torch.cat((*feature_parts, hidden), dim=1) if feature_parts else hidden
            if cache_demo or cache_robot:
                cached = block.attn1.attn_caches["g_pi_context"]
                keep = demo_tokens if cache_demo else offset + count
                keys_values.append((cached["k"][:, :keep].clone(), cached["v"][:, :keep].clone()))
        return features, tuple(keys_values)
    finally:
        for block, (self_mask, cross_mask, previous_cache) in zip(native.blocks, previous):
            block.attn1.self_block_mask, block.attn2.cross_block_mask = self_mask, cross_mask
            block.attn1.clear_cache("g_pi_context")
            if previous_cache is not None:
                block.attn1.attn_caches["g_pi_context"] = previous_cache
        ICLAttentionBackend.self_mask, ICLAttentionBackend.cross_mask = backend_masks
        for module, training in modes:
            module.training = training


def build_demo_cache(native, demonstration, config):
    features, keys_values = _context(native, demonstration, None, config, cache_demo=True)
    return DemoCache(id(native), _versions(native), _tensor_hash(demonstration),
                     config["icl_rope_h"], features[0].shape[1], features, keys_values)


def g_context_features(native, demonstration, history, config, *, demo_cache=None, current_index=None,
                       prefix_cache=None):
    """Read [demo, observed robot]; calls sharing one native must be sequential."""
    if demonstration is None:
        raise ValueError("G context requires a complete task demonstration")
    history = truncate_robot_history(history, current_index)
    if prefix_cache is not None:
        return _prefix_context(native, demonstration, history, config, prefix_cache,
                               owner="g", demo_cache=demo_cache)
    return _context(native, demonstration, history, config, demo_cache=demo_cache)[0]


def demo_context_features(native, demonstration, config, *, demo_cache=None):
    """Read a complete demonstration with no robot, state or task text inputs."""
    if demonstration is None:
        raise ValueError("demo-only context requires a complete task demonstration")
    if demo_cache is None:
        return _context(native, demonstration, None, config)[0]
    assert_frozen_base(native)
    _validate_config(native, config)
    _validate_video(native, demonstration, "demonstration")
    if (not isinstance(demo_cache, DemoCache) or demo_cache.native_id != id(native)
            or demo_cache.versions != _versions(native)
            or demo_cache.height_shift != config["icl_rope_h"]
            or demo_cache.demo_hash != _tensor_hash(demonstration)):
        raise ValueError("demo cache does not match the frozen base, demonstration, or RoPE namespace")
    return demo_cache.features


def split_g_context_features(native, demonstration, history, config, *, demo_cache=None,
                             current_index=None, prefix_cache=None):
    """Split frozen demo/robot memories; optionally isolate the entire u path."""
    if demonstration is None:
        raise ValueError("G context requires a complete task demonstration")
    route = config.get("demo_route", "one_way")
    if route not in ("one_way", "via_u_only"):
        raise ValueError("demo_route must be one_way or via_u_only")
    history = truncate_robot_history(history, current_index)
    if route == "via_u_only":
        if prefix_cache is not None:
            robot = _prefix_context(native, demonstration, history, config, prefix_cache,
                                    owner="g", demo_cache=demo_cache, conditioned=False)
            return prefix_cache.demo_cache.features, robot
        demo = demo_context_features(native, demonstration, config, demo_cache=demo_cache)
        return demo, pi_context_features(native, history, config)
    features = g_context_features(native, demonstration, history, config, demo_cache=demo_cache,
                                  prefix_cache=prefix_cache)
    _, ph, pw = native.patch_size
    count = demonstration.shape[2] * (demonstration.shape[3] // ph) * (demonstration.shape[4] // pw)
    return ({layer: value[:, :count] for layer, value in features.items()},
            {layer: value[:, count:] for layer, value in features.items()})


def pi_context_features(native, history, config, *, current_index=None, prefix_cache=None):
    """Read only observed robot frames using chunk-causal attention."""
    history = truncate_robot_history(history, current_index)
    if prefix_cache is not None:
        return _prefix_context(native, None, history, config, prefix_cache, owner="pi")
    return _context(native, None, history, config)[0]


class FrozenGoalEncoder(nn.Module):
    """E: fixed single-frame encoding rules over the shared frozen video base."""

    def __init__(self, native, *, layer, camera_layout, grid_size=(4, 4), base_id=None):
        super().__init__()
        if (not isinstance(grid_size, (tuple, list)) or len(grid_size) != 2
                or any(type(value) is not int or value < 1 for value in grid_size)):
            raise ValueError("E grid_size must contain two positive integers [height,width]")
        if type(layer) is not int or not 0 <= layer < len(native.blocks):
            raise ValueError("E layer must select an existing native block")
        if base_id is not None:
            if (not isinstance(base_id, (str, dict)) or not base_id
                    or (isinstance(base_id, str) and not base_id.strip())):
                raise ValueError("base_id must be a nonempty string or JSON identity dictionary")
            base_id = _json_identity(base_id)
        if not hasattr(native, "g_pi_empty_text") or not hasattr(native, "g_pi_empty_text_identity"):
            raise ValueError("E requires the pretrained empty text embedding and source identity")
        assert_frozen_base(native)
        # The owning G/pi system registers native once; E has no parameter tree.
        object.__setattr__(self, "native", native)
        self.grid_size = tuple(grid_size)
        self.camera_layout = validate_camera_layout(camera_layout)
        self.k_z = len(self.camera_layout) * grid_size[0] * grid_size[1]
        self.d_z, self.layer = native.inner_dim, layer
        checksum = frozen_base_checksum(self.native)
        empty_identity = _json_identity(native.g_pi_empty_text_identity)
        empty_identity["sha256"] = _tensor_hash(native.g_pi_empty_text)
        self._identity = {
            "layer": layer, "timestep": 0, "pooling": "adaptive_avg_pool2d_spatial",
            "grid_size": list(self.grid_size), "token_order": "camera_then_row_major",
            "camera_layout": deepcopy(self.camera_layout), "num_views": len(self.camera_layout),
            "k_z": self.k_z, "d_z": self.d_z, "normalization": "l2_last_dim",
            "text_conditioning": "pretrained_empty_prompt",
            "empty_text_identity": empty_identity,
            "precision": {"compute": "cuda_bfloat16_autocast",
                          "video_weights": str(native.patch_embedding_mlp.weight.dtype)},
            "base_id": deepcopy(base_id) or checksum,
            "base_sha256": checksum,
        }
        self.train(False)

    @property
    def identity(self):
        return deepcopy(self._identity)

    def validate_identity(self, identity):
        if _json_identity(identity) != self._identity:
            raise ValueError("E identity mismatch: layer, timestep, grid, pooling, normalization, empty prompt, or base weights differ")

    def train(self, mode=True):
        return super().train(False)

    @staticmethod
    def normalize(value):
        return F.normalize(value.float(), dim=-1)

    @torch.no_grad()
    def forward(self, frame):
        if not isinstance(frame, torch.Tensor) or frame.ndim != 5 or frame.shape[2] != 1:
            raise ValueError("E requires exactly one separately encoded target frame [1,C,1,H,W]")
        _validate_camera_canvas(self.native, frame, self.camera_layout)
        config = {"chunk_size": 1, "max_frame_chunk_size": 1,
                  "icl_rope_h": max(1, frame.shape[3] // self.native.patch_size[1]),
                  "window_size": self.native.attn_window}
        features = pi_context_features(self.native, frame, config)[self.layer]
        height, width = (frame.shape[3] // self.native.patch_size[1],
                         frame.shape[4] // self.native.patch_size[2])
        spatial = features.transpose(1, 2).float().reshape(1, self.d_z, height, width)
        pooled, start = [], 0
        for view in self.camera_layout:
            end = start + view["token_width"]
            pooled.append(F.adaptive_avg_pool2d(spatial[..., start:end], self.grid_size)
                          .flatten(2).transpose(1, 2))
            start = end
        pooled = torch.cat(pooled, dim=1)
        return self.normalize(pooled)
