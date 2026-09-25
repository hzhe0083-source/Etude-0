"""Native Wan features of observed robot history and a complete demonstration."""

import torch
from torch.nn import functional as F


def observed_context_features(native, demonstration, history, language, config, *,
                              demo_times=None, on_layer=None):
    """Read clean observed context once, returning every native layer in order.

    Each feature tensor is ``[1, history_tokens + demo_tokens, inner_dim]``.
    History is stored first and the full supplied demonstration second; all real
    tokens attend to one another. This packing order is not a causal ordering.
    The streams have independent latent-frame time grids, and the demonstration
    occupies the ``icl_rope_h`` spatial namespace. Optional ``demo_times`` checks
    the supplied frame metadata; RoPE retains native latent-frame index units.

    Cached VAE/text inputs are detached, but native embedding, text projection,
    and Transformer computation retain autograd. No future, diffusion sampler,
    video output head, or persistent KV cache participates in this read. If
    ``on_layer`` is supplied, it receives ``(index, features)`` after every block
    and the function returns an empty dictionary instead of storing features.
    Calls sharing one native model must be sequential, as in upstream ICL.
    """
    if hasattr(native, "demo_bottleneck"):
        raise ValueError("observed context uses the complete native demonstration, without demo_bottleneck")
    for key in ("chunk_size", "max_frame_chunk_size", "icl_rope_h", "window_size"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["chunk_size"] > config["max_frame_chunk_size"]:
        raise ValueError("chunk_size exceeds max_frame_chunk_size")
    if native.attn_window != config["window_size"]:
        raise ValueError("window_size must match the native attention window")
    if "feature_layers" in config:
        layers = config["feature_layers"]
        if (not isinstance(layers, list) or any(type(index) is not int for index in layers)
                or layers != list(range(len(native.blocks)))):
            raise ValueError("feature_layers must contain every native block in depth order")
    if on_layer is not None and not callable(on_layer):
        raise ValueError("on_layer must be callable")
    pf, ph, pw = native.patch_size
    if pf != 1:
        raise ValueError("observed context requires temporal patch size 1")
    for name, video in (("demonstration", demonstration), ("history", history)):
        if (not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[0] != 1
                or not video.is_floating_point() or not torch.isfinite(video).all()
                or video.shape[1] != native.config.in_channels or min(video.shape[-3:]) < 1
                or video.shape[3] % ph or video.shape[4] % pw):
            raise ValueError(f"{name} must be finite [1,C,F,H,W] matching the native patch grid")
    if config["icl_rope_h"] < history.shape[3] // ph:
        raise ValueError("icl_rope_h overlaps the robot history spatial grid")
    if (not isinstance(language, torch.Tensor) or language.ndim != 3 or language.shape[0] != 1
            or language.shape[1] < 1 or language.shape[2] != native.config.text_dim
            or not language.is_floating_point() or not torch.isfinite(language).all()):
        raise ValueError("language must be finite pure-text embeddings [1,L,text_dim]")
    if demo_times is not None and (
            not isinstance(demo_times, torch.Tensor) or demo_times.shape != (demonstration.shape[2],)
            or not demo_times.is_floating_point() or not torch.isfinite(demo_times).all()
            or torch.any(demo_times[1:] <= demo_times[:-1])):
        raise ValueError("demo_times must be strictly increasing finite latent-frame times")

    from torch.nn.attention.flex_attention import create_block_mask
    from wan_va.modules.icl_model import ICLAttentionBackend
    from wan_va.utils import get_mesh_id

    weight = native.patch_embedding_mlp.weight
    history, demonstration, language = (
        value.detach().to(weight) for value in (history, demonstration, language))
    modules = list(native.modules())
    modes = [(module, module.training) for module in modules]
    attention = [module for module in modules if hasattr(module, "attn_caches")]
    masks = [(module, module.self_block_mask, module.cross_block_mask) for module in attention]
    backend_masks = ICLAttentionBackend.self_mask, ICLAttentionBackend.cross_mask

    def clear_cache():
        # Native clear resets shared cache metadata; clear every attention cache
        # as well, including unused MCP blocks and names from earlier routes.
        native.clear_cache()
        for module in attention:
            module.attn_caches.clear()

    collected = {}
    try:
        clear_cache()
        hidden_parts, projection_parts, grids = [], [], []
        for video, height_shift in ((history, 0), (demonstration, config["icl_rope_h"])):
            frames, height, width = video.shape[-3:]
            timestep = torch.zeros(1, frames, device=weight.device, dtype=torch.float32)
            hidden, _, projection = native._training_embed(video, timestep, "video")
            hidden_parts.append(hidden)
            projection_parts.append(projection)
            grids.append(get_mesh_id(frames, height // ph, width // pw, 0,
                                     h_shift=height_shift).to(weight.device))
        hidden = torch.cat(hidden_parts, dim=1)
        projection = torch.cat(projection_parts, dim=1)
        count = hidden.shape[1]
        # Match native block padding, including a full pad block at multiples.
        padding = 128 - count % 128
        pad = hidden.new_zeros(1, padding, native.inner_dim)
        rotary = F.pad(native.rope(torch.cat(grids, dim=1)[None])[:, :, None],
                       (0, 0, 0, 0, 0, padding))
        text = language.to(native.condition_embedder.text_embedder.linear_1.weight)
        text_hidden = native.condition_embedder.text_embedder(text)

        # Every input is observed. Do not reuse temporal-forcing/streaming masks:
        # they would hide later history or history from demonstration queries.
        def valid_observed_pair(batch, head, query, key):
            return (query < count) & (key < count)

        def valid_text_pair(batch, head, query, key):
            return (query < count) & (key < language.shape[1])

        self_mask = create_block_mask(valid_observed_pair, 1, 1, count + padding,
                                      count + padding, device=weight.device)
        cross_mask = create_block_mask(valid_text_pair, 1, 1, count + padding,
                                       language.shape[1], device=weight.device)
        ICLAttentionBackend.self_mask, ICLAttentionBackend.cross_mask = self_mask, cross_mask
        for index, block in enumerate(native.blocks):
            block.attn1.self_block_mask = self_mask
            block.attn2.cross_block_mask = cross_mask
            hidden, _ = block(hidden, None, pad, text_hidden, projection, None, rotary,
                              update_cache=0, cache_name="observed_context")
            if (not isinstance(hidden, torch.Tensor) or hidden.shape != (1, count, native.inner_dim)
                    or not torch.isfinite(hidden).all()):
                raise ValueError("native block must return finite packed observed-context features")
            if on_layer is None:
                collected[index] = hidden
            else:
                on_layer(index, hidden)
        return collected
    finally:
        clear_cache()
        for module, self_mask, cross_mask in masks:
            module.self_block_mask, module.cross_block_mask = self_mask, cross_mask
        ICLAttentionBackend.self_mask, ICLAttentionBackend.cross_mask = backend_masks
        for module, training in modes:
            module.training = training
