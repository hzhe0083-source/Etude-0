"""Generated robot futures for the goal interface, with no teacher-future input."""

import math

import torch

from .zerowam import unpack_velocity


def generated_robot_features(native, demonstration, history, language, config, generator, *,
                             demo_times=None, on_layer=None):
    """Sample from A and observed history; read generated per-layer features.

    Sampling is detached. The final clean read and its replayed context retain
    autograd during training, while an enclosing no_grad remains supported at
    inference. This trains the contextual feature computation, not the sampler's
    trajectory. Each requested layer returns [1,F,Npatch,D], or calls
    on_layer(index, features) in native layer order instead of storing features.
    """
    if hasattr(native, "demo_bottleneck"):
        raise ValueError("the goal interface uses the raw native demonstration, without demo_bottleneck")
    for key in ("chunk_size", "max_frame_chunk_size", "icl_rope_h", "window_size", "sampling_steps"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    chunk = config["chunk_size"]
    if chunk > config["max_frame_chunk_size"]:
        raise ValueError("chunk_size exceeds max_frame_chunk_size")
    if native.attn_window != config["window_size"]:
        raise ValueError("window_size must match the native attention window")
    shift = config.get("video_snr_shift")
    if type(shift) not in (int, float) or not math.isfinite(shift) or shift <= 0:
        raise ValueError("video_snr_shift must be positive and finite")
    layers = config.get("feature_layers")
    if (not isinstance(layers, list) or not layers
            or any(type(i) is not int or not 0 <= i < len(native.blocks) for i in layers)
            or len(set(layers)) != len(layers)):
        raise ValueError("feature_layers must be unique native block indices")
    if on_layer is not None and not callable(on_layer):
        raise ValueError("on_layer must be callable")
    if not isinstance(generator, torch.Generator):
        raise ValueError("a torch.Generator is required for reproducible future sampling")
    pf, ph, pw = native.patch_size
    if pf != 1:
        raise ValueError("generated robot futures require temporal patch size 1")
    for name, video in (("demonstration", demonstration), ("history", history)):
        if (not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[0] != 1
                or not video.is_floating_point() or not torch.isfinite(video).all()
                or video.shape[1] != native.config.in_channels or min(video.shape[-3:]) < 1
                or video.shape[3] % ph or video.shape[4] % pw):
            raise ValueError(f"{name} must be finite [1,C,F,H,W] matching the native patch grid")
    if history.shape[1] != native.config.out_channels or history.shape[2] % chunk:
        raise ValueError("history must match output channels and end at a native chunk boundary")
    if config["icl_rope_h"] < history.shape[3] // ph:
        raise ValueError("icl_rope_h overlaps the target spatial grid")
    if (not isinstance(language, torch.Tensor) or language.ndim != 3 or language.shape[0] != 1
            or language.shape[1] < 1 or language.shape[2] != native.config.text_dim
            or not language.is_floating_point() or not torch.isfinite(language).all()):
        raise ValueError("language must be finite pure-text embeddings [1,L,text_dim]")
    if demo_times is not None and (
            not isinstance(demo_times, torch.Tensor) or demo_times.shape != (demonstration.shape[2],)
            or not demo_times.is_floating_point() or not torch.isfinite(demo_times).all()
            or torch.any(demo_times[1:] <= demo_times[:-1])):
        raise ValueError("demo_times must be strictly increasing finite latent-frame times")

    from wan_va.utils import FlowMatchScheduler, get_mesh_id

    weight = native.patch_embedding_mlp.weight
    demonstration, history, language = (
        value.detach().to(weight) for value in (demonstration, history, language))
    future_shape = (1, history.shape[1], chunk, history.shape[3], history.shape[4])
    hooks, collected = [], {}
    seen_layers = set()
    modes = [(module, module.training) for module in native.modules()]
    count = chunk * (history.shape[3] // ph) * (history.shape[4] // pw)

    def restore_modes():
        for module, training in modes:
            module.training = training

    def clear_cache():
        # Native metadata is global even though individual KV caches have names.
        names = {"pos"} | {name for block in native.blocks for name in block.attn1.attn_caches}
        for name in names:
            native.clear_cache(name)

    def stream(video, timestep, cache_type, start):
        frames, height, width = video.shape[-3:]
        grid = get_mesh_id(frames, height // ph, width // pw, 0,
                           f_shift=0 if cache_type == 2 else start,
                           h_shift=config["icl_rope_h"] if cache_type == 2 else 0).to(weight.device)
        count = grid.shape[1]
        return {"latent_res_lst": {"noisy_latents": video,
                    "timesteps": torch.full((frames,), float(timestep), device=weight.device),
                    "cache_type_ids": torch.full((count,), cache_type, device=weight.device, dtype=torch.int)},
                "latent_grid_id": grid, "text_emb": language,
                "current_seq_ids": torch.zeros(count, device=weight.device, dtype=torch.int),
                "current_frame_ids": torch.full((count,), 0 if cache_type == 2 else 2 * (start // chunk),
                                                device=weight.device, dtype=torch.int),
                "encoder_seq_ids": torch.zeros(language.shape[1], device=weight.device, dtype=torch.int)}

    def cache_context():
        native(stream(demonstration, 0, 2, 0), update_cache=1, mode="forward_latent_only")
        # ponytail: replay stores O(history) KV; use a rolling episode cache if profiling requires it.
        for start in range(0, history.shape[2], chunk):
            native(stream(history[:, :, start:start + chunk], 0, 0, start),
                   update_cache=1, mode="forward_latent_only")

    try:
        clear_cache()
        native.eval()
        # Detached casts cached by an outer autocast scope would otherwise be
        # reused by the differentiable replay, silently removing its gradients.
        with torch.no_grad(), torch.autocast(
                weight.device.type, enabled=torch.is_autocast_enabled(weight.device.type),
                dtype=torch.get_autocast_dtype(weight.device.type), cache_enabled=False):
            cache_context()
            generated = torch.randn(future_shape, generator=generator, device=generator.device).to(weight)
            scheduler = FlowMatchScheduler(shift=shift, sigma_min=0., extra_one_step=True)
            scheduler.set_timesteps(config["sampling_steps"])
            for timestep in scheduler.timesteps:
                velocity = native(stream(generated, timestep, 1, history.shape[2]),
                                  update_cache=0, mode="forward_latent_only")
                generated = scheduler.step(unpack_velocity(velocity, future_shape, native.patch_size),
                                           timestep, generated)
            generated = generated.detach()
            if not torch.isfinite(generated).all():
                raise ValueError("native sampler produced a nonfinite robot future")
        clear_cache()
        restore_modes()
        cache_context()
        for index in layers:
            def capture(module, args, output, index=index):
                features = output[0]
                if not isinstance(features, torch.Tensor) or features.shape != (1, count, native.inner_dim):
                    raise ValueError("native block did not return the requested generated future features")
                features = features.reshape(1, chunk, count // chunk, native.inner_dim)
                if not torch.isfinite(features).all():
                    raise ValueError("native generated future features are nonfinite")
                seen_layers.add(index)
                if on_layer is None:
                    collected[index] = features
                else:
                    on_layer(index, features)
            hooks.append(native.blocks[index].register_forward_hook(capture))
        native(stream(generated, 0, 1, history.shape[2]), update_cache=0, mode="forward_latent_only")
        if seen_layers != set(layers):
            raise ValueError("native blocks did not return the requested generated future features")
        return generated, collected if on_layer is None else None
    finally:
        for hook in hooks:
            hook.remove()
        restore_modes()
        clear_cache()
