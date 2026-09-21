"""Generated robot futures for the goal interface, with no teacher-future input."""

import math

import torch

from .zerowam import unpack_velocity


def generated_robot_features(native, demonstration, history, null, config, generator, *, demo_times=None):
    """Sample from A and observed history; read generated per-layer features.

    Sampling is detached. The final clean read and its replayed context retain
    autograd during training, while an enclosing no_grad remains supported at
    inference. This trains the contextual feature computation, not the sampler's
    trajectory. Returned features are [1,F,Npatch,len(feature_layers)*D].
    """
    if hasattr(native, "demo_bottleneck"):
        raise ValueError("the goal interface uses the raw native demonstration, without demo_bottleneck")
    if native.training:
        raise ValueError("generated robot futures require native.eval()")
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
    if (not isinstance(null, torch.Tensor) or null.ndim != 3 or null.shape[0] != 1
            or null.shape[1] < 1 or null.shape[2] != native.config.text_dim
            or not null.is_floating_point() or not torch.isfinite(null).all()):
        raise ValueError("null must be a finite native empty-text embedding [1,L,text_dim]")
    if demo_times is not None and (
            not isinstance(demo_times, torch.Tensor) or demo_times.shape != (demonstration.shape[2],)
            or not demo_times.is_floating_point() or not torch.isfinite(demo_times).all()
            or torch.any(demo_times[1:] <= demo_times[:-1])):
        raise ValueError("demo_times must be strictly increasing finite latent-frame times")

    from wan_va.utils import FlowMatchScheduler, get_mesh_id

    weight = native.patch_embedding_mlp.weight
    demonstration, history, null = (value.detach().to(weight) for value in (demonstration, history, null))
    future_shape = (1, history.shape[1], chunk, history.shape[3], history.shape[4])
    hooks, collected = [], {}

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
                "latent_grid_id": grid, "text_emb": null,
                "current_seq_ids": torch.zeros(count, device=weight.device, dtype=torch.int),
                "current_frame_ids": torch.full((count,), 0 if cache_type == 2 else 2 * (start // chunk),
                                                device=weight.device, dtype=torch.int),
                "encoder_seq_ids": torch.zeros(null.shape[1], device=weight.device, dtype=torch.int)}

    def cache_context():
        native(stream(demonstration, 0, 2, 0), update_cache=1, mode="forward_latent_only")
        # ponytail: replay stores O(history) KV; use a rolling episode cache if profiling requires it.
        for start in range(0, history.shape[2], chunk):
            native(stream(history[:, :, start:start + chunk], 0, 0, start),
                   update_cache=1, mode="forward_latent_only")

    try:
        clear_cache()
        with torch.no_grad():
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
        cache_context()
        for index in layers:
            def capture(module, args, output, index=index):
                collected[index] = output[0]
            hooks.append(native.blocks[index].register_forward_hook(capture))
        native(stream(generated, 0, 1, history.shape[2]), update_cache=0, mode="forward_latent_only")
        count = chunk * (history.shape[3] // ph) * (history.shape[4] // pw)
        if any(not isinstance(collected.get(i), torch.Tensor)
               or collected[i].shape != (1, count, native.inner_dim) for i in layers):
            raise ValueError("native blocks did not return the requested generated future features")
        features = torch.cat([collected[i] for i in layers], dim=-1).reshape(1, chunk, count // chunk, -1)
        if not torch.isfinite(features).all():
            raise ValueError("native generated future features are nonfinite")
        return generated, features
    finally:
        for hook in hooks:
            hook.remove()
        clear_cache()
