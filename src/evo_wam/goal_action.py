"""Run the native Action Expert using only goal-interface and robot-state tokens."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .zerowam import VideoLoRA, action_mask_for, unpack_velocity


_ACTION_TARGETS = ("attn1.action_to_q", "attn1.action_to_k", "attn1.action_to_v",
                   "attn1.action_to_out.0", "attn2.action_to_q", "attn2.action_to_out.0")


def install_action_lora(native, rank=8, alpha=8):
    """Install after video LoRA; never adapt the cross-attention's shared K/V."""
    projections = []
    for block in native.blocks:
        if (block.attn1.action_to_k is block.attn1.to_k
                or block.attn1.action_to_v is block.attn1.to_v):
            raise ValueError("action adapters require separate action self-attention K/V")
        for path in _ACTION_TARGETS:
            parent_path, name = path.rsplit(".", 1)
            parent = block.get_submodule(parent_path)
            base = getattr(parent, name)
            if not isinstance(base, nn.Linear):
                raise ValueError("action adapters require unwrapped native Linear projections")
            projections.append((parent, name, VideoLoRA(base, rank, alpha)))
    for parent, name, adapter in projections:
        setattr(parent, name, adapter)
        adapter.enable(True)
    return native


def set_action_lora(native, enabled):
    """Change only action LoRA trainability; video/interface stages remain caller-owned."""
    for block in native.blocks:
        for path in _ACTION_TARGETS:
            adapter = block.get_submodule(path)
            if isinstance(adapter, VideoLoRA):
                adapter.enable(enabled)
    return native


@torch.no_grad()
def merge_action_lora(native):
    """Restore checkpoint-compatible native action Linear layers in place."""
    for block in native.blocks:
        for path in _ACTION_TARGETS:
            parent_path, name = path.rsplit(".", 1)
            parent = block.get_submodule(parent_path)
            adapter = getattr(parent, name)
            if isinstance(adapter, VideoLoRA):
                update = adapter.up.weight.float() @ adapter.down.weight.float()
                adapter.base.weight.copy_((adapter.base.weight.float() + update * adapter.scale)
                                          .to(adapter.base.weight))
                setattr(parent, name, adapter.base)
    return native


def goal_action_forward(native, noisy_actions, timesteps, condition, *, cache_name="se3_action"):
    """Denoise one action block with zero video tokens and zero video-cache access.

    ``condition`` is the learned z plus current-state token, already projected
    to native.inner_dim. The same function serves training and inference.
    Calls on the shared native model are sequential, as in upstream ICL.
    """
    if (not isinstance(noisy_actions, torch.Tensor) or noisy_actions.ndim != 5
            or noisy_actions.shape[0] != 1 or noisy_actions.shape[1] != native.config.action_dim
            or noisy_actions.shape[-1] != 1 or min(noisy_actions.shape) < 1
            or not noisy_actions.is_floating_point() or not torch.isfinite(noisy_actions).all()):
        raise ValueError("noisy_actions must be finite native [1,A,F,N,1] floating point")
    if (not isinstance(condition, torch.Tensor) or condition.ndim != 3
            or condition.shape[0] != 1 or condition.shape[1] < 2
            or condition.shape[-1] != native.inner_dim or not condition.is_floating_point()
            or not torch.isfinite(condition).all()):
        raise ValueError("condition must be finite [1,K+1,native.inner_dim] goal/state tokens")
    if not isinstance(cache_name, str) or not cache_name.strip() or cache_name == "pos":
        raise ValueError("action cache_name must be nonempty and separate from the native pos cache")
    frames = noisy_actions.shape[2]
    time = torch.as_tensor(timesteps)
    if (time.shape not in {(frames,), (1, frames)} or time.is_complex()
            or not torch.isfinite(time).all() or (time < 0).any() or (time > 1000).any()):
        raise ValueError("timesteps must contain one finite value in [0,1000] per action frame")

    from wan_va.modules.icl_model import ICLAttentionBackend
    from wan_va.utils import get_mesh_id

    weight = native.action_embedder.weight
    actions, condition = noisy_actions.to(weight), condition.to(weight)
    hidden, temb, projection = native._embed_stream(
        {"noisy_latents": actions, "timesteps": time.to(device=weight.device, dtype=torch.float32)},
        "action")
    count = hidden.shape[1]
    padding = 128 - count % 128
    pad = hidden.new_zeros(1, padding, native.inner_dim)
    grid = get_mesh_id(*actions.shape[-3:], 1, action=False).to(weight.device)
    rotary = F.pad(native.rope(grid[None])[:, :, None], (0, 0, 0, 0, 0, padding))
    seq = F.pad(torch.zeros(count, device=weight.device, dtype=torch.int), (0, padding), value=-1)
    types = F.pad(torch.ones(count, device=weight.device, dtype=torch.int), (0, padding), value=-1)
    frame_ids = seq.clone()  # All queries belong to the same denoised action block.
    encoder_seq = torch.zeros(condition.shape[1], device=weight.device, dtype=torch.int)
    previous = [(block.attn1.self_block_mask, block.attn2.cross_block_mask) for block in native.blocks]
    backend_masks = ICLAttentionBackend.self_mask, ICLAttentionBackend.cross_mask
    try:
        self_mask = ICLAttentionBackend.build_self_mask(
            types, types, seq, seq, frame_ids, frame_ids, -1, weight.device, compile_mask=False)
        cross_mask = ICLAttentionBackend.build_cross_mask(seq, encoder_seq, weight.device, compile_mask=False)
        for block in native.blocks:
            # Never native.clear_cache(): even a named clear erases shared video metadata.
            block.attn1.clear_cache(cache_name)
            block.attn1.self_block_mask = self_mask
            block.attn2.cross_block_mask = cross_mask
            _, hidden = block(None, hidden, pad, condition, None, projection, rotary,
                              update_cache=0, cache_name=cache_name)
        shift, scale = (native.scale_shift_table_action[None] + temb[:, :, None]).unbind(2)
        hidden = (native.action_norm_out(hidden.float()) * (1 + scale) + shift).to(hidden.dtype)
        return unpack_velocity(native.action_proj_out(hidden), actions.shape)
    finally:
        for block, (self_mask, cross_mask) in zip(native.blocks, previous):
            block.attn1.clear_cache(cache_name)
            block.attn1.self_block_mask = self_mask
            block.attn2.cross_block_mask = cross_mask
        ICLAttentionBackend.self_mask, ICLAttentionBackend.cross_mask = backend_masks


@torch.no_grad()
def goal_action_sample(native, condition, shape, mask, generator, steps=4, shift=1.0):
    """Generate a fresh action block, masking inactive channels at every flow step."""
    if (not isinstance(shape, (tuple, list, torch.Size)) or len(shape) != 5
            or any(type(size) is not int or size < 1 for size in shape)
            or shape[0] != 1 or shape[1] != native.config.action_dim or shape[-1] != 1):
        raise ValueError("shape must be a positive native [1,A,F,N,1] shape")
    if type(steps) is not int or steps < 1 or not math.isfinite(shift) or shift <= 0:
        raise ValueError("steps and shift must be positive and finite")
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError("sampling requires a CPU torch.Generator for reproducible fresh noise")
    from wan_va.utils import FlowMatchScheduler

    sample = torch.randn(tuple(shape), generator=generator, device="cpu").to(native.action_embedder.weight)
    active = action_mask_for(mask, sample)
    if not active.any():
        raise ValueError("at least one action channel must be active")
    sample = torch.where(active, sample, 0)
    scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(steps)
    for timestep in scheduler.timesteps:
        velocity = goal_action_forward(native, sample, timestep.expand(shape[2]), condition)
        sample = torch.where(active, scheduler.step(velocity, timestep, sample), 0)
    if not torch.isfinite(sample).all():
        raise ValueError("goal-conditioned action sampler produced nonfinite commands")
    return sample
