"""Run the native Action Expert using only goal-interface and robot-state tokens."""

from copy import deepcopy
import math
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from .zerowam import action_mask_for, unpack_velocity


def _action_cross_project(self, hs_latent, hs_action, hs_pad, encoder_hidden_states):
    """Use independent action K/V; video queries keep the native projection path."""
    if hs_action is None:
        return type(self)._project(self, hs_latent, hs_action, hs_pad, encoder_hidden_states)
    if hs_latent is not None or encoder_hidden_states is None:
        raise ValueError("the goal action interface requires isolated action cross-attention")
    query = self.action_norm_q(self.action_to_q(hs_action)).unflatten(2, (self.heads, -1))
    padding = query.new_zeros(query.shape[0], hs_pad.shape[1], self.heads, query.shape[-1])
    query = torch.cat((query, padding), dim=1)
    key = self.action_norm_k(self.action_to_k(encoder_hidden_states)).unflatten(2, (self.heads, -1))
    value = self.action_to_v(encoder_hidden_states).unflatten(2, (self.heads, -1))
    return query, key, value, 0, hs_action.shape[1]


def install_action_interface(native):
    """Split native cross-attention aliases and route their actual action calls.

    Load pretrained native weights before installation; restore specialized full
    checkpoints after installation. The names in state_dict stay native-compatible,
    but their action K/V values can now differ from the video's values.
    """
    if getattr(native, "_goal_action_interface", False):
        raise ValueError("goal action interface is already installed")
    for block in native.blocks:
        for name in ("to_q", "to_k", "to_v", "action_to_q", "action_to_k", "action_to_v"):
            if not isinstance(getattr(block.attn1, name), nn.Linear) or not isinstance(getattr(block.attn2, name), nn.Linear):
                raise ValueError("goal action interface requires native projections without LoRA")
        if (block.attn2.action_to_k is not block.attn2.to_k
                or block.attn2.action_to_v is not block.attn2.to_v
                or block.attn2.action_norm_k is not block.attn2.norm_k):
            raise ValueError("expected native shared cross-attention aliases before installation")
    for block in native.blocks:
        attention = block.attn2
        attention.action_to_k = deepcopy(attention.to_k)
        attention.action_to_v = deepcopy(attention.to_v)
        attention.action_norm_k = deepcopy(attention.norm_k)
        attention._project = MethodType(_action_cross_project, attention)
    native._goal_action_interface = True
    return native


def action_named_parameters(native):
    """Full executed action branch, including its independent language projection.

    The caller embeds nonvisual language with condition_embedder_action.text_embedder.
    MCP action branches are unused by this route and deliberately excluded.
    """
    if not getattr(native, "_goal_action_interface", False):
        raise ValueError("install the goal action interface before selecting action parameters")
    prefixes = ("action_embedder.", "condition_embedder_action.", "action_norm_out.", "action_proj_out.")
    for name, parameter in native.named_parameters():
        if (name.startswith(prefixes) or name == "scale_shift_table_action"
                or (name.startswith("blocks.") and any(
                    part.startswith("action_") or part == "scale_shift_table_action"
                    for part in name.split(".")[2:]))):
            yield name, parameter


def goal_action_forward(native, noisy_actions, timesteps, condition, *, cache_name="se3_action"):
    """Denoise one action block with zero video tokens and zero video-cache access.

    ``condition`` contains one [1,K_layer,native.inner_dim] tensor per native
    layer, combining that layer's latent with independent language/state tokens.
    Training and inference use the same route, without raw visual K/V access.
    Calls on the shared native model are sequential, as in upstream ICL.
    """
    if (not isinstance(noisy_actions, torch.Tensor) or noisy_actions.ndim != 5
            or noisy_actions.shape[0] != 1 or noisy_actions.shape[1] != native.config.action_dim
            or noisy_actions.shape[-1] != 1 or min(noisy_actions.shape) < 1
            or not noisy_actions.is_floating_point() or not torch.isfinite(noisy_actions).all()):
        raise ValueError("noisy_actions must be finite native [1,A,F,N,1] floating point")
    if (not isinstance(condition, (tuple, list)) or len(condition) != len(native.blocks)
            or any(not isinstance(value, torch.Tensor) or value.ndim != 3
                   or value.shape[0] != 1 or value.shape[1] < 1
                   or value.shape[-1] != native.inner_dim or not value.is_floating_point()
                   or not torch.isfinite(value).all() for value in condition)):
        raise ValueError("condition must contain one finite [1,K_layer,native.inner_dim] tensor per layer")
    if not isinstance(cache_name, str) or not cache_name.strip() or cache_name == "pos":
        raise ValueError("action cache_name must be nonempty and separate from the native pos cache")
    frames = noisy_actions.shape[2]
    time = torch.as_tensor(timesteps)
    if (time.shape not in {(frames,), (1, frames)} or time.is_complex()
            or not torch.isfinite(time).all() or (time < 0).any() or (time > 1000).any()):
        raise ValueError("timesteps must contain one finite value in [0,1000] per action frame")

    from wan_va.modules.icl_model import ICLAttentionBackend
    from wan_va.utils import get_mesh_id

    if not getattr(native, "_goal_action_interface", False):
        raise ValueError("install the goal action interface before action decoding")
    weight = native.action_embedder.weight
    actions = noisy_actions.to(weight)
    condition = [value.to(weight) for value in condition]
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
    previous = [(block.attn1.self_block_mask, block.attn2.cross_block_mask) for block in native.blocks]
    backend_masks = ICLAttentionBackend.self_mask, ICLAttentionBackend.cross_mask
    try:
        self_mask = ICLAttentionBackend.build_self_mask(
            types, types, seq, seq, frame_ids, frame_ids, -1, weight.device, compile_mask=False)
        for block, layer_condition in zip(native.blocks, condition):
            encoder_seq = torch.zeros(layer_condition.shape[1], device=weight.device, dtype=torch.int)
            cross_mask = ICLAttentionBackend.build_cross_mask(seq, encoder_seq, weight.device, compile_mask=False)
            # Never native.clear_cache(): even a named clear erases shared video metadata.
            block.attn1.clear_cache(cache_name)
            block.attn1.self_block_mask = self_mask
            block.attn2.cross_block_mask = cross_mask
            _, hidden = block(None, hidden, pad, layer_condition, None, projection, rotary,
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
