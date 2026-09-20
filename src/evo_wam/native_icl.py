"""Human video training through the unchanged native Zero-WAM ICL blocks."""

import math

import torch
from torch.nn import functional as F

from .zerowam import VideoLoRA


_LORA_TARGETS = ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                 "attn2.to_q", "attn2.to_out.0")


def install_icl_lora(native, rank=8, alpha=8):
    """Adapt native video/context attention; preserve action and text K/V weights."""
    if any(isinstance(module, VideoLoRA) for module in native.modules()):
        raise ValueError("ICL adapters must be installed on an unwrapped native model")
    projections = []
    for block in native.blocks:
        if (block.attn1.to_k is block.attn1.action_to_k
                or block.attn1.to_v is block.attn1.action_to_v):
            raise ValueError("ICL adapters require separate video/action self-attention K/V")
        for path in _LORA_TARGETS:
            parent_path, name = path.rsplit(".", 1)
            parent = block.get_submodule(parent_path)
            projections.append((parent, name, VideoLoRA(getattr(parent, name), rank, alpha)))
    native.requires_grad_(False)
    for parent, name, adapter in projections:
        setattr(parent, name, adapter)
        adapter.enable(True)
    # eval disables dropout, not autograd; robot and human updates share these weights.
    native.eval()
    return native


@torch.no_grad()
def merge_icl_lora(native):
    """Merge in place into checkpoint-compatible Linear layers; perform no I/O."""
    for block in native.blocks:
        for path in _LORA_TARGETS:
            parent_path, name = path.rsplit(".", 1)
            parent = block.get_submodule(parent_path)
            adapter = getattr(parent, name)
            if isinstance(adapter, VideoLoRA):
                update = adapter.up.weight.float() @ adapter.down.weight.float()
                adapter.base.weight.copy_((adapter.base.weight.float() + update * adapter.scale)
                                          .to(adapter.base.weight))
                setattr(parent, name, adapter.base)
    native.clear_cache()
    return native


def forward_video_only(native, input_dict):
    """Native temporal forcing with absent actions, returning (video, MCP list).

    B's noisy queries see only earlier clean B chunks plus the complete demo A.
    This function reuses upstream attention masks and never creates action tokens.
    """
    if "action_dict" in input_dict:
        raise ValueError("Human video-only inputs must omit action_dict entirely")
    latent = input_dict["latent_dict"]
    if latent["noisy_latents"].shape[0] != 1:
        raise ValueError("Native ICL training requires batch_size=1")
    chunk_size = int(input_dict["chunk_size"])
    max_chunk_size = int(input_dict["max_frame_chunk_size"])
    if chunk_size <= 0 or max_chunk_size <= 0:
        raise ValueError("Chunk sizes must be positive")
    mcp_streams = input_dict.get("mcp_latent_dicts", [])
    if mcp_streams and (not native.enable_mcp
                        or len(mcp_streams) != len(native.mcp_blocks)):
        raise ValueError("MCP inputs must match the enabled native MCP groups")

    noisy_hs, target_temb, noisy_proj = native._training_embed(
        latent["noisy_latents"], latent["timesteps"], "video")
    clean_hs, _, clean_proj = native._training_embed(
        latent["latent"], latent["cond_timesteps"], "video")
    target_length = noisy_hs.shape[1]
    target_grid = latent["grid_id"][0]
    hidden_parts, projection_parts = [noisy_hs, clean_hs], [noisy_proj, clean_proj]
    grid_parts = [target_grid, target_grid]
    icl = input_dict.get("icl_latent_dict")
    if icl is not None:
        from .demo_context import prepare_demo_context
        icl = prepare_demo_context(native, icl)
    icl_length = 0
    if icl is not None:
        icl_hs, _, icl_proj = native._training_embed(
            icl["latent"], icl["timesteps"], "video")
        icl_length = icl_hs.shape[1]
        hidden_parts.append(icl_hs)
        projection_parts.append(icl_proj)
        grid_parts.append(icl["grid_id"][0])
    hidden = torch.cat(hidden_parts, dim=1)
    timestep_proj = torch.cat(projection_parts, dim=1)
    total_length = hidden.shape[1]
    # Match upstream padding, including the full pad block at exact multiples.
    padded_length = 128 - total_length % 128
    hs_pad = hidden.new_zeros(1, padded_length, native.inner_dim)
    rotary = native.rope(torch.cat(grid_parts, dim=1)[None])[:, :, None]
    rotary = F.pad(rotary, (0, 0, 0, 0, 0, padded_length))

    device = hidden.device
    target_frame_ids = (target_grid[0] // chunk_size * 2).to(torch.int)
    icl_frame_ids = (target_frame_ids[:0] if icl is None else
                     (icl["grid_id"][0][0] // max_chunk_size).to(torch.int))
    seq_ids = torch.zeros(total_length, device=device, dtype=torch.int)
    frame_ids = torch.cat([target_frame_ids, target_frame_ids, icl_frame_ids])
    noise_ids = torch.ones_like(seq_ids)
    noise_ids[:target_length] = 0
    type_ids = torch.zeros_like(seq_ids)
    icl_ids = torch.zeros_like(seq_ids)
    icl_ids[target_length * 2:] = 1
    cross_seq_ids = icl_ids.clone()
    metadata = [native._pad_metadata(value, padded_length) for value in
                (seq_ids, frame_ids, noise_ids, type_ids, icl_ids, cross_seq_ids)]
    text = input_dict["text_emb"].to(
        device=device, dtype=native.condition_embedder.text_embedder.linear_1.weight.dtype)
    text_hidden = native.condition_embedder.text_embedder(text)
    encoder_seq_ids = input_dict["encoder_seq_ids"].to(device=device, dtype=torch.int)
    native._build_training_masks(*metadata, encoder_seq_ids,
                                 input_dict["window_size"], native.blocks)

    collected = []
    for index, block in enumerate(native.blocks):
        hidden, _ = block(hidden, None, hs_pad, text_hidden, timestep_proj, None, rotary)
        if native.enable_mcp and index in native.mcp_hidden_collect_layers:
            collected.append(hidden[:, :target_length * 2])
    output = hidden[:, :target_length]
    shift, scale = (native.scale_shift_table[None] + target_temb[:, :, None]).unbind(2)
    output = (native.norm_out(output.float()) * (1 + scale) + shift).to(output.dtype)
    output = native.proj_out(output)
    output = output.reshape(1, -1, output.shape[-1] // math.prod(native.patch_size))
    mcp = native._forward_training_mcp(
        input_dict=input_dict, collected_hidden_states=collected,
        hs_action=None, action_timestep_proj=None, text_hidden_states=text_hidden,
        encoder_seq_ids=encoder_seq_ids, target_grid=target_grid,
        action_grid=target_grid[:, :0], target_frame_ids=target_frame_ids,
        action_frame_ids=target_frame_ids[:0], target_clean_timestep_proj=clean_proj,
        target_length=target_length, action_length=0)
    return output, mcp
