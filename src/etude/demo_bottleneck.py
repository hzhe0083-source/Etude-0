"""Ordered demonstration compression after the native frozen patch embedding.

The channel limit is structural, not a claim of view or appearance invariance.
All real frames contribute, including a final group shorter than group_frames.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TemporalDemoBottleneck(nn.Module):
    """Shared local queries followed by a Transformer over ordered group tokens.

Inputs are embedded patches [B,T,P,input_dim], real feature-frame timestamps
in seconds, and normalized xy patch centers. The returned timestamps describe
groups, not a fabricated image grid. ``adapter`` maps tokens to native width.
"""

    def __init__(self, input_dim: int, dim: int = 768, num_heads: int = 8,
                 group_frames: int = 4, tokens_per_group: int = 4, layers: int = 2):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in
               (input_dim, dim, num_heads, group_frames, tokens_per_group, layers)):
            raise ValueError("bottleneck dimensions, heads, groups and layers must be positive integers")
        if dim % num_heads:
            raise ValueError("bottleneck dim must be divisible by num_heads")
        self.input_dim, self.dim = input_dim, dim
        self.num_heads, self.layers = num_heads, layers
        self.group_frames, self.tokens_per_group = group_frames, tokens_per_group
        # Bind content, actual elapsed time and physical patch location BEFORE pooling.
        self.mix = nn.Sequential(nn.Linear(input_dim + 3, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.queries = nn.Parameter(torch.randn(tokens_per_group, dim) / math.sqrt(dim))
        self.cross_attention = nn.MultiheadAttention(dim, num_heads, dropout=0, batch_first=True)
        self.feed_forward = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim),
                                          nn.GELU(), nn.Linear(4 * dim, dim))
        self.group_time = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, num_heads, dim_feedforward=4 * dim,
                                           dropout=0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(dim),
                                              enable_nested_tensor=False)
        self.adapter = nn.Linear(dim, input_dim)

    def forward(self, features: Tensor, frame_times: Tensor,
                patch_coordinates: Tensor) -> tuple[Tensor, Tensor]:
        if (not isinstance(features, Tensor) or features.ndim != 4
                or not features.is_floating_point() or min(features.shape) < 1
                or features.shape[-1] != self.input_dim or not torch.isfinite(features).all()):
            raise ValueError("features must be finite nonempty floating [B,T,P,input_dim]")
        batch, length, patches, _ = features.shape
        if (not isinstance(frame_times, Tensor) or frame_times.shape != (length,)
                or not frame_times.is_floating_point() or not torch.isfinite(frame_times).all()
                or (frame_times[1:] <= frame_times[:-1]).any()):
            raise ValueError("frame_times must be finite strictly increasing floating seconds [T]")
        if (not isinstance(patch_coordinates, Tensor) or patch_coordinates.shape != (patches, 2)
                or not patch_coordinates.is_floating_point() or not torch.isfinite(patch_coordinates).all()
                or (patch_coordinates.abs() >= 1).any()
                or patch_coordinates.unique(dim=0).shape[0] != patches):
            raise ValueError("patch_coordinates must be unique finite normalized xy centers [P,2] in (-1,1)")
        # Keep timestamp arithmetic out of the model's possibly reduced precision.
        times = frame_times.to(device=features.device, dtype=torch.float64)
        elapsed = torch.log1p(times - times[0]).to(features.dtype)
        xy = patch_coordinates.to(features)
        positions = torch.cat((elapsed[:, None, None].expand(-1, patches, -1),
                               xy[None].expand(length, -1, -1)), -1)
        hidden = self.mix(torch.cat((features, positions[None].expand(batch, -1, -1, -1)), -1))

        groups = math.ceil(length / self.group_frames)
        padding = groups * self.group_frames - length
        hidden = F.pad(hidden, (0, 0, 0, 0, 0, padding))
        hidden = hidden.reshape(batch * groups, self.group_frames * patches, self.dim)
        padded = torch.arange(groups * self.group_frames, device=features.device) >= length
        padded = padded.reshape(groups, self.group_frames).repeat_interleave(patches, dim=1)
        padded = padded[None].expand(batch, -1, -1).reshape(batch * groups, -1)
        queries = self.queries[None].expand(batch * groups, -1, -1)
        pooled = queries + self.cross_attention(queries, hidden, hidden,
                                                key_padding_mask=padded, need_weights=False)[0]
        pooled = pooled + self.feed_forward(pooled)
        pooled = pooled.reshape(batch, groups, self.tokens_per_group, self.dim)

        group_times = torch.stack([times[start:start + self.group_frames].mean()
                                   for start in range(0, length, self.group_frames)])
        group_elapsed = torch.log1p(group_times - times[0]).to(features.dtype)
        pooled = pooled + self.group_time(group_elapsed[:, None])[None, :, None]
        tokens = self.temporal(pooled.flatten(1, 2))
        return tokens, group_times
