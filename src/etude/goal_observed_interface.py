"""Observed Wan context and supervised pose features jointly condition actions.

Inputs are hidden features of the packed observed robot history and demonstration.
This interface neither samples future video nor accepts goal labels as inputs.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .goal_interface import _rotation_from_6d, _translation_scale


class _ObservedPoseDecoder(nn.Module):
    """Read all context with one learned query per controlled effector."""

    def __init__(self, dim: int, effectors: int, num_heads: int, translation_scale: float):
        super().__init__()
        self.translation_scale = translation_scale
        self.queries = nn.Parameter(torch.randn(effectors, dim) / math.sqrt(dim))
        self.readout = nn.MultiheadAttention(dim, num_heads, dropout=0, batch_first=True)
        self.output_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)
        nn.init.normal_(self.head.weight, std=.01)
        with torch.no_grad():
            self.head.bias.copy_(self.head.bias.new_tensor([0., 0., 0., 1., 0., 0., 0., 1., 0., 0.]))

    def forward(self, context: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        queries = self.queries[None].expand(context.shape[0], -1, -1)
        hidden = self.output_norm(queries + self.readout(queries, context, context, need_weights=False)[0])
        raw = self.head(hidden)
        # Keep homogeneous poses in full precision, including under autocast.
        raw = raw if raw.dtype == torch.float64 else raw.float()
        poses = raw.new_zeros(*raw.shape[:-1], 4, 4)
        poses[..., :3, :3] = _rotation_from_6d(raw[..., 3:9])
        poses[..., :3, 3] = raw[..., :3] * self.translation_scale
        poses[..., 3, 3] = 1.
        return hidden, {"goal_poses": poses, "goal_gripper": raw[..., 9].sigmoid()}


class ObservedGoalInterface(nn.Module):
    """Form action context from language, state, Wan features, and pose features.

    The direct Wan branch preserves every observed token. The pose decoder also
    reads every token, plus independently encoded state and language, and exposes
    its per-effector hidden features to both the pose head and action branch.
    Pose/gripper targets belong only in the external supervised loss.
    """

    def __init__(self, *, native_dim: int, state_dim: int, effectors: int,
                 dim: int = 768, num_heads: int = 8, translation_scale: float = 1.):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in
               (native_dim, state_dim, effectors, dim, num_heads)):
            raise ValueError("interface dimensions, effectors and heads must be positive integers")
        if dim % num_heads:
            raise ValueError("interface dim must be divisible by num_heads")
        self.native_dim, self.state_dim, self.effectors = native_dim, state_dim, effectors
        self.dim, self.num_heads = dim, num_heads
        self.translation_scale = _translation_scale(translation_scale)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, dim), nn.SiLU(), nn.Linear(dim, native_dim),
            nn.Unflatten(-1, (1, native_dim)))
        self.wan_context_projection = nn.Sequential(nn.Linear(native_dim, dim), nn.LayerNorm(dim))
        self.state_context_projection = nn.Sequential(nn.Linear(native_dim, dim), nn.LayerNorm(dim))
        self.language_context_projection = nn.Sequential(nn.Linear(native_dim, dim), nn.LayerNorm(dim))
        self.pose_decoder = _ObservedPoseDecoder(dim, effectors, num_heads, self.translation_scale)
        # Independent normalizations keep both action feature sources explicit.
        self.wan_action_projection = nn.Sequential(nn.Linear(native_dim, native_dim), nn.LayerNorm(native_dim))
        self.pose_action_projection = nn.Sequential(nn.Linear(dim, native_dim), nn.LayerNorm(native_dim))

    def _sequence(self, value: Tensor, name: str, batch: int | None = None) -> Tensor:
        if (not isinstance(value, Tensor) or value.ndim != 3 or min(value.shape) < 1
                or value.shape[-1] != self.native_dim
                or (batch is not None and value.shape[0] != batch)
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite floating [B,S,native_dim] with matching batch")
        return value.to(self.wan_action_projection[0].weight)

    def condition_from_features(self, features: Tensor, state: Tensor,
                                language_hidden: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Consume [B,S,native_dim], [B,state_dim], and [B,L,native_dim].

        Return [B,L+1+S+effectors,native_dim] action conditions and only the
        ``goal_poses`` [B,E,4,4] and ``goal_gripper`` [B,E] predictions. Tensor
        conversions preserve gradients, including through the pose hidden branch.
        """
        features = self._sequence(features, "observed features")
        batch = features.shape[0]
        language = self._sequence(language_hidden, "language hidden", batch)
        if (not isinstance(state, Tensor) or state.shape != (batch, self.state_dim)
                or not state.is_floating_point() or not torch.isfinite(state).all()):
            raise ValueError("state must be finite floating [B,state_dim] with matching batch")
        state_token = self.state_encoder(state.to(self.state_encoder[0].weight))
        context = torch.cat((self.wan_context_projection(features),
                             self.state_context_projection(state_token),
                             self.language_context_projection(language)), dim=1)
        hidden, prediction = self.pose_decoder(context)
        wan_condition = self.wan_action_projection(features)
        pose_condition = self.pose_action_projection(hidden)
        condition = torch.cat((language.to(wan_condition), state_token.to(wan_condition),
                               wan_condition, pose_condition.to(wan_condition)), dim=1)
        return condition, prediction
