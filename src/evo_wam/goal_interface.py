"""Robot-frame SE(3) supervision for a continuous video-to-action interface.

Goal tokens are learned features, not decoded poses or complete action chunks.
The goal encoder has no visual or state input; the action condition receives
state separately. Visual goal inference uses future features in the target
robot scene, with explicit elapsed time and patch positions before pooling.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def validate_se3(poses: Tensor) -> None:
    """Require nonempty, finite homogeneous poses [...,4,4] with proper SO(3)."""
    if (not isinstance(poses, Tensor) or poses.ndim < 2 or poses.shape[-2:] != (4, 4)
            or min(poses.shape) < 1 or not poses.is_floating_point()
            or not torch.isfinite(poses).all()):
        raise ValueError("goal poses must be finite floating [...,4,4]")
    value = poses.detach().double()
    expected = value.new_tensor([0., 0., 0., 1.]).expand_as(value[..., 3, :])
    rotation = value[..., :3, :3]
    identity = torch.eye(3, device=value.device, dtype=value.dtype).expand_as(rotation)
    if (not torch.allclose(value[..., 3, :], expected, atol=1e-4, rtol=0)
            or not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-4, rtol=0)
            or not torch.allclose(torch.linalg.det(rotation), value.new_ones(rotation.shape[:-2]),
                                  atol=1e-4, rtol=0)):
        raise ValueError("goal poses must have proper SO(3) rotations and homogeneous bottom rows")


def validate_goal_poses(poses: Tensor) -> None:
    validate_se3(poses)
    if poses.ndim != 4:
        raise ValueError("goal poses must have batch and effector axes [B,E,4,4]")


def _translation_scale(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("translation_scale must be finite and positive")
    return float(value)


def _rotation_from_6d(value: Tensor) -> Tensor:
    # Work outside reduced precision so even a bf16 decoder produces proper SO(3).
    value = value if value.dtype == torch.float64 else value.float()
    first, second = value[..., :3], value[..., 3:]
    first = torch.where(first.norm(dim=-1, keepdim=True) > 1e-6,
                        first, value.new_tensor([1., 0., 0.]))
    first = F.normalize(first, dim=-1)
    # Cross products implement the same Gram-Schmidt projection without
    # subtracting nearly equal vectors, then restore orthogonality explicitly.
    normal = torch.linalg.cross(first, second, dim=-1)
    axis = F.one_hot(first.abs().argmin(-1), num_classes=3).to(value)
    fallback = torch.linalg.cross(first, axis, dim=-1)
    threshold = 1e-6 * second.norm(dim=-1, keepdim=True).clamp_min(1.)
    normal = torch.where(normal.norm(dim=-1, keepdim=True) > threshold, normal, fallback)
    third = F.normalize(normal, dim=-1)
    second = F.normalize(torch.linalg.cross(third, first, dim=-1), dim=-1)
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), -1)


class _PoseDecoder(nn.Module):
    def __init__(self, dim: int, effectors: int, num_heads: int, translation_scale: float):
        super().__init__()
        self.translation_scale = translation_scale
        self.queries = nn.Parameter(torch.randn(effectors, dim) / math.sqrt(dim))
        self.readout = nn.MultiheadAttention(dim, num_heads, dropout=0, batch_first=True)
        self.head = nn.Linear(dim, 9)
        nn.init.normal_(self.head.weight, std=.01)
        with torch.no_grad():
            self.head.bias.copy_(self.head.bias.new_tensor([0., 0., 0., 1., 0., 0., 0., 1., 0.]))

    def forward(self, tokens: Tensor) -> Tensor:
        queries = self.queries[None].expand(tokens.shape[0], -1, -1)
        hidden = queries + self.readout(queries, tokens, tokens, need_weights=False)[0]
        raw = self.head(hidden)
        raw = raw if raw.dtype == torch.float64 else raw.float()
        poses = raw.new_zeros(*raw.shape[:-1], 4, 4)
        poses[..., :3, :3] = _rotation_from_6d(raw[..., 3:])
        poses[..., :3, 3] = raw[..., :3] * self.translation_scale
        poses[..., 3, 3] = 1.
        return poses


class GoalInterface(nn.Module):
    """Shared continuous goal tokens and an explicit state-conditioned adapter.

``encode_goal`` is the nonvisual training path. ``read_future`` is the visual
path. Both return [B,K,dim]; ``condition`` maps these plus one state token to
[B,K+1,native_dim] for the action expert. Pose decoding is auxiliary only.
"""

    def __init__(self, native_dim: int, feature_dim: int, state_dim: int, effectors: int,
                 dim: int = 768, num_tokens: int = 64, num_heads: int = 8,
                 translation_scale: float = 1.):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in
               (native_dim, feature_dim, state_dim, effectors, dim, num_tokens, num_heads)):
            raise ValueError("interface dimensions, effectors, tokens and heads must be positive integers")
        if dim % num_heads:
            raise ValueError("interface dim must be divisible by num_heads")
        self.native_dim, self.feature_dim, self.state_dim = native_dim, feature_dim, state_dim
        self.effectors, self.dim, self.num_tokens = effectors, dim, num_tokens
        self.num_heads, self.translation_scale = num_heads, _translation_scale(translation_scale)
        self.goal_encoder = nn.Sequential(nn.Linear(effectors * 9, dim), nn.SiLU(),
            nn.Linear(dim, num_tokens * dim), nn.Unflatten(-1, (num_tokens, dim)), nn.LayerNorm(dim))
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, dim), nn.SiLU(), nn.Linear(dim, dim),
                                            nn.Unflatten(-1, (1, dim)))
        self.condition_adapter = nn.Linear(dim, native_dim)
        self.visual_project = nn.Sequential(nn.Linear(feature_dim + 3, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.visual_queries = nn.Parameter(torch.randn(num_tokens, dim) / math.sqrt(dim))
        self.visual_readout = nn.MultiheadAttention(dim, num_heads, dropout=0, batch_first=True)
        self.visual_norm = nn.LayerNorm(dim)
        self.pose_decoder = _PoseDecoder(dim, effectors, num_heads, self.translation_scale)

    def _state(self, state: Tensor, batch: int) -> Tensor:
        if (not isinstance(state, Tensor) or state.shape != (batch, self.state_dim)
                or not state.is_floating_point() or not torch.isfinite(state).all()):
            raise ValueError("state must be finite floating [B,state_dim]")
        return self.state_encoder(state.to(self.state_encoder[0].weight))

    def _tokens(self, tokens: Tensor) -> Tensor:
        if (not isinstance(tokens, Tensor) or tokens.ndim != 3 or tokens.shape[0] < 1
                or tokens.shape[1:] != (self.num_tokens, self.dim)
                or not tokens.is_floating_point() or not torch.isfinite(tokens).all()):
            raise ValueError("goal tokens must be finite floating [B,num_tokens,dim]")
        return tokens.to(self.condition_adapter.weight)

    def encode_goal(self, poses: Tensor) -> Tensor:
        validate_goal_poses(poses)
        if poses.shape[1] != self.effectors:
            raise ValueError("goal poses must contain the configured number of effectors")
        # Rotation uses columns, never Euler angles or arbitrary quaternion signs.
        value = poses.to(self.goal_encoder[0].weight)
        rotation = value[..., :3, :2].transpose(-1, -2).flatten(-2)
        goal = torch.cat((value[..., :3, 3] / self.translation_scale, rotation), -1)
        return self.goal_encoder(goal.flatten(1))

    def read_future(self, features: Tensor, state: Tensor, frame_times: Tensor,
                    patch_coordinates: Tensor) -> Tensor:
        if (not isinstance(features, Tensor) or features.ndim != 4 or min(features.shape) < 1
                or features.shape[-1] != self.feature_dim or not features.is_floating_point()
                or not torch.isfinite(features).all()):
            raise ValueError("future features must be finite floating [B,F,P,feature_dim]")
        batch, frames, patches, _ = features.shape
        if (not isinstance(frame_times, Tensor) or frame_times.shape != (frames,)
                or not frame_times.is_floating_point() or not torch.isfinite(frame_times).all()
                or (frame_times[1:] <= frame_times[:-1]).any()):
            raise ValueError("frame_times must be finite strictly increasing floating seconds [F]")
        if (not isinstance(patch_coordinates, Tensor) or patch_coordinates.shape != (patches, 2)
                or not patch_coordinates.is_floating_point() or not torch.isfinite(patch_coordinates).all()
                or (patch_coordinates.abs() >= 1).any()
                or patch_coordinates.unique(dim=0).shape[0] != patches):
            raise ValueError("patch_coordinates must be unique normalized xy centers [P,2] in (-1,1)")
        value = features.to(self.visual_project[0].weight)
        times = frame_times.to(device=value.device, dtype=torch.float64)
        elapsed = torch.log1p(times - times[0]).to(value.dtype)
        positions = torch.cat((elapsed[:, None, None].expand(-1, patches, -1),
                               patch_coordinates.to(value)[None].expand(frames, -1, -1)), -1)
        memory = self.visual_project(torch.cat((value, positions[None].expand(batch, -1, -1, -1)), -1))
        memory = memory.flatten(1, 2)
        queries = self.visual_queries[None] + self._state(state, batch)
        return self.visual_norm(queries + self.visual_readout(queries, memory, memory, need_weights=False)[0])

    def condition(self, tokens: Tensor, state: Tensor) -> Tensor:
        tokens = self._tokens(tokens)
        return self.condition_adapter(torch.cat((tokens, self._state(state, tokens.shape[0])), dim=1))

    def decode_goal(self, tokens: Tensor) -> Tensor:
        return self.pose_decoder(self._tokens(tokens))


def goal_pose_loss(prediction: Tensor, target: Tensor, translation_scale: float = 1.) -> dict[str, Tensor]:
    """Normalized translation MSE plus squared chordal SO(3) distance.

Unlike acos-based geodesic loss, the chordal term has finite gradients at
identity. Translation and rotation remain separate for interpretable reporting.
"""
    scale = _translation_scale(translation_scale)
    validate_goal_poses(target)
    if (not isinstance(prediction, Tensor) or prediction.shape != target.shape
            or not prediction.is_floating_point() or not torch.isfinite(prediction).all()):
        raise ValueError("predicted poses must be finite floating tensors matching target [B,E,4,4]")
    prediction = prediction if prediction.dtype == torch.float64 else prediction.float()
    target = target.detach().to(prediction)
    translation = F.mse_loss(prediction[..., :3, 3] / scale, target[..., :3, 3] / scale)
    rotation = (prediction[..., :3, :3] - target[..., :3, :3]).square().sum(dim=(-2, -1)).mean()
    return {"translation": translation, "rotation": rotation, "total": translation + rotation}
