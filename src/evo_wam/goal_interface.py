"""Robot-frame SE(3) supervision for a continuous video-to-action interface.

Stage 1 embeds robot goals without visual input. Stage 2 recurrently updates a
latent workspace at each video/action coupling layer. The pose head supervises
a subset of that workspace; it does not establish that irrelevant visual
information has been removed.
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


def validate_gripper(value: Tensor, shape: tuple[int, ...], name: str = "goal gripper") -> None:
    if (not isinstance(value, Tensor) or value.shape != shape or not value.is_floating_point()
            or not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any()):
        raise ValueError(f"{name} must be finite floating {shape} in [0,1]")


class _PoseDecoder(nn.Module):
    def __init__(self, dim: int, effectors: int, num_heads: int, translation_scale: float,
                 num_pose_tokens: int):
        super().__init__()
        self.translation_scale = translation_scale
        self.num_pose_tokens = num_pose_tokens
        self.queries = nn.Parameter(torch.randn(effectors, dim) / math.sqrt(dim))
        self.readout = nn.MultiheadAttention(dim, num_heads, dropout=0, batch_first=True)
        self.head = nn.Linear(dim, 10)
        nn.init.normal_(self.head.weight, std=.01)
        with torch.no_grad():
            self.head.bias.copy_(self.head.bias.new_tensor([0., 0., 0., 1., 0., 0., 0., 1., 0., 0.]))

    def forward(self, tokens: Tensor, *, per_effector: bool = False) -> dict[str, Tensor]:
        tokens = tokens[:, :self.num_pose_tokens]
        queries = self.queries[None].expand(tokens.shape[0], -1, -1)
        if per_effector:
            if tokens.shape[1] != self.queries.shape[0]:
                raise ValueError("per-effector pose decoding requires one token per effector")
            batch, effectors, dim = queries.shape
            query = queries.reshape(batch * effectors, 1, dim)
            memory = tokens.reshape(batch * effectors, 1, dim)
            attended = self.readout(query, memory, memory, need_weights=False)[0]
            hidden = queries + attended.reshape(batch, effectors, dim)
        else:
            hidden = queries + self.readout(queries, tokens, tokens, need_weights=False)[0]
        raw = self.head(hidden)
        raw = raw if raw.dtype == torch.float64 else raw.float()
        poses = raw.new_zeros(*raw.shape[:-1], 4, 4)
        poses[..., :3, :3] = _rotation_from_6d(raw[..., 3:9])
        poses[..., :3, 3] = raw[..., :3] * self.translation_scale
        poses[..., 3, 3] = 1.
        return {"goal_poses": poses, "goal_gripper": raw[..., 9].sigmoid()}


class _RecurrentGroup(nn.Module):
    """One parameter group shared by a contiguous set of coupling layers."""

    def __init__(self, native_dim: int, dim: int, num_heads: int):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, num_heads, dropout=0, batch_first=True)
        self.semantic_norm = nn.LayerNorm(dim)
        self.semantic_project = nn.Linear(native_dim, dim)
        self.semantic_attention = nn.MultiheadAttention(dim, num_heads, dropout=0, batch_first=True)
        self.visual_norm = nn.LayerNorm(dim)
        # Jointly encode content and physical positions before set aggregation.
        self.visual_project = nn.Sequential(nn.Linear(native_dim + 3, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.visual_attention = nn.MultiheadAttention(dim, num_heads, dropout=0, batch_first=True)
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, tokens: Tensor, semantic: Tensor, visual: Tensor) -> Tensor:
        query = self.self_norm(tokens)
        tokens = tokens + self.self_attention(query, query, query, need_weights=False)[0]
        memory = self.semantic_project(semantic)
        tokens = tokens + self.semantic_attention(self.semantic_norm(tokens), memory, memory, need_weights=False)[0]
        memory = self.visual_project(visual)
        tokens = tokens + self.visual_attention(self.visual_norm(tokens), memory, memory, need_weights=False)[0]
        return self.output_norm(tokens)


class GoalInterface(nn.Module):
    """Goal-conditioned Stage 1 and layer-recurrent visual conditions for Stage 2.

    Each query represents one action block. ``semantic`` contains only text and
    independently encoded robot state. ``read_layer`` updates [B,K,dim] tokens;
    each action layer receives its own ``condition``. Decoding is auxiliary.
    """

    def __init__(self, native_dim: int, feature_dim: int, state_dim: int, effectors: int,
                 dim: int = 768, num_tokens: int = 100, num_heads: int = 8,
                 translation_scale: float = 1., num_layers: int = 30,
                 num_layer_groups: int = 6, num_pose_tokens: int = 8):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in
               (native_dim, feature_dim, state_dim, effectors, dim, num_tokens, num_heads,
                num_layers, num_layer_groups, num_pose_tokens)):
            raise ValueError("interface dimensions, effectors, tokens, heads and groups must be positive integers")
        if feature_dim != native_dim:
            raise ValueError("feature_dim must equal native_dim for per-layer video features")
        if dim % num_heads:
            raise ValueError("interface dim must be divisible by num_heads")
        if num_layers % num_layer_groups:
            raise ValueError("num_layers must be divisible by num_layer_groups")
        if num_pose_tokens > num_tokens:
            raise ValueError("num_pose_tokens must not exceed num_tokens")
        self.native_dim, self.feature_dim, self.state_dim = native_dim, feature_dim, state_dim
        self.effectors, self.dim, self.num_tokens = effectors, dim, num_tokens
        self.num_heads, self.translation_scale = num_heads, _translation_scale(translation_scale)
        self.num_layers, self.num_layer_groups = num_layers, num_layer_groups
        self.num_pose_tokens = num_pose_tokens
        self.goal_encoder = nn.Sequential(nn.Linear(effectors * 10, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, num_tokens * dim),
            nn.Unflatten(-1, (num_tokens, dim)), nn.LayerNorm(dim))
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, dim), nn.SiLU(), nn.Linear(dim, native_dim),
                                            nn.Unflatten(-1, (1, native_dim)))
        self.condition_adapter = nn.Linear(dim, native_dim)
        self.visual_queries = nn.Parameter(torch.randn(num_tokens, dim) / math.sqrt(dim))
        self.recurrent_groups = nn.ModuleList(
            _RecurrentGroup(native_dim, dim, num_heads) for _ in range(num_layer_groups))
        self.pose_decoder = _PoseDecoder(dim, effectors, num_heads, self.translation_scale, num_pose_tokens)

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

    def _features(self, features: Tensor) -> Tensor:
        if (not isinstance(features, Tensor) or features.ndim != 4 or min(features.shape) < 1
                or features.shape[-1] != self.feature_dim or not features.is_floating_point()
                or not torch.isfinite(features).all()):
            raise ValueError("future features must be finite floating [B,F,P,native_dim]")
        return features.to(self.condition_adapter.weight)

    def _semantic(self, semantic: Tensor, batch: int) -> Tensor:
        if (batch < 1 or not isinstance(semantic, Tensor) or semantic.ndim != 3 or semantic.shape[0] != batch
                or semantic.shape[1] < 1 or semantic.shape[2] != self.native_dim
                or not semantic.is_floating_point() or not torch.isfinite(semantic).all()):
            raise ValueError("semantic features must be finite floating [B,L,native_dim]")
        return semantic.to(self.condition_adapter.weight)

    def encode_goal(self, poses: Tensor, gripper: Tensor) -> Tensor:
        validate_goal_poses(poses)
        if poses.shape[1] != self.effectors:
            raise ValueError("goal poses must contain the configured number of effectors")
        validate_gripper(gripper, poses.shape[:2])
        value = poses.to(self.goal_encoder[0].weight)
        rotation = value[..., :3, :2].transpose(-1, -2).flatten(-2)
        goal = torch.cat((value[..., :3, 3] / self.translation_scale, rotation,
                          gripper.to(value).unsqueeze(-1)), -1)
        return self.goal_encoder(goal.flatten(1))

    def initial_queries(self, batch: int) -> Tensor:
        if type(batch) is not int or batch < 1:
            raise ValueError("query batch must be a positive integer")
        return self.visual_queries[None].expand(batch, -1, -1)

    def semantic(self, language_hidden: Tensor, state: Tensor) -> Tensor:
        batch = state.shape[0] if isinstance(state, Tensor) and state.ndim > 0 else 0
        language = self._semantic(language_hidden, batch)
        return torch.cat((language, self._state(state, batch)), dim=1)

    def read_layer(self, tokens: Tensor, features: Tensor, semantic: Tensor,
                   frame_times: Tensor, patch_coordinates: Tensor, layer_index: int) -> Tensor:
        tokens, value = self._tokens(tokens), self._features(features)
        batch, frames, patches, _ = value.shape
        if tokens.shape[0] != batch:
            raise ValueError("tokens and future features must share a batch")
        semantic = self._semantic(semantic, batch)
        if type(layer_index) is not int or not 0 <= layer_index < self.num_layers:
            raise ValueError("layer_index must identify a configured coupling layer")
        if (not isinstance(frame_times, Tensor) or frame_times.shape != (frames,)
                or not frame_times.is_floating_point() or not torch.isfinite(frame_times).all()
                or (frame_times[1:] <= frame_times[:-1]).any()):
            raise ValueError("frame_times must be finite strictly increasing floating seconds [F]")
        if (not isinstance(patch_coordinates, Tensor) or patch_coordinates.shape != (patches, 2)
                or not patch_coordinates.is_floating_point() or not torch.isfinite(patch_coordinates).all()
                or (patch_coordinates.abs() >= 1).any()
                or patch_coordinates.unique(dim=0).shape[0] != patches):
            raise ValueError("patch_coordinates must be unique normalized xy centers [P,2] in (-1,1)")
        times = frame_times.to(device=value.device, dtype=torch.float64)
        elapsed = torch.log1p(times - times[0]).to(value.dtype)
        positions = torch.cat((elapsed[:, None, None].expand(-1, patches, -1),
                               patch_coordinates.to(value)[None].expand(frames, -1, -1)), -1)
        memory = torch.cat((value, positions[None].expand(batch, -1, -1, -1)), -1).flatten(1, 2)
        group = layer_index // (self.num_layers // self.num_layer_groups)
        return self.recurrent_groups[group](tokens, semantic, memory)

    def condition(self, tokens: Tensor, state: Tensor, language_hidden: Tensor) -> Tensor:
        tokens = self._tokens(tokens)
        semantic = self.semantic(language_hidden, state)
        if tokens.shape[0] != semantic.shape[0]:
            raise ValueError("tokens and semantic features must share a batch")
        return torch.cat((semantic, self.condition_adapter(tokens)), dim=1)

    def direct_condition(self, features: Tensor, state: Tensor, language_hidden: Tensor) -> Tensor:
        """Matched baseline: raw future features intentionally bypass the interface."""
        value = self._features(features)
        semantic = self.semantic(language_hidden, state)
        if value.shape[0] != semantic.shape[0]:
            raise ValueError("future and semantic features must share a batch")
        return torch.cat((semantic, value.flatten(1, 2)), dim=1)

    def decode_goal(self, tokens: Tensor) -> dict[str, Tensor]:
        return self.pose_decoder(self._tokens(tokens))


def goal_pose_loss(prediction: Tensor, target: Tensor, translation_scale: float = 1., *,
                   gripper_prediction: Tensor, gripper_target: Tensor) -> dict[str, Tensor]:
    """Three supervised terms plus detached errors in physical/interpretable units.

    The squared chordal rotation loss has finite gradients at identity. Angular
    error is only a reporting metric; no acos derivative enters optimization.
    """
    scale = _translation_scale(translation_scale)
    validate_goal_poses(target)
    if (not isinstance(prediction, Tensor) or prediction.shape != target.shape
            or not prediction.is_floating_point() or not torch.isfinite(prediction).all()):
        raise ValueError("predicted poses must be finite floating tensors matching target [B,E,4,4]")
    validate_gripper(gripper_prediction, target.shape[:2], "predicted gripper")
    validate_gripper(gripper_target, target.shape[:2])
    prediction = prediction if prediction.dtype == torch.float64 else prediction.float()
    target = target.detach().to(prediction)
    grip = gripper_prediction.to(prediction)
    grip_target = gripper_target.detach().to(prediction)
    delta = prediction[..., :3, 3] - target[..., :3, 3]
    translation = (delta / scale).square().mean()
    rotation = (prediction[..., :3, :3] - target[..., :3, :3]).square().sum(dim=(-2, -1)).mean()
    gripper = F.mse_loss(grip, grip_target)
    with torch.no_grad():
        # Elementwise trace avoids an autocast matmul rounding this diagnostic to bf16.
        trace = (prediction[..., :3, :3] * target[..., :3, :3]).sum(dim=(-2, -1))
        cosine = ((trace - 1) / 2).clamp(-1, 1)
        position_error = delta.norm(dim=-1).mean()
        orientation_error = torch.rad2deg(torch.acos(cosine)).mean()
        gripper_error = (grip - grip_target).abs().mean()
    return {"translation": translation, "rotation": rotation, "gripper": gripper,
            "total": translation + rotation + gripper, "position_error_m": position_error,
            "orientation_error_deg": orientation_error, "gripper_error": gripper_error}
