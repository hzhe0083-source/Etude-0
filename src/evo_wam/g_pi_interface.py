"""Goal translation and robot-only goal-conditioned action interfaces."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .goal_action import goal_action_sample
from .goal_interface import (
    GoalInterface, _PoseDecoder, _translation_scale, goal_pose_loss, validate_goal_poses, validate_gripper,
)
from .video_data import patch_grid_coordinates


def normalize_z(value: Tensor) -> Tensor:
    if (not isinstance(value, Tensor) or value.ndim != 3 or min(value.shape) < 1
            or not value.is_floating_point() or not torch.isfinite(value).all()):
        raise ValueError("z must be finite floating [B,K_z,d_z]")
    return F.normalize(value.float(), dim=-1)


class _GoalDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, num_heads, dropout=0, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, num_heads, dropout=0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, tokens: Tensor, memory: Tensor) -> Tensor:
        query = self.self_norm(tokens)
        tokens = tokens + self.self_attention(query, query, query, need_weights=False)[0]
        tokens = tokens + self.cross_attention(self.cross_norm(tokens), memory, memory, need_weights=False)[0]
        return tokens + self.ffn(self.ffn_norm(tokens))


class GGoalDecoder(nn.Module):
    """The complete trainable portion of G, reading one frozen native layer."""

    def __init__(self, native_dim: int, d_z: int, state_dim: int, effectors: int,
                 k_z: int = 8, dim: int = 768, num_heads: int = 8,
                 num_layers: int = 2, translation_scale: float = 1., use_state: bool = True):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in
               (native_dim, d_z, state_dim, effectors, k_z, dim, num_heads, num_layers)):
            raise ValueError("decoder dimensions, effectors, tokens, heads and layers must be positive integers")
        if dim % num_heads:
            raise ValueError("decoder dim must be divisible by num_heads")
        if type(use_state) is not bool:
            raise ValueError("use_state must be boolean")
        self.native_dim, self.d_z, self.state_dim = native_dim, d_z, state_dim
        self.k_z, self.dim, self.effectors = k_z, dim, effectors
        self.translation_scale, self.use_state = _translation_scale(translation_scale), use_state
        self.queries = nn.Parameter(torch.randn(k_z + 1, dim) / math.sqrt(dim))
        self.memory_projection = nn.Sequential(nn.Linear(native_dim, dim), nn.LayerNorm(dim))
        self.state_projection = nn.Linear(state_dim, dim) if use_state else None
        self.blocks = nn.ModuleList(_GoalDecoderBlock(dim, num_heads) for _ in range(num_layers))
        self.output_norm = nn.LayerNorm(dim)
        self.z_head = nn.Linear(dim, d_z)
        self.pose_decoder = _PoseDecoder(dim, effectors, num_heads, self.translation_scale, 1)

    def forward(self, features: Tensor, state: Tensor) -> dict[str, Tensor]:
        if (not isinstance(features, Tensor) or features.ndim != 3 or min(features.shape) < 1
                or features.shape[-1] != self.native_dim or not features.is_floating_point()
                or not torch.isfinite(features).all()):
            raise ValueError("G features must be finite floating [B,S,native_dim]")
        batch = features.shape[0]
        if (not isinstance(state, Tensor) or state.shape != (batch, self.state_dim)
                or not state.is_floating_point() or not torch.isfinite(state).all()):
            raise ValueError("state must be finite floating [B,state_dim]")
        memory = self.memory_projection(features.detach().to(self.queries))
        if self.state_projection is not None:
            memory = torch.cat((memory, self.state_projection(state.detach().to(self.queries))[:, None]), dim=1)
        tokens = self.queries[None].expand(batch, -1, -1)
        for block in self.blocks:
            tokens = block(tokens, memory)
        tokens = self.output_norm(tokens)
        return {"z": normalize_z(self.z_head(tokens[:, :self.k_z])),
                **self.pose_decoder(tokens[:, self.k_z:])}


def g_goal_loss(prediction: dict, target: dict, translation_scale: float = 1.,
                pose_weight: float = 1.) -> dict[str, Tensor]:
    if type(pose_weight) not in (int, float) or not math.isfinite(pose_weight) or pose_weight < 0:
        raise ValueError("pose_weight must be finite and nonnegative")
    if any(not isinstance(value, dict) or not {"z", "goal_poses", "goal_gripper"} <= value.keys()
           for value in (prediction, target)):
        raise ValueError("prediction and target must contain z, goal_poses and goal_gripper")
    predicted_z, target_z = normalize_z(prediction["z"]), normalize_z(target["z"]).detach()
    if predicted_z.shape != target_z.shape:
        raise ValueError("predicted and target z must have matching [B,K_z,d_z] shapes")
    z_loss = F.mse_loss(predicted_z, target_z.to(predicted_z))
    pose = goal_pose_loss(prediction["goal_poses"], target["goal_poses"], translation_scale,
                         gripper_prediction=prediction["goal_gripper"], gripper_target=target["goal_gripper"])
    return {"z": z_loss, **{f"pose_{key}": value for key, value in pose.items()},
            "total": z_loss + pose_weight * pose["total"]}


def perturb_goal(goal: dict, *, z_std: float = 0., translation_std: float = 0.,
                 rotation_std: float = 0., gripper_std: float = 0.,
                 generator: torch.Generator | None = None) -> dict[str, Tensor]:
    """Perturb detached goals; rotation_std is in radians, translation in metres."""
    scales = (z_std, translation_std, rotation_std, gripper_std)
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in scales):
        raise ValueError("goal noise scales must be finite and nonnegative")
    if not isinstance(goal, dict) or not {"z", "goal_poses", "goal_gripper"} <= goal.keys():
        raise ValueError("goal must contain z, goal_poses and goal_gripper")
    validate_goal_poses(goal["goal_poses"])
    validate_gripper(goal["goal_gripper"], goal["goal_poses"].shape[:2])
    z = normalize_z(goal["z"]).detach()
    poses, gripper = goal["goal_poses"].detach().clone(), goal["goal_gripper"].detach().clone()
    if z.shape[0] != poses.shape[0]:
        raise ValueError("z and poses must share a batch")
    if generator is not None and (not isinstance(generator, torch.Generator) or generator.device.type != "cpu"):
        raise ValueError("goal noise requires a CPU torch.Generator")

    def noise_like(value):
        return torch.randn(value.shape, generator=generator, device="cpu").to(value)

    if z_std:
        z = z + z_std * noise_like(z)
    if translation_std:
        poses[..., :3, 3] += translation_std * noise_like(poses[..., :3, 3])
    if rotation_std:
        vector = rotation_std * noise_like(poses[..., :3, 3]).float()
        x, y, z_axis = vector.unbind(-1)
        zeros = torch.zeros_like(x)
        skew = torch.stack((zeros, -z_axis, y, z_axis, zeros, -x, -y, x, zeros), -1).unflatten(-1, (3, 3))
        rotation = torch.matrix_exp(skew) @ poses[..., :3, :3].float()
        poses = poses.float()
        poses[..., :3, :3] = rotation
    if gripper_std:
        gripper = (gripper + gripper_std * noise_like(gripper)).clamp(0, 1)
    return {"z": z, "goal_poses": poses, "goal_gripper": gripper}


class PiGoalInterface(GoalInterface):
    """Recurrent LIT workspace with semantic [language, state, z, pose] memory."""

    def __init__(self, native_dim: int, feature_dim: int, state_dim: int, effectors: int,
                 d_z: int, k_z: int = 8, **kwargs):
        if any(type(value) is not int or value < 1 for value in (d_z, k_z)):
            raise ValueError("d_z and k_z must be positive integers")
        super().__init__(native_dim, feature_dim, state_dim, effectors, **kwargs)
        self.d_z, self.k_z = d_z, k_z
        # No auxiliary pose prediction is used by the robot-only action route.
        del self.pose_decoder
        self.z_projection = nn.Linear(d_z, native_dim)

    def goal_semantic(self, language_hidden: Tensor, state: Tensor, goal: dict) -> Tensor:
        semantic = self.semantic(language_hidden, state)
        if not isinstance(goal, dict) or not {"z", "goal_poses", "goal_gripper"} <= goal.keys():
            raise ValueError("goal must contain z, goal_poses and goal_gripper")
        z = goal["z"]
        if (not isinstance(z, Tensor) or z.shape != (semantic.shape[0], self.k_z, self.d_z)
                or not z.is_floating_point() or not torch.isfinite(z).all()):
            raise ValueError("goal z must be finite floating [B,K_z,d_z]")
        validate_goal_poses(goal["goal_poses"])
        validate_gripper(goal["goal_gripper"], goal["goal_poses"].shape[:2])
        poses = self.encode_goal(goal["goal_poses"].detach(), goal["goal_gripper"].detach())
        if poses.shape[0] != semantic.shape[0]:
            raise ValueError("goal and semantic features must share a batch")
        return torch.cat((semantic, self.z_projection(z.detach().to(self.z_projection.weight)),
                          self.condition_adapter(poses)), dim=1)

    def conditions(self, features: dict[int, Tensor], state: Tensor, language_hidden: Tensor,
                   goal: dict, frame_times: Tensor, patch_coordinates: Tensor) -> list[Tensor]:
        if not isinstance(features, dict) or list(features) != list(range(self.num_layers)):
            raise ValueError("robot features must contain every configured layer in depth order")
        if (not isinstance(frame_times, Tensor) or frame_times.ndim != 1
                or not isinstance(patch_coordinates, Tensor) or patch_coordinates.ndim != 2):
            raise ValueError("frame_times and patch_coordinates must have frame and patch axes")
        semantic = self.goal_semantic(language_hidden, state, goal)
        batch, frames, patches = semantic.shape[0], len(frame_times), len(patch_coordinates)
        tokens = self.initial_queries(batch)
        conditions = []
        for index, feature in features.items():
            if (not isinstance(feature, Tensor)
                    or feature.shape != (batch, frames * patches, self.native_dim)):
                raise ValueError("robot layer features must match the observed frame and patch grid")
            visual = feature.detach().unflatten(1, (frames, patches))
            tokens = self.read_layer(tokens, visual, semantic, frame_times, patch_coordinates, index)
            conditions.append(self.condition(tokens, state, language_hidden))
        return conditions


class GTranslator(nn.Module):
    """Offline G with task-local demonstration KV caching and no generated video."""

    def __init__(self, native, goal_decoder: GGoalDecoder, config: dict, *, feature_layer: int = -1):
        super().__init__()
        from .g_pi_context import assert_frozen_base
        assert_frozen_base(native)
        if feature_layer == -1:
            feature_layer = len(native.blocks) - 1
        if type(feature_layer) is not int or not 0 <= feature_layer < len(native.blocks):
            raise ValueError("feature_layer must identify a native block")
        if goal_decoder.native_dim != native.inner_dim:
            raise ValueError("goal decoder width must match the frozen native model")
        self.native, self.goal_decoder = native, goal_decoder
        self.config, self.feature_layer = dict(config), feature_layer
        self._demo_cache, self._demo_reference = None, None
        self.native.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.native.eval()
        return self

    def clear_demo_cache(self):
        self._demo_cache, self._demo_reference = None, None

    @torch.no_grad()
    def cache_demo(self, demo: Tensor):
        from .g_pi_context import build_demo_cache
        self._demo_cache = build_demo_cache(self.native, demo, self.config)
        self._demo_reference = demo.detach().clone()
        return self._demo_cache

    @torch.no_grad()
    def predict(self, demo: Tensor, robot_frames_t0_to_t: Tensor, state: Tensor, *,
                current_index: int | None = None, use_cache: bool = True) -> dict[str, Tensor]:
        from .g_pi_context import g_context_features
        from .goal_training import autocast_for
        if use_cache and (self._demo_reference is None or not torch.equal(demo, self._demo_reference)):
            self.cache_demo(demo)
        with autocast_for(self.native):
            features = g_context_features(self.native, demo, robot_frames_t0_to_t, self.config,
                                          demo_cache=self._demo_cache if use_cache else None,
                                          current_index=current_index)
            return self.goal_decoder(features[self.feature_layer], state)


class PiGoalPolicy(nn.Module):
    """Offline policy; its public and internal interfaces never accept a demo."""

    def __init__(self, native, interface: PiGoalInterface, config: dict, *,
                 action_shape, actions_mask: Tensor, seed: int = 0, video_native=None):
        super().__init__()
        from .g_pi_context import assert_frozen_base
        video_native = native if video_native is None else video_native
        assert_frozen_base(video_native)
        if (not isinstance(action_shape, (list, tuple, torch.Size)) or len(action_shape) != 5
                or any(type(size) is not int or size < 1 for size in action_shape)
                or action_shape[0] != 1 or action_shape[1] != native.config.action_dim or action_shape[-1] != 1):
            raise ValueError("action_shape must be positive native [1,A,F,N,1]")
        if not getattr(native, "_goal_action_interface", False):
            raise ValueError("install the goal action interface before constructing the policy")
        if (interface.native_dim != native.inner_dim or interface.native_dim != video_native.inner_dim
                or interface.num_layers != len(native.blocks) or interface.num_layers != len(video_native.blocks)):
            raise ValueError("policy, frozen video and action models must share widths and layer counts")
        from .zerowam import action_mask_for
        if not action_mask_for(actions_mask, torch.empty(action_shape)).any():
            raise ValueError("at least one action channel must be active")
        if type(seed) is not int or seed < 0:
            raise ValueError("sampling seed must be a nonnegative integer")
        self.native, self.video_native, self.interface = native, video_native, interface
        self.config, self.action_shape = dict(config), tuple(action_shape)
        self.register_buffer("actions_mask", actions_mask.detach().clone())
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.video_native.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.video_native.eval()
        return self

    @torch.no_grad()
    def predict(self, robot_frames_t0_to_t: Tensor, state: Tensor, language: Tensor, goal: dict, *,
                current_index: int | None = None, frame_times: Tensor | None = None) -> Tensor:
        from .g_pi_context import pi_context_features, truncate_robot_history
        from .goal_training import autocast_for
        history = truncate_robot_history(robot_frames_t0_to_t, current_index)
        features = pi_context_features(self.video_native, history, self.config)
        if frame_times is None:
            interval = self.config.get("latent_frame_dt", self.config.get(
                "control_dt", getattr(self.interface, "control_dt", None)))
            if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
                raise ValueError("frame_times or a positive latent_frame_dt/control_dt from the training registry are required")
            frame_times = torch.arange(history.shape[2], dtype=torch.float64) * interval
        else:
            if not isinstance(frame_times, Tensor) or frame_times.ndim != 1:
                raise ValueError("frame_times must be a floating [F] tensor")
            if current_index is not None:
                frame_times = frame_times[:current_index + 1]
        _, patch_h, patch_w = self.video_native.patch_size
        coordinates = patch_grid_coordinates((history.shape[3] // patch_h, history.shape[4] // patch_w))
        if (not isinstance(language, Tensor) or language.ndim != 3 or language.shape[0] != 1
                or min(language.shape) < 1 or language.shape[-1] != self.native.config.text_dim
                or not language.is_floating_point() or not torch.isfinite(language).all()):
            raise ValueError("language must be finite pure-text embeddings [1,L,text_dim]")
        with autocast_for(self.native):
            projection = self.native.condition_embedder_action.text_embedder
            language_hidden = projection(language.to(projection.linear_1.weight))
            conditions = self.interface.conditions(features, state, language_hidden, goal, frame_times, coordinates)
            return goal_action_sample(self.native, conditions, self.action_shape, self.actions_mask,
                                      self.generator, steps=self.config.get("action_sampling_steps", 4),
                                      shift=self.config.get("action_sampling_shift", 1.))
