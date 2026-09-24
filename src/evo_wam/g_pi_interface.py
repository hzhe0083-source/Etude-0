"""Goal translation and robot-only goal-conditioned action interfaces."""

from __future__ import annotations

import math
from copy import deepcopy

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
    """Demo-only intent queries followed by scene-conditioned goal queries."""

    def __init__(self, native_dim: int, d_z: int, state_dim: int, effectors: int,
                 k_z: int = 16, dim: int = 768, num_heads: int = 8,
                 num_layers: int = 2, translation_scale: float = 1., use_state: bool = True,
                 num_intent_tokens: int = 4, num_intent_layers: int | None = None,
                 intent_mode: str = "connected"):
        super().__init__()
        if num_intent_layers is None:
            num_intent_layers = num_layers
        if any(type(value) is not int or value < 1 for value in
               (native_dim, d_z, state_dim, effectors, k_z, dim, num_heads, num_layers,
                num_intent_tokens, num_intent_layers)):
            raise ValueError("decoder dimensions, effectors, tokens, heads and layers must be positive integers")
        if dim % num_heads:
            raise ValueError("decoder dim must be divisible by num_heads")
        if type(use_state) is not bool:
            raise ValueError("use_state must be boolean")
        if intent_mode not in ("connected", "independent", "regression_only"):
            raise ValueError("intent_mode must be connected, independent or regression_only")
        self.native_dim, self.d_z, self.state_dim = native_dim, d_z, state_dim
        self.k_z, self.dim, self.effectors = k_z, dim, effectors
        self.num_intent_tokens, self.intent_mode = num_intent_tokens, intent_mode
        self.translation_scale, self.use_state = _translation_scale(translation_scale), use_state
        self.queries = nn.Parameter(torch.randn(k_z + effectors, dim) / math.sqrt(dim))
        self.memory_projection = nn.Sequential(nn.Linear(native_dim, dim), nn.LayerNorm(dim))
        self.state_projection = nn.Linear(state_dim, dim) if use_state else None
        self.blocks = nn.ModuleList(_GoalDecoderBlock(dim, num_heads) for _ in range(num_layers))
        self.output_norm = nn.LayerNorm(dim)
        self.z_head = nn.Linear(dim, d_z)
        self.pose_decoder = _PoseDecoder(dim, effectors, num_heads, self.translation_scale, effectors)
        if intent_mode != "regression_only":
            self.intent_queries = nn.Parameter(torch.randn(num_intent_tokens, dim) / math.sqrt(dim))
            self.intent_projection = nn.Sequential(nn.Linear(native_dim, dim), nn.LayerNorm(dim))
            self.intent_blocks = nn.ModuleList(_GoalDecoderBlock(dim, num_heads)
                                               for _ in range(num_intent_layers))
            self.intent_norm = nn.LayerNorm(dim)
        if intent_mode == "connected":
            self.intent_position = nn.Parameter(torch.randn(1, num_intent_tokens, dim) / math.sqrt(dim))

    def _validate_features(self, features: Tensor, name: str):
        if (not isinstance(features, Tensor) or features.ndim != 3 or min(features.shape) < 1
                or features.shape[-1] != self.native_dim or not features.is_floating_point()
                or not torch.isfinite(features).all()):
            raise ValueError(f"G {name} features must be finite floating [B,S,native_dim]")

    def encode_intent(self, demo_features: Tensor) -> Tensor:
        if self.intent_mode == "regression_only":
            raise ValueError("regression_only has no intent queries")
        self._validate_features(demo_features, "demonstration")
        memory = self.intent_projection(demo_features.detach().to(self.intent_queries))
        tokens = self.intent_queries[None].expand(demo_features.shape[0], -1, -1)
        for block in self.intent_blocks:
            tokens = block(tokens, memory)
        return self.intent_norm(tokens)

    def decode_goal(self, u: Tensor | None, robot_features: Tensor, state: Tensor, *,
                    demo_features: Tensor | None = None) -> dict[str, Tensor]:
        self._validate_features(robot_features, "robot")
        batch = robot_features.shape[0]
        if (not isinstance(state, Tensor) or state.shape != (batch, self.state_dim)
                or not state.is_floating_point() or not torch.isfinite(state).all()):
            raise ValueError("state must be finite floating [B,state_dim]")
        robot_memory = self.memory_projection(robot_features.detach().to(self.queries))
        if self.intent_mode == "connected":
            if (not isinstance(u, Tensor) or u.shape != (batch, self.num_intent_tokens, self.dim)
                    or not u.is_floating_point() or not torch.isfinite(u).all()):
                raise ValueError("u must be finite floating [B,num_intent_tokens,dim]")
            if demo_features is not None:
                raise ValueError("connected decoder direct demo memory is disabled; pass intent only through u")
            # Cross-attention alone is invariant to permuting its memory keys.
            memory = torch.cat((u.to(robot_memory) + self.intent_position, robot_memory), dim=1)
        else:
            self._validate_features(demo_features, "demonstration")
            if demo_features.shape[0] != batch:
                raise ValueError("demonstration and robot features must share a batch")
            memory = torch.cat((self.memory_projection(demo_features.detach().to(self.queries)),
                                robot_memory), dim=1)
        if self.state_projection is not None:
            memory = torch.cat((memory, self.state_projection(state.detach().to(self.queries))[:, None]), dim=1)
        tokens = self.queries[None].expand(batch, -1, -1)
        for block in self.blocks:
            tokens = block(tokens, memory)
        tokens = self.output_norm(tokens)
        return {"z": normalize_z(self.z_head(tokens[:, :self.k_z])),
                **self.pose_decoder(tokens[:, self.k_z:], per_effector=True)}

    def forward(self, demo_features: Tensor, robot_features: Tensor, state: Tensor) -> dict:
        self._validate_features(demo_features, "demonstration")
        u = None if self.intent_mode == "regression_only" else self.encode_intent(demo_features)
        prediction = self.decode_goal(u, robot_features, state,
            demo_features=None if self.intent_mode == "connected" else demo_features)
        return {**prediction, "u": u}


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
                 translation_max_m: float | None = None, rotation_max_deg: float | None = None,
                 candidate_separation_m: float | None = None,
                 generator: torch.Generator | None = None) -> dict[str, Tensor]:
    """Detached hindsight goal noise with explicit spatial/angular support."""
    from .g_pi_noise import perturb_goal_bounded
    return perturb_goal_bounded(goal, z_std=z_std, translation_std=translation_std,
        rotation_std=rotation_std, gripper_std=gripper_std, translation_max_m=translation_max_m,
        rotation_max_deg=rotation_max_deg, candidate_separation_m=candidate_separation_m,
        generator=generator)


class PiGoalInterface(GoalInterface):
    """Recurrent LIT workspace with semantic [language, state, z, pose] memory."""

    def __init__(self, native_dim: int, feature_dim: int, state_dim: int, effectors: int,
                 d_z: int, k_z: int = 16, **kwargs):
        if any(type(value) is not int or value < 1 for value in (d_z, k_z)):
            raise ValueError("d_z and k_z must be positive integers")
        super().__init__(native_dim, feature_dim, state_dim, effectors, **kwargs)
        self.d_z, self.k_z = d_z, k_z
        # The prior encoder and semantic subgoal encoder never share parameters.
        self.endpoint_encoder = deepcopy(self.goal_encoder)
        # Every effector query reads the complete final workspace, not one key.
        self.pose_decoder = _PoseDecoder(self.dim, effectors, self.num_heads,
                                         self.translation_scale, self.num_tokens)
        self.z_projection = nn.Linear(d_z, native_dim)
        self.z_position = nn.Parameter(torch.randn(1, k_z, native_dim) / math.sqrt(native_dim))

    def encode_endpoint(self, poses: Tensor, gripper: Tensor) -> Tensor:
        """Encode the clean executed-block endpoint for the image-free prior."""
        validate_goal_poses(poses)
        if poses.shape[1] != self.effectors:
            raise ValueError("block endpoint poses must contain the configured number of effectors")
        validate_gripper(gripper, poses.shape[:2], "block endpoint gripper")
        value = poses.detach().to(self.endpoint_encoder[0].weight)
        rotation = value[..., :3, :2].transpose(-1, -2).flatten(-2)
        endpoint = torch.cat((value[..., :3, 3] / self.translation_scale, rotation,
                              gripper.detach().to(value).unsqueeze(-1)), -1)
        return self.endpoint_encoder(endpoint.flatten(1))

    def endpoint_conditions(self, poses: Tensor, gripper: Tensor, state: Tensor,
                            language_hidden: Tensor) -> list[Tensor]:
        """Stage 1 reads [language,state,E_eta(q)] without images or subgoals."""
        tokens = self.encode_endpoint(poses, gripper)
        condition = self.condition(tokens, state, language_hidden)
        return [condition] * self.num_layers

    def decode_endpoint(self, tokens: Tensor) -> dict[str, Tensor]:
        """Diagnostic/auxiliary readout from final visual tokens only."""
        return self.pose_decoder(self._tokens(tokens))

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
        return torch.cat((semantic, self.z_projection(z.detach().to(self.z_projection.weight))
                          + self.z_position,
                          self.condition_adapter(poses)), dim=1)

    def conditions(self, features: dict[int, Tensor], state: Tensor, language_hidden: Tensor,
                   goal: dict, frame_times: Tensor, patch_coordinates: Tensor, *,
                   return_tokens: bool = False) -> list[Tensor] | tuple[list[Tensor], Tensor]:
        if type(return_tokens) is not bool:
            raise ValueError("return_tokens must be Boolean")
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
        return (conditions, tokens) if return_tokens else conditions

    def stage2_conditions(self, features: dict[int, Tensor], state: Tensor, language_hidden: Tensor,
                          goal: dict, frame_times: Tensor, patch_coordinates: Tensor) -> tuple[list[Tensor], dict]:
        """Keep the auxiliary readout inside the same FSDP forward scope."""
        # Bypass a separately registered conditions wrapper while inside this
        # FSDP scope; its post-forward hook must not reshard before the readout.
        conditions, tokens = PiGoalInterface.conditions(self, features, state, language_hidden, goal,
                                                       frame_times, patch_coordinates, return_tokens=True)
        return conditions, self.decode_endpoint(tokens)


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
        demo_route = config.get("demo_route", "one_way")
        if demo_route not in ("one_way", "via_u_only"):
            raise ValueError("demo_route must be one_way or via_u_only")
        if demo_route == "via_u_only" and goal_decoder.intent_mode != "connected":
            raise ValueError("via_u_only requires a connected intent decoder")
        # The shared base is owned by the system, never by G's optimizer/state.
        object.__setattr__(self, "native", native)
        self.goal_decoder = goal_decoder
        self.config, self.feature_layer = dict(config), feature_layer
        self._demo_cache, self._demo_reference = None, None
        from .g_pi_context import RobotPrefixCache
        self._prefix_cache = RobotPrefixCache("g")

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def clear_demo_cache(self):
        self._demo_cache, self._demo_reference = None, None
        self.clear_context_cache()

    def clear_context_cache(self):
        self._prefix_cache.clear()

    @torch.no_grad()
    def cache_demo(self, demo: Tensor):
        from .g_pi_context import build_demo_cache
        self.clear_context_cache()
        self._demo_cache = build_demo_cache(self.native, demo, self.config)
        self._demo_reference = demo.detach().clone()
        return self._demo_cache

    @torch.no_grad()
    def predict(self, demo: Tensor, robot_frames_t0_to_t: Tensor, state: Tensor, *,
                current_index: int | None = None, use_cache: bool = True,
                use_prefix_cache: bool = False) -> dict[str, Tensor]:
        from .g_pi_context import split_g_context_features
        from .goal_training import autocast_for
        if use_cache and (self._demo_reference is None or not torch.equal(demo, self._demo_reference)):
            self.cache_demo(demo)
        with autocast_for(self.native):
            demonstration, robot = split_g_context_features(
                self.native, demo, robot_frames_t0_to_t, self.config,
                demo_cache=self._demo_cache if use_cache else None, current_index=current_index,
                prefix_cache=self._prefix_cache if use_prefix_cache else None)
            result = self.goal_decoder(demonstration[self.feature_layer], robot[self.feature_layer], state)
            return {name: value for name, value in result.items() if name != "u"}

    @torch.no_grad()
    def encode_intent(self, demo: Tensor, *, use_cache: bool = True) -> Tensor:
        from .g_pi_context import demo_context_features
        from .goal_training import autocast_for
        if use_cache and (self._demo_reference is None or not torch.equal(demo, self._demo_reference)):
            self.cache_demo(demo)
        with autocast_for(self.native):
            features = demo_context_features(self.native, demo, self.config,
                                             demo_cache=self._demo_cache if use_cache else None)
            return self.goal_decoder.encode_intent(features[self.feature_layer])

    @torch.no_grad()
    def predict_from_intent(self, demo: Tensor, robot_frames_t0_to_t: Tensor, state: Tensor,
                            u: Tensor, *, demo_route: str | None = None,
                            current_index: int | None = None, use_cache: bool = True) -> dict[str, Tensor]:
        """Offline intervention: replace ordered u slots and optionally remove the native demo path."""
        from .g_pi_context import split_g_context_features
        from .goal_training import autocast_for
        if self.goal_decoder.intent_mode != "connected":
            raise ValueError("intent intervention requires connected goal decoder")
        if use_cache and (self._demo_reference is None or not torch.equal(demo, self._demo_reference)):
            self.cache_demo(demo)
        config = dict(self.config)
        if demo_route is not None:
            config["demo_route"] = demo_route
        with autocast_for(self.native):
            _, robot = split_g_context_features(self.native, demo, robot_frames_t0_to_t, config,
                demo_cache=self._demo_cache if use_cache else None, current_index=current_index)
            return self.goal_decoder.decode_goal(u, robot[self.feature_layer], state)


    @torch.no_grad()
    def diagnose_intent(self, demo: Tensor, robot_frames_t0_to_t: Tensor, state: Tensor, *,
                        replacement_u: Tensor | None = None, current_index: int | None = None,
                        use_cache: bool = True) -> dict[str, dict[str, Tensor]]:
        """Intervene on u at fixed robot memory, then remove the native demo path."""
        from .g_pi_context import pi_context_features, split_g_context_features
        from .goal_training import autocast_for
        if self.goal_decoder.intent_mode != "connected":
            raise ValueError("intent diagnostic requires connected goal decoder")
        if use_cache and (self._demo_reference is None or not torch.equal(demo, self._demo_reference)):
            self.cache_demo(demo)
        with autocast_for(self.native):
            demonstration, robot = split_g_context_features(self.native, demo, robot_frames_t0_to_t,
                self.config, demo_cache=self._demo_cache if use_cache else None, current_index=current_index)
            u = self.goal_decoder.encode_intent(demonstration[self.feature_layer])
            robot_memory = robot[self.feature_layer]
            changed = u.flip(1) if replacement_u is None else replacement_u
            result = {"baseline": self.goal_decoder.decode_goal(u, robot_memory, state),
                      "u_permuted" if replacement_u is None else "u_replaced":
                          self.goal_decoder.decode_goal(changed, robot_memory, state)}
            isolated = pi_context_features(self.native, robot_frames_t0_to_t, self.config,
                                           current_index=current_index)
            result["robot_without_demo"] = self.goal_decoder.decode_goal(u, isolated[self.feature_layer], state)
            return result


class PiGoalPolicy(nn.Module):
    """Offline policy; its public and internal interfaces never accept a demo."""

    def __init__(self, native, interface: PiGoalInterface, config: dict, *,
                 action_shape, actions_mask: Tensor, seed: int = 0, video_native=None):
        super().__init__()
        from .g_pi_context import assert_frozen_base
        if video_native is not None and video_native is not native:
            raise ValueError("G, E and pi must share the same native object")
        assert_frozen_base(native)
        if (not isinstance(action_shape, (list, tuple, torch.Size)) or len(action_shape) != 5
                or any(type(size) is not int or size < 1 for size in action_shape)
                or action_shape[0] != 1 or action_shape[1] != native.config.action_dim or action_shape[-1] != 1):
            raise ValueError("action_shape must be positive native [1,A,F,N,1]")
        if not getattr(native, "_goal_action_interface", False):
            raise ValueError("install the goal action interface before constructing the policy")
        if interface.native_dim != native.inner_dim or interface.num_layers != len(native.blocks):
            raise ValueError("policy and shared native must share widths and layer counts")
        from .zerowam import action_mask_for
        if not action_mask_for(actions_mask, torch.empty(action_shape)).any():
            raise ValueError("at least one action channel must be active")
        if type(seed) is not int or seed < 0:
            raise ValueError("sampling seed must be a nonnegative integer")
        self.native, self.interface = native, interface
        self.config, self.action_shape = dict(config), tuple(action_shape)
        self.register_buffer("actions_mask", actions_mask.detach().clone())
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.video_native.eval()
        from .g_pi_context import RobotPrefixCache
        self._prefix_cache = RobotPrefixCache("pi")

    def clear_context_cache(self):
        self._prefix_cache.clear()

    @property
    def video_native(self):
        return self.native

    def train(self, mode: bool = True):
        super().train(mode)
        self.video_native.eval()
        return self

    def _read_conditions(self, robot_frames_t0_to_t: Tensor, state: Tensor, language: Tensor | None,
                         goal: dict | None, *, current_index: int | None = None,
                         frame_times: Tensor | None = None, use_prefix_cache: bool = False,
                         return_tokens: bool = False):
        from .g_pi_context import pi_context_features, truncate_robot_history
        from .goal_training import autocast_for
        if not isinstance(goal, dict):
            raise ValueError("pi requires goal; when language is omitted, pass goal as a keyword")
        history = truncate_robot_history(robot_frames_t0_to_t, current_index)
        features = pi_context_features(self.video_native, history, self.config,
            prefix_cache=self._prefix_cache if use_prefix_cache else None)
        if frame_times is None:
            interval = self.config.get("latent_frame_dt", getattr(self.interface, "latent_frame_dt", None))
            if interval is None:
                control_dt = self.config.get("control_dt", getattr(self.interface, "control_dt", None))
                if type(control_dt) in (int, float) and math.isfinite(control_dt) and control_dt > 0:
                    interval = self.action_shape[3] * control_dt
            if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
                raise ValueError("frame_times or latent_frame_dt=N*control_dt from the training registry are required")
            frame_times = torch.arange(history.shape[2], dtype=torch.float64) * interval
        else:
            if not isinstance(frame_times, Tensor) or frame_times.ndim != 1:
                raise ValueError("frame_times must be a floating [F] tensor")
            if current_index is not None:
                frame_times = frame_times[:current_index + 1]
        _, patch_h, patch_w = self.video_native.patch_size
        coordinates = patch_grid_coordinates((history.shape[3] // patch_h, history.shape[4] // patch_w))
        if language is None:
            if not hasattr(self.native, "g_pi_empty_text"):
                raise ValueError("pi without language requires the pretrained empty prompt embedding")
            language = self.native.g_pi_empty_text
        if (not isinstance(language, Tensor) or language.ndim != 3 or language.shape[0] != 1
                or min(language.shape) < 1 or language.shape[-1] != self.native.config.text_dim
                or not language.is_floating_point() or not torch.isfinite(language).all()):
            raise ValueError("language must be finite pure-text embeddings [1,L,text_dim]")
        with autocast_for(self.native):
            projection = self.native.condition_embedder_action.text_embedder
            language_hidden = projection(language.to(projection.linear_1.weight))
            return self.interface.conditions(features, state, language_hidden, goal, frame_times, coordinates,
                                             return_tokens=return_tokens)

    @torch.no_grad()
    def diagnose_endpoint(self, robot_frames_t0_to_t: Tensor, state: Tensor, language: Tensor | None = None,
                          goal: dict | None = None, *, current_index: int | None = None,
                          frame_times: Tensor | None = None, use_prefix_cache: bool = False) -> dict:
        """Read the inferred clean block endpoint without sampling an action."""
        from .goal_training import autocast_for
        _, tokens = self._read_conditions(robot_frames_t0_to_t, state, language, goal,
            current_index=current_index, frame_times=frame_times, use_prefix_cache=use_prefix_cache,
            return_tokens=True)
        with autocast_for(self.native):
            return self.interface.decode_endpoint(tokens)

    @torch.no_grad()
    def predict(self, robot_frames_t0_to_t: Tensor, state: Tensor, language: Tensor | None = None,
                goal: dict | None = None, *,
                current_index: int | None = None, frame_times: Tensor | None = None,
                use_prefix_cache: bool = False, return_endpoint: bool = False) -> Tensor | dict:
        from .goal_training import autocast_for
        if type(return_endpoint) is not bool:
            raise ValueError("return_endpoint must be Boolean")
        result = self._read_conditions(robot_frames_t0_to_t, state, language, goal,
            current_index=current_index, frame_times=frame_times, use_prefix_cache=use_prefix_cache,
            return_tokens=return_endpoint)
        conditions, tokens = result if return_endpoint else (result, None)
        with autocast_for(self.native):
            actions = goal_action_sample(self.native, conditions, self.action_shape, self.actions_mask,
                                         self.generator, steps=self.config.get("action_sampling_steps", 4),
                                         shift=self.config.get("action_sampling_shift", 1.))
            return ({"actions": actions, "endpoint": self.interface.decode_endpoint(tokens)}
                    if return_endpoint else actions)
