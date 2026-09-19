"""Shared normalized-action policy for file prediction and the RoboTwin bridge."""
from __future__ import annotations

import random
from dataclasses import replace

import torch

from .contracts import TaskRequirement
from .evaluation import Candidate
from .models import GoalTokens
from .zerowam import TaskConditions


class RequirementRejected(ValueError):
    """No sampling or robot command should occur after a rejected requirement."""


class NativePolicy:
    def __init__(self, trainer, null_text, config, *, sampling_steps=4, view=0, diagnostic=False):
        if type(sampling_steps) is not int or sampling_steps < 1:
            raise ValueError("sampling_steps must be positive")
        self.trainer, self.null_text, self.config = trainer, null_text, config
        self.sampling_steps, self.view = sampling_steps, view
        self.diagnostic = diagnostic
        self.source_stage = getattr(trainer, "stage", "reader")

    @classmethod
    def from_artifact(cls, artifact_path, *, checkpoint=None, device="cuda", sampling_steps=4, view=0, diagnostic=False):
        """Load once for a server loop; checkpoints are local paths only."""
        from .cli import build_trainer, restore_run, load_run
        artifact = load_run(artifact_path)
        config = artifact["config"]
        trainer, null = build_trainer(config, checkpoint=checkpoint, tiny_native=artifact["tiny_native"],
                                      device=device, stage="reader")
        restore_run(artifact_path, trainer, config, torch.Generator(), resume=False)
        trainer.eval().requires_grad_(False)
        policy = cls(trainer, null, config, sampling_steps=sampling_steps, view=view, diagnostic=diagnostic)
        policy.source_stage = artifact["stage"]
        return policy

    @torch.no_grad()
    def candidates(self, observation, requirement, rng: random.Random, *, count=1):
        """Oracle requirements and inferred requirements share the exact sampler."""
        from .cli import action_space, move
        if count not in (1, 4):
            raise ValueError("candidate protocol requires one or four candidates")
        trainer, config = self.trainer, self.config
        device = next(trainer.codec.parameters()).device
        observation = move(observation, device)
        ids = observation.entity_ids
        space = action_space({"action_space": observation.action_space}, config["dimensions"]["action_dim"])
        if trainer.action_spaces.get(space["normalization_id"]) != space:
            raise ValueError("observation normalization differs from the trained interface")
        current = torch.tensor(config["current_offsets"], device=device, dtype=torch.int64)
        remaining = torch.tensor(config["remaining_offsets"], device=device, dtype=torch.int64)
        if requirement is None:
            if self.source_stage not in {"reader", "joint"}:
                raise RequirementRejected("demonstration_inference_requires_trained_reader")
            if not self.diagnostic and (not config.get("validation_locked")
                                       or not config["binding_policy"]["validation_locked"]):
                raise RequirementRejected("demonstration_inference_requires_validation_locked_thresholds")
            if not 0 <= self.view < len(observation.demonstrations):
                raise ValueError("demonstration view is absent")
            goals = trainer.reader(observation.demonstrations[self.view], observation.robot_history,
                observation.proprio_history, observation.embodiment, current, remaining, entity_present=ids >= 0)
            binding = config["binding_policy"]
            try:
                parts = {name: trainer.codec.decode(getattr(goals, name), observation.robot_history[:, -1], offsets, ids)
                    .materialize(interface=config["interface"], min_binding_confidence=binding["confidence_threshold"],
                                 min_binding_margin=binding["margin_threshold"])
                    for name, offsets in (("current", current), ("remaining", remaining))}
                requirement = TaskRequirement(**parts).validate()
            except ValueError as error:
                raise RequirementRejected(f"invalid_decoded_requirement: {error}") from error
        else:
            try:
                requirement = move(requirement, device).validate()
            except ValueError as error:
                raise RequirementRejected(f"invalid_oracle_requirement: {error}") from error
            if not torch.equal(requirement.current.entity_ids, ids):
                raise RequirementRejected("oracle_requirement_entity_table_differs")
            if not torch.equal(requirement.current.step_offsets, current) or not torch.equal(requirement.remaining.step_offsets, remaining):
                raise RequirementRejected("oracle_requirement_query_grid_differs")
            if config["interface"] == "geometry":
                parts = []
                for part in (requirement.current, requirement.remaining):
                    masks = {name: value if name == "geometry" else torch.zeros_like(value)
                             for name, value in part.requirement_mask.items()}
                    parts.append(replace(part, requirement_mask=masks,
                        event_precedence=torch.full_like(part.event_precedence, -1),
                        label_valid={**part.label_valid, "event_precedence": torch.ones_like(part.label_valid["event_precedence"])}))
                requirement = TaskRequirement(*parts).validate()
            # Validate before G, which must not invent missing semantic labels.
            if not bool(requirement.semantics_known.all()):
                raise RequirementRejected("oracle_requirement_unknown_or_unresolved")
            goals = GoalTokens(*(trainer.codec.encode(getattr(requirement, name), observation.robot_history[:, -1])
                                for name in ("current", "remaining")))
        if not bool(requirement.current.has_requirement.all()):
            raise RequirementRejected("empty_current_requirement")
        if not bool(requirement.resolved.all()):
            raise RequirementRejected("unresolved_required_binding")
        if not bool(requirement.semantics_known.all()):
            raise RequirementRejected("unknown_requirement_semantics")
        chunk = observation.chunk_size
        horizon = chunk * observation.actions_per_frame
        required_windows = requirement.current.event_windows[requirement.current.requirement_mask["events"]]
        timed = (requirement.current.requirement_mask["geometry"].flatten(2).any(-1)
                 | requirement.current.requirement_mask["relations"].flatten(2).any(-1))
        if (timed & (current[None] > horizon)).any() or (required_windows.numel() and int(required_windows.max()) > horizon):
            raise RequirementRejected("current_requirement_exceeds_candidate_horizon")
        dtype = next(trainer.adapter.native.parameters()).dtype
        observed = observation.robot_latent.to(dtype=dtype)
        history = observation.native_history(dtype=dtype)
        generator = torch.Generator().manual_seed(rng.getrandbits(63))
        shape = (1, observed.shape[1], chunk, *observed.shape[-2:])
        noise = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
        conditions = TaskConditions(goals.current, goals.remaining, self.null_text)
        future = trainer.adapter.sample_video(noise, conditions, history=history, steps=self.sampling_steps,
            shift=config["video_snr_shift"], **observation.sampling_position())
        action_shape = (1, trainer.adapter.native.config.action_dim, chunk, observation.actions_per_frame, 1)
        mask = torch.tensor(space["valid_channels"], device=device)[None, :, None, None, None]
        candidates = []
        for index in range(count):
            initial = torch.randn(action_shape, generator=generator).to(device=device, dtype=dtype)
            action = trainer.adapter.sample_actions(initial, conditions, future, history=history,
                steps=self.sampling_steps, action_mask=mask)
            flattened = action.float().permute(0, 2, 3, 4, 1).reshape(-1, action.shape[1])
            candidates.append(Candidate(str(index), flattened.cpu().tolist()))
        return candidates, requirement

    def oracle_candidate(self, observation, requirement, rng):
        if requirement is None:
            raise ValueError("oracle mode requires explicitly annotated current-state requirements")
        return self.candidates(observation, requirement, rng)[0][0]
