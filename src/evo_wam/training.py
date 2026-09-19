"""Stage objectives and optimizer steps for the existing Evo-WAM modules.

Robot denoising inputs are constructed once outside the trainer and reused by
both view branches. Sampling accepts only noise plus actual history, never a
teacher-forced training dictionary. Native Zero-WAM currently packs B=1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

import torch
from torch import Tensor, nn

from .contracts import PhysicalOutcome, TaskRequirement
from .losses import paired_dropout_disabled
from .losses import aggregate_fields, masked_bce, masked_mean, paired_js
from .models import (CausalEffectPredictor, EffectReader, GoalTokens,
                     RequirementCodec, TemporalInteractionHead,
                     decoded_requirement_loss, physical_prediction_loss)
from .zerowam import TaskConditions, ZeroWAMAdapter, unpack_velocity


@dataclass(frozen=True)
class LossWeights:
    requirement: float = 1.0
    latent: float = 1.0
    execution: float = 1.0
    next_video: float = 1.0
    native_action: float = 1.0
    ifp: float = 1.0
    interaction: float = 1.0
    cv: float = 0.0
    physical: float = 1.0

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0 for value in vars(self).values()):
            raise ValueError("loss weights must be finite and nonnegative")


@dataclass
class TrainingBatch:
    # Pure observation features; never take these from task-conditioned WAM.
    entity_features: Tensor                  # [1,N,De]
    entity_history: Tensor                   # [1,L,N,De]
    proprio_history: Tensor                  # [1,L,Dp]
    embodiment: Tensor                       # [1,Db]
    null_text: Tensor                        # native encoding of empty text
    native_inputs: dict                      # one shared denoising payload
    conditional: bool = True
    requirements: TaskRequirement | None = None
    demonstrations: tuple[Tensor, ...] = ()
    outcome: PhysicalOutcome | None = None
    physical_actions: Tensor | None = None   # [1,H,Da], actually executed
    entity_patch_weights: Tensor | None = None  # fixed observation-based [1,N,S]
    sample_noise: Tensor | None = None       # independent video generation noise
    noisy_actions: Tensor | None = None      # shared teacher/student input
    action_timestep: Tensor | float | None = None
    execution_valid: Tensor | None = None    # bool, exactly noisy_actions shape
    sample_kwargs: dict = field(default_factory=dict)
    # Independent evidence per view; never replace these by their intersection.
    per_view_valid: tuple[Mapping[str, Tensor], ...] | None = None
    category_groups: Mapping[str, Tensor] = field(default_factory=dict)

    def validate(self) -> None:
        if self.entity_features.ndim != 3 or self.entity_features.shape[0] != 1:
            raise ValueError("native training currently requires entity_features [1,N,D]")
        if self.entity_history.ndim != 4 or self.entity_history.shape[0] != 1:
            raise ValueError("entity_history must be [1,L,N,D]")
        if self.entity_features.shape[1:] != self.entity_history.shape[2:]:
            raise ValueError("current entities and history must share their entity feature table")
        for name in ("entity_features", "entity_history", "proprio_history", "embodiment", "null_text"):
            value = getattr(self, name)
            if not value.is_floating_point() or not torch.isfinite(value).all() or value.requires_grad:
                raise ValueError(f"{name} must be finite, detached observation/input features")
        if self.null_text.ndim != 3 or self.null_text.shape[0] != 1:
            raise ValueError("null_text must be a native [1,N,text_dim] empty-text encoding")
        if self.native_inputs.get("icl_latent_dict") is not None:
            raise ValueError("raw ICL bypass is forbidden; route demonstrations through the reader")
        # No caller-supplied text may bypass the explicit empty/goal conditions.
        if "text_emb" in self.native_inputs or "encoder_seq_ids" in self.native_inputs:
            raise ValueError("native text conditions are installed only by the adapter")
        if set(self.sample_kwargs) - {"history", "steps", "shift", "frame_id", "grid_id", "rope_offset"}:
            raise ValueError("sampling accepts only actual history and sampler configuration")
        for name in ("sample_noise", "noisy_actions"):
            value = getattr(self, name)
            if value is not None and (value.requires_grad or not torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite and detached")
        if self.requirements is not None:
            self.requirements.validate()
            if self.requirements.current.entity_ids.shape != self.entity_features.shape[:2]:
                raise ValueError("requirements and pure observation entity table differ")
        if self.per_view_valid is not None:
            if self.requirements is None or len(self.per_view_valid) != len(self.demonstrations):
                raise ValueError("per_view_valid must match demonstrations with requirement labels")
            shapes = {f"{part}.{name}": getattr(self.requirements, part).label_valid[name]
                      for part in ("current", "remaining")
                      for name in getattr(self.requirements, part).label_valid}
            if self.outcome is not None:
                shapes.update({name: self.outcome.label_valid[name] for name in ("relations", "events")})
            for evidence in self.per_view_valid:
                if set(evidence) != set(shapes):
                    raise ValueError("per_view_valid must explicitly annotate every requirement and interaction field")
                for name, reference in shapes.items():
                    value = evidence[name]
                    if value.dtype != torch.bool or value.shape != reference.shape or value.device != reference.device:
                        raise ValueError(f"per_view_valid[{name}] must be a data-side bool mask with the label shape/device")
        if self.outcome is not None:
            self.outcome.validate()
            if self.outcome.entity_ids.shape != self.entity_features.shape[:2]:
                raise ValueError("outcomes and pure observation entity table differ")
            if self.requirements is not None and not torch.equal(
                    self.outcome.entity_ids, self.requirements.current.entity_ids):
                raise ValueError("physical labels and requirements must share entity IDs")


def _native_loss(prediction: Tensor, stream: dict, patch_size) -> Tensor:
    """Upstream flow target order, with fixed scheduler weights and data masks."""
    target = stream["targets"]
    if target.ndim != 5 or not target.is_floating_point() or target.requires_grad:
        raise ValueError("native targets must be detached [B,C,F,H,W]")
    prediction = unpack_velocity(prediction, target.shape, patch_size)
    valid = torch.ones_like(target, dtype=torch.bool)
    for key in ("valid_mask", "actions_mask"):
        if key in stream:
            mask = stream[key]
            if mask.requires_grad or not ((mask == 0) | (mask == 1)).all():
                raise ValueError(f"{key} must be a fixed binary annotation mask")
            valid = valid & torch.broadcast_to(mask.bool(), target.shape)
    safe_target = torch.where(valid, target.float(), 0)
    safe_prediction = torch.where(valid, prediction.float(), 0)
    error = (safe_prediction - safe_target).square()
    weight = stream.get("training_weight")
    if weight is not None:
        if (weight.requires_grad or tuple(weight.shape) != (target.shape[0], target.shape[2])
                or not torch.isfinite(weight).all() or (weight < 0).any()):
            raise ValueError("training_weight must be fixed, finite nonnegative [B,F]")
        error = error * weight[:, None, :, None, None].float()
    return masked_mean(error, valid).loss


class EvoTrainer(nn.Module):
    """One optimizer over the parameters selected by the current training stage.

    ``set_stage`` deliberately rebuilds AdamW: moments from interface fitting do
    not silently carry into the distinct reader/joint optimization problems.
    Save/reload optimizer state only when resuming the *same* stage.
    """

    def __init__(self, adapter: ZeroWAMAdapter, codec: RequirementCodec,
                 reader: EffectReader, physical: CausalEffectPredictor,
                 interaction: TemporalInteractionHead | None = None, *,
                 stage: str = "interface", weights: LossWeights | None = None,
                 enable_ifp: bool = True, enable_interaction: bool = True,
                 ifp_weights: tuple[float, ...] | None = None,
                 cv_field_weights: Mapping[str, float] | None = None,
                 learning_rate: float = 1e-4, weight_decay: float = 0.0,
                 max_grad_norm: float = 1.0, exec_start_step: int = 100):
        super().__init__()
        self.adapter, self.codec, self.reader = adapter, codec, reader
        self.physical, self.interaction = physical, interaction
        self.weights = weights or LossWeights()
        self.enable_ifp, self.enable_interaction = enable_ifp, enable_interaction
        self.ifp_weights = ifp_weights
        self.cv_field_weights = dict(cv_field_weights or {})
        if enable_interaction and interaction is None:
            raise ValueError("enabled interaction supervision requires its existing shared head")
        if ifp_weights is not None and any(not math.isfinite(w) or w < 0 for w in ifp_weights):
            raise ValueError("IFP weights must be finite and nonnegative")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight decay must be finite and nonnegative")
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("gradient clipping norm must be finite and positive")
        if type(exec_start_step) is not int or exec_start_step < 0:
            raise ValueError("exec_start_step must be a nonnegative integer of successful stage updates")
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.exec_start_step = exec_start_step
        self.max_grad_norm, self.updates = max_grad_norm, 0
        self.set_stage(stage)

    def set_stage(self, stage: str) -> None:
        if stage not in {"interface", "reader", "joint"}:
            raise ValueError("stage must be interface, reader or joint")
        self.requires_grad_(False)
        self.zero_grad(set_to_none=True)
        self.adapter.set_stage(stage)
        self.codec.requires_grad_(stage == "interface")
        self.physical.requires_grad_(stage == "interface")
        self.reader.requires_grad_(stage != "interface")
        if self.interaction is not None:
            self.interaction.requires_grad_(stage == "joint" and self.enable_interaction)
        self.stage = stage
        self.updates = 0
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("stage has no trainable parameters")
        self.optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate,
                                          weight_decay=self.weight_decay)

    @property
    def execution_enabled(self) -> bool:
        """Count successful stage updates; reader-only null batches do not update."""
        return (self.stage != "interface" and self.weights.execution > 0
                and self.updates >= self.exec_start_step)

    def _true_goals(self, batch: TrainingBatch) -> GoalTokens:
        if batch.requirements is None:
            raise ValueError("conditional training requires current and remaining labels")
        outputs = [self.codec.encode(requirement, batch.entity_features)
                   for requirement in (batch.requirements.current, batch.requirements.remaining)]
        return GoalTokens(*outputs)

    def _decode(self, goals: GoalTokens, batch: TrainingBatch):
        return tuple(self.codec.decode(tokens, batch.entity_features,
                                       requirement.step_offsets, requirement.entity_ids)
                     for tokens, requirement in zip((goals.current, goals.remaining),
                         (batch.requirements.current, batch.requirements.remaining)))

    def _view_identifiable(self, batch: TrainingBatch, evidence: Mapping[str, Tensor]) -> bool:
        """Conservative data-only gate for a unique latent/execution target.

        A physical future is conditioned on current *and* remaining goals, so
        an identifiable local grasp cannot authorize a hidden destination.
        Geometry controls use only their own explicit interface content.
        """
        names = ("geometry",) if self.codec.interface == "geometry" else ("geometry", "relations", "events")
        for part in ("current", "remaining"):
            target = getattr(batch.requirements, part)
            if self.codec.interface == "full" and not bool(target.semantics_known.all()):
                return False
            used = torch.zeros_like(target.binding, dtype=torch.bool)
            for name in names:
                mask = target.requirement_mask[name]
                used |= (mask.any((1, 3)) if name == "geometry" else
                         mask.any((1, 3, 4)) | mask.any((1, 2, 4)))
            if (used & ((target.binding < 0) | ~target.label_valid["binding"]
                        | ~evidence[f"{part}.binding"])).any():
                return False
            value_fields = {"geometry": "geometry", "geometry_tolerance": "geometry"}
            if self.codec.interface == "full":
                value_fields.update({"relations": "relations", "events": "events", "event_windows": "events"})
            for name, mask_name in value_fields.items():
                # Mask existence itself is part of G's input: evidence must
                # identify it for active roles, even where the target is false.
                support = (used[:, None, :, None] if mask_name == "geometry" else
                           used[:, None, :, None, None] & used[:, None, None, :, None])
                if (support & ~evidence[f"{part}.{name}"]).any():
                    return False
                if (target.requirement_mask[mask_name] & ~target.label_valid[name]).any():
                    return False
            if self.codec.interface == "full" and not bool(evidence[f"{part}.event_precedence"].all()):
                return False
        return True

    def _native(self, batch: TrainingBatch, conditions: TaskConditions,
                *, include_action: bool, include_interaction: bool,
                interaction_valid: Mapping[str, Tensor] | None = None):
        # Do not mutate the shared robot payload or noise between view calls.
        inputs = dict(batch.native_inputs)
        if not self.enable_ifp:
            inputs["mcp_latent_dicts"] = []
        output = self.adapter.forward_train(inputs, conditions,
                                            history=batch.sample_kwargs.get("history", ()))
        terms = {}
        if self.weights.next_video:
            terms["next_video"] = _native_loss(output.video, inputs["latent_dict"],
                                                self.adapter.native.patch_size)
        if include_action and self.weights.native_action:
            terms["native_action"] = _native_loss(output.action, inputs["action_dict"], (1, 1, 1))
        if self.enable_ifp and self.weights.ifp:
            streams = inputs.get("mcp_latent_dicts", [])
            if not streams or len(output.mcp) != len(streams):
                raise ValueError("enabled IFP requires one output for each shared future target")
            weights = self.ifp_weights or (1.0,) * len(streams)
            if len(weights) != len(streams):
                raise ValueError("IFP loss weights must match configured future targets")
            terms["ifp"] = sum(weight * _native_loss(pred, stream, self.adapter.native.patch_size)
                               for weight, pred, stream in zip(weights, output.mcp, streams))
        interaction = None
        if include_interaction:
            if batch.outcome is None or batch.entity_patch_weights is None or interaction_valid is None:
                raise ValueError("interaction supervision needs physical labels, per-view evidence and observation patch weights")
            pool = batch.entity_patch_weights
            expected = (1, batch.entity_features.shape[1], output.phi.shape[1])
            present = batch.outcome.entity_ids >= 0
            if (tuple(pool.shape) != expected or pool.requires_grad or not torch.isfinite(pool).all()
                    or (pool < 0).any() or ((pool.sum(-1) <= 0) & present).any()
                    or (pool[~present] != 0).any()):
                raise ValueError("entity patch weights must be fixed, nonnegative [1,N,S] with visible entity support")
            # Native teacher forcing packs multiple future chunks. Interaction
            # queries are relative to the window start, so they may pool only
            # its first prediction chunk, never a later teacher-forced future.
            video = inputs["latent_dict"]["noisy_latents"]
            pt, ph, pw = self.adapter.native.patch_size
            chunk = min(inputs.get("chunk_size", video.shape[2]), video.shape[2])
            prefix_tokens = (chunk // pt) * (video.shape[3] // ph) * (video.shape[4] // pw)
            if prefix_tokens < 1 or (pool[..., prefix_tokens:] != 0).any():
                raise ValueError("interaction pooling must use only the current prediction chunk, not later future tokens")
            pool = torch.where(present[..., None], pool, 0)
            pool = pool / pool.sum(-1, keepdim=True).clamp_min(torch.finfo(pool.dtype).tiny)
            phi = torch.bmm(pool.to(output.phi), output.phi)
            # The large native backbone may be BF16 while this small head is
            # FP32; this differentiable cast must not detach the deployed Phi.
            interaction = self.interaction(phi.to(next(self.interaction.parameters())), batch.outcome.step_offsets)
            terms["interaction"] = aggregate_fields({
                name: masked_bce(interaction[key], getattr(batch.outcome, name),
                                 batch.outcome.label_valid[name] & interaction_valid[name])
                for name, key in (("relations", "relation_logits"), ("events", "event_logits"))
            }).loss
        return terms, interaction

    def _execution(self, batch: TrainingBatch, predicted: TaskConditions,
                   truth: TaskConditions) -> Tensor:
        if batch.sample_noise is None or batch.noisy_actions is None or batch.action_timestep is None:
            raise ValueError("execution distillation needs independent video noise, noisy actions and timestep")
        with torch.no_grad():
            future = self.adapter.sample_video(batch.sample_noise, predicted, **batch.sample_kwargs)
            teacher = self.adapter.action_velocity(batch.noisy_actions, batch.action_timestep,
                                                  truth.detached(), future,
                                                  history=batch.sample_kwargs.get("history", ()))
        # Frozen action parameters still transmit the direct current-goal gradient.
        student = self.adapter.action_velocity(batch.noisy_actions, batch.action_timestep,
                                              predicted, future,
                                              history=batch.sample_kwargs.get("history", ()))
        if student.shape != teacher.shape or student.shape != batch.noisy_actions.shape:
            raise ValueError("action expert must return the same shape as its noisy input")
        valid = batch.execution_valid
        if valid is None:
            valid = torch.ones_like(student, dtype=torch.bool)
        if valid.shape != student.shape or valid.dtype != torch.bool:
            raise ValueError("execution_valid must be a data-side bool mask matching actions")
        error = (torch.where(valid, student.float(), 0) - torch.where(valid, teacher.float(), 0)).square()
        return masked_mean(error, valid).loss

    def _cv(self, first, second, batch: TrainingBatch):
        if batch.per_view_valid is None or len(batch.per_view_valid) != 2:
            raise ValueError("CV requires explicit data-side visibility for both demonstration views")
        left, right = batch.per_view_valid
        fields = {}
        for index, name in enumerate(("current", "remaining")):
            key = f"{name}.binding"
            target = getattr(batch.requirements, name)
            fields[key] = paired_js(first[0][index].binding_logits, second[0][index].binding_logits,
                                   left[key] & target.label_valid["binding"],
                                   right[key] & target.label_valid["binding"], kind="categorical",
                                   category_groups=batch.category_groups.get(key))
        if first[1] is not None and second[1] is not None:
            for name, key in (("relations", "relation_logits"), ("events", "event_logits")):
                fields[name] = paired_js(first[1][key], second[1][key],
                                        left[name] & batch.outcome.label_valid[name],
                                        right[name] & batch.outcome.label_valid[name], kind="bernoulli")
        return aggregate_fields(fields, self.cv_field_weights)

    def objective(self, batch: TrainingBatch) -> dict[str, Tensor]:
        """Return differentiable losses; no optimizer mutation or hidden sampling."""
        batch.validate()
        with paired_dropout_disabled(self):
            return self._objective(batch)

    def _objective(self, batch: TrainingBatch) -> dict[str, Tensor]:
        zero = batch.entity_features.new_zeros((), dtype=torch.float32)
        if not batch.conditional:
            # Deliberately never encode true goals, call the reader or distill a
            # teacher here, even if annotation objects remain in the dataset.
            terms, _ = self._native(batch, TaskConditions(None, None, batch.null_text),
                                    include_action=False, include_interaction=False)
            total = sum((getattr(self.weights, key) * value for key, value in terms.items()), zero)
            return {**terms, "cv_coverage": zero, "total": total}

        if self.stage == "interface":
            truth = self._true_goals(batch)
            goals_by_view = (truth,)
            identifiable_by_view = (True,)
        else:
            if batch.requirements is None:
                raise ValueError("conditional training requires current and remaining labels")
            if len(batch.demonstrations) not in (1, 2):
                raise ValueError("reader/joint conditional training expects one or two demonstration views")
            if batch.per_view_valid is None:
                raise ValueError("reader/joint supervision requires independent per_view_valid annotations")
            identifiable_by_view = tuple(self._view_identifiable(batch, evidence)
                                         for evidence in batch.per_view_valid)
            truth = None
            if any(identifiable_by_view) and (self.weights.latent or self.execution_enabled):
                with torch.no_grad():
                    truth = self._true_goals(batch)
            goals_by_view = tuple(self.reader(demo, batch.entity_history, batch.proprio_history,
                                              batch.embodiment, batch.requirements.current.step_offsets,
                                              batch.requirements.remaining.step_offsets,
                                              entity_present=batch.requirements.current.entity_ids >= 0)
                                  for demo in batch.demonstrations)
        truth_condition = None if truth is None else TaskConditions(truth.current, truth.remaining, batch.null_text)
        views, predictions = [], []
        for index, goals in enumerate(goals_by_view):
            conditions = TaskConditions(goals.current, goals.remaining, batch.null_text)
            decoded = self._decode(goals, batch)
            evidence = None if self.stage == "interface" else batch.per_view_valid[index]
            identifiable = identifiable_by_view[index]
            terms = {}
            if self.weights.requirement:
                terms["requirement"] = sum(decoded_requirement_loss(
                    pred, getattr(batch.requirements, part), self.codec.interface,
                    evidence_valid=None if evidence is None else {
                        name: evidence[f"{part}.{name}"] for name in getattr(batch.requirements, part).label_valid})
                    for part, pred in zip(("current", "remaining"), decoded)) / 2
            if self.stage != "interface" and self.weights.latent:
                terms["latent"] = (((goals.current.float() - truth.current.float()).square().mean()
                                   + (goals.remaining.float() - truth.remaining.float()).square().mean()) / 2
                                  if identifiable else zero)
            interaction = None
            if self.stage != "reader":
                use_interaction = self.stage == "joint" and self.enable_interaction
                native_terms, interaction = self._native(batch, conditions,
                    include_action=self.stage == "interface", include_interaction=use_interaction,
                    interaction_valid=evidence)
                terms.update(native_terms)
            if self.execution_enabled:
                terms["execution"] = self._execution(batch, conditions, truth_condition) if identifiable else zero
            views.append(terms)
            predictions.append((decoded, interaction))

        # Arithmetic mean, including when one view has fewer valid labels.
        result = {key: sum(view[key] for view in views) / len(views) for key in views[0]}
        result["cv_coverage"] = zero
        if self.stage == "joint" and self.weights.cv:
            if len(views) != 2:
                raise ValueError("nonzero CV weight requires exactly two conditional views")
            cv = self._cv(predictions[0], predictions[1], batch)
            result["cv"], result["cv_coverage"] = cv.loss, cv.coverage
        if self.stage == "interface" and self.weights.physical:
            if batch.outcome is None or batch.physical_actions is None:
                raise ValueError("interface physical training needs actually executed actions and outcome labels")
            prediction = self.physical(batch.entity_history, batch.proprio_history,
                                       batch.physical_actions, batch.embodiment,
                                       entity_present=batch.outcome.entity_ids >= 0)
            if int(batch.outcome.step_offsets.max()) > batch.physical_actions.shape[1]:
                raise ValueError("cannot supervise an unexecuted physical future")
            result["physical"] = physical_prediction_loss(prediction, batch.outcome)
        result["total"] = sum((getattr(self.weights, key) * value for key, value in result.items()
                               if key != "cv_coverage"), zero)
        return result

    def train_step(self, batch: TrainingBatch) -> dict[str, float | bool]:
        self.train()
        self.zero_grad(set_to_none=True)
        losses = self.objective(batch)
        total = losses["total"]
        if not torch.isfinite(total):
            raise ValueError("non-finite training objective; optimizer was not stepped")
        # Reader-only null examples have no trainable conditional path. They are
        # measured but cannot update a frozen generator or leak a true goal.
        updated = total.requires_grad
        norm = total.new_zeros(())
        if updated:
            total.backward()
            parameters = [p for p in self.parameters() if p.requires_grad and p.grad is not None]
            norm = nn.utils.clip_grad_norm_(parameters, self.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
            self.updates += 1
        return {**{name: float(value.detach()) for name, value in losses.items()},
                "grad_norm": float(norm), "updated": updated}
