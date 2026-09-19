"""CPU graph/optimizer checks, deliberately not a native Zero-WAM smoke test."""

from dataclasses import replace
import unittest

import torch
from torch import nn

from evo_wam.contracts import EffectRequirement, PhysicalOutcome, TaskRequirement
from evo_wam.models import (CausalEffectPredictor, EffectReader, RequirementCodec,
                            TemporalInteractionHead)
from evo_wam.training import EvoTrainer, LossWeights, TrainingBatch
from evo_wam.zerowam import GeneratedFuture, NativeOutput


def sequence(value):
    return value.permute(0, 2, 3, 4, 1).reshape(1, -1, value.shape[1])


class TinyAdapter(nn.Module):
    """Small actual autograd graph with the native adapter's public calls."""

    def __init__(self):
        super().__init__()
        self.native = nn.Module()
        self.native.patch_size = (1, 1, 1)
        self.native.video = nn.Linear(4, 4)
        self.native.action = nn.Linear(4, 4)
        self.native.mcp = nn.Linear(4, 4)
        self.condition_projection = nn.Linear(4, 4)
        self.dropout = nn.Dropout(0.9)
        self.forward_inputs = []
        self.forward_histories = []
        self.action_calls = []
        self.sample_calls = []
        self.detach_current = False

    def set_stage(self, stage):
        self.requires_grad_(False)
        self.condition_projection.requires_grad_(stage == "interface")
        self.native.video.requires_grad_(stage in {"interface", "joint"})
        self.native.action.requires_grad_(stage == "interface")
        self.native.mcp.requires_grad_(stage == "joint")

    def task(self, conditions, remaining=False):
        if conditions.current is None:
            return conditions.null.mean(1, keepdim=True) * 0
        tokens = conditions.current.detach() if self.detach_current else conditions.current
        current = self.condition_projection(tokens).mean(1, keepdim=True)
        if remaining:
            current = current + self.condition_projection(conditions.remaining).mean(1, keepdim=True)
        return current

    def forward_train(self, inputs, conditions, *, history=()):
        self.forward_inputs.append(inputs)
        self.forward_histories.append(history)
        phi = self.dropout(self.native.video(sequence(inputs["latent_dict"]["noisy_latents"])
                                              + self.task(conditions, remaining=True)))
        action = self.native.action(sequence(inputs["action_dict"]["noisy_latents"])
                                    + self.task(conditions))
        mcp = [self.native.mcp(sequence(stream["noisy_latents"]) + phi)
               for stream in inputs.get("mcp_latent_dicts", [])]
        return NativeOutput(phi, action, mcp, phi)

    def sample_video(self, initial_noise, conditions, **kwargs):
        self.sample_calls.append(torch.is_grad_enabled())
        # Intentionally not decorated: the trainer must disconnect this graph.
        generated = initial_noise + self.native.video(self.task(conditions, True)).mean()
        return GeneratedFuture(generated, conditions.detached(), torch.zeros(1), 2)

    def action_velocity(self, noisy_action, timestep, conditions, future, **kwargs):
        self.action_calls.append((torch.is_grad_enabled(), id(noisy_action), id(timestep), id(future),
                                  future.latents.requires_grad, future.latents.grad_fn))
        output = self.native.action(sequence(noisy_action) + sequence(future.latents)
                                    + self.task(conditions))
        return output.reshape(1, 1, 1, 2, 4).permute(0, 4, 1, 2, 3)


def make_batch():
    ids, offsets = torch.tensor([[7, 9]]), torch.tensor([1, 2])
    geometry = torch.randn(1, 2, 2, 2)
    relations = torch.randint(0, 2, (1, 2, 2, 2, 2)).float()
    events = torch.randint(0, 2, (1, 2, 2, 2, 1)).float()
    values = {"geometry": geometry, "relations": relations, "events": events}
    masks = {name: torch.ones_like(value, dtype=torch.bool) for name, value in values.items()}
    requirement = EffectRequirement(ids, offsets, torch.tensor([[0, 1]]), **values,
        requirement_mask=masks, label_valid={**masks, "binding": torch.ones(1, 2, dtype=torch.bool)})
    remaining = replace(requirement, geometry=geometry + 1)
    outcome = PhysicalOutcome(ids, offsets, **values, label_valid=masks)

    def stream():
        return {"noisy_latents": torch.randn(1, 4, 1, 1, 2),
                "targets": torch.randn(1, 4, 1, 1, 2),
                "timesteps": torch.tensor([[500.]]),
                "training_weight": torch.ones(1, 1)}

    native = {"latent_dict": stream(), "action_dict": stream(), "mcp_latent_dicts": [stream()]}
    demo = torch.randn(1, 3, 5)
    valid = {f"{part}.{name}": torch.ones_like(value, dtype=torch.bool)
             for part in ("current", "remaining") for name, value in requirement.label_valid.items()}
    valid.update({"relations": masks["relations"], "events": masks["events"]})
    return TrainingBatch(
        entity_features=torch.randn(1, 2, 3), entity_history=torch.randn(1, 2, 2, 3),
        proprio_history=torch.randn(1, 2, 2), embodiment=torch.randn(1, 2),
        null_text=torch.zeros(1, 1, 4), native_inputs=native,
        requirements=TaskRequirement(requirement, remaining), demonstrations=(demo, demo.clone()),
        outcome=outcome, physical_actions=torch.randn(1, 2, 4),
        entity_patch_weights=torch.eye(2)[None], sample_noise=torch.randn(1, 4, 1, 1, 2),
        noisy_actions=torch.randn(1, 4, 1, 1, 2), action_timestep=torch.tensor(500.),
        per_view_valid=(valid, {name: value.clone() for name, value in valid.items()}))


def trainer(stage="joint", weights=None, exec_start_step=0):
    return EvoTrainer(TinyAdapter(), RequirementCodec(3, 2, 2, 1, roles=2, token_dim=4),
                      EffectReader(5, 3, 2, 2, roles=2, token_dim=4),
                      CausalEffectPredictor(3, 2, 4, 2, 2, 2, 1, hidden_dim=4),
                      TemporalInteractionHead(4, 2, 1, hidden_dim=4),
                      stage=stage, weights=weights, learning_rate=0.01,
                      exec_start_step=exec_start_step)


def has_gradient(module):
    return any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
               for parameter in module.parameters())


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_reader_warmup_skips_entire_execution_graph_until_successful_updates(self):
        model, batch = trainer("reader", exec_start_step=2), make_batch()
        self.assertFalse(model.execution_enabled)
        for expected_updates in (1, 2):
            report = model.train_step(batch)
            self.assertTrue(report["updated"])
            self.assertNotIn("execution", report)
            self.assertTrue(has_gradient(model.reader))
            self.assertEqual(model.updates, expected_updates)
            self.assertEqual(model.adapter.sample_calls, [])
            self.assertEqual(model.adapter.action_calls, [])
        self.assertTrue(model.execution_enabled)
        self.assertIn("execution", model.train_step(batch))
        self.assertEqual(len(model.adapter.sample_calls), 2)
        self.assertEqual(len(model.adapter.action_calls), 4)

    def test_warmup_ignores_null_attempts_and_phase_switch_resets_counter(self):
        model, batch = trainer("reader", exec_start_step=1), make_batch()
        self.assertFalse(model.train_step(replace(batch, conditional=False))["updated"])
        self.assertEqual(model.updates, 0)
        model.train_step(batch)
        self.assertTrue(model.execution_enabled)
        model.set_stage("joint")
        self.assertEqual(model.updates, 0)
        self.assertFalse(model.execution_enabled)
        # Same-stage resume restores this recorded successful-update counter.
        model.updates = 1
        self.assertTrue(model.execution_enabled)

    def test_execution_zero_ablation_never_samples_after_warmup(self):
        model, batch = trainer("reader", LossWeights(execution=0), exec_start_step=0), make_batch()
        model.updates = 10000
        report = model.train_step(batch)
        self.assertNotIn("execution", report)
        self.assertTrue(report["updated"])
        self.assertEqual(model.adapter.sample_calls, [])
        self.assertEqual(model.adapter.action_calls, [])

    def test_occluded_view_skips_unique_latent_and_execution_targets(self):
        model, batch = trainer("reader"), make_batch()
        first, second = batch.per_view_valid
        second = {name: value.clone() for name, value in second.items()}
        second["remaining.binding"][0, 0] = False
        paired = replace(batch, per_view_valid=(first, second))
        report = model.train_step(paired)
        self.assertGreater(report["requirement"], 0)
        self.assertTrue(has_gradient(model.reader))
        self.assertEqual(len(model.adapter.sample_calls), 1)
        self.assertEqual(len(model.adapter.action_calls), 2)
        model.adapter.sample_calls.clear()
        model.adapter.action_calls.clear()
        hidden = replace(batch, demonstrations=batch.demonstrations[1:], per_view_valid=(second,))
        losses = model.objective(hidden)
        self.assertEqual(float(losses["latent"]), 0)
        self.assertEqual(float(losses["execution"]), 0)
        self.assertEqual(model.adapter.sample_calls, [])
        self.assertEqual(model.adapter.action_calls, [])

    def test_hidden_binding_truth_cannot_change_uncertain_view_loss_or_gradient(self):
        model, batch = trainer("reader"), make_batch()
        hidden = {name: torch.zeros_like(value) for name, value in batch.per_view_valid[0].items()}
        batch = replace(batch, demonstrations=batch.demonstrations[:1], per_view_valid=(hidden,))

        def forbidden_goal(*args):
            raise AssertionError("a view without unique task evidence must not construct a privileged latent target")

        model.codec.encode = forbidden_goal
        first = model.objective(batch)
        first["total"].backward()
        gradients = [None if p.grad is None else p.grad.clone() for p in model.reader.parameters()]
        model.zero_grad(set_to_none=True)
        changed = TaskRequirement(
            replace(batch.requirements.current, binding=batch.requirements.current.binding.flip(1)),
            replace(batch.requirements.remaining, binding=batch.requirements.remaining.binding.flip(1)))
        second = model.objective(replace(batch, requirements=changed))
        second["total"].backward()
        self.assertTrue(torch.allclose(first["total"], second["total"]))
        for before, parameter in zip(gradients, model.reader.parameters()):
            self.assertTrue(before is None and parameter.grad is None or
                            before is not None and torch.allclose(before, parameter.grad))
        self.assertEqual(model.adapter.sample_calls, [])

    def test_reader_rejects_missing_per_view_evidence(self):
        model, batch = trainer("reader"), make_batch()
        with self.assertRaisesRegex(ValueError, "per_view_valid"):
            model.objective(replace(batch, per_view_valid=None))
        incomplete = dict(batch.per_view_valid[0])
        incomplete.pop("current.geometry")
        with self.assertRaisesRegex(ValueError, "every requirement"):
            model.objective(replace(batch, per_view_valid=(incomplete, incomplete)))

    def test_geometry_control_does_not_gate_on_unavailable_control_relation_evidence(self):
        model, batch = trainer("reader"), make_batch()
        evidence = []
        for view in batch.per_view_valid:
            masked = {name: value.clone() for name, value in view.items()}
            for part in ("current", "remaining"):
                for name in ("relations", "events"):
                    masked[f"{part}.{name}"][:] = False
            evidence.append(masked)
        batch = replace(batch, per_view_valid=tuple(evidence))
        full = model.objective(batch)
        self.assertEqual(float(full["execution"]), 0)
        self.assertEqual(model.adapter.sample_calls, [])
        model.codec.interface = "geometry"
        geometry = model.objective(batch)
        self.assertGreater(float(geometry["execution"].detach()), 0)
        self.assertEqual(len(model.adapter.sample_calls), 2)

    def test_missing_tolerance_or_event_window_evidence_blocks_unique_targets(self):
        for key in ("current.geometry_tolerance", "remaining.event_windows"):
            model, batch = trainer("reader"), make_batch()
            evidence = {name: value.clone() for name, value in batch.per_view_valid[0].items()}
            evidence[key].flatten()[0] = False
            batch = replace(batch, demonstrations=batch.demonstrations[:1], per_view_valid=(evidence,))
            losses = model.objective(batch)
            self.assertEqual(float(losses["latent"]), 0)
            self.assertEqual(float(losses["execution"]), 0)
            self.assertEqual(model.adapter.sample_calls, [])

    def test_interaction_head_cannot_pool_later_teacher_forced_chunks(self):
        model = trainer(weights=LossWeights(next_video=0, ifp=0))
        model.enable_ifp = False
        model.adapter.dropout.p = 0
        batch = make_batch()
        stream = batch.native_inputs["latent_dict"]
        for key in ("noisy_latents", "targets"):
            stream[key] = stream[key].repeat(1, 1, 2, 1, 1)
        batch.native_inputs["chunk_size"] = 1
        batch.entity_patch_weights = torch.ones(1, 2, 4) / 4
        from evo_wam.zerowam import TaskConditions
        conditions = TaskConditions(torch.ones(1, 2, 4), torch.ones(1, 2, 4), batch.null_text)
        with self.assertRaises(ValueError):
            model._native(batch, conditions, include_action=False, include_interaction=True,
                          interaction_valid=batch.per_view_valid[0])
        batch.entity_patch_weights[..., 2:] = 0
        _, first = model._native(batch, conditions, include_action=False, include_interaction=True,
                                 interaction_valid=batch.per_view_valid[0])
        stream["noisy_latents"][:, :, 1:] += 100
        _, second = model._native(batch, conditions, include_action=False, include_interaction=True,
                                  interaction_valid=batch.per_view_valid[0])
        torch.testing.assert_close(first["relation_logits"], second["relation_logits"])
        torch.testing.assert_close(first["event_logits"], second["event_logits"])

    def test_interface_updates_codec_physics_and_native_but_not_reader(self):
        model, batch = trainer("interface"), make_batch()
        report = model.train_step(batch)
        self.assertTrue(report["updated"])
        self.assertIn("native_action", report)
        self.assertIn("physical", report)
        self.assertNotIn("execution", report)
        self.assertTrue(has_gradient(model.codec))
        self.assertTrue(has_gradient(model.physical))
        self.assertTrue(has_gradient(model.adapter.native.action))
        self.assertFalse(has_gradient(model.reader))
        old_optimizer = model.optimizer
        model.set_stage("reader")
        self.assertIsNot(model.optimizer, old_optimizer)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        actual = {id(p) for group in model.optimizer.param_groups for p in group["params"]}
        self.assertEqual(actual, {id(p) for p in model.reader.parameters()})

    def test_execution_only_updates_reader_through_direct_condition(self):
        weights = LossWeights(requirement=0, latent=0, execution=1, next_video=0,
                              native_action=0, ifp=0, interaction=0, cv=0, physical=0)
        model, batch = trainer("reader", weights), make_batch()
        report = model.train_step(batch)
        self.assertGreater(report["execution"], 0)
        self.assertTrue(has_gradient(model.reader))
        self.assertFalse(has_gradient(model.adapter))
        self.assertFalse(has_gradient(model.codec))
        self.assertEqual(model.adapter.forward_inputs, [])
        self.assertEqual(model.adapter.sample_calls, [False, False])
        calls = model.adapter.action_calls
        for teacher, student in zip(calls[::2], calls[1::2]):
            self.assertFalse(teacher[0])
            self.assertTrue(student[0])
            self.assertEqual(teacher[1:4], student[1:4])
            self.assertEqual(student[4:], (False, None))

    def test_joint_updates_deployed_video_head_and_reader_only(self):
        model, batch = trainer(weights=LossWeights(cv=0.5)), make_batch()
        report = model.train_step(batch)
        self.assertTrue(has_gradient(model.adapter.native.video))
        self.assertTrue(has_gradient(model.adapter.native.mcp))
        self.assertTrue(has_gradient(model.interaction))
        self.assertTrue(has_gradient(model.reader))
        self.assertFalse(has_gradient(model.adapter.native.action))
        self.assertFalse(has_gradient(model.adapter.condition_projection))
        self.assertFalse(has_gradient(model.codec))
        self.assertFalse(has_gradient(model.physical))
        self.assertNotIn("native_action", report)
        self.assertEqual(report["cv_coverage"], 1.0)
        first, second = model.adapter.forward_inputs
        self.assertIs(first["latent_dict"], second["latent_dict"])
        self.assertIs(first["mcp_latent_dicts"][0], second["mcp_latent_dicts"][0])

    def test_native_training_receives_same_observed_history_for_pairs_and_null_examples(self):
        from evo_wam.zerowam import NativeHistoryChunk
        model, batch = trainer(weights=LossWeights(execution=0)), make_batch()
        history = (NativeHistoryChunk("video", torch.ones(1, 4, 1, 1, 2), 0, 0),
                   NativeHistoryChunk("action", torch.zeros(1, 4, 1, 2, 1), 1, 0))
        batch = replace(batch, sample_kwargs={"history": history})
        model.objective(batch)
        self.assertEqual(len(model.adapter.forward_histories), 2)
        self.assertTrue(all(actual is history for actual in model.adapter.forward_histories))
        model.objective(replace(batch, conditional=False))
        self.assertIs(model.adapter.forward_histories[-1], history)
        # The future-only payload is unchanged; prepending belongs to the
        # native adapter and must not duplicate observed frames in the trainer.
        self.assertEqual(batch.native_inputs["latent_dict"]["noisy_latents"].shape[2], 1)

    def test_cutting_direct_goal_path_removes_execution_gradient(self):
        weights = LossWeights(requirement=0, latent=0, execution=1, next_video=0,
                              native_action=0, ifp=0, interaction=0, cv=0, physical=0)
        model, batch = trainer("reader", weights), make_batch()
        model.adapter.detach_current = True
        report = model.train_step(batch)
        self.assertFalse(report["updated"])
        self.assertFalse(has_gradient(model.reader))

    def test_identical_view_zero_js_and_supervision_not_doubled(self):
        model, batch = trainer(weights=LossWeights(cv=0.5)), make_batch()
        pair = model.objective(batch)
        self.assertLess(abs(float(pair["cv"].detach())), 1e-7)
        model.weights = replace(model.weights, cv=0)
        v0 = model.objective(batch)
        single = model.objective(replace(batch, demonstrations=batch.demonstrations[:1],
                                         per_view_valid=batch.per_view_valid[:1]))
        self.assertTrue(torch.allclose(v0["total"], single["total"], atol=1e-6))
        self.assertTrue(torch.allclose(pair["total"], v0["total"], atol=1e-6))
        self.assertTrue(model.adapter.dropout.training)  # scoped dropout override restored

    def test_unconditional_never_calls_task_or_true_goal_paths(self):
        model, batch = trainer(weights=LossWeights(cv=0.5)), make_batch()

        def forbidden(*args, **kwargs):
            raise AssertionError("task path must not execute on null examples")

        model.codec.encode = forbidden
        model.reader.forward = forbidden
        model.interaction.forward = forbidden
        model.physical.forward = forbidden
        report = model.train_step(replace(batch, conditional=False))
        self.assertEqual(set(report), {"next_video", "ifp", "cv_coverage", "total", "grad_norm", "updated"})
        self.assertEqual(model.adapter.action_calls, [])
        self.assertEqual(model.adapter.sample_calls, [])
        self.assertTrue(report["updated"])
        model.set_stage("reader")
        self.assertFalse(model.train_step(replace(batch, conditional=False))["updated"])

    def test_empty_common_mask_retains_supervision_and_has_zero_coverage(self):
        model, batch = trainer(weights=LossWeights(cv=1)), make_batch()
        empty = {name: torch.zeros_like(value) for name, value in batch.per_view_valid[0].items()}
        losses = model.objective(replace(batch, per_view_valid=(empty, empty)))
        self.assertEqual(float(losses["cv_coverage"]), 0)
        self.assertEqual(float(losses["cv"].detach()), 0)
        self.assertGreater(float(losses["requirement"].detach()), 0)
        self.assertTrue(torch.isfinite(losses["total"]))

    def test_bfloat_native_phi_keeps_gradient_to_fp32_interaction_head(self):
        weights = LossWeights(requirement=0, latent=0, execution=0, next_video=0,
                              native_action=0, ifp=0, interaction=1, cv=0, physical=0)
        model, batch = trainer(weights=weights), make_batch()
        original = model.adapter.forward_train

        def mixed_precision_forward(*args, **kwargs):
            output = original(*args, **kwargs)
            output.phi = output.phi.bfloat16()
            return output

        model.adapter.forward_train = mixed_precision_forward
        model.train_step(batch)
        self.assertTrue(has_gradient(model.adapter.native.video))
        self.assertTrue(has_gradient(model.interaction))

    def test_padding_entities_have_zero_pool_and_no_physical_supervision(self):
        model, batch = trainer(), make_batch()
        ids = torch.tensor([[7, -1]])
        current = replace(batch.requirements.current, entity_ids=ids, binding=torch.tensor([[0, 0]]))
        remaining = replace(batch.requirements.remaining, entity_ids=ids, binding=torch.tensor([[0, 0]]))
        valid = {name: value.clone() for name, value in batch.outcome.label_valid.items()}
        for name, mask in valid.items():
            mask[:, :, 1] = False
            if name != "geometry":
                mask[:, :, :, 1] = False
        batch = replace(batch, requirements=TaskRequirement(current, remaining),
                        outcome=replace(batch.outcome, entity_ids=ids, label_valid=valid),
                        entity_patch_weights=torch.tensor([[[1., 0.], [0., 0.]]]))
        self.assertTrue(torch.isfinite(model.objective(batch)["total"]))
        with self.assertRaisesRegex(ValueError, "patch weights"):
            model.objective(replace(batch, entity_patch_weights=torch.eye(2)[None]))

    def test_rejects_conditional_bypass_and_unsupervised_physical_future(self):
        model, batch = trainer(), make_batch()
        with self.assertRaisesRegex(ValueError, "raw ICL"):
            model.objective(replace(batch, native_inputs={**batch.native_inputs, "icl_latent_dict": {}}))
        with self.assertRaisesRegex(ValueError, "actual history"):
            model.objective(replace(batch, sample_kwargs={"training_cache": object()}))
        model.set_stage("interface")
        with self.assertRaisesRegex(ValueError, "unexecuted"):
            model.objective(replace(batch, physical_actions=batch.physical_actions[:, :1]))

    def test_gradient_norm_failure_never_steps_optimizer(self):
        model, batch = trainer("reader"), make_batch()
        before = [parameter.detach().clone() for parameter in model.reader.parameters()]
        hook = next(model.reader.parameters()).register_hook(lambda grad: grad * float("nan"))
        try:
            with self.assertRaises(RuntimeError):
                model.train_step(batch)
        finally:
            hook.remove()
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(before, model.reader.parameters())))
        self.assertEqual(model.updates, 0)


if __name__ == "__main__":
    unittest.main()
