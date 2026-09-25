"""CPU policy-boundary tests with sampler spies, not policy performance tests.

The requirement codec and data-history packing are real implementations. The
sampler only records calls and returns its noise; no native WAM, task success,
simulator interaction or robot command is claimed by these tests.
"""

from dataclasses import replace
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from etude.cli import make_fixture
from etude.contracts import TaskRequirement
from etude.data import (ObservedActionHistory, load_experiment, load_observation,
                          load_sample)
from etude.inference import NativePolicy, RequirementRejected
from etude.models import GoalTokens, RequirementCodec


class SamplerSpy(nn.Module):
    """The public adapter signature, deliberately no learned action behavior."""

    def __init__(self):
        super().__init__()
        self.native = nn.Linear(1, 1).to(dtype=torch.bfloat16)
        self.native.config = SimpleNamespace(action_dim=3)
        self.video_calls, self.action_calls = [], []

    def sample_video(self, noise, conditions, **kwargs):
        future = SimpleNamespace(latents=noise, conditions=conditions)
        self.video_calls.append((noise, conditions, kwargs, future, torch.is_grad_enabled()))
        return future

    def sample_actions(self, initial, conditions, future, **kwargs):
        self.action_calls.append((initial, conditions, future, kwargs, torch.is_grad_enabled()))
        return initial * kwargs["action_mask"]


class InferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        folder = Path(self.directory.name)
        make_fixture(folder)
        sample = load_sample(folder / "sample.json")
        self.requirement = sample.requirement
        self.observation = load_observation(folder / "observation.json")
        self.config = load_experiment(Path(__file__).resolve().parents[1] / "configs" / "full.json")
        d = self.config["dimensions"]
        self.codec = RequirementCodec(d["entity_dim"], d["geometry_dim"], d["relation_dim"],
            d["event_dim"], d["roles"], d["token_dim"], "full", d["max_precedence_edges"])
        self.adapter = SamplerSpy()
        self.reader = Mock(side_effect=AssertionError("oracle path must not invoke the task reader"))
        self.trainer = SimpleNamespace(codec=self.codec, reader=self.reader, adapter=self.adapter,
            action_spaces={self.observation.action_space["normalization_id"]: self.observation.action_space})
        self.policy = NativePolicy(self.trainer, torch.zeros(1, 2, 8), self.config, sampling_steps=3, diagnostic=True)

    def assert_not_sampled(self):
        self.assertEqual(self.adapter.video_calls, [])
        self.assertEqual(self.adapter.action_calls, [])

    def test_interface_checkpoint_cannot_call_untrained_reader(self):
        self.policy.source_stage = "interface"
        with self.assertRaisesRegex(RequirementRejected, "trained_reader"):
            self.policy.candidates(self.observation, None, random.Random(0))
        self.reader.assert_not_called()
        self.assert_not_sampled()

    def test_direct_reader_policy_requires_validation_lock_or_explicit_diagnostic(self):
        self.policy.diagnostic = False
        with self.assertRaisesRegex(RequirementRejected, "validation_locked"):
            self.policy.candidates(self.observation, None, random.Random(0))
        self.reader.assert_not_called()
        self.assert_not_sampled()

    def test_raw_features_and_encoded_demonstrations_cannot_be_silently_mixed(self):
        encoding = {"kind": "video_effect_tokens", "encoder_sha256": "1" * 64,
                    "encoder_version": 2,
                    "feature_space_id": "frozen-test-v1", "token_dim": 4,
                    "window_frames": 3, "num_tokens": 2}
        encoded = replace(self.observation, demonstration_encoding=encoding)
        with self.assertRaisesRegex(RequirementRejected, "encoding_differs"):
            self.policy.candidates(encoded, None, random.Random(0))
        self.trainer.demonstration_encoding = encoding
        with self.assertRaisesRegex(RequirementRejected, "encoding_differs"):
            self.policy.candidates(self.observation, None, random.Random(0))
        self.reader.assert_not_called()
        self.assert_not_sampled()

    def test_oracle_uses_real_codec_and_the_shared_native_sampling_path(self):
        with torch.no_grad():
            expected = tuple(self.codec.encode(getattr(self.requirement, name), self.observation.robot_history[:, -1])
                             for name in ("current", "remaining"))
        with patch.object(self.codec, "encode", wraps=self.codec.encode) as encode:
            candidates, requirement = self.policy.candidates(self.observation, self.requirement, random.Random(31))
            self.assertEqual(encode.call_count, 2)
        self.reader.assert_not_called()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(candidates[0].actions), self.observation.chunk_size * self.observation.actions_per_frame)
        self.assertEqual(len(candidates[0].actions[0]), 3)
        self.assertTrue(requirement.semantics_known.all())
        noise, conditions, kwargs, future, grad_enabled = self.adapter.video_calls[0]
        torch.testing.assert_close(conditions.current, expected[0])
        torch.testing.assert_close(conditions.remaining, expected[1])
        self.assertEqual(noise.dtype, torch.bfloat16)
        self.assertFalse(grad_enabled)
        self.assertEqual(kwargs["steps"], 3)
        self.assertEqual(kwargs["shift"], self.config["video_snr_shift"])
        action_noise, action_conditions, action_future, action_kwargs, action_grad = self.adapter.action_calls[0]
        self.assertIs(action_conditions, conditions)
        self.assertIs(action_future, future)
        self.assertIs(action_kwargs["history"], kwargs["history"])
        self.assertFalse(action_grad)
        self.assertEqual(action_noise.shape, (1, 3, 2, 2, 1))

    def test_four_candidates_share_one_future_and_replay_rng(self):
        first, _ = self.policy.candidates(self.observation, self.requirement, random.Random(8), count=4)
        self.assertEqual(len(self.adapter.video_calls), 1)
        self.assertEqual(len(self.adapter.action_calls), 4)
        future = self.adapter.video_calls[0][3]
        self.assertTrue(all(call[2] is future for call in self.adapter.action_calls))
        second, _ = self.policy.candidates(self.observation, self.requirement, random.Random(8), count=4)
        self.assertEqual([c.actions for c in first], [c.actions for c in second])
        self.assertNotEqual(first[0].actions, first[1].actions)

    def test_oracle_and_reader_routes_use_identical_sampling_inputs_for_identical_goals(self):
        # This is an input-routing check; the reader outputs are supplied, not
        # learned. It says nothing about demonstration understanding accuracy.
        goals = GoalTokens(*(self.codec.encode(getattr(self.requirement, name), self.observation.robot_history[:, -1])
                             for name in ("current", "remaining")))
        self.trainer.reader = Mock(return_value=goals)
        decoded = [SimpleNamespace(materialize=Mock(return_value=getattr(self.requirement, name)))
                   for name in ("current", "remaining")]
        oracle, _ = self.policy.candidates(self.observation, self.requirement, random.Random(77))
        with patch.object(self.codec, "decode", side_effect=decoded):
            inferred, _ = self.policy.candidates(self.observation, None, random.Random(77))
        self.trainer.reader.assert_called_once()
        self.assertEqual(oracle[0].actions, inferred[0].actions)
        first, second = self.adapter.video_calls
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1].current, second[1].current)
        torch.testing.assert_close(first[1].remaining, second[1].remaining)
        for part in decoded:
            part.materialize.assert_called_once_with(interface="full", min_binding_confidence=0.5,
                                                      min_binding_margin=0.1)

    def test_empty_unresolved_unknown_and_invalid_requirements_never_sample(self):
        current = self.requirement.current
        empty = replace(current, requirement_mask={name: torch.zeros_like(value)
                                                   for name, value in current.requirement_mask.items()})
        missing_binding = current.binding.clone()
        missing_binding[0, 0] = -2
        unresolved = replace(current, binding=missing_binding)
        uncertain_binding = current.binding.clone()
        uncertain_binding[0, 0] = -3
        uncertain = replace(current, binding=uncertain_binding)
        unknown = replace(current, label_valid={**current.label_valid,
            "geometry_tolerance": torch.zeros_like(current.label_valid["geometry_tolerance"])})
        invalid = replace(current, geometry_tolerance=-torch.ones_like(current.geometry_tolerance))
        for name, part in (("empty", empty), ("unmatched", unresolved), ("uncertain", uncertain),
                           ("unknown", unknown), ("invalid", invalid)):
            with self.subTest(name=name), self.assertRaises(RequirementRejected):
                self.policy.candidates(self.observation, TaskRequirement(part, self.requirement.remaining), random.Random(0))
            self.assert_not_sampled()

    def test_oracle_entity_query_grid_and_horizon_mismatch_never_sample(self):
        ids = self.requirement.current.entity_ids + 100
        wrong_entities = TaskRequirement(replace(self.requirement.current, entity_ids=ids),
                                        replace(self.requirement.remaining, entity_ids=ids))
        wrong_grid = TaskRequirement(replace(self.requirement.current, step_offsets=self.requirement.current.step_offsets + 1),
                                    self.requirement.remaining)
        for requirement in (wrong_entities, wrong_grid):
            with self.assertRaises(RequirementRejected):
                self.policy.candidates(self.observation, requirement, random.Random(0))
            self.assert_not_sampled()
        # Identical oracle grid, but a shorter actual candidate cannot meet it.
        with self.assertRaises(RequirementRejected):
            self.policy.candidates(replace(self.observation, chunk_size=1), self.requirement, random.Random(0))
        self.assert_not_sampled()

    def test_geometry_oracle_ignores_unknown_control_semantics_while_full_refuses(self):
        parts = []
        for part in (self.requirement.current, self.requirement.remaining):
            valid = {name: value.clone() for name, value in part.label_valid.items()}
            for name in ("relations", "events", "event_windows", "event_precedence"):
                valid[name][:] = False
            parts.append(replace(part, label_valid=valid))
        requirement = TaskRequirement(*parts).validate()
        with self.assertRaises(RequirementRejected):
            self.policy.candidates(self.observation, requirement, random.Random(0))
        self.assert_not_sampled()

        self.codec.interface = "geometry"
        geometry_policy = NativePolicy(self.trainer, self.policy.null_text,
                                       {**self.config, "interface": "geometry"})
        candidates, geometry = geometry_policy.candidates(self.observation, requirement, random.Random(0))
        self.assertEqual(len(candidates), 1)
        self.assertTrue(geometry.semantics_known.all())
        for part in (geometry.current, geometry.remaining):
            self.assertFalse(part.requirement_mask["relations"].any())
            self.assertFalse(part.requirement_mask["events"].any())
            self.assertTrue((part.event_precedence == -1).all())
            self.assertTrue(part.label_valid["event_precedence"].all())
        # The full oracle annotation is not changed in place by the control.
        self.assertTrue(requirement.current.requirement_mask["relations"].any())
        self.assertFalse(requirement.current.label_valid["event_precedence"].any())

    def test_required_event_window_past_horizon_rejects_before_sampling(self):
        current = self.requirement.current
        windows = current.event_windows.clone()
        horizon = self.observation.chunk_size * self.observation.actions_per_frame
        windows[0, 0, 0, 0, 0] = torch.tensor([1, horizon + 1])
        requirement = TaskRequirement(replace(current, event_windows=windows), self.requirement.remaining).validate()
        self.assertLessEqual(int(current.step_offsets.max()), horizon)
        with self.assertRaisesRegex(RequirementRejected, "exceeds_candidate_horizon"):
            self.policy.candidates(self.observation, requirement, random.Random(0))
        self.assert_not_sampled()

    def test_observed_action_history_is_packed_by_data_and_passed_to_both_samplers(self):
        commands = torch.tensor([[[0.2, -0.1, 0.3], [0.4, 0.5, -0.2], [0.7, 0.1, 0.6]]])
        history = ObservedActionHistory(commands, torch.tensor([-2, -1, 0]), self.observation.action_space,
                                        observation_step=3, control_dt=0.05)
        observation = replace(self.observation, observed_action_history=history,
            observed_video_step_offsets=torch.tensor([-3]), history_chunks=(
                {"mode": "video", "slice": [0, 1], "frame_id": 0, "rope_offset": 0},
                {"mode": "action", "slice": [0, 3], "frame_id": 1, "rope_offset": 0}))
        expected = observation.native_history(dtype=torch.bfloat16)
        self.policy.candidates(observation, self.requirement, random.Random(1))
        passed = self.adapter.video_calls[0][2]["history"]
        self.assertEqual([chunk.mode for chunk in passed], ["video", "action"])
        for actual, target in zip(passed, expected):
            torch.testing.assert_close(actual.latent, target.latent)
            self.assertEqual((actual.frame_id, actual.rope_offset), (target.frame_id, target.rope_offset))
            if target.token_valid is not None:
                torch.testing.assert_close(actual.token_valid, target.token_valid)
        self.assertEqual(passed[1].token_valid.tolist(), [True, True, True, False])
        self.assertEqual(self.adapter.video_calls[0][2]["frame_id"], 2)
        self.assertEqual(self.adapter.video_calls[0][2]["rope_offset"], 2)
        self.assertIs(passed, self.adapter.action_calls[0][3]["history"])

    def test_action_normalization_mismatch_and_invalid_count_reject_before_sampling(self):
        wrong = {**self.observation.action_space, "normalization_id": "other-calibration"}
        with self.assertRaises(ValueError):
            self.policy.candidates(replace(self.observation, action_space=wrong), self.requirement, random.Random(0))
        with self.assertRaises(ValueError):
            self.policy.candidates(self.observation, self.requirement, random.Random(0), count=2)
        self.assert_not_sampled()

    def test_future_actions_in_programmatic_observation_never_reach_sampling(self):
        # Server-side observation providers can construct the dataclass without
        # load_observation; the history packing boundary must validate it too.
        history = ObservedActionHistory(torch.zeros(1, 1, 3), torch.tensor([1]),
            self.observation.action_space, observation_step=1, control_dt=0.05)
        observation = replace(self.observation, observed_action_history=history,
            observed_video_step_offsets=torch.tensor([-1]), history_chunks=(
                {"mode": "video", "slice": [0, 1], "frame_id": 0, "rope_offset": 0},
                {"mode": "action", "slice": [0, 1], "frame_id": 1, "rope_offset": 0}))
        with self.assertRaises(ValueError):
            self.policy.candidates(observation, self.requirement, random.Random(0))
        self.assert_not_sampled()


if __name__ == "__main__":
    unittest.main()
