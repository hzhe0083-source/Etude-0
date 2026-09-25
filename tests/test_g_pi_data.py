from dataclasses import asdict
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from etude.g_pi_data import (EventRules, GTranslatorSample, GripperEventDetector, PiGoalSample,
                              _block_endpoint, detect_gripper_events, g_pi_sample_files, load_g_pi_index,
                              load_g_pi_sample, next_subgoal_time, subgoal_control_indices, validate_latent_grid)
from etude.icl_data import LATENT_NORMALIZATION
from test_goal_language import write_goal_language


def write_g_pi_task(root, name="task", *, gripper=None, demonstration=True, frame_stride=1):
    if gripper is None:
        gripper = np.array([[0.]] * 5 + [[1.]] * 12, dtype=np.float32)
    gripper = np.asarray(gripper, dtype=np.float32)
    frames, effectors = gripper.shape
    count = 4 * frame_stride
    latent_frames = (frames - 1) // count + 1
    metadata = {
        "format_version": 3, "kind": "g_pi_task", "sample_id": name, "arrays": f"{name}.npz",
        "robot_source": {"source_id": f"{name}-robot", "source_group": f"{name}-robot",
                         "domain": "robot", "trajectory_id": f"{name}-trajectory"},
        "action_space": {"representation": "zero-wam-normalized", "normalization_id": "fixture",
                         "dimension": 2, "valid_channels": [True, False]},
        "state_space_id": "fixture-joint-state", "coordinate_frame": "robot_base", "pose_units": "m",
        "goal_source": "measured_endpoint", "end_effectors": [f"gripper-{i}" for i in range(effectors)],
        "pose_representation": "absolute_robot_base_tool", "tool_frames": [f"tool-{i}" for i in range(effectors)],
        "gripper_space": {"normalization_id": "fixture-width", "closed": [0.] * effectors,
                          "open": [.08] * effectors, "units": "m"},
        "language": f"{name}-language.json", "feature_space_id": "fixture-wan-v1",
        "latent_normalization": LATENT_NORMALIZATION, "control_dt": .1,
        "action_frames": 2, "actions_per_frame": count, "task_start_time": 0., "success": True,
        "frame_stride": frame_stride, "temporal_down_rate": 4,
        "alignment": "zerowam_causal_first_then_four", "subgoal_encoding": "wan_vae_single_frame",
        "event_rules": asdict(EventRules()),
    }
    write_goal_language(root, f"{name}-language")
    poses = np.tile(np.eye(4, dtype=np.float32), (frames, effectors, 1, 1))
    poses[:, :, 0, 3] = np.arange(frames, dtype=np.float32)[:, None]
    times = np.arange(frames, dtype=np.float64) * .1
    goals = subgoal_control_indices(torch.from_numpy(gripper), torch.from_numpy(times)).numpy()
    arrays = {"latent": np.arange(2 * latent_frames * 4, dtype=np.float32).reshape(2, latent_frames, 2, 2),
              "control_times": times, "latent_available_times": times[::count],
              "states": np.arange(frames * 4, dtype=np.float32).reshape(frames, 4),
              "poses": poses, "gripper": gripper,
              "actions": 1 + np.arange(2 * frames, dtype=np.float32).reshape(2, frames),
              "actions_mask": np.ones((2, frames), dtype=np.bool_), "subgoal_times": times[goals],
              "subgoal_latents": np.stack([np.full((2, 1, 2, 2), 1000 + int(index), dtype=np.float32)
                                             for index in goals])}
    if demonstration:
        metadata.update(demonstration={"source_id": f"{name}-demo", "source_group": f"{name}-demo",
                                       "domain": "human", "arrays": f"{name}-demo.npz"},
                        compatibility={"kind": "audited_semantic_task", "evidence": "fixture task-level audit"})
        np.savez_compressed(root / f"{name}-demo.npz", latent=np.ones((2, 3, 2, 2), dtype=np.float32),
                            frame_times=np.array([0., .3, .9], dtype=np.float64))
    path = root / f"{name}.json"
    path.write_text(json.dumps(metadata))
    np.savez_compressed(root / metadata["arrays"], **arrays)
    return path, metadata, arrays


class GripperEventsTest(unittest.TestCase):
    def events(self, values, **rules):
        values = torch.tensor(values, dtype=torch.float32)
        if values.ndim == 1:
            values = values[:, None]
        return detect_gripper_events(values, torch.arange(len(values), dtype=torch.float64), EventRules(**rules))

    def test_no_transition_only_terminal(self):
        events = self.events([0., 0., 0., 0.])
        self.assertEqual(events, ())
        self.assertEqual(next_subgoal_time(0., 3., events), 3.)

    def test_hysteresis_and_debounce_reject_chatter(self):
        events = self.events([0., 0., .76, .74, .76, .76, .24, .26, .24, .24])
        self.assertEqual([(event.time, event.kind) for event in events], [(5., "open"), (9., "close")])

    def test_multiple_effectors_fixed_order_and_online_equivalence(self):
        values = torch.tensor([[0., 1.], [0., 1.], [1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        times = torch.arange(len(values), dtype=torch.float64)
        events = detect_gripper_events(values, times)
        self.assertEqual([(e.time, e.effector, e.kind) for e in events],
                         [(3., 0, "open"), (3., 1, "close"), (5., 0, "close"), (5., 1, "open")])
        detector = GripperEventDetector(2)
        online = tuple(event for value, time in zip(values, times.tolist()) for event in detector.update(value, time))
        self.assertEqual(online, events)
        detector.reset()
        self.assertEqual(detector.update(values[0], 0.), ())

    def test_event_confirmation_never_uses_later_samples(self):
        values = torch.tensor([[0.], [0.], [1.], [1.], [0.], [0.]])
        for length in range(1, len(values) + 1):
            prefix = detect_gripper_events(values[:length], torch.arange(length, dtype=torch.float64))
            whole = detect_gripper_events(values, torch.arange(len(values), dtype=torch.float64))
            self.assertEqual(prefix, tuple(event for event in whole if event.time < length))

    def test_next_event_is_strictly_future_then_terminal(self):
        events = self.events([0., 0., 1., 1., 0., 0., 0.])
        self.assertEqual(next_subgoal_time(2., 6., events), 3.)
        self.assertEqual(next_subgoal_time(3., 6., events), 5.)
        self.assertEqual(next_subgoal_time(5., 6., events), 6.)
        for time in (6., 7., -1., float("nan")):
            with self.subTest(time=time), self.assertRaises(ValueError):
                next_subgoal_time(time, 6., events)

    def test_invalid_rules_and_signals(self):
        for values in ({"signal_source": "unknown"}, {"close_threshold": .8}, {"debounce_steps": True},
                       {"debounce_steps": 0}, {"open_threshold": float("nan")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                EventRules(**values)
        detector = GripperEventDetector(1)
        detector.update(torch.tensor([0.]), 0.)
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            detector.update(torch.tensor([0.]), 0.)
        with self.assertRaisesRegex(ValueError, "measured gripper"):
            detector.update(torch.tensor([1.1]), 1.)


class GPiDataTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def save(self, path, metadata, arrays):
        path.write_text(json.dumps(metadata))
        np.savez_compressed(path.parent / metadata["arrays"], **arrays)

    def test_midchunk_event_masks_control_actions_and_slices_latent_history(self):
        path, _, arrays = write_g_pi_task(self.root)
        sample = load_g_pi_sample(path, current_time=.4)
        self.assertIs(type(sample), PiGoalSample)
        self.assertAlmostEqual(sample.subgoal_time, .6)
        self.assertEqual(sample.terminal_time, 1.6)
        self.assertEqual(sample.history.shape, (1, 2, 2, 2, 2))
        self.assertEqual(sample.history_times.tolist(), [0., .4])
        self.assertEqual(sample.target_frame.shape, (1, 2, 1, 2, 2))
        self.assertEqual(sample.actions.shape, (1, 2, 2, 4, 1))
        self.assertEqual(sample.actions_mask.flatten().tolist(), [True, True] + [False] * 14)
        self.assertEqual(sample.actions.flatten()[:2].tolist(), [5., 6.])
        torch.testing.assert_close(sample.state[0], torch.from_numpy(arrays["states"][4]))
        torch.testing.assert_close(sample.target_frame[0], torch.from_numpy(arrays["subgoal_latents"][0]))
        torch.testing.assert_close(sample.goal_poses[0], torch.from_numpy(arrays["poses"][6]))
        torch.testing.assert_close(sample.goal_gripper[0], torch.from_numpy(arrays["gripper"][6]))
        self.assertFalse(hasattr(sample, "demonstration"))
        self.assertNotIn("demonstration", inspect.signature(PiGoalSample).parameters)
        self.assertNotIn("demonstration", sample.metadata)

    def test_block_endpoint_is_next_recorded_state_after_horizon(self):
        path, _, arrays = write_g_pi_task(self.root, gripper=np.zeros((17, 1)))
        sample = load_g_pi_sample(path, current_time=0.)
        self.assertEqual(sample.actions_mask.any(1).sum().item(), 8)
        self.assertEqual(sample.subgoal_time, 1.6)
        self.assertEqual(sample.block_end_index, 8)
        self.assertEqual(sample.block_end_time, .8)
        self.assertTrue(sample.block_end_valid)
        self.assertFalse(sample.reaches_subgoal)
        torch.testing.assert_close(sample.block_end_poses[0], torch.from_numpy(arrays["poses"][8]), atol=0, rtol=0)
        torch.testing.assert_close(sample.block_end_gripper[0], torch.from_numpy(arrays["gripper"][8]), atol=0, rtol=0)
        self.assertFalse(torch.equal(sample.goal_poses, sample.block_end_poses))
        self.assertEqual(sample.metadata["block_end_index"], 8)
        self.assertNotIn("block_end_poses", sample.metadata)

    def test_subgoal_cut_and_terminal_padding_determine_block_endpoint(self):
        path, _, arrays = write_g_pi_task(self.root)
        for time, endpoint in ((.4, 6), (1.2, 16)):
            with self.subTest(current_time=time):
                sample = load_g_pi_sample(path, current_time=time)
                self.assertEqual(sample.block_end_index, endpoint)
                self.assertTrue(sample.block_end_valid)
                self.assertTrue(sample.reaches_subgoal)
                self.assertEqual(sample.block_end_time, sample.subgoal_time)
                torch.testing.assert_close(sample.block_end_poses, sample.goal_poses, atol=0, rtol=0)
                torch.testing.assert_close(sample.block_end_gripper, sample.goal_gripper, atol=0, rtol=0)
                torch.testing.assert_close(sample.block_end_poses[0], torch.from_numpy(arrays["poses"][endpoint]),
                                           atol=0, rtol=0)

    def test_mask_holes_use_last_supervised_step_not_count_or_disabled_channel(self):
        path, metadata, arrays = write_g_pi_task(self.root)
        arrays["actions_mask"][0] = False
        arrays["actions_mask"][0, [0, 3, 4]] = True
        # Channel 1 has future labels but action_space declares it inactive.
        self.save(path, metadata, arrays)
        sample = load_g_pi_sample(path, current_time=0.)
        self.assertEqual(sample.actions_mask.sum().item(), 3)
        self.assertEqual(sample.block_end_index, 5)
        self.assertEqual(sample.block_end_time, .5)
        self.assertFalse(sample.reaches_subgoal)
        metadata["action_space"]["valid_channels"][1] = True
        arrays["actions_mask"][1] = False
        arrays["actions_mask"][1, 5] = True
        self.save(path, metadata, arrays)
        sample = load_g_pi_sample(path, current_time=0.)
        self.assertEqual(sample.block_end_index, 6)
        self.assertTrue(sample.reaches_subgoal)
        arrays["actions_mask"][:] = False
        self.save(path, metadata, arrays)
        with self.assertRaisesRegex(ValueError, "valid action supervision"):
            load_g_pi_sample(path, current_time=0.)

    def test_missing_post_action_measurement_marks_endpoint_invalid(self):
        poses = torch.eye(4).repeat(3, 1, 1, 1)
        gripper, times = torch.zeros(3, 1), torch.arange(3, dtype=torch.float64)
        label = _block_endpoint(torch.tensor([[True, False, True]]), 0, poses, gripper, times, 4)
        self.assertEqual(label["block_end_index"], 3)
        self.assertFalse(label["block_end_valid"])
        self.assertFalse(label["reaches_subgoal"])
        for key in ("block_end_poses", "block_end_gripper", "block_end_time"):
            self.assertIsNone(label[key])
        with self.assertRaisesRegex(ValueError, "nonempty Boolean"):
            _block_endpoint(torch.zeros(1, 3, dtype=torch.bool), 0, poses, gripper, times, 2)

    def test_prior_never_fetches_visual_arrays_or_demonstration_and_has_no_g(self):
        path, metadata, _ = write_g_pi_task(self.root)
        visual = load_g_pi_sample(path, current_time=.4)
        (self.root / metadata["demonstration"]["arrays"]).unlink()
        original = np.lib.npyio.NpzFile.__getitem__
        accessed = []

        def read(archive, key):
            accessed.append(key)
            if key in {"latent", "subgoal_latents"}:
                raise AssertionError("the nonvisual prior must not fetch video or goal latents")
            return original(archive, key)

        with patch.object(np.lib.npyio.NpzFile, "__getitem__", read):
            prior = load_g_pi_sample(path, route="pi_prior", current_time=.4)
        self.assertIn("language", accessed)
        self.assertIn("latent_available_times", accessed)
        for field in ("history", "history_times", "target_frame", "goal_poses", "goal_gripper"):
            self.assertIsNone(getattr(prior, field))
        self.assertNotIn("demonstration", prior.metadata)
        for field in ("state", "actions", "actions_mask", "block_end_poses", "block_end_gripper", "language"):
            torch.testing.assert_close(getattr(prior, field), getattr(visual, field), atol=0, rtol=0)
        self.assertEqual(prior.block_end_index, visual.block_end_index)
        self.assertEqual(len(g_pi_sample_files(path, prior)), 4)

    def test_prior_preserves_grid_boundary_and_schema_validation(self):
        path, metadata, arrays = write_g_pi_task(self.root)
        for key, value, message in (("latent_available_times", arrays["latent_available_times"] + .1, "causal grid"),
                                    ("subgoal_times", arrays["subgoal_times"] + .1, "recomputed")):
            self.save(path, metadata, {**arrays, key: value})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                load_g_pi_sample(path, route="pi_prior", current_time=.4)
        missing = {key: value for key, value in arrays.items() if key != "subgoal_latents"}
        self.save(path, metadata, missing)
        with self.assertRaisesRegex(ValueError, "version-3 task NPZ requires"):
            load_g_pi_sample(path, route="pi_prior", current_time=.4)
        self.save(path, metadata, arrays)
        with self.assertRaisesRegex(ValueError, "latent availability"):
            load_g_pi_sample(path, route="pi_prior", current_time=.3)
        with patch("etude.g_pi_data._block_endpoint", return_value={"block_end_valid": False}):
            with self.assertRaisesRegex(ValueError, "recorded state after"):
                load_g_pi_sample(path, route="pi_prior", current_time=.4)

    def test_prior_random_time_and_action_mask_match_visual_route(self):
        path, _, _ = write_g_pi_task(self.root)
        one, two = torch.Generator().manual_seed(74), torch.Generator().manual_seed(74)
        for _ in range(8):
            visual = load_g_pi_sample(path, generator=one)
            prior = load_g_pi_sample(path, route="pi_prior", generator=two)
            self.assertEqual(visual.metadata["current_time"], prior.metadata["current_time"])
            self.assertEqual(visual.block_end_index, prior.block_end_index)
            torch.testing.assert_close(visual.actions_mask, prior.actions_mask, atol=0, rtol=0)

    def test_after_last_event_uses_terminal_including_remaining_motion(self):
        path, _, _ = write_g_pi_task(self.root)
        for time in (.8, 1.2):
            sample = load_g_pi_sample(path, current_time=time)
            self.assertEqual(sample.subgoal_time, 1.6)
            self.assertGreater(sample.subgoal_time, sample.metadata["current_time"])
        self.assertEqual(load_g_pi_sample(path, current_time=1.2).actions_mask.sum().item(), 4)

    def test_no_events_target_terminal_only(self):
        path, _, _ = write_g_pi_task(self.root, gripper=np.zeros((17, 1)))
        for time in (0., .4, 1.2):
            sample = load_g_pi_sample(path, current_time=time)
            self.assertEqual(sample.events, ())
            self.assertEqual(sample.subgoal_time, 1.6)

    def test_terminal_and_nonavailability_samples_rejected(self):
        path, _, _ = write_g_pi_task(self.root)
        for time in (1.6, 1.7, 2., -.1, .2, .25, .6):
            with self.subTest(time=time), self.assertRaises(ValueError):
                load_g_pi_sample(path, current_time=time)

    def test_event_at_sample_time_uses_strictly_later_subgoal(self):
        path, _, _ = write_g_pi_task(self.root, gripper=[[0.]] * 3 + [[1.]] * 14)
        sample = load_g_pi_sample(path, current_time=.4)
        self.assertAlmostEqual(sample.events[0].time, .4)
        self.assertEqual(sample.subgoal_time, 1.6)
        initial = load_g_pi_sample(path, current_time=0.)
        self.assertEqual(initial.subgoal_time, .4)
        self.assertEqual(initial.actions_mask.sum().item(), 4)

    def test_multiple_events_between_latents_and_simultaneous_effectors(self):
        grip = np.array([[0.]] * 5 + [[1.]] * 4 + [[0.]] * 4 + [[1.]] * 4)
        grip = np.concatenate([grip, 1 - grip], axis=1)
        path, _, arrays = write_g_pi_task(self.root, gripper=grip)
        self.assertEqual(len(arrays["subgoal_times"]), 4)
        for time, expected in ((0., .6), (.4, .6), (.8, 1.), (1.2, 1.4)):
            sample = load_g_pi_sample(path, current_time=time)
            self.assertAlmostEqual(sample.subgoal_time, expected)
            self.assertEqual(len(sample.events), 6)
        self.assertEqual([e.effector for e in sample.events], [0, 1, 0, 1, 0, 1])

    def test_event_at_terminal_shares_one_single_frame_target(self):
        path, _, arrays = write_g_pi_task(self.root, gripper=[[0.]] * 15 + [[1.]] * 2)
        sample = load_g_pi_sample(path, current_time=1.2)
        self.assertEqual(arrays["subgoal_times"].tolist(), [1.6])
        self.assertEqual(sample.events[0].time, sample.terminal_time)
        self.assertEqual(sample.subgoal_time, 1.6)

    def test_g_reads_task_pair_and_pi_never_opens_demo(self):
        path, metadata, _ = write_g_pi_task(self.root)
        sample = load_g_pi_sample(path, route="g_translator", current_time=.4)
        self.assertIs(type(sample), GTranslatorSample)
        self.assertEqual(sample.demonstration.shape, (1, 2, 3, 2, 2))
        self.assertIn(self.root / metadata["demonstration"]["arrays"], g_pi_sample_files(path, sample))
        (self.root / metadata["demonstration"]["arrays"]).unlink()
        sample = load_g_pi_sample(path, current_time=.4)
        self.assertEqual(len(g_pi_sample_files(path, sample)), 4)
        path, _, _ = write_g_pi_task(self.root, "unpaired", demonstration=False)
        load_g_pi_sample(path, current_time=.4)
        with self.assertRaisesRegex(ValueError, "task-paired human"):
            load_g_pi_sample(path, route="g_translator", current_time=.4)

    def test_g_never_loads_text_and_does_not_require_language_files(self):
        path, metadata, _ = write_g_pi_task(self.root)
        (self.root / metadata["language"]).unlink()
        with patch("etude.g_pi_data.load_goal_language", side_effect=AssertionError("G must not read text")):
            sample = load_g_pi_sample(path, route="g_translator", current_time=.4)
        self.assertIsNone(sample.language)
        self.assertIsNone(sample.language_identity)
        self.assertNotIn("language", sample.metadata)
        self.assertEqual(len(g_pi_sample_files(path, sample)), 3)
        metadata.pop("language")
        path.write_text(json.dumps(metadata))
        load_g_pi_sample(path, route="g_translator", current_time=.4)
        with self.assertRaisesRegex(ValueError, "pi training requires"):
            load_g_pi_sample(path, current_time=.4)

    def test_offline_measurements_can_read_robot_goals_without_text_or_demo(self):
        path, metadata, _ = write_g_pi_task(self.root, demonstration=False)
        metadata.pop("language")
        path.write_text(json.dumps(metadata))
        with patch("etude.g_pi_data.load_goal_language", side_effect=AssertionError("no text")):
            sample = load_g_pi_sample(path, current_time=.4, read_language=False)
        self.assertIsNone(sample.language)
        self.assertEqual(len(g_pi_sample_files(path, sample)), 2)
        self.assertNotIn("demonstration", sample.metadata)

    def test_future_sequence_perturbations_never_enter_history_or_single_frame_target(self):
        path, metadata, arrays = write_g_pi_task(self.root)
        baseline = load_g_pi_sample(path, current_time=.4)
        arrays["latent"][:, 2:] += 1000
        arrays["states"][5:] += 1000
        self.save(path, metadata, arrays)
        changed = load_g_pi_sample(path, current_time=.4)
        self.assertTrue(torch.equal(baseline.history, changed.history))
        self.assertTrue(torch.equal(baseline.state, changed.state))
        self.assertTrue(torch.equal(baseline.target_frame, changed.target_frame))
        self.assertEqual(baseline.history.untyped_storage().nbytes(),
                         baseline.history.numel() * baseline.history.element_size())
        self.assertTrue(torch.equal(changed.target_frame, torch.full_like(changed.target_frame, 1006)))

    def test_random_sampling_generator_reproducibility_and_availability_bounds(self):
        path, _, _ = write_g_pi_task(self.root)
        one, two = torch.Generator().manual_seed(12), torch.Generator().manual_seed(12)
        times = [load_g_pi_sample(path, generator=one).metadata["current_time"] for _ in range(12)]
        self.assertEqual(times, [load_g_pi_sample(path, generator=two).metadata["current_time"] for _ in range(12)])
        self.assertGreater(len(set(times)), 1)
        self.assertTrue(all(any(abs(time - allowed) < 1e-6 for allowed in (0., .4, .8, 1.2)) for time in times))

    def test_stride_two_aligns_causal_latents_and_unpadded_future_actions(self):
        path, metadata, arrays = write_g_pi_task(self.root, frame_stride=2)
        sample = load_g_pi_sample(path, current_time=.8)
        self.assertEqual(sample.history_times.tolist(), [0., .8])
        self.assertEqual(sample.actions.shape, (1, 2, 2, 8, 1))
        self.assertEqual(sample.actions[0, 0, 0, :, 0].tolist(), list(range(9, 17)))
        self.assertEqual(sample.actions_mask.sum().item(), 8)
        indices = validate_latent_grid(metadata, torch.from_numpy(arrays["control_times"]),
                                       torch.from_numpy(arrays["latent_available_times"]))
        self.assertEqual(indices.tolist(), [0, 8, 16])
        with self.assertRaisesRegex(ValueError, "actions_per_frame"):
            validate_latent_grid({**metadata, "actions_per_frame": 4},
                                 torch.from_numpy(arrays["control_times"]),
                                 torch.from_numpy(arrays["latent_available_times"]))

    def test_short_task_with_only_initial_latent_and_terminal_off_latent_grid(self):
        path, _, _ = write_g_pi_task(self.root, gripper=np.zeros((3, 1)))
        sample = load_g_pi_sample(path, current_time=0.)
        self.assertEqual(sample.history_times.tolist(), [0.])
        self.assertEqual(sample.subgoal_time, .2)
        self.assertEqual(sample.actions_mask.sum().item(), 2)
        self.assertEqual(sample.actions.flatten()[:2].tolist(), [1., 2.])

    def test_small_control_dt_preserves_exact_event_terminal_and_sample_indices(self):
        grip = np.array([[0.], [0.], [1.], [1.], [1.]], dtype=np.float32)
        path, metadata, arrays = write_g_pi_task(self.root, gripper=grip)
        metadata["control_dt"] = 1e-7
        arrays["control_times"] = np.arange(5, dtype=np.float64) * metadata["control_dt"]
        arrays["latent_available_times"] = arrays["control_times"][::4]
        arrays["subgoal_times"] = arrays["control_times"][[3, 4]]
        self.save(path, metadata, arrays)
        indices = subgoal_control_indices(torch.from_numpy(grip), torch.from_numpy(arrays["control_times"]))
        self.assertEqual(indices.tolist(), [3, 4])
        sample = load_g_pi_sample(path, current_time=0.)
        self.assertEqual(sample.subgoal_time, arrays["control_times"][3])
        self.assertEqual(sample.goal_poses[0, 0, 0, 3].item(), 3.)
        self.assertTrue(torch.equal(sample.target_frame, torch.full_like(sample.target_frame, 1003)))
        self.assertEqual(sample.actions_mask.sum().item(), 3)
        with self.assertRaisesRegex(ValueError, "latent availability"):
            load_g_pi_sample(path, current_time=1e-7)

        path, metadata, arrays = write_g_pi_task(self.root, "small-dt", gripper=np.zeros((9, 1)))
        metadata["control_dt"] = 1e-7
        arrays["control_times"] = np.arange(9, dtype=np.float64) * metadata["control_dt"]
        arrays["latent_available_times"] = arrays["control_times"][::4]
        arrays["subgoal_times"] = arrays["control_times"][[-1]]
        self.save(path, metadata, arrays)
        sample = load_g_pi_sample(path, current_time=4e-7)
        self.assertEqual(sample.metadata["current_time"], 4e-7)
        self.assertEqual(sample.state[0, 0].item(), 16.)
        self.assertEqual(sample.subgoal_time, 8e-7)
        self.assertTrue(torch.equal(sample.target_frame, torch.full_like(sample.target_frame, 1008)))
        invalid_times = torch.from_numpy(arrays["latent_available_times"].copy())
        invalid_times[1] = 3e-7
        with self.assertRaisesRegex(ValueError, "complete causal grid"):
            validate_latent_grid(metadata, torch.from_numpy(arrays["control_times"]), invalid_times)

    def test_metadata_rejects_incompatible_grid_conventions_and_signals(self):
        path, metadata, arrays = write_g_pi_task(self.root)
        for change in ({"success": False}, {"task_start_time": .1}, {"goal_source": "controller_target"},
                       {"pose_units": "cm"}, {"event_rules": {**metadata["event_rules"], "signal_source": "command"}},
                       {"action_frames": False}, {"arrays": "../unsafe.npz"}, {"frame_stride": 0},
                       {"temporal_down_rate": 2}, {"actions_per_frame": 3},
                       {"alignment": "one_latent_per_control_step"}, {"subgoal_encoding": "sequence_slice"}):
            self.save(path, {**metadata, **change}, arrays)
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_g_pi_sample(path, current_time=.4)
        self.save(path, metadata, arrays)
        for times in (arrays["control_times"] + .1, arrays["control_times"] * 2):
            self.save(path, metadata, {**arrays, "control_times": times})
            with self.assertRaisesRegex(ValueError, "every control step"):
                load_g_pi_sample(path, current_time=.4)

    def test_command_gripper_requires_explicit_evidence_and_remains_weak(self):
        path, metadata, arrays = write_g_pi_task(self.root)
        metadata["event_rules"]["signal_source"] = "command"
        metadata["gripper_signal_source"] = "command"
        self.save(path, metadata, arrays)
        with self.assertRaisesRegex(ValueError, "gripper_source_evidence"):
            load_g_pi_sample(path, current_time=.4)
        metadata["gripper_source_evidence"] = "fixture collector stores commanded finger target"
        self.save(path, metadata, arrays)
        sample = load_g_pi_sample(path, current_time=.4)
        self.assertEqual(sample.metadata["gripper_signal_source"], "command")
        self.assertEqual(sample.metadata["subgoal_annotation"]["detector_version"], "command_gripper_v1")
        self.assertTrue(sample.metadata["subgoal_annotation"]["weak_label"])
        self.assertEqual(sample.metadata["subgoal_annotation"]["source_evidence"], metadata["gripper_source_evidence"])
        torch.testing.assert_close(sample.goal_gripper, torch.ones(1, 1), rtol=0, atol=0)
        for source in ("candidate_match", "sim_relation", "pedal"):
            self.save(path, {**metadata, "subgoal_source": source}, arrays)
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "command gripper"):
                load_g_pi_sample(path, current_time=.4)

    def test_rejects_latents_before_raw_coverage_is_available_and_missing_history(self):
        path, metadata, arrays = write_g_pi_task(self.root)
        bad = arrays["latent_available_times"].copy()
        bad[1] = .3
        for change in ({"latent_available_times": bad},
                       {"latent": arrays["latent"][:, 1:], "latent_available_times": arrays["latent_available_times"][1:]},
                       {"latent": arrays["latent"][:, :-1], "latent_available_times": arrays["latent_available_times"][:-1]}):
            self.save(path, metadata, {**arrays, **change})
            with self.subTest(change=list(change)), self.assertRaisesRegex(ValueError, "complete causal grid"):
                load_g_pi_sample(path, current_time=.4)

    def test_rejects_missing_stale_or_sequence_subgoal_caches(self):
        path, metadata, arrays = write_g_pi_task(self.root)
        for change in ({"subgoal_times": arrays["subgoal_times"][:1]},
                       {"subgoal_times": arrays["subgoal_times"][1:]},
                       {"subgoal_times": np.array([.5, 1.6])},
                       {"subgoal_times": arrays["subgoal_times"] + 1e-8},
                       {"subgoal_times": np.array([.6, .6, 1.6])},
                       {"subgoal_latents": np.ones((2, 2, 2, 2, 2), dtype=np.float32)},
                       {"subgoal_latents": np.full((2, 2, 1, 2, 2), np.nan, dtype=np.float32)}):
            self.save(path, metadata, {**arrays, **change})
            with self.subTest(change=list(change)), self.assertRaisesRegex(ValueError, "subgoal"):
                load_g_pi_sample(path, current_time=.4)
        changed = arrays["gripper"].copy()
        changed[5] = 0.
        self.save(path, metadata, {**arrays, "gripper": changed})
        with self.assertRaisesRegex(ValueError, "recomputed"):
            load_g_pi_sample(path, current_time=.4)

    def test_version_two_rejected_with_subgoal_migration_reason(self):
        path, metadata, arrays = write_g_pi_task(self.root)
        self.save(path, {**metadata, "format_version": 2}, arrays)
        with self.assertRaisesRegex(ValueError, "version 2 is unsupported.*auditable subgoal"):
            load_g_pi_sample(path)

    def test_version_one_rejected_with_migration_reason(self):
        path, metadata, arrays = write_g_pi_task(self.root)
        self.save(path, {**metadata, "format_version": 1}, arrays)
        with self.assertRaisesRegex(ValueError, "version 1 is unsupported.*separate control and latent grids"):
            load_g_pi_sample(path)

    def test_native_history_padding_is_not_shifted_into_future_labels(self):
        import ast
        source = Path(__file__).resolve().parents[1] / "third_party/Zero-WAM/wan_va/dataset/robotwin_action.py"
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "preprocess_robotwin_actions")
        namespace = {"np": np, "MODEL_ACTION_DIM": 30, "relative_robotwin_action": lambda action, state: action}
        exec(compile(ast.Module([function], type_ignores=[]), str(source), "exec"), namespace)
        raw = np.arange(17 * 16, dtype=np.float64).reshape(17, 16) / 300.
        kwargs = dict(action=raw, state=raw, q01=-np.ones(30), q99=np.ones(30),
                      inverse_used_action_channel_ids=list(range(16)) + [16] * 14)
        unpadded, valid = namespace["preprocess_robotwin_actions"](**kwargs, history_size=0, required_size=17)
        padded, _ = namespace["preprocess_robotwin_actions"](**kwargs, history_size=4, required_size=20)
        np.testing.assert_array_equal(padded[:4], 0.)
        np.testing.assert_array_equal(padded[4:8], unpadded[:4])
        path, metadata, arrays = write_g_pi_task(self.root, gripper=np.zeros((17, 1)))
        metadata["action_space"].update(dimension=30, valid_channels=valid[0].tolist())
        arrays.update(actions=unpadded.T.astype(np.float32), actions_mask=valid.T)
        self.save(path, metadata, arrays)
        initial = load_g_pi_sample(path, current_time=0.)
        np.testing.assert_array_equal(initial.actions[0, :, 0, :, 0].numpy(), padded[4:8].T.astype(np.float32))
        following = load_g_pi_sample(path, current_time=.4)
        np.testing.assert_array_equal(following.actions[0, :, 0, :, 0].numpy(), padded[8:12].T.astype(np.float32))

    def test_index_audits_source_splits_without_array_reads(self):
        first, _, _ = write_g_pi_task(self.root, "first")
        second, metadata, _ = write_g_pi_task(self.root, "second")
        index = self.root / "index.json"
        index.write_text(json.dumps({"format_version": 1, "kind": "g_pi_index", "samples": [
            {"manifest": first.name, "split": "train"}, {"manifest": second.name, "split": "test"}]}))
        with patch("numpy.load", side_effect=AssertionError("metadata-only index")):
            selected, records = load_g_pi_index(index)
        self.assertEqual(selected, [first])
        self.assertEqual(len(records), 4)
        metadata["demonstration"]["source_group"] = "first-demo"
        second.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "split"):
            load_g_pi_index(index)


if __name__ == "__main__":
    unittest.main()
