from dataclasses import asdict
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from evo_wam.g_pi_data import (EventRules, GTranslatorSample, GripperEventDetector, PiGoalSample,
                              detect_gripper_events, g_pi_sample_files, load_g_pi_index,
                              load_g_pi_sample, next_subgoal_time)
from evo_wam.icl_data import LATENT_NORMALIZATION
from test_goal_language import write_goal_language


def write_g_pi_task(root, name="task", *, gripper=None, demonstration=True):
    if gripper is None:
        gripper = np.array([[0.], [0.], [0.], [1.], [1.], [1.], [1.], [1.]], dtype=np.float32)
    gripper = np.asarray(gripper, dtype=np.float32)
    frames, effectors = gripper.shape
    metadata = {
        "format_version": 1, "kind": "g_pi_task", "sample_id": name, "arrays": f"{name}.npz",
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
        "action_frames": 2, "actions_per_frame": 3, "task_start_time": 0., "success": True,
        "event_rules": asdict(EventRules()),
    }
    write_goal_language(root, f"{name}-language")
    poses = np.tile(np.eye(4, dtype=np.float32), (frames, effectors, 1, 1))
    poses[:, :, 0, 3] = np.arange(frames, dtype=np.float32)[:, None]
    arrays = {"latent": np.arange(2 * frames * 4, dtype=np.float32).reshape(2, frames, 2, 2),
              "frame_times": np.arange(frames, dtype=np.float64) * .1,
              "states": np.arange(frames * 4, dtype=np.float32).reshape(frames, 4),
              "poses": poses, "gripper": gripper,
              "actions": np.arange(2 * frames, dtype=np.float32).reshape(2, frames),
              "actions_mask": np.ones((2, frames), dtype=np.bool_)}
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
        for values in ({"signal_source": "command"}, {"close_threshold": .8}, {"debounce_steps": True},
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

    def test_midchunk_event_masks_actions_and_slices_history(self):
        path, _, arrays = write_g_pi_task(self.root)
        sample = load_g_pi_sample(path, current_time=.2)
        self.assertIs(type(sample), PiGoalSample)
        self.assertEqual(sample.subgoal_time, .4)
        self.assertAlmostEqual(sample.terminal_time, .7)
        self.assertEqual(sample.history.shape, (1, 2, 3, 2, 2))
        self.assertEqual(sample.target_frame.shape, (1, 2, 1, 2, 2))
        self.assertEqual(sample.actions.shape, (1, 2, 2, 3, 1))
        self.assertEqual(sample.actions_mask.flatten().tolist(), [True, True] + [False] * 10)
        torch.testing.assert_close(sample.target_frame[0, :, 0], torch.from_numpy(arrays["latent"][:, 4]))
        torch.testing.assert_close(sample.goal_poses[0], torch.from_numpy(arrays["poses"][4]))
        self.assertFalse(hasattr(sample, "demonstration"))
        self.assertNotIn("demonstration", inspect.signature(PiGoalSample).parameters)
        self.assertNotIn("demonstration", sample.metadata)

    def test_after_last_event_uses_terminal_including_remaining_motion(self):
        path, _, _ = write_g_pi_task(self.root)
        for time in (.4, .5, .6):
            sample = load_g_pi_sample(path, current_time=time)
            self.assertAlmostEqual(sample.subgoal_time, .7)
            self.assertGreater(sample.subgoal_time, sample.metadata["current_time"])
        self.assertEqual(load_g_pi_sample(path, current_time=.6).actions_mask.sum().item(), 1)

    def test_no_events_target_terminal_only(self):
        path, _, _ = write_g_pi_task(self.root, gripper=np.zeros((8, 1)))
        for time in (0., .2, .6):
            sample = load_g_pi_sample(path, current_time=time)
            self.assertEqual(sample.events, ())
            self.assertAlmostEqual(sample.subgoal_time, .7)

    def test_terminal_and_after_terminal_rejected(self):
        path, _, _ = write_g_pi_task(self.root)
        for time in (.7, .8, 1., -.1, .25):
            with self.subTest(time=time), self.assertRaises(ValueError):
                load_g_pi_sample(path, current_time=time)

    def test_g_reads_task_pair_and_pi_never_opens_demo(self):
        path, metadata, _ = write_g_pi_task(self.root)
        sample = load_g_pi_sample(path, route="g_translator", current_time=.2)
        self.assertIs(type(sample), GTranslatorSample)
        self.assertEqual(sample.demonstration.shape, (1, 2, 3, 2, 2))
        self.assertIn(self.root / metadata["demonstration"]["arrays"], g_pi_sample_files(path, sample))
        (self.root / metadata["demonstration"]["arrays"]).unlink()
        sample = load_g_pi_sample(path, current_time=.2)
        self.assertEqual(len(g_pi_sample_files(path, sample)), 4)
        path, _, _ = write_g_pi_task(self.root, "unpaired", demonstration=False)
        load_g_pi_sample(path, current_time=.2)
        with self.assertRaisesRegex(ValueError, "task-paired human"):
            load_g_pi_sample(path, route="g_translator", current_time=.2)

    def test_future_frame_perturbations_never_enter_history(self):
        path, metadata, arrays = write_g_pi_task(self.root)
        baseline = load_g_pi_sample(path, current_time=.2)
        arrays["latent"][:, 3:] += 1000
        arrays["states"][3:] += 1000
        np.savez_compressed(self.root / metadata["arrays"], **arrays)
        changed = load_g_pi_sample(path, current_time=.2)
        self.assertTrue(torch.equal(baseline.history, changed.history))
        self.assertTrue(torch.equal(baseline.state, changed.state))
        self.assertEqual(baseline.history.untyped_storage().nbytes(),
                         baseline.history.numel() * baseline.history.element_size())

    def test_random_sampling_generator_reproducibility_and_bounds(self):
        path, _, _ = write_g_pi_task(self.root)
        one, two = torch.Generator().manual_seed(12), torch.Generator().manual_seed(12)
        times = [load_g_pi_sample(path, generator=one).metadata["current_time"] for _ in range(12)]
        self.assertEqual(times, [load_g_pi_sample(path, generator=two).metadata["current_time"] for _ in range(12)])
        self.assertGreater(len(set(times)), 1)
        self.assertTrue(all(0 <= time < .7 for time in times))

    def test_metadata_rejects_incompatible_frames_and_signals(self):
        path, metadata, arrays = write_g_pi_task(self.root)
        for change in ({"success": False}, {"task_start_time": .1}, {"goal_source": "controller_target"},
                       {"pose_units": "cm"}, {"event_rules": {**metadata["event_rules"], "signal_source": "command"}},
                       {"action_frames": False}, {"arrays": "../unsafe.npz"}):
            path.write_text(json.dumps({**metadata, **change}))
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_g_pi_sample(path, current_time=.2)
        path.write_text(json.dumps(metadata))
        for times in (arrays["frame_times"] + .1, arrays["frame_times"] * 2):
            np.savez_compressed(self.root / metadata["arrays"], **{**arrays, "frame_times": times})
            with self.assertRaisesRegex(ValueError, "every control step"):
                load_g_pi_sample(path, current_time=.2)

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
