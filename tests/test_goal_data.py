import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from etude.goal_data import load_goal_index, load_goal_observation, load_goal_sample
from test_icl_data import LATENT_NORMALIZATION, write_sample
from test_goal_language import write_goal_language


def write_goal_sample(root, name="goal", *, visual=False):
    """Synthetic measured/controller labels are explicit, never derived from actions."""
    metadata = {
        "format_version": 2, "kind": "se3_goal_sample", "sample_id": name, "arrays": f"{name}.npz",
        "robot_source": {"source_id": f"{name}-robot", "source_group": f"{name}-robot",
                         "domain": "robot", "trajectory_id": f"{name}-trajectory"},
        "action_space": {"representation": "zero-wam-normalized", "normalization_id": "fixture",
                         "dimension": 2, "valid_channels": [True, False]},
        "state_space_id": "fixture-joint-state", "coordinate_frame": "robot_base", "pose_units": "m",
        "goal_source": "measured_endpoint", "end_effectors": ["gripper"],
        "pose_representation": "absolute_robot_base_tool", "tool_frames": ["gripper_tool"],
        "gripper_space": {"normalization_id": "fixture-gripper-width", "closed": [0.], "open": [.08], "units": "m"},
        "language": f"{name}-language.json",
        "current_time": 0.3, "goal_time": 0.9, "control_dt": 0.1,
    }
    write_goal_language(root, f"{name}-language")
    arrays = {"state": np.arange(4, dtype=np.float32), "goal_poses": np.eye(4, dtype=np.float32)[None],
              "goal_gripper": np.array([.5], dtype=np.float32),
              "actions": np.ones((2, 2, 3, 1), dtype=np.float32),
              "actions_mask": np.ones((2, 2, 3, 1), dtype=np.bool_)}
    if visual:
        pair_path, pair, _, _ = write_sample(root, f"{name}-visual", robot=True)
        pair["target"].update(metadata["robot_source"])
        pair_path.write_text(json.dumps(pair))
        metadata["visual_pair"] = pair_path.name
    path = root / f"{name}.json"
    path.write_text(json.dumps(metadata))
    np.savez_compressed(root / metadata["arrays"], **arrays)
    return path, metadata, arrays


def write_goal_observation(root, name="observation"):
    metadata = {
        "format_version": 2, "kind": "se3_goal_observation", "arrays": f"{name}.npz",
        "demonstration": {"arrays": f"{name}-demo.npz", "source_id": f"{name}-demo",
                          "source_group": f"{name}-demo", "domain": "human"},
        "feature_space_id": "fixture-wan-v1", "latent_normalization": LATENT_NORMALIZATION,
        "action_space": {"representation": "zero-wam-normalized", "normalization_id": "fixture",
                         "dimension": 2, "valid_channels": [True, False]},
        "state_space_id": "fixture-joint-state", "coordinate_frame": "robot_base", "pose_units": "m",
        "end_effectors": ["gripper"], "current_time": 0.3, "control_dt": 0.1, "actions_per_frame": 3,
        "pose_representation": "absolute_robot_base_tool", "tool_frames": ["gripper_tool"],
        "gripper_space": {"normalization_id": "fixture-gripper-width", "closed": [0.], "open": [.08], "units": "m"},
        "language": f"{name}-language.json",
    }
    write_goal_language(root, f"{name}-language")
    arrays = {"state": np.arange(4, dtype=np.float32),
              "history_latent": np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2),
              "history_times": np.array([0., 0.3], dtype=np.float64)}
    demo = {"latent": np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2),
            "frame_times": np.array([0., 0.2, 0.4], dtype=np.float64)}
    path = root / f"{name}.json"
    path.write_text(json.dumps(metadata))
    np.savez_compressed(root / metadata["arrays"], **arrays)
    np.savez_compressed(root / metadata["demonstration"]["arrays"], **demo)
    return path, metadata, arrays, demo


class GoalDataTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def save(self, path, metadata, arrays):
        path.write_text(json.dumps(metadata))
        np.savez_compressed(path.parent / metadata["arrays"], **arrays)

    def write_index(self, samples, **extra):
        path = self.root / "index.json"
        path.write_text(json.dumps({"format_version": 2, "kind": "se3_goal_index", "samples": samples, **extra}))
        return path

    def test_stage1_reads_only_goal_arrays_and_keeps_goal_semantics_explicit(self):
        path, metadata, arrays = write_goal_sample(self.root)
        metadata["visual_pair"] = "missing.json"
        original_load = np.load
        for source in ("measured_endpoint", "controller_target"):
            metadata["goal_source"] = source
            self.save(path, metadata, arrays)
            with patch("etude.goal_data.load_icl_sample", side_effect=AssertionError("Stage 1 must not load videos")), \
                    patch("numpy.load", wraps=original_load) as loads:
                sample = load_goal_sample(path)
            self.assertEqual(loads.call_count, 2)
            self.assertEqual([call.args[0] for call in loads.call_args_list],
                             [self.root / "goal.npz", self.root / "goal-language.npz"])
            self.assertEqual(sample.metadata["goal_source"], source)
            self.assertEqual(sample.state.shape, (1, 4))
            self.assertEqual(sample.goal_poses.shape, (1, 1, 4, 4))
            self.assertEqual(sample.goal_gripper.shape, (1, 1))
            self.assertEqual(sample.language.shape, (1, 3, 8))
            self.assertEqual(sample.actions.shape, (1, 2, 2, 3, 1))
            self.assertIsNone(sample.visual)
            torch.testing.assert_close(sample.goal_poses[0], torch.from_numpy(arrays["goal_poses"]))
        self.save(path, {**metadata, "goal_source": "inferred_from_normalized_actions"}, arrays)
        with self.assertRaisesRegex(ValueError, "goal_source"):
            load_goal_sample(path)

    def test_metadata_requires_robot_goal_identity_units_and_time_conventions(self):
        path, metadata, arrays = write_goal_sample(self.root)
        invalid = [{"robot_source": {**metadata["robot_source"], "domain": "human"}},
                   {"robot_source": {key: value for key, value in metadata["robot_source"].items() if key != "trajectory_id"}},
                   {"state_space_id": " "}, {"coordinate_frame": ""}, {"pose_units": "mm"},
                   {"end_effectors": []}, {"end_effectors": ["gripper", "gripper"]},
                   {"end_effectors": [1]}, {"goal_time": 0.3}, {"control_dt": 0},
                   {"control_dt": True}, {"goal_time": float("nan")}, {"current_time": float("inf")},
                   {"visual_pair": "../outside.json"}, {"arrays": str(self.root / "absolute.npz")},
                   {"provenance": "text"}, {"human_poses": "invented"}]
        invalid += [{"format_version": 1}, {"pose_representation": "relative"},
                    {"tool_frames": []}, {"tool_frames": [""]}, {"language": "../outside.json"},
                    {"gripper_space": {**metadata["gripper_space"], "closed": [.08]}},
                    {"gripper_space": {**metadata["gripper_space"], "open": [float("nan")]}},
                    {"gripper_space": {**metadata["gripper_space"], "closed": [False]}},
                    {"gripper_space": {**metadata["gripper_space"], "units": ""}}]
        for changed in invalid:
            path.write_text(json.dumps({**metadata, **changed}))
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                load_goal_sample(path)

    def test_poses_reject_invalid_rotation_reflection_bottom_row_and_nonfinite(self):
        path, metadata, arrays = write_goal_sample(self.root)
        invalid = []
        for row, column, value in ((0, 0, 2), (0, 0, -1), (0, 1, 0.2), (3, 0, 1),
                                   (3, 3, 0), (2, 3, np.nan), (1, 1, np.inf)):
            poses = arrays["goal_poses"].copy()
            poses[0, row, column] = value
            invalid.append(poses)
        invalid += [np.eye(4, dtype=np.int64)[None], np.eye(4, dtype=np.float32),
                    np.repeat(arrays["goal_poses"], 2, axis=0)]
        for poses in invalid:
            self.save(path, metadata, {**arrays, "goal_poses": poses})
            with self.subTest(poses=poses), self.assertRaises(ValueError):
                load_goal_sample(path)
        poses = arrays["goal_poses"].copy()
        poses[0, :3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        poses[0, :3, 3] = [0.1, -0.2, 0.4]
        self.save(path, metadata, {**arrays, "goal_poses": poses})
        torch.testing.assert_close(load_goal_sample(path).goal_poses[0], torch.from_numpy(poses))

    def test_invalid_actions_are_sanitized_and_active_future_is_required(self):
        path, metadata, arrays = write_goal_sample(self.root)
        arrays["actions"][1] = np.nan
        arrays["actions"][0, 0, 0, 0] = np.nan
        arrays["actions_mask"][0, 0, 0, 0] = False
        self.save(path, metadata, arrays)
        sample = load_goal_sample(path)
        self.assertTrue(torch.isfinite(sample.actions).all())
        self.assertEqual(sample.actions[0, 0, 0, 0, 0], 0)
        self.assertFalse(sample.actions_mask[:, 1].any())
        arrays["actions_mask"][0, 0, 0, 0] = True
        self.save(path, metadata, arrays)
        with self.assertRaisesRegex(ValueError, "finite"):
            load_goal_sample(path)
        arrays["actions_mask"][0] = False
        self.save(path, metadata, arrays)
        with self.assertRaisesRegex(ValueError, "future action supervision"):
            load_goal_sample(path)

    def test_npz_requires_exact_floating_state_pose_action_and_boolean_mask_shapes(self):
        path, metadata, arrays = write_goal_sample(self.root)
        changes = [{"state": np.zeros(4, dtype=np.int64)}, {"state": np.zeros((1, 4), dtype=np.float32)},
                   {"state": np.full(4, np.nan, dtype=np.float32)},
                   {"actions": np.ones((2, 2, 3), dtype=np.float32)},
                   {"actions": np.ones_like(arrays["actions"], dtype=np.int64)},
                   {"actions_mask": arrays["actions_mask"].astype(np.int64)},
                   {"actions_mask": np.ones((2, 2, 2, 1), dtype=np.bool_)},
                   {"unknown_label": np.ones(1)}]
        changes += [{"goal_gripper": np.array([value], dtype=np.float32)} for value in (-.1, 1.1, np.nan)]
        changes += [{"goal_gripper": np.array([1], dtype=np.int64)},
                    {"goal_gripper": np.zeros((1, 1), dtype=np.float32)}]
        for changed in changes:
            self.save(path, metadata, {**arrays, **changed})
            with self.subTest(changed=list(changed)), self.assertRaises(ValueError):
                load_goal_sample(path)
        self.save(path, metadata, {key: value for key, value in arrays.items() if key != "state"})
        with self.assertRaisesRegex(ValueError, "exactly"):
            load_goal_sample(path)

    def test_single_future_block_duration_and_stage2_time_alignment(self):
        path, metadata, arrays = write_goal_sample(self.root, visual=True)
        sample = load_goal_sample(path, visual=True)
        self.assertIsNotNone(sample.visual)
        torch.testing.assert_close(sample.actions, sample.visual.actions[:, :, 2:])
        for changed, error in (({"control_dt": 0.2}, "F\\*N\\*control_dt"),
                               ({"current_time": 0.4, "goal_time": 1.0}, "history endpoint/final time"),
                               ({"goal_time": 0.8}, "F\\*N\\*control_dt")):
            self.save(path, {**metadata, **changed}, arrays)
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, error):
                load_goal_sample(path, visual=True)
        self.save(path, {key: value for key, value in metadata.items() if key != "visual_pair"}, arrays)
        with self.assertRaisesRegex(ValueError, "visual_pair"):
            load_goal_sample(path, visual=True)

    def test_stage2_rejects_robot_identity_action_space_future_actions_and_masks(self):
        path, metadata, arrays = write_goal_sample(self.root, visual=True)
        pair_path = self.root / metadata["visual_pair"]
        pair = json.loads(pair_path.read_text())
        for name in ("source_id", "source_group", "trajectory_id"):
            changed = copy.deepcopy(pair)
            changed["target"][name] = "other"
            pair_path.write_text(json.dumps(changed))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "robot_source"):
                load_goal_sample(path, visual=True)
        changed = copy.deepcopy(pair)
        changed["target"]["action_space"]["normalization_id"] = "other"
        pair_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "action_space"):
            load_goal_sample(path, visual=True)
        pair_path.write_text(json.dumps(pair))
        target_path = self.root / pair["target"]["arrays"]
        with np.load(target_path) as archive:
            target = {name: archive[name].copy() for name in archive.files}
        for key, value in (("actions", 0.5), ("actions_mask", False)):
            changed = {name: data.copy() for name, data in target.items()}
            changed[key][0, 2, 0, 0] = value
            np.savez_compressed(target_path, **changed)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "future actions and validity"):
                load_goal_sample(path, visual=True)
        np.savez_compressed(target_path, **target)
        changed = {**metadata, "goal_time": 0.6}
        shortened = {**arrays, "actions": arrays["actions"][:, :1], "actions_mask": arrays["actions_mask"][:, :1]}
        self.save(path, changed, shortened)
        with self.assertRaisesRegex(ValueError, "future frame count"):
            load_goal_sample(path, visual=True)

    def test_stage2_checks_every_future_time_without_restricting_history_cadence(self):
        path, metadata, _ = write_goal_sample(self.root, visual=True)
        pair = json.loads((self.root / metadata["visual_pair"]).read_text())
        target_path = self.root / pair["target"]["arrays"]
        with np.load(target_path) as archive:
            target = {name: archive[name].copy() for name in archive.files}
        target["frame_times"][0] = 0.29  # Observed history need not have the generated future cadence.
        np.savez_compressed(target_path, **target)
        load_goal_sample(path, visual=True)
        target["frame_times"][2] = 0.31  # Same history/goal endpoints, wrong first future frame time.
        np.savez_compressed(target_path, **target)
        with self.assertRaisesRegex(ValueError, "future times.*N\\*control_dt"):
            load_goal_sample(path, visual=True)

    def test_goal_visual_robot_reference_must_be_independent_from_target(self):
        path, metadata, _ = write_goal_sample(self.root, visual=True)
        pair_path = self.root / metadata["visual_pair"]
        pair = json.loads(pair_path.read_text())
        pair["demonstration"]["domain"] = "robot"
        pair["demonstration"]["trajectory_id"] = "independent-demo-trajectory"
        index = self.write_index([{"manifest": path.name, "split": "train"}])
        pair_path.write_text(json.dumps(pair))
        load_goal_sample(path, visual=True)
        load_goal_index(index)
        for name in ("source_id", "source_group", "trajectory_id"):
            changed = copy.deepcopy(pair)
            changed["demonstration"][name] = pair["target"][name]
            pair_path.write_text(json.dumps(changed))
            with self.subTest(name=name, path="sample"), self.assertRaisesRegex(ValueError, "independent recording/trajectory"):
                load_goal_sample(path, visual=True)
            with self.subTest(name=name, path="index"), self.assertRaisesRegex(ValueError, "independent recording/trajectory"):
                load_goal_index(index)
        # A human-to-robot provenance bridge does not establish identical video pixels.
        human = copy.deepcopy(pair)
        human["demonstration"].pop("trajectory_id")
        human["demonstration"]["domain"] = "human"
        pair_path.write_text(json.dumps(human))
        index = self.write_index([{"manifest": path.name, "split": "train"}], bridge_sources=[{
            "source_id": human["demonstration"]["source_id"],
            "trajectory_id": pair["target"]["trajectory_id"], "split": "train"}])
        load_goal_sample(path, visual=True)
        load_goal_index(index)

    def test_index_audits_robot_visual_sources_all_splits_without_arrays(self):
        train, _, _ = write_goal_sample(self.root, "train", visual=True)
        test, metadata, _ = write_goal_sample(self.root, "test", visual=True)
        index = self.write_index([{"manifest": train.name, "split": "train"}, {"manifest": test.name, "split": "test"}])
        with patch("numpy.load", side_effect=AssertionError("index is metadata-only")):
            selected, records = load_goal_index(index)
        self.assertEqual(selected, [train])
        self.assertEqual(len(records), 6)
        pair_path = self.root / metadata["visual_pair"]
        pair = json.loads(pair_path.read_text())
        pair["demonstration"]["source_id"] = "train-visual-A"
        pair_path.write_text(json.dumps(pair))
        with self.assertRaisesRegex(ValueError, "crosses"):
            load_goal_index(index)

    def test_index_preserves_transitive_alias_bridge_and_trajectory_leakage(self):
        train, _, _ = write_goal_sample(self.root, "train")
        test, metadata, _ = write_goal_sample(self.root, "test")
        entries = [{"manifest": train.name, "split": "train"}, {"manifest": test.name, "split": "test"}]
        index = self.write_index(entries, source_aliases=[{"source_id": "alias", "source_group": "train-robot",
                                                          "domain": "robot", "split": "train"}],
                                 bridge_sources=[{"source_id": "alias", "trajectory_id": "test-trajectory", "split": "train"}])
        with patch("numpy.load", side_effect=AssertionError("index is metadata-only")):
            with self.assertRaisesRegex(ValueError, "crosses"):
                load_goal_index(index)
        index = self.write_index(entries)
        metadata["robot_source"]["trajectory_id"] = "train-trajectory"
        test.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "crosses"):
            load_goal_index(index)

    def test_index_rejects_duplicates_and_path_escape(self):
        path, _, _ = write_goal_sample(self.root)
        for entries in ([{"manifest": path.name, "split": "train"}] * 2,
                        [{"manifest": "../outside.json", "split": "train"}],
                        [{"manifest": path.name, "split": "invalid"}]):
            index = self.write_index(entries)
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                load_goal_index(index)

    def test_index_rejects_mixed_measured_and_controller_targets(self):
        train, _, _ = write_goal_sample(self.root, "train")
        other, metadata, _ = write_goal_sample(self.root, "other")
        metadata["goal_source"] = "controller_target"
        other.write_text(json.dumps(metadata))
        index = self.write_index([{"manifest": train.name, "split": "train"},
                                  {"manifest": other.name, "split": "train"}])
        with self.assertRaisesRegex(ValueError, "must not mix"):
            load_goal_index(index)


class GoalObservationTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_loads_only_observed_history_state_and_demo(self):
        path, metadata, arrays, demo = write_goal_observation(self.root)
        # Inference control frequency does not invent or resample observation timestamps.
        metadata["control_dt"] = 0.007
        path.write_text(json.dumps(metadata))
        sample = load_goal_observation(path)
        self.assertEqual(sample.state.shape, (1, 4))
        self.assertEqual(sample.history.shape, (1, 2, 2, 2, 2))
        self.assertEqual(sample.demonstration.shape, (1, 2, 3, 2, 2))
        self.assertEqual(sample.language.shape, (1, 3, 8))
        torch.testing.assert_close(sample.state[0], torch.from_numpy(arrays["state"]))
        torch.testing.assert_close(sample.history[0], torch.from_numpy(arrays["history_latent"]))
        torch.testing.assert_close(sample.history_times, torch.from_numpy(arrays["history_times"]))
        torch.testing.assert_close(sample.demonstration[0], torch.from_numpy(demo["latent"]))
        torch.testing.assert_close(sample.demonstration_times, torch.from_numpy(demo["frame_times"]))
        self.assertFalse(hasattr(sample, "goal_poses"))
        self.assertFalse(hasattr(sample, "actions"))

    def test_rejects_supervision_in_metadata_history_archive_or_demo_archive(self):
        path, metadata, arrays, demo = write_goal_observation(self.root)
        for name in ("goal_poses", "goal_gripper", "future_latent", "actions", "actions_mask", "goal_time", "goal_source", "target"):
            path.write_text(json.dumps({**metadata, name: "not-an-inference-input"}))
            with self.subTest(location="metadata", name=name), self.assertRaisesRegex(ValueError, "without supervision"):
                load_goal_observation(path)
        path.write_text(json.dumps(metadata))
        for name in ("goal_poses", "future_latent", "actions", "actions_mask"):
            np.savez_compressed(self.root / metadata["arrays"], **arrays, **{name: np.zeros(1)})
            with self.subTest(location="history", name=name), self.assertRaisesRegex(ValueError, "without supervision"):
                load_goal_observation(path)
        np.savez_compressed(self.root / metadata["arrays"], **arrays)
        for name in ("goal_poses", "future_latent", "actions", "actions_mask"):
            np.savez_compressed(self.root / metadata["demonstration"]["arrays"], **demo, **{name: np.zeros(1)})
            with self.subTest(location="demonstration", name=name), self.assertRaisesRegex(ValueError, "exactly"):
                load_goal_observation(path)

    def test_rejects_invalid_inference_metadata_and_demonstration_identity(self):
        path, metadata, _, _ = write_goal_observation(self.root)
        invalid = [{"actions_per_frame": 0}, {"actions_per_frame": True}, {"actions_per_frame": 2.5},
                   {"feature_space_id": " "}, {"latent_normalization": "raw"}, {"current_time": 0.31},
                   {"state_space_id": ""}, {"coordinate_frame": ""}, {"pose_units": "cm"},
                   {"end_effectors": []}, {"end_effectors": ["gripper", "gripper"]},
                   {"control_dt": 0}, {"current_time": float("nan")}, {"provenance": []},
                   {"arrays": "../outside.npz"},
                   {"demonstration": {**metadata["demonstration"], "actions": "labels.npz"}},
                   {"demonstration": {**metadata["demonstration"], "trajectory_id": "fake-human-pose"}},
                   {"demonstration": {**metadata["demonstration"], "arrays": "../outside.npz"}}]
        for changed in invalid:
            path.write_text(json.dumps({**metadata, **changed}))
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                load_goal_observation(path)

    def test_rejects_invalid_state_history_timestamps_and_channel_mismatch(self):
        path, metadata, arrays, demo = write_goal_observation(self.root)
        invalid = [{"state": np.zeros(4, dtype=np.int64)}, {"state": np.full(4, np.nan)},
                   {"history_latent": np.zeros((2, 0, 2, 2), dtype=np.float32)},
                   {"history_latent": np.zeros((2, 2, 2), dtype=np.float32)},
                   {"history_latent": np.zeros_like(arrays["history_latent"], dtype=np.int64)},
                   {"history_latent": np.full_like(arrays["history_latent"], np.inf)},
                   {"history_times": np.array([0, 1], dtype=np.int64)},
                   {"history_times": np.array([0.3, 0.3])}, {"history_times": np.array([0.4, 0.3])},
                   {"history_times": np.array([0., np.nan])}, {"history_times": np.array([0.3])}]
        for changed in invalid:
            np.savez_compressed(self.root / metadata["arrays"], **{**arrays, **changed})
            with self.subTest(changed=list(changed)), self.assertRaises(ValueError):
                load_goal_observation(path)
        np.savez_compressed(self.root / metadata["arrays"], **arrays)
        np.savez_compressed(self.root / metadata["demonstration"]["arrays"], **{**demo, "latent": demo["latent"][:1]})
        with self.assertRaisesRegex(ValueError, "channel dimension"):
            load_goal_observation(path)


if __name__ == "__main__":
    unittest.main()
