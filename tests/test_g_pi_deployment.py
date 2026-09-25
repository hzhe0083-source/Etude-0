import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from etude.g_pi_deployment import (GObservation, PiObservation, load_g_pi_observation,
                                     load_goal_prediction, save_goal_prediction)
from etude.goal_language import load_goal_language
from test_goal_data import write_goal_observation


class GDeploymentTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path, self.metadata, self.arrays, _ = write_goal_observation(self.root)
        self.metadata.update(format_version=3, kind="g_pi_observation", current_time=.5,
                             actions_per_frame=4, frame_stride=1, temporal_down_rate=4,
                             alignment="zerowam_causal_first_then_four", subgoal_encoding="wan_vae_single_frame")
        self.arrays.pop("history_times")
        self.arrays["latent_available_times"] = np.array([0., .4], dtype=np.float64)
        self.language_path = self.metadata.pop("language")
        _, language_identity = load_goal_language(self.root / self.language_path)
        keys = ("action_space", "state_space_id", "coordinate_frame", "pose_units", "end_effectors",
                "control_dt", "actions_per_frame", "pose_representation", "tool_frames", "gripper_space",
                "frame_stride", "temporal_down_rate", "alignment", "subgoal_encoding")
        self.registry = {key: self.metadata[key] for key in keys}
        self.registry.update(language_identity=language_identity, goal_source="measured_endpoint", subgoal_source="gripper",
                             event_rules={"signal_source": "measured"})
        self.identity = {"k_z": 16, "grid_size": [4, 4], "num_views": 1,
                         "camera_layout": [{"name": "head", "token_width": 2}], "d_z": 4, "layer": 1, "base_sha256": "fixture"}
        self.payload = {"config": {"interface_type": "g_translator"}, "registry": self.registry,
                        "native_config": {"patch_size": [1, 1, 1]},
                        "visual_feature_space": self.metadata["feature_space_id"],
                        "encoder_identity": self.identity}
        self.goal = {"z": torch.nn.functional.normalize(torch.ones(1, 16, 4), dim=-1),
                     "goal_poses": torch.eye(4)[None, None], "goal_gripper": torch.ones(1, 1)}
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.metadata))
        np.savez_compressed(self.root / self.metadata["arrays"], **self.arrays)

    def test_goal_identity_and_coordinate_roundtrip(self):
        path = save_goal_prediction(self.root / "goal.npz", self.goal, self.identity, self.registry)
        loaded = load_goal_prediction(path, encoder_identity=self.identity, registry=self.registry)
        for name in self.goal:
            torch.testing.assert_close(loaded[name], self.goal[name], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "E identity"):
            load_goal_prediction(path, encoder_identity={**self.identity, "layer": 0}, registry=self.registry)
        with self.assertRaisesRegex(ValueError, "coordinate"):
            load_goal_prediction(path, encoder_identity=self.identity,
                                 registry={**self.registry, "coordinate_frame": "another_base"})
        with self.assertRaisesRegex(ValueError, "fresh"):
            save_goal_prediction(self.root / "goal.npz", self.goal, self.identity, self.registry)
        np.savez_compressed(self.root / "goal.npz", **{key: value.numpy() * 2 for key, value in self.goal.items()})
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_goal_prediction(path, encoder_identity=self.identity, registry=self.registry)

    def test_separate_observation_contracts(self):
        observation = load_g_pi_observation(self.path, self.payload)
        self.assertIsInstance(observation, GObservation)
        self.assertFalse(hasattr(observation, "goal"))
        self.assertFalse(hasattr(observation, "language"))
        self.assertEqual(observation.history.shape[2], 2)
        demo_file = self.root / self.metadata.pop("demonstration")["arrays"]
        demo_file.unlink()
        goal_path = save_goal_prediction(self.root / "goal.npz", self.goal, self.identity, self.registry)
        self.metadata.update(language=self.language_path, goal=goal_path.name)
        self.payload["config"]["interface_type"] = "pi_goal"
        self.save()
        observation = load_g_pi_observation(self.path, self.payload)
        self.assertIsInstance(observation, PiObservation)
        self.assertFalse(hasattr(observation, "demonstration"))
        self.metadata["demonstration"] = {"arrays": "missing.npz"}
        self.save()
        with self.assertRaisesRegex(ValueError, "route-specific"):
            load_g_pi_observation(self.path, self.payload)

    def test_pi_observation_without_language_does_not_open_a_language_file(self):
        self.metadata.pop("demonstration")
        self.metadata["goal"] = save_goal_prediction(self.root / "goal.npz", self.goal,
                                                    self.identity, self.registry).name
        self.payload["config"]["interface_type"] = "pi_goal"
        (self.root / self.language_path).unlink()
        self.save()
        observation = load_g_pi_observation(self.path, self.payload)
        self.assertIsInstance(observation, PiObservation)
        self.assertIsNone(observation.language)
        self.assertIsNone(observation.language_identity)
        self.metadata["language"] = self.language_path
        self.save()
        with self.assertRaises((FileNotFoundError, ValueError)):
            load_g_pi_observation(self.path, self.payload)

    def test_rejects_future_previous_task_and_supervision(self):
        original = copy.deepcopy(self.metadata)
        for key in ("goal_poses", "actions", "terminal_time", "subgoal_time", "subgoal_source",
                    "subgoal_annotation", "object_states", "relation", "stage"):
            self.metadata = {**original, key: "not-an-input"}
            self.save()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "without supervision"):
                load_g_pi_observation(self.path, self.payload)
        self.metadata = original
        for times, reason in (([.1, .4], "causal grid"), ([0., .6], "current_time"),
                              ([0., float("nan")], "current_time"), ([0., .2], "causal grid")):
            self.arrays["latent_available_times"] = np.array(times)
            self.save()
            with self.subTest(times=times), self.assertRaisesRegex(ValueError, reason):
                load_g_pi_observation(self.path, self.payload)

    def test_event_between_latents_uses_latest_available_history(self):
        observation = load_g_pi_observation(self.path, self.payload)
        self.assertEqual(observation.metadata["current_time"], .5)
        torch.testing.assert_close(observation.history_times, torch.tensor([0., .4], dtype=torch.float64))
        self.metadata["current_time"] = .35
        self.save()
        with self.assertRaisesRegex(ValueError, "control grid"):
            load_g_pi_observation(self.path, self.payload)
        self.metadata["current_time"] = .8
        self.save()
        with self.assertRaisesRegex(ValueError, "complete causal grid"):
            load_g_pi_observation(self.path, self.payload)

    def test_observation_canvas_matches_policy_camera_layout(self):
        self.payload["encoder_identity"] = {**self.identity,
            "camera_layout": [{"name": "head", "token_width": 2}, {"name": "wrist", "token_width": 1}]}
        with self.assertRaisesRegex(ValueError, "camera_layout"):
            load_g_pi_observation(self.path, self.payload)
        self.payload["encoder_identity"]["camera_layout"] = [{"name": "head", "token_width": 1},
                                                             {"name": "wrist", "token_width": 1}]
        self.assertEqual(load_g_pi_observation(self.path, self.payload).history.shape[-1], 2)

    def test_old_observation_version_is_rejected_explicitly(self):
        for version in (1, 2):
            self.metadata["format_version"] = version
            self.save()
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, f"version {version}.*version 3"):
                load_g_pi_observation(self.path, self.payload)

    def test_time_tolerance_never_reaches_another_control_step(self):
        self.metadata.update(control_dt=1e-7, current_time=5e-7)
        self.registry["control_dt"] = 1e-7
        self.arrays["latent_available_times"] = np.array([0., 4e-7], dtype=np.float64)
        self.save()
        self.assertEqual(load_g_pi_observation(self.path, self.payload).history.shape[2], 2)
        self.metadata["current_time"] = 4.5e-7
        self.save()
        with self.assertRaisesRegex(ValueError, "control grid"):
            load_g_pi_observation(self.path, self.payload)
        self.metadata["current_time"] = 3e-7
        self.save()
        with self.assertRaisesRegex(ValueError, "current_time"):
            load_g_pi_observation(self.path, self.payload)


if __name__ == "__main__":
    unittest.main()
