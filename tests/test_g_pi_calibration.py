import copy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from evo_wam.g_pi_calibration import (DISTANCES, _subgoal_keys, calibrate_g_pi_thresholds, calibrate_goal_thresholds,
    calibrate_validation, load_calibration_artifact, load_calibration_policy)
from evo_wam.g_pi_context import save_target_cache
from evo_wam.g_pi_controller import goal_distances, goal_reached
from evo_wam.g_pi_data import load_g_pi_sample
from evo_wam.g_pi_deployment import save_goal_prediction
from evo_wam.goal_training import goal_registry
from test_g_pi_data import write_g_pi_task


def encoder_identity():
    return {"layer": 1, "timestep": 0, "pooling": "adaptive_avg_pool2d_spatial", "grid_size": [2, 2],
            "token_order": "camera_then_row_major", "num_views": 1,
            "camera_layout": [{"name": "head", "token_width": 2}],
            "k_z": 4, "d_z": 2, "normalization": "l2_last_dim",
            "base_id": {"kind": "fixture"}, "base_sha256": "a" * 64,
            "empty_text_identity": {"source": {"kind": "fixture"}, "sha256": "b" * 64}}


def goal(angle=0., position=0., rotation=0., gripper=.5):
    angle = torch.tensor(float(angle), dtype=torch.float64)
    z = torch.stack([angle.cos(), angle.sin()])[None, None].repeat(1, 4, 1)
    poses = torch.eye(4, dtype=torch.float64)[None, None]
    poses[0, 0, 0, 3] = position
    theta = torch.deg2rad(torch.tensor(float(rotation), dtype=torch.float64))
    poses[0, 0, :2, :2] = torch.tensor([[theta.cos(), -theta.sin()], [theta.sin(), theta.cos()]])
    return {"z": z, "goal_poses": poses, "goal_gripper": torch.tensor([[gripper]], dtype=torch.float64)}


def recordings(offset=.04):
    return [{"sample_id": "a", "intent_group": "intent-1", "source_group": "person-a", "goals": [goal(), goal(1.)]},
            {"sample_id": "b", "intent_group": "intent-1", "source_group": "person-b", "goals": [goal(offset), goal(1. + offset)]}]


class GPiCalibrationTest(unittest.TestCase):
    def calibrate(self, records=None, **kwargs):
        return calibrate_goal_thresholds(records or recordings(), encoder_identity=encoder_identity(),
                                        registry={"coordinate_frame": "robot_base"}, **kwargs)

    def test_joint_and_threshold_accepts_spread_and_separates_z_only_goals(self):
        artifact = self.calibrate()
        thresholds = load_calibration_policy(artifact, encoder_identity(), {"coordinate_frame": "robot_base"})
        evidence = artifact["evidence"]
        self.assertGreater(thresholds.z, evidence["positive_bounds"]["z"])
        self.assertLess(thresholds.z, min(row["z"] for row in evidence["adjacent_distances"]))
        self.assertEqual(thresholds.position_m, 0.)
        self.assertEqual(thresholds.rotation_deg, 0.)
        self.assertEqual(thresholds.gripper, 0.)
        for first, second in zip(recordings()[0]["goals"], recordings()[1]["goals"]):
            self.assertTrue(goal_reached(first, second, thresholds))
        self.assertFalse(goal_reached(*recordings()[0]["goals"], thresholds))

    def test_each_adjacent_pair_can_have_different_separating_coordinate(self):
        left = [goal(), goal(1.), goal(1., position=1.), goal(1., position=1., rotation=60.),
                goal(1., position=1., rotation=60., gripper=.9)]
        right = [goal(.02), goal(1.02), goal(1.02, position=1.), goal(1.02, position=1., rotation=60.),
                 goal(1.02, position=1., rotation=60., gripper=.9)]
        data = recordings()
        data[0]["goals"], data[1]["goals"] = left, right
        result = self.calibrate(data)
        for distances in result["evidence"]["adjacent_distances"]:
            self.assertTrue(any(distances[key] > result["thresholds"][key] for key in DISTANCES))
        self.assertTrue(all(result["thresholds"][key] > 0 for key in DISTANCES))

    def test_oracle_arrival_error_sets_the_lower_bound(self):
        arrival = {"reference": goal(), "achieved": goal(.2, position=.03, rotation=2., gripper=.52)}
        result = self.calibrate(arrivals=[arrival])
        errors = goal_distances(arrival["reference"], arrival["achieved"])
        for key in DISTANCES:
            self.assertGreaterEqual(result["thresholds"][key], errors[key])
        self.assertGreater(result["thresholds"]["z"], self.calibrate()["thresholds"]["z"])

    def test_overlapping_arrival_or_cross_execution_spread_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no feasible"):
            self.calibrate(arrivals=[{"reference": goal(), "achieved": goal(1.2)}])
        with self.assertRaisesRegex(ValueError, "layout heterogeneity"):
            self.calibrate(recordings(offset=1.2))

    def test_insufficient_source_groups_and_no_adjacent_bound_rejected(self):
        data = recordings()
        data[1]["source_group"] = data[0]["source_group"]
        with self.assertRaisesRegex(ValueError, "independent robot source"):
            self.calibrate(data)
        data = recordings()
        for record in data:
            record["goals"] = record["goals"][:1]
        with self.assertRaisesRegex(ValueError, "adjacent subgoals"):
            self.calibrate(data)

    def test_mismatched_robot_subgoals_and_duplicate_source_samples_rejected(self):
        data = recordings()
        data[1]["goals"] = data[1]["goals"][:1]
        with self.assertRaisesRegex(ValueError, "subgoal counts"):
            self.calibrate(data)
        data = recordings()
        data[0]["subgoal_keys"], data[1]["subgoal_keys"] = ["in_hand:o:0", "terminal"], ["opened:a:0", "terminal"]
        with self.assertRaisesRegex(ValueError, "ordered robot subgoal"):
            self.calibrate(data)
        with self.assertRaisesRegex(ValueError, "sample_id must be unique"):
            self.calibrate([recordings()[0], recordings()[0]])

    def test_bad_identity_and_tampered_threshold_evidence_rejected(self):
        artifact = self.calibrate()
        bad = {**encoder_identity(), "grid_size": [1, 4]}
        with self.assertRaisesRegex(ValueError, "E identity mismatch"):
            load_calibration_artifact(artifact, bad)
        with self.assertRaisesRegex(ValueError, "registry mismatch"):
            load_calibration_artifact(artifact, encoder_identity(), {"coordinate_frame": "world"})
        tampered = copy.deepcopy(artifact)
        tampered["thresholds"]["z"] = 2.
        with self.assertRaisesRegex(ValueError, "validation evidence"):
            load_calibration_artifact(tampered, encoder_identity())
        bad = {**encoder_identity(), "pooling": "adaptive_avg_pool1d_spatial"}
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            calibrate_goal_thresholds(recordings(), encoder_identity=bad, registry={"x": 1})

    def test_nonunit_z_and_nonfinite_margin_rejected(self):
        data = recordings()
        data[0]["goals"][0]["z"] *= 2
        with self.assertRaisesRegex(ValueError, "unit normalized"):
            self.calibrate(data)
        for margin in (0., 1., float("nan"), True):
            with self.subTest(margin=margin), self.assertRaisesRegex(ValueError, "margin_fraction"):
                self.calibrate(margin_fraction=margin)

    def test_relation_instance_at_terminal_keeps_both_identities(self):
        relations = [{"predicate": "in_hand", "object_role": "cup", "occurrence": 0},
                     {"predicate": "placed", "object_role": "cup", "occurrence": 0}]
        sample = SimpleNamespace(metadata={"subgoal_source": "sim_relation", "subgoal_annotation": {
            "relations": relations, "relation_control_indices": [4, 8]}})
        keys = _subgoal_keys(sample, torch.tensor([4, 8]), torch.arange(9).double())
        self.assertEqual(json.loads(keys[0].split(":", 2)[2]), [relations[0]])
        self.assertEqual(json.loads(keys[1].split(":", 2)[2]), [relations[1], "terminal"])

    def test_camera_layout_changes_token_count_and_invalidates_calibration(self):
        identity = encoder_identity()
        identity.update(camera_layout=[{"name": "head", "token_width": 2}, {"name": "wrist", "token_width": 2}],
                        num_views=2, k_z=8)
        data = recordings()
        for record in data:
            for target in record["goals"]:
                target["z"] = target["z"].repeat(1, 2, 1)
        artifact = calibrate_goal_thresholds(data, encoder_identity=identity, registry={"coordinate_frame": "robot_base"})
        self.assertEqual(artifact["encoder_identity"]["k_z"], 8)
        reordered = {**identity, "camera_layout": list(reversed(identity["camera_layout"]))}
        with self.assertRaisesRegex(ValueError, "E identity mismatch"):
            load_calibration_artifact(artifact, reordered)
        invalid = {**identity, "k_z": 4}
        with self.assertRaisesRegex(ValueError, "matching k_z"):
            calibrate_goal_thresholds(data, encoder_identity=invalid, registry={"x": 1})


class GPiCalibrationFileTest(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        records = []
        self.identity = encoder_identity()
        for index, name in enumerate(("robot-a", "robot-b")):
            path, metadata, arrays = write_g_pi_task(self.root, name, demonstration=False)
            arrays["poses"][..., 0, 3] += index * .01
            np.savez(self.root / metadata["arrays"], **arrays)
            z = torch.cat([goal(.6 + .02 * index)["z"], goal(1.6 + .02 * index)["z"]]).float()
            cache = self.root / f"{name}-cache.npz"
            save_target_cache(cache, z, self.identity)
            records.append({"task": path.name, "intent_group": "opaque-intent", "target_cache": cache.name})
        self.registry = goal_registry(load_g_pi_sample(path, current_time=0.))
        self.document = {"format_version": 1, "kind": "g_pi_validation", "split": "validation",
                         "encoder_identity": self.identity, "registry": self.registry, "recordings": records}
        self.manifest = self.root / "validation.json"
        self.output = self.root / "calibration.json"
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.document))

    def test_cache_only_command_reads_robot_tasks_without_native_or_text(self):
        args = SimpleNamespace(manifest=self.manifest, output=self.output)
        with patch("evo_wam.g_pi_training.load_g_pi_policy", side_effect=AssertionError("no model needed")):
            summary = calibrate_g_pi_thresholds(args)
        self.assertEqual(summary["recordings"], 2)
        artifact = load_calibration_artifact(self.output, self.identity, self.registry)
        self.assertEqual(len(artifact["validation_files"]), 11)
        self.assertTrue(all(len(value) == 64 for value in artifact["validation_files"].values()))
        self.assertEqual(asdict(load_calibration_policy(self.output, self.identity)), summary["thresholds"])
        with self.assertRaisesRegex(ValueError, "new artifact"):
            calibrate_g_pi_thresholds(args)

    def test_cached_identity_count_split_and_registry_fail_closed(self):
        first = self.root / self.document["recordings"][0]["target_cache"]
        save_target_cache(first, goal()["z"].float(), self.identity)
        with self.assertRaisesRegex(ValueError, "every subgoal"):
            calibrate_validation(self.manifest, self.output)
        self.document["split"] = "train"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "validation"):
            calibrate_validation(self.manifest, self.output)
        self.document["split"] = "validation"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "registry mismatch"):
            calibrate_validation(self.manifest, self.output, expected_registry={})

    def test_single_frame_encoder_receives_no_offline_group_or_history(self):
        calls = []
        outer = self

        class Encoder:
            native = torch.nn.Linear(1, 1)
            identity = outer.identity

            def __call__(self, frame):
                calls.append(frame.clone())
                return goal((float(frame.flatten()[0]) - 1000) / 10)["z"].float()

        for record in self.document["recordings"]:
            record.pop("target_cache")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "target_cache or"):
            calibrate_validation(self.manifest, self.output)
        result = calibrate_validation(self.manifest, self.output, encoder=Encoder())
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(tuple(value.shape) == (1, 2, 1, 2, 2) for value in calls))
        self.assertGreater(result["thresholds"]["z"], 0)

    def test_first_calibration_loads_training_artifact_without_preexisting_policy(self):
        fake_encoder = SimpleNamespace(identity=self.identity)
        payload = {"encoder_identity": self.identity, "registry": self.registry}
        args = SimpleNamespace(manifest=self.manifest, output=self.output,
                               artifact="training.pt", device="cpu", checkpoint=None)
        with patch("evo_wam.g_pi_training.load_g_pi_encoder", return_value=(fake_encoder, payload)) as loader:
            result = calibrate_g_pi_thresholds(args)
        loader.assert_called_once_with("training.pt", device="cpu", checkpoint=None)
        self.assertEqual(result["recordings"], 2)

    def test_oracle_arrival_sidecars_are_identity_checked_and_hashed(self):
        reference = save_goal_prediction(self.root / "reference.npz", goal(.6), self.identity, self.registry)
        achieved = save_goal_prediction(self.root / "achieved.npz", goal(.7), self.identity, self.registry)
        self.document["oracle_goal_arrivals"] = [{"reference": reference.name, "achieved": achieved.name}]
        self.write_manifest()
        result = calibrate_validation(self.manifest, self.output)
        self.assertEqual(len(result["evidence"]["oracle_arrival_errors"]), 1)
        self.assertEqual(len(result["validation_files"]), 15)
        self.assertGreaterEqual(result["thresholds"]["z"], goal_distances(goal(.6), goal(.7))["z"])

    def test_reused_trajectory_cannot_fake_independent_source_groups(self):
        task_a = self.root / self.document["recordings"][0]["task"]
        task_b = self.root / self.document["recordings"][1]["task"]
        first, second = json.loads(task_a.read_text()), json.loads(task_b.read_text())
        second["robot_source"]["trajectory_id"] = first["robot_source"]["trajectory_id"]
        task_b.write_text(json.dumps(second))
        with self.assertRaisesRegex(ValueError, "cannot count as independent"):
            calibrate_validation(self.manifest, self.output)

    def test_validation_does_not_require_action_supervision_at_task_start(self):
        for record in self.document["recordings"]:
            task = self.root / record["task"]
            metadata = json.loads(task.read_text())
            path = self.root / metadata["arrays"]
            with np.load(path) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            arrays["actions_mask"][:, :4] = False
            np.savez(path, **arrays)
        result = calibrate_validation(self.manifest, self.output)
        self.assertEqual(result["evidence"]["recordings"], 2)

    def test_g_calibration_needs_neither_task_language_nor_human_video(self):
        self.document["registry"]["language_identity"] = None
        for record in self.document["recordings"]:
            path = self.root / record["task"]
            metadata = json.loads(path.read_text())
            metadata.pop("language")
            path.write_text(json.dumps(metadata))
        self.write_manifest()
        with patch("evo_wam.g_pi_data.load_goal_language", side_effect=AssertionError("G calibration has no text")):
            result = calibrate_validation(self.manifest, self.output)
        self.assertIsNone(result["registry"]["language_identity"])
        self.assertEqual(len(result["validation_files"]), 7)
