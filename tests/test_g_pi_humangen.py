from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from evo_wam.g_pi_data import EventRules, load_g_pi_index, load_g_pi_sample, subgoal_control_indices
from evo_wam.g_pi_humangen import (audit_humangen, convert_humangen, load_humangen_robot_latents,
                                 read_humangen_episode, read_humangen_latent)
from evo_wam.g_pi_subgoals import (CANDIDATE_THRESHOLDS, RELATION_REGISTRY, VERSIONS,
                                 generate_candidates, resolve_subgoal_indices)
from evo_wam.icl_data import LATENT_NORMALIZATION
from evo_wam.robotwin import RobotwinActionTransform, USED_CHANNELS
from evo_wam.vision import sha256
from test_goal_language import write_goal_language


def write_humangen_fixture(root, *, signal="measured"):
    count = 17
    state = np.zeros((count, 16), dtype=np.float32)
    state[:, 6] = state[:, 14] = 1.
    state[:, 0] = np.arange(count) * .01
    state[:, 8] = np.arange(count) * -.02
    state[5:, 7] = .08
    state[:, 15] = .08
    action = state.copy()
    action[:, 0] += .015
    action[:, 8] -= .025
    times = np.arange(count, dtype=np.float64) * .1
    raw = {"observation.state": state, "action": action, "timestamp": times}
    np.savez_compressed(root / "raw.npz", **raw)
    stats = {"method": "abs", "norm_stats": {
        "action.hand.position": {"q01": [-1.] * 14, "q99": [1.] * 14},
        "action.effector.position": {"q01": [0., 0.], "q99": [.08, .08]}}}
    (root / "stats.json").write_text(json.dumps(stats))
    rules = asdict(EventRules(signal_source=signal))
    grip = torch.from_numpy(state[:, [7, 15]] / .08)
    indices = subgoal_control_indices(grip, torch.from_numpy(times), EventRules(**rules))
    layout = [{"name": name, "token_width": 1} for name in ("head", "left_wrist", "right_wrist")]
    visual = {"latent": np.arange(2 * 5 * 2 * 3, dtype=np.float32).reshape(2, 5, 2, 3),
              "latent_available_times": times[::4], "subgoal_times": times[indices.numpy()],
              "subgoal_latents": np.stack([np.full((2, 1, 2, 3), 100 + i, np.float32) for i in indices.tolist()])}
    np.savez_compressed(root / "visual.npz", **visual)
    visual_meta = {"format_version": 1, "kind": "humangen_robotwin_visuals", "arrays": "visual.npz",
                   "arrays_sha256": sha256(root / "visual.npz"), "raw_sha256": sha256(root / "raw.npz"),
                   "feature_space_id": "fixture-wan", "latent_normalization": LATENT_NORMALIZATION,
                   "frame_stride": 1, "camera_layout": layout, "patch_size": [1, 1],
                   "subgoal_encoding": "wan_vae_single_frame", "subgoal_control_indices": indices.tolist(),
                   "vae_identity": {"fixture-vae": "a" * 64}, "evidence": "constructed isolated-frame fixture"}
    (root / "visual.json").write_text(json.dumps(visual_meta))
    np.savez_compressed(root / "human.npz", latent=np.ones((2, 3, 2, 3), np.float32), frame_times=np.array([0., .5, 1.]))
    write_goal_language(root, "language")
    spec = {"format_version": 1, "kind": "humangen_robotwin_conversion",
            "repository": {"id": "fixture/HumanGen", "revision": "fixture-v1", "evidence": "constructed fixture, not downloaded HumanGen"},
            "fields": {"state": "observation.state", "action": "action", "timestamp": "timestamp"},
            "pose_convention": {"quaternion_order": "xyzw", "translation_units": "m", "source_frame": "world",
                                "coordinate_frame": "robot_base", "base_from_source": np.eye(4).tolist(),
                                "tool_frames": ["left_tcp", "right_tcp"], "pose_source": "measured_endpoint", "evidence": "fixture calibration"},
            "gripper_signal": {"source": signal, "field": "observation.state", "indices": [7, 15],
                               "closed": [0., 0.], "open": [.08, .08], "units": "m", "normalization_id": "fixture-grip", "evidence": f"fixture {signal} source audit"},
            "normalization": {"path": "stats.json", "sha256": sha256(root / "stats.json")},
            "robot": {"control_dt": .1, "frame_stride": 1, "action_frames": 2, "state_space_id": "fixture-world-xyzw",
                      "feature_space_id": "fixture-wan", "camera_layout": layout, "patch_size": [1, 1]},
            "event_rules": rules,
            "episodes": [{"sample_id": "pick-0", "split": "train", "robot_source_id": "robot-0", "trajectory_id": "trajectory-0",
                          "raw": "raw.npz", "success": True, "success_evidence": "fixture explicitly successful", "visual_cache": "visual.json",
                          "language": "language.json", "human": {"source_id": "human-0", "source_group": "person-session-0",
                          "video_id": "run0/human.mp4", "video_sha256": "b" * 64, "latent": "human.npz",
                          "pairing_evidence": "fixture task-level matching only", "origin_evidence": "fixture known generation parent",
                          "origin_robot_trajectory_id": "human-parent-trajectory"}}]}
    path = root / "conversion.json"
    path.write_text(json.dumps(spec))
    return path, spec, raw, visual_meta


class HumanGenConversionTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path, self.spec, self.raw, self.visual_meta = write_humangen_fixture(self.root)
        self.output = self.root / "converted"

    def convert(self):
        self.path.write_text(json.dumps(self.spec))
        return convert_humangen(self.path, self.output)

    def refresh_raw(self):
        np.savez_compressed(self.root / "raw.npz", **self.raw)
        self.visual_meta["raw_sha256"] = sha256(self.root / "raw.npz")
        (self.root / "visual.json").write_text(json.dumps(self.visual_meta))

    def test_fixture_converts_and_matches_native_action_mapping_and_dual_grid(self):
        report = self.convert()
        paths, _ = load_g_pi_index(report["index"])
        pi = load_g_pi_sample(paths[0], current_time=.4)
        g = load_g_pi_sample(paths[0], current_time=.4, route="g_translator")
        self.assertEqual(report["episodes"], 1)
        self.assertFalse(report["model_loaded"])
        self.assertEqual(report["missing_pi_language"], [])
        self.assertEqual(pi.history.shape, (1, 2, 2, 2, 3))
        self.assertEqual(pi.subgoal_time, .6000000000000001)
        self.assertTrue(torch.equal(pi.target_frame, torch.full_like(pi.target_frame, 106.)))
        self.assertEqual(pi.actions_mask.sum().item(), len(USED_CHANNELS) * 2)
        self.assertTrue(torch.equal(pi.actions_mask[0, :, 0, :2, 0].any(-1), torch.from_numpy(np.isin(np.arange(30), USED_CHANNELS))))
        torch.testing.assert_close(pi.goal_poses[0, :, 0, 3], torch.tensor([.06, -.12]))
        self.assertIsNone(g.language)
        self.assertIsNotNone(pi.language)
        self.assertTrue(g.metadata["subgoal_annotation"]["weak_label"])
        # Read the pinned upstream helper, without loading any model or changing the submodule.
        module_path = Path(__file__).resolve().parents[1] / "third_party/Zero-WAM/wan_va/dataset/robotwin_action.py"
        module_spec = importlib.util.spec_from_file_location("humangen_native_action_fixture", module_path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        transform = RobotwinActionTransform.from_stats(self.root / "stats.json", expected_sha256=sha256(self.root / "stats.json"))
        inverse = [16] * 30
        for i, j in enumerate(USED_CHANNELS):
            inverse[j] = i
        expected, mask = module.preprocess_robotwin_actions(self.raw["action"], self.raw["observation.state"],
            transform.q01, transform.q99, inverse, history_size=0, required_size=17)
        with np.load(paths[0].parent / "robot.npz") as values:
            np.testing.assert_allclose(values["actions"].T, expected, atol=1e-6)
            np.testing.assert_array_equal(values["actions_mask"].T, mask)

    def test_command_gripper_is_honest_weak_event_source(self):
        self.spec["gripper_signal"].update(source="command", evidence="fixture command register, not measured width")
        self.spec["event_rules"]["signal_source"] = "command"
        report = self.convert()
        sample = load_g_pi_sample(Path(report["index"]).parent / "episode_000000/task.json", route="g_translator")
        self.assertEqual(sample.metadata["gripper_signal_source"], "command")
        self.assertEqual(sample.metadata["subgoal_annotation"]["evidence"], "gripper:command")
        self.assertTrue(sample.metadata["subgoal_annotation"]["weak_label"])

    def test_missing_language_still_converts_for_g_but_is_reported_for_pi(self):
        self.spec["episodes"][0].pop("language")
        report = self.convert()
        self.assertEqual(report["missing_pi_language"], ["pick-0"])
        task = self.output / "episode_000000/task.json"
        self.assertIsNone(load_g_pi_sample(task, route="g_translator").language)
        with self.assertRaisesRegex(ValueError, "language cache"):
            load_g_pi_sample(task)

    def test_quaternion_order_units_and_base_calibration_are_applied(self):
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_euler("z", 90, degrees=True).as_quat()
        for value in (self.raw["observation.state"], self.raw["action"]):
            for start in (0, 8):
                value[:, start:start + 3] *= 1000
                value[:, start + 3:start + 7] = rotation[[3, 0, 1, 2]]
        base = np.eye(4)
        base[1, 3] = .5
        self.spec["pose_convention"].update(quaternion_order="wxyz", translation_units="mm", base_from_source=base.tolist())
        self.refresh_raw()
        report = self.convert()
        sample = load_g_pi_sample(self.output / "episode_000000/task.json", current_time=.4)
        np.testing.assert_allclose(sample.goal_poses[0, 0, :3, :3], Rotation.from_euler("z", 90, degrees=True).as_matrix(), atol=1e-6)
        np.testing.assert_allclose(sample.goal_poses[0, 0, :3, 3], [.06, .5, 0.], atol=1e-6)

    def test_invalid_success_and_unverified_conventions_are_rejected(self):
        for field, value, message in (("success", False, "verified success"), ("success_evidence", "", "success_evidence")):
            original = self.spec["episodes"][0][field]
            self.spec["episodes"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                self.convert()
            self.spec["episodes"][0][field] = original
            self.assertFalse(self.output.exists())
        self.spec["pose_convention"]["pose_source"] = "controller_target"
        with self.assertRaisesRegex(ValueError, "measured_endpoint"):
            self.convert()

    def test_timestamp_gaps_do_not_silently_resample(self):
        self.raw["timestamp"][7] += .01
        self.refresh_raw()
        with self.assertRaisesRegex(ValueError, "every control row"):
            self.convert()
        self.assertFalse(self.output.exists())

    def test_visual_identity_and_independent_goal_indices_are_checked(self):
        for key, bad, message in (("raw_sha256", "c" * 64, "raw episode identity"),
                                  ("arrays_sha256", "c" * 64, "arrays identity"),
                                  ("subgoal_control_indices", [8, 16], "subgoal_control_indices"),
                                  ("subgoal_encoding", "covering_video_latent", "single-frame"),
                                  ("camera_layout", [{"name": "merged", "token_width": 3}], "camera_layout")):
            (self.root / "visual.json").write_text(json.dumps({**self.visual_meta, key: bad}))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                self.convert()
            self.assertFalse(self.output.exists())

    def test_subgoal_times_must_match_exact_control_boundary(self):
        with np.load(self.root / "visual.npz") as data:
            arrays = {name: data[name].copy() for name in data.files}
        arrays["subgoal_times"][0] = .8
        np.savez_compressed(self.root / "visual.npz", **arrays)
        self.visual_meta["arrays_sha256"] = sha256(self.root / "visual.npz")
        (self.root / "visual.json").write_text(json.dumps(self.visual_meta))
        with self.assertRaisesRegex(ValueError, "exactly match"):
            self.convert()

    def test_candidate_match_rejects_command_gripper_and_object_sources(self):
        entry = self.spec["episodes"][0]
        entry["subgoal_source"] = "sim_relation"
        with self.assertRaisesRegex(ValueError, "only weak"):
            self.convert()
        entry["subgoal_source"] = "candidate_match"
        self.spec["event_rules"]["signal_source"] = "command"
        self.spec["gripper_signal"]["source"] = "command"
        with self.assertRaisesRegex(ValueError, "measured gripper widths"):
            self.convert()

    def test_candidate_match_measured_fixture_regenerates_boundaries(self):
        state = self.raw["observation.state"]
        state[:, 7] = .08
        state[4:8, 7] = .024
        self.refresh_raw()
        poses = np.tile(np.eye(4, dtype=np.float32), (17, 2, 1, 1))
        poses[:, 0, 0, 3] = state[:, 0]
        poses[:, 1, 0, 3] = state[:, 8]
        arrays = {"control_times": torch.from_numpy(self.raw["timestamp"]), "poses": torch.from_numpy(poses),
                  "gripper": torch.from_numpy(state[:, [7, 15]] / .08)}
        annotation = {"detector_version": VERSIONS["candidate_match"], "thresholds": CANDIDATE_THRESHOLDS,
                      "stable_steps": 2, "evidence": "fixture measured widths", "weak_label": True,
                      "relation_registry": RELATION_REGISTRY, "template_version": "fixture-v1",
                      "relations": [{"predicate": "in_hand", "object_role": "cup", "effector": 0, "occurrence": 1},
                                    {"predicate": "placed", "object_role": "cup", "effector": 0, "occurrence": 1, "relation": "in", "target_role": "basket"}],
                      "candidates": generate_candidates(arrays, EventRules())}
        metadata = {"subgoal_source": "candidate_match", "event_rules": self.spec["event_rules"], "subgoal_annotation": annotation}
        indices, audit = resolve_subgoal_indices(metadata, arrays, check_indices=False)
        self.spec["episodes"][0].update(subgoal_source="candidate_match", subgoal_annotation=audit)
        with np.load(self.root / "visual.npz") as data:
            visual = {name: data[name].copy() for name in data.files}
        visual["subgoal_times"] = self.raw["timestamp"][indices.numpy()]
        visual["subgoal_latents"] = np.stack([np.full((2, 1, 2, 3), 100 + i, np.float32) for i in indices.tolist()])
        np.savez_compressed(self.root / "visual.npz", **visual)
        self.visual_meta.update(arrays_sha256=sha256(self.root / "visual.npz"), subgoal_control_indices=indices.tolist())
        (self.root / "visual.json").write_text(json.dumps(self.visual_meta))
        self.convert()
        sample = load_g_pi_sample(self.output / "episode_000000/task.json", current_time=.4)
        self.assertTrue(sample.metadata["subgoal_annotation"]["weak_label"])
        # Closing at 4, two stable measured widths at 5 and 6: confirmation at 6.
        self.assertAlmostEqual(sample.subgoal_time, .6)

    def second_episode(self):
        second = deepcopy(self.spec["episodes"][0])
        second.update(sample_id="pick-1", split="test", robot_source_id="robot-1", trajectory_id="trajectory-1")
        # Distinct robot content avoids the independent robot-identity guard obscuring the human guard.
        raw = {key: value.copy() for key, value in self.raw.items()}
        raw["observation.state"][:, 1] += .01
        np.savez_compressed(self.root / "raw2.npz", **raw)
        meta = {**self.visual_meta, "raw_sha256": sha256(self.root / "raw2.npz")}
        (self.root / "visual2.json").write_text(json.dumps(meta))
        second.update(raw="raw2.npz", visual_cache="visual2.json")
        self.spec["episodes"].append(second)
        return second

    def test_reused_human_source_cannot_cross_splits(self):
        self.second_episode()
        with self.assertRaisesRegex(ValueError, "crosses.*splits"):
            self.convert()
        self.assertFalse(self.output.exists())

    def test_repacked_human_content_and_generation_parent_close_split_aliases(self):
        second = self.second_episode()
        second["human"].update(source_id="alias", source_group="alias", video_id="alias.mp4", video_sha256="d" * 64,
                               origin_robot_trajectory_id="other")
        with self.assertRaisesRegex(ValueError, "crosses.*splits"):
            self.convert()
        with np.load(self.root / "human.npz") as data:
            np.savez_compressed(self.root / "different-human.npz", latent=data["latent"] + 1, frame_times=data["frame_times"])
        second["human"].update(latent="different-human.npz", origin_robot_trajectory_id="trajectory-0")
        with self.assertRaisesRegex(ValueError, "crosses.*splits"):
            self.convert()

    def test_unknown_gripper_source_missing_calibration_and_normalization_rejected(self):
        self.spec["gripper_signal"]["source"] = "unknown"
        with self.assertRaisesRegex(ValueError, "measured/command"):
            self.convert()
        self.spec["gripper_signal"]["source"] = "measured"
        self.spec["normalization"]["sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "normalization artifact"):
            self.convert()
        self.spec["normalization"]["sha256"] = sha256(self.root / "stats.json")
        self.spec["pose_convention"].pop("evidence")
        with self.assertRaisesRegex(ValueError, "pose_convention"):
            self.convert()

    def test_no_overwrite_and_no_automatic_downloads(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "new directory"):
            self.convert()
        self.output.rmdir()
        self.spec["episodes"][0]["raw"] = "https://example.org/episode.parquet"
        with self.assertRaisesRegex(ValueError, "local inputs only"):
            self.convert()

    def test_missing_optional_parquet_dependency_has_actionable_error(self):
        from unittest.mock import patch
        with patch.dict("sys.modules", {"pyarrow": None, "pyarrow.parquet": None}):
            with self.assertRaisesRegex(ValueError, "already-installed pyarrow.*NPZ"):
                read_humangen_episode(self.root / "absent.parquet", self.spec["fields"])

    def write_published_latent(self, name="published", *, bias=0.):
        latent = torch.arange(5 * 2 * 3 * 2, dtype=torch.float32).reshape(5 * 2 * 3, 2) + bias
        payload = {"latent": latent, "latent_num_frames": 5, "latent_height": 2, "latent_width": 3,
                   "frame_ids": list(range(17)), "ori_fps": 10., "text": "must not enter G",
                   "text_emb": torch.full((512, 4096), float("nan"))}
        path = self.root / f"{name}.pth"
        torch.save(payload, path)
        return path, payload

    def test_published_packed_latents_keep_layout_and_discard_text(self):
        path, payload = self.write_published_latent()
        arrays, audit = read_humangen_latent(path)
        self.assertEqual(set(arrays), {"latent", "frame_times"})
        np.testing.assert_array_equal(arrays["latent"], payload["latent"].reshape(5, 2, 3, 2).permute(3, 0, 1, 2))
        np.testing.assert_array_equal(arrays["frame_times"], np.array([0., .4, .8, 1.2, 1.6]))
        self.assertEqual(audit["text_fields_discarded"], ["text", "text_emb"])
        self.spec["episodes"][0]["human"]["latent"] = path.name
        self.spec["episodes"][0]["human"].pop("video_sha256")
        self.convert()
        sample = load_g_pi_sample(self.output / "episode_000000/task.json", route="g_translator")
        self.assertIsNone(sample.language)
        self.assertTrue(torch.isfinite(sample.demonstration).all())

    def test_camera_import_preserves_order_and_rejects_future_or_missing_coverage(self):
        left, _ = self.write_published_latent("left", bias=0)
        right, payload = self.write_published_latent("right", bias=1000)
        times = np.arange(17, dtype=np.float64) / 10
        arrays, audits = load_humangen_robot_latents([right, left], times, frame_stride=1)
        self.assertEqual(arrays["latent"].shape, (2, 5, 2, 6))
        np.testing.assert_array_equal(arrays["latent"][..., :3] - arrays["latent"][..., 3:], 1000)
        payload["frame_ids"][-1] = 18
        torch.save(payload, right)
        with self.assertRaisesRegex(ValueError, "causal grid"):
            load_humangen_robot_latents([left, right], times, frame_stride=1)

    def test_real_input_preflight_never_fabricates_training_labels(self):
        robot, _ = self.write_published_latent()
        human, _ = self.write_published_latent("human-published")
        spec = {"format_version": 1, "kind": "humangen_robotwin_audit", "fields": self.spec["fields"],
                "control_dt": .1, "frame_stride": 1, "episodes": [{"episode_id": 0, "raw": "raw.npz",
                "robot_latents": [robot.name], "human_latent": human.name}]}
        path = self.root / "audit-manifest.json"
        info = {"fps": 10, "features": {self.spec["fields"][key]: {"shape": [16]}
                                        for key in ("state", "action")}}
        (self.root / "info.json").write_text(json.dumps(info))
        spec["info"] = "info.json"
        path.write_text(json.dumps(spec))
        report = audit_humangen(path, self.root / "audit-result.json")
        self.assertFalse(report["training_ready"])
        self.assertFalse(report["model_loaded"])
        self.assertEqual(report["gripper_signal_source"], "unknown")
        self.assertEqual(report["info_identity"]["fps"], 10)
        self.assertEqual(report["episodes"][0]["control_rows"], 17)
        self.assertTrue(any("gripper source" in text for text in report["requires_audit"]))
        self.assertTrue(any("independently encode" in text for text in report["requires_preprocessing"]))
        self.assertFalse(self.output.exists())
        info["fps"] = 50
        (self.root / "info.json").write_text(json.dumps(info))
        with self.assertRaisesRegex(ValueError, "fps differs"):
            audit_humangen(path, self.root / "other-report.json")


if __name__ == "__main__":
    unittest.main()
