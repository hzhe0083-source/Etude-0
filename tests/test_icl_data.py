import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from evo_wam.icl_data import LATENT_NORMALIZATION, load_icl_index, load_icl_sample


def write_sample(root, name="pair", *, robot=False):
    metadata = {
        "format_version": 1, "kind": "native_icl_sample", "sample_id": name,
        "feature_space_id": "fixture-wan-v1", "latent_normalization": LATENT_NORMALIZATION,
        "history_frames": 2, "compatibility": {"kind": "audited_semantic_task", "evidence": "fixture-review"},
        "demonstration": {"arrays": f"{name}-A.npz", "source_id": f"{name}-A", "source_group": f"{name}-A",
                          "domain": "human"},
        "target": {"arrays": f"{name}-B.npz", "source_id": f"{name}-B", "source_group": f"{name}-B",
                   "domain": "robot" if robot else "human"},
    }
    demo = {"latent": np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2),
            "frame_times": np.array([0., 0.2, 0.4], dtype=np.float64)}
    target = {"latent": np.arange(32, dtype=np.float32).reshape(2, 4, 2, 2),
              "frame_times": np.array([0., 0.3, 0.6, 0.9], dtype=np.float64)}
    if robot:
        metadata["target"]["action_space"] = {"representation": "zero-wam-normalized", "normalization_id": "fixture",
                                                "dimension": 2, "valid_channels": [True, False]}
        target.update(actions=np.ones((2, 4, 3, 1), dtype=np.float32),
                      actions_mask=np.ones((2, 4, 3, 1), dtype=np.bool_))
    path = root / f"{name}.json"
    save_sample(path, metadata, demo, target)
    return path, metadata, demo, target


def save_sample(path, metadata, demo, target):
    path.write_text(json.dumps(metadata))
    np.savez_compressed(path.parent / metadata["demonstration"]["arrays"], **demo)
    np.savez_compressed(path.parent / metadata["target"]["arrays"], **target)


class NativeICLDataTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_human_pair_preserves_full_videos_without_robot_labels(self):
        path, metadata, demo, target = write_sample(self.root)
        sample = load_icl_sample(path)
        self.assertEqual(sample.metadata, metadata)
        self.assertEqual(sample.demonstration.shape, (1, 2, 3, 2, 2))
        self.assertEqual(sample.target.shape, (1, 2, 4, 2, 2))
        torch.testing.assert_close(sample.demonstration[0], torch.from_numpy(demo["latent"]))
        torch.testing.assert_close(sample.target[0], torch.from_numpy(target["latent"]))
        torch.testing.assert_close(sample.target_times, torch.from_numpy(target["frame_times"]))
        self.assertEqual(sample.history_frames, 2)
        self.assertIsNone(sample.actions)
        self.assertIsNone(sample.actions_mask)
        self.assertIsNone(sample.appearance_demonstration)

    def test_appearance_pair_preserves_same_demo_lineage_without_new_source_edges(self):
        path, metadata, demo, _ = write_sample(self.root)
        metadata["appearance_variant"] = {"arrays": "appearance.npz", "derived_from": "pair-A",
                                          "evidence": "audited appearance-only transform of A"}
        path.write_text(json.dumps(metadata))
        variant = {**demo, "latent": demo["latent"] + 0.25}
        np.savez_compressed(self.root / "appearance.npz", **variant)
        sample = load_icl_sample(path)
        torch.testing.assert_close(sample.appearance_demonstration[0], torch.from_numpy(variant["latent"]))
        self.assertEqual(sample.appearance_demonstration.shape, sample.demonstration.shape)
        torch.testing.assert_close(sample.demonstration[0], torch.from_numpy(demo["latent"]))
        self.assertIsNone(sample.actions)
        index = self.write_index([{"manifest": path.name, "split": "train"}])
        with patch("numpy.load", side_effect=AssertionError("index must not read appearance arrays")):
            selected, records = load_icl_index(index)
        self.assertEqual(selected, [path])
        self.assertEqual([record["source_id"] for record in records], ["pair-A", "pair-B"])

    def test_appearance_metadata_requires_evidence_same_lineage_and_separate_local_path(self):
        path, metadata, _, _ = write_sample(self.root)
        variant = {"arrays": "appearance.npz", "derived_from": "pair-A", "evidence": "audited transform"}
        invalid = [None, {key: value for key, value in variant.items() if key != "evidence"},
                   {**variant, "evidence": " "}, {**variant, "derived_from": "pair-B"},
                   {**variant, "arrays": "pair-A.npz"}, {**variant, "arrays": "pair-B.npz"},
                   {**variant, "arrays": "../appearance.npz"},
                   {**variant, "arrays": str(self.root / "appearance.npz")},
                   {**variant, "arrays": "appearance.json"}]
        invalid.extend({**variant, name: "invented"} for name in ("actions", "geometry", "flow"))
        index = self.write_index([{"manifest": path.name, "split": "train"}])
        for changed in invalid:
            path.write_text(json.dumps({**metadata, "appearance_variant": changed}))
            with self.subTest(variant=changed), patch("numpy.load", side_effect=AssertionError("metadata only")):
                with self.assertRaises(ValueError):
                    load_icl_index(index)

    def test_appearance_payload_requires_exact_shape_times_and_no_extra_labels(self):
        path, metadata, demo, _ = write_sample(self.root)
        metadata["appearance_variant"] = {"arrays": "appearance.npz", "derived_from": "pair-A",
                                          "evidence": "audited transform"}
        path.write_text(json.dumps(metadata))
        invalid = [{**demo, "latent": demo["latent"][:, :, :, :1]},
                   {**demo, "latent": np.full_like(demo["latent"], np.nan)},
                   {**demo, "frame_times": demo["frame_times"] + 1e-12},
                   {**demo, "frame_times": np.array([0., 0.2, np.nan])},
                   {"latent": demo["latent"]}]
        invalid.extend({**demo, name: np.zeros(1)} for name in ("actions", "geometry", "flow"))
        for number, payload in enumerate(invalid):
            np.savez_compressed(self.root / "appearance.npz", **payload)
            with self.subTest(number=number), self.assertRaises(ValueError):
                load_icl_sample(path)

    def test_human_rejects_invented_actions_proprio_and_effects_in_either_video(self):
        path, metadata, demo, target = write_sample(self.root)
        for role in ("demonstration", "target"):
            for name in ("actions", "actions_mask", "proprio", "geometry", "relations", "events"):
                a, b = copy.deepcopy(demo), copy.deepcopy(target)
                (a if role == "demonstration" else b)[name] = np.zeros((1,), dtype=np.float32)
                save_sample(path, metadata, a, b)
                with self.subTest(role=role, name=name), self.assertRaisesRegex(ValueError, "NPZ"):
                    load_icl_sample(path)
        for name in ("trajectory_id", "action_space", "proprio"):
            changed = copy.deepcopy(metadata)
            changed["target"][name] = "invented"
            save_sample(path, changed, demo, target)
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_icl_sample(path)

    def test_semantic_pair_requires_review_and_independent_sources(self):
        path, metadata, demo, target = write_sample(self.root)
        changes = [{"compatibility": {"kind": "synchronized_views", "evidence": "review"}},
                   {"compatibility": {"kind": "audited_semantic_task", "evidence": " "}},
                   {"compatibility": {"kind": "audited_semantic_task"}}]
        for key in ("source_id", "source_group", "arrays"):
            changes.append({"target": {**metadata["target"], key: metadata["demonstration"][key]}})
        changes.append({"demonstration": {**metadata["demonstration"], "domain": "robot"}})
        for change in changes:
            path.write_text(json.dumps({**metadata, **change}))
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_icl_sample(path)

    def test_history_normalization_latent_and_time_boundaries(self):
        path, metadata, demo, target = write_sample(self.root)
        for frames in (0, -1, True, 1.5, 4, 5):
            save_sample(path, {**metadata, "history_frames": frames}, demo, target)
            with self.subTest(frames=frames), self.assertRaisesRegex(ValueError, "history_frames"):
                load_icl_sample(path)
        save_sample(path, {**metadata, "latent_normalization": "unscaled"}, demo, target)
        with self.assertRaisesRegex(ValueError, "latent_normalization"):
            load_icl_sample(path)
        for name, value in (("latent", np.full((2, 4, 2, 2), np.nan, dtype=np.float32)),
                            ("latent", np.zeros((3, 4, 2, 2), dtype=np.float32)),
                            ("latent", np.zeros((2, 4, 2, 2), dtype=np.int64)),
                            ("frame_times", np.array([0., 0.1, 0.1, 0.2])),
                            ("frame_times", np.array([0., np.nan, 0.2, 0.3])),
                            ("frame_times", np.array([0., 0.1]))):
            save_sample(path, metadata, demo, {**target, name: value})
            with self.subTest(name=name, shape=value.shape), self.assertRaises(ValueError):
                load_icl_sample(path)

    def test_robot_masks_invalid_and_inactive_channels_before_arithmetic(self):
        path, metadata, demo, target = write_sample(self.root, robot=True)
        target["actions"][1] = np.nan
        target["actions"][0, 0, 0, 0] = np.nan
        target["actions_mask"][0, 0, 0, 0] = False
        save_sample(path, metadata, demo, target)
        sample = load_icl_sample(path)
        self.assertEqual(sample.actions.shape, (1, 2, 4, 3, 1))
        self.assertTrue(torch.isfinite(sample.actions).all())
        self.assertFalse(sample.actions_mask[:, 1].any())
        self.assertEqual(sample.actions[0, 0, 0, 0, 0], 0)
        self.assertTrue(sample.actions_mask[:, 0, 2:].all())
        target["actions_mask"][0, 0, 0, 0] = True
        save_sample(path, metadata, demo, target)
        with self.assertRaisesRegex(ValueError, "finite"):
            load_icl_sample(path)
        target["actions_mask"][0, 0, 0, 0] = False
        target["actions_mask"][:, 2:] = False
        save_sample(path, metadata, demo, target)
        with self.assertRaisesRegex(ValueError, "future action supervision"):
            load_icl_sample(path)

    def write_index(self, samples, **extra):
        path = self.root / "index.json"
        path.write_text(json.dumps({"format_version": 1, "kind": "native_icl_index", "samples": samples, **extra}))
        return path

    def test_index_is_metadata_only_and_audits_both_video_roles(self):
        train, _, _, _ = write_sample(self.root, "train")
        test, metadata, _, _ = write_sample(self.root, "test")
        path = self.write_index([{"manifest": train.name, "split": "train"}, {"manifest": test.name, "split": "test"}])
        with patch("numpy.load", side_effect=AssertionError("index must not read arrays")):
            selected, records = load_icl_index(path)
        self.assertEqual(selected, [train])
        self.assertEqual(len(records), 4)
        for role, training_id in (("demonstration", "train-B"), ("target", "train-A")):
            changed = copy.deepcopy(metadata)
            changed[role]["source_id"] = training_id
            test.write_text(json.dumps(changed))
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "crosses"):
                load_icl_index(path)

    def test_index_retains_alias_and_robot_bridge_leakage(self):
        train, _, _, _ = write_sample(self.root, "train")
        test, _, _, _ = write_sample(self.root, "test")
        aliases = [{"source_id": "alias-A", "source_group": "train-A", "domain": "human", "split": "train"}]
        bridges = [{"source_id": "alias-A", "trajectory_id": "robot-bridge", "split": "train"},
                   {"source_id": "test-B", "trajectory_id": "robot-bridge", "split": "test"}]
        path = self.write_index([{"manifest": train.name, "split": "train"}, {"manifest": test.name, "split": "test"}],
                                source_aliases=aliases, bridge_sources=bridges)
        with self.assertRaisesRegex(ValueError, "crosses"):
            load_icl_index(path)
        path = self.write_index([{"manifest": train.name, "split": "train"}], source_aliases=aliases,
                                bridge_sources=bridges[:1])
        _, records = load_icl_index(path)
        self.assertEqual(len(records), 4)
        self.assertEqual(records[-1]["trajectory_id"], "robot-bridge")

    def test_index_rejects_human_target_reposts_through_transitive_source_links(self):
        pair, _, _, _ = write_sample(self.root)
        aliases = [{"source_id": "repost", "source_group": "pair-A", "domain": "human", "split": "train"},
                   {"source_id": "repost", "source_group": "pair-B", "domain": "human", "split": "train"}]
        path = self.write_index([{"manifest": pair.name, "split": "train"}], source_aliases=aliases)
        with patch("numpy.load", side_effect=AssertionError("index must not read arrays")):
            with self.assertRaisesRegex(ValueError, "independent source components"):
                load_icl_index(path)

    def test_shared_robot_supervision_does_not_imply_same_human_recording(self):
        pair, _, _, _ = write_sample(self.root)
        bridges = [{"source_id": "pair-A", "trajectory_id": "shared", "split": "train"},
                   {"source_id": "pair-B", "trajectory_id": "shared", "split": "train"}]
        aliases = [{"source_id": "robot-A", "source_group": "pair-A", "domain": "robot",
                    "trajectory_id": "shared", "split": "train"},
                   {"source_id": "robot-B", "source_group": "pair-B", "domain": "robot",
                    "trajectory_id": "shared", "split": "train"}]
        for extras in ({"bridge_sources": bridges}, {"source_aliases": aliases}):
            path = self.write_index([{"manifest": pair.name, "split": "train"}], **extras)
            with self.subTest(extras=extras), patch("numpy.load", side_effect=AssertionError("index must not read arrays")):
                selected, _ = load_icl_index(path)
            self.assertEqual(selected, [pair])

    def test_reversed_pair_does_not_join_independent_sources(self):
        first, metadata, _, _ = write_sample(self.root)
        second = self.root / "reverse.json"
        second.write_text(json.dumps({**metadata, "sample_id": "reverse", "history_frames": 1,
                                     "demonstration": metadata["target"], "target": metadata["demonstration"]}))
        path = self.write_index([{"manifest": first.name, "split": "train"}, {"manifest": second.name, "split": "train"}])
        with patch("numpy.load", side_effect=AssertionError("index must not read arrays")):
            selected, _ = load_icl_index(path)
        self.assertEqual(selected, [first, second])

    def test_index_rejects_duplicate_ids_feature_mismatch_and_path_escape(self):
        train, _, _, _ = write_sample(self.root, "train")
        test, metadata, _, _ = write_sample(self.root, "test")
        path = self.write_index([{"manifest": train.name, "split": "train"}, {"manifest": test.name, "split": "test"}])
        for change in ({"sample_id": "train"}, {"feature_space_id": "other-wan"}):
            test.write_text(json.dumps({**metadata, **change}))
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_icl_index(path)
        path = self.write_index([{"manifest": "../outside.json", "split": "train"}])
        with self.assertRaisesRegex(ValueError, "within"):
            load_icl_index(path)


if __name__ == "__main__":
    unittest.main()
