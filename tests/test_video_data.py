import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from evo_wam.video_data import load_video_index, load_video_window, validate_video_sources


def write_window(root, name="window", *, domain="human", tracked=False, effects=()):
    metadata = {
        "format_version": 1, "kind": "video_pretrain", "arrays": f"{name}.npz",
        "sample_id": name, "source_id": f"original-{name}", "source_group": f"group-{name}",
        "domain": domain, "feature_space_id": "fixed-wan-posterior-v1",
        "feature_kind": "tracked_entities" if tracked else "patches", "context_frames": 2,
        "provenance": {"fixture": "synthetic-numeric-data/no-robot-action-labels"},
    }
    arrays = {
        "features": np.arange(30, dtype=np.float32).reshape(5, 2, 3),
        "feature_valid": np.ones((5, 2, 3), dtype=np.bool_),
        "frame_times": np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32),
    }
    if tracked:
        arrays["entity_ids"] = np.array([11, 22], dtype=np.int64)
    if effects:
        metadata.update(effect_schema_id="observed-object-effects-v1", geometry_frame="object-relative-v1",
                        geometry_units="metres", evidence_source="audited-synthetic-fixture")
    for field in effects:
        shape = (3, 2, 3) if field == "geometry" else (3, 2, 2, 2)
        arrays[field] = np.zeros(shape, dtype=np.float32)
        arrays[f"{field}_valid"] = np.ones(shape, dtype=np.bool_)
    path = root / f"{name}.json"
    save_window(path, metadata, arrays)
    return path, metadata, arrays


def save_window(path, metadata, arrays):
    path.write_text(json.dumps(metadata))
    np.savez_compressed(path.parent / metadata["arrays"], **arrays)


def source(name, split="train", *, domain="human", group=None, **extra):
    return {"record_kind": "video", "source_id": name, "source_group": group or f"group-{name}",
            "domain": domain, "split": split, **extra}


class VideoWindowTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_human_patch_window_needs_no_robot_pair_action_or_effect_label(self):
        path, metadata, _ = write_window(self.root)
        window = load_video_window(path)
        self.assertEqual(window.features.shape, (1, 5, 2, 3))
        self.assertEqual(window.feature_valid.shape, window.features.shape)
        self.assertEqual(window.frame_times.shape, (5,))
        self.assertEqual(window.context_frames, 2)
        self.assertEqual(window.metadata, metadata)
        self.assertEqual(window.effect_targets, {})
        self.assertEqual(window.effect_valid, {})
        self.assertIsNone(window.entity_ids)
        self.assertTrue(window.has_training_signal)

    def test_robot_replay_uses_same_unpaired_schema_without_actions(self):
        path, _, _ = write_window(self.root, domain="robot")
        self.assertEqual(load_video_window(path).metadata["domain"], "robot")

    def test_partial_effect_subset_preserves_data_masks_and_sanitizes_invalid_values(self):
        path, metadata, arrays = write_window(self.root, tracked=True, effects=("geometry", "events"))
        arrays["geometry"][0, 0, 0] = np.nan
        arrays["geometry_valid"][0, 0, 0] = False
        arrays["events"][1, 0, 1, 0] = np.inf
        arrays["events_valid"][1, 0, 1, 0] = False
        arrays["features"][0, 0, 0] = np.nan
        arrays["feature_valid"][0, 0, 0] = False
        save_window(path, metadata, arrays)
        window = load_video_window(path)
        self.assertEqual(set(window.effect_targets), {"geometry", "events"})
        self.assertEqual(window.effect_targets["geometry"].shape, (1, 3, 2, 3))
        self.assertEqual(window.effect_targets["events"].shape, (1, 3, 2, 2, 2))
        self.assertEqual(window.entity_ids.tolist(), [[11, 22]])
        self.assertEqual(window.features[0, 0, 0, 0].item(), 0)
        self.assertFalse(window.feature_valid[0, 0, 0, 0])
        for field in window.effect_targets:
            self.assertTrue(torch.isfinite(window.effect_targets[field]).all())
            torch.testing.assert_close(window.effect_valid[field], torch.from_numpy(arrays[f"{field}_valid"])[None])

    def test_all_invisible_or_targetless_windows_are_loadable_but_skip_updates(self):
        path, metadata, arrays = write_window(self.root)
        arrays["features"][:] = np.nan
        arrays["feature_valid"][:] = False
        save_window(path, metadata, arrays)
        window = load_video_window(path)
        self.assertFalse(window.has_training_signal)
        self.assertFalse(window.features.any())
        arrays["features"][:2] = 1
        arrays["feature_valid"][:2] = True
        save_window(path, metadata, arrays)
        self.assertFalse(load_video_window(path).has_training_signal)
        arrays["features"][:] = 1
        arrays["feature_valid"][:] = True
        arrays["feature_valid"][:2] = False
        save_window(path, metadata, arrays)
        self.assertFalse(load_video_window(path).has_training_signal)

    def test_valid_future_effect_can_supervise_an_occluded_future_feature(self):
        path, metadata, arrays = write_window(self.root, tracked=True, effects=("relations",))
        arrays["feature_valid"][2:] = False
        arrays["features"][2:] = np.nan
        save_window(path, metadata, arrays)
        self.assertTrue(load_video_window(path).has_training_signal)
        arrays["relations_valid"][:] = False
        save_window(path, metadata, arrays)
        self.assertFalse(load_video_window(path).has_training_signal)

    def test_context_size_and_time_units_are_explicit(self):
        path, metadata, arrays = write_window(self.root)
        for context in (0, 4, True, 2.5):
            changed = dict(metadata, context_frames=context)
            save_window(path, changed, arrays)
            with self.subTest(context=context), self.assertRaises(ValueError):
                load_video_window(path)
        for times in (np.array([0, 1, 2, 3, 4]), np.array([0.0, 0.1, 0.1, 0.3, 0.4]),
                      np.array([0.0, 0.1, np.nan, 0.3, 0.4])):
            save_window(path, metadata, {**arrays, "frame_times": times})
            with self.subTest(times=times), self.assertRaises(ValueError):
                load_video_window(path)

    def test_bad_feature_dtype_mask_and_valid_nan_fail(self):
        path, metadata, arrays = write_window(self.root)
        mutations = [
            {"features": arrays["features"].astype(np.int64)},
            {"feature_valid": arrays["feature_valid"].astype(np.float32)},
            {"feature_valid": np.ones((5, 2), dtype=np.bool_)},
            {"features": np.full((5, 2, 3), np.nan, dtype=np.float32)},
        ]
        for changed in mutations:
            save_window(path, metadata, {**arrays, **changed})
            with self.subTest(keys=changed.keys()), self.assertRaises(ValueError):
                load_video_window(path)

    def test_effect_values_and_validity_must_appear_together_and_match_future(self):
        path, metadata, arrays = write_window(self.root, tracked=True, effects=("events",))
        for changed in (
            {name: value for name, value in arrays.items() if name != "events_valid"},
            {**arrays, "events": np.zeros((5, 2, 2, 2), dtype=np.float32)},
            {**arrays, "events": np.full((3, 2, 2, 2), 0.5, dtype=np.float32)},
        ):
            save_window(path, metadata, changed)
            with self.assertRaises(ValueError):
                load_video_window(path)

    def test_effect_identity_and_stable_entities_are_required(self):
        path, metadata, arrays = write_window(self.root, tracked=True, effects=("geometry",))
        for key in ("effect_schema_id", "geometry_frame", "geometry_units", "evidence_source"):
            save_window(path, {name: value for name, value in metadata.items() if name != key}, arrays)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                load_video_window(path)
        for ids in (np.array([11, 11]), np.array([11, -1]), np.array([11.0, 22.0])):
            save_window(path, metadata, {**arrays, "entity_ids": ids})
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                load_video_window(path)
        save_window(path, {**metadata, "feature_kind": "patches"}, {name: value for name, value in arrays.items() if name != "entity_ids"})
        with self.assertRaises(ValueError):
            load_video_window(path)

    def test_actions_forced_pairs_unknown_arrays_and_path_escape_are_rejected(self):
        path, metadata, arrays = write_window(self.root)
        save_window(path, metadata, {**arrays, "actions": np.zeros((3, 7), dtype=np.float32)})
        with self.assertRaises(ValueError):
            load_video_window(path)
        for changed in ({"pair_kind": "same-task"}, {"trajectory_id": "arbitrary-robot"}, {"arrays": "../outside.npz"}):
            path.write_text(json.dumps({**metadata, **changed}))
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                load_video_window(path)


class VideoSourcesTest(unittest.TestCase):
    def test_source_reposts_adjacent_windows_and_transitive_links_cannot_cross_splits(self):
        cases = [
            [source("same", "train", group="a"), source("same", "test", group="b")],
            [source("original", "train", group="same"), source("repost", "validation", group="same")],
            [source("a", "train", group="one"), source("b", "train", group="one"),
             source("b", "test", group="two")],
        ]
        for records in cases:
            with self.subTest(records=records), self.assertRaises(ValueError):
                validate_video_sources(records)

    def test_downstream_bridge_checks_human_sources_and_robot_trajectories(self):
        bridge = {"record_kind": "bridge", "source_id": "human", "trajectory_id": "robot", "split": "test"}
        for record in (source("human"), source("robot", domain="robot"),
                       source("camera-export", domain="robot", trajectory_id="robot")):
            with self.subTest(record=record), self.assertRaises(ValueError):
                validate_video_sources([record, bridge])
        validate_video_sources([source("human", "test"), bridge])
        validate_video_sources([source("independent"), bridge])

    def test_shared_robot_trajectory_links_bridge_source_components(self):
        records = [source("video-a"),
                   {"record_kind": "bridge", "source_id": "video-a", "trajectory_id": "robot", "split": "train"},
                   {"record_kind": "bridge", "source_id": "video-b", "trajectory_id": "robot", "split": "validation"}]
        with self.assertRaises(ValueError):
            validate_video_sources(records)

    def test_index_accepts_robot_replay_and_exports_all_source_records_without_features(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            human, _, _ = write_window(root, "human")
            robot, _, _ = write_window(root, "robot", domain="robot")
            index = root / "index.json"
            document = {
                "format_version": 1, "kind": "video_pretrain_index",
                "samples": [{"manifest": human.name, "split": "train"}, {"manifest": robot.name, "split": "validation"}],
                "bridge_sources": [{"source_id": "separate-human", "trajectory_id": "separate-robot", "split": "test"}],
            }
            index.write_text(json.dumps(document))
            selected, records = load_video_index(index)
            self.assertEqual(selected, [human])
            self.assertEqual([record["split"] for record in records], ["train", "validation", "test"])
            self.assertEqual(records[-1]["record_kind"], "bridge")
            self.assertTrue(all("features" not in record and "arrays" not in record for record in records))
            validate_video_sources(records)

    def test_index_rejects_mixed_feature_space_duplicates_and_hidden_heldout_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, _, _ = write_window(root, "first")
            second, metadata, arrays = write_window(root, "second")
            index = root / "index.json"
            document = {"format_version": 1, "kind": "video_pretrain_index", "samples": [
                {"manifest": first.name, "split": "train"}, {"manifest": second.name, "split": "test"}]}
            index.write_text(json.dumps(document))
            for changed in ({"feature_space_id": "another-encoder"}, {"sample_id": "first"},
                            {"source_group": "group-first"}):
                save_window(second, {**metadata, **changed}, arrays)
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    load_video_index(index, "train")
            save_window(second, metadata, arrays)
            document["bridge_sources"] = [{"source_id": "original-first", "trajectory_id": "heldout", "split": "test"}]
            index.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                load_video_index(index, "train")

    def test_index_rejects_unknown_split_and_external_manifest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "index.json"
            index.write_text(json.dumps({"format_version": 1, "kind": "video_pretrain_index",
                                        "samples": [{"manifest": "../outside.json", "split": "train"}]}))
            with self.assertRaises(ValueError):
                load_video_index(index)
            with self.assertRaises(ValueError):
                load_video_index(index, "anything")


if __name__ == "__main__":
    unittest.main()
