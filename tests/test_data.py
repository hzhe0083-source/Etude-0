import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from evo_wam.data import (
    TaskCondition, assign_splits, connected_components, executed_prefix_valid,
    load_experiment, load_observation, load_sample, paired_dropout_disabled, prepare_training_input,
    shared_denoising_inputs, validate_demo_encoding, validate_splits,
)


def write_fixture(directory: Path):
    """Small numeric fixture: two entities, a partially executed eight-step plan."""
    meta = {
        "format_version": 2, "kind": "training_sample", "arrays": "arrays.npz", "source_id": "human-1",
        "trajectory_id": "robot-1", "history_id": "robot-1-at-12",
        "coordinate_frame": "robot_base", "task_annotation": "audited-required-events-v1",
        "window_start": 12, "observation_step": 12, "executed_steps": 2, "control_dt": 0.05,
        "view_ids": ["front", "side"],
        "pair_kind": "synchronized_views",
        "actions_per_frame": 2,
        "history_chunks": [
            {"mode": "video", "slice": [0, 2], "frame_id": 0, "rope_offset": 0},
            {"mode": "action", "slice": [0, 3], "frame_id": 1, "rope_offset": 0},
        ],
    }
    meta["action_space"] = {"representation": "zero-wam-normalized", "normalization_id": "test-v2", "dimension": 3, "valid_channels": [True, True, True]}
    meta["observed_action_space"] = dict(meta["action_space"])
    arrays = {
        "entity_ids": np.array([11, 22], dtype=np.int64),
        "step_offsets": np.array([1, 2, 4, 8], dtype=np.int64),
        "robot_history": np.zeros((3, 2, 4), dtype=np.float32),
        "proprio_history": np.zeros((3, 3), dtype=np.float32),
        "embodiment": np.zeros(2, dtype=np.float32),
        "entity_patch_weights": np.ones((2, 8), dtype=np.float32) / 8,
        "robot_latent": np.zeros((4, 2, 2, 2), dtype=np.float32),
        "observed_action_history": np.arange(9, dtype=np.float32).reshape(3, 3) / 10,
        "observed_action_step_offsets": np.array([-2, -1, 0], dtype=np.int64),
        "observed_video_step_offsets": np.array([-2, 0], dtype=np.int64),
        "actions": np.zeros((8, 3), dtype=np.float32),
        "demo_view_0": np.zeros((3, 4), dtype=np.float32),
        "demo_view_1": np.ones((3, 4), dtype=np.float32),
    }
    for field, shape in (("geometry", (4, 2, 3)), ("relations", (4, 2, 2, 3)), ("events", (4, 2, 2, 2))):
        arrays[f"outcome_{field}"] = np.zeros(shape, dtype=np.float32)
        arrays[f"outcome_{field}_valid"] = np.ones(shape, dtype=np.bool_)
    for part, offsets in (("current", [1, 2]), ("remaining", [4, 8])):
        arrays[f"{part}_step_offsets"] = np.array(offsets, dtype=np.int64)
        arrays[f"{part}_binding"] = np.array([0, 1], dtype=np.int64)
        arrays[f"{part}_binding_valid"] = np.ones(2, dtype=np.bool_)
        for field, shape in (("geometry", (2, 2, 3)), ("relations", (2, 2, 2, 3)), ("events", (2, 2, 2, 2))):
            arrays[f"{part}_{field}"] = np.zeros(shape, dtype=np.float32)
            arrays[f"{part}_{field}_valid"] = np.ones(shape, dtype=np.bool_)
            arrays[f"{part}_{field}_required"] = np.ones(shape, dtype=np.bool_)
        arrays[f"{part}_geometry_tolerance"] = np.full((2, 2, 3), .25, dtype=np.float32)
        arrays[f"{part}_geometry_tolerance_valid"] = np.ones((2, 2, 3), dtype=np.bool_)
        windows = np.array(offsets, dtype=np.int64).reshape(2, 1, 1, 1, 1)
        arrays[f"{part}_event_windows"] = np.broadcast_to(windows, (2, 2, 2, 2, 2)).copy()
        arrays[f"{part}_event_windows_valid"] = np.ones((2, 2, 2, 2), dtype=np.bool_)
        arrays[f"{part}_events"].flat[[0, 8]] = 1
        arrays[f"{part}_event_precedence"] = np.array([[0, 8], [-1, -1]], dtype=np.int64)
        arrays[f"{part}_event_precedence_valid"] = np.ones(2, dtype=np.bool_)
    for view in (0, 1):
        for part in ("current", "remaining"):
            for field in ("binding", "geometry", "relations", "events", "geometry_tolerance", "event_windows", "event_precedence"):
                arrays[f"view{view}_{part}_{field}_valid"] = np.ones_like(arrays[f"{part}_{field}_valid"])
        for field in ("relations", "events"):
            arrays[f"view{view}_{field}_valid"] = np.ones_like(arrays[f"outcome_{field}"], dtype=np.bool_)
    arrays["view1_current_binding_valid"][0] = False
    (directory / "sample.json").write_text(json.dumps(meta), encoding="utf-8")
    np.savez(directory / "arrays.npz", **arrays)
    return meta, arrays


class DataTests(unittest.TestCase):
    def test_demo_encoding_identity_is_explicit_and_shape_checked(self):
        self.assertEqual(validate_demo_encoding({}, 4), {"kind": "raw_features"})
        encoding = {"kind": "video_effect_tokens", "encoder_sha256": "AB" * 32,
                    "feature_space_id": "frozen-visual-v1", "token_dim": 4,
                    "window_frames": 3, "num_tokens": 2}
        normalized = validate_demo_encoding({"demonstration_encoding": encoding}, 4)
        self.assertEqual(normalized["encoder_sha256"], "ab" * 32)
        self.assertEqual(encoding["encoder_sha256"], "AB" * 32)
        invalid = [None, {"kind": "raw_features", "encoder_sha256": "a" * 64},
                   {**encoding, "encoder_sha256": "g" * 64},
                   {**encoding, "feature_space_id": " "}, {**encoding, "token_dim": 5},
                   {**encoding, "window_frames": 2}, {**encoding, "num_tokens": 0},
                   {**encoding, "num_tokens": True}, {**encoding, "unregistered": 1}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_demo_encoding({"demonstration_encoding": value}, 4)

    def test_demo_encoding_is_preserved_by_both_loaders_and_widths_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta, arrays = write_fixture(root)
            self.assertEqual(load_sample(root / "sample.json").demonstration_encoding, {"kind": "raw_features"})
            meta["demonstration_encoding"] = {
                "kind": "video_effect_tokens", "encoder_sha256": "cd" * 32,
                "feature_space_id": "frozen-visual-v1", "token_dim": 4,
                "window_frames": 3, "num_tokens": 2,
            }
            (root / "sample.json").write_text(json.dumps(meta))
            self.assertEqual(load_sample(root / "sample.json").demonstration_encoding, meta["demonstration_encoding"])
            arrays["demo_view_1"] = np.ones((3, 5), dtype=np.float32)
            np.savez(root / "arrays.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "same feature width"):
                load_sample(root / "sample.json")
            arrays["demo_view_1"] = np.ones((3, 4), dtype=np.float32)
            keys = {"entity_ids", "robot_history", "proprio_history", "embodiment", "robot_latent", "demo_view_0", "demo_view_1",
                    "observed_action_history", "observed_action_step_offsets", "observed_video_step_offsets"}
            observed = {key: arrays[key] for key in keys}
            meta.update(kind="observation", chunk_size=2)
            (root / "observation.json").write_text(json.dumps(meta))
            np.savez(root / "arrays.npz", **observed)
            loaded = load_observation(root / "observation.json")
            self.assertEqual(loaded.demonstration_encoding, meta["demonstration_encoding"])
            observed["demo_view_1"] = np.ones((3, 5), dtype=np.float32)
            np.savez(root / "arrays.npz", **observed)
            with self.assertRaisesRegex(ValueError, "same feature width"):
                load_observation(root / "observation.json")

    def test_shared_noise_and_encode_once(self):
        calls = {"history": 0, "target": 0}

        def history_encoder(value):
            calls["history"] += 1
            return value + torch.randn_like(value)

        def target_encoder(value):
            calls["target"] += 1
            return value + 0.3

        targets = {"next": torch.zeros(1, 2, 3), "ifp1": torch.ones(1, 2, 3)}
        conditions = (TaskCondition(torch.zeros(1, 3, 4)), TaskCondition(torch.zeros(1, 3, 4)))
        paired = prepare_training_input(
            torch.zeros(1, 2, 4), targets, conditions,
            {"next": torch.tensor([0.25]), "ifp1": torch.tensor([0.75])},
            {"next": 1, "ifp1": 3}, history_encoder=history_encoder,
            target_encoder=target_encoder, generator=torch.Generator().manual_seed(10),
            unconditional_probability=0, pair_kind="synchronized_views",
        )
        self.assertEqual(calls, {"history": 1, "target": 2})
        first, second = paired.branches
        self.assertTrue(paired.loss_enabled["cv"])
        self.assertIs(first.robot_history, second.robot_history)
        self.assertIs(first.denoising, second.denoising)
        self.assertIs(first.denoising["next"].noise, second.denoising["next"].noise)
        self.assertFalse(torch.equal(first.denoising["next"].noise, first.denoising["ifp1"].noise))
        self.assertEqual(first.denoising["next"].tau.item(), 0.25)
        self.assertEqual(first.denoising["ifp1"].tau.item(), 0.75)
        target = first.denoising["next"]
        torch.testing.assert_close(target.noisy, 0.75 * target.clean + 0.25 * target.noise)

    def test_per_frame_noise_and_invalid_schedule(self):
        inputs = {"next": torch.zeros(2, 4, 3, 2, 2)}
        draws = shared_denoising_inputs(inputs, {"next": torch.linspace(1, 0.1, 10)}, {"next": 1}, generator=torch.Generator().manual_seed(4), time_dim=2)
        self.assertEqual(draws["next"].tau.shape, (2, 3))
        with self.assertRaises(ValueError):
            shared_denoising_inputs(inputs, {"next": torch.tensor([1.1])}, {"next": 1}, generator=torch.Generator())

    def test_bfloat16_latents_preserve_float32_scheduler_times(self):
        clean = torch.ones(2, 4, 3, 2, 2, dtype=torch.bfloat16)
        table = torch.tensor([0.1234567, 0.7654321], dtype=torch.float32)
        seed = 41
        expected_indices = torch.randint(table.numel(), (2, 3), generator=torch.Generator().manual_seed(seed))
        sample = shared_denoising_inputs(
            {"next": clean}, {"next": table}, {"next": 1},
            generator=torch.Generator().manual_seed(seed), time_dim=2,
        )["next"]
        self.assertEqual(sample.tau.dtype, torch.float32)
        self.assertTrue(torch.equal(sample.tau, table[expected_indices]))
        self.assertFalse(torch.equal(sample.tau, sample.tau.bfloat16().float()))
        self.assertEqual(sample.noisy.dtype, torch.bfloat16)
        self.assertEqual(sample.flow_target.dtype, torch.bfloat16)
        weights = sample.tau.reshape(2, 1, 3, 1, 1).bfloat16()
        torch.testing.assert_close(sample.noisy, (1 - weights) * clean + weights * sample.noise)

    def test_unconditional_has_no_teacher_or_task_inputs(self):
        conditions = tuple(TaskCondition(torch.ones(1, 2, 4), torch.ones(1, 2, 4), torch.ones(1, 2, 4), task_cache={"old": "task"}) for _ in range(2))
        kwargs = dict(history_encoder=lambda x: x, target_encoder=lambda x: x, unconditional_probability=1)
        pair = prepare_training_input(torch.zeros(1, 2, 4), {"next": torch.zeros(1, 2, 4)}, conditions, {"next": torch.tensor([0.5])}, {"next": 1}, generator=torch.Generator().manual_seed(1), **kwargs)
        self.assertFalse(pair.conditional)
        self.assertEqual(len(pair.branches), 1)
        self.assertEqual(pair.branches[0].condition, TaskCondition())
        self.assertEqual({k for k, enabled in pair.loss_enabled.items() if enabled}, {"next_video", "ifp"})
        other = prepare_training_input(torch.zeros(1, 2, 4), {"next": torch.zeros(1, 2, 4)}, (TaskCondition(), TaskCondition()), {"next": torch.tensor([0.5])}, {"next": 1}, generator=torch.Generator().manual_seed(1), **kwargs)
        torch.testing.assert_close(pair.branches[0].denoising["next"].noisy, other.branches[0].denoising["next"].noisy)

    def test_single_view_and_unaudited_pairs_do_not_enable_cv(self):
        condition = TaskCondition(torch.ones(1, 3, 4))
        def prepare(count, kind="none"):
            return prepare_training_input(
                torch.zeros(1, 2, 4), {"next": torch.zeros(1, 2, 4)},
                (condition,) * count, {"next": torch.tensor([0.5])}, {"next": 1},
                history_encoder=lambda value: value, target_encoder=lambda value: value,
                generator=torch.Generator().manual_seed(3), unconditional_probability=0,
                enable_cv=True, pair_kind=kind,
            )
        single = prepare(1)
        self.assertEqual(len(single.branches), 1)
        self.assertIs(single.branches[0].condition, condition)
        self.assertTrue(single.loss_enabled["requirement"])
        self.assertFalse(single.loss_enabled["cv"])
        self.assertFalse(prepare(2).loss_enabled["cv"])
        self.assertTrue(prepare(2, "synchronized_views").loss_enabled["cv"])
        with self.assertRaises(ValueError):
            prepare(1, "synchronized_views")
        with self.assertRaises(ValueError):
            prepare(2, "same_task")

    def test_dropout_pair_determinism_and_restore(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.9))
        model.train()
        value = torch.ones(10, 4)
        with paired_dropout_disabled(model):
            self.assertTrue(model.training)
            torch.testing.assert_close(model(value), model(value))
        self.assertTrue(model[1].training)

    def test_transitive_connected_split_is_order_invariant(self):
        records = [
            {"source_id": "h1", "trajectory_id": "r1"},
            {"source_id": "h1", "trajectory_id": "r2"},
            {"source_id": "h2", "trajectory_id": "r2"},
            {"source_id": "h3", "trajectory_id": "r3"},
        ]
        self.assertEqual(connected_components(records), [(0, 1, 2), (3,)])
        splits = assign_splits(records, seed=42)
        self.assertEqual(splits[:3], [splits[0]] * 3)
        self.assertEqual(assign_splits(records[::-1], seed=42), splits[::-1])
        assigned = [dict(row, split=split) for row, split in zip(records, splits)]
        validate_splits(assigned)
        assigned[2]["split"] = "test" if splits[0] != "test" else "train"
        with self.assertRaises(ValueError):
            validate_splits(assigned)

    def test_sparse_prefix_and_mask_type(self):
        mask = torch.ones(1, 3, 2, dtype=torch.bool)
        valid = executed_prefix_valid(mask, 3, time_dim=1, step_offsets=torch.tensor([1, 3, 8]))
        self.assertTrue(valid[:, :2].all())
        self.assertFalse(valid[:, 2:].any())
        with self.assertRaises(ValueError):
            executed_prefix_valid(mask.float(), 1)

    def test_safe_loader_shapes_provenance_and_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta, arrays = write_fixture(root)
            sample = load_sample(root / "sample.json")
            self.assertEqual(sample.outcome.entity_ids.tolist(), [[11, 22]])
            self.assertEqual(sample.requirement.current.step_offsets.tolist(), [1, 2])
            self.assertEqual(sample.requirement.remaining.step_offsets.tolist(), [4, 8])
            self.assertTrue(sample.outcome.label_valid["geometry"][:, :2].all())
            self.assertFalse(sample.outcome.label_valid["geometry"][:, 2:].any())
            self.assertTrue(sample.per_view_valid[0]["current.binding"][0, 0])
            self.assertFalse(sample.per_view_valid[1]["current.binding"][0, 0])
            torch.testing.assert_close(sample.requirement.current.geometry_tolerance, torch.full((1, 2, 2, 3), .25))
            self.assertEqual(sample.requirement.current.event_precedence.tolist(), [[[0, 8], [-1, -1]]])
            self.assertTrue(sample.requirement.remaining.label_valid["geometry"].all())
            self.assertEqual(sample.native_inputs, {})
            self.assertEqual(sample.pair_kind, "synchronized_views")
            arrays["outcome_relations"] = np.zeros((4, 2, 3), dtype=np.float32)
            np.savez(root / "arrays.npz", **arrays)
            with self.assertRaises(ValueError):
                load_sample(root / "sample.json")

    def test_single_view_schema_preserves_its_masks_without_a_duplicate_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta, arrays = write_fixture(root)
            # Legacy v2 files are readable but no longer silently assert sync.
            del meta["pair_kind"]
            (root / "sample.json").write_text(json.dumps(meta))
            old_pair = load_sample(root / "sample.json")
            self.assertEqual(old_pair.pair_kind, "none")
            self.assertEqual(len(old_pair.demonstrations), 2)
            meta["view_ids"] = ["front"]
            meta["pair_kind"] = "none"
            arrays = {key: value for key, value in arrays.items()
                      if key != "demo_view_1" and not key.startswith("view1_")}
            arrays["view0_remaining_geometry_valid"][0, 0, 0] = False
            (root / "sample.json").write_text(json.dumps(meta))
            np.savez(root / "arrays.npz", **arrays)
            single = load_sample(root / "sample.json")
            self.assertEqual(len(single.demonstrations), 1)
            self.assertEqual(len(single.per_view_valid), 1)
            self.assertFalse(single.per_view_valid[0]["remaining.geometry"][0, 0, 0, 0])
            self.assertTrue(single.requirement.remaining.label_valid["geometry"].all())
            self.assertEqual(single.pair_kind, "none")
            meta["pair_kind"] = "synchronized_views"
            (root / "sample.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "requires two"):
                load_sample(root / "sample.json")
            meta["pair_kind"] = "none"
            (root / "sample.json").write_text(json.dumps(meta))
            arrays["demo_view_1"] = arrays["demo_view_0"].copy()
            np.savez(root / "arrays.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "extra="):
                load_sample(root / "sample.json")

    def test_video_without_robot_labels_is_not_a_robot_training_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = {"format_version": 2, "kind": "video_pretrain", "arrays": "video.npz",
                    "source_id": "single-video", "view_ids": ["ego"]}
            (root / "video.json").write_text(json.dumps(meta))
            np.savez(root / "video.npz", demo_view_0=np.ones((3, 4), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "training_sample"):
                load_sample(root / "video.json")

    def test_loader_rejects_pickle_paths_and_task_bypass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta, arrays = write_fixture(root)
            arrays["demo_view_0"] = np.array([{"execute": "never"}], dtype=object)
            np.savez(root / "arrays.npz", **arrays)
            with self.assertRaises(ValueError):
                load_sample(root / "sample.json")
            meta, arrays = write_fixture(root)
            meta["arrays"] = "../arrays.npz"
            (root / "sample.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "within the manifest directory"):
                load_sample(root / "sample.json")
            meta["arrays"] = "arrays.npz"
            meta["native_arrays"] = {"text_emb": {"target": "native_answer"}}
            (root / "sample.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "never task conditions"):
                load_sample(root / "sample.json")

    def test_observed_action_prefix_has_time_format_and_padding_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta, arrays = write_fixture(root)
            # Computational positions are deliberately not inferred from length.
            meta["history_chunks"][0].update(frame_id=8, rope_offset=13)
            meta["history_chunks"][1].update(frame_id=9, rope_offset=13)
            (root / "sample.json").write_text(json.dumps(meta))
            sample = load_sample(root / "sample.json")
            history = sample.native_history(dtype=torch.bfloat16)
            self.assertEqual([chunk.mode for chunk in history], ["video", "action"])
            action = history[1]
            self.assertEqual(action.latent.shape, (1, 3, 2, 2, 1))
            self.assertEqual(action.latent.dtype, torch.bfloat16)
            self.assertEqual(action.token_valid.tolist(), [True, True, True, False])
            actual = action.latent.permute(0, 2, 3, 4, 1).reshape(1, 4, 3)
            torch.testing.assert_close(actual[:, :3].float(), sample.observed_action_history.commands,
                                       rtol=0.01, atol=0.01)
            self.assertFalse(actual[:, 3].any())
            sample.actions.fill_(999)
            torch.testing.assert_close(sample.native_history(dtype=torch.bfloat16)[1].latent, action.latent)
            self.assertEqual(sample.sampling_position(), {"frame_id": 10, "rope_offset": 15})
            self.assertEqual(sample.observed_action_history.step_offsets.tolist(), [-2, -1, 0])
            forged = replace(sample, observed_action_history=replace(
                sample.observed_action_history, step_offsets=torch.tensor([-2, -1, 1])))
            with self.assertRaisesRegex(ValueError, "never future"):
                forged.native_history()
            with self.assertRaisesRegex(ValueError, "never future"):
                forged.sampling_position()

    def test_v2_rejects_future_unsorted_or_misnormalized_history_and_missing_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta, arrays = write_fixture(root)
            for case in ("future_action", "future_video", "unsorted", "format", "source", "missing_history", "missing_semantics", "legacy"):
                changed_meta, changed_arrays = copy.deepcopy(meta), {key: value.copy() for key, value in arrays.items()}
                if case == "future_action":
                    changed_arrays["observed_action_step_offsets"][-1] = 1
                elif case == "future_video":
                    changed_arrays["observed_video_step_offsets"][-1] = 1
                elif case == "unsorted":
                    changed_arrays["observed_action_step_offsets"][:] = [0, -1, -2]
                elif case == "format":
                    changed_meta["observed_action_space"]["normalization_id"] = "other-robot"
                elif case == "source":
                    changed_meta["history_chunks"][1]["source"] = "actions"
                elif case == "missing_history":
                    del changed_arrays["observed_action_history"]
                elif case == "missing_semantics":
                    del changed_arrays["current_event_windows_valid"]
                else:
                    changed_meta["format_version"] = 1
                (root / "sample.json").write_text(json.dumps(changed_meta))
                np.savez(root / "arrays.npz", **changed_arrays)
                with self.subTest(case=case), self.assertRaises(ValueError):
                    load_sample(root / "sample.json")

    def test_observation_keeps_only_past_commands_and_accepts_explicit_empty_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta, arrays = write_fixture(root)
            keys = {"entity_ids", "robot_history", "proprio_history", "embodiment", "robot_latent", "demo_view_0", "demo_view_1",
                    "observed_action_history", "observed_action_step_offsets", "observed_video_step_offsets"}
            observed = {key: arrays[key] for key in keys}
            meta.update(kind="observation", chunk_size=2)
            path = root / "observation.json"
            path.write_text(json.dumps(meta))
            np.savez(root / "arrays.npz", **observed)
            sample = load_observation(path)
            self.assertFalse(hasattr(sample, "actions"))
            self.assertEqual(sample.observed_action_history.commands.shape, (1, 3, 3))
            self.assertEqual(len(sample.native_history()), 2)
            forged = replace(sample, observed_action_history=replace(
                sample.observed_action_history, step_offsets=torch.tensor([-2, -1, 1])))
            with self.assertRaisesRegex(ValueError, "never future"):
                forged.native_history()
            observed["observed_action_history"] = np.empty((0, 3), dtype=np.float32)
            observed["observed_action_step_offsets"] = np.empty(0, dtype=np.int64)
            meta["history_chunks"] = meta["history_chunks"][:1]
            path.write_text(json.dumps(meta))
            np.savez(root / "arrays.npz", **observed)
            self.assertEqual(len(load_observation(path).native_history()), 1)
            observed["actions"] = arrays["actions"]
            np.savez(root / "arrays.npz", **observed)
            with self.assertRaises(ValueError):
                load_observation(path)

    def test_native_numeric_streams_remain_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta, arrays = write_fixture(root)
            meta["native_arrays"] = {"latent_dict": {"latent": "native_clean"}, "mcp_latent_dicts": [{"targets": "native_future"}]}
            meta["native_scalars"] = {"chunk_size": 2, "max_frame_chunk_size": 4, "window_size": 8}
            arrays["native_clean"] = np.zeros((1, 4, 2, 2, 2), dtype=np.float32)
            arrays["native_future"] = arrays["native_clean"].copy()
            (root / "sample.json").write_text(json.dumps(meta))
            np.savez(root / "arrays.npz", **arrays)
            sample = load_sample(root / "sample.json")
            self.assertEqual(sample.native_inputs["latent_dict"]["latent"].shape, (1, 4, 2, 2, 2))
            self.assertEqual(sample.native_inputs["chunk_size"], 2)

    def test_experiment_controls(self):
        directory = Path(__file__).resolve().parents[1] / "configs"
        configs = {name: json.loads((directory / f"{name}.json").read_text()) for name in ("T0", "T1", "T2", "V0", "V1", "geometry", "full")}
        v0, v1 = copy.deepcopy(configs["V0"]), copy.deepcopy(configs["V1"])
        self.assertEqual(v0.pop("lambda_cv"), 0)
        self.assertGreater(v1.pop("lambda_cv"), 0)
        self.assertEqual(v0["training"]["loss_weights"].pop("cv"), 0)
        self.assertGreater(v1["training"]["loss_weights"].pop("cv"), 0)
        self.assertEqual(v0, v1)
        geo, full = copy.deepcopy(configs["geometry"]), copy.deepcopy(configs["full"])
        self.assertEqual(geo.pop("interface"), "geometry")
        self.assertEqual(full.pop("interface"), "full")
        self.assertEqual(geo, full)
        for config in configs.values():
            self.assertEqual(config["training"]["seeds"], [0, 1, 2])
            self.assertEqual(config["candidate_count"], 1)
            self.assertEqual(config["conditioning"]["unconditional_probability"], 0.1)
            self.assertFalse(config["f_ranking"])
        self.assertEqual(load_experiment(directory / "V0.json"), configs["V0"])
        with self.assertRaises(ValueError):
            load_experiment(directory / "V1.json", for_test=True)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "experiment.json"
            for field in ("confidence_threshold", "margin_threshold"):
                for endpoint in (0, 1):
                    invalid = copy.deepcopy(configs["V1"])
                    invalid["binding_policy"][field] = endpoint
                    path.write_text(json.dumps(invalid))
                    with self.subTest(field=field, endpoint=endpoint), self.assertRaises(ValueError):
                        load_experiment(path)


if __name__ == "__main__":
    unittest.main()
