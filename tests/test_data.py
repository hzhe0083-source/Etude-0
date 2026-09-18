import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from evo_wam.data import (
    TaskCondition, assign_splits, connected_components, executed_prefix_valid,
    load_experiment, load_sample, paired_dropout_disabled, prepare_training_input,
    shared_denoising_inputs, validate_splits,
)


def write_fixture(directory: Path):
    """Small numeric fixture: two entities, a partially executed eight-step plan."""
    meta = {
        "format_version": 1, "arrays": "arrays.npz", "source_id": "human-1",
        "trajectory_id": "robot-1", "history_id": "robot-1-at-12",
        "coordinate_frame": "robot_base", "task_annotation": "audited-required-events-v1",
        "window_start": 12, "executed_steps": 2, "control_dt": 0.05,
        "view_ids": ["front", "side"],
    }
    arrays = {
        "entity_ids": np.array([11, 22], dtype=np.int64),
        "step_offsets": np.array([1, 2, 4, 8], dtype=np.int64),
        "robot_history": np.zeros((3, 2, 4), dtype=np.float32),
        "proprio_history": np.zeros((3, 3), dtype=np.float32),
        "embodiment": np.zeros(2, dtype=np.float32),
        "entity_patch_weights": np.ones((2, 8), dtype=np.float32) / 8,
        "robot_latent": np.zeros((4, 2, 2, 2), dtype=np.float32),
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
    for view in (0, 1):
        for field in ("current_binding", "remaining_binding"):
            arrays[f"view{view}_{field}_valid"] = np.ones(2, dtype=np.bool_)
        for field in ("relations", "events"):
            arrays[f"view{view}_{field}_valid"] = np.ones_like(arrays[f"outcome_{field}"], dtype=np.bool_)
    arrays["view1_current_binding_valid"][0] = False
    (directory / "sample.json").write_text(json.dumps(meta), encoding="utf-8")
    np.savez(directory / "arrays.npz", **arrays)
    return meta, arrays


class DataTests(unittest.TestCase):
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
            unconditional_probability=0,
        )
        self.assertEqual(calls, {"history": 1, "target": 2})
        first, second = paired.branches
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
            self.assertFalse(sample.common_valid["current_binding"][0, 0])
            self.assertTrue(sample.requirement.remaining.label_valid["geometry"].all())
            self.assertEqual(sample.native_inputs, {})
            arrays["outcome_relations"] = np.zeros((4, 2, 3), dtype=np.float32)
            np.savez(root / "arrays.npz", **arrays)
            with self.assertRaises(ValueError):
                load_sample(root / "sample.json")

    def test_loader_rejects_pickle_paths_and_task_bypass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta, arrays = write_fixture(root)
            arrays["demo_view_0"] = np.array([{"execute": "never"}], dtype=object)
            np.savez(root / "arrays.npz", **arrays)
            with self.assertRaises(ValueError):
                load_sample(root / "sample.json")
            meta["arrays"] = "../arrays.npz"
            (root / "sample.json").write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                load_sample(root / "sample.json")
            meta["arrays"] = "arrays.npz"
            meta["native_arrays"] = {"text_emb": {"target": "native_answer"}}
            (root / "sample.json").write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                load_sample(root / "sample.json")

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


if __name__ == "__main__":
    unittest.main()
