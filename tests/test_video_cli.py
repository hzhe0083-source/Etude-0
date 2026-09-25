"""Real CPU unpaired-video training/export checks using synthetic feature windows.

These tests run the small effect encoder/predictor and the existing reader. They
do not run WAM, recover real geometry, measure transfer, or command a robot.
"""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from etude.cli import make_fixture, file_sha256
from etude.data import load_sample
from etude.models import EffectReader
from etude.video_cli import (build_video_models, encode_demonstrations, evaluate_video,
                               load_video_config, load_video_encoder, train_video)
from etude.video_data import load_video_window, patch_grid_coordinates, validate_video_sources


CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "video"


def write_windows(folder, *, supervised=True):
    """Independent human patches and a partly annotated robot entity window."""
    folder.mkdir(parents=True)
    rng = np.random.default_rng(13)
    entries = []
    for domain in ("robot", "human"):
        features = rng.normal(size=(3, 2, 4)).astype("float32")
        valid = np.ones_like(features, dtype="bool")
        if not supervised:
            valid[1:] = False
            features[1:] = np.nan
        arrays = {"features": features, "feature_valid": valid,
                  "frame_times": np.array([0., .5, 1.], dtype="float32")}
        meta = {"format_version": 2, "kind": "video_pretrain", "arrays": f"{domain}.npz",
                "sample_id": f"{domain}-window", "source_id": f"{domain}-independent-source",
                "source_group": f"{domain}-independent-recording", "domain": domain,
                "feature_space_id": "synthetic-frozen-visual-v1", "context_frames": 1,
                "feature_kind": "tracked_entities" if domain == "robot" else "patches"}
        if domain == "robot":
            arrays["entity_ids"] = np.array([10, 20], dtype="int64")
            for field, shape in (("geometry", (2, 2, 3)), ("relations", (2, 2, 2, 3)),
                                 ("events", (2, 2, 2, 2))):
                values = np.full(shape, np.nan, dtype="float32")
                evidence = np.zeros(shape, dtype="bool")
                if supervised:
                    values.flat[0], evidence.flat[0] = 1., True
                arrays[field], arrays[f"{field}_valid"] = values, evidence
            meta.update(effect_schema_id="synthetic-effect-v1", geometry_frame="robot-relative",
                        geometry_units="meters", evidence_source="synthetic-explicit-labels")
        else:
            meta.update(patch_grid=[1, 2], patch_coordinate_system="normalized_xy_patch_centers")
            arrays["patch_coordinates"] = patch_grid_coordinates([1, 2]).numpy()
        (folder / f"{domain}.json").write_text(json.dumps(meta))
        np.savez_compressed(folder / meta["arrays"], **arrays)
        entries.append({"manifest": f"{domain}.json", "split": "train"})
    index = folder / "index.json"
    index.write_text(json.dumps({"format_version": 1, "kind": "video_pretrain_index", "samples": entries}))
    return index


class VideoCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.index = write_windows(self.root / "data")
        self.config = CONFIGS / "U2_effect_constraints.json"

    def arguments(self, output="run", *, steps=4, resume=None, config=None, index=None):
        return SimpleNamespace(config=str(config or self.config), index=str(index or self.index),
            output=str(self.root / output), steps=steps, resume=None if resume is None else str(resume),
            seed=37, device="cpu")

    def artifact(self, output="run"):
        return torch.load(self.root / output / "video_encoder.pt", map_location="cpu", weights_only=True)

    def assert_state_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)

    def test_four_real_cpu_updates_domain_counts_and_encoder_roundtrip(self):
        human = load_video_window(self.index.parent / "human.json")
        self.assertEqual(human.effect_targets, {})
        self.assertFalse(hasattr(human, "actions"))
        self.assertFalse(hasattr(human, "requirement"))
        report = train_video(self.arguments())
        self.assertEqual(report["updates"], 4)
        self.assertEqual(report["domain_windows"], {"robot": 1, "human": 3})
        self.assertEqual(report["domain_updates"], {"robot": 1, "human": 3})
        self.assertEqual(report["feature_kind_updates"], {"tracked_entities": 1, "patches": 3})
        self.assertFalse(report["wam_updated_by_video_loss"])
        self.assertFalse(report["robot_execution_evaluated"])
        metrics = [json.loads(line) for line in (self.root / "run" / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["domain"] for row in metrics], ["robot", "human", "human", "human"])
        for row in metrics[1:]:
            self.assertEqual([row[key] for key in ("geometry", "relations", "events")], [0, 0, 0])
        encoder, payload = load_video_encoder(report["artifact"])
        self.assertEqual(payload["format_version"], 2)
        other, _ = load_video_encoder(report["artifact"])
        self.assertFalse(encoder.training)
        self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))
        with torch.no_grad():
            first = encoder.encode_demo(human.features, human.feature_valid, human.frame_times, window_frames=3,
                feature_kind="patches", patch_coordinates=human.patch_coordinates)
            second = other.encode_demo(human.features, human.feature_valid, human.frame_times, window_frames=3,
                feature_kind="patches", patch_coordinates=human.patch_coordinates)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(first.shape, (1, 2, payload["config"]["model"]["latent_dim"]))

    def test_resume_matches_continuous_parameters_optimizer_and_rng_exactly(self):
        train_video(self.arguments("continuous", steps=4))
        train_video(self.arguments("resumed", steps=2))
        train_video(self.arguments("resumed", steps=2, resume=self.root / "resumed" / "video_encoder.pt"))
        first, second = self.artifact("continuous"), self.artifact("resumed")
        for name in ("encoder", "predictor"):
            self.assert_state_equal(first[name], second[name])
        self.assertEqual(first["optimizer"]["param_groups"], second["optimizer"]["param_groups"])
        self.assertEqual(first["optimizer"]["state"].keys(), second["optimizer"]["state"].keys())
        for key in first["optimizer"]["state"]:
            self.assert_state_equal(first["optimizer"]["state"][key], second["optimizer"]["state"][key])
        torch.testing.assert_close(first["torch_rng"], second["torch_rng"], rtol=0, atol=0)
        for name in ("updates", "attempted_steps", "domain_windows", "domain_updates", "feature_kind_updates", "data_identity"):
            self.assertEqual(first[name], second[name])
        rows = [json.loads(line) for line in (self.root / "resumed" / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["step"] for row in rows], [0, 1, 2, 3])

    def test_no_supervision_skips_encoder_predictor_optimizer_and_capacity(self):
        index = write_windows(self.root / "unknown", supervised=False)
        config = load_video_config(self.config)
        config["loss_weights"]["capacity"] = 1000.
        config_path = self.root / "capacity.json"
        config_path.write_text(json.dumps(config))
        torch.manual_seed(37)
        initial_encoder, initial_predictor = build_video_models(config, "cpu")
        with patch("etude.video_cli.VideoEffectEncoder.forward", side_effect=AssertionError("no supervised target")), \
             patch("etude.video_cli.EffectFeaturePredictor.forward", side_effect=AssertionError("no supervised target")):
            report = train_video(self.arguments(index=index, config=config_path))
        self.assertEqual(report["updates"], 0)
        self.assertEqual(report["domain_updates"], {"robot": 0, "human": 0})
        self.assertEqual(report["feature_kind_updates"], {"tracked_entities": 0, "patches": 0})
        artifact = self.artifact()
        self.assert_state_equal(initial_encoder.state_dict(), artifact["encoder"])
        self.assert_state_equal(initial_predictor.state_dict(), artifact["predictor"])
        self.assertEqual(artifact["optimizer"]["state"], {})
        with self.assertRaisesRegex(ValueError, "without a successful supervised update"):
            load_video_encoder(report["artifact"])

    def test_legacy_encoder_artifact_rejects_loading_and_resume(self):
        report = train_video(self.arguments(steps=1))
        legacy = self.artifact()
        legacy["format_version"] = 1
        path = self.root / "legacy.pt"
        torch.save(legacy, path)
        with self.assertRaisesRegex(ValueError, "re-pretrain and re-export"):
            load_video_encoder(path)
        with self.assertRaisesRegex(ValueError, "re-pretrain and re-export"):
            train_video(self.arguments(steps=1, resume=path))
        self.assertEqual(self.artifact()["format_version"], 2)

    def test_encode_demonstrations_retains_labels_and_feeds_real_reader(self):
        artifact = train_video(self.arguments())["artifact"]
        bridge = self.root / "bridge"
        make_fixture(bridge)
        path = bridge / "sample.json"
        meta = json.loads(path.read_text())
        meta["demo_feature_space_id"] = "synthetic-frozen-visual-v1"
        meta["demonstration_layouts"] = [{"frames": 3, "tokens_per_frame": 2,
                                           "frame_times": [0., .5, 1.], "feature_kind": "patches",
                                           "patch_grid": [1, 2], "patch_coordinate_system": "normalized_xy_patch_centers",
                                           "token_order": "time,height,width,channel"} for _ in meta["view_ids"]]
        path.write_text(json.dumps(meta))
        before = load_sample(path)
        report = encode_demonstrations(SimpleNamespace(manifest=str(path), artifact=artifact,
            output=str(self.root / "encoded"), device="cpu"))
        encoded = load_sample(report["manifest"])
        self.assertEqual(encoded.demonstration_encoding["kind"], "video_effect_tokens")
        self.assertEqual(encoded.demonstration_encoding["encoder_version"], 2)
        self.assertEqual(encoded.demonstration_encoding["encoder_sha256"], file_sha256(artifact))
        self.assertEqual(len(before.demonstrations), len(encoded.demonstrations))
        for original, tokens in zip(before.demonstrations, encoded.demonstrations):
            self.assertEqual(original.shape, (1, 6, 4))
            self.assertEqual(tokens.shape, (1, 2, 4))
        torch.testing.assert_close(encoded.requirement.current.geometry, before.requirement.current.geometry)
        torch.testing.assert_close(encoded.actions, before.actions)
        torch.testing.assert_close(encoded.outcome.relations, before.outcome.relations)
        reader = EffectReader(demo_dim=4, entity_dim=4, proprio_dim=3, embodiment_dim=2, roles=2, token_dim=8)
        goals = reader(encoded.demonstrations[0], encoded.robot_history, encoded.proprio_history, encoded.embodiment,
            encoded.requirement.current.step_offsets, encoded.requirement.remaining.step_offsets,
            entity_present=encoded.requirement.current.entity_ids >= 0)
        self.assertEqual(goals.current.shape, (1, 8, 8))
        self.assertEqual(goals.remaining.shape, (1, 8, 8))
        self.assertNotEqual(goals.current.shape, encoded.demonstrations[0].shape)
        self.assertEqual(report["commands_sent"], 0)
        with self.assertRaisesRegex(ValueError, "already encoded"):
            encode_demonstrations(SimpleNamespace(manifest=report["manifest"], artifact=artifact,
                output=str(self.root / "double-encoded"), device="cpu"))

    def test_export_requires_audited_patch_layout_or_stable_entity_tracks(self):
        artifact = train_video(self.arguments(steps=2))["artifact"]
        bridge = self.root / "bridge"
        make_fixture(bridge)
        path = bridge / "sample.json"
        meta = json.loads(path.read_text())
        meta["demo_feature_space_id"] = "synthetic-frozen-visual-v1"
        layout = {"frames": 3, "tokens_per_frame": 2, "frame_times": [0., .5, 1.],
                  "feature_kind": "patches", "patch_grid": [1, 2],
                  "patch_coordinate_system": "normalized_xy_patch_centers", "token_order": "time,height,width,channel"}
        args = SimpleNamespace(manifest=str(path), artifact=artifact, output=str(self.root / "encoded"), device="cpu")
        invalid = [{key: value for key, value in layout.items() if key != missing}
                   for missing in ("feature_kind", "patch_grid", "patch_coordinate_system", "token_order")]
        invalid += [{**layout, "patch_grid": [1, 3]}, {**layout, "token_order": "unknown"},
                    {"frames": 3, "tokens_per_frame": 2, "frame_times": [0., .5, 1.],
                     "feature_kind": "tracked_entities", "entity_ids": [10, 10], "token_order": "time,entity,channel"}]
        for wrong in invalid:
            meta["demonstration_layouts"] = [wrong for _ in meta["view_ids"]]
            path.write_text(json.dumps(meta))
            with self.subTest(layout=wrong), self.assertRaises(ValueError):
                encode_demonstrations(args)
            self.assertFalse(Path(args.output).exists())
        tracked = {"frames": 3, "tokens_per_frame": 2, "frame_times": [0., .5, 1.],
                   "feature_kind": "tracked_entities", "entity_ids": [20, 10], "token_order": "time,entity,channel"}
        meta["demonstration_layouts"] = [tracked for _ in meta["view_ids"]]
        path.write_text(json.dumps(meta))
        report = encode_demonstrations(args)
        self.assertEqual(load_sample(report["manifest"]).demonstrations[0].shape, (1, 2, 4))
        self.assertEqual(json.loads(Path(report["manifest"]).read_text())["raw_demonstration_layouts"],
                         meta["demonstration_layouts"])

    def test_bridge_source_cross_split_is_rejected_before_training(self):
        document = json.loads(self.index.read_text())
        document["bridge_sources"] = [{"source_id": "human-independent-source",
            "trajectory_id": "heldout-robot", "split": "test"}]
        self.index.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "crosses"):
            train_video(self.arguments())
        self.assertFalse((self.root / "run" / "video_encoder.pt").exists())

    def test_heldout_diagnostics_are_finite_and_reject_a_training_source(self):
        artifact = train_video(self.arguments())["artifact"]
        index = write_windows(self.root / "heldout")
        document = json.loads(index.read_text())
        for entry in document["samples"]:
            entry["split"] = "test"
            path = index.parent / entry["manifest"]
            meta = json.loads(path.read_text())
            for key in ("sample_id", "source_id", "source_group"):
                meta[key] = "heldout-" + meta[key]
            path.write_text(json.dumps(meta))
            array_path = index.parent / meta["arrays"]
            with np.load(array_path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            arrays["features"] += .7
            np.savez_compressed(array_path, **arrays)
        index.write_text(json.dumps(document))
        args = SimpleNamespace(artifact=artifact, index=str(index), split="test", max_samples=2,
                               output=str(self.root / "diagnostics.json"), device="cpu")
        report = evaluate_video(args)
        self.assertEqual(report["windows"], 2)
        self.assertEqual(report["valid_feature_values"], 32)
        self.assertEqual(set(report["mse"]), {"correct_z", "zero_z", "shuffled_z"})
        self.assertTrue(all(np.isfinite(value) and value >= 0 for value in report["mse"].values()))
        self.assertFalse(report["robot_execution_evaluated"])
        self.assertFalse(report["view_invariance_established"])
        # Small random-feature diagnostics must not assert correct-z wins.
        path = index.parent / "robot.json"
        meta = json.loads(path.read_text())
        meta["source_id"] = "robot-independent-source"
        path.write_text(json.dumps(meta))
        args.output = str(self.root / "leaked-diagnostics.json")
        with self.assertRaisesRegex(ValueError, "crosses"):
            evaluate_video(args)
        self.assertFalse(Path(args.output).exists())

    def test_changed_consumed_npz_rejects_resume_without_overwriting_artifact(self):
        report = train_video(self.arguments(steps=1))
        saved = self.artifact()
        self.assertEqual(set(saved["visited_arrays"]), {"robot.json"})
        self.assertEqual(saved["geometry_basis"], ["robot-relative", "meters"])
        artifact_hash = file_sha256(report["artifact"])
        path = self.index.parent / "robot.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays["features"][0, 0, 0] += .125
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, "previously consumed.*changed"):
            train_video(self.arguments(steps=1, resume=report["artifact"]))
        self.assertEqual(file_sha256(report["artifact"]), artifact_hash)

    def test_geometry_unit_conflicts_reject_but_relation_only_labels_do_not(self):
        human_path, robot_path = self.index.parent / "human.json", self.index.parent / "robot.json"
        meta, robot = json.loads(human_path.read_text()), json.loads(robot_path.read_text())
        meta["feature_kind"] = "tracked_entities"
        meta.pop("patch_grid")
        meta.pop("patch_coordinate_system")
        for key in ("effect_schema_id", "geometry_frame", "evidence_source"):
            meta[key] = robot[key]
        meta["geometry_units"] = "pixels"
        human_path.write_text(json.dumps(meta))
        path = self.index.parent / "human.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays.pop("patch_coordinates")
        arrays.update(entity_ids=np.array([30, 40], dtype="int64"),
                      geometry=np.zeros((2, 2, 3), dtype="float32"),
                      geometry_valid=np.ones((2, 2, 3), dtype="bool"))
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, "frame convention and unit system"):
            train_video(self.arguments(steps=2))
        self.assertFalse((self.root / "run" / "video_encoder.pt").exists())
        arrays.pop("geometry")
        arrays.pop("geometry_valid")
        arrays.update(relations=np.zeros((2, 2, 2, 3), dtype="float32"),
                      relations_valid=np.ones((2, 2, 2, 3), dtype="bool"))
        np.savez_compressed(path, **arrays)
        report = train_video(self.arguments("relation-only", steps=2))
        self.assertEqual(report["updates"], 2)
        self.assertEqual(self.artifact("relation-only")["geometry_basis"], ["robot-relative", "meters"])

    def test_u0_does_not_claim_unused_human_sources_were_training_inputs(self):
        report = train_video(self.arguments(config=CONFIGS / "U0_robot_only.json", steps=1))
        artifact = self.artifact()
        self.assertEqual({record["domain"] for record in artifact["training_source_records"]}, {"robot"})
        self.assertEqual(set(artifact["visited_arrays"]), {"robot.json"})
        # The unchanged index contains human metadata, but U0 never consumed it.
        # Two test windows from that human source are therefore not train leakage.
        meta = json.loads((self.index.parent / "human.json").read_text())
        second = {**meta, "sample_id": "second-human-window"}
        (self.index.parent / "second-human.json").write_text(json.dumps(second))
        index = self.index.parent / "human-test.json"
        index.write_text(json.dumps({"format_version": 1, "kind": "video_pretrain_index", "samples": [
            {"manifest": "human.json", "split": "test"},
            {"manifest": "second-human.json", "split": "test"}]}))
        # Leakage accounting accepts these sources, but the tracked-only run
        # has not trained the separate patch-position branch.
        with self.assertRaisesRegex(ValueError, "no successful patches updates"):
            evaluate_video(SimpleNamespace(artifact=report["artifact"], index=str(index), split="test",
                max_samples=2, output=str(self.root / "u0-human-diagnostics.json"), device="cpu"))

    def test_human_only_pretraining_needs_no_robot_window_or_action_labels(self):
        document = json.loads(self.index.read_text())
        document["samples"] = [entry for entry in document["samples"] if entry["manifest"] == "human.json"]
        self.index.write_text(json.dumps(document))
        config = load_video_config(self.config)
        config["domain_schedule"] = ["human"]
        path = self.root / "human-only.json"
        path.write_text(json.dumps(config))
        report = train_video(self.arguments(steps=2, config=path))
        self.assertEqual(report["updates"], 2)
        self.assertEqual(report["domain_windows"], {"human": 2})
        self.assertEqual(report["domain_updates"], {"human": 2})
        self.assertEqual(report["feature_kind_updates"], {"tracked_entities": 0, "patches": 2})
        artifact = self.artifact()
        self.assertEqual({record["domain"] for record in artifact["training_source_records"]}, {"human"})
        self.assertEqual(set(artifact["visited_arrays"]), {"human.json"})
        self.assertIsNone(artifact["geometry_basis"])
        self.assertIsNone(artifact["effect_schema_id"])
        sample = load_video_window(self.index.parent / "human.json")
        self.assertEqual(sample.effect_targets, {})
        self.assertFalse(hasattr(sample, "actions"))
        bridge = self.root / "bridge"
        make_fixture(bridge)
        manifest = bridge / "sample.json"
        meta = json.loads(manifest.read_text())
        meta["demo_feature_space_id"] = "synthetic-frozen-visual-v1"
        meta["demonstration_layouts"] = [{"frames": 3, "tokens_per_frame": 2, "frame_times": [0., .5, 1.],
            "feature_kind": "tracked_entities", "entity_ids": [10, 20], "token_order": "time,entity,channel"}
            for _ in meta["view_ids"]]
        manifest.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "no successful tracked_entities updates"):
            encode_demonstrations(SimpleNamespace(manifest=str(manifest), artifact=report["artifact"],
                output=str(self.root / "untrained-tracked-export"), device="cpu"))
        heldout = write_windows(self.root / "heldout")
        index = json.loads(heldout.read_text())
        for entry in index["samples"]:
            entry["split"] = "test"
            target = heldout.parent / entry["manifest"]
            meta = json.loads(target.read_text())
            for field in ("sample_id", "source_id", "source_group"):
                meta[field] = "heldout-" + meta[field]
            target.write_text(json.dumps(meta))
        heldout.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "no successful tracked_entities updates"):
            evaluate_video(SimpleNamespace(artifact=report["artifact"], index=str(heldout), split="test",
                max_samples=2, output=str(self.root / "untrained-tracked-diagnostics.json"), device="cpu"))

    def test_known_bridge_edges_survive_artifact_and_block_indirect_test_leakage(self):
        document = json.loads(self.index.read_text())
        document["bridge_sources"] = [{"source_id": "human-independent-source",
                                      "trajectory_id": "linked-robot", "split": "train"}]
        self.index.write_text(json.dumps(document))
        report = train_video(self.arguments())
        ledger = self.artifact()["training_source_records"]
        self.assertTrue(any(record["record_kind"] == "bridge" and record["trajectory_id"] == "linked-robot"
                            for record in ledger))
        # This is the same downstream source-validation helper used by the
        # robot-training CLI; neither a WAM nor a mocked GPU is needed here.
        with self.assertRaisesRegex(ValueError, "crosses"):
            validate_video_sources([*ledger, {"record_kind": "bridge", "source_id": "new-task-human",
                                              "trajectory_id": "linked-robot", "split": "test"}])
        index = write_windows(self.root / "indirect-test")
        heldout = json.loads(index.read_text())
        for entry in heldout["samples"]:
            entry["split"] = "test"
            path = index.parent / entry["manifest"]
            meta = json.loads(path.read_text())
            for key in ("sample_id", "source_id", "source_group"):
                meta[key] = "new-test-" + meta[key]
            if meta["domain"] == "robot":
                meta["trajectory_id"] = "linked-robot"
            path.write_text(json.dumps(meta))
        index.write_text(json.dumps(heldout))
        output = self.root / "indirect-leak.json"
        with self.assertRaisesRegex(ValueError, "crosses"):
            evaluate_video(SimpleNamespace(artifact=report["artifact"], index=str(index), split="test",
                                          max_samples=2, output=str(output), device="cpu"))
        self.assertFalse(output.exists())

    def test_registered_experiments_match_robot_budget_and_video_control_capacity(self):
        zero = load_video_config(CONFIGS / "U0_robot_only.json")
        one = load_video_config(CONFIGS / "U1_feature_prediction.json")
        two = load_video_config(CONFIGS / "U2_effect_constraints.json")
        budgets = []
        counts = []
        for config in (zero, one, two):
            steps, schedule = config["training"]["max_steps"], config["domain_schedule"]
            budgets.append(sum(schedule[i % len(schedule)] == "robot" for i in range(steps)))
            encoder, predictor = build_video_models(config, "cpu")
            counts.append(sum(p.numel() for module in (encoder, predictor) for p in module.parameters()))
        self.assertEqual(budgets, [250, 250, 250])
        self.assertEqual(len(set(counts)), 1)
        self.assertEqual(one["domain_schedule"], two["domain_schedule"])
        self.assertEqual(one["training"], two["training"])
        for key in ("window_frames", "context_frames", "evaluation"):
            self.assertEqual(one[key], two[key])
        for name in ("geometry", "relations", "events", "capacity"):
            self.assertEqual(one["loss_weights"][name], 0)
            self.assertGreater(two["loss_weights"][name], 0)
        # U0 fixes robot exposure, not total training compute: report this
        # explicitly rather than claiming all three have equal total steps.
        self.assertLess(zero["training"]["max_steps"], one["training"]["max_steps"])


if __name__ == "__main__":
    unittest.main()
