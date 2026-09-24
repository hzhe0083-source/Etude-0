from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from evo_wam.g_pi_context import frozen_base_checksum
from evo_wam.g_pi_intent import load_intent_table
from evo_wam.g_pi_probe import (_data_identity, extract_probe_features, fit_intent_probe,
    load_probe_features, ordered_probe_features, probe_g_pi_intent, save_probe_features,
    validate_probe_split)
from evo_wam.icl_data import LATENT_NORMALIZATION


def write_probe_table(folder):
    folder = Path(folder)
    rows = []
    for split in ("train", "test"):
        for label in range(2):
            for index in range(2):
                name = f"{split}-{label}-{index}"
                random = np.random.default_rng(len(rows))
                np.savez(folder / f"{name}.npz", latent=random.normal(size=(4, 3, 1, 2)).astype(np.float32),
                         frame_times=np.arange(3, dtype=np.float32))
                rows.append({"demo_id": name, "purpose_group": f"purpose-{label}", "split": split,
                    "source_id": name, "source_group": name, "person_id": f"{split}-person-{index}",
                    "scene_id": f"{split}-scene-{index}", "view_id": f"view-{index}",
                    "object_ids": ["cup", "bowl"], "object_family": "containers",
                    "arrays": f"{name}.npz", "complete_demo": True})
    metadata = {"format_version": 1, "kind": "g_pi_intent_groups", "data_version": "v1",
        "feature_space_id": "fixture", "latent_normalization": LATENT_NORMALIZATION,
        "grouping_evidence": "offline reviewed purpose groups", "entries": rows}
    path = folder / "purposes.json"
    path.write_text(json.dumps(metadata))
    return path


def fixture_features(table):
    # Every row's unordered average is zero; only role/step slot order is useful.
    return {entry.demo_id: torch.tensor([[1., 0.], [-1., 0.]]) *
            (1 if entry.purpose_group == "purpose-0" else -1) for entry in table.entries}


def fixture_identity(table):
    return {"kind": "g_pi_demo_features", "layer": 1, "pool_grid": [2, 1, 1],
        "token_order": "time_row_column", "conditioning": "pretrained_empty_prompt_only",
        "base_sha256": "a" * 64, "empty_text_identity": {"source": "fixture", "sha256": "b" * 64},
        "context": {"chunk_size": 2, "max_frame_chunk_size": 4, "icl_rope_h": 4, "window_size": 8},
        "data_files": _data_identity(table)}


class IntentProbeTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = write_probe_table(self.root)
        self.table = load_intent_table(self.path)

    def test_ordered_readout_distinguishes_role_swap_without_mean_pooling(self):
        features = fixture_features(self.table)
        self.assertTrue(all(torch.equal(row.mean(0), torch.zeros(2)) for row in features.values()))
        report = fit_intent_probe(self.table, features, steps=40, learning_rate=.1)
        self.assertEqual(report["test_accuracy"], 1.)
        self.assertEqual(report["random_accuracy"], .5)
        self.assertEqual(report["majority_accuracy"], .5)
        self.assertEqual(report["split_counts"]["test"]["independent_sources"], 4)
        self.assertEqual(report["shared_purposes"], ["purpose-0", "purpose-1"])

    def test_pooling_preserves_time_row_and_column_order(self):
        grid = (2, 2, 3)
        features = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
        pooled = ordered_probe_features(features, grid, grid)
        torch.testing.assert_close(pooled, features, rtol=0, atol=0)
        reversed_time = features.reshape(1, *grid, 1).flip(1).flatten(1, 3)
        self.assertFalse(torch.equal(ordered_probe_features(reversed_time, grid, grid), pooled))
        for bad in ((0, 2, 2), (2, 2), (True, 1, 1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ordered_probe_features(features, grid, bad)
        with self.assertRaisesRegex(ValueError, "finite"):
            ordered_probe_features(features * float("nan"), grid, grid)

    def test_holdout_person_scene_and_transitive_source_aliases(self):
        for field in ("person_id", "scene_id"):
            entries = list(self.table.entries)
            entries[-1] = replace(entries[-1], **{field: getattr(entries[0], field)})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                validate_probe_split(replace(self.table, entries=tuple(entries)))
        document = json.loads(self.path.read_text())
        document["source_aliases"] = [
            {"source_id": "train-0-0", "source_group": "bridge", "domain": "human", "split": "train"},
            {"source_id": "test-0-0", "source_group": "bridge", "domain": "human", "split": "test"}]
        self.path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "split"):
            load_intent_table(self.path)

    def test_renamed_copied_video_cannot_cross_probe_split(self):
        (self.root / "test-0-0.npz").write_bytes((self.root / "train-0-0.npz").read_bytes())
        with self.assertRaisesRegex(ValueError, "split"):
            load_intent_table(self.path)

    def test_repeat_sources_do_not_inflate_accuracy_or_sample_count(self):
        features = fixture_features(self.table)
        entries = list(self.table.entries)
        wrong = entries[-1]
        features[wrong.demo_id] = -features[wrong.demo_id]
        first = fit_intent_probe(self.table, features, steps=40, learning_rate=.1)
        for number in range(5):
            name = f"reuse-{number}"
            entries.append(replace(wrong, demo_id=name))
            features[name] = features[wrong.demo_id]
        repeated = fit_intent_probe(replace(self.table, entries=tuple(entries)), features,
                                    steps=40, learning_rate=.1)
        self.assertAlmostEqual(first["test_accuracy"], .75)
        self.assertAlmostEqual(first["test_accuracy"], repeated["test_accuracy"], places=6)
        self.assertEqual(repeated["split_counts"]["test"]["independent_sources"], 4)

    def test_test_features_cannot_change_fitted_training_loss_or_rng(self):
        features = fixture_features(self.table)
        before = torch.get_rng_state().clone()
        first = fit_intent_probe(self.table, features, steps=20)
        for entry in self.table.entries:
            if entry.split == "test":
                features[entry.demo_id] = features[entry.demo_id] * 100 + 500
        changed = fit_intent_probe(self.table, features, steps=20)
        self.assertEqual(first["train_loss"], changed["train_loss"])
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)

    def test_unknown_purposes_are_reported_and_not_silently_omitted(self):
        entries = list(self.table.entries)
        entries[-1] = replace(entries[-1], purpose_group="novel-composition")
        table = replace(self.table, entries=tuple(entries))
        report = fit_intent_probe(table, fixture_features(table), steps=30, learning_rate=.1)
        self.assertEqual(report["unseen_test_purposes"], ["novel-composition"])
        self.assertEqual(report["test_accuracy"], .75)
        self.assertEqual(report["known_purpose_accuracy"], 1.)
        self.assertEqual(report["random_accuracy"], .375)

    def test_cache_only_command_roundtrip_and_file_identity(self):
        cache = self.root / "features.npz"
        features, identity = fixture_features(self.table), fixture_identity(self.table)
        save_probe_features(cache, self.table, features, identity)
        restored, recorded = load_probe_features(cache, self.table, layer=1, pool_grid=(2, 1, 1))
        self.assertEqual(identity, recorded)
        for key in features:
            torch.testing.assert_close(features[key], restored[key], rtol=0, atol=0)
        manifest = {"format_version": 1, "kind": "g_pi_intent_probe", "purpose_table": self.path.name,
                    "layer": 1, "pool_grid": [2, 1, 1], "feature_cache": cache.name,
                    "steps": 40, "learning_rate": .1}
        path = self.root / "probe.json"
        path.write_text(json.dumps(manifest))
        output = self.root / "report.json"
        result = probe_g_pi_intent(SimpleNamespace(manifest=path, output=output))
        self.assertEqual(result["test_accuracy"], 1.)
        self.assertEqual(json.loads(output.read_text())["feature_identity"], identity)
        with self.assertRaisesRegex(ValueError, "pooling identity mismatch"):
            load_probe_features(cache, self.table, layer=1, pool_grid=(1, 1, 2))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            probe_g_pi_intent(SimpleNamespace(manifest=path, output=output, artifact="unused.pt"))
        self.path.write_text(self.path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "input file identity changed"):
            load_probe_features(cache, self.table, layer=1, pool_grid=(2, 1, 1))

    def test_finite_shape_and_configuration_checks(self):
        for value in (float("nan"), float("inf")):
            bad = fixture_features(self.table)
            bad[self.table.entries[0].demo_id][0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                fit_intent_probe(self.table, bad)
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            fit_intent_probe(self.table, {})
        with self.assertRaisesRegex(ValueError, "steps"):
            fit_intent_probe(self.table, fixture_features(self.table), steps=0)
        with self.assertRaisesRegex(ValueError, "overflowed"):
            fit_intent_probe(self.table, {key: value * 1e38
                for key, value in fixture_features(self.table).items()})

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_tiny_native_demo_only_frozen_extraction_and_artifact_entry(self):
        from test_g_pi_training import config_for
        from evo_wam.g_pi_training import (build_g_pi_system, _base_reference, _system_state,
            g_pi_artifact_version, g_pi_architecture, conditioning_mode)
        from evo_wam.zerowam import ZERO_WAM_COMMIT

        config = config_for("g_translator")
        empty_path = self.root / "empty.pt"
        torch.save(torch.arange(24, dtype=torch.float32).reshape(1, 3, 8) / 24, empty_path)
        config["empty_text_emb_path"] = str(empty_path)
        registry = {"action_space": {"dimension": 3}, "end_effectors": ["arm"],
                    "event_rules": config["event_rules"]}
        native, interface, encoder, base_identity, layers = build_g_pi_system(
            config, registry, stage="g", tiny_native=True, device="cuda")
        before = frozen_base_checksum(native)
        with patch.object(native.action_embedder, "forward", side_effect=AssertionError("robot action read")), \
             patch.object(native.condition_embedder_action, "forward", side_effect=AssertionError("robot state read")):
            features, identity = extract_probe_features(self.table, native, config, layer=1, pool_grid=(2, 1, 1))
        self.assertEqual(identity["base_sha256"], before)
        self.assertEqual(frozen_base_checksum(native), before)
        self.assertTrue(all(row.shape == (2, native.inner_dim) and not row.requires_grad
                            and torch.isfinite(row).all() for row in features.values()))
        manifest = {"format_version": 1, "kind": "g_pi_intent_probe", "purpose_table": self.path.name,
                    "layer": 1, "pool_grid": [2, 1, 1], "steps": 2,
                    "save_feature_cache": "native-features.npz"}
        path = self.root / "probe.json"
        path.write_text(json.dumps(manifest))
        artifact = self.root / "training.pt"
        torch.save({"format_version": g_pi_artifact_version(config), "kind": "g_pi_training",
            "config": config, "interface_type": "g_translator", "stage": "g", "p_drop": 0,
            "architecture": g_pi_architecture(config), "conditioning_mode": conditioning_mode(config),
            "demo_route": config.get("demo_route", "one_way"), "upstream_commit": ZERO_WAM_COMMIT,
            "precision": "float32", "encoder_precision": "bfloat16", "tiny_native": True,
            "registry": registry, "event_rules": config["event_rules"], "base_identity": base_identity,
            "encoder_identity": encoder.identity, "empty_text_identity": encoder.identity["empty_text_identity"],
            "k_z": encoder.k_z, "d_z": encoder.d_z, "feature_layers": layers,
            "base_reference": _base_reference(config, None, True, base_identity, encoder),
            "model": _system_state(native, interface, encoder)}, artifact)
        result = probe_g_pi_intent(SimpleNamespace(manifest=path, output=self.root / "report.json",
                                                   artifact=artifact, device="cuda", checkpoint=None))
        self.assertTrue(0 <= result["test_accuracy"] <= 1)
        self.assertEqual(frozen_base_checksum(native), before)
        self.assertTrue((self.root / "native-features.npz").exists())


if __name__ == "__main__":
    unittest.main()
