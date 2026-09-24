from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from evo_wam.cli import file_sha256
from evo_wam.g_pi_context import FrozenGoalEncoder, save_target_cache
from evo_wam.g_pi_data import load_g_pi_sample
from evo_wam.g_pi_targets import build_target_cache_index, cache_g_pi_targets, load_target_cache_index
from test_g_pi_context import CAMERAS, context_model
from test_g_pi_data import write_g_pi_task


class _FixtureEncoder:
    identity = {"k_z": 2, "d_z": 3, "base_sha256": "fixture-base", "layer": 1}

    def __init__(self):
        self.calls = []

    def __call__(self, frame):
        self.calls.append(frame.clone())
        value = frame.flatten()[0].float()
        token = torch.stack((value, value + 1, value + 2))
        return torch.nn.functional.normalize(torch.stack((token, token.flip(0)))[None], dim=-1)


class TargetCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.task, self.metadata, self.arrays = write_g_pi_task(self.root)
        self.index = self.root / "index.json"
        self.index.write_text(json.dumps({"format_version": 1, "kind": "g_pi_index",
                                         "samples": [{"manifest": self.task.name, "split": "train"}]}))
        self.encoder = _FixtureEncoder()

    def build(self):
        return build_target_cache_index(self.index, self.root / "cache", self.encoder)

    def test_independent_single_frames_exact_order_and_no_text(self):
        (self.root / self.metadata["language"]).unlink()
        (self.root / self.metadata["demonstration"]["arrays"]).unlink()
        path = self.build()
        self.assertEqual(len(self.encoder.calls), len(self.arrays["subgoal_times"]))
        for actual, expected in zip(self.encoder.calls, self.arrays["subgoal_latents"]):
            torch.testing.assert_close(actual, torch.from_numpy(expected[None]), atol=0, rtol=0)
            self.assertEqual(actual.shape[:3], (1, 2, 1))
        cache = load_target_cache_index(path, self.encoder.identity, task_paths=[self.task])
        sample = load_g_pi_sample(self.task, current_time=.4, read_language=False)
        expected = self.encoder(sample.target_frame)
        with patch.object(_FixtureEncoder, "__call__", side_effect=AssertionError("E must not run")):
            actual = cache.goal(sample, self.task)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.zero_()
        torch.testing.assert_close(cache.goal(sample, self.task), expected, atol=0, rtol=0)
        self.assertTrue({self.task, self.index, path, self.root / self.metadata["arrays"]} <= set(cache.files))

    def test_later_current_time_selects_terminal_cache(self):
        path = self.build()
        cache = load_target_cache_index(path, self.encoder.identity)
        sample = load_g_pi_sample(self.task, current_time=.8)
        self.assertEqual(sample.subgoal_time, sample.terminal_time)
        torch.testing.assert_close(cache.goal(sample, self.task), self.encoder(sample.target_frame), atol=0, rtol=0)

    def test_existing_cache_is_never_overwritten(self):
        path = self.build()
        expected = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "fresh directory"):
            self.build()
        self.assertEqual(path.read_bytes(), expected)

    def test_wrong_encoder_identity_rejected(self):
        path = self.build()
        with self.assertRaisesRegex(ValueError, "E identity mismatch"):
            load_target_cache_index(path, dict(self.encoder.identity, base_sha256="another-base"))

    def test_changed_source_manifest_or_arrays_rejected(self):
        path = self.build()
        self.task.write_text(json.dumps(dict(self.metadata, sample_id="renamed")))
        with self.assertRaisesRegex(ValueError, "source task manifest changed"):
            load_target_cache_index(path, self.encoder.identity)
        self.task.write_text(json.dumps(self.metadata))
        self.arrays["subgoal_latents"][0] += 10
        np.savez_compressed(self.root / self.metadata["arrays"], **self.arrays)
        with self.assertRaisesRegex(ValueError, "source arrays changed"):
            load_target_cache_index(path, self.encoder.identity)

    def test_changed_dataset_index_rejected(self):
        path = self.build()
        self.index.write_text(self.index.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "dataset index changed"):
            load_target_cache_index(path, self.encoder.identity)

    def test_boundary_and_latent_order_rejected(self):
        path = self.build()
        original = path.read_text()
        for field, message in (("subgoal_times", "boundary order"),
                               ("subgoal_latent_sha256", "latent identity or order")):
            document = json.loads(original)
            document["records"][0][field].reverse()
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, message):
                load_target_cache_index(path, self.encoder.identity)

    def test_wrong_token_archive_and_missing_targets_rejected(self):
        path = self.build()
        document = json.loads(path.read_text())
        record = document["records"][0]
        target = path.parent / record["cache"]
        save_target_cache(target, torch.ones(1, 2, 3), self.encoder.identity)
        with self.assertRaisesRegex(ValueError, "token archive changed"):
            load_target_cache_index(path, self.encoder.identity)
        record["cache_sha256"] = file_sha256(target)
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "every subgoal"):
            load_target_cache_index(path, self.encoder.identity)

    def test_token_archive_mutation_after_index_load_rejected(self):
        path = self.build()
        cache = load_target_cache_index(path, self.encoder.identity)
        sample = load_g_pi_sample(self.task, current_time=.4)
        record = json.loads(path.read_text())["records"][0]
        save_target_cache(path.parent / record["cache"], torch.ones(2, 2, 3), self.encoder.identity)
        with self.assertRaisesRegex(ValueError, "changed after loading"):
            cache.goal(sample, self.task)

    def test_duplicate_or_incomplete_index_rejected(self):
        path = self.build()
        document = json.loads(path.read_text())
        document["records"].append(document["records"][0])
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "unique"):
            load_target_cache_index(path, self.encoder.identity)
        path = build_target_cache_index(self.index, self.root / "another-cache", self.encoder)
        with self.assertRaisesRegex(ValueError, "every requested training task"):
            load_target_cache_index(path, self.encoder.identity, task_paths=[self.root / "other.json"])

    def test_selected_sample_must_match_cache(self):
        path = self.build()
        cache = load_target_cache_index(path, self.encoder.identity)
        sample = load_g_pi_sample(self.task, current_time=.4)
        with self.assertRaisesRegex(ValueError, "single-frame latent differs"):
            cache.goal(replace(sample, target_frame=sample.target_frame + 1), self.task)
        with self.assertRaisesRegex(ValueError, "subgoal_time is absent"):
            cache.goal(replace(sample, subgoal_time=.7), self.task)
        with self.assertRaisesRegex(ValueError, "sample identity"):
            cache.goal(replace(sample, metadata=dict(sample.metadata, sample_id="wrong")), self.task)
        with self.assertRaisesRegex(ValueError, "task is absent"):
            cache.goal(sample, self.root / "other.json")

    def test_source_mutation_during_encoding_rejected(self):
        encoder = self.encoder

        class ChangedEncoder:
            identity = encoder.identity

            def __call__(inner, frame):
                self.task.write_text(self.task.read_text() + "\n")
                return encoder(frame)

        with self.assertRaisesRegex(ValueError, "changed during offline encoding"):
            build_target_cache_index(self.index, self.root / "cache", ChangedEncoder())

    def test_command_reconstructs_encoder_and_emits_usable_index(self):
        args = SimpleNamespace(index=self.index, output=self.root / "cache", artifact="fixture.pt",
                               checkpoint=None, device="cpu", split="train")
        with patch("evo_wam.g_pi_training.load_g_pi_encoder", return_value=(self.encoder, {})) as factory:
            result = cache_g_pi_targets(args)
        factory.assert_called_once_with("fixture.pt", device="cpu", checkpoint=None)
        cache = load_target_cache_index(result["target_cache_index"], self.encoder.identity)
        self.assertEqual(result["encoder_identity"], self.encoder.identity)
        self.assertGreater(len(cache.files), 3)

    def test_fresh_config_command_needs_no_training_artifact_or_task_language(self):
        config = self.root / "training-config.json"
        config.write_text(json.dumps({"interface_type": "pi_goal"}))
        (self.root / self.metadata["language"]).unlink()
        (self.root / self.metadata["demonstration"]["arrays"]).unlink()
        args = SimpleNamespace(index=self.index, output=self.root / "cache", config=config,
                               checkpoint="base", device="cpu", tiny_native=True)
        with patch("evo_wam.g_pi_training.build_g_pi_encoder",
                   return_value=(None, self.encoder, {}, [])) as factory:
            result = cache_g_pi_targets(args)
        factory.assert_called_once_with({"interface_type": "pi_goal"}, stage="pi", device="cpu",
                                        checkpoint="base", tiny_native=True)
        self.assertTrue(Path(result["target_cache_index"]).is_file())
        with self.assertRaisesRegex(ValueError, "exactly one"):
            cache_g_pi_targets(SimpleNamespace(artifact="a.pt", config=config))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            cache_g_pi_targets(SimpleNamespace())

    def test_fresh_encoder_matches_full_system_identity_without_task_language(self):
        from evo_wam.g_pi_training import build_g_pi_encoder, build_g_pi_system
        from evo_wam.goal_training import goal_registry
        from test_g_pi_training import config_for
        from test_native_icl import tiny_model

        def build(config, **kwargs):
            native = tiny_model("cpu").float()
            return native, torch.ones(1, 3, 8), {"kind": "native-test-fixture"}

        sample = load_g_pi_sample(self.task, current_time=0.)
        registry = goal_registry(sample)
        registry["action_space"] = dict(registry["action_space"], dimension=3,
                                         valid_channels=[True, True, False])
        with patch("evo_wam.g_pi_training.build_icl_model", side_effect=build):
            for route, stage in (("pi_goal", "pi"), ("g_translator", "g")):
                config = config_for(route)
                native, encoder, identity, layers = build_g_pi_encoder(config, stage=stage,
                                                                       tiny_native=True, device="cpu")
                _, _, trained_encoder, trained_identity, _ = build_g_pi_system(config, registry,
                    stage=stage, tiny_native=True, device="cpu")
                self.assertEqual(identity, trained_identity)
                self.assertEqual(encoder.identity, trained_encoder.identity)
                self.assertEqual(layers, [0, 1])
                self.assertTrue(all(not p.requires_grad for p in native.parameters()))

    def test_fresh_g_cache_does_not_require_decoder_purpose_groups(self):
        config = self.root / "g.json"
        original = {"interface_type": "g_translator", "intent_training": {
                    "contrastive_weight": 1., "data_version": "v1"}}
        config.write_text(json.dumps(original))
        args = SimpleNamespace(index=self.index, output=self.root / "cache", config=config,
                               device="cpu", tiny_native=True)
        with patch("evo_wam.g_pi_training.build_g_pi_encoder",
                   return_value=(None, self.encoder, {}, [])) as factory:
            result = cache_g_pi_targets(args)
        settings = factory.call_args.args[0]["intent_training"]
        self.assertEqual(settings["contrastive_weight"], 0.)
        self.assertIsNone(settings["manifest"])
        self.assertEqual(json.loads(config.read_text()), original)
        self.assertTrue(Path(result["target_cache_index"]).is_file())

    @unittest.skipUnless(torch.cuda.is_available(), "native E target caching requires CUDA")
    def test_native_target_cache_is_bitwise_equal_to_independent_encoding(self):
        native = context_model("cuda")
        encoder = FrozenGoalEncoder(native, camera_layout=CAMERAS, layer=1, grid_size=(2, 2))
        for field in ("latent", "subgoal_latents"):
            self.arrays[field] = np.concatenate([self.arrays[field], self.arrays[field]],
                                                axis=0 if field == "latent" else 1)
        np.savez_compressed(self.root / self.metadata["arrays"], **self.arrays)
        path = build_target_cache_index(self.index, self.root / "cache", encoder)
        cache = load_target_cache_index(path, encoder.identity)
        for time in (0., .4, .8, 1.2):
            sample = load_g_pi_sample(self.task, current_time=time)
            online = encoder(sample.target_frame).cpu()
            torch.testing.assert_close(cache.goal(sample, self.task), online, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
