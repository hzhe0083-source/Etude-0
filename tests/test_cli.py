"""CPU CLI/data/checkpoint integration; no released model or robot is evaluated."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from evo_wam import cli
from evo_wam.data import load_experiment, load_sample, load_observation
from evo_wam.training import EvoTrainer, LossWeights
from evo_wam.zerowam import TaskConditions, ZERO_WAM_COMMIT


class CheckpointFixture(nn.Module):
    """A small serialization fixture, deliberately not a native WAM substitute."""

    def __init__(self):
        super().__init__()
        self.adapter = nn.Module()
        self.adapter.native = nn.Linear(2, 2).requires_grad_(False)
        self.codec = nn.Linear(2, 2)
        self.optimizer = torch.optim.AdamW(self.codec.parameters(), lr=0.01)
        self.stage = "interface"
        self.updates = 0
        self.tiny_native = True
        self.base_identity = {"kind": "tiny-native-full-state", "config": {"test_only": True}}

    def populate_optimizer(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.codec(torch.ones(1, 2)).square().sum().backward()
        self.optimizer.step()
        self.updates += 1


def configuration():
    return load_experiment(cli.ROOT / "configs" / "V1.json")


class CliTests(unittest.TestCase):
    def test_deployment_observation_has_no_future_or_goal_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            cli.make_fixture(directory)
            path = Path(directory) / "observation.json"
            sample = load_observation(path)
            self.assertEqual(sample.robot_latent.shape, (1, 4, 1, 1, 2))
            self.assertFalse(hasattr(sample, "outcome"))
            self.assertFalse(hasattr(sample, "requirement"))
            self.assertFalse(hasattr(sample, "actions"))
            cli.action_space({"action_space": sample.action_space}, 3)
            array_path = Path(directory) / "observation.npz"
            with np.load(array_path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            arrays["future_labels"] = np.ones(3)
            np.savez(array_path, **arrays)
            with self.assertRaises(ValueError):
                load_observation(path)

    def test_physical_actions_cannot_silently_use_different_units(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = cli.make_fixture(directory)
            sample = load_sample(cli.load_index(paths["index"])[0][0])
            sample.actions.mul_(100)
            with self.assertRaises(ValueError):
                cli.fresh_native(sample, configuration(), torch.Generator().manual_seed(0), "cpu", torch.float32)
            with self.assertRaises(ValueError):
                cli.action_space({"action_space": None}, 3)

    def test_real_fixture_load_and_four_mcp_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = cli.make_fixture(directory)
            samples, records = cli.load_index(paths["index"])
            self.assertEqual(len(samples), 1)
            self.assertEqual(records[0]["provenance"], "synthetic")
            self.assertIsInstance(samples[0], Path)
            sample = load_sample(samples[0])
            self.assertEqual(sample.robot_history.shape, (1, 2, 3, 4))
            self.assertEqual(sample.proprio_history.shape, (1, 2, 3))
            config = configuration()
            generator = torch.Generator().manual_seed(53)
            native = cli.fresh_native(sample, config, generator, "cpu", torch.float32)
            self.assertEqual(len(native["mcp_latent_dicts"]), 4)
            video = native["latent_dict"]
            # These counts follow the actual pinned shift utility, not mocked data.
            for stream, count in zip(native["mcp_latent_dicts"], (16, 12, 8, 4)):
                self.assertEqual(int(stream["valid_mask"].sum()), count)
                self.assertTrue(bool(stream["valid_mask"].any()))
                self.assertEqual(stream["targets"].shape, video["targets"].shape)
                self.assertTrue(torch.isfinite(stream["targets"]).all())
                self.assertTrue(torch.isfinite(stream["training_weight"]).all())
            again = cli.fresh_native(sample, config, generator, "cpu", torch.float32)
            self.assertFalse(torch.equal(video["noisy_latents"], again["latent_dict"]["noisy_latents"]))
            replay = cli.fresh_native(sample, config, torch.Generator().manual_seed(53), "cpu", torch.float32)
            torch.testing.assert_close(video["noisy_latents"], replay["latent_dict"]["noisy_latents"], rtol=0, atol=0)
            self.assertEqual(native["chunk_size"], 2)
            with self.assertRaises(ValueError):
                cli.make_fixture(directory)

    def test_build_batch_reuses_one_native_payload_for_view_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = cli.make_fixture(directory)
            sample = load_sample(cli.load_index(paths["index"])[0][0])
            fixture = CheckpointFixture()
            batch = cli.build_batch(sample, configuration(), fixture, torch.zeros(1, 2, 8),
                                    torch.Generator().manual_seed(5), conditional=True)
            batch.validate()
            self.assertEqual(len(batch.demonstrations), 2)
            # Exercise the actual view-forward wrapper without claiming model execution.
            adapter = SimpleNamespace(forward_train=Mock(return_value=None))
            proxy = SimpleNamespace(adapter=adapter, enable_ifp=True,
                                    weights=LossWeights(next_video=0, native_action=0, ifp=0))
            condition = TaskConditions(torch.zeros(1, 8, 32), torch.zeros(1, 8, 32), batch.null_text)
            EvoTrainer._native(proxy, batch, condition, include_action=False, include_interaction=False)
            EvoTrainer._native(proxy, batch, condition, include_action=False, include_interaction=False)
            first, second = [call.args[0] for call in adapter.forward_train.call_args_list]
            self.assertIs(first["latent_dict"], second["latent_dict"])
            self.assertIs(first["action_dict"], second["action_dict"])
            for left, right in zip(first["mcp_latent_dicts"], second["mcp_latent_dicts"]):
                self.assertIs(left["noisy_latents"], right["noisy_latents"])
                self.assertIs(left["timesteps"], right["timesteps"])
            null_batch = cli.build_batch(sample, configuration(), fixture, batch.null_text,
                                         torch.Generator().manual_seed(5), conditional=False)
            self.assertFalse(null_batch.conditional)
            self.assertIsNone(null_batch.requirements)
            self.assertEqual(null_batch.demonstrations, ())
            for forbidden in ("text_emb", "encoder_seq_ids", "icl_latent_dict"):
                self.assertNotIn(forbidden, null_batch.native_inputs)

    def test_index_rejects_transitive_source_leakage_and_missing_split(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = cli.make_fixture(directory)
            index = Path(paths["index"])
            with patch("numpy.load", side_effect=AssertionError("index must not eagerly load arrays")):
                selected, _ = cli.load_index(index)
            self.assertEqual(selected, [Path(paths["manifest"]).resolve()])
            index.write_text(json.dumps({"samples": [
                {"manifest": "sample.json", "split": "train"},
                {"manifest": "sample.json", "split": "test"},
            ]}))
            with self.assertRaises(ValueError):
                cli.load_index(index)
            for entry in ({"manifest": "sample.json"}, {"manifest": "../outside.json", "split": "train"}):
                index.write_text(json.dumps({"samples": [entry]}))
                with self.subTest(entry=entry), self.assertRaises(ValueError):
                    cli.load_index(index)
            index.write_text(json.dumps({"samples": [{"manifest": "sample.json", "split": "train"}]}))
            with self.assertRaisesRegex(ValueError, "no validation samples"):
                cli.load_index(index, "validation")
            index.write_text(json.dumps({"samples": [{"manifest": "sample.json", "split": "unknown"}]}))
            with self.assertRaises(ValueError):
                cli.load_index(index)

    def test_prefix_only_labels_cannot_train_unexecuted_native_future(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = cli.make_fixture(directory)
            manifest = Path(paths["manifest"])
            meta = json.loads(manifest.read_text())
            meta["executed_steps"] = 2
            manifest.write_text(json.dumps(meta))
            sample = load_sample(cli.load_index(paths["index"])[0][0])
            self.assertFalse(sample.outcome.label_valid["events"][:, 2:].any())
            with self.assertRaisesRegex(ValueError, "fully executed paired window"):
                cli.fresh_native(sample, configuration(), torch.Generator(), "cpu", torch.float32)

    def test_tiny_checkpoint_saves_full_frozen_base_and_restores_rng(self):
        torch.manual_seed(17)
        trainer = CheckpointFixture()
        trainer.populate_optimizer()
        config = configuration()
        generator = torch.Generator().manual_seed(29)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.pt"
            cli.save_run(path, trainer, config, generator, attempted_steps=7, tiny_native=True)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            self.assertIn("adapter.native.weight", saved["model"])
            self.assertIn("adapter.native.bias", saved["model"])
            self.assertEqual(saved["upstream_commit"], ZERO_WAM_COMMIT)
            expected_local = torch.randn(5, generator=generator)
            expected_global = torch.randn(5)
            restored = CheckpointFixture()
            restored_generator = torch.Generator().manual_seed(888)
            start = cli.restore_run(path, restored, config, restored_generator, resume=True)
            self.assertEqual(start, 7)
            self.assertEqual(restored.updates, trainer.updates)
            for name, value in trainer.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
            torch.testing.assert_close(torch.randn(5, generator=restored_generator), expected_local, rtol=0, atol=0)
            torch.testing.assert_close(torch.randn(5), expected_global, rtol=0, atol=0)
            states = list(restored.optimizer.state.values())
            self.assertTrue(states)
            self.assertTrue(all(int(state["step"]) == 1 for state in states))
            # Small serialization training above is not a native model run.
            self.assertTrue(saved["base_identity"]["config"]["test_only"])

    def test_checkpoint_rejects_source_interface_mode_and_base_changes(self):
        trainer = CheckpointFixture()
        config = configuration()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.pt"
            cli.save_run(path, trainer, config, torch.Generator(), 1, True)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            for field, invalid in (("upstream_commit", "wrong-source"), ("format_version", 1)):
                corrupted = dict(saved, **{field: invalid})
                bad = Path(directory) / f"bad-{field}.pt"
                torch.save(corrupted, bad)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    cli.restore_run(bad, CheckpointFixture(), config, torch.Generator(), resume=True)
            wrong_interface = copy.deepcopy(config)
            wrong_interface["interface"] = "geometry"
            with self.assertRaises(ValueError):
                cli.restore_run(path, CheckpointFixture(), wrong_interface, torch.Generator(), resume=True)
            wrong_mode = CheckpointFixture()
            wrong_mode.tiny_native = False
            with self.assertRaisesRegex(ValueError, "base identity or tiny/native mode"):
                cli.restore_run(path, wrong_mode, config, torch.Generator(), resume=True)
            wrong_base = CheckpointFixture()
            wrong_base.base_identity = {"kind": "different-frozen-weights"}
            with self.assertRaisesRegex(ValueError, "base identity or tiny/native mode"):
                cli.restore_run(path, wrong_base, config, torch.Generator(), resume=True)

    def test_rejected_resume_does_not_mutate_destination(self):
        config = configuration()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.pt"
            cli.save_run(path, CheckpointFixture(), config, torch.Generator(), 1, True)
            for mismatch in ("stage", "budget"):
                destination = CheckpointFixture()
                before = {name: tensor.clone() for name, tensor in destination.state_dict().items()}
                other = copy.deepcopy(config)
                if mismatch == "stage":
                    destination.stage = "joint"
                else:
                    other["training"]["max_steps"] += 1
                with self.subTest(mismatch=mismatch):
                    with self.assertRaisesRegex(ValueError, "same stage and full config"):
                        cli.restore_run(path, destination, other, torch.Generator(), resume=True)
                    for name, value in before.items():
                        torch.testing.assert_close(destination.state_dict()[name], value, rtol=0, atol=0)

    def test_help_and_doctor_do_not_load_or_download_weights(self):
        with patch.object(cli, "build_trainer", side_effect=AssertionError("no model load")), \
             patch("torch.load", side_effect=AssertionError("no checkpoint load")), \
             patch("urllib.request.urlopen", side_effect=AssertionError("no downloads")):
            output = io.StringIO()
            with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
                cli.main(["--help"])
            self.assertEqual(stopped.exception.code, 0)
            self.assertIn("make-fixture", output.getvalue())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(cli.main(["doctor"]), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["upstream_commit"], ZERO_WAM_COMMIT)
            self.assertFalse(report["full_checkpoint_verified"])
            self.assertFalse(report["simulator_verified"])
            self.assertFalse(report["real_robot_verified"])


if __name__ == "__main__":
    unittest.main()
