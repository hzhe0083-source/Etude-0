import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from etude.icl_data import load_icl_sample
from etude.icl_training import (build_icl_model, load_icl_config,
                                  native_icl_loss, prepare_icl_inputs, train_native_icl)
from etude.zerowam import NativeDependencyError, load_native_class
from test_icl_data import save_sample, write_sample


class NativeICLTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            load_native_class()
        except NativeDependencyError as exc:
            raise unittest.SkipTest(str(exc))

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = load_icl_config(Path(__file__).parents[1] / "configs/icl/H1_cross_video.json")
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config))
        rng = np.random.default_rng(19)
        manifests = []
        for name, robot in (("human", False), ("robot", True)):
            path, metadata, _, _ = write_sample(self.root, name, robot=robot)
            demo = {"latent": rng.normal(size=(4, 6, 1, 2)).astype(np.float32),
                    "frame_times": np.arange(6, dtype=np.float64) * 0.2}
            target = {"latent": rng.normal(size=(4, 6, 1, 2)).astype(np.float32),
                      "frame_times": np.arange(6, dtype=np.float64) * 0.3}
            if robot:
                metadata["target"]["action_space"].update(dimension=3, valid_channels=[True, True, False])
                target.update(actions=rng.normal(size=(3, 6, 2, 1)).astype(np.float32),
                              actions_mask=np.ones((3, 6, 2, 1), dtype=np.bool_))
                target["actions"][2] = np.nan
            save_sample(path, metadata, demo, target)
            manifests.append(path.name)
            if not robot:
                reverse = {**metadata, "sample_id": "reverse", "demonstration": metadata["target"],
                           "target": metadata["demonstration"]}
                (self.root / "reverse.json").write_text(json.dumps(reverse))
                manifests.append("reverse.json")
        self.index = self.root / "index.json"
        self.index.write_text(json.dumps({"format_version": 1, "kind": "native_icl_index",
            "samples": [{"manifest": name, "split": "train"} for name in manifests]}))

    def args(self, output, steps, resume=None):
        return SimpleNamespace(config=str(self.config_path), index=str(self.index),
            output=str(self.root / output), steps=steps, seed=19, device="cuda",
            tiny_native=True, checkpoint=None, resume=resume)

    def assert_same(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_same(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_same(first, second)
        else:
            self.assertEqual(left, right)

    def test_human_control_keeps_identical_noisy_targets_and_ifp_without_actions(self):
        model, null, _ = build_icl_model(self.config, tiny_native=True, device="cpu")
        sample = load_icl_sample(self.root / "human.json")
        cross = prepare_icl_inputs(sample, self.config, model, null, torch.Generator().manual_seed(11))
        ordinary = prepare_icl_inputs(sample, {**self.config, "human_context": "none"}, model, null,
                                      torch.Generator().manual_seed(11))
        self.assertNotIn("action_dict", cross)
        self.assertNotIn("action_dict", ordinary)
        self.assertNotIn("icl_latent_dict", ordinary)
        self.assert_same(cross["icl_latent_dict"]["latent"], sample.demonstration)
        self.assertEqual(cross["text_emb"].shape[1], ordinary["text_emb"].shape[1] * 2)
        self.assertEqual(set(cross) - set(ordinary), {"icl_latent_dict"})
        for key in ordinary.keys() - {"text_emb", "encoder_seq_ids"}:
            self.assert_same(cross[key], ordinary[key])
        self.assertFalse(cross["latent_dict"]["valid_mask"][:, :, :2].any())
        self.assertTrue(cross["latent_dict"]["valid_mask"][:, :, 2:].all())
        self.assertTrue(cross["mcp_latent_dicts"][0]["valid_mask"].any())
        self.assertFalse(cross["mcp_latent_dicts"][-1]["valid_mask"].any())
        robot = prepare_icl_inputs(load_icl_sample(self.root / "robot.json"), self.config, model, null,
                                   torch.Generator().manual_seed(11))
        self.assertTrue(robot["latent_dict"]["valid_mask"].all())
        self.assertTrue(robot["action_dict"]["valid_mask"][:, :2, :2].all())
        self.assertFalse(robot["action_dict"]["valid_mask"][:, 2].any())
        self.assertEqual(robot["action_dict"]["latent"][:, 2].count_nonzero().item(), 0)
        self.assertGreater(robot["action_dict"]["noisy_latents"][:, 2, :2].count_nonzero().item(), 0)
        self.assertFalse(robot["action_dict"]["valid_mask"][:, 2, 2:].any())

    def test_robot_loss_matches_native_full_window_and_masked_action_mean(self):
        video = torch.tensor([[[2.], [4.]]])
        action = torch.tensor([[[2., 7.], [4., 9.]]])
        native = Mock(patch_size=(1, 1, 1), return_value=(video, action, [video]))
        target = torch.zeros(1, 1, 2, 1, 1)
        latent = {"targets": target, "valid_mask": torch.ones_like(target, dtype=torch.bool),
                  "training_weight": torch.tensor([[3., 1.]])}
        target_action = torch.zeros(1, 2, 2, 1, 1)
        action_stream = {"targets": target_action, "valid_mask": torch.ones_like(target_action, dtype=torch.bool),
                         "training_weight": torch.ones(1, 2)}
        action_stream["valid_mask"][:, 1] = False
        future = {**latent, "valid_mask": latent["valid_mask"].clone()}
        future["valid_mask"][:, :, 0] = False
        losses = native_icl_loss(native, {"latent_dict": latent, "action_dict": action_stream,
            "mcp_latent_dicts": [future]}, self.config, human=False)
        self.assertEqual(losses["video"].item(), 14.)  # (2² * 3 + 4²) / all 2 frames
        self.assertEqual(losses["action"].item(), 5.)  # (2² + 4²) / all 4 action elements
        self.assertEqual(losses["ifp"].item(), 8.)  # 0.5 * 4² / 1 valid future element
        self.assertEqual(losses["total"].item(), 27.)

    def test_control_coverage_requires_the_same_clip_cache_not_just_source_id(self):
        reverse = self.root / "reverse.json"
        metadata = json.loads(reverse.read_text())
        cache = self.root / metadata["target"]["arrays"]
        duplicate = self.root / "different-window.npz"
        duplicate.write_bytes(cache.read_bytes())
        metadata["target"]["arrays"] = duplicate.name
        reverse.write_text(json.dumps(metadata))
        with patch("etude.icl_training.build_icl_model", side_effect=AssertionError("validate coverage before model loading")):
            with self.assertRaisesRegex(ValueError, "every demo video cache"):
                train_native_icl(self.args("unmatched", 1))

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_native_mixed_updates_exact_resume_and_original_class_export(self):
        # Snapshot the identical seeded base to prove the training loop only changes adapters.
        torch.manual_seed(19)
        model, null, _ = build_icl_model(self.config, tiny_native=True, device="cuda")
        initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        human = prepare_icl_inputs(load_icl_sample(self.root / "human.json"), self.config, model, null,
                                   torch.Generator().manual_seed(11))
        with patch.object(model.action_embedder, "forward", side_effect=AssertionError("human action embedding")), \
             patch.object(model.action_proj_out, "forward", side_effect=AssertionError("human action head")):
            losses = native_icl_loss(model, human, self.config, human=True)
        self.assertEqual(set(losses), {"video", "ifp", "total"})
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        self.assertGreater(losses["ifp"].item(), 0)
        del losses, human, model, null

        full_report = train_native_icl(self.args("full", 4))
        full = torch.load(full_report["artifact"], weights_only=True, map_location="cpu")
        self.assertEqual(full["domain_updates"], {"robot": 1, "human": 3})
        changed = {name for name in initial if not torch.equal(initial[name], full["model"][name])}
        self.assertTrue(changed)
        self.assertTrue(all(".up.weight" in name or ".down.weight" in name for name in changed), changed)
        rows = [json.loads(row) for row in (self.root / "full/metrics.jsonl").read_text().splitlines()]
        self.assertTrue(all(("action" in row) == (row["domain"] == "robot") for row in rows))
        self.assertFalse(full_report["action_parameters_updated"])
        self.assertFalse(full_report["test_time_updates"])

        partial = train_native_icl(self.args("resumed", 2))
        resumed_report = train_native_icl(self.args("resumed", 2, partial["artifact"]))
        resumed = torch.load(resumed_report["artifact"], weights_only=True, map_location="cpu")
        for key in ("model", "optimizer", "domain_samples", "domain_updates", "domain_rng", "torch_rng", "cuda_rng"):
            self.assert_same(full[key], resumed[key])
        self.assertEqual(resumed["attempted_steps"], 4)

        # Diffusers' private default-value metadata has hash-dependent ordering;
        # artifact identity must survive exporting in a fresh Python process.
        hash_seed = "98765" if os.environ.get("PYTHONHASHSEED") != "98765" else "98766"
        result = subprocess.run([sys.executable, "-m", "etude", "export-native-icl",
            "--artifact", full_report["artifact"], "--output", str(self.root / "export"), "--device", "cpu"],
            env={**os.environ, "PYTHONHASHSEED": hash_seed}, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        exported = json.loads(result.stdout)
        cls = load_native_class()
        deployed, info = cls.from_pretrained(exported["transformer"], local_files_only=True,
            use_safetensors=True, output_loading_info=True)
        self.assertEqual(type(deployed), cls)
        self.assertFalse(info["missing_keys"])
        self.assertFalse(info["unexpected_keys"])
        self.assertFalse(info["mismatched_keys"])
        self.assertFalse(any(".up." in key or ".down." in key or ".base." in key for key in deployed.state_dict()))
        cls(**dict(deployed.config)).load_state_dict(deployed.state_dict(), strict=True)
        deployed.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
        self.assertTrue(all(not value.requires_grad for value in deployed.parameters()))
        self.assertFalse(exported["test_time_updates"])
        self.assertEqual(exported["commands_sent"], 0)
        adapted, null, _ = build_icl_model(self.config, tiny_native=True, device="cuda")
        adapted.load_state_dict(full["model"], strict=True)
        robot = prepare_icl_inputs(load_icl_sample(self.root / "robot.json"), self.config, adapted, null,
                                   torch.Generator().manual_seed(11))
        frozen = {name: value.clone() for name, value in deployed.state_dict().items()}
        with torch.inference_mode():
            expected = adapted(robot, train_mode=True)
            actual = deployed(robot, train_mode=True)
        for before, after in zip([*expected[:2], *expected[2]], [*actual[:2], *actual[2]]):
            torch.testing.assert_close(before, after, rtol=0.025, atol=0.025)
        self.assert_same(frozen, deployed.state_dict())

        # Resume must reject modified arrays already used, even with unchanged metadata.
        consumed = self.root / "human-B.npz"
        original = consumed.read_bytes()
        with np.load(consumed, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays["latent"][0, 0, 0, 0] += 1
        np.savez_compressed(consumed, **arrays)
        with self.assertRaisesRegex(ValueError, "consumed.*changed"):
            train_native_icl(self.args("resumed", 1, resumed_report["artifact"]))
        consumed.write_bytes(original)
        index = json.loads(self.index.read_text())
        index["samples"][1]["split"] = "test"
        self.index.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "crosses"):
            train_native_icl(self.args("resumed", 1, resumed_report["artifact"]))


if __name__ == "__main__":
    unittest.main()
