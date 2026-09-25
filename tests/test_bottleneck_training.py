"""Native H1/H2/H3 controls, objective gradients and deployable training state."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from etude.demo_context import PreparedDemoContext, prepare_demo_context
from etude.icl_data import load_icl_sample
from etude.icl_deployment import load_bottleneck_deployment
from etude.icl_training import (_model_state, build_icl_model, export_native_icl,
                                  native_icl_loss, prepare_icl_inputs, train_native_icl)
from etude.native_icl import forward_video_only
import test_icl_training as fixtures


class BottleneckTrainingTests(unittest.TestCase):
    # Reuse the audited reciprocal human pair and robot fixture, not its test cases.
    setUpClass = classmethod(fixtures.NativeICLTrainingTests.setUpClass.__func__)
    setUp = fixtures.NativeICLTrainingTests.setUp
    args = fixtures.NativeICLTrainingTests.args
    assert_same = fixtures.NativeICLTrainingTests.assert_same

    def configure(self, weight=0.):
        self.config["demo_bottleneck"] = dict(dim=32, num_heads=4, group_frames=2,
            tokens_per_group=2, layers=1, consistency_weight=weight)
        self.config_path.write_text(json.dumps(self.config))

    def add_appearance_variants(self):
        for name in ("human", "reverse", "robot"):
            path = self.root / f"{name}.json"
            metadata = json.loads(path.read_text())
            with np.load(self.root / metadata["demonstration"]["arrays"], allow_pickle=False) as archive:
                latent, times = archive["latent"].copy(), archive["frame_times"].copy()
            variant = self.root / f"{name}-appearance.npz"
            # Synthetic appearance perturbation; preserve the clip's spatial/time axes.
            latent += np.array([.5, -.3, .2, -.1], dtype=latent.dtype)[:, None, None, None]
            np.savez_compressed(variant, latent=latent, frame_times=times)
            metadata["appearance_variant"] = {"arrays": variant.name,
                "derived_from": metadata["demonstration"]["source_id"],
                "evidence": "Synthetic fixture: channel offsets only; identical source, geometry and frame times."}
            path.write_text(json.dumps(metadata))

    def test_h1_raw_context_and_h2_preserve_all_target_streams(self):
        self.add_appearance_variants()
        raw_model, null, _ = build_icl_model(self.config, tiny_native=True, device="cpu")
        self.assertFalse(hasattr(raw_model, "demo_bottleneck"))
        raw_config = dict(self.config)
        self.configure()
        native, null, _ = build_icl_model(self.config, tiny_native=True, device="cpu")
        for name in ("human", "robot"):
            sample = load_icl_sample(self.root / f"{name}.json")
            raw = prepare_icl_inputs(sample, raw_config, raw_model, null, torch.Generator().manual_seed(11))
            with patch("etude.demo_context.prepare_demo_context", wraps=prepare_demo_context) as compress:
                compressed = prepare_icl_inputs(sample, self.config, native, null, torch.Generator().manual_seed(11))
            self.assertEqual(compress.call_count, 1)  # H2 never encodes the appearance copy.
            self.assertNotIn("appearance_tokens", compressed)
            self.assert_same(raw["icl_latent_dict"]["latent"], sample.demonstration)
            self.assert_same({key: value for key, value in raw.items() if key != "icl_latent_dict"},
                             {key: value for key, value in compressed.items() if key != "icl_latent_dict"})
            context = compressed["icl_latent_dict"]["latent"]
            self.assertIsInstance(context, PreparedDemoContext)
            self.assertEqual(context.hidden.shape, (1, 6, native.inner_dim))
            self.assertEqual(context.tokens.shape, (1, 6, 32))
            self.assertEqual(context.group_times.numel(), 3)
            self.assertFalse(any(isinstance(value, torch.Tensor) and value.ndim == 5
                                 for value in compressed["icl_latent_dict"].values()))

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_human_video_and_robot_action_update_bottleneck_and_adapter_only(self):
        self.configure()
        for name, objective in (("human", "video"), ("robot", "action")):
            with self.subTest(objective=objective):
                torch.manual_seed(19)
                native, null, _ = build_icl_model(self.config, tiny_native=True, device="cuda")
                initial = {key: value.detach().clone() for key, value in native.named_parameters()}
                inputs = prepare_icl_inputs(load_icl_sample(self.root / f"{name}.json"), self.config,
                                            native, null, torch.Generator().manual_seed(11))
                if name == "human":
                    with patch.object(native.action_embedder, "forward", side_effect=AssertionError("human action embedding")), \
                         patch.object(native.action_proj_out, "forward", side_effect=AssertionError("human action head")):
                        losses = native_icl_loss(native, inputs, self.config, human=True)
                    self.assertNotIn("action", losses)
                else:
                    losses = native_icl_loss(native, inputs, self.config, human=False)
                self.assertTrue(torch.isfinite(losses[objective]))
                losses[objective].backward()
                groups = {
                    "compressor": [p for key, p in native.named_parameters()
                                   if key.startswith("demo_bottleneck.") and not key.startswith("demo_bottleneck.adapter.")],
                    "adapter": list(native.demo_bottleneck.adapter.parameters()),
                    "wam_lora": [p for key, p in native.named_parameters() if ".up." in key or ".down." in key],
                }
                for group, parameters in groups.items():
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters), group)
                optimizer = torch.optim.AdamW([p for p in native.parameters() if p.requires_grad], lr=.01)
                optimizer.step()
                changed = {key for key, value in native.named_parameters() if not torch.equal(initial[key], value)}
                self.assertTrue(any(key.startswith("demo_bottleneck.adapter.") for key in changed))
                self.assertTrue(any(key.startswith("demo_bottleneck.") and not key.startswith("demo_bottleneck.adapter.") for key in changed))
                for key, parameter in native.named_parameters():
                    if not parameter.requires_grad:
                        self.assertIsNone(parameter.grad, key)
                        self.assert_same(parameter, initial[key])

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_h2_exact_resume_full_interface_registry_and_cross_process_export(self):
        self.configure()
        native, _, _ = build_icl_model(self.config, tiny_native=True, device="cpu")
        adapter_only = _model_state(native, False)
        interface = {"demo_bottleneck." + key for key in native.demo_bottleneck.state_dict()}
        self.assertEqual({key for key in adapter_only if key.startswith("demo_bottleneck.")}, interface)
        self.assertIn("demo_bottleneck.adapter.weight", adapter_only)
        self.assertTrue(all(key.startswith("demo_bottleneck.") or ".up." in key or ".down." in key for key in adapter_only))
        del native
        full_report = train_native_icl(self.args("h2-full", 4))
        partial = train_native_icl(self.args("h2-resume", 2))
        resumed_report = train_native_icl(self.args("h2-resume", 2, partial["artifact"]))
        full = torch.load(full_report["artifact"], weights_only=True, map_location="cpu")
        resumed = torch.load(resumed_report["artifact"], weights_only=True, map_location="cpu")
        for key in ("model", "optimizer", "domain_samples", "domain_updates", "domain_rng", "torch_rng", "cuda_rng", "visited_arrays"):
            self.assert_same(full[key], resumed[key])
        self.assertEqual(full["domain_updates"], {"robot": 1, "human": 3})
        self.assertTrue(full_report["demo_bottleneck_updated"])
        destination = self.root / "h2-export"
        result = subprocess.run([sys.executable, "-m", "etude", "export-native-icl",
            "--artifact", full_report["artifact"], "--output", str(destination), "--device", "cpu"],
            env={**os.environ, "PYTHONHASHSEED": "91873"}, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["deployment"], str(destination))
        self.check_export(destination, full)

    def check_export(self, destination, payload):
        self.assertTrue((destination / "backbone/config.json").is_file())
        self.assertTrue((destination / "demo_bottleneck.pt").is_file())
        self.assertTrue((destination / "bottleneck.json").is_file())
        self.assertFalse((destination / "transformer").exists())
        self.assertFalse((destination / "config.json").exists())
        deployed, null = load_bottleneck_deployment(destination, device="cuda")
        self.assertFalse(deployed.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in deployed.parameters()))
        expected_interface = {key.removeprefix("demo_bottleneck."): value
            for key, value in payload["model"].items() if key.startswith("demo_bottleneck.")}
        self.assert_same(expected_interface, {key: value.cpu() for key, value in deployed.demo_bottleneck.state_dict().items()})
        adapted, _, _ = build_icl_model(self.config, tiny_native=True, device="cuda")
        adapted.load_state_dict(payload["model"], strict=True)
        robot = load_icl_sample(self.root / "robot.json")
        # Deployment only needs the original demonstration, never its training augmentation.
        config = {**self.config, "demo_bottleneck": {**self.config["demo_bottleneck"], "consistency_weight": 0.}}
        frozen = {key: value.clone() for key, value in deployed.state_dict().items()}
        with torch.inference_mode():
            original = prepare_icl_inputs(robot, config, adapted, null, torch.Generator().manual_seed(11))
            restored = prepare_icl_inputs(robot, config, deployed, null, torch.Generator().manual_seed(11))
            expected, actual = adapted(original, train_mode=True), deployed(restored, train_mode=True)
        for first, second in zip([*expected[:2], *expected[2]], [*actual[:2], *actual[2]]):
            torch.testing.assert_close(first, second, rtol=.025, atol=.025)
        self.assert_same(frozen, deployed.state_dict())

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_h3_normalized_consistency_without_extra_wam_and_consumed_variant_integrity(self):
        self.configure(weight=.01)
        native, null, _ = build_icl_model(self.config, tiny_native=True, device="cuda")
        with self.assertRaisesRegex(ValueError, "appearance_variant"):
            prepare_icl_inputs(load_icl_sample(self.root / "human.json"), self.config, native,
                               null, torch.Generator().manual_seed(11))
        self.add_appearance_variants()
        sample = load_icl_sample(self.root / "human.json")
        h2_config = {**self.config, "demo_bottleneck": {**self.config["demo_bottleneck"], "consistency_weight": 0.}}
        h2 = prepare_icl_inputs(sample, h2_config, native, null, torch.Generator().manual_seed(11))
        with patch("etude.demo_context.prepare_demo_context", wraps=prepare_demo_context) as compress:
            h3 = prepare_icl_inputs(sample, self.config, native, null, torch.Generator().manual_seed(11))
        self.assertEqual(compress.call_count, 2)
        for key in h2.keys() - {"icl_latent_dict"}:
            self.assert_same(h2[key], h3[key])
        self.assert_same(h2["icl_latent_dict"]["latent"].hidden, h3["icl_latent_dict"]["latent"].hidden)
        with patch("etude.icl_training.forward_video_only", wraps=forward_video_only) as wam:
            losses = native_icl_loss(native, h3, self.config, human=True)
        self.assertEqual(wam.call_count, 1)
        original, augmented = h3["icl_latent_dict"]["latent"].tokens, h3["appearance_tokens"]
        normalized_mse = (F.normalize(original.float(), dim=-1) - F.normalize(augmented.float(), dim=-1)).square().mean()
        self.assert_same(losses["appearance_consistency"], normalized_mse)
        self.assertGreater(normalized_mse.item(), 0.)
        gradient = torch.autograd.grad(normalized_mse, native.demo_bottleneck.mix[0].weight)[0]
        self.assertGreater(gradient.abs().sum().item(), 0.)
        torch.testing.assert_close(losses["total"], losses["video"] + losses["ifp"] + .01 * normalized_mse)
        del losses, h2, h3, native, null
        report = train_native_icl(self.args("h3", 2))
        payload = torch.load(report["artifact"], weights_only=True, map_location="cpu")
        consumed = self.root / "robot-appearance.npz"
        self.assertIn(str(consumed), payload["visited_arrays"])
        destination = self.root / "h3-export"
        export_native_icl(SimpleNamespace(artifact=report["artifact"], output=str(destination),
                                         checkpoint=None, device="cpu"))
        self.check_export(destination, payload)
        with np.load(consumed, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays["latent"][0, 0, 0, 0] += 1
        np.savez_compressed(consumed, **arrays)
        with self.assertRaisesRegex(ValueError, "consumed.*changed"):
            train_native_icl(self.args("h3", 1, report["artifact"]))


if __name__ == "__main__":
    unittest.main()
