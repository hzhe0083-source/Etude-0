import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from evo_wam.g_pi_context import (
    FrozenGoalEncoder, assert_frozen_base, build_demo_cache, frozen_base_checksum,
    g_attention_mask, g_context_features, install_empty_text, load_target_cache, pi_context_features,
    save_target_cache, truncate_robot_history,
)
from evo_wam.zerowam import NativeDependencyError, load_native_class
from test_native_icl import tiny_model


CONFIG = {"chunk_size": 2, "max_frame_chunk_size": 4, "icl_rope_h": 4, "window_size": 8}


def context_model(device="cpu"):
    native = tiny_model(device)
    install_empty_text(native, torch.randn(1, 4, 8), "fixture_empty_prompt")
    return native


class GPiContextTests(unittest.TestCase):
    def test_directional_chunk_mask(self):
        mask = g_attention_mask(3, 5, 2, 2)
        self.assertEqual(mask.shape, (13, 13))
        self.assertTrue(mask[:3, :3].all())
        self.assertFalse(mask[:3, 3:].any())
        self.assertTrue(mask[3:, :3].all())
        self.assertTrue(mask[3:7, 3:7].all())
        self.assertFalse(mask[3:7, 7:].any())
        self.assertTrue(mask[7:11, :11].all())
        self.assertFalse(mask[7:11, 11:].any())
        self.assertTrue(mask[11:].all())
        self.assertTrue(torch.equal(g_attention_mask(0, 5, 2, 2), mask[3:, 3:]))

    def test_physical_truncation_before_native_call(self):
        frames = torch.randn(1, 4, 5, 1, 2)
        frames[:, :, 3:] = float("nan")
        seen = []

        def record(native, demonstration, history, config, **kwargs):
            seen.append(history)
            return {0: history}, ()

        with patch("evo_wam.g_pi_context._context", side_effect=record):
            g_context_features(None, torch.zeros_like(frames), frames, CONFIG, current_index=2)
            pi_context_features(None, frames, CONFIG, current_index=2)
        self.assertTrue(all(value.shape[2] == 3 and torch.isfinite(value).all() for value in seen))
        with self.assertRaisesRegex(ValueError, "current_index"):
            truncate_robot_history(frames, 5)
        self.assertNotIn("demonstration", inspect.signature(pi_context_features).parameters)
        with self.assertRaises(TypeError):
            pi_context_features(None, frames, CONFIG, demonstration=frames)

    def test_frozen_assertion_and_full_checksum(self):
        module = torch.nn.Linear(3, 2)
        with self.assertRaisesRegex(ValueError, "every parameter frozen"):
            assert_frozen_base(module)
        module.requires_grad_(False)
        assert_frozen_base(module)
        before = frozen_base_checksum(module)
        module.weight.add_(1)
        self.assertNotEqual(before, frozen_base_checksum(module))

    def test_target_cache_identity_and_roundtrip(self):
        identity = {"layer": 1, "timestep": 0, "pooling": "adaptive_avg_pool1d_spatial",
                    "k_z": 8, "d_z": 4, "normalization": "l2_last_dim",
                    "base_id": {"kind": "fixture", "patch_size": (1, 1, 1)},
                    "empty_text_identity": {"source": {"shape": (1, 4, 8)}}}
        z = FrozenGoalEncoder.normalize(torch.randn(2, 8, 4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "goal.npz"
            save_target_cache(path, z, identity)
            torch.testing.assert_close(load_target_cache(path, identity), z, atol=0, rtol=0)
            for field, value in (("layer", 0), ("base_id", "other"), ("k_z", 2),
                                 ("normalization", "none"), ("timestep", 1)):
                with self.assertRaisesRegex(ValueError, "E identity mismatch"):
                    load_target_cache(path, dict(identity, **{field: value}))


class GPiNativeContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            load_native_class()
        except NativeDependencyError as exc:
            raise unittest.SkipTest(str(exc))

    def test_snapshot_independent_frozen_and_identity(self):
        native = context_model()
        base_id = {"kind": "fixture", "version": 1, "patch_size": (1, 1, 1)}
        encoder = FrozenGoalEncoder(native, layer=1, k_z=8, base_id=base_id)
        base_id["version"] = 2
        self.assertEqual(encoder.identity["base_id"],
                         {"kind": "fixture", "version": 1, "patch_size": [1, 1, 1]})
        encoder.train(True)
        self.assertFalse(any(module.training for module in encoder.modules()))
        assert_frozen_base(encoder.native)
        for source, snapshot in zip(native.parameters(), encoder.native.parameters()):
            self.assertNotEqual(source.data_ptr(), snapshot.data_ptr())
        before = frozen_base_checksum(encoder.native)
        with torch.no_grad():
            next(native.parameters()).add_(2)
        self.assertEqual(before, frozen_base_checksum(encoder.native))
        encoder.validate_identity(encoder.identity)
        with self.assertRaisesRegex(ValueError, "E identity mismatch"):
            encoder.validate_identity(dict(encoder.identity, layer=0))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            encoder(torch.randn(1, 4, 2, 1, 2))

    def test_pretrained_empty_text_required_and_hashed(self):
        native = tiny_model()
        with self.assertRaisesRegex(ValueError, "empty text embedding"):
            FrozenGoalEncoder(native, layer=1)
        empty = torch.randn(1, 4, 8)
        install_empty_text(native, empty, {"kind": "fixture_a", "shape": (1, 4, 8)})
        first = FrozenGoalEncoder(native, layer=1)
        self.assertEqual(first.identity["empty_text_identity"]["source"]["shape"], [1, 4, 8])
        self.assertIn("g_pi_empty_text", native.state_dict())
        self.assertFalse(native.g_pi_empty_text.requires_grad)
        install_empty_text(native, empty + 1, "fixture_b")
        second = FrozenGoalEncoder(native, layer=1)
        self.assertNotEqual(first.identity, second.identity)
        with self.assertRaisesRegex(ValueError, "E identity mismatch"):
            first.validate_identity(second.identity)

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_native_directionality_and_real_demo_cache(self):
        torch.manual_seed(27)
        native = context_model("cuda").requires_grad_(False)
        demonstration = torch.randn(1, 4, 3, 1, 2, device="cuda", dtype=torch.bfloat16)
        history = torch.randn_like(demonstration)
        baseline = g_context_features(native, demonstration, history, CONFIG)
        changed = g_context_features(native, demonstration, history + 5, CONFIG)
        other_demo = g_context_features(native, demonstration - 5, history, CONFIG)
        for index in range(2):
            torch.testing.assert_close(baseline[index][:, :6], changed[index][:, :6], rtol=0, atol=0)
            self.assertGreater((baseline[index][:, 6:] - other_demo[index][:, 6:]).abs().max().item(), 0)
        cache = build_demo_cache(native, demonstration, CONFIG)
        self.assertEqual(len(cache.keys_values), 2)
        self.assertTrue(all(key.shape[1] == 6 for key, _ in cache.keys_values))
        projected = []
        handle = native.blocks[0].attn1.to_k.register_forward_pre_hook(
            lambda module, args: projected.append(args[0].shape[1]))
        try:
            cached = g_context_features(native, demonstration, history, CONFIG, demo_cache=cache)
        finally:
            handle.remove()
        self.assertEqual(projected, [6])
        for index in range(2):
            torch.testing.assert_close(baseline[index], cached[index], rtol=0.015, atol=0.015)
        with self.assertRaisesRegex(ValueError, "demo cache"):
            g_context_features(native, demonstration + 1, history, CONFIG, demo_cache=cache)

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_native_fp32_weights_use_fixed_internal_autocast(self):
        native = tiny_model("cuda").float().requires_grad_(False)
        install_empty_text(native, torch.randn(1, 4, 8), "fixture_fp32_empty")
        encoder = FrozenGoalEncoder(native, layer=1)
        frame = torch.randn(1, 4, 1, 1, 2, device="cuda")
        standalone = encoder(frame)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nested = encoder(frame)
        torch.testing.assert_close(standalone, nested, rtol=0, atol=0)
        cache = build_demo_cache(native, frame, CONFIG)
        standalone = g_context_features(native, frame, frame, CONFIG, demo_cache=cache)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nested = g_context_features(native, frame, frame, CONFIG, demo_cache=cache)
        for index in range(2):
            torch.testing.assert_close(standalone[index], nested[index], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_native_future_invariance_and_snapshot_output(self):
        torch.manual_seed(28)
        native = context_model("cuda").requires_grad_(False)
        encoder = FrozenGoalEncoder(native, layer=1)
        video = torch.randn(1, 4, 4, 1, 2, device="cuda", dtype=torch.bfloat16)
        before_hash = frozen_base_checksum(native)
        before_z = encoder(video[:, :, :1])
        self.assertEqual(before_z.shape, (1, 8, 36))
        torch.testing.assert_close(before_z.norm(dim=-1), torch.ones(1, 8, device="cuda"))
        demo = video[:, :, :2].clone()
        for read in (lambda value: g_context_features(native, demo, value, CONFIG, current_index=1),
                     lambda value: pi_context_features(native, value, CONFIG, current_index=1)):
            baseline = read(video)
            changed = video.clone()
            changed[:, :, 2:] = float("nan")
            for index, value in read(changed).items():
                torch.testing.assert_close(value, baseline[index], rtol=0, atol=0)
        self.assertEqual(before_hash, frozen_base_checksum(native))
        torch.testing.assert_close(before_z, encoder(video[:, :, :1]), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
