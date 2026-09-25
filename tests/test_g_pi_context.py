import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from etude.g_pi_context import (
    FrozenGoalEncoder, assert_frozen_base, build_demo_cache, frozen_base_checksum,
    g_attention_mask, g_context_features, install_empty_text, load_target_cache, pi_context_features,
    demo_context_features, split_g_context_features,
    save_target_cache, truncate_robot_history, validate_camera_layout,
)
from etude.goal_action import action_named_parameters, install_action_interface
from etude.zerowam import NativeDependencyError, load_native_class
from test_native_icl import tiny_model


CAMERAS = [{"name": "camera", "token_width": 2}]


CONFIG = {"chunk_size": 2, "max_frame_chunk_size": 4, "icl_rope_h": 4, "window_size": 8}


def context_model(device="cpu"):
    native = tiny_model(device).requires_grad_(False)
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

        with patch("etude.g_pi_context._context", side_effect=record):
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
        with self.assertRaisesRegex(ValueError, "every non-action parameter frozen"):
            assert_frozen_base(module)
        module.requires_grad_(False)
        assert_frozen_base(module)
        before = frozen_base_checksum(module)
        module.weight.add_(1)
        self.assertNotEqual(before, frozen_base_checksum(module))

    def test_target_cache_identity_and_roundtrip(self):
        identity = {"layer": 1, "timestep": 0, "pooling": "adaptive_avg_pool2d_spatial",
                    "grid_size": [2, 4], "token_order": "camera_then_row_major",
                    "camera_layout": CAMERAS, "num_views": 1,
                    "k_z": 8, "d_z": 4, "normalization": "l2_last_dim",
                    "base_id": {"kind": "fixture", "patch_size": (1, 1, 1)},
                    "empty_text_identity": {"source": {"shape": (1, 4, 8)}}}
        z = FrozenGoalEncoder.normalize(torch.randn(2, 8, 4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "goal.npz"
            save_target_cache(path, z, identity)
            torch.testing.assert_close(load_target_cache(path, identity), z, atol=0, rtol=0)
            for field, value in (("layer", 0), ("base_id", "other"), ("k_z", 2),
                                 ("normalization", "none"), ("timestep", 1), ("grid_size", [4, 2]),
                                 ("pooling", "adaptive_avg_pool1d_spatial")):
                with self.assertRaisesRegex(ValueError, "E identity mismatch"):
                    load_target_cache(path, dict(identity, **{field: value}))


class GPiNativeContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            load_native_class()
        except NativeDependencyError as exc:
            raise unittest.SkipTest(str(exc))

    def test_two_dimensional_pooling_keeps_both_axes_in_row_order(self):
        native = context_model()
        encoder = FrozenGoalEncoder(native, camera_layout=[{"name": "camera", "token_width": 4}],
                                    layer=1, grid_size=(2, 2))
        spatial = torch.zeros(1, native.inner_dim, 4, 4)
        spatial[:, 0] = 1
        spatial[:, 1, :, :2], spatial[:, 1, :, 2:] = -1, 1
        spatial[:, 2, :2, :], spatial[:, 2, 2:, :] = -2, 2
        features = spatial.flatten(2).transpose(1, 2)
        with patch("etude.g_pi_context.pi_context_features", return_value={1: features}):
            z = encoder(torch.zeros(1, 4, 1, 4, 4))
        expected = torch.zeros(1, 4, native.inner_dim)
        expected[0, :, :3] = torch.tensor([[1., -1., -2.], [1., 1., -2.],
                                          [1., -1., 2.], [1., 1., 2.]])
        torch.testing.assert_close(z, FrozenGoalEncoder.normalize(expected), atol=0, rtol=0)
        self.assertFalse(torch.equal(z[:, 0], z[:, 1]))
        self.assertFalse(torch.equal(z[:, 0], z[:, 2]))
        self.assertEqual(encoder.identity["grid_size"], [2, 2])
        self.assertEqual(encoder.identity["token_order"], "camera_then_row_major")
        self.assertEqual(encoder.identity["pooling"], "adaptive_avg_pool2d_spatial")
        self.assertEqual(FrozenGoalEncoder(native, camera_layout=CAMERAS, layer=1).k_z, 16)
        for grid in ((0, 2), (2,), (True, 2), "4x4"):
            with self.assertRaisesRegex(ValueError, "grid_size"):
                FrozenGoalEncoder(native, camera_layout=CAMERAS, layer=1, grid_size=grid)
        with self.assertRaises(TypeError):
            FrozenGoalEncoder(native, camera_layout=CAMERAS, layer=1, k_z=8)

    def test_same_token_count_with_different_grids_rejects_cache(self):
        native = context_model()
        encoder = FrozenGoalEncoder(native, camera_layout=CAMERAS, layer=1, grid_size=(2, 4))
        other = FrozenGoalEncoder(native, camera_layout=CAMERAS, layer=1, grid_size=(4, 2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "goal.npz"
            z = FrozenGoalEncoder.normalize(torch.randn(1, 8, native.inner_dim))
            save_target_cache(path, z, encoder.identity)
            with self.assertRaisesRegex(ValueError, "E identity mismatch"):
                load_target_cache(path, other.identity)
            old = dict(encoder.identity, pooling="adaptive_avg_pool1d_spatial")
            old.pop("grid_size")
            old.pop("token_order")
            save_target_cache(path, z, old)
            with self.assertRaisesRegex(ValueError, "E identity mismatch"):
                load_target_cache(path, encoder.identity)

    def test_camera_pooling_never_averages_across_view_boundaries(self):
        native = context_model()
        layout = [{"name": "head", "token_width": 3}, {"name": "wrist", "token_width": 2}]
        encoder = FrozenGoalEncoder(native, layer=1, camera_layout=layout, grid_size=(2, 2))
        spatial = torch.zeros(1, native.inner_dim, 2, 5)
        spatial[:, 0, :, :3] = 1
        spatial[:, 1, :, 3:] = 1
        with patch("etude.g_pi_context.pi_context_features",
                   return_value={1: spatial.flatten(2).transpose(1, 2)}):
            z = encoder(torch.zeros(1, 4, 1, 2, 5))
        expected = torch.zeros(1, 8, native.inner_dim)
        expected[:, :4, 0], expected[:, 4:, 1] = 1, 1
        torch.testing.assert_close(z, expected, atol=0, rtol=0)
        self.assertEqual(encoder.k_z, 8)
        self.assertEqual(encoder.identity["num_views"], 2)
        self.assertEqual(encoder.identity["camera_layout"], layout)
        layout[0]["token_width"] = 100
        self.assertEqual(encoder.identity["camera_layout"][0]["token_width"], 3)
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            encoder(torch.zeros(1, 4, 1, 2, 6))
        config = dict(CONFIG, goal_encoder={"camera_layout": encoder.identity["camera_layout"]})
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            pi_context_features(native, torch.zeros(1, 4, 2, 2, 6), config)
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            g_context_features(native, torch.zeros(1, 4, 1, 2, 5),
                               torch.zeros(1, 4, 2, 2, 6), config)
        other = FrozenGoalEncoder(native, layer=1, grid_size=(2, 2),
                                  camera_layout=list(reversed(encoder.identity["camera_layout"])))
        with self.assertRaisesRegex(ValueError, "E identity mismatch"):
            encoder.validate_identity(other.identity)
        for layout in (None, [], [{"name": "head", "token_width": 0}],
                       [{"name": "head", "token_width": True}],
                       [{"name": "head", "token_width": 2}, {"name": "head", "token_width": 2}],
                       [{"name": "head", "token_width": 2, "unused": True}]):
            with self.assertRaisesRegex(ValueError, "camera_layout"):
                validate_camera_layout(layout)

    def test_shared_base_rules_ownership_and_identity(self):
        native = context_model()
        install_action_interface(native)
        for _, parameter in action_named_parameters(native):
            parameter.data = parameter.data.float()
            parameter.requires_grad_(True)
        native.train()
        modes = [module.training for module in native.modules()]
        trainable = [parameter.requires_grad for parameter in native.parameters()]
        base_id = {"kind": "fixture", "version": 1, "patch_size": (1, 1, 1)}
        encoder = FrozenGoalEncoder(native, camera_layout=CAMERAS, layer=1, grid_size=(2, 4), base_id=base_id)
        base_id["version"] = 2
        self.assertEqual(encoder.identity["base_id"],
                         {"kind": "fixture", "version": 1, "patch_size": [1, 1, 1]})
        encoder.train(True)
        encoder.requires_grad_(False)
        self.assertFalse(any(module.training for module in encoder.modules()))
        self.assertEqual(modes, [module.training for module in native.modules()])
        self.assertEqual(trainable, [parameter.requires_grad for parameter in native.parameters()])
        assert_frozen_base(encoder.native)
        self.assertIs(encoder.native, native)
        self.assertEqual(list(encoder.parameters()), [])
        self.assertEqual(encoder.state_dict(), {})
        before = frozen_base_checksum(encoder.native)
        actions = [parameter for _, parameter in action_named_parameters(native)]
        optimizer = torch.optim.SGD(actions, lr=0.1)
        sum(parameter.square().sum() for parameter in actions).backward()
        optimizer.step()
        self.assertEqual(before, frozen_base_checksum(encoder.native))
        encoder.validate_identity(encoder.identity)
        with self.assertRaisesRegex(ValueError, "E identity mismatch"):
            encoder.validate_identity(dict(encoder.identity, layer=0))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            encoder(torch.randn(1, 4, 2, 1, 2))

    def test_pretrained_empty_text_required_and_hashed(self):
        native = tiny_model().requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "empty text embedding"):
            FrozenGoalEncoder(native, camera_layout=CAMERAS, layer=1)
        empty = torch.randn(1, 4, 8)
        install_empty_text(native, empty, {"kind": "fixture_a", "shape": (1, 4, 8)})
        first = FrozenGoalEncoder(native, camera_layout=CAMERAS, layer=1)
        self.assertEqual(first.identity["empty_text_identity"]["source"]["shape"], [1, 4, 8])
        self.assertIn("g_pi_empty_text", native.state_dict())
        self.assertFalse(native.g_pi_empty_text.requires_grad)
        install_empty_text(native, empty + 1, "fixture_b")
        second = FrozenGoalEncoder(native, camera_layout=CAMERAS, layer=1)
        self.assertNotEqual(first.identity, second.identity)
        with self.assertRaisesRegex(ValueError, "E identity mismatch"):
            first.validate_identity(second.identity)

    def test_action_dtype_and_updates_leave_base_identity_unchanged(self):
        native = context_model()
        install_action_interface(native)
        first = FrozenGoalEncoder(native, camera_layout=CAMERAS, layer=1)
        for _, parameter in action_named_parameters(native):
            parameter.data = parameter.data.float()
            parameter.requires_grad_(True)
            with torch.no_grad():
                parameter.add_(1)
        second = FrozenGoalEncoder(native, camera_layout=CAMERAS, layer=1)
        self.assertEqual(first.identity, second.identity)
        native.blocks[0].attn2.action_to_k.weight = native.blocks[0].attn2.to_k.weight
        with self.assertRaisesRegex(ValueError, "independent action attention"):
            assert_frozen_base(native)

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_split_default_preserves_native_demo_path_and_ablation_removes_it(self):
        torch.manual_seed(315)
        native = context_model("cuda").requires_grad_(False)
        demonstration = torch.randn(1, 4, 2, 1, 2, device="cuda")
        history = torch.randn(1, 4, 3, 1, 2, device="cuda")
        legacy = g_context_features(native, demonstration, history, CONFIG)
        demo, robot = split_g_context_features(native, demonstration, history, CONFIG)
        changed_demo, changed_robot = split_g_context_features(native, demonstration + 7, history, CONFIG)
        for layer in range(2):
            torch.testing.assert_close(demo[layer], legacy[layer][:, :4], rtol=0, atol=0)
            torch.testing.assert_close(robot[layer], legacy[layer][:, 4:], rtol=0, atol=0)
            self.assertFalse(torch.equal(robot[layer], changed_robot[layer]))
            self.assertFalse(torch.equal(demo[layer], changed_demo[layer]))
        isolated = dict(CONFIG, demo_route="via_u_only")
        demo, robot = split_g_context_features(native, demonstration, history, isolated)
        other_demo, other_robot = split_g_context_features(native, demonstration + 7, history, isolated)
        cached = build_demo_cache(native, demonstration, isolated)
        with patch("etude.g_pi_context._context", wraps=__import__(
                "etude.g_pi_context", fromlist=["_context"])._context) as context:
            cached_demo = demo_context_features(native, demonstration, isolated, demo_cache=cached)
            self.assertEqual(context.call_count, 0)
        for layer in range(2):
            torch.testing.assert_close(robot[layer], other_robot[layer], rtol=0, atol=0)
            torch.testing.assert_close(demo[layer], cached_demo[layer], rtol=0, atol=0)
            self.assertFalse(torch.equal(demo[layer], other_demo[layer]))
        with self.assertRaisesRegex(ValueError, "demo cache"):
            demo_context_features(native, demonstration + 1, isolated, demo_cache=cached)
        with self.assertRaisesRegex(ValueError, "demo_route"):
            split_g_context_features(native, demonstration, history, dict(CONFIG, demo_route="unknown"))

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
        encoder = FrozenGoalEncoder(native, camera_layout=CAMERAS, layer=1)
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
    def test_native_future_invariance_and_shared_encoder_output(self):
        torch.manual_seed(28)
        native = context_model("cuda").requires_grad_(False)
        install_action_interface(native)
        actions = [parameter for _, parameter in action_named_parameters(native)]
        for parameter in actions:
            parameter.data = parameter.data.float()
            parameter.requires_grad_(True)
        encoder = FrozenGoalEncoder(native, camera_layout=CAMERAS, layer=1)
        video = torch.randn(1, 4, 4, 1, 2, device="cuda", dtype=torch.bfloat16)
        before_hash = frozen_base_checksum(native)
        before_z = encoder(video[:, :, :1])
        self.assertEqual(before_z.shape, (1, 16, 36))
        torch.testing.assert_close(before_z.norm(dim=-1), torch.ones(1, 16, device="cuda"))
        demo = video[:, :, :2].clone()
        for read in (lambda value: g_context_features(native, demo, value, CONFIG, current_index=1),
                     lambda value: pi_context_features(native, value, CONFIG, current_index=1)):
            baseline = read(video)
            changed = video.clone()
            changed[:, :, 2:] = float("nan")
            for index, value in read(changed).items():
                torch.testing.assert_close(value, baseline[index], rtol=0, atol=0)
        self.assertEqual(before_hash, frozen_base_checksum(native))
        cache = build_demo_cache(native, demo, CONFIG)
        cached = g_context_features(native, demo, video[:, :, :2], CONFIG, demo_cache=cache)
        optimizer = torch.optim.SGD(actions, lr=0.1)
        sum(parameter.square().sum() for parameter in actions).backward()
        optimizer.step()
        self.assertEqual(before_hash, frozen_base_checksum(native))
        torch.testing.assert_close(before_z, encoder(video[:, :, :1]), rtol=0, atol=0)
        after = g_context_features(native, demo, video[:, :, :2], CONFIG, demo_cache=cache)
        for index in range(2):
            torch.testing.assert_close(cached[index], after[index], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
