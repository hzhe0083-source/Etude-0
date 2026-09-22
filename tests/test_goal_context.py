"""Observed-only Wan context: packing, visibility, gradients and state isolation."""

from contextlib import ExitStack
import inspect
import unittest
from unittest.mock import patch

import torch

from evo_wam.goal_context import observed_context_features
from evo_wam.zerowam import NativeDependencyError, load_native_class
from test_native_icl import tiny_model


class GoalContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            load_native_class()
        except NativeDependencyError as exc:
            raise unittest.SkipTest(str(exc))

    def inputs(self, device="cpu"):
        torch.manual_seed(71)
        native = tiny_model(device).float().train()
        native.blocks[1].eval()
        weight = native.patch_embedding_mlp.weight
        # Different spatial grids, an entire three-frame demo, and history that
        # does not end on a chunk boundary are all valid observed context.
        demonstration = torch.randn(1, 4, 3, 2, 1).to(weight)
        history = torch.randn(1, 4, 2, 1, 2).to(weight)
        language = torch.randn(1, 2, 8).to(weight)
        config = dict(chunk_size=3, max_frame_chunk_size=4, icl_rope_h=4, window_size=8)
        return native, demonstration, history, language, config

    def read(self, values, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=values[0].device.type == "cuda"):
            return observed_context_features(*values, **kwargs)

    def seed_state(self, native):
        from wan_va.modules.icl_model import ICLAttentionBackend
        masks = []
        for module in native.modules():
            if not hasattr(module, "attn_caches"):
                continue
            module.attn_caches["stale"] = {"k": None, "v": None}
            module.attn_caches["observed_context"] = {"k": None, "v": None}
            module.self_block_mask, module.cross_block_mask = object(), object()
            masks.append((module, module.self_block_mask, module.cross_block_mask))
        for name in ("type_ids_cache", "seq_ids_cache", "frame_ids_cache", "cache_type_ids_cache"):
            setattr(native, name, torch.tensor([2]))
        backend = ICLAttentionBackend.self_mask, ICLAttentionBackend.cross_mask
        modes = [(module, module.training) for module in native.modules()]
        return masks, backend, modes

    def assert_clean_state(self, native, state):
        from wan_va.modules.icl_model import ICLAttentionBackend
        masks, backend, modes = state
        self.assertEqual(native.cache_counts(), {})
        for name in ("type_ids_cache", "seq_ids_cache", "frame_ids_cache", "cache_type_ids_cache"):
            self.assertIsNone(getattr(native, name))
        for module, self_mask, cross_mask in masks:
            self.assertFalse(module.attn_caches)
            self.assertIs(module.self_block_mask, self_mask)
            self.assertIs(module.cross_block_mask, cross_mask)
        self.assertIs(ICLAttentionBackend.self_mask, backend[0])
        self.assertIs(ICLAttentionBackend.cross_mask, backend[1])
        self.assertEqual([module.training for module, _ in modes], [mode for _, mode in modes])
        self.assertTrue(all(not block._forward_hooks for block in native.blocks))

    def test_contract_and_validation_precede_native_execution(self):
        self.assertEqual(list(inspect.signature(observed_context_features).parameters),
                         ["native", "demonstration", "history", "language", "config", "demo_times", "on_layer"])
        values = self.inputs()
        native, demo, history, language, config = values
        with patch.object(native, "_training_embed", side_effect=AssertionError("native execution")):
            for change in ({"chunk_size": 0}, {"chunk_size": 5}, {"chunk_size": True},
                           {"max_frame_chunk_size": 0}, {"window_size": 4}, {"window_size": -1},
                           {"icl_rope_h": 0}, {"feature_layers": [1, 0]}, {"feature_layers": [0]},
                           {"feature_layers": [False, True]}, {"feature_layers": [0, 0]}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.read((native, demo, history, language, config | change))
            for bad in (history[:, :, :0], history.expand(2, -1, -1, -1, -1),
                        history[:, :3], history * float("nan")):
                with self.subTest(history_shape=bad.shape), self.assertRaisesRegex(ValueError, "history"):
                    self.read((native, demo, bad, language, config))
            with self.assertRaisesRegex(ValueError, "demonstration"):
                self.read((native, demo.long(), history, language, config))
            for bad in (language[:, :, :1], language[:, :0], language * float("inf")):
                with self.assertRaisesRegex(ValueError, "language"):
                    self.read((native, demo, history, bad, config))
            for bad in (torch.zeros(3), torch.tensor([0., 2., 1.]), torch.arange(3),
                        torch.tensor([0., 1., float("nan")]), torch.arange(2).float()):
                with self.assertRaisesRegex(ValueError, "demo_times"):
                    self.read(values, demo_times=bad)
            with self.assertRaisesRegex(ValueError, "on_layer"):
                self.read(values, on_layer=3)
            native.patch_size = (2, 1, 1)
            with self.assertRaisesRegex(ValueError, "temporal patch"):
                self.read(values)
            native.patch_size = (1, 2, 1)
            with self.assertRaisesRegex(ValueError, "history"):
                self.read(values)
            native.patch_size = (1, 1, 1)
            with self.assertRaisesRegex(ValueError, "overlaps"):
                self.read((native, demo, history.expand(-1, -1, -1, 5, -1), language, config))
            native.demo_bottleneck = torch.nn.Identity()
            with self.assertRaisesRegex(ValueError, "without demo_bottleneck"):
                self.read(values)

    def test_history_first_independent_grids_and_explicit_full_prefix_masks(self):
        values = self.inputs()
        native, demo, history, language, _ = values
        expected_history = native._training_embed(history, torch.zeros(1, 2), "video")[0]
        expected_demo = native._training_embed(demo, torch.zeros(1, 3), "video")[0]
        state = self.seed_state(native)
        seen = []

        def observe(hidden, action, pad, text, projection, action_projection, rotary, **kwargs):
            index = len(seen)
            block = native.blocks[index]
            torch.testing.assert_close(hidden, torch.cat([expected_history, expected_demo], dim=1))
            self.assertIsNone(action)
            self.assertIsNone(action_projection)
            self.assertEqual(pad.shape, (1, 118, native.inner_dim))
            self.assertEqual(projection.shape, (1, 10, 6, native.inner_dim))
            self.assertEqual(rotary.shape[1], 128)
            self.assertEqual(kwargs, {"update_cache": 0, "cache_name": "observed_context"})
            self.assertTrue(all(not module.attn_caches for module, _, _ in state[0]))
            # Both streams see every real query/key regardless of storage order,
            # with neither padding queries nor padding keys visible.
            query = torch.arange(128)[:, None]
            key = torch.arange(128)[None, :]
            actual = block.attn1.self_block_mask.mask_mod(0, 0, query, key)
            expected = (query < 10) & (key < 10)
            torch.testing.assert_close(actual, expected)
            actual_cross = block.attn2.cross_block_mask.mask_mod(0, 0, query, key)
            torch.testing.assert_close(actual_cross, (query < 10) & (key < language.shape[1]))
            seen.append(index)
            return hidden, None

        with ExitStack() as stack:
            for block in native.blocks:
                stack.enter_context(patch.object(block, "forward", side_effect=observe))
            for module in (native, native.proj_out, native.action_embedder, native.action_proj_out):
                stack.enter_context(patch.object(module, "forward", side_effect=AssertionError("forbidden forward")))
            stack.enter_context(patch("wan_va.utils.FlowMatchScheduler", side_effect=AssertionError("video sampler")))
            embed = stack.enter_context(patch.object(native, "_training_embed", wraps=native._training_embed))
            rope = stack.enter_context(patch.object(native.rope, "forward", wraps=native.rope.forward))
            features = self.read(values, demo_times=torch.tensor([8., 8.2, 9.]))
        self.assertEqual(seen, [0, 1])
        self.assertEqual(list(features), [0, 1])
        self.assertEqual(embed.call_count, 2)
        for call, video in zip(embed.call_args_list, (history, demo)):
            torch.testing.assert_close(call.args[0], video)
            self.assertTrue(torch.all(call.args[1] == 0))
            self.assertEqual(call.args[2], "video")
        torch.testing.assert_close(rope.call_args.args[0], torch.tensor([[
            [0, 0, 1, 1, 0, 0, 1, 1, 2, 2],
            [0, 0, 0, 0, 4, 5, 4, 5, 4, 5],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]))
        self.assert_clean_state(native, state)

    def test_embedding_block_and_callback_failures_restore_attention_state(self):
        values = self.inputs()
        native = values[0]
        for failure in ("embedding", "block", "callback"):
            with self.subTest(failure=failure):
                state = self.seed_state(native)
                with ExitStack() as stack:
                    if failure == "embedding":
                        stack.enter_context(patch.object(native, "_training_embed", side_effect=RuntimeError(failure)))
                    elif failure == "block":
                        stack.enter_context(patch.object(native.blocks[0], "forward", side_effect=RuntimeError(failure)))
                    else:
                        for block in native.blocks:
                            stack.enter_context(patch.object(block, "forward", side_effect=lambda hidden, *a, **kw: (hidden, None)))
                    def fail_callback(*_):
                        native.train(False)
                        raise RuntimeError("callback")
                    with self.assertRaisesRegex(RuntimeError, failure):
                        self.read(values, on_layer=fail_callback if failure == "callback" else None)
                self.assert_clean_state(native, state)

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_real_native_bidirectional_influence_gradients_and_determinism(self):
        values = self.inputs("cuda")
        native, demo, history, language, config = values
        demo.requires_grad_()
        history.requires_grad_()
        language.requires_grad_()
        state = self.seed_state(native)
        embeddings, text_embeddings = [], []

        def retain(collection):
            def capture(module, args, output):
                output.retain_grad()
                collection.append(output)
            return capture

        with ExitStack() as stack:
            for module in (native, native.proj_out, native.action_embedder, native.action_proj_out):
                stack.enter_context(patch.object(module, "forward", side_effect=AssertionError("forbidden forward")))
            stack.enter_context(patch("wan_va.utils.FlowMatchScheduler", side_effect=AssertionError("video sampler")))
            hooks = [native.patch_embedding_mlp.register_forward_hook(retain(embeddings)),
                     native.condition_embedder.text_embedder.register_forward_hook(retain(text_embeddings))]
            try:
                features = self.read(values)
            finally:
                for hook in hooks:
                    hook.remove()
        self.assertEqual(list(features), [0, 1])
        self.assertTrue(all(feature.shape == (1, 10, native.inner_dim) for feature in features.values()))
        # Supervision on history tokens alone must reach both input streams.
        feature = features[1][:, :4].float()
        (feature[..., :18].square().mean() + feature[..., 18:].abs().mean()).backward()
        for tensor in [*embeddings, *text_embeddings, native.patch_embedding_mlp.weight,
                       native.blocks[0].attn1.to_k.weight, native.blocks[1].attn1.to_v.weight]:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.float().abs().sum().item(), 0)
        self.assertTrue(all(value.grad is None for value in (demo, history, language)))
        self.assert_clean_state(native, state)
        with torch.no_grad():
            baseline = self.read(values)
            again = self.read(values)
            changed_demo = self.read((native, -demo * 3, history, language, config))
            changed_history = self.read((native, demo, history + 3, language, config))
            callbacks = []
            def consume(index, feature):
                callbacks.append(index)
                torch.testing.assert_close(feature, baseline[index], rtol=0, atol=0)
            omitted = self.read(values, on_layer=consume)
        self.assertEqual(omitted, {})
        self.assertEqual(callbacks, [0, 1])
        for index in baseline:
            torch.testing.assert_close(baseline[index], again[index], rtol=0, atol=0)
            torch.testing.assert_close(features[index], baseline[index], rtol=.02, atol=.008)
            self.assertFalse(baseline[index].requires_grad)
        # Test cross-stream effects, not the changed stream's own residual.
        self.assertGreater((baseline[1][:, :4] - changed_demo[1][:, :4]).float().abs().max().item(), 1e-3)
        self.assertGreater((baseline[1][:, 4:] - changed_history[1][:, 4:]).float().abs().max().item(), 1e-3)
        self.assert_clean_state(native, state)


if __name__ == "__main__":
    unittest.main()
