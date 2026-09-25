"""The goal interface sees native generated futures, never a teacher future."""

import inspect
import unittest
from unittest.mock import patch

import torch

from etude.goal_future import generated_robot_features
from etude.zerowam import NativeDependencyError, load_native_class
from test_native_icl import tiny_model


class GoalFutureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            load_native_class()
        except NativeDependencyError as exc:
            raise unittest.SkipTest(str(exc))

    def inputs(self, device="cpu"):
        torch.manual_seed(71)
        native = tiny_model(device).float().train()
        native.blocks[1].eval()  # Mixed per-module modes must survive sampling.
        weight = native.patch_embedding_mlp.weight
        demonstration = torch.randn(1, 4, 2, 1, 2).to(weight)
        history = torch.randn(1, 4, 2, 1, 2).to(weight)
        language = torch.randn(1, 2, 8).to(weight)
        config = dict(chunk_size=1, max_frame_chunk_size=4, icl_rope_h=4,
                      window_size=8, video_snr_shift=3., sampling_steps=1, feature_layers=[0, 1])
        return native, demonstration, history, language, config

    def run_future(self, values, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=values[0].device.type == "cuda"):
            return generated_robot_features(*values, torch.Generator().manual_seed(9), **kwargs)

    def test_history_only_contract_and_invalid_inputs_fail_before_execution(self):
        self.assertEqual(list(inspect.signature(generated_robot_features).parameters),
                         ["native", "demonstration", "history", "language", "config", "generator",
                          "demo_times", "on_layer"])
        values = self.inputs()
        native, demo, history, language, config = values
        with patch.object(native, "forward", side_effect=AssertionError("native execution")):
            for change in ({"feature_layers": []}, {"feature_layers": [0, 0]}, {"feature_layers": [2]},
                           {"sampling_steps": 0}, {"chunk_size": 3}, {"video_snr_shift": float("nan")},
                           {"window_size": 4}, {"icl_rope_h": 0}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.run_future((native, demo, history, language, config | change))
            with self.assertRaisesRegex(ValueError, "history"):
                self.run_future((native, demo, history[:, :, :0], language, config))
            with self.assertRaisesRegex(ValueError, "language"):
                self.run_future((native, demo, history, language[:, :, :1], config))
            with self.assertRaisesRegex(ValueError, "demo_times"):
                self.run_future(values, demo_times=torch.zeros(2))
            with self.assertRaisesRegex(ValueError, "on_layer"):
                self.run_future(values, on_layer=3)
            native.demo_bottleneck = torch.nn.Identity()
            with self.assertRaisesRegex(ValueError, "without demo_bottleneck"):
                self.run_future(values)

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_native_sampling_determinism_causality_and_final_feature_gradients(self):
        values = self.inputs("cuda")
        native, demo, history, language, config = values
        modes = [module.training for module in native.modules()]
        seen = []
        replay_embeddings, replay_keys, replay_text = [], [], []
        stream_type = None
        def observe(module, args, kwargs):
            nonlocal stream_type
            item = args[0]
            self.assertEqual(kwargs["mode"], "forward_latent_only")
            self.assertNotIn("action_res_lst", item)
            stream = item["latent_res_lst"]
            stream_type = int(stream["cache_type_ids"][0])
            self.assertEqual([module.training for module in native.modules()],
                             modes if torch.is_grad_enabled() else [False] * len(modes))
            torch.testing.assert_close(item["text_emb"], language)
            seen.append((int(stream["cache_type_ids"][0]),
                         int(item["current_frame_ids"][0]), int(item["latent_grid_id"][0, 0]),
                         torch.is_grad_enabled(), kwargs["update_cache"]))
        def retain_embedding(module, args, output):
            if torch.is_grad_enabled() and stream_type in (2, 0):
                output.retain_grad()
                replay_embeddings.append((stream_type, output))
        def retain_text(module, args, output):
            if torch.is_grad_enabled() and stream_type in (2, 0):
                output.retain_grad()
                replay_text.append((stream_type, output))
        def retain_cache(module, args, kwargs, output):
            if torch.is_grad_enabled() and stream_type in (2, 0):
                for block in native.blocks:
                    for tensor in block.attn1.attn_caches["pos"].values():
                        tensor.retain_grad()
                        replay_keys.append((stream_type, tensor))
        hooks = [native.register_forward_pre_hook(observe, with_kwargs=True),
                 native.patch_embedding_mlp.register_forward_hook(retain_embedding),
                 native.condition_embedder.text_embedder.register_forward_hook(retain_text),
                 native.register_forward_hook(retain_cache, with_kwargs=True)]
        try:
            with patch.object(native.action_embedder, "forward", side_effect=AssertionError("action embedding")):
                generated, features = self.run_future(values, demo_times=torch.tensor([0., .5]))
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(generated.shape, (1, 4, 1, 1, 2))
        self.assertEqual(list(features), [0, 1])
        self.assertTrue(all(f.shape == (1, 1, 2, 36) for f in features.values()))
        self.assertFalse(generated.requires_grad)
        self.assertTrue(all(f.requires_grad for f in features.values()))
        self.assertEqual(seen, [(2, 0, 0, False, 1), (0, 0, 0, False, 1), (0, 2, 1, False, 1),
                                (1, 4, 2, False, 0), (2, 0, 0, True, 1), (0, 0, 0, True, 1),
                                (0, 2, 1, True, 1), (1, 4, 2, True, 0)])
        # Two downstream heads stand in for action and pose supervision. Their
        # only input is final generated-future features, so intermediate replay
        # gradients prove the context graph survived sampling and cache cleanup.
        feature = features[1].float()
        action_proxy = feature[..., :18].mean(-2).square().mean()
        pose_proxy = feature[..., 18:].mean(-2).abs().mean()
        (action_proxy + pose_proxy).backward()
        for records in (replay_embeddings, replay_keys, replay_text):
            self.assertEqual({kind for kind, _ in records}, {0, 2})
            for kind, tensor in records:
                with self.subTest(replay_kind=kind, shape=tuple(tensor.shape)):
                    self.assertIsNotNone(tensor.grad)
                    self.assertTrue(torch.isfinite(tensor.grad).all())
                    self.assertGreater(tensor.grad.float().abs().sum().item(), 0)
        grads = [p.grad for p in native.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(g.float().abs().sum().item() for g in grads), 0)
        self.assertEqual(native.cache_counts(), {})
        self.assertEqual([module.training for module in native.modules()], modes)
        self.assertTrue(all(p.dtype == torch.float32 for p in native.parameters()))
        self.assertTrue(all(not block._forward_hooks for block in native.blocks))
        with torch.no_grad():
            again, same = self.run_future(values)
            torch.testing.assert_close(generated, again, rtol=0, atol=0)
            for index in features:
                # Native grad/no-grad attention kernels may round bf16 differently.
                torch.testing.assert_close(features[index], same[index], rtol=.02, atol=.008)
                self.assertFalse(same[index].requires_grad)
            _, changed_demo = self.run_future((native, -demo * 3, history, language, config))
            _, changed_history = self.run_future((native, demo, history + 3, language, config))
            callbacks = []
            def consume(index, feature):
                callbacks.append(index)
                torch.testing.assert_close(feature, same[index], rtol=0, atol=0)
            callback_generated, omitted = self.run_future(values, on_layer=consume)
        self.assertEqual(callbacks, [0, 1])
        self.assertIsNone(omitted)
        torch.testing.assert_close(generated, callback_generated, rtol=0, atol=0)
        self.assertGreater((features[1] - changed_demo[1]).float().abs().max().item(), 1e-3)
        self.assertGreater((features[1] - changed_history[1]).float().abs().max().item(), 1e-3)

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_failure_during_final_read_removes_hooks_and_all_cache_names(self):
        values = self.inputs("cuda")
        native = values[0]
        modes = [module.training for module in native.modules()]
        original = native.blocks[1].forward
        def fail(*args, **kwargs):
            if native.blocks[0]._forward_hooks:
                raise RuntimeError("final read failure")
            return original(*args, **kwargs)
        for block in native.blocks:
            block.attn1.attn_caches["stale"] = {"k": None, "v": None}
        with patch.object(native.blocks[1], "forward", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "final read failure"):
                self.run_future(values)
        self.assertEqual(native.cache_counts(), {})
        self.assertTrue(all(not block._forward_hooks and not block.attn1.attn_caches for block in native.blocks))
        self.assertEqual([module.training for module in native.modules()], modes)
        def fail_callback(*_):
            raise RuntimeError("callback failure")
        with self.assertRaisesRegex(RuntimeError, "callback failure"):
            self.run_future(values, on_layer=fail_callback)
        self.assertEqual(native.cache_counts(), {})
        self.assertTrue(all(not block._forward_hooks for block in native.blocks))
        self.assertEqual([module.training for module in native.modules()], modes)
        with patch.object(native, "forward", side_effect=RuntimeError("sampler failure")):
            with self.assertRaisesRegex(RuntimeError, "sampler failure"):
                self.run_future(values)
        self.assertEqual([module.training for module in native.modules()], modes)


if __name__ == "__main__":
    unittest.main()
