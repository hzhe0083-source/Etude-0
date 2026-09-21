"""The goal interface sees native generated futures, never a teacher future."""

import inspect
import unittest
from unittest.mock import patch

import torch

from evo_wam.goal_future import generated_robot_features
from evo_wam.native_icl import install_icl_lora
from evo_wam.zerowam import NativeDependencyError, load_native_class
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
        native = tiny_model(device)
        install_icl_lora(native, rank=2, alpha=2)
        weight = native.patch_embedding_mlp.weight
        demonstration = torch.randn(1, 4, 2, 1, 2).to(weight)
        history = torch.randn(1, 4, 2, 1, 2).to(weight)
        null = torch.zeros(1, 2, 8).to(weight)
        config = dict(chunk_size=1, max_frame_chunk_size=4, icl_rope_h=4,
                      window_size=8, video_snr_shift=3., sampling_steps=1, feature_layers=[0, 1])
        return native, demonstration, history, null, config

    def run_future(self, values, **kwargs):
        return generated_robot_features(*values, torch.Generator().manual_seed(9), **kwargs)

    def test_history_only_contract_and_invalid_inputs_fail_before_execution(self):
        self.assertEqual(list(inspect.signature(generated_robot_features).parameters),
                         ["native", "demonstration", "history", "null", "config", "generator", "demo_times"])
        values = self.inputs()
        native, demo, history, null, config = values
        with patch.object(native, "forward", side_effect=AssertionError("native execution")):
            for change in ({"feature_layers": []}, {"feature_layers": [0, 0]}, {"feature_layers": [2]},
                           {"sampling_steps": 0}, {"chunk_size": 3}, {"video_snr_shift": float("nan")},
                           {"window_size": 4}, {"icl_rope_h": 0}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.run_future((native, demo, history, null, config | change))
            with self.assertRaisesRegex(ValueError, "history"):
                self.run_future((native, demo, history[:, :, :0], null, config))
            with self.assertRaisesRegex(ValueError, "null"):
                self.run_future((native, demo, history, null[:, :, :1], config))
            with self.assertRaisesRegex(ValueError, "demo_times"):
                self.run_future(values, demo_times=torch.zeros(2))
            native.demo_bottleneck = torch.nn.Identity()
            with self.assertRaisesRegex(ValueError, "without demo_bottleneck"):
                self.run_future(values)

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_native_sampling_determinism_causality_and_final_feature_gradients(self):
        values = self.inputs("cuda")
        native, demo, history, null, config = values
        seen = []
        def observe(module, args, kwargs):
            item = args[0]
            self.assertEqual(kwargs["mode"], "forward_latent_only")
            self.assertNotIn("action_res_lst", item)
            stream = item["latent_res_lst"]
            seen.append((int(stream["cache_type_ids"][0]),
                         int(item["current_frame_ids"][0]), int(item["latent_grid_id"][0, 0]),
                         torch.is_grad_enabled(), kwargs["update_cache"]))
        hook = native.register_forward_pre_hook(observe, with_kwargs=True)
        try:
            with patch.object(native.action_embedder, "forward", side_effect=AssertionError("action embedding")):
                generated, features = self.run_future(values, demo_times=torch.tensor([0., .5]))
        finally:
            hook.remove()
        self.assertEqual(generated.shape, (1, 4, 1, 1, 2))
        self.assertEqual(features.shape, (1, 1, 2, 72))
        self.assertFalse(generated.requires_grad)
        self.assertTrue(features.requires_grad)
        self.assertEqual(seen, [(2, 0, 0, False, 1), (0, 0, 0, False, 1), (0, 2, 1, False, 1),
                                (1, 4, 2, False, 0), (2, 0, 0, True, 1), (0, 0, 0, True, 1),
                                (0, 2, 1, True, 1), (1, 4, 2, True, 0)])
        features.float().square().mean().backward()
        grads = [p.grad for p in native.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(g.float().abs().sum().item() for g in grads), 0)
        self.assertEqual(native.cache_counts(), {})
        self.assertTrue(all(not block._forward_hooks for block in native.blocks))
        with torch.no_grad():
            again, same = self.run_future(values)
            torch.testing.assert_close(generated, again, rtol=0, atol=0)
            torch.testing.assert_close(features, same, rtol=0, atol=0)
            self.assertFalse(same.requires_grad)
            _, changed_demo = self.run_future((native, -demo * 3, history, null, config))
            _, changed_history = self.run_future((native, demo, history + 3, null, config))
        self.assertGreater((features - changed_demo).float().abs().max().item(), 1e-3)
        self.assertGreater((features - changed_history).float().abs().max().item(), 1e-3)

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_failure_during_final_read_removes_hooks_and_all_cache_names(self):
        values = self.inputs("cuda")
        native = values[0]
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


if __name__ == "__main__":
    unittest.main()
