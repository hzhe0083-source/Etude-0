from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from evo_wam.demo_context import (PreparedDemoContext, cache_demo_context,
                                   install_demo_interface, prepare_demo_context)
from evo_wam.native_icl import forward_video_only, install_icl_lora
from evo_wam.zerowam import NativeDependencyError, load_native_class
from test_native_icl import inputs, tiny_model


CONFIG = {"enabled": True, "dim": 12, "num_heads": 3, "group_frames": 2,
          "tokens_per_group": 1, "layers": 1}


def setup(device="cpu"):
    native = tiny_model(device)
    install_icl_lora(native, rank=2, alpha=2)
    install_demo_interface(native, CONFIG)
    payload, action = inputs(native)
    payload["icl_latent_dict"]["frame_times"] = torch.tensor([3., 3.1, 3.4], device=device, dtype=torch.float64)
    return native, payload, action


class DemoContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            load_native_class()
        except NativeDependencyError as exc:
            raise unittest.SkipTest(str(exc))

    def test_typed_context_compresses_once_and_uses_actual_patch_geometry(self):
        fresh = tiny_model()
        fresh.blocks[0].attn1.attn_caches["old_episode"] = {"k": torch.zeros(1), "v": torch.zeros(1)}
        fresh.cache_type_ids_cache = torch.tensor([2])
        install_demo_interface(fresh, CONFIG)
        self.assertIsNone(fresh.cache_type_ids_cache)
        self.assertFalse(any(block.attn1.attn_caches for block in fresh.blocks))
        native, payload, _ = setup()
        raw = payload["icl_latent_dict"]
        with patch.object(native, "_demo_original_training_embed", wraps=native._demo_original_training_embed) as embed, \
             patch.object(native.demo_bottleneck, "forward", wraps=native.demo_bottleneck.forward) as compress:
            prepared = prepare_demo_context(native, raw)
            context = prepared["latent"]
            self.assertIsInstance(context, PreparedDemoContext)
            self.assertEqual(context.hidden.shape, (1, 2, native.inner_dim))
            self.assertEqual(context.tokens.shape, (1, 2, 12))
            self.assertIs(prepare_demo_context(native, prepared), prepared)
            native._training_embed(context, prepared["timesteps"], "video")
            self.assertEqual(embed.call_count, 1)
            self.assertEqual(compress.call_count, 1)
            features, times, xy = compress.call_args.args
            self.assertEqual(features.shape, (1, 3, 2, native.inner_dim))
            torch.testing.assert_close(times, raw["frame_times"])
            torch.testing.assert_close(xy, torch.tensor([[-.5, 0.], [.5, 0.]]))
        self.assertEqual(set(prepared), {"latent", "grid_id", "timesteps", "frame_times"})
        torch.testing.assert_close(context.grid_id, torch.tensor([[[0, 1], [4, 4], [0, 0], [0, 0]]]))
        self.assertEqual(context.timestep_proj.shape, (1, 2, 6, native.inner_dim))
        self.assertIsInstance(raw["latent"], torch.Tensor)
        target = payload["latent_dict"]
        expected = native._demo_original_training_embed(target["latent"], target["timesteps"], "video")
        actual = native._training_embed(target["latent"], target["timesteps"], "video")
        for first, second in zip(expected, actual):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_missing_or_mismatched_demo_metadata_fails_before_embedding(self):
        native, payload, _ = setup()
        raw = payload["icl_latent_dict"]
        for change in ({"frame_times": None}, {"frame_times": torch.tensor([0., 0., 1.])},
                       {"grid_id": raw["grid_id"][:, :, :-1]},
                       {"timesteps": torch.ones_like(raw["timesteps"])}):
            with patch.object(native, "_demo_original_training_embed", side_effect=AssertionError("raw bypass")):
                with self.assertRaises(ValueError):
                    prepare_demo_context(native, {**raw, **change})
        prepared = prepare_demo_context(native, raw)
        for change in ({"grid_id": None}, {"timesteps": None},
                       {"latent": replace(prepared["latent"], owner=object())}):
            with self.assertRaises(ValueError):
                prepare_demo_context(native, {**prepared, **change})
        with self.assertRaisesRegex(ValueError, "action"):
            native._training_embed(prepared["latent"], prepared["timesteps"], "action")
        raw_grid = raw["grid_id"][0]
        stream = {"noisy_latents": raw["latent"], "timesteps": raw["timesteps"],
                  "cache_type_ids": torch.full((6,), 2)}
        with self.assertRaisesRegex(ValueError, "frame_times"):
            native({"latent_res_lst": stream, "latent_grid_id": raw_grid})
        with self.assertRaisesRegex(ValueError, "mixed"):
            native({"latent_res_lst": {**stream, "cache_type_ids": torch.tensor([2, 2, 2, 0, 0, 0])}})
        with self.assertRaisesRegex(ValueError, "never actions"):
            native({"action_res_lst": stream}, mode="forward_action_only")
        with self.assertRaisesRegex(ValueError, "metadata length"):
            native({"latent_res_lst": {**stream, "cache_type_ids": torch.tensor([2])}})

    def test_trained_namespace_is_default_and_rejects_raw_or_prepared_mismatch(self):
        native, payload, _ = setup()
        native.demo_icl_rope_h = 4
        raw = payload["icl_latent_dict"]
        null = payload["text_emb"][:, :2]
        with patch.object(native, "_demo_original_forward_stream") as forward:
            cache_demo_context(native, raw["latent"], raw["frame_times"], null)
            cached_grid = forward.call_args.args[0]["latent_grid_id"]
            self.assertTrue(torch.all(cached_grid[1] == 4))
        with self.assertRaisesRegex(ValueError, "namespace"):
            cache_demo_context(native, raw["latent"], raw["frame_times"], null, icl_rope_h=24)
        wrong_grid = raw["grid_id"].clone()
        wrong_grid[:, 1] += 1
        with self.assertRaisesRegex(ValueError, "namespace"):
            prepare_demo_context(native, {**raw, "grid_id": wrong_grid})
        prepared = prepare_demo_context(native, raw)
        wrong_grid = prepared["grid_id"].clone()
        wrong_grid[:, 1] += 1
        tampered = replace(prepared["latent"], grid_id=wrong_grid)
        with self.assertRaisesRegex(ValueError, "namespace"):
            prepare_demo_context(native, {**prepared, "latent": tampered, "grid_id": wrong_grid})

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_real_native_human_robot_gradients_disabled_parity_and_cache(self):
        torch.manual_seed(37)
        native, payload, action = setup("cuda")
        for human in (True, False):
            native.zero_grad(set_to_none=True)
            data = payload if human else {**payload, "action_dict": action}
            with patch.object(native.demo_bottleneck, "forward", wraps=native.demo_bottleneck.forward) as compress, \
                 patch.object(native.blocks[0], "forward", wraps=native.blocks[0].forward) as first_block:
                result = forward_video_only(native, data) if human else native(data, train_mode=True)
            self.assertEqual(compress.call_count, 1)
            self.assertEqual(first_block.call_args.args[0].shape[1], 6 * 2 + 2)
            self.assertEqual(result[0].shape, (1, 6, 4))
            if human:
                self.assertIsNone(first_block.call_args.args[1])
                loss = result[0].float().square().mean()
            else:
                self.assertEqual(first_block.call_args.args[1].shape[1], 12)
                self.assertEqual(result[1].shape, (1, 6, 3))
                # The robot action objective alone reaches the video interface.
                loss = result[1].float().square().mean()
            loss.backward()
            for parameter in (native.demo_bottleneck.queries, native.demo_bottleneck.adapter.weight):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.float().abs().sum().item(), 0)
            self.assertTrue(all(value.grad is None for value in native.parameters() if not value.requires_grad))

        baseline = tiny_model("cuda")
        baseline_payload, baseline_action = inputs(baseline)
        baseline_payload["action_dict"] = baseline_action
        with torch.inference_mode():
            expected = baseline(baseline_payload, train_mode=True)
            install_demo_interface(baseline, {"enabled": False})
            self.assertFalse(hasattr(baseline, "demo_bottleneck"))
            actual = baseline(baseline_payload, train_mode=True)
        for left, right in zip([*expected[:2], *expected[2]], [*actual[:2], *actual[2]]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

        native.eval().requires_grad_(False)
        raw = payload["icl_latent_dict"]
        null = payload["text_emb"][:, :2]
        with patch.object(native.demo_bottleneck, "forward", wraps=native.demo_bottleneck.forward) as compress:
            counts = cache_demo_context(native, raw["latent"], raw["frame_times"], null, icl_rope_h=4)
            self.assertEqual(compress.call_count, 1)
        self.assertEqual(counts[2], 2)
        with self.assertRaisesRegex(ValueError, "empty cache"):
            cache_demo_context(native, raw["latent"], raw["frame_times"], null, icl_rope_h=4)
        self.assertEqual(native.cache_counts(), counts)
        active_demo = (native.cache_type_ids_cache == 2) & (native.frame_ids_cache >= 0)
        self.assertTrue(torch.all(native.frame_ids_cache[active_demo] == 0))
        target = payload["latent_dict"]
        with torch.inference_mode():
            observed = native({"latent_res_lst": {"noisy_latents": target["latent"],
                "timesteps": target["cond_timesteps"], "cache_type_ids": torch.zeros(6, device="cuda", dtype=torch.int)},
                "latent_grid_id": target["grid_id"][0], "text_emb": null}, update_cache=1)
            generated = native({"action_res_lst": {"noisy_latents": action["latent"],
                "timesteps": action["timesteps"]}, "action_grid_id": action["grid_id"][0],
                "text_emb": null}, mode="forward_action_only", update_cache=1)
        self.assertEqual(observed.shape, (1, 6, 4))
        self.assertEqual(generated.shape, (1, 6, 3))
        self.assertEqual(native.cache_counts(), {0: 6, 1: 6, 2: 2})
        native.clear_prediction_cache()
        self.assertEqual(native.cache_counts(), {0: 6, 1: 0, 2: 2})


if __name__ == "__main__":
    unittest.main()
