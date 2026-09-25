from contextlib import ExitStack
import unittest
from unittest.mock import patch

import torch

from etude.goal_action import (action_named_parameters, goal_action_forward,
                                 goal_action_sample, install_action_interface)
from etude.zerowam import NativeDependencyError, load_native_class
from test_native_icl import tiny_model


class GoalActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            load_native_class()
        except NativeDependencyError as exc:
            raise unittest.SkipTest(str(exc))

    def test_independent_projections_route_restore_and_parameter_selection(self):
        model = tiny_model().float()
        keys = set(model.state_dict())
        attention = model.blocks[0].attn2
        video, action, pad, condition = (torch.randn(1, n, 36) for n in (2, 3, 1, 4))
        original = attention._project(video, None, pad, condition)
        install_action_interface(model)
        for before, after in zip(original[:3], attention._project(video, None, pad, condition)[:3]):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertEqual(set(model.state_dict()), keys)
        names = dict(action_named_parameters(model))
        for name in ("action_embedder.weight", "condition_embedder_action.time_proj.weight",
                     "condition_embedder_action.text_embedder.linear_1.weight",
                     "blocks.0.attn1.action_to_q.weight", "blocks.0.attn2.action_to_k.weight",
                     "blocks.0.attn2.action_to_v.weight", "blocks.0.attn2.action_norm_k.weight",
                     "blocks.0.action_ffn.net.0.proj.weight", "blocks.0.action_norm2.weight",
                     "blocks.0.scale_shift_table_action", "action_proj_out.weight"):
            self.assertIn(name, names)
        self.assertFalse(any(name.startswith("mcp_") for name in names))
        for block in model.blocks:
            for action_name, video_name in (("action_to_k", "to_k"), ("action_to_v", "to_v"),
                                            ("action_norm_k", "norm_k")):
                self.assertIsNot(getattr(block.attn2, action_name), getattr(block.attn2, video_name))
                torch.testing.assert_close(getattr(block.attn2, action_name).weight,
                                           getattr(block.attn2, video_name).weight)
        expected = attention._project(None, action, pad, condition)
        with torch.no_grad():
            attention.to_k.weight.add_(10)
            attention.to_v.weight.mul_(0)
            attention.norm_k.weight.mul_(0)
        for before, after in zip(expected[:3], attention._project(None, action, pad, condition)[:3]):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        with torch.no_grad():
            attention.action_to_k.weight.neg_()
        changed = attention._project(None, action, pad, condition)
        self.assertGreater((expected[1] - changed[1]).abs().max().item(), 0)
        restored = install_action_interface(tiny_model().float())
        restored.load_state_dict(model.state_dict(), strict=True)
        for before, after in zip(changed[:3], restored.blocks[0].attn2._project(None, action, pad, condition)[:3]):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "already installed"):
            install_action_interface(model)
        with self.assertRaisesRegex(ValueError, "isolated"):
            attention._project(video, action, pad, condition)

    def test_input_validation(self):
        model = install_action_interface(tiny_model().float())
        action = torch.zeros(1, 3, 2, 2, 1)
        condition = [torch.zeros(1, 3, 36), torch.zeros(1, 5, 36)]
        for bad in (action[..., :0], action[..., 0], action.long(), action + float("nan")):
            with self.assertRaisesRegex(ValueError, "noisy_actions"):
                goal_action_forward(model, bad, torch.ones(2), condition)
        for bad in (condition[0], condition[:1], [condition[0][:, :0], condition[1]],
                    [condition[0][..., :4], condition[1]], [condition[0].long(), condition[1]],
                    [condition[0] + float("inf"), condition[1]]):
            with self.assertRaisesRegex(ValueError, "condition"):
                goal_action_forward(model, action, torch.ones(2), bad)
        for bad in (1., [1.], [-1., 1.], [1001., 1.], [float("nan"), 1.]):
            with self.assertRaisesRegex(ValueError, "timesteps"):
                goal_action_forward(model, action, bad, condition)
        with self.assertRaisesRegex(ValueError, "cache_name"):
            goal_action_forward(model, action, torch.ones(2), condition, cache_name="pos")
        generator = torch.Generator().manual_seed(12)
        for bad in (torch.ones(3) * -1, torch.zeros(1, 3, 1, 1, 1)):
            with self.assertRaises(ValueError):
                goal_action_sample(model, condition, action.shape, bad, generator)
        with self.assertRaisesRegex(ValueError, "steps"):
            goal_action_sample(model, condition, action.shape, None, generator, steps=0)

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_full_action_updates_layer_conditions_and_visual_cache_isolation(self):
        from wan_va.utils import get_mesh_id

        torch.manual_seed(73)
        model = tiny_model("cuda").float()
        weight = model.action_embedder.weight
        video = torch.randn(1, 4, 1, 1, 2).to(weight)
        video_input = {"latent_res_lst": {"noisy_latents": video,
                          "timesteps": torch.zeros(1, device="cuda")},
                       "latent_grid_id": get_mesh_id(1, 1, 2, 0).to("cuda"),
                       "text_emb": torch.zeros(1, 2, 8).to(weight)}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            original_video = model(video_input, update_cache=1)
        model.clear_cache()
        install_action_interface(model).requires_grad_(False)
        action_params = dict(action_named_parameters(model))
        for parameter in action_params.values():
            parameter.requires_grad_(True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            after_video = model(video_input, update_cache=1)
        torch.testing.assert_close(original_video, after_video, rtol=0, atol=0)
        metadata = {name: getattr(model, name) for name in
                    ("seq_ids_cache", "type_ids_cache", "frame_ids_cache", "cache_type_ids_cache")}
        masks = [(block.attn1.self_block_mask, block.attn2.cross_block_mask) for block in model.blocks]
        caches = [block.attn1.attn_caches["pos"] for block in model.blocks]
        actions = torch.randn(1, 3, 2, 2, 1).to(weight)
        latent = [torch.randn(1, n, 36, device="cuda", requires_grad=True) for n in (2, 4)]
        state = torch.randn(1, 1, 36, device="cuda", requires_grad=True)
        text = torch.randn(1, 2, 8).to(weight)
        time = torch.full((1, 2), 500., device="cuda")
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            language = model.condition_embedder_action.text_embedder(text)
            conditions = [torch.cat((z, state, language), dim=1) for z in latent]
            expected = goal_action_forward(model, actions, time, conditions).detach()
        seen = []
        with ExitStack() as guards:
            for module in (model.patch_embedding_mlp, model.patch_embedding, model.proj_out,
                           model.condition_embedder, model.condition_embedder.text_embedder):
                guards.enter_context(patch.object(module, "forward", side_effect=AssertionError("visual bypass")))
            for block in model.blocks:
                for module in (block.attn1.to_q, block.attn1.to_k, block.attn1.to_v,
                               block.attn1.to_out[0], block.attn2.to_q, block.attn2.to_k,
                               block.attn2.to_v, block.attn2.norm_k, block.attn2.to_out[0], block.ffn):
                    guards.enter_context(patch.object(module, "forward", side_effect=AssertionError("video projection")))
                handle = block.register_forward_pre_hook(lambda module, args: seen.append(args[3]))
                guards.callback(handle.remove)
                block.attn1.attn_caches["se3_action"] = block.attn1.attn_caches["pos"]
            for cache in caches:
                cache["k"].fill_(200)
                cache["v"].fill_(-200)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = goal_action_forward(model, actions, time, conditions)
                torch.testing.assert_close(predicted, expected, rtol=0, atol=0)
                loss = predicted.float().square().mean()
            loss.backward()
            self.assertEqual([value.shape[1] for value in seen], [5, 7])
            for z in latent:
                self.assertGreater(z.grad.abs().sum().item(), 0)
            self.assertGreater(state.grad.abs().sum().item(), 0)
            for name, parameter in action_params.items():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertEqual(parameter.dtype, torch.float32)
                self.assertEqual(parameter.grad.dtype, torch.float32)
            optimizer = torch.optim.AdamW(action_params.values(), lr=0.01, weight_decay=0.)
            optimizer.step()
            for name, parameter in model.named_parameters():
                if name not in action_params:
                    self.assertIsNone(parameter.grad, name)
                    torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            for name in ("action_embedder.weight", "condition_embedder_action.time_proj.weight",
                         "condition_embedder_action.text_embedder.linear_1.weight",
                         "blocks.0.attn1.action_to_q.weight", "blocks.0.attn2.action_to_k.weight",
                         "blocks.0.action_ffn.net.0.proj.weight", "action_proj_out.weight"):
                self.assertFalse(torch.equal(action_params[name], before[name]), name)
            for values in optimizer.state.values():
                self.assertEqual(values["exp_avg"].dtype, torch.float32)
                self.assertEqual(values["exp_avg_sq"].dtype, torch.float32)
        for name, value in metadata.items():
            self.assertIs(getattr(model, name), value)
        for block, (self_mask, cross_mask), cache in zip(model.blocks, masks, caches):
            self.assertIs(block.attn1.attn_caches["pos"], cache)
            self.assertNotIn("se3_action", block.attn1.attn_caches)
            self.assertIs(block.attn1.self_block_mask, self_mask)
            self.assertIs(block.attn2.cross_block_mask, cross_mask)
        with patch.object(model.blocks[0], "forward", side_effect=RuntimeError("injected")):
            with torch.autocast("cuda", dtype=torch.bfloat16), self.assertRaisesRegex(RuntimeError, "injected"):
                goal_action_forward(model, actions, time, conditions)
        self.assertIs(model.blocks[0].attn1.self_block_mask, masks[0][0])
        self.assertIs(model.seq_ids_cache, metadata["seq_ids_cache"])

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_native_sampling_uses_every_layer_and_masks_every_step(self):
        torch.manual_seed(18)
        model = install_action_interface(tiny_model("cuda").float()).requires_grad_(False)
        conditions = [torch.randn(1, n, 36, device="cuda") for n in (3, 5)]
        shape, mask = (1, 3, 2, 2, 1), torch.tensor([1, 0, 1]).reshape(1, 3, 1, 1, 1)
        seen = []
        hook = model.action_embedder.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sample = goal_action_sample(model, conditions, shape, mask, torch.Generator().manual_seed(9), steps=2)
                again = goal_action_sample(model, conditions, shape, mask, torch.Generator().manual_seed(9), steps=2)
                changed = []
                for layer in range(len(conditions)):
                    perturb = list(conditions)
                    perturb[layer] = -conditions[layer] * 2
                    changed.append(goal_action_sample(model, perturb, shape, mask,
                                                      torch.Generator().manual_seed(9), steps=2))
        finally:
            hook.remove()
        torch.testing.assert_close(sample, again, rtol=0, atol=0)
        for value in changed:
            self.assertGreater((sample.float() - value.float()).abs().max().item(), 1e-3)
        self.assertEqual(len(seen), 8)
        self.assertTrue(all(not value[..., 1].any() for value in seen))
        self.assertFalse(sample[:, 1].any())
        self.assertFalse(sample.requires_grad)
        self.assertTrue(torch.isfinite(sample).all())


if __name__ == "__main__":
    unittest.main()
