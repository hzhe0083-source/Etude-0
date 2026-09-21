from contextlib import ExitStack
import unittest
from unittest.mock import patch

import torch

from evo_wam.goal_action import (goal_action_forward, goal_action_sample, install_action_lora,
                                 merge_action_lora, set_action_lora)
from evo_wam.native_icl import install_icl_lora, merge_icl_lora
from evo_wam.zerowam import NativeDependencyError, VideoLoRA, load_native_class
from test_native_icl import tiny_model


class GoalActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            load_native_class()
        except NativeDependencyError as exc:
            raise unittest.SkipTest(str(exc))

    def test_action_adapters_preserve_video_trainability_and_native_aliases(self):
        model = tiny_model()
        keys = set(model.state_dict())
        install_icl_lora(model, rank=2, alpha=2)
        video = {name for name, value in model.named_parameters() if value.requires_grad}
        install_action_lora(model, rank=2, alpha=2)
        all_trainable = {name for name, value in model.named_parameters() if value.requires_grad}
        action = all_trainable - video
        self.assertEqual(len(action), 24)
        self.assertTrue(video <= all_trainable)
        self.assertTrue(all("action" in name and name.endswith((".up.weight", ".down.weight"))
                            for name in action))
        for block in model.blocks:
            self.assertIs(block.attn2.to_k, block.attn2.action_to_k)
            self.assertIs(block.attn2.to_v, block.attn2.action_to_v)
            self.assertIs(block.attn2.norm_k, block.attn2.action_norm_k)
            self.assertNotIsInstance(block.attn2.to_k, VideoLoRA)
        with self.assertRaisesRegex(ValueError, "unwrapped"):
            install_action_lora(model, rank=2)
        set_action_lora(model, False)
        self.assertEqual({name for name, value in model.named_parameters() if value.requires_grad}, video)
        set_action_lora(model, True)
        self.assertEqual({name for name, value in model.named_parameters() if value.requires_grad}, all_trainable)
        merge_action_lora(model)
        merge_icl_lora(model)
        self.assertEqual(set(model.state_dict()), keys)
        tiny_model().load_state_dict(model.state_dict(), strict=True)

    def test_input_validation(self):
        model = tiny_model()
        action, condition = torch.zeros(1, 3, 2, 2, 1), torch.zeros(1, 3, 36)
        for bad in (action[..., :0], action[..., 0], action.long(), action + float("nan")):
            with self.assertRaisesRegex(ValueError, "noisy_actions"):
                goal_action_forward(model, bad, torch.ones(2), condition)
        for bad in (condition[:, :1], condition[..., :4], condition.long(), condition + float("inf")):
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
    def test_native_action_only_cache_isolation_gradients_and_merge(self):
        from wan_va.utils import get_mesh_id

        torch.manual_seed(73)
        model = tiny_model("cuda")
        keys = set(model.state_dict())
        install_icl_lora(model, rank=2, alpha=2)
        install_action_lora(model, rank=2, alpha=2)
        weight = model.action_embedder.weight
        actions = torch.randn(1, 3, 2, 2, 1).to(weight)
        z = torch.randn(1, 2, 36, device="cuda", dtype=weight.dtype, requires_grad=True)
        state = torch.randn(1, 1, 36, device="cuda", dtype=weight.dtype, requires_grad=True)
        condition = torch.cat((z, state), dim=1)
        time = torch.full((1, 2), 500., device="cuda")
        # Compare cache isolation under the same autograd mode: bf16 kernels
        # can differ slightly between grad-enabled and no-grad execution.
        empty = goal_action_forward(model, actions, time, condition).detach()
        with torch.no_grad():
            # Populate actual upstream robot-video K/V and global metadata.
            video = torch.randn(1, 4, 1, 1, 2).to(weight)
            model({"latent_res_lst": {"noisy_latents": video,
                       "timesteps": torch.zeros(1, device="cuda")},
                   "latent_grid_id": get_mesh_id(1, 1, 2, 0).to("cuda"),
                   "text_emb": torch.zeros(1, 2, 8).to(weight)}, update_cache=1)
        metadata = {name: getattr(model, name) for name in
                    ("seq_ids_cache", "type_ids_cache", "frame_ids_cache", "cache_type_ids_cache")}
        masks = [(block.attn1.self_block_mask, block.attn2.cross_block_mask) for block in model.blocks]
        caches = [block.attn1.attn_caches["pos"] for block in model.blocks]
        bases = {name: value.detach().clone() for name, value in model.named_parameters()
                 if not value.requires_grad}
        with ExitStack() as guards:
            # No video/image/text projections may execute during goal-conditioned action decoding.
            for module in (model.patch_embedding_mlp, model.patch_embedding, model.proj_out,
                           model.condition_embedder, model.condition_embedder.text_embedder):
                guards.enter_context(patch.object(module, "forward", side_effect=AssertionError("visual bypass")))
            for block in model.blocks:
                for module in (block.attn1.to_q, block.attn1.to_k, block.attn1.to_v,
                               block.attn1.to_out[0], block.attn2.to_q, block.attn2.to_out[0], block.ffn):
                    guards.enter_context(patch.object(module, "forward", side_effect=AssertionError("video projection")))
                # Even an intentionally stale action-local cache must be discarded.
                block.attn1.attn_caches["se3_action"] = block.attn1.attn_caches["pos"]
            prediction = goal_action_forward(model, actions, time, condition)
            torch.testing.assert_close(prediction, empty, rtol=0, atol=0)
            prediction.float().square().mean().backward()
            self.assertGreater(z.grad.float().abs().sum().item(), 0)
            self.assertGreater(state.grad.float().abs().sum().item(), 0)
            action_loras = [module for name, module in model.named_modules()
                            if "action" in name and isinstance(module, VideoLoRA)]
            self.assertTrue(all(module.up.weight.grad is not None and torch.isfinite(module.up.weight.grad).all()
                                for module in action_loras))
            self.assertGreater(sum(module.up.weight.grad.abs().sum().item() for module in action_loras), 0)
            self.assertTrue(all(value.grad is None for name, value in model.named_parameters()
                                if "action" not in name))
            optimizer = torch.optim.SGD([value for value in model.parameters() if value.requires_grad], lr=0.1)
            optimizer.step()
            self.assertTrue(any(module.up.weight.count_nonzero() for module in action_loras))
            for name, value in model.named_parameters():
                if name in bases:
                    torch.testing.assert_close(value, bases[name], rtol=0, atol=0)
            with torch.no_grad():
                expected = goal_action_forward(model, actions, time, condition)
                for cache in caches:
                    cache["k"].fill_(200)
                    cache["v"].fill_(-200)
                perturbed = goal_action_forward(model, actions, time, condition)
                torch.testing.assert_close(expected, perturbed, rtol=0, atol=0)

        for name, value in metadata.items():
            self.assertIs(getattr(model, name), value)
        for block, (self_mask, cross_mask), cache in zip(model.blocks, masks, caches):
            self.assertIs(block.attn1.attn_caches["pos"], cache)
            self.assertNotIn("se3_action", block.attn1.attn_caches)
            self.assertIs(block.attn1.self_block_mask, self_mask)
            self.assertIs(block.attn2.cross_block_mask, cross_mask)
        with patch.object(model.blocks[0], "forward", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                goal_action_forward(model, actions, time, condition)
        self.assertIs(model.blocks[0].attn1.self_block_mask, masks[0][0])
        self.assertIs(model.seq_ids_cache, metadata["seq_ids_cache"])
        merge_action_lora(model)
        merge_icl_lora(model)
        self.assertEqual(set(model.state_dict()), keys)
        with torch.no_grad():
            merged = goal_action_forward(model, actions, time, condition)
        torch.testing.assert_close(expected, merged, rtol=0.03, atol=0.03)

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_native_sampling_is_conditioned_and_masks_every_step(self):
        torch.manual_seed(18)
        model = tiny_model("cuda").requires_grad_(False)
        condition = torch.randn(1, 3, 36).to(model.action_embedder.weight)
        shape, mask = (1, 3, 2, 2, 1), torch.tensor([1, 0, 1]).reshape(1, 3, 1, 1, 1)
        seen = []
        hook = model.action_embedder.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
        try:
            sample = goal_action_sample(model, condition, shape, mask, torch.Generator().manual_seed(9), steps=2)
            again = goal_action_sample(model, condition, shape, mask, torch.Generator().manual_seed(9), steps=2)
            changed = goal_action_sample(model, -condition * 2, shape, mask, torch.Generator().manual_seed(9), steps=2)
        finally:
            hook.remove()
        torch.testing.assert_close(sample, again, rtol=0, atol=0)
        self.assertGreater((sample.float() - changed.float()).abs().max().item(), 1e-3)
        self.assertEqual(len(seen), 6)
        self.assertTrue(all(not value[..., 1].any() for value in seen))
        self.assertFalse(sample[:, 1].any())
        self.assertFalse(sample.requires_grad)
        self.assertTrue(torch.isfinite(sample).all())


if __name__ == "__main__":
    unittest.main()
