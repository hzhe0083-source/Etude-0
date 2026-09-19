import copy
import unittest
from unittest.mock import patch

import torch

from evo_wam.native_icl import forward_video_only, install_icl_lora, merge_icl_lora
from evo_wam.zerowam import NativeDependencyError, VideoLoRA, load_native_class


def tiny_model(device="cpu"):
    return load_native_class()(
        patch_size=(1, 1, 1), num_attention_heads=2, attention_head_dim=18,
        in_channels=4, out_channels=4, action_dim=3, text_dim=8, freq_dim=4,
        ffn_dim=32, num_layers=2, rope_max_seq_len=32, action_inner_dim=36,
        action_ffn_dim=32, attn_window=8, enable_mcp=True,
        num_mcp_modules=1, mcp_hidden_collect_layers=(0, 1),
    ).to(device=device, dtype=torch.bfloat16).eval()


def inputs(model):
    from wan_va.utils import get_mesh_id
    weight = next(model.parameters())

    def stream(channels, shift=0, action=False):
        data = torch.randn(1, channels, 3, 1, 2).to(weight)
        grid = get_mesh_id(3, 1, 2, int(action), action=action).to(weight.device)
        grid[0] += shift
        return {"latent": data, "noisy_latents": torch.randn_like(data),
                "timesteps": torch.ones(1, 3, device=weight.device) * 500,
                "cond_timesteps": torch.zeros(1, 3, device=weight.device),
                "grid_id": grid[None]}

    payload = {"latent_dict": stream(4), "icl_latent_dict": stream(4),
               "text_emb": torch.randn(1, 4, 8).to(weight),
               "encoder_seq_ids": torch.tensor([0, 0, 1, 1], device=weight.device),
               "chunk_size": 1, "max_frame_chunk_size": 4, "window_size": 8,
               "mcp_latent_dicts": [stream(4, shift=1)]}
    payload["icl_latent_dict"]["timesteps"].zero_()
    payload["icl_latent_dict"]["grid_id"][:, 1] += 4
    return payload, stream(3, action=True)


class NativeICLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            load_native_class()
        except NativeDependencyError as exc:
            raise unittest.SkipTest(str(exc))

    def test_adapter_targets_and_merged_schema(self):
        model = tiny_model()
        keys = set(model.state_dict())
        install_icl_lora(model, rank=2, alpha=2)
        self.assertFalse(model.training)
        adapters = [module for module in model.modules() if isinstance(module, VideoLoRA)]
        self.assertEqual(len(adapters), 12)
        for block in model.blocks:
            self.assertNotIsInstance(block.attn1.action_to_k, VideoLoRA)
            self.assertNotIsInstance(block.attn1.action_to_v, VideoLoRA)
            self.assertIs(block.attn2.to_k, block.attn2.action_to_k)
            self.assertNotIsInstance(block.attn2.to_k, VideoLoRA)
        trainable = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(".up.weight" in name or ".down.weight" in name for name in trainable))
        self.assertTrue(all(name.startswith("blocks.") and "action" not in name for name in trainable))
        with self.assertRaisesRegex(ValueError, "unwrapped"):
            install_icl_lora(model)
        merge_icl_lora(model)
        self.assertEqual(set(model.state_dict()), keys)
        tiny_model().load_state_dict(model.state_dict(), strict=True)

    def test_missing_actions_are_not_zero_actions(self):
        model = tiny_model()
        payload, action = inputs(model)
        for value in (None, action):
            with self.assertRaisesRegex(ValueError, "omit action_dict"):
                forward_video_only(model, dict(payload, action_dict=value))

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_native_human_robot_causality_gradients_and_merge(self):
        torch.manual_seed(42)
        model = tiny_model("cuda")
        keys = set(model.state_dict())
        install_icl_lora(model, rank=2, alpha=2)
        payload, action = inputs(model)
        # Native actions are absent, including their embedders and output heads.
        with patch.object(model.action_embedder, "forward", side_effect=AssertionError("action embedding")), \
             patch.object(model.condition_embedder_action, "forward", side_effect=AssertionError("action time")), \
             patch.object(model.action_proj_out, "forward", side_effect=AssertionError("action head")):
            prediction, mcp = forward_video_only(model, payload)
            self.assertEqual(prediction.shape, (1, 6, 4))
            self.assertEqual(len(mcp), 1)
            self.assertTrue(torch.isfinite(prediction).all())
            self.assertTrue(torch.isfinite(mcp[0]).all())
            (prediction.float().square().mean() + mcp[0].float().square().mean()).backward()
        trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
        self.assertTrue(all(value.grad is not None and torch.isfinite(value.grad).all()
                            for _, value in trainable))
        self.assertGreater(sum(value.grad.float().abs().sum().item() for _, value in trainable), 0)
        self.assertTrue(all(value.grad is None for value in model.parameters() if not value.requires_grad))
        # Also verify that the distant-future objective reaches the adapters by itself.
        model.zero_grad(set_to_none=True)
        _, mcp = forward_video_only(model, payload)
        mcp[0].float().square().mean().backward()
        self.assertGreater(sum(value.grad.float().abs().sum().item() for _, value in trainable
                               if value.grad is not None), 0)

        with torch.no_grad():
            baseline, baseline_mcp = forward_video_only(model, payload)
            changed = copy.deepcopy(payload)
            changed["icl_latent_dict"]["latent"] = -changed["icl_latent_dict"]["latent"] * 3
            other, _ = forward_video_only(model, changed)
            self.assertGreater((baseline.float() - other.float()).abs().max().item(), 1e-3)
            # Same/future clean target chunks cannot change the middle noisy chunk.
            changed = copy.deepcopy(payload)
            changed["latent_dict"]["latent"][:, :, 1:] += 20
            causal, causal_mcp = forward_video_only(model, changed)
            torch.testing.assert_close(baseline[:, :4], causal[:, :4], rtol=0, atol=0)
            torch.testing.assert_close(baseline_mcp[0][:, :4], causal_mcp[0][:, :4], rtol=0, atol=0)
            # Earlier observed chunks do condition later predictions.
            changed = copy.deepcopy(payload)
            changed["latent_dict"]["latent"][:, :, :1] += 20
            past_changed, _ = forward_video_only(model, changed)
            self.assertGreater((baseline[:, 2:].float() - past_changed[:, 2:].float()).abs().max().item(), 1e-3)
            ordinary = dict(payload, icl_latent_dict=None, text_emb=payload["text_emb"][:, :2],
                            encoder_seq_ids=payload["encoder_seq_ids"][:2])
            ordinary_prediction, _ = forward_video_only(model, ordinary)
            self.assertTrue(torch.isfinite(ordinary_prediction).all())

        # An actual optimizer update makes merge parity meaningful (LoRA starts at zero).
        optimizer = torch.optim.SGD([value for _, value in trainable], lr=0.2)
        optimizer.step()
        model.zero_grad(set_to_none=True)
        robot_input = dict(payload, action_dict=action)
        robot_video, robot_action, robot_mcp = model(robot_input, train_mode=True)
        self.assertTrue(torch.isfinite(robot_action).all())
        (robot_video.float().square().mean() + robot_action.float().square().mean()
         + robot_mcp[0].float().square().mean()).backward()
        self.assertGreater(sum(value.grad.float().abs().sum().item() for _, value in trainable
                               if value.grad is not None), 0)
        with torch.no_grad():
            before, before_mcp = forward_video_only(model, payload)
            robot_before = model(robot_input, train_mode=True)
            merge_icl_lora(model)
            self.assertEqual(set(model.state_dict()), keys)
            after, after_mcp = forward_video_only(model, payload)
            torch.testing.assert_close(before, after, rtol=0.025, atol=0.025)
            torch.testing.assert_close(before_mcp[0], after_mcp[0], rtol=0.025, atol=0.025)
            video_after, action_after, mcp_after = model(robot_input, train_mode=True)
            self.assertTrue(torch.isfinite(action_after).all())
            torch.testing.assert_close(robot_before[0], video_after, rtol=0.025, atol=0.025)
            torch.testing.assert_close(robot_before[1], action_after, rtol=0.025, atol=0.025)
            torch.testing.assert_close(robot_before[2][0], mcp_after[0], rtol=0.025, atol=0.025)


if __name__ == "__main__":
    unittest.main()
