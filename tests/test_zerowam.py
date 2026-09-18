import unittest
import torch
from torch import nn

from evo_wam.zerowam import (NativeDependencyError, TaskConditions, VideoLoRA, ZeroWAMAdapter,
                             load_native_class,
                             route_allowed, tiny_native_smoke, unpack_velocity,
                             verify_source)


class ZeroWAMStructureTests(unittest.TestCase):
    def test_pinned_source(self):
        self.assertTrue((verify_source() / "wan_va/modules/icl_model.py").is_file())

    def test_task_slots_and_unconditional(self):
        q, k = torch.arange(5)[:, None], torch.arange(3)[None, :]
        args = dict(latent_length=2, action_length=2, null_length=1, current_length=1)
        mask = route_allowed(q, k, conditional=True, mcp=False, **args)
        self.assertEqual(mask.tolist(), [[True, True, True], [True, True, True],
                                         [True, True, False], [True, True, False],
                                         [False, False, False]])
        for conditional, mcp in [(False, False), (True, True)]:
            mask = route_allowed(q, k, conditional=conditional, mcp=mcp, **args)
            self.assertFalse(mask[:, 1:].any())
        c = TaskConditions(torch.ones(1, 1, 2), torch.ones(1, 1, 2), torch.zeros(1, 1, 2))
        self.assertIsNone(c.unconditional().current)
        self.assertIsNone(c.unconditional().remaining)

    def test_lora_preserves_base_and_receives_gradients(self):
        base = nn.Linear(6, 4)
        lora = VideoLoRA(base, rank=2)
        self.assertEqual(lora.scale, 1.0)
        self.assertEqual(VideoLoRA(nn.Linear(6, 4), rank=2, alpha=8).scale, 4.0)
        x = torch.randn(3, 6, requires_grad=True)
        self.assertTrue(torch.equal(lora(x), base(x)))
        lora(x).square().mean().backward()
        self.assertIsNone(base.weight.grad)
        self.assertGreater(lora.up.weight.grad.abs().sum(), 0)
        self.assertGreater(x.grad.abs().sum(), 0)

    def test_unpack_matches_upstream_patch_order(self):
        shape, patch_size = (1, 2, 2, 4, 4), (1, 2, 2)
        clean = torch.arange(64).reshape(shape)
        packed = clean.reshape(1, 2, 2, 1, 2, 2, 2, 2).permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(1, 32, 2)
        self.assertTrue(torch.equal(unpack_velocity(packed, shape, patch_size), clean))
        with self.assertRaises(ValueError):
            unpack_velocity(packed[:, :-1], shape, patch_size)


class ZeroWAMNativeTests(unittest.TestCase):
    def test_native_cpu_parameter_isolation(self):
        try:
            native = load_native_class()
        except NativeDependencyError as exc:
            self.skipTest(str(exc))
        model = native(patch_size=(1, 1, 1), num_attention_heads=2,
                       attention_head_dim=18, in_channels=4, out_channels=4,
                       action_dim=3, text_dim=8, freq_dim=4, ffn_dim=16,
                       num_layers=1, action_inner_dim=36, action_ffn_dim=16,
                       num_mcp_modules=1, mcp_hidden_collect_layers=(0,))
        adapter = ZeroWAMAdapter(model, condition_dim=8, lora_rank=2)
        attention = model.blocks[0].attn2
        self.assertIs(attention.to_k, attention.action_to_k)
        self.assertIs(attention.to_v, attention.action_to_v)
        adapter.set_stage("interface")
        self.assertTrue(adapter.condition_projection.weight.requires_grad)
        self.assertTrue(attention.action_to_q.up.weight.requires_grad)
        self.assertFalse(attention.to_k.weight.requires_grad)
        adapter.set_stage("reader")
        self.assertFalse(any(p.requires_grad for p in adapter.parameters()))
        adapter.set_stage("joint")
        self.assertTrue(attention.to_q.up.weight.requires_grad)
        self.assertFalse(attention.action_to_q.up.weight.requires_grad)
        self.assertFalse(adapter.condition_projection.weight.requires_grad)
        self.assertFalse(attention.to_k.weight.requires_grad)
        self.assertTrue(any(p.requires_grad for p in model.mcp_blocks.parameters()))

    def test_tiny_native_cuda(self):
        try:
            result = tiny_native_smoke()
        except NativeDependencyError as exc:
            self.skipTest(str(exc))
        self.assertTrue(result["native"])
        self.assertTrue(result["direct_condition_gradient"])
        self.assertTrue(result["mcp_phi_only_task_path"])


if __name__ == "__main__":
    unittest.main()
