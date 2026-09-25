import unittest
from unittest.mock import patch
import builtins
import sys
import torch
from torch import nn

from etude.zerowam import (NativeDependencyError, NativeHistoryChunk, TaskConditions, VideoLoRA, ZeroWAMAdapter,
                             load_native_class,
                             optional_flash_source, _LEGACY_FLASH_IMPORT, _OptionalFlashFinder,
                             route_allowed, tiny_native_smoke, unpack_velocity,
                             action_mask_for,
                             verify_source)


class ZeroWAMStructureTests(unittest.TestCase):
    def test_pinned_source(self):
        self.assertTrue((verify_source() / "wan_va/modules/icl_model.py").is_file())

    def test_optional_flash_guard_never_implements_attention(self):
        original_import = builtins.__import__

        def missing_flash(name, *args, **kwargs):
            if name in {"flash_attn", "flash_attn_interface"}:
                raise ImportError("deliberately unavailable in guard test")
            return original_import(name, *args, **kwargs)

        namespace = {}
        module_before = sys.modules.get("flash_attn")
        with patch("builtins.__import__", side_effect=missing_flash):
            exec(optional_flash_source(_LEGACY_FLASH_IMPORT), namespace)
        self.assertTrue(namespace["_EVO_FLASH_UNAVAILABLE"])
        with self.assertRaisesRegex(ImportError, "real flash-attn"):
            namespace["flash_attn_func"](torch.ones(1))
        self.assertIs(sys.modules.get("flash_attn"), module_before)
        with self.assertRaises(RuntimeError):
            optional_flash_source("unexpected source")
        self.assertIsNone(_OptionalFlashFinder(verify_source()).find_spec("flash_attn"))

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

    def test_action_mask_broadcast_and_validation(self):
        sample = torch.randn(1, 3, 2, 4, 1)
        for mask in [torch.tensor([True, False, True]), torch.tensor([1., 0., 1.])]:
            active = action_mask_for(mask.reshape(1, 3, 1, 1, 1), sample)
            self.assertEqual(active.shape, sample.shape)
            self.assertTrue(active[:, 0].all())
            self.assertFalse(active[:, 1].any())
        self.assertTrue(action_mask_for(None, sample).all())
        invalid = [torch.ones(1, 2, 1, 1, 1), torch.tensor([0.5]),
                   torch.tensor([float("nan")]), torch.tensor([float("inf")]),
                   torch.ones(1, requires_grad=True)]
        for mask in invalid:
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                action_mask_for(mask, sample)

    def test_history_chunk_metadata_validation(self):
        latent = torch.zeros(1, 3, 2, 2, 1)
        chunk = NativeHistoryChunk("action", latent, frame_id=3, rope_offset=2,
                                   token_valid=torch.tensor([True, True, True, False]))
        self.assertEqual((chunk.frame_id, chunk.rope_offset), (3, 2))
        for mode, frame_id, offset in [("other", 0, 0), ("video", 1, 0),
                                       ("action", 2, 0), ("video", 0, -1)]:
            with self.assertRaises(ValueError):
                NativeHistoryChunk(mode, latent, frame_id, offset)
        with self.assertRaises(ValueError):
            NativeHistoryChunk("action", latent, 1, 0, torch.ones(4))


class ZeroWAMNativeTests(unittest.TestCase):
    def test_native_cpu_parameter_isolation(self):
        try:
            native = load_native_class()
        except NativeDependencyError as exc:
            self.skipTest(str(exc))
        self.assertFalse(any(isinstance(finder, _OptionalFlashFinder) for finder in sys.meta_path))
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
        self.assertTrue(adapter.condition_types.requires_grad)
        self.assertTrue(attention.action_to_q.up.weight.requires_grad)
        self.assertFalse(attention.to_k.weight.requires_grad)
        adapter.set_stage("reader")
        self.assertFalse(any(p.requires_grad for p in adapter.parameters()))
        adapter.set_stage("joint")
        self.assertTrue(attention.to_q.up.weight.requires_grad)
        self.assertFalse(attention.action_to_q.up.weight.requires_grad)
        self.assertFalse(adapter.condition_projection.weight.requires_grad)
        self.assertFalse(adapter.condition_types.requires_grad)
        self.assertFalse(attention.to_k.weight.requires_grad)
        self.assertTrue(any(p.requires_grad for p in model.mcp_blocks.parameters()))
        stream = adapter._stream(torch.ones(1, 3, 2, 2, 1), "action", 0, frame_id=3,
            rope_offset=2, token_valid=torch.tensor([True, True, True, False]))
        self.assertEqual(stream["action_grid_id"][0].tolist(), [2, 2, 3, 3])
        self.assertEqual(stream["current_frame_ids"].tolist(), [3, 3, 3, -1])
        self.assertEqual(stream["current_seq_ids"].tolist(), [0, 0, 0, -1])
        self.assertFalse(stream["action_res_lst"]["noisy_latents"][:, :, -1, -1].any())
        for mask in [torch.ones(3, dtype=torch.bool), torch.zeros(4, dtype=torch.bool)]:
            with self.assertRaises(ValueError):
                adapter._stream(torch.ones(1, 3, 2, 2, 1), "action", 0, 3,
                                rope_offset=2, token_valid=mask)

    def test_tiny_native_cuda(self):
        try:
            result = tiny_native_smoke()
        except NativeDependencyError as exc:
            self.skipTest(str(exc))
        self.assertTrue(result["native"])
        self.assertTrue(result["direct_condition_gradient"])
        self.assertTrue(result["mcp_phi_only_task_path"])
        self.assertTrue(result["current_remaining_types"])
        self.assertTrue(result["inactive_action_channels_zero_each_step"])
        self.assertTrue(result["observed_action_history"])
        self.assertTrue(result["history_padding_excluded"])
        self.assertTrue(result["independent_rope_offset"])
        self.assertTrue(result["training_observed_history"])
        self.assertTrue(result["differentiable_history_prefix"])
        self.assertTrue(result["paired_training_grids_unchanged"])


if __name__ == "__main__":
    unittest.main()
