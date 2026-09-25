import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from etude.cli import file_sha256
from etude.demo_context import install_demo_interface
from etude.icl_deployment import attach_bottleneck_server, load_bottleneck_deployment
from etude.zerowam import NativeDependencyError, ZERO_WAM_COMMIT, load_native_class
from test_native_icl import tiny_model


class BottleneckDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.native_cls = load_native_class()
        except NativeDependencyError as exc:
            raise unittest.SkipTest(str(exc))

    def setUp(self):
        from safetensors.torch import save_file

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = self.root / "bundle"
        native = tiny_model()
        native.save_config(self.bundle / "backbone")
        save_file({name: value.detach().clone().contiguous() for name, value in native.state_dict().items()},
                  str(self.bundle / "backbone/diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
        self.options = dict(dim=12, num_heads=3, group_frames=2, tokens_per_group=1, layers=1)
        install_demo_interface(native, self.options)
        self.state = {name: value.detach().clone() for name, value in native.demo_bottleneck.state_dict().items()}
        torch.save(self.state, self.bundle / "demo_bottleneck.pt")
        self.manifest = dict(format_version=1, kind="native_icl_temporal_bottleneck", upstream_commit=ZERO_WAM_COMMIT,
                             demo_bottleneck=self.options, icl_rope_h=4, feature_space_id="test-wan", tiny_native=True,
                             files={path.relative_to(self.bundle).as_posix(): file_sha256(path)
                                    for path in self.bundle.rglob("*") if path.is_file()})
        self.write_manifest()

    def write_manifest(self):
        (self.bundle / "bottleneck.json").write_text(json.dumps(self.manifest))

    def make_server(self, device="cpu"):
        native = self.native_cls.from_pretrained(str(self.bundle / "backbone"), local_files_only=True,
                                                use_safetensors=True).to(device=device, dtype=torch.bfloat16)
        return SimpleNamespace(transformer=native, device=torch.device(device), dtype=torch.bfloat16,
                               use_icl_model=True, cache_name="pos", job_config=SimpleNamespace(icl_rope_h=4),
                               _infer_icl=lambda: "original inference", _compute_icl_kv_cache=lambda: "original history")

    def make_clip(self):
        path = self.root / "clip.npz"
        np.savez_compressed(path, latent=np.random.default_rng(9).normal(size=(4, 5, 1, 2)).astype(np.float32),
                            frame_times=np.array([0., .1, .3, .6, .7], dtype=np.float64))
        metadata = {"format_version": 1, "kind": "native_icl_clip", "arrays": path.name,
                    "source_id": "demo-A", "source_group": "recording-A", "domain": "human",
                    "feature_space_id": "test-wan", "provenance": {
                        "latent_normalization": "(posterior_mode-mean)/std",
                        "latent_layout": "channel,time,height,width", "feature_encoder": "frozen-wan-vae",
                        "continuous_segment_verified": True, "injected_test_encoder": True}}
        (self.root / "clip.json").write_text(json.dumps(metadata))
        return path, metadata

    def test_complete_restore_frozen_and_stock_root_rejected(self):
        native, null = load_bottleneck_deployment(self.bundle, device="cpu")
        self.assertEqual(type(native), self.native_cls)
        self.assertEqual(null.shape, (1, 2, 8))
        self.assertEqual(native.demo_feature_space_id, "test-wan")
        self.assertEqual(native.demo_icl_rope_h, 4)
        self.assertFalse(native.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in native.parameters()))
        for name, value in self.state.items():
            torch.testing.assert_close(native.demo_bottleneck.state_dict()[name], value.float(), rtol=0, atol=0)
        with self.assertRaises(OSError):
            self.native_cls.from_pretrained(str(self.bundle), local_files_only=True, use_safetensors=True)

    def test_missing_tampered_untracked_and_escaping_files_fail_before_load(self):
        sidecar = self.bundle / "demo_bottleneck.pt"
        original = sidecar.read_bytes()
        with patch("etude.icl_deployment.load_native_class", side_effect=AssertionError("loaded before auditing")):
            sidecar.unlink()
            with self.assertRaisesRegex(ValueError, "missing or untracked"):
                load_bottleneck_deployment(self.bundle, device="cpu")
            sidecar.write_bytes(original + b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_bottleneck_deployment(self.bundle, device="cpu")
            sidecar.write_bytes(original)
            extra = self.bundle / "backbone/extra.safetensors.index.json"
            extra.write_text("{}")
            with self.assertRaisesRegex(ValueError, "missing or untracked"):
                load_bottleneck_deployment(self.bundle, device="cpu")
            extra.unlink()
            self.manifest["files"]["../outside.safetensors"] = "0" * 64
            self.write_manifest()
            with self.assertRaisesRegex(ValueError, "local native files"):
                load_bottleneck_deployment(self.bundle, device="cpu")

    def test_shard_index_cannot_reference_untracked_or_remote_weights(self):
        index = self.bundle / "backbone/diffusion_pytorch_model.safetensors.index.json"
        for shard in ("../outside.safetensors", "missing.safetensors", "https://remote/model.safetensors"):
            index.write_text(json.dumps({"weight_map": {"weight": shard}}))
            self.manifest["files"][index.relative_to(self.bundle).as_posix()] = file_sha256(index)
            self.write_manifest()
            with self.subTest(shard=shard), self.assertRaisesRegex(ValueError, "shard index"):
                load_bottleneck_deployment(self.bundle, device="cpu")

    def test_server_binding_in_place_checks_backbone_rope_and_clip_identity(self):
        server = self.make_server()
        native = server.transformer
        infer, history = server._infer_icl, server._compute_icl_kv_cache
        server.job_config.icl_rope_h = 5
        with self.assertRaisesRegex(ValueError, "icl_rope_h"):
            attach_bottleneck_server(server, self.bundle)
        self.assertFalse(hasattr(native, "demo_bottleneck"))
        server.job_config.icl_rope_h = 4
        with patch.object(self.native_cls, "from_pretrained", side_effect=AssertionError("second full backbone load")):
            self.assertIs(attach_bottleneck_server(server, self.bundle), server)
        self.assertIs(server.transformer, native)
        self.assertIs(server._infer_icl, infer)
        self.assertIs(server._compute_icl_kv_cache, history)
        for video, latent in (("demo.mp4", ""), ("", "legacy.pt")):
            with self.assertRaisesRegex(ValueError, "preprocess-icl-video"):
                server._cache_icl_context(video, latent)
        path, metadata = self.make_clip()
        for change in ({"feature_space_id": "another-vae"}, {"arrays": "another.npz"}):
            (self.root / "clip.json").write_text(json.dumps({**metadata, **change}))
            with self.assertRaisesRegex(ValueError, "feature space"):
                server._cache_icl_context("", path)
        (self.root / "clip.json").write_text(json.dumps(metadata))
        np.savez_compressed(path, latent=np.zeros((4, 5, 1, 2), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "latent/frame_times"):
            server._cache_icl_context("", path)
        wrong = self.make_server()
        wrong.transformer.register_to_config(_name_or_path=str(self.root))
        with self.assertRaisesRegex(ValueError, "pointing to this bundle/backbone"):
            attach_bottleneck_server(wrong, self.bundle)

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_original_server_cache_entry_uses_actual_compressed_native_cache(self):
        server = attach_bottleneck_server(self.make_server("cuda"), self.bundle)
        path, _ = self.make_clip()
        native = server.transformer
        frozen = {name: value.clone() for name, value in native.state_dict().items()}
        with patch.object(native.demo_bottleneck, "forward", wraps=native.demo_bottleneck.forward) as compress:
            counts = server._cache_icl_context("", path)
        self.assertEqual(compress.call_count, 1)
        self.assertEqual(counts[2], 3)  # ceil(5 frames / 2) groups, one token each.
        self.assertEqual(native.cache_counts()[2], 3)
        for name, value in native.state_dict().items():
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
