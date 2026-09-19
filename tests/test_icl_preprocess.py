"""Native ICL RGB cache checks; tiny random weights do not establish task quality."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from evo_wam.icl_preprocess import preprocess_icl_video
from evo_wam.vision import encode_rgb, read_video, sha256


def raw_spec():
    return {"format_version": 1, "kind": "raw_icl_video", "video": "clip.avi",
            "vae_path": "tiny-vae", "vae_sha256": {"config.json": "0" * 64},
            "size": [32, 48], "fps": 5, "domain": "human", "source_id": "human-clip",
            "source_group": "original-recording", "feature_space_id": "tiny-wan-2channels-v1",
            "continuous_segment_verified": True}


class IclPreprocessContractTest(unittest.TestCase):
    def test_invalid_metadata_is_rejected_before_video_io(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "raw.json"
            for change in ({"format_version": True}, {"kind": "native_icl_sample"}, {"fps": float("nan")},
                           {"fps": True}, {"size": [16, 17]}, {"size": "32"}, {"source_group": ""},
                           {"domain": "unknown"}, {"continuous_segment_verified": False},
                           {"trajectory_id": "human-action-label"}, {"actions": [0]}, {"proprio": []},
                           {"geometry": []}, {"roi": []}, {"vae_sha256": {}},
                           {"vae_sha256": {"config.json": "bad-digest"}}, {"video": "https://example.com/clip"}):
                path.write_text(json.dumps({**raw_spec(), **change}))
                with self.subTest(change=change), patch("evo_wam.icl_preprocess.read_video") as reader:
                    with self.assertRaises(ValueError):
                        preprocess_icl_video(path, root / "out", device="cpu", vae=object())
                    reader.assert_not_called()
                self.assertFalse((root / "out").exists())

    def test_invalid_latent_and_sampling_cannot_create_cache(self):
        frames = np.zeros((9, 32, 48, 3), dtype=np.uint8)
        sampling = {"source_fps": 10., "frame_indices": list(range(9))}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clip.avi").write_bytes(b"metadata-only fixture")
            path = root / "raw.json"
            path.write_text(json.dumps(raw_spec()))
            for latent in (torch.zeros(1, 2, 2, 2, 3), torch.full((1, 2, 3, 2, 3), float("nan")),
                           torch.zeros(1, 2, 3, 2, 3, dtype=torch.int64)):
                with self.subTest(shape=latent.shape, dtype=latent.dtype), \
                        patch("evo_wam.icl_preprocess.read_video", return_value=(frames, sampling)), \
                        patch("evo_wam.icl_preprocess.encode_rgb", return_value=latent):
                    with self.assertRaisesRegex(ValueError, "one time per causal endpoint"):
                        preprocess_icl_video(path, root / "out", vae=object())
                self.assertFalse((root / "out").exists())
            for malformed in ({"source_fps": float("inf")}, {"frame_indices": [0] * 9},
                              {"frame_indices": list(range(8))}, {"frame_indices": list(np.arange(9.))}):
                with self.subTest(sampling=malformed), \
                        patch("evo_wam.icl_preprocess.read_video", return_value=(frames, {**sampling, **malformed})), \
                        patch("evo_wam.icl_preprocess.encode_rgb") as encoder:
                    with self.assertRaises(ValueError):
                        preprocess_icl_video(path, root / "out", vae=object())
                    encoder.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("diffusers") and importlib.util.find_spec("cv2"),
                     "native Diffusers/OpenCV extras are unavailable")
class NativeIclPreprocessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_vision import NativeWanVaeTest
        cls.fixture = NativeWanVaeTest
        cls.fixture.setUpClass()
        cls.vae = cls.fixture.vae

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDownClass()

    def test_real_video_cache_preserves_native_axes_normalization_time_and_provenance(self):
        import cv2

        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 10, (48, 32))
            if not writer.isOpened():
                writer.release()
                self.skipTest("OpenCV build has no FFV1 lossless encoder")
            try:
                for frame in self.fixture.frames(self, 17):
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            weights = root / "tiny-vae"
            self.vae.save_pretrained(weights)
            identity = {file.name: sha256(file) for file in weights.iterdir()}
            path = root / "raw.json"
            path.write_text(json.dumps({**raw_spec(), "vae_sha256": identity}))
            report = preprocess_icl_video(path, root / "cache", device="cpu", vae=self.vae)
            decoded, sampling = read_video(video, 5)
            expected = encode_rgb(self.vae, decoded, [32, 48])[0].numpy()
            self.assertEqual(report["shape"], [2, 3, 2, 3])
            self.assertEqual(report["commands_sent"], 0)
            with np.load(report["arrays"], allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {"latent", "frame_times"})
                np.testing.assert_array_equal(archive["latent"], expected)
                np.testing.assert_array_equal(archive["frame_times"], [0, .8, 1.6])
            metadata = json.loads(Path(report["manifest"]).read_text())
            self.assertEqual(metadata["kind"], "native_icl_clip")
            self.assertEqual(metadata["arrays"], "clip.npz")
            self.assertEqual(metadata["source_group"], raw_spec()["source_group"])
            provenance = metadata["provenance"]
            self.assertEqual(provenance["source_video_sha256"], sha256(video))
            self.assertEqual(provenance["manifest_sha256"], sha256(path))
            self.assertEqual(provenance["vae_sha256"], identity)
            self.assertEqual(provenance["frame_indices"], sampling["frame_indices"])
            self.assertEqual(provenance["latent_endpoint_source_frame_indices"], [0, 8, 16])
            self.assertEqual(provenance["latent_normalization"], "(posterior_mode-mean)/std")
            self.assertEqual(provenance["frame_time_basis"], "source_frame_index/source_fps")
            self.assertTrue(provenance["injected_test_encoder"])
            with self.assertRaisesRegex(ValueError, "fresh directory"):
                preprocess_icl_video(path, root / "cache", device="cpu", vae=self.vae)

            # Exercise the real local-only loader with the same tiny random weights.
            robot_spec = {**raw_spec(), "vae_sha256": identity, "domain": "robot", "trajectory_id": "trajectory-1"}
            path.write_text(json.dumps(robot_spec))
            local = preprocess_icl_video(path, root / "robot-cache", device="cpu")
            robot = json.loads(Path(local["manifest"]).read_text())
            self.assertEqual(robot["trajectory_id"], "trajectory-1")
            self.assertFalse(robot["provenance"]["injected_test_encoder"])
            with np.load(local["arrays"], allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["latent"], expected)
            path.write_text(json.dumps({**robot_spec, "vae_sha256": {**identity, "config.json": "0" * 64}}))
            with patch("evo_wam.icl_preprocess.read_video") as reader:
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    preprocess_icl_video(path, root / "bad-identity", device="cpu")
                reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
