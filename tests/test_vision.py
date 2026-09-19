"""Preprocessing contracts plus a real, randomly initialized tiny Wan VAE.

No released weights, detector, learned tracking accuracy or robot performance
are tested. Capture doubles below isolate sampling; a lossless local video and
the native Diffusers encoder exercise the actual decoding/encoding libraries.
"""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from evo_wam.vision import encode_rgb, load_vae, preprocess, preprocess_video, read_video, robot_features, sha256


class Capture:
    """A sampling-protocol double, not evidence about a real video decoder."""

    def __init__(self, frames, fps=10.0, opened=True):
        self.frames, self.fps, self.opened = iter(frames), fps, opened
        self.released = False

    def get(self, _property):
        return self.fps

    def isOpened(self):
        return self.opened

    def read(self):
        frame = next(self.frames, None)
        return frame is not None, frame

    def release(self):
        self.released = True


@unittest.skipUnless(importlib.util.find_spec("cv2"), "native OpenCV extra is unavailable")
class VideoSamplingTest(unittest.TestCase):
    def test_sampling_is_ordered_rgb_and_truncates_only_incomplete_causal_tail(self):
        frames = [np.full((16, 16, 3), (i, 30, 200 - i), dtype=np.uint8) for i in range(14)]
        capture = Capture(frames)
        with patch("cv2.VideoCapture", return_value=capture):
            sampled, metadata = read_video("contract-double.avi", 4)
        self.assertEqual(metadata, {"source_fps": 10.0, "frame_indices": [0, 3, 5, 8, 10]})
        self.assertEqual(sampled.shape, (5, 16, 16, 3))
        np.testing.assert_array_equal(sampled[:, 0, 0], [[200 - i, 30, i] for i in metadata["frame_indices"]])
        self.assertTrue(capture.released)

    def test_faster_requested_rate_does_not_duplicate_frames(self):
        capture = Capture([np.full((16, 16, 3), i, dtype=np.uint8) for i in range(9)])
        with patch("cv2.VideoCapture", return_value=capture):
            frames, metadata = read_video("contract-double.avi", 30)
        self.assertEqual(metadata["frame_indices"], list(range(9)))
        np.testing.assert_array_equal(frames[:, 0, 0, 0], np.arange(9))

    def test_bad_rate_unreadable_and_empty_captures_are_rejected_and_released(self):
        for rate in (0, -1, float("nan"), float("inf")):
            with self.subTest(rate=rate), patch("cv2.VideoCapture") as factory:
                with self.assertRaises(ValueError):
                    read_video("unused.avi", rate)
                factory.assert_not_called()
        for capture in (Capture([], opened=False), Capture([], fps=float("nan")), Capture([])):
            with self.subTest(capture=capture), patch("cv2.VideoCapture", return_value=capture):
                with self.assertRaises(ValueError):
                    read_video("contract-double.avi", 10)
                self.assertTrue(capture.released)

    def test_real_lossless_video_rgb_order_and_frame_indices(self):
        import cv2

        with TemporaryDirectory() as directory:
            path = Path(directory) / "ordered.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 10, (32, 32))
            if not writer.isOpened():
                writer.release()
                self.skipTest("OpenCV build has no FFV1 lossless encoder")
            try:
                for index in range(9):
                    writer.write(np.full((32, 32, 3), (index * 10, 50, 200 - index * 10), dtype=np.uint8))
            finally:
                writer.release()
            frames, metadata = read_video(path, 10)
            self.assertEqual(metadata["frame_indices"], list(range(9)))
            self.assertAlmostEqual(metadata["source_fps"], 10)
            np.testing.assert_array_equal(frames[:, 0, 0], [[200 - i * 10, 50, i * 10] for i in range(9)])


class RobotFeatureTest(unittest.TestCase):
    def setUp(self):
        # Native layout is [1,channels,time,height,width], cameras join on width.
        self.latents = [torch.empty(1, 2, 3, 2, 2) for _ in range(2)]
        for camera, latent in enumerate(self.latents):
            for time in range(3):
                for channel in range(2):
                    latent[0, channel, time] = (
                        torch.arange(4).reshape(2, 2) + 100 * camera + 10 * time + 1000 * channel
                    )
        self.ids = np.array([10, 20], dtype=np.int64)
        self.masks = np.zeros((3, 2, 2, 2, 2), dtype=np.float32)
        self.masks[:, 0, 0, 0, 1] = 1
        self.masks[:, 1, 1, 1, 0] = 1

    def test_camera_time_and_entity_order_are_preserved(self):
        packed, features = robot_features(self.latents, self.masks, self.ids)
        np.testing.assert_array_equal(packed, torch.cat(self.latents, dim=-1)[0].numpy())
        expected = np.array([[[1 + 10 * t, 1001 + 10 * t], [102 + 10 * t, 1102 + 10 * t]] for t in range(3)])
        np.testing.assert_allclose(features, expected)
        _, permuted = robot_features(self.latents, self.masks[:, ::-1].copy(), self.ids[::-1].copy())
        np.testing.assert_array_equal(permuted, features[:, ::-1])

    def test_moving_mask_tracks_each_endpoint_and_area_pooling(self):
        masks = np.zeros((3, 1, 2, 4, 4), dtype=np.float32)
        masks[0, 0, 0, :2, :2] = 1
        masks[1, 0, 1, 2:, 2:] = 1
        masks[2, 0, 0, :, :] = 0.5
        masks[2, 0, 1, :, :] = 0.5
        _, features = robot_features(self.latents, masks, np.array([10], dtype=np.int64))
        np.testing.assert_allclose(features[:, 0], [[0, 1000], [113, 1113], [71.5, 1071.5]])

    def test_absence_misaligned_time_bad_masks_and_unstable_ids_are_rejected(self):
        bad_masks = [self.masks[:2], self.masks.copy(), self.masks.copy(), self.masks.copy()]
        bad_masks[1][1, 0] = 0
        bad_masks[2][0, 0, 0, 0, 0] = float("nan")
        bad_masks[3][0, 0, 0, 0, 0] = 2
        for masks in bad_masks:
            with self.subTest(shape=masks.shape), self.assertRaises(ValueError):
                robot_features(self.latents, masks, self.ids)
        for ids in (np.array([10, 10]), np.array([-1, 10]), np.array([10.0, 20.0])):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                robot_features(self.latents, self.masks, ids)

    def test_latent_batch_and_nonfinite_values_are_not_silently_discarded(self):
        # This API returns one observation. It must not silently select batch 0.
        with self.assertRaises(ValueError):
            robot_features([x.repeat(2, 1, 1, 1, 1) for x in self.latents], self.masks, self.ids)
        bad = [x.clone() for x in self.latents]
        bad[0][0, 0, 0, 0, 1] = float("nan")
        with self.assertRaises(ValueError):
            robot_features(bad, self.masks, self.ids)


class VideoPretrainManifestTest(unittest.TestCase):
    def test_invalid_windows_and_unscreened_segments_fail_before_video_io(self):
        spec = {"format_version": 1, "kind": "raw_video_pretrain", "video": "segment.avi",
                "vae_path": "local-vae", "vae_sha256": {"config.json": "0" * 64},
                "size": [32, 32], "fps": 10, "domain": "human", "source_id": "original-clip",
                "source_group": "original-recording", "feature_space_id": "wan-test-v1", "split": "train",
                "window_frames": 3, "context_frames": 1, "continuous_segment_verified": True}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for change in ({"window_frames": 2}, {"context_frames": 0}, {"context_frames": 2},
                           {"window_stride_frames": 4}, {"continuous_segment_verified": False},
                           {"source_group": ""}, {"clip_start_seconds": 1}, {"split": "validation/test"}):
                path = root / "raw.json"
                path.write_text(json.dumps({**spec, **change}))
                with self.subTest(change=change), patch("evo_wam.vision.read_video") as reader:
                    with self.assertRaises(ValueError):
                        preprocess_video(path, root / "output", device="cpu", vae=object())
                    reader.assert_not_called()
                self.assertFalse((root / "output").exists())


@unittest.skipUnless(importlib.util.find_spec("diffusers"), "native Diffusers extra is unavailable")
class NativeWanVaeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from diffusers import AutoencoderKLWan

        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        with torch.random.fork_rng():
            torch.manual_seed(7)
            # Actual Wan-2.2-style residual/patchified implementation, tiny random
            # channels only. This is not a released or trained visual encoder.
            cls.vae = AutoencoderKLWan(
                base_dim=4, decoder_base_dim=4, z_dim=2, dim_mult=[1, 1, 1, 1],
                num_res_blocks=1, temperal_downsample=[False, True, True],
                latents_mean=[0.5, -1.5], latents_std=[2.0, 4.0],
                is_residual=True, patch_size=2, scale_factor_spatial=16,
                in_channels=12, out_channels=12,
            ).eval().requires_grad_(False)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def frames(self, count=9):
        y, x = np.indices((32, 48))
        return np.stack([np.stack(((x + 17 * t) % 256, (y + 31 * t) % 256, (x + y + 7 * t) % 256), -1)
                         for t in range(count)]).astype(np.uint8)

    def test_native_encode_shape_rgb_axes_and_per_channel_normalization(self):
        frames = self.frames()
        actual = encode_rgb(self.vae, frames, [32, 32])
        # Independent input layout follows the pinned upstream server's C,T,H,W
        # interpolation; compare against a real native posterior mode.
        pixels = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
        pixels = F.interpolate(pixels, size=(32, 32), mode="bilinear", align_corners=False)
        with torch.no_grad():
            raw = self.vae.encode((pixels / 255 * 2 - 1).unsqueeze(0)).latent_dist.mode()
        expected = (raw - torch.tensor([0.5, -1.5])[None, :, None, None, None]) / torch.tensor([2.0, 4.0])[None, :, None, None, None]
        self.assertEqual(actual.shape, (1, 2, 3, 2, 2))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.device.type, "cpu")
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_real_native_temporal_prefix_is_causal_and_repeatable(self):
        frames = self.frames()
        original = encode_rgb(self.vae, frames, [32, 32])
        changed = frames.copy()
        changed[5:] = 255 - changed[5:]
        modified = encode_rgb(self.vae, changed, [32, 32])
        prefix = encode_rgb(self.vae, frames[:5], [32, 32])
        torch.testing.assert_close(original[:, :, :2], prefix)
        torch.testing.assert_close(original[:, :, :2], modified[:, :, :2])
        self.assertGreater((original[:, :, 2] - modified[:, :, 2]).abs().max().item(), 1e-6)
        endpoint_frames = frames.copy()
        endpoint_frames[4] = 255 - endpoint_frames[4]
        endpoint = encode_rgb(self.vae, endpoint_frames, [32, 32])
        torch.testing.assert_close(original[:, :, 0], endpoint[:, :, 0])
        self.assertGreater((original[:, :, 1] - endpoint[:, :, 1]).abs().max().item(), 1e-6)
        torch.testing.assert_close(original, encode_rgb(self.vae, frames, [32, 32]))

    def test_bad_rgb_dtype_clip_length_resize_and_native_scale_are_rejected(self):
        frames = self.frames()
        for bad in (frames.astype(np.float32), frames[:4], frames[..., :2]):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                encode_rgb(self.vae, bad, [32, 32])
        for size in ([31, 32], [0, 32], [True, 32], [32]):
            with self.subTest(size=size), self.assertRaises(ValueError):
                encode_rgb(self.vae, frames, size)
        original = self.vae.config.latents_std
        try:
            self.vae.register_to_config(latents_std=[0.0, 1.0])
            with self.assertRaises(ValueError):
                encode_rgb(self.vae, frames[:1], [32, 32])
        finally:
            self.vae.register_to_config(latents_std=original)

    def test_local_saved_native_weights_load_frozen_with_verified_identity(self):
        from diffusers import AutoencoderKLWan

        with TemporaryDirectory() as directory:
            path = Path(directory)
            self.vae.save_pretrained(path)
            expected = {file.name: sha256(file) for file in path.iterdir()}
            with patch.object(AutoencoderKLWan, "from_pretrained", wraps=AutoencoderKLWan.from_pretrained) as loader:
                loaded = load_vae(path, expected, device="cpu")
                self.assertIs(loader.call_args.kwargs["local_files_only"], True)
                self.assertIs(loader.call_args.kwargs.get("use_safetensors"), True)
            self.assertFalse(loaded.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))
            torch.testing.assert_close(encode_rgb(loaded, self.frames(5), [32, 32]), encode_rgb(self.vae, self.frames(5), [32, 32]))
            with self.assertRaises(ValueError):
                load_vae(path, {"config.json": expected["config.json"]}, device="cpu")
            with (path / "config.json").open("a") as stream:
                stream.write(" ")
            with self.assertRaises(ValueError):
                load_vae(path, expected, device="cpu")

    def test_loader_cannot_fallback_to_unhashed_pickle_weights(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            self.vae.save_pretrained(path, safe_serialization=False)
            # This file is hashed but is not the weights Diffusers would load.
            (path / "unrelated.safetensors").write_bytes(b"not the model weights")
            expected = {name: sha256(path / name) for name in ("config.json", "unrelated.safetensors")}
            with self.assertRaises((ValueError, OSError)):
                load_vae(path, expected, device="cpu")

    def test_sharded_native_checkpoint_identity_includes_index(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            self.vae.save_pretrained(path, max_shard_size="10KB")
            expected = {file.name: sha256(file) for file in path.iterdir()}
            indices = [name for name in expected if name.endswith(".safetensors.index.json")]
            self.assertEqual(len(indices), 1)
            with self.assertRaises(ValueError):
                load_vae(path, {name: digest for name, digest in expected.items() if name not in indices}, device="cpu")
            loaded = load_vae(path, expected, device="cpu")
            torch.testing.assert_close(encode_rgb(loaded, self.frames(1), [32, 32]), encode_rgb(self.vae, self.frames(1), [32, 32]))

    def test_shard_index_cannot_reference_an_unhashed_external_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "vae"
            self.vae.save_pretrained(path, max_shard_size="10KB")
            index_path = next(path.glob("*.safetensors.index.json"))
            index = json.loads(index_path.read_text())
            first_shard = next(iter(index["weight_map"].values()))
            (path / first_shard).rename(Path(directory) / "outside.safetensors")
            index["weight_map"] = {key: "../outside.safetensors" if name == first_shard else name
                                   for key, name in index["weight_map"].items()}
            index_path.write_text(json.dumps(index))
            expected = {file.name: sha256(file) for file in path.iterdir()}
            with self.assertRaises(ValueError):
                load_vae(path, expected, device="cpu")

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "native OpenCV extra is unavailable")
    def test_single_unpaired_video_emits_patch_windows_at_actual_source_seconds(self):
        import cv2
        from evo_wam.video_data import load_video_index, load_video_window

        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "single-human.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 10, (48, 32))
            if not writer.isOpened():
                writer.release()
                self.skipTest("OpenCV build has no FFV1 lossless encoder")
            try:
                for rgb in self.frames(49):
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            spec = {"format_version": 1, "kind": "raw_video_pretrain", "video": video.name,
                    "vae_path": "unused-injected-test-encoder", "vae_sha256": {"config.json": "0" * 64},
                    "size": [32, 32], "fps": 5, "domain": "human", "source_id": "human-clip",
                    "source_group": "original-human-recording", "feature_space_id": "tiny-wan-2channels-v1",
                    "split": "train", "window_frames": 3, "context_frames": 1,
                    "continuous_segment_verified": True}
            path = root / "raw-video.json"
            path.write_text(json.dumps(spec))
            report = preprocess_video(path, root / "encoded", device="cpu", vae=self.vae)
            self.assertEqual(report["windows"], 2)
            self.assertEqual(report["tokens_per_frame"], 4)
            self.assertEqual(report["feature_dim"], 2)
            self.assertFalse(report["automatic_effect_labels"])
            self.assertFalse(report["released_vae_run"])
            index = json.loads(Path(report["index"]).read_text())
            load_video_index(report["index"])
            self.assertEqual(index["bridge_sources"], [])
            self.assertEqual([row["split"] for row in index["samples"]], ["train", "train"])
            frames, sampling = read_video(video, 5)
            expected = encode_rgb(self.vae, frames, [32, 32])[0].permute(1, 2, 3, 0).flatten(1, 2)
            expected_times = np.asarray(sampling["frame_indices"])[::4] / sampling["source_fps"]
            np.testing.assert_allclose(expected_times, [0, .8, 1.6, 2.4, 3.2, 4, 4.8])
            for i, entry in enumerate(index["samples"]):
                window_path = Path(report["index"]).parent / entry["manifest"]
                window = load_video_window(window_path)
                torch.testing.assert_close(window.features[0], expected[i * 3:i * 3 + 3])
                np.testing.assert_allclose(window.frame_times.numpy(), expected_times[i * 3:i * 3 + 3])
                self.assertTrue(window.feature_valid.all())
                self.assertEqual(window.effect_targets, {})
                self.assertEqual(window.context_frames, 1)
                self.assertEqual(window.metadata["source_group"], spec["source_group"])
                self.assertEqual(window.metadata["feature_kind"], "patches")
                provenance = window.metadata["provenance"]
                self.assertTrue(provenance["injected_test_encoder"])
                self.assertEqual(provenance["source_video_sha256"], sha256(video))
                self.assertEqual(provenance["window_policy"]["discarded_tail_frames"], 1)
                with np.load(window_path.with_suffix(".npz"), allow_pickle=False) as arrays:
                    self.assertEqual(set(arrays.files), {"features", "feature_valid", "frame_times"})
            with self.assertRaises(ValueError):
                preprocess_video(path, root / "encoded", device="cpu", vae=self.vae)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "native OpenCV extra is unavailable")
    def test_real_raw_preprocessing_loads_v2_observation_with_explicit_test_provenance(self):
        import cv2
        from evo_wam.data import load_observation

        with TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "tiny-random-vae"
            self.vae.save_pretrained(weights)
            vae_identity = {file.name: sha256(file) for file in weights.iterdir()}
            raw_front = self.frames(9)
            raw_wrist = 255 - raw_front
            demonstrations = []
            # Synthetic demonstration pixels in actual lossless video files;
            # this verifies preprocessing, not human action understanding.
            for view, frames in (("front", raw_front), ("side", raw_wrist)):
                video = root / f"human-fixture-{view}.avi"
                writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 10, (48, 32))
                if not writer.isOpened():
                    writer.release()
                    self.skipTest("OpenCV build has no FFV1 lossless encoder")
                try:
                    for rgb in frames:
                        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                finally:
                    writer.release()
                demonstrations.append({"view_id": view, "video": video.name})

            masks = np.zeros((3, 2, 2, 32, 48), dtype=np.float32)
            masks[:, 0, 0, :, :24] = 1
            masks[:, 1, 1, :, 24:] = 1
            commands = np.array([[0.1, 0.2, -0.1], [0.3, -0.4, 0.2], [0.4, 0.0, 0.2]], dtype=np.float32)
            proprio = np.arange(9, dtype=np.float32).reshape(3, 3) / 10
            arrays_path = root / "raw-observed.npz"
            np.savez_compressed(
                arrays_path, entity_ids=np.array([11, 22], dtype=np.int64),
                entity_masks=masks, proprio_history=proprio,
                embodiment=np.array([0.1, 0.2], dtype=np.float32),
                observed_action_history=commands,
                observed_action_step_offsets=np.array([-2, -1, 0], dtype=np.int64),
                rgb_step_offsets=np.arange(-8, 1, dtype=np.int64),
                rgb_front=raw_front, rgb_wrist=raw_wrist,
            )
            space = {"representation": "zero-wam-normalized", "normalization_id": "synthetic-vision-fixture-v1",
                     "dimension": 3, "valid_channels": [True, True, True]}
            manifest = {
                "format_version": 1, "kind": "raw_visual_observation", "arrays": arrays_path.name,
                "camera_order": ["front", "wrist"], "robot_size": [32, 32], "demo_size": [32, 32],
                "entity_source": "visual_tracks", "tracker_identity": "synthetic-test-roi-annotations/no-detector",
                "coordinate_frame": "synthetic-test-robot-base", "vae_path": str(weights),
                "vae_sha256": vae_identity, "demonstrations": demonstrations, "demo_fps": 10,
                "chunk_size": 2, "actions_per_frame": 4,
                "action_space": space, "observed_action_space": dict(space),
                "observation_step": 8, "control_dt": 0.05,
                "history_chunks": [
                    {"mode": "video", "slice": [0, 1], "frame_id": 0, "rope_offset": 0},
                    {"mode": "video", "slice": [1, 2], "frame_id": 2, "rope_offset": 1},
                    {"mode": "video", "slice": [2, 3], "frame_id": 4, "rope_offset": 2},
                    {"mode": "action", "slice": [0, 3], "frame_id": 5, "rope_offset": 2},
                ],
            }
            manifest_path = root / "raw.json"
            manifest_path.write_text(json.dumps(manifest))
            report = preprocess(manifest_path, root / "processed", device="cpu", vae=self.vae)
            loaded = load_observation(report["manifest"])
            metadata = json.loads(Path(report["manifest"]).read_text())
            provenance = metadata["visual_provenance"]

            self.assertEqual(metadata["format_version"], 2)
            self.assertEqual(metadata["view_ids"], ["front", "side"])
            self.assertEqual(metadata["demonstration_encoding"], {"kind": "raw_features"})
            self.assertEqual(metadata["demonstration_layouts"], [
                {"frames": 3, "tokens_per_frame": 4, "frame_times": [0., .4, .8]},
                {"frames": 3, "tokens_per_frame": 4, "frame_times": [0., .4, .8]},
            ])
            self.assertTrue(provenance["injected_test_encoder"])
            self.assertFalse(report["released_vae_run"])
            self.assertEqual(report["commands_sent"], 0)
            self.assertEqual(provenance["entity_source"], "visual_tracks")
            self.assertEqual(provenance["tracker_identity"], manifest["tracker_identity"])
            self.assertEqual(provenance["camera_order"], ["front", "wrist"])
            self.assertEqual(provenance["manifest_sha256"], sha256(manifest_path))
            self.assertEqual(provenance["robot_arrays_sha256"], sha256(arrays_path))
            self.assertEqual(provenance["vae_sha256"], vae_identity)
            self.assertEqual(provenance["cache_identity"], report["cache_identity"])
            self.assertEqual(provenance["latent_normalization"], "(posterior_mode-mean)/std")
            for source, record in zip(demonstrations, provenance["demonstrations"]):
                self.assertEqual(record["source_sha256"], sha256(root / source["video"]))
                self.assertEqual(record["frame_indices"], list(range(9)))

            front = encode_rgb(self.vae, raw_front, [32, 32])
            wrist = encode_rgb(self.vae, raw_wrist, [32, 32])
            torch.testing.assert_close(loaded.robot_latent, torch.cat((front, wrist), -1))
            expected_history = torch.stack((front[0, :, :, :, 0].mean(-1).T,
                                            wrist[0, :, :, :, 1].mean(-1).T), 1)[None]
            torch.testing.assert_close(loaded.robot_history, expected_history)
            torch.testing.assert_close(loaded.proprio_history, torch.from_numpy(proprio)[None])
            self.assertEqual(loaded.observed_video_step_offsets.tolist(), [-8, -4, 0])
            self.assertEqual(loaded.observed_action_history.step_offsets.tolist(), [-2, -1, 0])
            self.assertEqual(loaded.observed_action_history.observation_step, 8)
            for latent, tokens in zip((front, wrist), loaded.demonstrations):
                expected_tokens = latent[0].permute(1, 2, 3, 0).reshape(1, -1, 2)
                torch.testing.assert_close(tokens, expected_tokens)

            native = loaded.native_history()
            self.assertEqual([chunk.mode for chunk in native], ["video", "video", "video", "action"])
            self.assertEqual([chunk.frame_id for chunk in native], [0, 2, 4, 5])
            self.assertEqual(native[-1].token_valid.tolist(), [True, True, True, False])
            packed_commands = native[-1].latent.permute(0, 2, 3, 4, 1).reshape(1, 4, 3)
            torch.testing.assert_close(packed_commands[:, :3], torch.from_numpy(commands)[None])
            self.assertFalse(packed_commands[:, 3].any())
            self.assertEqual(loaded.sampling_position(), {"frame_id": 6, "rope_offset": 3})

            # Exercise the real local B artifact loader after one synthetic
            # future-feature update; this is not a trained robotics result.
            from evo_wam.video_effects import EffectFeaturePredictor, VideoEffectEncoder
            with torch.random.fork_rng():
                torch.manual_seed(19)
                effect_encoder = VideoEffectEncoder(2, latent_dim=4, num_tokens=2, hidden_dim=8, noise_std=.2)
                predictor = EffectFeaturePredictor(2, latent_dim=4, hidden_dim=8)
                optimizer = torch.optim.Adam([*effect_encoder.parameters(), *predictor.parameters()], lr=.001)
                feature_window = front.permute(0, 2, 3, 4, 1).flatten(2, 3)
                valid_window = torch.ones_like(feature_window, dtype=torch.bool)
                times = torch.tensor([0., .4, .8])
                z = effect_encoder(feature_window, valid_window, times)
                prediction = predictor(feature_window[:, :1], valid_window[:, :1], z, times[1:], past_times=times[:1])
                update_loss = (prediction["features"] - feature_window[:, 1:]).square().mean()
                update_loss.backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in effect_encoder.parameters()))
                optimizer.step()
            effect_encoder.eval().requires_grad_(False)
            artifact = root / "synthetic-video-encoder.pt"
            payload = {"format_version": 1, "kind": "video_effect_pretrain", "updates": 1,
                       "feature_space_id": "tiny-wan-2channels-v1", "encoder": effect_encoder.state_dict(),
                       "config": {"window_frames": 3, "context_frames": 1,
                                  "model": {"feature_dim": 2, "latent_dim": 4, "num_tokens": 2,
                                            "hidden_dim": 8, "noise_std": .2}}}
            torch.save(payload, artifact)
            encoded_spec = {**manifest, "demo_encoder_artifact": artifact.name,
                            "demo_encoder_sha256": sha256(artifact),
                            "demo_feature_space_id": payload["feature_space_id"]}
            encoded_path = root / "raw-with-effect-encoder.json"
            encoded_path.write_text(json.dumps(encoded_spec))
            encoded_report = preprocess(encoded_path, root / "effect-tokens", device="cpu", vae=self.vae)
            encoded_loaded = load_observation(encoded_report["manifest"])
            encoded_meta = json.loads(Path(encoded_report["manifest"]).read_text())
            self.assertNotIn("demonstration_layouts", encoded_meta)
            self.assertEqual(encoded_meta["demonstration_encoding"], {
                "kind": "video_effect_tokens", "encoder_sha256": sha256(artifact),
                "feature_space_id": payload["feature_space_id"], "token_dim": 4,
                "window_frames": 3, "num_tokens": 2})
            self.assertEqual(encoded_meta["visual_provenance"]["demonstration_window_policy"]["stride"], "nonoverlapping")
            for latent, tokens in zip((front, wrist), encoded_loaded.demonstrations):
                values = latent.permute(0, 2, 3, 4, 1).flatten(2, 3)
                with torch.no_grad():
                    expected_tokens = effect_encoder.encode_demo(values, torch.ones_like(values, dtype=torch.bool), times, window_frames=3)
                torch.testing.assert_close(tokens, expected_tokens)
            for mismatch in ({"demo_encoder_sha256": "0" * 64}, {"demo_feature_space_id": "wrong-space"}):
                encoded_path.write_text(json.dumps({**encoded_spec, **mismatch}))
                with self.subTest(mismatch=mismatch), self.assertRaises(ValueError):
                    preprocess(encoded_path, root / "rejected-artifact", device="cpu", vae=self.vae)
                self.assertFalse((root / "rejected-artifact").exists())


if __name__ == "__main__":
    unittest.main()
