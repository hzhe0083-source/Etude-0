import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from etude.g_pi_data import EventRules, load_g_pi_sample
from etude.icl_preprocess import encode_g_pi_frames
from etude.vision import encode_rgb
from test_g_pi_data import write_g_pi_task


class GPreprocessContractTest(unittest.TestCase):
    def test_invalid_control_grid_is_rejected_before_encoding(self):
        frames = np.zeros((11, 32, 48, 3), dtype=np.uint8)
        gripper = torch.zeros(11, 1)
        for times, stride in ((torch.arange(10).double() * .1, 1),
                              (torch.arange(11).double() * .2, 1),
                              (torch.arange(11).double() * .1, 0)):
            with self.subTest(stride=stride), patch("etude.icl_preprocess.encode_rgb") as encode:
                with self.assertRaises(ValueError):
                    encode_g_pi_frames(None, frames, times, gripper, frame_stride=stride,
                                       control_dt=.1, event_rules=EventRules(), size=[32, 48])
                encode.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("diffusers"), "native Diffusers extras are unavailable")
class NativeGPreprocessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_vision import NativeWanVaeTest
        cls.fixture = NativeWanVaeTest
        cls.fixture.setUpClass()
        cls.vae = cls.fixture.vae

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDownClass()

    def test_single_frame_targets_preserve_tail_and_exact_event_control_step(self):
        frames = self.fixture.frames(self, 19)
        times = np.arange(19, dtype=np.float64) * .1
        gripper = np.array([[0.]] * 10 + [[1.]] * 9, dtype=np.float32)
        with patch("etude.icl_preprocess.encode_rgb", wraps=encode_rgb) as encode:
            arrays, metadata = encode_g_pi_frames(self.vae, frames, times, gripper,
                frame_stride=2, control_dt=.1, event_rules=EventRules(), size=[32, 48])
        self.assertEqual([len(call.args[1]) for call in encode.call_args_list], [9, 1, 1])
        self.assertEqual(metadata["actions_per_frame"], 8)
        self.assertEqual(metadata["subgoal_encoding"], "wan_vae_single_frame")
        np.testing.assert_array_equal(arrays["latent_available_times"], times[[0, 8, 16]])
        np.testing.assert_array_equal(arrays["subgoal_times"], times[[11, 18]])
        torch.testing.assert_close(torch.from_numpy(arrays["latent"]),
                                   encode_rgb(self.vae, frames[:17:2], [32, 48])[0], rtol=0, atol=0)
        for index, control in enumerate((11, 18)):
            expected = encode_rgb(self.vae, frames[control:control + 1], [32, 48])[0]
            torch.testing.assert_close(torch.from_numpy(arrays["subgoal_latents"][index]), expected,
                                       rtol=0, atol=0)
        with TemporaryDirectory() as folder:
            root = Path(folder)
            path, manifest, values = write_g_pi_task(root, gripper=gripper, frame_stride=2)
            values.update(arrays)
            manifest.update(metadata)
            path.write_text(json.dumps(manifest))
            np.savez_compressed(root / manifest["arrays"], **values)
            sample = load_g_pi_sample(path, current_time=.8)
            self.assertEqual(sample.history.shape[2], 2)
            self.assertEqual(sample.subgoal_time, times[11])
            self.assertEqual(sample.goal_poses[0, 0, 0, 3].item(), 11.)
            torch.testing.assert_close(sample.target_frame[0], torch.from_numpy(arrays["subgoal_latents"][0]),
                                       rtol=0, atol=0)

    def test_goal_image_is_independent_of_all_other_video_frames(self):
        frames = self.fixture.frames(self, 11)
        times = np.arange(11, dtype=np.float64) * .1
        gripper = np.array([[0.]] * 5 + [[1.]] * 6, dtype=np.float32)
        options = dict(frame_stride=1, control_dt=.1, event_rules=EventRules(), size=[32, 48])
        first, _ = encode_g_pi_frames(self.vae, frames, times, gripper, **options)
        changed = 255 - frames
        changed[[6, 10]] = frames[[6, 10]]
        second, _ = encode_g_pi_frames(self.vae, changed, times, gripper, **options)
        np.testing.assert_array_equal(first["subgoal_times"], times[[6, 10]])
        np.testing.assert_array_equal(first["subgoal_latents"], second["subgoal_latents"])
        self.assertFalse(np.array_equal(first["latent"], second["latent"]))


if __name__ == "__main__":
    unittest.main()
