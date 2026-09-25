"""Structural/learnability checks; these do not establish robot transfer."""

import unittest

import torch
from torch import nn

from etude.demo_bottleneck import TemporalDemoBottleneck


class TemporalDemoBottleneckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(31)
        self.model = TemporalDemoBottleneck(6, dim=16, num_heads=4,
                                             group_frames=2, tokens_per_group=2)
        self.features = torch.randn(2, 5, 3, 6)
        self.times = torch.tensor([0., .2, .7, 1.5, 3.1], dtype=torch.float64)
        self.coordinates = torch.tensor([[-.6, 0.], [0., 0.], [.6, 0.]])

    def test_ordered_groups_actual_timestamps_and_short_tail_gradients(self):
        features = self.features.clone().requires_grad_()
        tokens, times = self.model(features, self.times, self.coordinates)
        self.assertEqual(tokens.shape, (2, 6, 16))
        torch.testing.assert_close(times, torch.tensor([.1, 1.1, 3.1], dtype=torch.float64))
        context = self.model.adapter(tokens)
        self.assertEqual(context.shape, (2, 6, 6))
        context.square().mean().backward()
        self.assertGreater(features.grad[:, -1].abs().sum().item(), 0)
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        one, one_time = self.model(self.features[:, :1], self.times[:1], self.coordinates)
        self.assertEqual(one.shape, (2, 2, 16))
        torch.testing.assert_close(one_time, self.times[:1])
        # More padded positions must not dilute the same one-frame group.
        larger_group = TemporalDemoBottleneck(6, dim=16, num_heads=4,
                                              group_frames=4, tokens_per_group=2)
        larger_group.load_state_dict(self.model.state_dict())
        padded, _ = larger_group(self.features[:, :1], self.times[:1], self.coordinates)
        torch.testing.assert_close(one, padded, atol=2e-6, rtol=2e-6)

    def test_complete_patch_storage_permutation_preserves_context(self):
        order = torch.tensor([2, 0, 1])
        tokens, times = self.model(self.features, self.times, self.coordinates)
        reordered, reordered_times = self.model(self.features[:, :, order], self.times,
                                                self.coordinates[order])
        torch.testing.assert_close(tokens, reordered, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(times, reordered_times)
        torch.testing.assert_close(self.model.adapter(tokens), self.model.adapter(reordered),
                                   atol=2e-6, rtol=2e-6)

    def test_elapsed_seconds_change_encoding_without_random_training_noise(self):
        tokens, times = self.model(self.features, self.times, self.coordinates)
        repeated, _ = self.model(self.features, self.times, self.coordinates)
        torch.testing.assert_close(tokens, repeated, rtol=0, atol=0)
        stretched, stretched_times = self.model(self.features, self.times * 3, self.coordinates)
        self.assertFalse(torch.allclose(tokens, stretched))
        torch.testing.assert_close(stretched_times, times * 3)
        shifted, shifted_times = self.model(self.features, self.times + 100, self.coordinates)
        torch.testing.assert_close(tokens, shifted)
        torch.testing.assert_close(shifted_times, times + 100)

    def test_same_past_swaps_and_equal_endpoints_different_processes_can_be_fit(self):
        # Every frame contains the same set of two values. Three clips also
        # share endpoints; two exchange only their intermediate event ordering.
        # A set/temporal mean cannot fit all four futures from this common past.
        features = torch.tensor([
            [[-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.]],
            [[-1., 1.], [1., -1.], [1., -1.], [1., -1.], [1., -1.]],
            [[-1., 1.], [1., -1.], [-1., 1.], [-1., 1.], [-1., 1.]],
            [[-1., 1.], [-1., 1.], [1., -1.], [-1., 1.], [-1., 1.]],
        ])[..., None]
        times = torch.tensor([0., .4, 1.2, 2.3, 3.1])
        xy = torch.tensor([[-.5, 0.], [.5, 0.]])
        model = TemporalDemoBottleneck(1, dim=16, num_heads=4, group_frames=4, tokens_per_group=2)
        readout = nn.Linear(4 * 16, 8)
        optimizer = torch.optim.Adam([*model.parameters(), *readout.parameters()], lr=.006)
        target = features[:, 1:].flatten(1)
        for _ in range(350):
            tokens, _ = model(features, times, xy)
            prediction = readout(tokens.flatten(1))
            loss = (prediction - target).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            tokens, _ = model(features, times, xy)
            torch.testing.assert_close(readout(tokens.flatten(1)), target, atol=.04, rtol=0)
            # The readout sees only compressed tokens; swapping them changes
            # the predicted operation despite identical initial observations.
            order = torch.tensor([3, 2, 0, 1])
            torch.testing.assert_close(readout(tokens[order].flatten(1)), target[order], atol=.04, rtol=0)

    def test_constructor_and_input_validation(self):
        for key in ("input_dim", "dim", "num_heads", "group_frames", "tokens_per_group", "layers"):
            for value in (0, -1, True, 1.5):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    TemporalDemoBottleneck(**{"input_dim": 6, key: value})
        with self.assertRaisesRegex(ValueError, "divisible"):
            TemporalDemoBottleneck(6, dim=17, num_heads=4)
        for features in (self.features[:, :0], self.features[..., :5], self.features.long(),
                         self.features * float("nan")):
            with self.assertRaisesRegex(ValueError, "features"):
                self.model(features, self.times, self.coordinates)
        for times in (self.times[:4], self.times.flip(0), self.times * 0,
                      self.times * float("nan"), self.times.long()):
            with self.assertRaisesRegex(ValueError, "frame_times"):
                self.model(self.features, times, self.coordinates)
        for coordinates in (self.coordinates[:2], self.coordinates * 0, self.coordinates.long(),
                            self.coordinates * float("inf"), self.coordinates * 2):
            with self.assertRaisesRegex(ValueError, "patch_coordinates"):
                self.model(self.features, self.times, coordinates)


if __name__ == "__main__":
    unittest.main()
