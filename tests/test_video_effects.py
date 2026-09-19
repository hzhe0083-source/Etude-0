"""CPU bottleneck checks, not evidence of cross-view or robot task success."""

import inspect
import unittest
from unittest.mock import patch

import torch

from evo_wam.video_effects import (VideoEffectEncoder, EffectFeaturePredictor,
                                    effect_pretraining_loss)


class VideoEffectTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.features = torch.randn(2, 5, 3, 6)
        self.valid = torch.ones_like(self.features, dtype=torch.bool)
        self.times = torch.tensor([0., .1, .3, .5, .9])
        self.encoder = VideoEffectEncoder(6, latent_dim=8, num_tokens=4, hidden_dim=12)
        self.predictor = EffectFeaturePredictor(6, 8, 12, geometry_dim=3, relation_dim=2, event_dim=1)

    def predict(self, tokens=None):
        if tokens is None:
            tokens = self.encoder(self.features, self.valid, self.times, noise=False)
        return self.predictor(self.features[:, :2], self.valid[:, :2], tokens,
                              self.times[2:] - self.times[1], past_times=self.times[:2]), tokens

    def test_shapes_and_gradient_through_only_bottleneck(self):
        prediction, tokens = self.predict()
        self.assertEqual(tokens.shape, (2, 4, 8))
        for key, shape in {"features": (2, 3, 3, 6), "geometry": (2, 3, 3, 3),
                           "relations": (2, 3, 3, 3, 2), "events": (2, 3, 3, 3, 1)}.items():
            self.assertEqual(prediction[key].shape, shape)
        targets = {"geometry": torch.randn_like(prediction["geometry"]),
                   "relations": torch.zeros_like(prediction["relations"]),
                   "events": torch.ones_like(prediction["events"])}
        losses = effect_pretraining_loss(prediction, self.features[:, 2:], self.valid[:, 2:],
            targets, {key: torch.ones_like(value, dtype=torch.bool) for key, value in targets.items()}, tokens)
        self.assertGreater(losses["valid_count"].item(), 0)
        losses["total"].backward()
        self.assertGreater(self.encoder.queries.grad.abs().sum().item(), 0)
        self.assertGreater(self.predictor.token_values.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(parameter.grad).all()
                            for model in (self.encoder, self.predictor)
                            for parameter in model.parameters() if parameter.grad is not None))

    def test_predictor_has_no_direct_process_or_future_input(self):
        signature = set(inspect.signature(self.predictor.forward).parameters)
        self.assertEqual(signature, {"past", "past_valid", "tokens", "query_times", "past_times", "effect_fields"})
        prediction, tokens = self.predict()
        with self.assertRaises(TypeError):
            self.predictor(self.features[:, :2], self.valid[:, :2], tokens, self.times[2:], future=self.features[:, 2:])
        # Fix z while changing the held-out future used only as a loss target.
        other_future = self.features[:, 2:] + 100
        effect_pretraining_loss(prediction, other_future, self.valid[:, 2:], {}, {}, tokens)
        again, _ = self.predict(tokens)
        for key in prediction:
            torch.testing.assert_close(prediction[key], again[key], rtol=0, atol=0)
        changed, _ = self.predict(tokens + .7)
        self.assertFalse(torch.allclose(prediction["features"], changed["features"]))

    def test_frozen_targets_receive_no_gradient(self):
        prediction, tokens = self.predict()
        target = self.features[:, 2:].clone().requires_grad_()
        geometry = torch.randn_like(prediction["geometry"], requires_grad=True)
        losses = effect_pretraining_loss(prediction, target, self.valid[:, 2:], {"geometry": geometry},
                                         {"geometry": torch.ones_like(geometry, dtype=torch.bool)}, tokens)
        losses["total"].backward()
        self.assertIsNone(target.grad)
        self.assertIsNone(geometry.grad)

    def test_process_order_matters_even_with_equal_endpoints(self):
        # The target also needs intermediate times: input order alone does not
        # make terminal-only training distinguish an excursion from doing nothing.
        shuffled = self.features.clone()
        shuffled[:, 1:4] = shuffled[:, 1:4].flip(1)
        first = self.encoder(self.features, self.valid, self.times, noise=False)
        second = self.encoder(shuffled, self.valid, self.times, noise=False)
        self.assertFalse(torch.allclose(first, second))
        with self.assertRaisesRegex(ValueError, "at least two"):
            self.predictor(self.features[:, :2], self.valid[:, :2], first, torch.tensor([.8]))
        prediction, _ = self.predict(first)
        target = self.features[:, 2:].clone()
        original = effect_pretraining_loss(prediction, target, self.valid[:, 2:], {}, {}, first)["features"]
        target[:, 0] += 10  # Terminal target remains unchanged.
        changed = effect_pretraining_loss(prediction, target, self.valid[:, 2:], {}, {}, first)["features"]
        self.assertFalse(torch.equal(original, changed))

    def test_input_masks_cannot_leak_invalid_values(self):
        valid = self.valid.clone()
        valid[:, 2, 0, :] = False
        first = self.encoder(self.features, valid, self.times, noise=False)
        hidden = self.features.clone()
        hidden[:, 2, 0, :] = float("nan")
        second = self.encoder(hidden, valid, self.times, noise=False)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            self.encoder(hidden, self.valid, self.times, noise=False)
        with self.assertRaises(ValueError):
            self.encoder(hidden, torch.zeros_like(valid), self.times, noise=False)

    def test_missing_targets_are_masked_before_arithmetic(self):
        prediction, tokens = self.predict()
        target = self.features[:, 2:].clone()
        valid = self.valid[:, 2:].clone()
        target[0, 0, 0, 0] = float("nan")
        valid[0, 0, 0, 0] = False
        effect = torch.full_like(prediction["relations"], float("nan"))
        loss = effect_pretraining_loss(prediction, target, valid, {"relations": effect},
            {"relations": torch.zeros_like(effect, dtype=torch.bool)}, tokens)
        self.assertTrue(torch.isfinite(loss["total"]))
        self.assertEqual(loss["relations"].item(), 0)
        loss["total"].backward()
        self.assertTrue(torch.isfinite(self.encoder.features.weight.grad).all())

    def test_empty_or_zero_weight_supervision_never_updates_from_capacity_alone(self):
        prediction, tokens = self.predict()
        for mask, weights in ((torch.zeros_like(self.valid[:, 2:]), {"capacity": 100}),
                              (self.valid[:, 2:], {"features": 0, "capacity": 100})):
            losses = effect_pretraining_loss(prediction, self.features[:, 2:], mask, {}, {}, tokens, weights)
            self.assertEqual(losses["valid_count"].item(), 0)
            self.assertEqual(losses["capacity"].item(), 0)
            self.assertEqual(losses["total"].item(), 0)
            self.assertFalse(losses["total"].requires_grad)

    def test_same_weights_work_without_action_or_relation_labels(self):
        prediction, tokens = self.predict()
        losses = effect_pretraining_loss(prediction, self.features[:, 2:], self.valid[:, 2:], None, None, tokens)
        self.assertGreater(losses["features"].item(), 0)
        self.assertEqual(losses["geometry"].item(), 0)
        losses["total"].backward()
        self.assertIsNone(self.predictor.geometry_head.weight.grad)
        self.assertGreater(self.encoder.output[-1].weight.grad.abs().sum().item(), 0)

    def test_entity_permutation_is_preserved_without_entity_id_features(self):
        order = torch.tensor([2, 0, 1])
        tokens = self.encoder(self.features, self.valid, self.times, noise=False)
        permuted_tokens = self.encoder(self.features[:, :, order], self.valid[:, :, order], self.times, noise=False)
        torch.testing.assert_close(tokens, permuted_tokens)
        first = self.predict(tokens)[0]
        second = self.predictor(self.features[:, :2, order], self.valid[:, :2, order], permuted_tokens,
                                self.times[2:] - self.times[1], past_times=self.times[:2])
        for name in ("features", "geometry"):
            torch.testing.assert_close(first[name][:, :, order], second[name])
        for name in ("relations", "events"):
            torch.testing.assert_close(first[name][:, :, order][:, :, :, order], second[name])

    def test_past_supplies_last_observed_appearance(self):
        with torch.no_grad():
            self.predictor.feature_head.weight.zero_()
            self.predictor.feature_head.bias.zero_()
        past, valid = self.features[:, :2].clone(), self.valid[:, :2].clone()
        valid[:, 1, 0, 0] = False
        past[:, 1, 0, 0] = float("nan")
        tokens = self.encoder(self.features, self.valid, self.times, noise=False)
        prediction = self.predictor(past, valid, tokens, torch.tensor([.2, .5]))
        torch.testing.assert_close(prediction["features"][:, :, 0, 0], past[:, :1, 0, 0].expand(-1, 2))
        torch.testing.assert_close(prediction["features"][:, :, 1, 1], past[:, 1:, 1, 1].expand(-1, 2))

    def test_noise_is_training_only_and_demo_encoding_preserves_tail(self):
        self.encoder.train()
        first = self.encoder(self.features, self.valid, self.times)
        second = self.encoder(self.features, self.valid, self.times)
        self.assertFalse(torch.equal(first, second))
        self.assertTrue((first.abs() <= 1).all())
        self.encoder.eval()
        torch.testing.assert_close(self.encoder(self.features, self.valid, self.times),
                                   self.encoder(self.features, self.valid, self.times), rtol=0, atol=0)
        features = torch.cat((self.features, self.features[:, :2] + 2), 1)
        valid = torch.ones_like(features, dtype=torch.bool)
        times = torch.arange(7).float()
        self.encoder.train()  # encode_demo must still be deterministic.
        encoded = self.encoder.encode_demo(features, valid, times, window_frames=5)
        self.assertEqual(encoded.shape, (2, 8, 8))
        tail = self.encoder(features[:, 5:], valid[:, 5:], times[5:], noise=False)
        torch.testing.assert_close(encoded[:, 4:], tail)
        torch.testing.assert_close(encoded[:, :4], self.encoder(features[:, :5], valid[:, :5], times[:5], noise=False))
        torch.testing.assert_close(encoded, self.encoder.encode_demo(features, valid, times, 5), rtol=0, atol=0)
        changed = features.clone()
        changed[:, -1] += 10
        self.assertFalse(torch.allclose(encoded[:, 4:], self.encoder.encode_demo(changed, valid, times, 5)[:, 4:]))

    def test_feature_only_configuration_and_validation(self):
        minimal = EffectFeaturePredictor(6, 8, 12)
        tokens = self.encoder(self.features, self.valid, self.times, noise=False)
        result = minimal(self.features[:, :2], self.valid[:, :2], tokens, torch.tensor([1., 2.]))
        self.assertEqual(result["geometry"].shape[-1], 0)
        self.assertEqual(result["relations"].shape[-1], 0)
        self.assertEqual(result["events"].shape[-1], 0)
        with self.assertRaises(ValueError):
            self.encoder(self.features, self.valid, self.times.flip(0))
        with self.assertRaises(ValueError):
            minimal(self.features[:, :2], self.valid[:, :2], tokens, torch.tensor([0., 2.]))
        with self.assertRaises(ValueError):
            effect_pretraining_loss(result, self.features[:, 2:4], self.valid[:, 2:4], {}, {}, tokens, {"capacity": -1})

    def test_patch_features_without_effect_labels_never_construct_pairs(self):
        past = torch.randn(1, 2, 2048, 6)
        valid = torch.ones_like(past, dtype=torch.bool)
        tokens = torch.zeros(1, 4, 8)
        with patch("evo_wam.video_effects._pair_features", side_effect=AssertionError("quadratic pair allocation")):
            output = self.predictor(past, valid, tokens, torch.tensor([1., 2.]), effect_fields=())
        self.assertEqual(output["features"].shape, (1, 2, 2048, 6))
        self.assertEqual(output["geometry"].numel(), 0)
        self.assertEqual(output["relations"].shape, (1, 2, 2048, 2048, 0))
        self.assertEqual(output["relations"].numel(), 0)
        self.assertEqual(output["events"].numel(), 0)


if __name__ == "__main__":
    unittest.main()
