"""CPU bottleneck checks, not evidence of cross-view or robot task success."""

import inspect
import unittest
from unittest.mock import patch

import torch

from evo_wam.video_effects import (VideoEffectEncoder, EffectFeaturePredictor,
                                    effect_pretraining_loss)


class VideoEffectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(23)
        self.features = torch.randn(2, 5, 3, 6)
        self.valid = torch.ones_like(self.features, dtype=torch.bool)
        self.times = torch.tensor([0., .1, .3, .5, .9])
        self.encoder = VideoEffectEncoder(6, latent_dim=8, num_tokens=4, hidden_dim=12)
        self.predictor = EffectFeaturePredictor(6, 8, 12, geometry_dim=3, relation_dim=2, event_dim=1)

    def predict(self, tokens=None):
        if tokens is None:
            tokens = self.encoder(self.features, self.valid, self.times, noise=False, feature_kind="tracked_entities")
        return self.predictor(self.features[:, :2], self.valid[:, :2], tokens,
                              self.times[2:] - self.times[1], past_times=self.times[:2], feature_kind="tracked_entities"), tokens

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
        self.assertEqual(signature, {"past", "past_valid", "tokens", "query_times", "past_times", "effect_fields",
                                     "feature_kind", "patch_coordinates"})
        prediction, tokens = self.predict()
        with self.assertRaises(TypeError):
            self.predictor(self.features[:, :2], self.valid[:, :2], tokens, self.times[2:],
                           future=self.features[:, 2:], feature_kind="tracked_entities")
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
        first = self.encoder(self.features, self.valid, self.times, noise=False, feature_kind="tracked_entities")
        second = self.encoder(shuffled, self.valid, self.times, noise=False, feature_kind="tracked_entities")
        self.assertFalse(torch.allclose(first, second))
        with self.assertRaisesRegex(ValueError, "at least two"):
            self.predictor(self.features[:, :2], self.valid[:, :2], first, torch.tensor([.8]), feature_kind="tracked_entities")
        prediction, _ = self.predict(first)
        target = self.features[:, 2:].clone()
        original = effect_pretraining_loss(prediction, target, self.valid[:, 2:], {}, {}, first)["features"]
        target[:, 0] += 10  # Terminal target remains unchanged.
        changed = effect_pretraining_loss(prediction, target, self.valid[:, 2:], {}, {}, first)["features"]
        self.assertFalse(torch.equal(original, changed))

    def test_input_masks_cannot_leak_invalid_values(self):
        valid = self.valid.clone()
        valid[:, 2, 0, :] = False
        first = self.encoder(self.features, valid, self.times, noise=False, feature_kind="tracked_entities")
        hidden = self.features.clone()
        hidden[:, 2, 0, :] = float("nan")
        second = self.encoder(hidden, valid, self.times, noise=False, feature_kind="tracked_entities")
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            self.encoder(hidden, self.valid, self.times, noise=False, feature_kind="tracked_entities")
        with self.assertRaises(ValueError):
            self.encoder(hidden, torch.zeros_like(valid), self.times, noise=False, feature_kind="tracked_entities")

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
        tokens = self.encoder(self.features, self.valid, self.times, noise=False, feature_kind="tracked_entities")
        permuted_tokens = self.encoder(self.features[:, :, order], self.valid[:, :, order], self.times,
                                       noise=False, feature_kind="tracked_entities")
        torch.testing.assert_close(tokens, permuted_tokens)
        first = self.predict(tokens)[0]
        second = self.predictor(self.features[:, :2, order], self.valid[:, :2, order], permuted_tokens,
                                self.times[2:] - self.times[1], past_times=self.times[:2], feature_kind="tracked_entities")
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
        tokens = self.encoder(self.features, self.valid, self.times, noise=False, feature_kind="tracked_entities")
        prediction = self.predictor(past, valid, tokens, torch.tensor([.2, .5]), feature_kind="tracked_entities")
        torch.testing.assert_close(prediction["features"][:, :, 0, 0], past[:, :1, 0, 0].expand(-1, 2))
        torch.testing.assert_close(prediction["features"][:, :, 1, 1], past[:, 1:, 1, 1].expand(-1, 2))

    def test_noise_is_training_only_and_demo_encoding_preserves_tail(self):
        self.encoder.train()
        first = self.encoder(self.features, self.valid, self.times, feature_kind="tracked_entities")
        second = self.encoder(self.features, self.valid, self.times, feature_kind="tracked_entities")
        self.assertFalse(torch.equal(first, second))
        self.assertTrue((first.abs() <= 1).all())
        self.encoder.eval()
        torch.testing.assert_close(self.encoder(self.features, self.valid, self.times, feature_kind="tracked_entities"),
                                   self.encoder(self.features, self.valid, self.times, feature_kind="tracked_entities"), rtol=0, atol=0)
        features = torch.cat((self.features, self.features[:, :2] + 2), 1)
        valid = torch.ones_like(features, dtype=torch.bool)
        times = torch.arange(7).float()
        self.encoder.train()  # encode_demo must still be deterministic.
        encoded = self.encoder.encode_demo(features, valid, times, window_frames=5, feature_kind="tracked_entities")
        self.assertEqual(encoded.shape, (2, 8, 8))
        tail = self.encoder(features[:, 5:], valid[:, 5:], times[5:], noise=False, feature_kind="tracked_entities")
        torch.testing.assert_close(encoded[:, 4:], tail)
        torch.testing.assert_close(encoded[:, :4], self.encoder(features[:, :5], valid[:, :5], times[:5],
                                                               noise=False, feature_kind="tracked_entities"))
        torch.testing.assert_close(encoded, self.encoder.encode_demo(features, valid, times, 5,
                                                                    feature_kind="tracked_entities"), rtol=0, atol=0)
        changed = features.clone()
        changed[:, -1] += 10
        self.assertFalse(torch.allclose(encoded[:, 4:], self.encoder.encode_demo(changed, valid, times, 5,
                                                                              feature_kind="tracked_entities")[:, 4:]))

    def test_feature_only_configuration_and_validation(self):
        minimal = EffectFeaturePredictor(6, 8, 12)
        tokens = self.encoder(self.features, self.valid, self.times, noise=False, feature_kind="tracked_entities")
        result = minimal(self.features[:, :2], self.valid[:, :2], tokens, torch.tensor([1., 2.]), feature_kind="tracked_entities")
        self.assertEqual(result["geometry"].shape[-1], 0)
        self.assertEqual(result["relations"].shape[-1], 0)
        self.assertEqual(result["events"].shape[-1], 0)
        with self.assertRaises(ValueError):
            self.encoder(self.features, self.valid, self.times.flip(0), feature_kind="tracked_entities")
        with self.assertRaises(ValueError):
            minimal(self.features[:, :2], self.valid[:, :2], tokens, torch.tensor([0., 2.]), feature_kind="tracked_entities")
        with self.assertRaises(ValueError):
            effect_pretraining_loss(result, self.features[:, 2:4], self.valid[:, 2:4], {}, {}, tokens, {"capacity": -1})

    def test_patch_features_without_effect_labels_never_construct_pairs(self):
        past = torch.randn(1, 2, 2048, 6)
        valid = torch.ones_like(past, dtype=torch.bool)
        tokens = torch.zeros(1, 4, 8)
        with patch("evo_wam.video_effects._pair_features", side_effect=AssertionError("quadratic pair allocation")):
            # Explicit 32 x 64 pixel-center grid, never inferred from N.
            y, x = torch.meshgrid((torch.arange(32) + .5) / 16 - 1,
                                  (torch.arange(64) + .5) / 32 - 1, indexing="ij")
            coordinates = torch.stack((x.flatten(), y.flatten()), -1)
            output = self.predictor(past, valid, tokens, torch.tensor([1., 2.]), effect_fields=(),
                                    feature_kind="patches", patch_coordinates=coordinates)
        self.assertEqual(output["features"].shape, (1, 2, 2048, 6))
        self.assertEqual(output["geometry"].numel(), 0)
        self.assertEqual(output["relations"].shape, (1, 2, 2048, 2048, 0))
        self.assertEqual(output["relations"].numel(), 0)
        self.assertEqual(output["events"].numel(), 0)

    def test_patch_swaps_and_equal_endpoint_excursions_fit_different_futures(self):
        # All three clips have the SAME past and the same per-frame feature set.
        # Static and excursion also have equal endpoints. A per-frame set pool
        # cannot fit these targets, regardless of how long it is trained.
        features = torch.tensor([
            [[-1., 1.], [-1., 1.], [-1., 1.]],
            [[-1., 1.], [1., -1.], [1., -1.]],
            [[-1., 1.], [1., -1.], [-1., 1.]],
        ])[..., None]
        valid = torch.ones_like(features, dtype=torch.bool)
        times = torch.arange(3).float()
        layout = {"feature_kind": "patches", "patch_coordinates": torch.tensor([[-.5, 0.], [.5, 0.]])}
        encoder = VideoEffectEncoder(1, latent_dim=8, num_tokens=2, hidden_dim=32, noise_std=0)
        predictor = EffectFeaturePredictor(1, latent_dim=8, hidden_dim=32)
        optimizer = torch.optim.Adam([*encoder.parameters(), *predictor.parameters()], lr=.01)
        for _ in range(500):
            tokens = encoder(features, valid, times, **layout)
            prediction = predictor(features[:, :1], valid[:, :1], tokens, times[1:], **layout)
            loss = effect_pretraining_loss(prediction, features[:, 1:], valid[:, 1:], {}, {}, tokens,
                                           {"capacity": 0})["total"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            tokens = encoder(features, valid, times, **layout)
            predicted = predictor(features[:, :1], valid[:, :1], tokens, times[1:], **layout)["features"]
            torch.testing.assert_close(predicted, features[:, 1:], rtol=0, atol=.05)
            # Swapping only z must swap the predicted operation with identical past.
            order = torch.tensor([1, 2, 0])
            swapped = predictor(features[:, :1], valid[:, :1], tokens[order], times[1:], **layout)["features"]
            torch.testing.assert_close(swapped, features[order, 1:], rtol=0, atol=.05)

    def test_predictor_positions_distinguish_identical_past_slots(self):
        torch.manual_seed(23)
        past = torch.zeros(1, 1, 2, 1)
        valid = torch.ones_like(past, dtype=torch.bool)
        tokens = torch.zeros(1, 2, 4)
        target = torch.tensor([[[[-1.], [1.]], [[-.5], [.5]]]])
        layout = {"feature_kind": "patches", "patch_coordinates": torch.tensor([[-.5, 0.], [.5, 0.]])}
        predictor = EffectFeaturePredictor(1, latent_dim=4, hidden_dim=16)
        optimizer = torch.optim.Adam(predictor.parameters(), lr=.01)
        for _ in range(200):
            predicted = predictor(past, valid, tokens, torch.tensor([1., 2.]), **layout)["features"]
            loss = (predicted - target).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            predicted = predictor(past, valid, tokens, torch.tensor([1., 2.]), **layout)["features"]
            torch.testing.assert_close(predicted, target, rtol=0, atol=.03)

    def test_patch_storage_permutation_preserves_tokens_and_reorders_predictions(self):
        coordinates = torch.tensor([[-2 / 3, 0.], [0., 0.], [2 / 3, 0.]])
        valid = self.valid.clone()
        valid[:, 0, 1] = False
        valid[:, 2, 2, :3] = False
        order = torch.tensor([2, 0, 1])
        tokens = self.encoder(self.features, valid, self.times, noise=False,
                              feature_kind="patches", patch_coordinates=coordinates)
        reordered = self.encoder(self.features[:, :, order], valid[:, :, order], self.times, noise=False,
                                 feature_kind="patches", patch_coordinates=coordinates[order])
        torch.testing.assert_close(tokens, reordered)
        first = self.predictor(self.features[:, :2], valid[:, :2], tokens, self.times[2:] - self.times[1],
                               past_times=self.times[:2], feature_kind="patches", patch_coordinates=coordinates)
        second = self.predictor(self.features[:, :2, order], valid[:, :2, order], reordered,
                                self.times[2:] - self.times[1], past_times=self.times[:2],
                                feature_kind="patches", patch_coordinates=coordinates[order])
        for name in ("features", "geometry"):
            torch.testing.assert_close(first[name][:, :, order], second[name])
        for name in ("relations", "events"):
            torch.testing.assert_close(first[name][:, :, order][:, :, :, order], second[name])
        demos = self.encoder.encode_demo(self.features, valid, self.times, 3,
                                        feature_kind="patches", patch_coordinates=coordinates)
        reordered_demos = self.encoder.encode_demo(self.features[:, :, order], valid[:, :, order], self.times, 3,
                                                  feature_kind="patches", patch_coordinates=coordinates[order])
        torch.testing.assert_close(demos, reordered_demos)

    def test_tracked_entity_time_correspondence_survives_set_pooling(self):
        static = torch.tensor([[[[-1.], [1.]], [[-1.], [1.]], [[-1.], [1.]]]])
        changed = static.clone()
        changed[:, 1:] = changed[:, 1:].flip(2)
        valid = torch.ones_like(static, dtype=torch.bool)
        times = torch.arange(3).float()
        encoder = VideoEffectEncoder(1, latent_dim=8, num_tokens=2, hidden_dim=16, noise_std=0)
        first = encoder(static, valid, times, feature_kind="tracked_entities")
        second = encoder(changed, valid, times, feature_kind="tracked_entities")
        self.assertFalse(torch.allclose(first, second))
        permuted = encoder(changed.flip(2), valid.flip(2), times, feature_kind="tracked_entities")
        torch.testing.assert_close(second, permuted)

    def test_patch_coordinates_are_required_and_cannot_be_entity_ids(self):
        coordinates = torch.tensor([[-2 / 3, 0.], [0., 0.], [2 / 3, 0.]])
        tokens = torch.zeros(2, 4, 8)
        malformed = (None, coordinates[:2], torch.zeros_like(coordinates), coordinates.long(),
                     coordinates * float("nan"), coordinates * float("inf"),
                     torch.tensor([[-1., 0.], [0., 0.], [1., 0.]]))
        for value in malformed:
            with self.subTest(coordinates=value):
                with self.assertRaisesRegex(ValueError, "patches require"):
                    self.encoder(self.features, self.valid, self.times,
                                 feature_kind="patches", patch_coordinates=value)
                with self.assertRaisesRegex(ValueError, "patches require"):
                    self.predictor(self.features[:, :2], self.valid[:, :2], tokens, torch.tensor([1., 2.]),
                                   feature_kind="patches", patch_coordinates=value)
        for feature_kind, message in (("tracked_entities", "must not use patch coordinates"),
                                      ("unknown", "feature_kind must be")):
            with self.assertRaisesRegex(ValueError, message):
                self.encoder(self.features, self.valid, self.times,
                             feature_kind=feature_kind, patch_coordinates=coordinates)
            with self.assertRaisesRegex(ValueError, message):
                self.predictor(self.features[:, :2], self.valid[:, :2], tokens, torch.tensor([1., 2.]),
                               feature_kind=feature_kind, patch_coordinates=coordinates)


if __name__ == "__main__":
    unittest.main()
