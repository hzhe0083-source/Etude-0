import copy
import inspect
import unittest

import torch

from evo_wam.contracts import EffectRequirement, PhysicalOutcome
from evo_wam.models import (
    CausalEffectPredictor, EffectReader, PhysicalPrediction, RequirementCodec,
    TemporalInteractionHead, decoded_requirement_loss, effect_cost,
    physical_prediction_loss,
)


def requirement(batch=2):
    values = {
        "geometry": torch.randn(batch, 2, 2, 3),
        "relations": torch.randint(0, 2, (batch, 2, 2, 2, 2)).float(),
        "events": torch.randint(0, 2, (batch, 2, 2, 2, 1)).float(),
    }
    masks = {name: torch.ones_like(value, dtype=torch.bool) for name, value in values.items()}
    return EffectRequirement(
        torch.tensor([[13, 8, 91]]).expand(batch, -1).clone(), torch.tensor([1, 4]),
        torch.tensor([[0, 2]]).expand(batch, -1).clone(), **values,
        requirement_mask=copy.deepcopy(masks),
        label_valid={**masks, "binding": torch.ones(batch, 2, dtype=torch.bool)},
    ).validate()


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)

    def test_codec_shapes_gradient_and_entity_permutation(self):
        req = requirement()
        entities = torch.randn(2, 3, 5)
        codec = RequirementCodec(5, 3, 2, 1, 2, token_dim=12)
        encoded = codec.encode(req, entities)
        self.assertEqual(encoded.shape, (2, 4, 12))
        decoded = codec.decode(encoded, entities, req.step_offsets, req.entity_ids)
        loss = decoded_requirement_loss(decoded, req)
        loss.backward()
        self.assertGreater(codec.encoder[0].weight.grad.abs().sum().item(), 0)
        order = torch.tensor([2, 0, 1])
        permuted = req.permute_entities(order)
        encoded_p = codec.encode(permuted, entities[:, order])
        torch.testing.assert_close(encoded, encoded_p)
        decoded_p = codec.decode(encoded_p, entities[:, order], req.step_offsets, permuted.entity_ids)
        torch.testing.assert_close(decoded.binding_logits[:, :, order], decoded_p.binding_logits)
        torch.testing.assert_close(decoded.geometry, decoded_p.geometry)

    def test_geometry_control_cannot_read_relation_labels_or_masks(self):
        first, entities = requirement(), torch.randn(2, 3, 5)
        second = copy.deepcopy(first)
        for name in ("relations", "events"):
            setattr(second, name, 1 - getattr(second, name))
            second.requirement_mask[name].zero_()
            second.label_valid[name].zero_()
        codec = RequirementCodec(5, 3, 2, 1, 2, 12, interface="geometry")
        torch.testing.assert_close(codec.encode(first, entities), codec.encode(second, entities), rtol=0, atol=0)
        decoded = codec(first, entities)
        torch.testing.assert_close(decoded_requirement_loss(decoded, first, "geometry"),
                                   decoded_requirement_loss(decoded, second, "geometry"))
        full = RequirementCodec(5, 3, 2, 1, 2, 12)
        full.load_state_dict(codec.state_dict())
        self.assertFalse(torch.allclose(full.encode(first, entities), full.encode(second, entities)))

    def test_predicted_mask_does_not_turn_off_supervision(self):
        req, entities = requirement(), torch.randn(2, 3, 5)
        codec = RequirementCodec(5, 3, 2, 1, 2, 12)
        decoded = codec(req, entities)
        loss_before = decoded_requirement_loss(decoded, req)
        for logits in decoded.requirement_mask_logits.values():
            logits.data.fill_(-100)
        loss_after = decoded_requirement_loss(decoded, req)
        self.assertGreater(loss_after.item(), loss_before.item())

    def test_annotation_coverage_is_not_a_goal_token_feature(self):
        first, entities = requirement(), torch.randn(2, 3, 5)
        first.requirement_mask["geometry"][..., 0] = False
        second = copy.deepcopy(first)
        second.label_valid["geometry"][..., 0] = False
        second.geometry[..., 0] = float("nan")
        codec = RequirementCodec(5, 3, 2, 1, 2, 12)
        torch.testing.assert_close(codec.encode(first, entities), codec.encode(second, entities), rtol=0, atol=0)

    def test_geometry_materialization_never_enables_unsupervised_control_requirements(self):
        req, entities = requirement(), torch.randn(2, 3, 5)
        codec = RequirementCodec(5, 3, 2, 1, 2, 12, interface="geometry")
        decoded = codec(req, entities)
        for logits in decoded.requirement_mask_logits.values():
            logits.data.fill_(100)
        geometry = decoded.materialize(interface="geometry")
        self.assertTrue(geometry.requirement_mask["geometry"].all())
        self.assertFalse(geometry.requirement_mask["relations"].any())
        self.assertFalse(geometry.requirement_mask["events"].any())
        self.assertTrue(decoded.materialize().requirement_mask["relations"].any())

    def test_unknown_label_is_not_nan_input(self):
        req, entities = requirement(), torch.randn(2, 3, 5)
        req.geometry[0, 0, 0, 0] = float("nan")
        req.label_valid["geometry"][0, 0, 0, 0] = False
        codec = RequirementCodec(5, 3, 2, 1, 2, 12)
        with self.assertRaises(ValueError):
            codec.encode(req, entities)
        req.requirement_mask["geometry"][0, 0, 0, 0] = False
        loss = decoded_requirement_loss(codec(req, entities), req)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all()
                            for parameter in codec.parameters() if parameter.grad is not None))

    def test_reader_current_remaining_and_task_response(self):
        reader = EffectReader(6, 5, 2, 2, roles=2, token_dim=12)
        demo = torch.randn(2, 3, 6)
        history, proprio, embodiment = torch.randn(2, 4, 3, 5), torch.randn(2, 4, 2), torch.randn(2, 2)
        first = reader(demo, history, proprio, embodiment, torch.tensor([1]), torch.tensor([2, 4]))
        self.assertEqual(first.current.shape, (2, 2, 12))
        self.assertEqual(first.remaining.shape, (2, 4, 12))
        permuted = reader(demo, history[:, :, [2, 0, 1]], proprio, embodiment, torch.tensor([1]), torch.tensor([2, 4]))
        torch.testing.assert_close(first.current, permuted.current)
        second = reader(demo + 1, history, proprio, embodiment, torch.tensor([1]), torch.tensor([2, 4]))
        self.assertFalse(torch.allclose(first.remaining, second.remaining))
        reordered = reader(demo.flip(1), history, proprio, embodiment, torch.tensor([1]), torch.tensor([2, 4]))
        self.assertFalse(torch.allclose(first.remaining, reordered.remaining))
        first.remaining.square().mean().backward()
        self.assertGreater(reader.demo.weight.grad.abs().sum().item(), 0)

    def test_physics_is_causal_equivariant_and_has_no_task_input(self):
        model = CausalEffectPredictor(5, 2, 3, 2, 3, 2, 1, hidden_dim=12)
        history, proprio = torch.randn(2, 4, 3, 5), torch.randn(2, 4, 2)
        actions, embodiment = torch.randn(2, 6, 3, requires_grad=True), torch.randn(2, 2)
        first = model(history, proprio, actions, embodiment)
        second_actions = actions.detach().clone()
        second_actions[:, 3:] += 100
        second = model(history, proprio, second_actions, embodiment)
        for name in ("geometry", "relation_logits", "event_logits", "prediction_uncertainty"):
            torch.testing.assert_close(getattr(first, name)[:, :3], getattr(second, name)[:, :3])
        first.geometry[:, :3].sum().backward()
        self.assertEqual(actions.grad[:, 3:].abs().sum().item(), 0)
        order = torch.tensor([2, 0, 1])
        permuted = model(history[:, :, order], proprio, actions, embodiment)
        torch.testing.assert_close(first.geometry[:, :, order], permuted.geometry)
        torch.testing.assert_close(first.relation_logits[:, :, order][:, :, :, order], permuted.relation_logits)
        self.assertEqual(set(inspect.signature(model.forward).parameters),
                         {"history", "proprio", "actions", "embodiment", "entity_present"})
        with self.assertRaises(TypeError):
            model(history, proprio, actions, embodiment, goal=torch.randn(2, 1))

    def test_physics_supervision_not_gated_by_uncertainty(self):
        model = CausalEffectPredictor(5, 2, 3, 2, 3, 2, 1, 12)
        prediction = model(torch.randn(2, 4, 3, 5), torch.randn(2, 4, 2), torch.randn(2, 6, 3), torch.randn(2, 2))
        targets = {"geometry": torch.zeros(2, 2, 3, 3), "relations": torch.zeros(2, 2, 3, 3, 2),
                   "events": torch.zeros(2, 2, 3, 3, 1)}
        outcome = PhysicalOutcome(torch.tensor([[0, 1, 2]]).expand(2, -1), torch.tensor([2, 5]),
                                  **targets, label_valid={k: torch.ones_like(v, dtype=torch.bool) for k, v in targets.items()})
        loss = physical_prediction_loss(prediction, outcome)
        loss.backward()
        self.assertGreater(model.future.weight_ih_l0.grad.abs().sum().item(), 0)
        bad = copy.copy(prediction)
        bad.prediction_uncertainty = prediction.prediction_uncertainty + 100
        self.assertGreater(physical_prediction_loss(bad, outcome).item(), loss.item())

    def test_interaction_head_uses_phi_and_is_entity_equivariant(self):
        head = TemporalInteractionHead(8, 2, 1, 12)
        phi = torch.randn(2, 3, 8, requires_grad=True)
        first = head(phi, torch.tensor([1, 4]))
        self.assertEqual(first["relation_logits"].shape, (2, 2, 3, 3, 2))
        first["relation_logits"].sum().backward()
        self.assertGreater(phi.grad.abs().sum().item(), 0)
        permuted = head(phi[:, [2, 0, 1]], torch.tensor([1, 4]))
        torch.testing.assert_close(first["event_logits"][:, :, [2, 0, 1]][:, :, :, [2, 0, 1]], permuted["event_logits"])

    def test_scoring_respects_terminal_time_missing_targets_and_uncertainty(self):
        req = requirement(batch=1)
        for mask in req.requirement_mask.values():
            mask.zero_()
        req.geometry.zero_()
        req.geometry[0, 1, 0, 0] = 1
        req.requirement_mask["geometry"][0, 1, 0, 0] = True
        prediction = PhysicalPrediction(torch.zeros(1, 4, 3, 3), torch.zeros(1, 4, 3, 3, 2),
                                        torch.zeros(1, 4, 3, 3, 1), torch.zeros(1, 4, 3))
        prediction.geometry[0, 3, 0, 0] = 1
        prediction.geometry[0, 0, 0, 0] = -50  # No requirement to finish in prefix.
        self.assertEqual(effect_cost(prediction, req, prefix_steps=1).item(), 0)
        uncertain = copy.copy(prediction)
        uncertain.prediction_uncertainty = torch.ones_like(prediction.prediction_uncertainty)
        self.assertGreater(effect_cost(uncertain, req).item(), 0)
        visible = torch.tensor([[False, True, True]])
        self.assertTrue(torch.isinf(effect_cost(prediction, req, entity_visible=visible)).all())
        req.label_valid["geometry"][0, 1, 0, 0] = False
        self.assertTrue(torch.isinf(effect_cost(prediction, req)).all())
        req.requirement_mask["geometry"].zero_()
        self.assertTrue(torch.isinf(effect_cost(prediction, req)).all())
        self.assertEqual(effect_cost(prediction, req, active=torch.tensor([False])).item(), 0)

    def test_cost_entity_permutation(self):
        req = requirement()
        prediction = CausalEffectPredictor(5, 2, 3, 2, 3, 2, 1, 12)(
            torch.randn(2, 3, 3, 5), torch.randn(2, 3, 2), torch.randn(2, 4, 3), torch.randn(2, 2))
        order = torch.tensor([2, 0, 1])
        swapped = PhysicalPrediction(prediction.geometry[:, :, order],
                                     prediction.relation_logits[:, :, order][:, :, :, order],
                                     prediction.event_logits[:, :, order][:, :, :, order],
                                     prediction.prediction_uncertainty[:, :, order])
        torch.testing.assert_close(effect_cost(prediction, req), effect_cost(swapped, req.permute_entities(order)))


if __name__ == "__main__":
    unittest.main()
