import math
import unittest
from dataclasses import replace

import torch

from evo_wam.contracts import (
    BINDING_UNCERTAIN, BINDING_UNMATCHED, BINDING_UNUSED,
    EffectRequirement, PhysicalOutcome, TaskRequirement,
)
from evo_wam.losses import (
    MaskedLoss, aggregate_fields, masked_bce, masked_mean, masked_mse,
    merge_category_mass, paired_js, pair_supervised_mean,
)


def outcome():
    geometry = torch.arange(24, dtype=torch.float32).reshape(2, 2, 3, 2)
    relations = torch.zeros(2, 2, 3, 3, 2)
    relations[:, :, 0, 2, :] = 1
    events = torch.zeros(2, 2, 3, 3, 1)
    fields = dict(geometry=geometry, relations=relations, events=events)
    return PhysicalOutcome(
        torch.tensor([[10, 20, 30], [40, 50, 60]]), torch.tensor([1, 4]),
        **fields, label_valid={name: torch.ones_like(value, dtype=torch.bool) for name, value in fields.items()},
    )


def requirement():
    physical = outcome()
    fields = dict(
        geometry=torch.zeros(2, 2, 2, 2),
        relations=torch.zeros(2, 2, 2, 2, 2),
        events=torch.zeros(2, 2, 2, 2, 1),
    )
    masks = {name: torch.zeros_like(value, dtype=torch.bool) for name, value in fields.items()}
    masks["geometry"][:, -1, 0, :] = True
    validity = {name: torch.ones_like(value, dtype=torch.bool) for name, value in fields.items()}
    validity["binding"] = torch.ones(2, 2, dtype=torch.bool)
    return EffectRequirement(
        physical.entity_ids, physical.step_offsets, torch.tensor([[0, 2], [2, 1]]),
        **fields, requirement_mask=masks, label_valid=validity,
    )


def temporal_requirement():
    req = requirement()
    # Two event constraints, not two compulsory event timestamps.
    req.events[:, 0, 0, 0, 0] = 1
    req.events[:, 1, 1, 1, 0] = 1
    req.requirement_mask["events"][:, 0, 0, 0, 0] = True
    req.requirement_mask["events"][:, 1, 1, 1, 0] = True
    req.event_windows[..., 0] = 1
    req.event_windows[..., 1] = 8
    req.event_precedence = torch.tensor([[[0, 7], [-1, -1]], [[0, 7], [-1, -1]]])
    req.label_valid["event_precedence"] = torch.ones(2, 2, dtype=torch.bool)
    return req


class ContractsTest(unittest.TestCase):
    def test_scene_and_requirement_validate(self):
        self.assertIs(outcome().validate().__class__, PhysicalOutcome)
        req = requirement()
        self.assertIs(req.validate(active=torch.ones(2, dtype=torch.bool)), req)
        TaskRequirement(req, req).validate(active=torch.ones(2, dtype=torch.bool))

    def test_entity_permutation_maps_both_relation_axes_and_bindings(self):
        physical, req = outcome(), requirement()
        order = torch.tensor([[2, 0, 1], [1, 2, 0]])
        permuted = physical.permute_entities(order).validate()
        mapped = req.permute_entities(order).validate()
        self.assertTrue(torch.equal(permuted.entity_ids, torch.tensor([[30, 10, 20], [50, 60, 40]])))
        self.assertTrue(torch.equal(mapped.binding, torch.tensor([[1, 0], [1, 0]])))
        self.assertTrue(torch.equal(permuted.geometry[0, :, 0], physical.geometry[0, :, 2]))
        self.assertTrue(torch.equal(permuted.relations[0, :, 1, 0], physical.relations[0, :, 0, 2]))
        self.assertIs(mapped.geometry, req.geometry)
        restored = permuted.permute_entities(order.argsort(1))
        for name in ("entity_ids", "geometry", "relations", "events"):
            self.assertTrue(torch.equal(getattr(restored, name), getattr(physical, name)))
        self.assertTrue(torch.equal(mapped.entity_ids.gather(1, mapped.binding), req.entity_ids.gather(1, req.binding)))

    def test_invalid_permutation_and_time_grid_fail(self):
        with self.assertRaises(ValueError):
            outcome().permute_entities(torch.tensor([0, 0, 1]))
        with self.assertRaises(ValueError):
            replace(outcome(), step_offsets=torch.tensor([4, 1])).validate()
        with self.assertRaises(ValueError):
            replace(outcome(), entity_ids=torch.tensor([[1, 1, 2], [3, 4, 5]])).validate()

    def test_unknown_is_not_negative_and_padding_cannot_be_valid(self):
        labels = outcome()
        labels.events[0, 0, 0, 1, 0] = float("nan")
        labels.label_valid["events"][0, 0, 0, 1, 0] = False
        labels.validate()
        labels.label_valid["events"][0, 0, 0, 1, 0] = True
        with self.assertRaises(ValueError):
            labels.validate()
        labels = outcome()
        labels.entity_ids[0, 0] = -1
        with self.assertRaises(ValueError):
            labels.validate()

    def test_required_and_label_valid_are_independent(self):
        req = requirement()
        req.label_valid["geometry"][:] = False
        req.geometry[:] = float("nan")
        req.validate(active=torch.ones(2, dtype=torch.bool))
        self.assertTrue(req.has_requirement.all())

    def test_active_empty_and_unbound_requirements_fail(self):
        req = requirement()
        for mask in req.requirement_mask.values():
            mask.zero_()
        req.validate(active=torch.zeros(2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            req.validate(active=torch.ones(2, dtype=torch.bool))
        req = requirement()
        req.binding[0, 0] = -1
        req.label_valid["binding"][0, 0] = False
        with self.assertRaises(ValueError):
            req.validate()

    def test_task_activity_is_explicit_and_can_use_remaining_only(self):
        current, remaining = requirement(), requirement()
        for mask in current.requirement_mask.values():
            mask.zero_()
        TaskRequirement(current, remaining).validate(active=torch.ones(2, dtype=torch.bool))
        future = replace(
            remaining, step_offsets=torch.tensor([8]),
            geometry=remaining.geometry[:, :1], relations=remaining.relations[:, :1], events=remaining.events[:, :1],
            requirement_mask={name: value[:, :1] for name, value in remaining.requirement_mask.items()},
            geometry_tolerance=remaining.geometry_tolerance[:, :1],
            event_windows=remaining.event_windows[:, :1],
            label_valid={name: value if name in {"binding", "event_precedence"} else value[:, :1]
                         for name, value in remaining.label_valid.items()},
        )
        TaskRequirement(current, future).validate()
        wrong = replace(remaining, entity_ids=remaining.entity_ids + 100)
        with self.assertRaises(ValueError):
            TaskRequirement(current, wrong).validate()

    def test_invalid_mask_shapes_and_relation_labels_fail(self):
        physical = outcome()
        physical.label_valid["geometry"] = torch.ones(2, 2, 3, dtype=torch.bool)
        with self.assertRaises(ValueError):
            physical.validate()
        physical = outcome()
        physical.relations[0, 0, 0, 1, 0] = 0.3
        with self.assertRaises(ValueError):
            physical.validate()

    def test_legacy_defaults_are_exact_fixed_time_and_known(self):
        req = requirement().validate()
        self.assertFalse(req.geometry_tolerance.any())
        self.assertTrue(torch.equal(req.event_windows[:, 0], torch.ones_like(req.event_windows[:, 0])))
        self.assertTrue(torch.equal(req.event_windows[:, 1], torch.full_like(req.event_windows[:, 1], 4)))
        self.assertEqual(req.event_precedence.shape, (2, 0, 2))
        self.assertTrue(req.semantics_known.all())
        self.assertTrue(TaskRequirement(req, req).resolved.all())

    def test_missing_and_uncertain_bindings_preserve_requirements(self):
        for status in (BINDING_UNMATCHED, BINDING_UNCERTAIN):
            req = requirement()
            before = {name: mask.clone() for name, mask in req.requirement_mask.items()}
            req.binding[:, 0] = status
            req.validate(active=torch.ones(2, dtype=torch.bool))
            self.assertFalse(req.resolved.any())
            self.assertFalse(req.semantics_known.any())
            self.assertTrue(req.has_requirement.all())
            for name, mask in before.items():
                self.assertTrue(torch.equal(mask, req.requirement_mask[name]))
            permuted = req.permute_entities(torch.tensor([2, 0, 1])).validate()
            self.assertTrue((permuted.binding[:, 0] == status).all())
        req = requirement()
        req.binding[:, 1] = BINDING_UNUSED  # Role 1 has no conditions in this fixture.
        req.validate()
        self.assertTrue(req.resolved.all())
        req.binding[:, 0] = BINDING_UNUSED
        with self.assertRaises(ValueError):
            req.validate()

    def test_geometry_intervals_and_semantic_validity_are_independent(self):
        req = requirement()
        req.geometry.fill_(3.0)
        req.geometry_tolerance.fill_(0.5)
        req.validate()
        low, high = req.geometry_bounds
        self.assertTrue((low == 2.5).all())
        self.assertTrue((high == 3.5).all())
        req.geometry_tolerance[0, -1, 0, 0] = -0.1
        with self.assertRaises(ValueError):
            req.validate()
        req.label_valid["geometry_tolerance"][0, -1, 0, 0] = False
        req.geometry_tolerance[0, -1, 0, 0] = float("nan")
        req.validate()
        self.assertTrue(torch.equal(req.semantics_known, torch.tensor([False, True])))

    def test_explicit_semantics_require_explicit_validity(self):
        req = requirement()
        validity = {name: value for name, value in req.label_valid.items() if name != "geometry_tolerance"}
        with self.assertRaises(ValueError):
            replace(req, label_valid=validity).validate()
        with self.assertRaises(ValueError):
            replace(req, event_windows=req.event_windows.float()).validate()

    def test_windows_are_inclusive_intervals_and_unknown_is_not_absence(self):
        req = temporal_requirement().validate()
        self.assertTrue(req.semantics_known.all())
        req.event_windows[0, 0, 0, 0, 0] = torch.tensor([5, 3])
        with self.assertRaises(ValueError):
            req.validate()
        req.label_valid["event_windows"][0, 0, 0, 0, 0] = False
        req.validate()
        self.assertFalse(req.semantics_known[0])
        req = temporal_requirement()
        req.label_valid["event_precedence"][0, 1] = False
        req.validate()
        self.assertFalse(req.semantics_known[0])
        self.assertTrue(req.semantics_known[1])

    def test_precedence_rejects_bad_indices_padding_and_nonpositive_nodes(self):
        for pair in ((0, 8), (-1, 0), (-2, -2), (0, 0), (0, 1)):
            req = temporal_requirement()
            req.event_precedence[0, 0] = torch.tensor(pair)
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                req.validate()
        req = temporal_requirement()
        req.requirement_mask["events"][0, 0, 0, 0, 0] = False
        with self.assertRaises(ValueError):
            req.validate()

    def test_precedence_rejects_duplicates_cycles_and_infeasible_windows(self):
        req = temporal_requirement()
        req.event_precedence[0, 1] = req.event_precedence[0, 0]
        with self.assertRaises(ValueError):
            req.validate()
        req = temporal_requirement()
        req.event_precedence[0, 1] = torch.tensor([7, 0])
        with self.assertRaises(ValueError):
            req.validate()
        req = temporal_requirement()
        req.event_windows[0, 0, 0, 0, 0] = torch.tensor([3, 5])
        req.event_windows[0, 1, 1, 1, 0] = torch.tensor([1, 3])
        with self.assertRaises(ValueError):
            req.validate()
        req.event_windows[0, 1, 1, 1, 0] = torch.tensor([4, 8])
        req.validate()

    def test_entity_permutation_does_not_reindex_role_temporal_constraints(self):
        req = temporal_requirement().validate()
        permuted = req.permute_entities(torch.tensor([1, 2, 0])).validate()
        for name in ("geometry_tolerance", "event_windows", "event_precedence"):
            self.assertTrue(torch.equal(getattr(permuted, name), getattr(req, name)))
            self.assertTrue(torch.equal(permuted.label_valid[name], req.label_valid[name]))
        self.assertTrue(torch.equal(permuted.required_roles, req.required_roles))
        self.assertTrue(torch.equal(permuted.entity_ids.gather(1, permuted.binding), req.entity_ids.gather(1, req.binding)))


class LossesTest(unittest.TestCase):
    def test_masked_mean_normalizes_examples_not_annotation_counts(self):
        values = torch.tensor([[1.0, 5.0], [7.0, float("nan")], [float("nan"), float("inf")]], requires_grad=True)
        valid = torch.tensor([[True, True], [True, False], [False, False]])
        result = masked_mean(values, valid, torch.tensor([1.0, 3.0]))
        self.assertTrue(torch.equal(result.per_sample, torch.tensor([4.0, 7.0, 0.0])))
        self.assertAlmostEqual(result.loss.item(), 5.5)
        self.assertEqual(result.valid_count.item(), 2)
        result.loss.backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertEqual(values.grad[1, 1].item(), 0)

    def test_mse_sanitizes_before_arithmetic_and_bce_is_multilabel(self):
        prediction = torch.tensor([[2.0, float("nan")]], requires_grad=True)
        result = masked_mse(prediction, torch.tensor([[0.0, float("nan")]]), torch.tensor([[True, False]]))
        result.loss.backward()
        self.assertTrue(torch.equal(prediction.grad, torch.tensor([[4.0, 0.0]])))
        logits = torch.tensor([[8.0, 8.0]], requires_grad=True)
        bce = masked_bce(logits, torch.ones_like(logits), torch.ones_like(logits, dtype=torch.bool))
        self.assertLess(bce.loss.item(), 0.001)
        bce.loss.backward()
        self.assertTrue((logits.grad < 0).all())

    def test_masked_bce_ignores_unknown_targets(self):
        logits = torch.tensor([[0.0, float("nan")]], requires_grad=True)
        result = masked_bce(logits, torch.tensor([[1.0, float("nan")]]), torch.tensor([[True, False]]))
        result.loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0, 1].item(), 0)

    def test_js_identity_symmetry_and_bound(self):
        first = torch.tensor([[1000.0, -1000.0], [2.0, -1.0]], requires_grad=True)
        second = torch.tensor([[-1000.0, 1000.0], [-1.0, 2.0]], requires_grad=True)
        valid = torch.ones(2, dtype=torch.bool)
        identity = paired_js(first, first, valid, valid, kind="categorical")
        self.assertLess(identity.loss.item(), 1e-7)
        result = paired_js(first, second, valid, valid, kind="categorical")
        reverse = paired_js(second, first, valid, valid, kind="categorical")
        self.assertTrue(torch.allclose(result.per_sample, reverse.per_sample))
        self.assertAlmostEqual(result.per_sample[0].item(), math.log(2), places=6)
        self.assertTrue((result.per_sample <= math.log(2) + 1e-6).all())
        result.loss.backward()
        self.assertGreater(first.grad[1].abs().sum().item(), 0)
        self.assertGreater(second.grad[1].abs().sum().item(), 0)

    def test_bernoulli_does_not_softmax_mutually_compatible_relations(self):
        first = torch.tensor([[10.0, 10.0]], requires_grad=True)
        second = -first.detach().clone().requires_grad_()
        valid = torch.ones_like(first, dtype=torch.bool)
        result = paired_js(first, second, valid, valid, kind="bernoulli")
        self.assertGreater(result.loss.item(), 0.69)

    def test_common_mask_and_empty_pair_keep_both_gradient_paths(self):
        first = torch.tensor([[2.0, float("nan")], [float("nan"), float("nan")]], requires_grad=True)
        second = torch.tensor([[-1.0, float("nan")], [float("nan"), float("nan")]], requires_grad=True)
        a_valid = torch.tensor([[True, False], [False, False]])
        b_valid = torch.tensor([[True, True], [False, False]])
        result = paired_js(first, second, a_valid, b_valid, kind="bernoulli")
        self.assertEqual(result.valid_count.item(), 1)
        self.assertEqual(result.coverage.item(), 0.5)
        result.loss.backward()
        for source in (first, second):
            self.assertTrue(torch.isfinite(source.grad).all())
            self.assertGreater(source.grad[0, 0].abs().item(), 0)
            self.assertEqual(source.grad[1].abs().sum().item(), 0)
        empty = paired_js(first, second, torch.zeros_like(a_valid), b_valid, kind="bernoulli")
        self.assertEqual(empty.loss.item(), 0)
        self.assertTrue(empty.loss.requires_grad)
        self.assertEqual(empty.valid_count.item(), 0)

    def test_weights_cannot_be_predicted_or_negative(self):
        values = torch.ones(1, 2)
        mask = torch.ones_like(values, dtype=torch.bool)
        for weights in (torch.ones(2, requires_grad=True), torch.tensor([-1.0, 1.0])):
            with self.assertRaises(ValueError):
                masked_mean(values, mask, weights)
        with self.assertRaises(ValueError):
            masked_mean(values, torch.ones(1, dtype=torch.bool))

    def test_legal_equivalence_merges_mass_but_retains_illegal_mass(self):
        first = torch.tensor([[0.2, 0.6, 0.2]], requires_grad=True)
        second = torch.tensor([[0.6, 0.2, 0.2]], requires_grad=True)
        groups = torch.tensor([0, 0, 1])
        merged = merge_category_mass(first, groups)
        self.assertTrue(torch.allclose(merged, torch.tensor([[0.8, 0.2]])))
        mask = torch.ones(1, dtype=torch.bool)
        result = paired_js(first.log(), second.log(), mask, mask, kind="categorical", category_groups=groups)
        self.assertLess(result.loss.item(), 1e-7)
        illegal = torch.tensor([[0.1, 0.1, 0.8]])
        result = paired_js(first.log(), illegal.log(), mask, mask, kind="categorical", category_groups=groups)
        self.assertGreater(result.loss.item(), 0.1)
        result.loss.backward()
        self.assertTrue(torch.isfinite(first.grad).all())
        with self.assertRaises(ValueError):
            merge_category_mass(first, torch.tensor([0, 0, -1]))

    def test_group_gaps_with_zero_mass_have_finite_gradients(self):
        first = torch.tensor([[1.0, 2.0, -1.0]], requires_grad=True)
        second = torch.tensor([[0.0, -1.0, 3.0]], requires_grad=True)
        valid = torch.ones(1, dtype=torch.bool)
        result = paired_js(first, second, valid, valid, kind="categorical", category_groups=torch.tensor([0, 0, 2]))
        result.loss.backward()
        self.assertTrue(torch.isfinite(first.grad).all())
        self.assertTrue(torch.isfinite(second.grad).all())

    def test_aggregate_fields_uses_available_field_weights_per_example(self):
        first = MaskedLoss(torch.tensor([2.0, 4.0, 0.0]), torch.tensor([True, True, False]))
        second = MaskedLoss(torch.tensor([6.0, 0.0, 0.0]), torch.tensor([True, False, False]))
        result = aggregate_fields({"binding": first, "contact": second}, {"binding": 1.0, "contact": 3.0})
        self.assertTrue(torch.equal(result.per_sample, torch.tensor([5.0, 4.0, 0.0])))
        self.assertEqual(result.loss.item(), 4.5)
        self.assertEqual(result.valid_count.item(), 2)

    def test_supervision_is_mean_and_survives_empty_cv(self):
        first = masked_mean(torch.tensor([[2.0], [4.0]]), torch.ones(2, 1, dtype=torch.bool))
        second = masked_mean(torch.tensor([[4.0], [6.0]]), torch.ones(2, 1, dtype=torch.bool))
        result = pair_supervised_mean(first, second)
        self.assertEqual(result.loss.item(), 4.0)
        self.assertEqual(pair_supervised_mean(torch.tensor(2.0), torch.tensor(6.0)).item(), 4.0)
        zero = torch.zeros(2, 1, requires_grad=True)
        empty = paired_js(zero, zero, torch.zeros_like(zero, dtype=torch.bool), torch.zeros_like(zero, dtype=torch.bool), kind="bernoulli")
        self.assertEqual((result.loss + empty.loss).item(), 4.0)


if __name__ == "__main__":
    unittest.main()
