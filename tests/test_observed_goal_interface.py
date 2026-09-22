"""Structural checks for joint observed-Wan and pose-hidden action conditions."""

import unittest

import torch

from evo_wam.goal_interface import goal_pose_loss, validate_goal_poses
from evo_wam.goal_observed_interface import ObservedGoalInterface


class ObservedGoalInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(47)
        self.options = dict(native_dim=12, state_dim=4, effectors=2, dim=16,
                            num_heads=4, translation_scale=.5)
        self.model = ObservedGoalInterface(**self.options)
        self.features = torch.randn(2, 9, 12)
        self.state = torch.randn(2, 4)
        self.language = torch.randn(2, 3, 12)

    def run_interface(self, **changes):
        return self.model.condition_from_features(
            changes.get('features', self.features), changes.get('state', self.state),
            changes.get('language', self.language))

    def test_shape_order_valid_poses_and_only_supervised_predictions(self):
        captured = {}
        handle = self.model.pose_decoder.register_forward_hook(
            lambda _module, _args, output: captured.update(hidden=output[0]))
        try:
            condition, prediction = self.run_interface()
        finally:
            handle.remove()
        self.assertEqual(condition.shape, (2, 15, 12))
        self.assertEqual(set(prediction), {'goal_poses', 'goal_gripper'})
        self.assertEqual(prediction['goal_poses'].shape, (2, 2, 4, 4))
        self.assertEqual(prediction['goal_gripper'].shape, (2, 2))
        self.assertEqual(captured['hidden'].shape, (2, 2, 16))
        validate_goal_poses(prediction['goal_poses'])
        self.assertTrue(((prediction['goal_gripper'] > 0) & (prediction['goal_gripper'] < 1)).all())
        torch.testing.assert_close(condition[:, :3], self.language)
        torch.testing.assert_close(condition[:, 3:4], self.model.state_encoder(self.state))
        torch.testing.assert_close(condition[:, 4:13], self.model.wan_action_projection(self.features))
        torch.testing.assert_close(condition[:, 13:], self.model.pose_action_projection(captured['hidden']))
        raw = self.model.pose_decoder.head(captured['hidden'])
        torch.testing.assert_close(prediction['goal_poses'][..., :3, 3], raw[..., :3] * .5)
        torch.testing.assert_close(prediction['goal_gripper'], raw[..., 9].sigmoid())
        self.assertIsNot(self.model.wan_action_projection[-1], self.model.pose_action_projection[-1])

    def test_action_gradients_use_wan_and_pose_hidden_independently(self):
        self.features.requires_grad_()
        captured = {}

        def record(_module, _args, output):
            output[0].retain_grad()
            captured['hidden'] = output[0]

        handle = self.model.pose_decoder.register_forward_hook(record)
        try:
            condition, _ = self.run_interface()
        finally:
            handle.remove()
        # Nonuniform weights avoid the constant sums induced by LayerNorm.
        weights = torch.randn_like(condition)
        wan_loss = (condition[:, 4:13] * weights[:, 4:13]).sum()
        pose_loss = (condition[:, 13:] * weights[:, 13:]).sum()
        wan_grad = torch.autograd.grad(wan_loss, (self.features, captured['hidden']),
                                       allow_unused=True, retain_graph=True)
        self.assertGreater(wan_grad[0].abs().sum().item(), 0)
        # Concatenation can produce an explicit zero derivative for the other branch.
        self.assertTrue(wan_grad[1] is None or wan_grad[1].count_nonzero() == 0)
        pose_grad = torch.autograd.grad(pose_loss, (self.features, captured['hidden']), retain_graph=True)
        for value in pose_grad:
            self.assertTrue(torch.isfinite(value).all())
            self.assertGreater(value.abs().sum().item(), 0)
        (wan_loss + pose_loss).backward()
        for module in (self.model.wan_action_projection, self.model.pose_action_projection,
                       self.model.pose_decoder.readout):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)
        # Action learning uses hidden pose features, not the numeric pose head output.
        self.assertTrue(all(p.grad is None for p in self.model.pose_decoder.head.parameters()))

    def test_hidden_intervention_changes_only_pose_action_tokens(self):
        baseline, baseline_prediction = self.run_interface()

        def replace_hidden(_module, _args, output):
            hidden, prediction = output
            return hidden.flip(-1), prediction

        handle = self.model.pose_decoder.register_forward_hook(replace_hidden)
        try:
            changed, prediction = self.run_interface()
        finally:
            handle.remove()
        torch.testing.assert_close(changed[:, :13], baseline[:, :13])
        self.assertFalse(torch.allclose(changed[:, 13:], baseline[:, 13:]))
        for key in prediction:
            torch.testing.assert_close(prediction[key], baseline_prediction[key])

    def test_decoder_reads_all_wan_tokens_and_independent_state_language(self):
        condition, prediction = self.run_interface()
        features = self.features.clone().requires_grad_()
        state = self.state.clone().requires_grad_()
        language = self.language.clone().requires_grad_()
        _, differentiable_prediction = self.run_interface(features=features, state=state, language=language)
        targets = torch.eye(4).repeat(2, 2, 1, 1)
        loss = goal_pose_loss(differentiable_prediction['goal_poses'], targets, .5,
                             gripper_prediction=differentiable_prediction['goal_gripper'],
                             gripper_target=torch.zeros(2, 2))['total']
        loss.backward()
        for value in (features, state, language):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertTrue((value.grad.abs().sum(-1) > 0).all())
        late_features = self.features.clone()
        late_features[:, -1] = torch.randn_like(late_features[:, -1]) * 3
        for changes in ({'features': late_features}, {'state': self.state + 1},
                        {'language': self.language + 1}):
            with self.subTest(changes=tuple(changes)):
                altered, altered_prediction = self.run_interface(**changes)
                self.assertFalse(torch.allclose(condition[:, 13:], altered[:, 13:]))
                self.assertFalse(torch.allclose(prediction['goal_poses'], altered_prediction['goal_poses']))
        # State and language cannot alter the direct observed-Wan action branch.
        changed, _ = self.run_interface(state=self.state + 1, language=self.language + 1)
        torch.testing.assert_close(condition[:, 4:13], changed[:, 4:13])

    def test_model_dtype_and_autocast_preserve_finite_gradients_and_rotations(self):
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.model = ObservedGoalInterface(**self.options).to(dtype)
                features = self.features.clone().requires_grad_()
                condition, prediction = self.run_interface(features=features)
                self.assertEqual(condition.dtype, dtype)
                self.assertEqual(prediction['goal_poses'].dtype,
                                 torch.float64 if dtype == torch.float64 else torch.float32)
                validate_goal_poses(prediction['goal_poses'])
                loss = (condition * torch.randn_like(condition)).sum() + prediction['goal_poses'].sum()
                loss.backward()
                self.assertTrue(torch.isfinite(features.grad).all())
                self.assertGreater(features.grad.abs().sum().item(), 0)
        self.model = ObservedGoalInterface(**self.options)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            condition, prediction = self.run_interface()
        self.assertEqual(condition.dtype, torch.bfloat16)
        validate_goal_poses(prediction['goal_poses'])

    def test_reject_invalid_dimensions_and_inputs(self):
        for key in ('native_dim', 'state_dim', 'effectors', 'dim', 'num_heads'):
            for value in (0, -1, True, 1.5):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    ObservedGoalInterface(**{**self.options, key: value})
        with self.assertRaisesRegex(ValueError, 'divisible'):
            ObservedGoalInterface(**{**self.options, 'dim': 15})
        for value in (0, -1, True, float('nan'), float('inf')):
            with self.subTest(scale=value), self.assertRaises(ValueError):
                ObservedGoalInterface(**{**self.options, 'translation_scale': value})
        for key, tensor in (('features', self.features), ('state', self.state), ('language', self.language)):
            invalid = [None, tensor.long(), tensor * float('nan'), tensor * float('inf'),
                       tensor[:1], tensor[..., :-1], tensor[:0], tensor.unsqueeze(0)]
            if tensor.ndim == 3:
                invalid.append(tensor[:, :0])
            for value in invalid:
                with self.subTest(key=key, shape=getattr(value, 'shape', None)), self.assertRaises(ValueError):
                    self.run_interface(**{key: value})


if __name__ == '__main__':
    unittest.main()
