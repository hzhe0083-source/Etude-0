"""Structural and small-sample checks, not evidence of robot transfer."""

import unittest

import torch

from evo_wam.goal_interface import GoalInterface, _rotation_from_6d, goal_pose_loss, validate_goal_poses, validate_se3


class GoalInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(43)
        self.options = dict(native_dim=12, feature_dim=12, state_dim=4, effectors=2,
                            dim=16, num_tokens=4, num_heads=4, translation_scale=.5,
                            num_layers=2, num_layer_groups=1, num_pose_tokens=2)
        self.model = GoalInterface(**self.options)
        self.poses = torch.eye(4).repeat(2, 2, 1, 1)
        self.poses[:, 0, :3, 3] = torch.tensor([.2, -.1, .3])
        self.poses[:, 1, :3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self.gripper = torch.tensor([[.2, .8], [.2, .8]])
        self.state = torch.randn(2, 4)
        self.language = torch.randn(2, 3, 12)
        self.features = torch.randn(2, 3, 2, 12)
        self.times = torch.tensor([.1, .6, 1.8], dtype=torch.float64)
        self.xy = torch.tensor([[-.5, 0.], [.5, 0.]])

    def read(self, features=None, state=None, times=None, xy=None, language=None):
        features = self.features if features is None else features
        semantic = self.model.semantic(self.language if language is None else language,
                                       self.state if state is None else state)
        tokens = self.model.initial_queries(features.shape[0])
        for index in range(self.model.num_layers):
            tokens = self.model.read_layer(tokens, features, semantic,
                                          self.times if times is None else times,
                                          self.xy if xy is None else xy, index)
        return tokens

    def loss(self, tokens):
        prediction = self.model.decode_goal(tokens)
        return goal_pose_loss(prediction['goal_poses'], self.poses, .5,
                              gripper_prediction=prediction['goal_gripper'], gripper_target=self.gripper)

    def test_goal_encoder_and_independent_language_state_conditions(self):
        tokens = self.model.encode_goal(self.poses, self.gripper)
        self.assertEqual(tokens.shape, (2, 4, 16))
        torch.testing.assert_close(tokens[0], tokens[1])
        changed = self.gripper.clone()
        changed[1] = changed[1].flip(0)
        self.assertFalse(torch.allclose(self.model.encode_goal(self.poses, changed)[1], tokens[1]))
        conditions = self.model.condition(tokens, self.state, self.language)
        self.assertEqual(conditions.shape, (2, 8, 12))
        torch.testing.assert_close(conditions[:, :3], self.language)
        torch.testing.assert_close(conditions[:, 3:4], self.model.state_encoder(self.state))
        torch.testing.assert_close(conditions[0, 4:], conditions[1, 4:])
        baseline = self.model.direct_condition(self.features, self.state, self.language)
        self.assertEqual(baseline.shape, (2, 10, 12))
        torch.testing.assert_close(baseline[:, 4:], self.features.flatten(1, 2))
        with self.assertRaises(TypeError):
            self.model.encode_goal(self.poses)

    def test_decoder_reads_only_pose_prefix_and_produces_valid_rotations(self):
        tokens = self.model.encode_goal(self.poses, self.gripper).detach().requires_grad_()
        prediction = self.model.decode_goal(tokens)
        self.assertEqual(prediction['goal_poses'].shape, (2, 2, 4, 4))
        self.assertEqual(prediction['goal_gripper'].shape, (2, 2))
        validate_goal_poses(prediction['goal_poses'])
        changed = tokens.detach().clone()
        changed[:, 2:] += 10
        for name, value in prediction.items():
            torch.testing.assert_close(self.model.decode_goal(changed)[name], value)
        self.loss(tokens)['total'].backward()
        self.assertGreater(tokens.grad[:, :2].abs().sum().item(), 0)
        self.assertEqual(tokens.grad[:, 2:].abs().sum().item(), 0)
        rotation_inputs = torch.tensor([
            [1., 0., 0., 0., 1., 0.], [0., 1., 0., -1., 0., 0.],
            [0., 0., 0., 0., 0., 0.], [1., 1., 1., 2., 2., 2.],
            [1., 1., 1., 1., 1., 1.0001],
        ], requires_grad=True)
        rotations = _rotation_from_6d(rotation_inputs)
        torch.testing.assert_close(rotations.transpose(-1, -2) @ rotations, torch.eye(3).expand(5, -1, -1),
                                   atol=2e-6, rtol=0)
        torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(5), atol=2e-6, rtol=0)
        rotations.sum().backward()
        self.assertTrue(torch.isfinite(rotation_inputs.grad).all())
        self.model.to(torch.bfloat16)
        validate_goal_poses(self.model.decode_goal(tokens)['goal_poses'])

    def test_loss_separates_components_and_detaches_metrics_and_targets(self):
        target = torch.eye(4).repeat(1, 1, 1, 1).requires_grad_()
        identity = target.detach().clone().requires_grad_()
        grip = torch.tensor([[.5]], requires_grad=True)
        goal_grip = grip.detach().clone().requires_grad_()
        losses = goal_pose_loss(identity, target, .5, gripper_prediction=grip, gripper_target=goal_grip)
        self.assertEqual(losses['total'].item(), 0)
        losses['total'].backward()
        self.assertTrue(torch.isfinite(identity.grad).all())
        self.assertTrue(torch.isfinite(grip.grad).all())
        self.assertIsNone(target.grad)
        self.assertIsNone(goal_grip.grad)
        for name in ('position_error_m', 'orientation_error_deg', 'gripper_error'):
            self.assertFalse(losses[name].requires_grad)
        moved = target.detach().clone()
        moved[..., :3, 3] = .5
        moved[..., :3, :3] = self.poses[0, 1, :3, :3]
        losses = goal_pose_loss(moved, target, .5, gripper_prediction=torch.ones(1, 1),
                                gripper_target=torch.zeros(1, 1))
        self.assertAlmostEqual(losses['translation'].item(), 1.)
        self.assertAlmostEqual(losses['rotation'].item(), 4.)
        self.assertAlmostEqual(losses['gripper'].item(), 1.)
        self.assertAlmostEqual(losses['position_error_m'].item(), 3 ** .5 / 2, places=6)
        self.assertAlmostEqual(losses['orientation_error_deg'].item(), 90.)
        self.assertAlmostEqual(losses['gripper_error'].item(), 1.)

    def test_recursive_conditions_use_early_features_and_all_token_queries(self):
        features = [self.features.clone().requires_grad_(), (self.features + .5).requires_grad_()]
        semantic = self.model.semantic(self.language, self.state)
        first = self.model.read_layer(self.model.initial_queries(2), features[0], semantic, self.times, self.xy, 0)
        first.retain_grad()
        final = self.model.read_layer(first, features[1], semantic, self.times, self.xy, 1)
        self.assertFalse(torch.allclose(first, final))
        self.loss(final)['total'].backward()
        for value in (first, *features):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        # The unsliced earlier workspace participates in the next self-attention.
        self.assertGreater(first.grad[:, 2:].abs().sum().item(), 0)
        self.assertGreater(self.model.visual_queries.grad.abs().sum().item(), 0)
        gradients = [p.grad for p in self.model.recurrent_groups.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        reset = self.model.read_layer(self.model.initial_queries(2), features[1], semantic, self.times, self.xy, 1)
        self.assertFalse(torch.allclose(final, reset))

    def test_groups_are_shared_by_contiguous_layers(self):
        self.model = GoalInterface(**{**self.options, 'num_layers': 4, 'num_layer_groups': 2})
        calls = []
        handles = [group.register_forward_hook(lambda _module, _args, _out, index=index: calls.append(index))
                   for index, group in enumerate(self.model.recurrent_groups)]
        try:
            self.read()
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(calls, [0, 0, 1, 1])

    def test_future_context_uses_state_time_and_xy_not_storage_order(self):
        tokens = self.read()
        for kwargs in ({'state': self.state + 1}, {'language': self.language + 1},
                       {'features': self.features.flip(1)}, {'features': self.features.flip(2)}):
            self.assertFalse(torch.allclose(tokens, self.read(**kwargs)))
        permuted = self.read(features=self.features.flip(2), xy=self.xy.flip(0))
        torch.testing.assert_close(tokens, permuted, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(tokens, self.read(times=self.times + 100))

    def test_position_swap_windows_fit_distinct_targets_with_identical_semantics(self):
        # Every frame has the same feature set. Only content-position-time binding differs.
        self.features = torch.randn(1, 1, 2, 12).expand(2, 3, -1, -1).clone()
        self.features[1, 1:] = self.features[1, 1:].flip(1)
        self.language[:] = self.language[0].clone()
        self.state[:] = self.state[0].clone()
        self.poses = torch.eye(4).repeat(2, 2, 1, 1)
        self.poses[0, :, 0, 3], self.poses[1, :, 0, 3] = -.3, .3
        self.gripper = torch.tensor([[.2, .2], [.8, .8]])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.01)
        for _ in range(300):
            optimizer.zero_grad(set_to_none=True)
            loss = self.loss(self.read())['total']
            loss.backward()
            optimizer.step()
        metrics = self.loss(self.read())
        self.assertLess(metrics['position_error_m'].item(), .025)
        self.assertLess(metrics['gripper_error'].item(), .04)
        self.assertLess(metrics['rotation'].item(), .005)

    def test_validations_reject_bad_goals_grippers_and_context(self):
        validate_se3(torch.eye(4))
        validate_se3(self.poses[None])
        for key in ('native_dim', 'feature_dim', 'state_dim', 'effectors', 'dim', 'num_tokens', 'num_heads',
                    'num_layers', 'num_layer_groups', 'num_pose_tokens'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                GoalInterface(**{**self.options, key: 0})
        for changes in ({'feature_dim': 5}, {'num_pose_tokens': 5}, {'num_layers': 3, 'num_layer_groups': 2}):
            with self.assertRaises(ValueError):
                GoalInterface(**{**self.options, **changes})
        for value in (0, -1, True, float('nan'), float('inf')):
            with self.subTest(scale=value), self.assertRaises(ValueError):
                goal_pose_loss(self.poses, self.poses, value, gripper_prediction=self.gripper,
                               gripper_target=self.gripper)
        invalid = self.poses.clone()
        invalid[..., 0, 0] = -1
        for poses in (invalid, self.poses.long(), self.poses[..., :3], self.poses * float('nan')):
            with self.assertRaisesRegex(ValueError, 'goal poses'):
                self.model.encode_goal(poses, self.gripper)
        for grip in (self.gripper[:, :1], self.gripper + 2, self.gripper * float('nan'), self.gripper.long()):
            with self.assertRaisesRegex(ValueError, 'gripper'):
                self.model.encode_goal(self.poses, grip)
        for kwargs in ({'features': self.features[..., :4]}, {'state': self.state[:, :3]},
                       {'times': self.times.flip(0)}, {'xy': self.xy * 0}, {'language': self.language[..., :2]}):
            with self.assertRaises(ValueError):
                self.read(**kwargs)
        with self.assertRaisesRegex(ValueError, 'goal tokens'):
            self.model.decode_goal(torch.randn(2, 4, 15))
        with self.assertRaisesRegex(ValueError, 'layer_index'):
            self.model.read_layer(self.model.initial_queries(2), self.features,
                                  self.model.semantic(self.language, self.state), self.times, self.xy, 2)


if __name__ == '__main__':
    unittest.main()
