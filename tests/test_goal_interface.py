"""Small structural checks, not evidence of cross-view or robot transfer."""

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
        self.model = GoalInterface(12, 5, 4, 2, dim=16, num_tokens=3, num_heads=4, translation_scale=.5)
        self.poses = torch.eye(4).repeat(2, 2, 1, 1)
        self.poses[:, 0, :3, 3] = torch.tensor([.2, -.1, .3])
        self.poses[:, 1, :3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self.state = torch.randn(2, 4)
        self.features = torch.randn(2, 3, 2, 5)
        self.times = torch.tensor([.1, .6, 1.8], dtype=torch.float64)
        self.xy = torch.tensor([[-.5, 0.], [.5, 0.]])

    def test_goal_only_tokens_state_condition_and_action_feature_width(self):
        tokens = self.model.encode_goal(self.poses)
        self.assertEqual(tokens.shape, (2, 3, 16))
        torch.testing.assert_close(tokens[0], tokens[1])
        torch.testing.assert_close(self.model.encode_goal(self.poses), tokens)
        conditions = self.model.condition(tokens, self.state)
        self.assertEqual(conditions.shape, (2, 4, 12))
        torch.testing.assert_close(conditions[0, :-1], conditions[1, :-1])
        self.assertFalse(torch.allclose(conditions[0, -1], conditions[1, -1]))
        self.assertEqual(self.model.state_encoder(self.state).shape, (2, 1, 16))
        with self.assertRaises(TypeError):
            self.model.encode_goal(self.poses, self.state)
        changed_goal = self.poses.clone()
        changed_goal[0, 0, 0, 3] += 1.
        self.assertFalse(torch.allclose(self.model.encode_goal(changed_goal), tokens))

    def test_decoder_and_degenerate_6d_output_proper_so3(self):
        tokens = self.model.encode_goal(self.poses)
        prediction = self.model.decode_goal(tokens)
        self.assertEqual(prediction.shape, (2, 2, 4, 4))
        validate_goal_poses(prediction)
        torch.testing.assert_close(self.model.pose_decoder(tokens), prediction)
        rotation_inputs = torch.tensor([
            [1., 0., 0., 0., 1., 0.], [0., 1., 0., -1., 0., 0.],
            [0., 0., 0., 0., 0., 0.], [1., 1., 1., 2., 2., 2.],
            [1., 1., 1., 1., 1., 1.0001],
        ], requires_grad=True)
        rotations = _rotation_from_6d(rotation_inputs)
        torch.testing.assert_close(rotations[0], torch.eye(3))
        torch.testing.assert_close(rotations[1], self.poses[0, 1, :3, :3])
        torch.testing.assert_close(rotations.transpose(-1, -2) @ rotations, torch.eye(3).expand(5, -1, -1),
                                   atol=2e-6, rtol=0)
        torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(5), atol=2e-6, rtol=0)
        rotations.sum().backward()
        self.assertTrue(torch.isfinite(rotation_inputs.grad).all())
        self.model.to(torch.bfloat16)
        validate_goal_poses(self.model.decode_goal(tokens))

    def test_loss_separates_translation_rotation_and_is_finite_at_identity(self):
        target = torch.eye(4).repeat(1, 1, 1, 1)
        identity = target.clone().requires_grad_()
        losses = goal_pose_loss(identity, target, translation_scale=.5)
        self.assertEqual(losses['total'].item(), 0)
        losses['total'].backward()
        self.assertTrue(torch.isfinite(identity.grad).all())
        translated = target.clone()
        translated[..., :3, 3] = .5
        losses = goal_pose_loss(translated, target, translation_scale=.5)
        self.assertAlmostEqual(losses['translation'].item(), 1.)
        self.assertEqual(losses['rotation'].item(), 0)
        rotated = target.clone()
        rotated[..., :3, :3] = self.poses[0, 1, :3, :3]
        losses = goal_pose_loss(rotated, target)
        self.assertEqual(losses['translation'].item(), 0)
        self.assertAlmostEqual(losses['rotation'].item(), 4.)

    def test_nonvisual_and_visual_paths_propagate_finite_nonzero_gradients(self):
        tokens = self.model.encode_goal(self.poses)
        loss = goal_pose_loss(self.model.decode_goal(tokens), self.poses, .5)['total']
        loss = loss + self.model.condition(tokens, self.state).square().mean()
        loss.backward()
        for name in ('goal_encoder', 'pose_decoder', 'state_encoder', 'condition_adapter'):
            gradients = [p.grad for p in getattr(self.model, name).parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients), name)
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0, name)
        self.model.zero_grad(set_to_none=True)
        for parameter in self.model.state_encoder.parameters():
            parameter.requires_grad_(False)
        features = self.features.clone().requires_grad_()
        visual = self.model.read_future(features, self.state, self.times, self.xy)
        goal_pose_loss(self.model.decode_goal(visual), self.poses, .5)['total'].backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in self.model.state_encoder.parameters()))
        for name in ('visual_project', 'visual_readout', 'visual_norm'):
            gradients = [p.grad for p in getattr(self.model, name).parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients), name)
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0, name)
        self.assertGreater(self.model.visual_queries.grad.abs().sum().item(), 0)

    def test_future_context_uses_state_time_and_physical_xy_not_storage_order(self):
        tokens = self.model.read_future(self.features, self.state, self.times, self.xy)
        variants = (
            (self.features, self.state + 1., self.times, self.xy),
            (self.features.flip(1), self.state, self.times, self.xy),
            (self.features.flip(2), self.state, self.times, self.xy),
        )
        for arguments in variants:
            self.assertFalse(torch.allclose(tokens, self.model.read_future(*arguments)))
        permuted = self.model.read_future(self.features.flip(2), self.state, self.times, self.xy.flip(0))
        torch.testing.assert_close(tokens, permuted, atol=2e-6, rtol=2e-6)
        shifted = self.model.read_future(self.features, self.state, self.times + 100, self.xy)
        torch.testing.assert_close(tokens, shifted)

    def test_validations_reject_invalid_poses_dimensions_and_context(self):
        validate_se3(torch.eye(4))
        validate_se3(self.poses[None])
        for key in ('native_dim', 'feature_dim', 'state_dim', 'effectors', 'dim', 'num_tokens', 'num_heads'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                GoalInterface(**{**dict(native_dim=12, feature_dim=5, state_dim=4, effectors=2), key: 0})
        for value in (0, -1, True, float('nan'), float('inf')):
            with self.subTest(scale=value), self.assertRaises(ValueError):
                goal_pose_loss(self.poses, self.poses, value)
        invalid = self.poses.clone()
        invalid[..., 0, 0] = -1
        for poses in (invalid, self.poses.long(), self.poses[..., :3], self.poses * float('nan')):
            with self.assertRaisesRegex(ValueError, 'goal poses'):
                self.model.encode_goal(poses)
        invalid = self.poses.clone()
        invalid[..., 3, 0] = .1
        with self.assertRaisesRegex(ValueError, 'bottom rows'):
            self.model.encode_goal(invalid)
        with self.assertRaisesRegex(ValueError, 'effectors'):
            self.model.encode_goal(self.poses[:, :1])
        for arguments in (
            (self.features[..., :4], self.state, self.times, self.xy),
            (self.features, self.state[:, :3], self.times, self.xy),
            (self.features, self.state, self.times.flip(0), self.xy),
            (self.features, self.state, self.times, self.xy * 0),
        ):
            with self.assertRaises(ValueError):
                self.model.read_future(*arguments)
        with self.assertRaisesRegex(ValueError, 'goal tokens'):
            self.model.decode_goal(torch.randn(2, 3, 15))


if __name__ == '__main__':
    unittest.main()
