"""G decoding, robot-only LIT semantics, and frozen native ownership."""

import inspect
import unittest
from unittest.mock import patch

import torch

from etude.g_pi_interface import (
    GGoalDecoder, GTranslator, PiGoalInterface, PiGoalPolicy, g_goal_loss, normalize_z, perturb_goal,
)
from etude.goal_interface import _PoseDecoder, _RecurrentGroup, validate_goal_poses


class GPiInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(123)
        self.decoder = GGoalDecoder(12, 12, 4, 2, k_z=3, dim=16, num_heads=4, num_layers=2)
        self.interface = PiGoalInterface(native_dim=12, feature_dim=12, state_dim=4, effectors=2,
            d_z=12, k_z=3, dim=16, num_tokens=4, num_heads=4, num_layers=2,
            num_layer_groups=1, num_pose_tokens=1)
        self.state = torch.randn(2, 4)
        self.language = torch.randn(2, 2, 12)
        self.features = {index: torch.randn(2, 6, 12) for index in range(2)}
        self.times = torch.tensor([0., .2, .4], dtype=torch.float64)
        self.xy = torch.tensor([[-.5, 0.], [.5, 0.]])
        self.goal = {"z": normalize_z(torch.randn(2, 3, 12)),
                     "goal_poses": torch.eye(4).repeat(2, 2, 1, 1),
                     "goal_gripper": torch.tensor([[.2, .7], [.9, .1]])}
        self.goal["goal_poses"][..., :3, 3] = torch.randn(2, 2, 3) * .2

    def conditions(self, goal=None, features=None):
        return self.interface.conditions(self.features if features is None else features,
            self.state, self.language, self.goal if goal is None else goal, self.times, self.xy)

    def test_decoder_queries_attention_order_and_pose_contract(self):
        self.assertEqual(self.decoder.queries.shape, (5, 16))
        calls, handles = [], []
        for index, block in enumerate(self.decoder.blocks):
            for name in ("self_attention", "cross_attention", "ffn"):
                handles.append(getattr(block, name).register_forward_hook(
                    lambda _module, _args, _out, label=(index, name): calls.append(label)))
        try:
            prediction = self.decoder(self.features[0], self.features[1], self.state)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(calls, [(i, name) for i in range(2)
                                for name in ("self_attention", "cross_attention", "ffn")])
        self.assertEqual(prediction["z"].shape, (2, 3, 12))
        torch.testing.assert_close(prediction["z"].norm(dim=-1), torch.ones(2, 3))
        validate_goal_poses(prediction["goal_poses"])
        self.assertTrue(((prediction["goal_gripper"] >= 0) & (prediction["goal_gripper"] <= 1)).all())

    def test_g_loss_only_updates_decoder_and_detaches_targets(self):
        features = self.features[1].clone().requires_grad_()
        state = self.state.clone().requires_grad_()
        target = {name: value.clone().requires_grad_() for name, value in self.goal.items()}
        before = {name: value.detach().clone() for name, value in self.decoder.named_parameters()}
        optimizer = torch.optim.SGD(self.decoder.parameters(), lr=.05)
        prediction = self.decoder(features, features, state)
        losses = g_goal_loss(prediction, target, pose_weight=.3)
        torch.testing.assert_close(losses["total"], losses["z"] + .3 * losses["pose_total"])
        losses["total"].backward()
        optimizer.step()
        self.assertIsNone(features.grad)
        self.assertIsNone(state.grad)
        self.assertTrue(all(value.grad is None for value in target.values()))
        self.assertTrue(any(not torch.equal(value, before[name]) for name, value in self.decoder.named_parameters()))
        self.assertTrue(all(value.grad is not None and torch.isfinite(value.grad).all()
                            for value in self.decoder.parameters()))

    def test_g_state_is_optional_decoder_context(self):
        before = self.decoder(self.features[0], self.features[1], self.state)
        after = self.decoder(self.features[0], self.features[1], self.state + 10)
        self.assertFalse(torch.allclose(before["z"], after["z"]))
        decoder = GGoalDecoder(12, 12, 4, 2, k_z=3, dim=16, num_heads=4, use_state=False)
        self.assertIsNone(decoder.state_projection)
        first = decoder(self.features[0], self.features[1], self.state)
        second = decoder(self.features[0], self.features[1], self.state + 10)
        for name in first:
            torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)

    def test_pose_readout_uses_only_its_own_effector_token(self):
        head = _PoseDecoder(16, 2, 4, 1., 2)
        tokens = torch.randn(3, 2, 16)
        first = head(tokens, per_effector=True)
        tokens[:, 1] += 10
        second = head(tokens, per_effector=True)
        for name in first:
            torch.testing.assert_close(first[name][:, 0], second[name][:, 0], atol=0, rtol=0)
            self.assertFalse(torch.equal(first[name][:, 1], second[name][:, 1]))
        with self.assertRaisesRegex(ValueError, "one token per effector"):
            head(tokens[:, :1], per_effector=True)

    def test_single_effector_decoder_matches_legacy_readout_exactly(self):
        decoder = GGoalDecoder(12, 12, 4, 1, k_z=3, dim=16, num_heads=4, num_layers=2)
        self.assertEqual(decoder.queries.shape, (4, 16))
        self.assertEqual(decoder.pose_decoder.num_pose_tokens, 1)
        current = decoder(self.features[0], self.features[1], self.state)
        original = decoder.pose_decoder.forward
        with patch.object(decoder.pose_decoder, "forward", side_effect=lambda tokens, **_: original(tokens)):
            legacy = decoder(self.features[0], self.features[1], self.state)
        for name in current:
            torch.testing.assert_close(current[name], legacy[name], atol=0, rtol=0)

    def test_decoder_can_fit_independent_dual_arm_targets(self):
        torch.manual_seed(921)
        decoder = GGoalDecoder(12, 12, 4, 2, k_z=3, dim=16, num_heads=4, num_layers=2)
        features = torch.randn(4, 6, 12)
        state = torch.eye(4)
        target = {"z": normalize_z(torch.randn(4, 3, 12)),
                  "goal_poses": torch.eye(4).repeat(4, 2, 1, 1),
                  "goal_gripper": torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])}
        target["goal_poses"][..., :3, 3] = torch.tensor([
            [[.2, -.1, .1], [-.1, .2, .3]], [[-.2, .1, .2], [.3, -.1, .1]],
            [[.1, .3, -.2], [-.3, -.2, .2]], [[-.1, -.3, .3], [.2, .1, -.1]]])
        optimizer = torch.optim.Adam(decoder.parameters(), lr=.01)
        for _ in range(400):
            optimizer.zero_grad()
            prediction = decoder(features, features, state)
            g_goal_loss(prediction, target)["total"].backward()
            optimizer.step()
        prediction = decoder(features, features, state)
        self.assertLess((prediction["goal_gripper"] - target["goal_gripper"]).abs().max().item(), .06)
        self.assertLess((prediction["goal_poses"][..., :3, 3]
                         - target["goal_poses"][..., :3, 3]).abs().max().item(), .025)

    def test_intent_all_layers_only_read_demo_and_ignore_goal_queries_robot_state(self):
        demo = self.features[0]
        first = self.decoder(demo, self.features[1], self.state)
        memories, handles = [], []
        for block in self.decoder.intent_blocks:
            handles.append(block.cross_attention.register_forward_pre_hook(
                lambda _module, args: memories.append(args[1].detach().clone())))
        with torch.no_grad():
            self.decoder.queries.add_(torch.randn_like(self.decoder.queries) * 10)
        try:
            second = self.decoder(demo, self.features[1] + 10, self.state - 10)
        finally:
            for handle in handles:
                handle.remove()
        torch.testing.assert_close(first["u"], second["u"], rtol=0, atol=0)
        self.assertFalse(torch.equal(first["z"], second["z"]))
        projected = self.decoder.intent_projection(demo)
        self.assertEqual(len(memories), 2)
        for memory in memories:
            torch.testing.assert_close(memory, projected, rtol=0, atol=0)
        self.assertFalse(torch.equal(first["u"], self.decoder.encode_intent(demo + 10)))
        self.assertEqual(set(inspect.signature(self.decoder.encode_intent).parameters), {"demo_features"})
        with self.assertRaises(TypeError):
            self.decoder(demo, self.features[1], self.state, language=self.language)

    def test_connected_goal_uses_u_and_retains_slot_order(self):
        u = self.decoder.encode_intent(self.features[0])
        first = self.decoder.decode_goal(u, self.features[1], self.state)
        changed = self.decoder.decode_goal(u + torch.randn_like(u), self.features[1], self.state)
        swapped = self.decoder.decode_goal(u.flip(1), self.features[1], self.state)
        self.assertFalse(torch.allclose(first["z"], changed["z"]))
        self.assertGreater((first["z"] - swapped["z"]).abs().max().item(), 1e-5)
        with torch.no_grad():
            self.decoder.intent_position.zero_()
        original = self.decoder.decode_goal(u, self.features[1], self.state)
        swapped = self.decoder.decode_goal(u.flip(1), self.features[1], self.state)
        torch.testing.assert_close(original["z"], swapped["z"], atol=1e-6, rtol=1e-6)
        with self.assertRaisesRegex(ValueError, "only through u"):
            self.decoder.decode_goal(u, self.features[1], self.state, demo_features=self.features[0])

    def test_regression_backpropagates_through_connected_intent_but_not_independent_head(self):
        prediction = self.decoder(self.features[0], self.features[1], self.state)
        g_goal_loss(prediction, self.goal)["total"].backward()
        self.assertGreater(self.decoder.intent_queries.grad.abs().sum().item(), 0)
        for name, parameter in self.decoder.intent_blocks.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for mode in ("independent", "regression_only"):
            decoder = GGoalDecoder(12, 12, 4, 2, k_z=3, dim=16, num_heads=4, intent_mode=mode)
            prediction = decoder(self.features[0], self.features[1], self.state)
            g_goal_loss(prediction, self.goal)["total"].backward()
            self.assertGreater(decoder.queries.grad.abs().sum().item(), 0)
            if mode == "independent":
                self.assertIsNone(decoder.intent_queries.grad)
                with torch.no_grad():
                    decoder.intent_queries.add_(100 * torch.randn_like(decoder.intent_queries))
                second = decoder(self.features[0], self.features[1], self.state)
                for name in self.goal:
                    torch.testing.assert_close(prediction[name], second[name], rtol=0, atol=0)
                self.assertFalse(torch.equal(prediction["u"], second["u"]))
            else:
                self.assertIsNone(prediction["u"])
                self.assertFalse(hasattr(decoder, "intent_queries"))
                with self.assertRaisesRegex(ValueError, "no intent queries"):
                    decoder.encode_intent(self.features[0])

    def test_contrastive_and_regression_share_live_intent_parameters(self):
        from types import SimpleNamespace
        from etude.g_pi_intent import intent_contrastive_loss, ordered_intent_similarity
        demo = torch.randn(4, 6, 12, requires_grad=True)
        entries = [SimpleNamespace(demo_id=str(i), component=str(i), purpose_group=str(i // 2))
                   for i in range(4)]
        u = self.decoder.encode_intent(demo)
        contrastive = intent_contrastive_loss(u, entries)
        contrastive.backward()
        self.assertGreater(self.decoder.intent_queries.grad.abs().sum().item(), 0)
        self.assertGreater(self.decoder.intent_projection[0].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(self.decoder.queries.grad)
        self.assertIsNone(demo.grad)
        self.decoder.zero_grad(set_to_none=True)
        g_goal_loss(self.decoder(demo[:2], self.features[1], self.state), self.goal)["total"].backward()
        self.assertGreater(self.decoder.intent_queries.grad.abs().sum().item(), 0)
        self.assertGreater(self.decoder.intent_projection[0].weight.grad.abs().sum().item(), 0)
        role_program = torch.eye(3).unsqueeze(0)
        ordered = torch.cat((role_program, role_program.flip(1), role_program.roll(1, dims=1)), dim=0)
        similarity = ordered_intent_similarity(ordered)
        torch.testing.assert_close(similarity.diag(), torch.ones(3))
        self.assertLess(similarity[0, 1].item(), .5)
        self.assertLess(similarity[0, 2].item(), .5)

    def test_pi_semantics_reuse_pose_encoder_and_keep_actions_separate(self):
        semantic = self.interface.goal_semantic(self.language, self.state, self.goal)
        self.assertEqual(semantic.shape, (2, 2 + 1 + 3 + 4, 12))
        torch.testing.assert_close(semantic[:, :2], self.language)
        torch.testing.assert_close(semantic[:, 3:6], self.interface.z_projection(self.goal["z"])
                                   + self.interface.z_position)
        pose = self.interface.encode_goal(self.goal["goal_poses"], self.goal["goal_gripper"])
        torch.testing.assert_close(semantic[:, 6:], self.interface.condition_adapter(pose))
        conditions = self.conditions()
        self.assertEqual(len(conditions), 2)
        self.assertTrue(all(value.shape == (2, 2 + 1 + 4, 12) for value in conditions))
        self.assertIsInstance(self.interface.pose_decoder, _PoseDecoder)
        self.assertEqual(self.interface.pose_decoder.num_pose_tokens, self.interface.num_tokens)
        self.assertTrue(all(isinstance(group, _RecurrentGroup) for group in self.interface.recurrent_groups))
        for value in conditions:
            torch.testing.assert_close(value[:, :2], self.language)
            torch.testing.assert_close(value[:, 2:3], self.interface.state_encoder(self.state))

    def test_endpoint_prior_has_independent_encoder_and_never_reads_subgoal_or_images(self):
        first = {id(value) for value in self.interface.goal_encoder.parameters()}
        second = {id(value) for value in self.interface.endpoint_encoder.parameters()}
        self.assertFalse(first & second)
        poses = self.goal["goal_poses"].clone().requires_grad_()
        gripper = self.goal["goal_gripper"].clone().requires_grad_()
        with patch.object(self.interface, "goal_semantic", side_effect=AssertionError("subgoal read")), \
             patch.object(self.interface, "read_layer", side_effect=AssertionError("visual read")), \
             patch.object(self.interface, "decode_endpoint", side_effect=AssertionError("auxiliary readout")):
            conditions = self.interface.endpoint_conditions(poses, gripper, self.state, self.language)
        self.assertEqual(len(conditions), self.interface.num_layers)
        expected = self.interface.condition(self.interface.encode_endpoint(poses, gripper), self.state, self.language)
        for value in conditions:
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
            torch.testing.assert_close(value[:, :2], self.language)
        self.assertEqual(self.interface.encode_endpoint(poses, gripper).shape, (2, 4, 16))
        conditions[-1].square().mean().backward()
        for module in (self.interface.endpoint_encoder, self.interface.state_encoder, self.interface.condition_adapter):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None), 0)
        self.assertTrue(all(p.grad is None for module in (self.interface.goal_encoder, self.interface.recurrent_groups,
                            self.interface.pose_decoder) for p in module.parameters()))
        self.assertIsNone(self.interface.visual_queries.grad)
        self.assertIsNone(poses.grad)
        self.assertIsNone(gripper.grad)
        self.assertEqual(set(inspect.signature(self.interface.endpoint_conditions).parameters),
                         {"poses", "gripper", "state", "language_hidden"})

    def test_clean_endpoint_readout_uses_final_visual_tokens_with_live_gradients(self):
        clean_endpoint = self.goal["goal_poses"].clone()
        clean_endpoint[..., :3, 3] += .2
        captured = []
        hook = self.interface.pose_decoder.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
        try:
            conditions, tokens = self.interface.conditions(self.features, self.state, self.language,
                self.goal, self.times, self.xy, return_tokens=True)
            self.assertFalse(captured)
            predicted = self.interface.decode_endpoint(tokens)
        finally:
            hook.remove()
        self.assertIs(captured[0], tokens)
        self.assertEqual(tokens.shape, (2, 4, 16))
        self.assertEqual(self.interface.pose_decoder.queries.shape, (2, 16))
        self.assertEqual(self.interface.pose_decoder.num_pose_tokens, 4)
        torch.testing.assert_close(conditions[-1][:, 3:], self.interface.condition_adapter(tokens))
        from etude.goal_interface import goal_pose_loss
        loss = goal_pose_loss(predicted["goal_poses"], clean_endpoint,
            gripper_prediction=predicted["goal_gripper"], gripper_target=self.goal["goal_gripper"])["total"]
        loss.backward()
        for module in (self.interface.recurrent_groups, self.interface.pose_decoder):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None), 0)
        self.assertGreater(self.interface.visual_queries.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in self.interface.endpoint_encoder.parameters()))
        self.assertEqual(set(inspect.signature(self.interface.decode_endpoint).parameters), {"tokens"})
        joint_conditions, joint_prediction = self.interface.stage2_conditions(self.features, self.state,
            self.language, self.goal, self.times, self.xy)
        for first, second in zip(conditions, joint_conditions):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        for name in predicted:
            torch.testing.assert_close(predicted[name], joint_prediction[name], rtol=0, atol=0)

    def test_endpoint_decoder_can_fit_independent_bimanual_block_ends(self):
        from etude.goal_interface import goal_pose_loss
        torch.manual_seed(624)
        interface = PiGoalInterface(native_dim=12, feature_dim=12, state_dim=4, effectors=2,
            d_z=12, k_z=3, dim=16, num_tokens=4, num_heads=4, num_layers=2,
            num_layer_groups=1, num_pose_tokens=1)
        self.assertEqual(interface.pose_decoder.num_pose_tokens, 4)
        workspace = torch.randn(4, 4, 16)
        poses = torch.eye(4).repeat(4, 2, 1, 1)
        poses[..., 0, 3] = torch.tensor([[-.2, -.3], [-.2, .3], [.2, -.3], [.2, .3]])
        gripper = torch.tensor([[.1, .1], [.1, .9], [.9, .1], [.9, .9]])
        optimizer = torch.optim.Adam(interface.pose_decoder.parameters(), lr=.01)
        for _ in range(300):
            optimizer.zero_grad()
            prediction = interface.decode_endpoint(workspace)
            loss = goal_pose_loss(prediction["goal_poses"], poses,
                gripper_prediction=prediction["goal_gripper"], gripper_target=gripper)
            loss["total"].backward()
            optimizer.step()
        prediction = interface.decode_endpoint(workspace)
        self.assertLess((prediction["goal_poses"][..., :3, 3] - poses[..., :3, 3]).abs().max().item(), .025)
        self.assertLess((prediction["goal_gripper"] - gripper).abs().max().item(), .06)

    def test_goal_position_embeddings_distinguish_permuted_tokens(self):
        self.assertEqual(self.interface.z_position.shape, (1, 3, 12))
        baseline = self.conditions()
        flipped = {**self.goal, "z": self.goal["z"].flip(1)}
        changed = self.conditions(flipped)
        self.assertFalse(torch.allclose(baseline[-1], changed[-1]))
        with torch.no_grad():
            self.interface.z_position.zero_()
        torch.testing.assert_close(self.conditions()[-1], self.conditions(flipped)[-1],
                                   atol=1e-6, rtol=1e-6)

    def test_pi_each_goal_component_and_early_visual_layer_change_action_conditions(self):
        baseline = self.conditions()
        for key in self.goal:
            goal = {name: value.clone() for name, value in self.goal.items()}
            if key == "goal_poses":
                goal[key][..., 0, 3] += 1
            else:
                goal[key] = 1 - goal[key]
            self.assertFalse(torch.allclose(baseline[-1], self.conditions(goal)[-1]), key)
        features = {index: value.clone() for index, value in self.features.items()}
        features[0] *= -3
        changed = self.conditions(features=features)
        self.assertFalse(torch.allclose(baseline[0], changed[0]))
        self.assertFalse(torch.allclose(baseline[1], changed[1]))

    def test_pi_update_detaches_visual_and_goal_labels(self):
        features = {index: value.clone().requires_grad_() for index, value in self.features.items()}
        goal = {name: value.clone().requires_grad_() for name, value in self.goal.items()}
        before = {name: value.detach().clone() for name, value in self.interface.named_parameters()}
        optimizer = torch.optim.Adam(self.interface.parameters(), lr=.01)
        conditions = self.conditions(goal, features)
        sum(value.square().mean() for value in conditions).backward()
        optimizer.step()
        self.assertTrue(all(value.grad is None for value in features.values()))
        self.assertTrue(all(value.grad is None for value in goal.values()))
        changed = {name for name, value in self.interface.named_parameters() if not torch.equal(value, before[name])}
        for prefix in ("goal_encoder.", "z_projection.", "z_position", "state_encoder.", "condition_adapter.",
                       "recurrent_groups.", "visual_queries"):
            self.assertTrue(any(name.startswith(prefix) for name in changed), prefix)

    def test_pi_has_no_demo_parameter(self):
        for method in (PiGoalInterface.goal_semantic, PiGoalInterface.conditions, PiGoalPolicy.predict):
            self.assertFalse(any("demo" in name for name in inspect.signature(method).parameters))
        with self.assertRaises(TypeError):
            self.interface.conditions(self.features, self.state, self.language, self.goal,
                                      self.times, self.xy, demonstration=torch.zeros(1))

    def test_noise_disabled_and_reproducible_small_pose_noise(self):
        original = {name: value.clone() for name, value in self.goal.items()}
        zero = perturb_goal(self.goal)
        for name, value in zero.items():
            torch.testing.assert_close(value, original[name])
        options = dict(z_std=.02, translation_std=.001, rotation_std=.01, gripper_std=.005,
                       translation_max_m=.005, rotation_max_deg=3., candidate_separation_m=.1)
        first = perturb_goal(self.goal, **options, generator=torch.Generator().manual_seed(19))
        second = perturb_goal(self.goal, **options, generator=torch.Generator().manual_seed(19))
        validate_goal_poses(first["goal_poses"])
        for name in self.goal:
            torch.testing.assert_close(self.goal[name], original[name], rtol=0, atol=0)
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
            self.assertFalse(torch.equal(first[name], zero[name]))

    def test_validation_rejects_broadcasting_and_inconsistent_layer_grids(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            GGoalDecoder(12, 12, 4, 2, k_z=0)
        with self.assertRaisesRegex(ValueError, "divisible"):
            GGoalDecoder(12, 12, 4, 2, dim=15, num_heads=4)
        with self.assertRaisesRegex(ValueError, "state"):
            self.decoder(self.features[0], self.features[1], self.state[:, :1])
        with self.assertRaisesRegex(ValueError, "matching"):
            g_goal_loss(self.goal, {**self.goal, "z": self.goal["z"][:1]})
        with self.assertRaisesRegex(ValueError, "pose_weight"):
            g_goal_loss(self.goal, self.goal, pose_weight=float("nan"))
        with self.assertRaisesRegex(ValueError, "K_z"):
            self.conditions({**self.goal, "z": self.goal["z"][:, :1]})
        with self.assertRaisesRegex(ValueError, "poses"):
            self.conditions({**self.goal, "goal_poses": None})
        with self.assertRaisesRegex(ValueError, "depth order"):
            self.conditions(features={1: self.features[1], 0: self.features[0]})
        with self.assertRaisesRegex(ValueError, "patch grid"):
            self.conditions(features={0: self.features[0][:, :-1], 1: self.features[1]})
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            perturb_goal(self.goal, rotation_std=-.1)


class GPiNativeInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from etude.zerowam import NativeDependencyError, load_native_class
        try:
            load_native_class()
        except NativeDependencyError as exc:
            raise unittest.SkipTest(str(exc))

    def fixture(self, device="cpu"):
        from test_native_icl import tiny_model
        from etude.g_pi_context import FrozenGoalEncoder, install_empty_text
        from etude.g_pi_training import _set_precision
        from etude.goal_action import action_named_parameters, install_action_interface
        torch.manual_seed(243)
        native = install_action_interface(tiny_model(device).float()).requires_grad_(False).eval()
        _set_precision(native, video_precision="bfloat16", route="pi_goal")
        install_empty_text(native, torch.randn(1, 2, 8, device=device), "unit-test empty-text fixture")
        encoder = FrozenGoalEncoder(native, layer=1, grid_size=(1, 3), base_id="native-fixture",
                                    camera_layout=[{"name": "camera", "token_width": 2}])
        for _, parameter in action_named_parameters(native):
            parameter.requires_grad_(True)
        decoder = GGoalDecoder(36, 36, 4, 1, k_z=3, dim=16, num_heads=4, num_layers=2).to(device)
        interface = PiGoalInterface(native_dim=36, feature_dim=36, state_dim=4, effectors=1,
            d_z=36, k_z=3, dim=16, num_tokens=4, num_heads=4, num_layers=2,
            num_layer_groups=1, num_pose_tokens=1).to(device)
        config = dict(chunk_size=1, max_frame_chunk_size=4, icl_rope_h=4, window_size=8,
                      action_sampling_steps=1, latent_frame_dt=.2)
        demo = torch.randn(1, 4, 2, 1, 2, device=device)
        history = torch.randn(1, 4, 3, 1, 2, device=device)
        state = torch.randn(1, 4, device=device)
        language = torch.randn(1, 2, 8, device=device)
        return native, encoder, decoder, interface, config, demo, history, state, language

    def test_wrappers_only_enable_expected_parameters_and_keep_video_eval(self):
        native, encoder, decoder, interface, config, demo, history, state, language = self.fixture()
        self.assertIs(encoder.native, native)
        self.assertEqual(list(encoder.parameters()), [])
        self.assertEqual(dict(encoder.state_dict()), {})
        native.train()
        translator = GTranslator(encoder.native, decoder, config)
        translator.train()
        self.assertIs(translator.native, native)
        self.assertTrue(native.training)
        self.assertTrue(translator.goal_decoder.training)
        self.assertTrue(all(name.startswith("goal_decoder.") for name, value in translator.named_parameters()
                            if value.requires_grad))
        action_flags = [(parameter, parameter.requires_grad) for parameter in native.parameters()]
        encoder.train()
        self.assertTrue(native.training)
        self.assertTrue(all(parameter.requires_grad == flag for parameter, flag in action_flags))
        policy = PiGoalPolicy(native, interface, config, action_shape=(1, 3, 1, 2, 1),
                              actions_mask=torch.ones(1, 3, 1, 2, 1, dtype=torch.bool))
        policy.train()
        self.assertIs(policy.native, policy.video_native)
        self.assertIs(policy.native, translator.native)
        self.assertNotIn("video_native", policy._modules)
        self.assertFalse(policy.video_native.training)
        self.assertTrue(policy.interface.training)
        from copy import deepcopy
        with self.assertRaisesRegex(ValueError, "same native"):
            PiGoalPolicy(native, interface, config, action_shape=(1, 3, 1, 2, 1),
                         actions_mask=policy.actions_mask, video_native=deepcopy(native))
        from etude.goal_action import action_named_parameters
        action_ids = {id(parameter) for _, parameter in action_named_parameters(native)}
        self.assertTrue(all(parameter.dtype == (torch.float32 if id(parameter) in action_ids else torch.bfloat16)
                            for parameter in native.parameters()))
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in interface.parameters()))
        with self.assertRaises(TypeError):
            policy.predict(history, state, language, {}, demonstration=demo)

    def test_translator_validates_demo_route_and_connected_intent_requirement(self):
        native, encoder, decoder, interface, config, demo, history, state, language = self.fixture()
        with self.assertRaisesRegex(ValueError, "demo_route"):
            GTranslator(native, decoder, dict(config, demo_route="unknown"))
        GTranslator(native, decoder, dict(config, demo_route="via_u_only"))
        for mode in ("independent", "regression_only"):
            other = GGoalDecoder(36, 36, 4, 1, k_z=3, dim=16, num_heads=4, intent_mode=mode)
            with self.assertRaisesRegex(ValueError, "requires a connected"):
                GTranslator(native, other, dict(config, demo_route="via_u_only"))
            GTranslator(native, other, config)

    def test_policy_physically_truncates_before_native_read(self):
        native, encoder, decoder, interface, config, demo, history, state, language = self.fixture()
        policy = PiGoalPolicy(native, interface, config, action_shape=(1, 3, 1, 2, 1),
                              actions_mask=torch.ones(1, 3, 1, 2, 1, dtype=torch.bool), video_native=encoder.native)
        goal = decoder(torch.randn(1, 4, 36), torch.randn(1, 4, 36), state)
        history[:, :, 2:] = float("nan")
        features = {index: torch.randn(1, 4, 36) for index in range(2)}
        with patch("etude.g_pi_context.pi_context_features", return_value=features) as read, \
             patch("etude.g_pi_interface.goal_action_sample", return_value=torch.zeros(1, 3, 1, 2, 1)):
            policy.predict(history, state, language, goal, current_index=1)
        observed = read.call_args.args[1]
        self.assertEqual(observed.shape[2], 2)
        self.assertTrue(torch.isfinite(observed).all())
        del policy.config["latent_frame_dt"]
        with patch("etude.g_pi_context.pi_context_features", return_value=features), \
             self.assertRaisesRegex(ValueError, "control_dt"):
            policy.predict(history, state, language, goal, current_index=1)
        interface.control_dt = .1
        with patch("etude.g_pi_context.pi_context_features", return_value=features), \
             patch("etude.g_pi_interface.goal_action_sample", return_value=torch.zeros(1, 3, 1, 2, 1)), \
             patch.object(interface, "conditions", wraps=interface.conditions) as condition:
            policy.predict(history, state, language, goal, current_index=1)
        torch.testing.assert_close(condition.call_args.args[4], torch.tensor([0., .2], dtype=torch.float64))
        interface.latent_frame_dt = .3
        with patch("etude.g_pi_context.pi_context_features", return_value=features), \
             patch("etude.g_pi_interface.goal_action_sample", return_value=torch.zeros(1, 3, 1, 2, 1)), \
             patch.object(interface, "conditions", wraps=interface.conditions) as condition:
            policy.predict(history, state, language, goal, current_index=1)
        torch.testing.assert_close(condition.call_args.args[4], torch.tensor([0., .3], dtype=torch.float64))

    def test_policy_optional_language_uses_the_same_fixed_empty_prompt(self):
        native, encoder, decoder, interface, config, demo, history, state, language = self.fixture()
        policy = PiGoalPolicy(native, interface, config, action_shape=(1, 3, 1, 2, 1),
                              actions_mask=torch.ones(1, 3, 1, 2, 1, dtype=torch.bool))
        goal = decoder(torch.randn(1, 4, 36), torch.randn(1, 4, 36), state)
        features = {index: torch.randn(1, 4, 36) for index in range(2)}
        inputs = []
        projection = native.condition_embedder_action.text_embedder
        hook = projection.register_forward_pre_hook(lambda _module, args: inputs.append(args[0].clone()))
        try:
            with patch("etude.g_pi_context.pi_context_features", return_value=features), \
                 patch("etude.g_pi_interface.goal_action_sample",
                       return_value=torch.zeros(1, 3, 1, 2, 1)) as sample:
                policy.predict(history, state, goal=goal, current_index=1)
                absent = sample.call_args.args[1]
                policy.predict(history, state, native.g_pi_empty_text, goal, current_index=1)
                explicit = sample.call_args.args[1]
                policy.predict(history, state, language, goal, current_index=1)
        finally:
            hook.remove()
        self.assertEqual(len(inputs), 3)
        torch.testing.assert_close(inputs[0], native.g_pi_empty_text.float(), atol=0, rtol=0)
        torch.testing.assert_close(inputs[0], inputs[1], atol=0, rtol=0)
        torch.testing.assert_close(inputs[2], language, atol=0, rtol=0)
        for first, second in zip(absent, explicit):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "requires goal"):
            policy.predict(history, state)

    def test_endpoint_diagnostic_is_opt_in_and_does_not_change_action_rng(self):
        native, encoder, decoder, interface, config, demo, history, state, language = self.fixture()
        policy = PiGoalPolicy(native, interface, config, action_shape=(1, 3, 1, 2, 1),
                              actions_mask=torch.ones(1, 3, 1, 2, 1, dtype=torch.bool))
        goal = decoder(torch.randn(1, 4, 36), torch.randn(1, 4, 36), state)
        features = {index: torch.randn(1, 6, 36) for index in range(2)}
        def sample(_native, _conditions, shape, _mask, generator, **kwargs):
            return torch.randn(shape, generator=generator)
        with patch("etude.g_pi_context.pi_context_features", return_value=features), \
             patch("etude.g_pi_interface.goal_action_sample", side_effect=sample), \
             patch.object(interface.endpoint_encoder, "forward", side_effect=AssertionError("prior read")), \
             patch.object(interface.pose_decoder, "forward", wraps=interface.pose_decoder.forward) as readout:
            seed = policy.generator.get_state()
            ordinary = policy.predict(history, state, language, goal)
            self.assertEqual(readout.call_count, 0)
            policy.generator.set_state(seed)
            diagnostic = policy.predict(history, state, language, goal, return_endpoint=True)
            self.assertEqual(readout.call_count, 1)
        torch.testing.assert_close(diagnostic["actions"], ordinary, rtol=0, atol=0)
        before = policy.generator.get_state()
        with patch("etude.g_pi_context.pi_context_features", return_value=features), \
             patch("etude.g_pi_interface.goal_action_sample", side_effect=AssertionError("action sampled")), \
             patch.object(interface.endpoint_encoder, "forward", side_effect=AssertionError("prior read")):
            standalone = policy.diagnose_endpoint(history, state, language, goal)
        torch.testing.assert_close(policy.generator.get_state(), before, rtol=0, atol=0)
        for name in standalone:
            torch.testing.assert_close(standalone[name], diagnostic["endpoint"][name], rtol=0, atol=0)
        self.assertEqual(set(diagnostic["endpoint"]), {"goal_poses", "goal_gripper"})
        validate_goal_poses(diagnostic["endpoint"]["goal_poses"])
        with self.assertRaisesRegex(ValueError, "return_endpoint"):
            policy.predict(history, state, language, goal, return_endpoint="yes")

    def test_intent_diagnostic_holds_computed_robot_memory_fixed(self):
        native, encoder, decoder, interface, config, demo, history, state, language = self.fixture()
        translator = GTranslator(native, decoder, config)
        demo_memory = torch.randn(1, 4, 36)
        robot_memory = torch.randn(1, 6, 36)
        isolated_memory = robot_memory + 5
        with patch("etude.g_pi_context.split_g_context_features",
                   return_value=({1: demo_memory}, {1: robot_memory})), \
             patch("etude.g_pi_context.pi_context_features", return_value={1: isolated_memory}), \
             patch.object(decoder, "decode_goal", wraps=decoder.decode_goal) as read:
            results = translator.diagnose_intent(demo, history, state, use_cache=False)
        self.assertIs(read.call_args_list[0].args[1], robot_memory)
        self.assertIs(read.call_args_list[1].args[1], robot_memory)
        self.assertIs(read.call_args_list[2].args[1], isolated_memory)
        torch.testing.assert_close(read.call_args_list[0].args[0], read.call_args_list[2].args[0],
                                   atol=0, rtol=0)
        self.assertEqual(set(results), {"baseline", "u_permuted", "robot_without_demo"})
        for result in results.values():
            self.assertEqual(set(result), {"z", "goal_poses", "goal_gripper"})
        self.assertFalse(torch.allclose(results["baseline"]["z"], results["u_permuted"]["z"]))
        self.assertFalse(torch.allclose(results["baseline"]["z"], results["robot_without_demo"]["z"]))

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_native_g_step_changes_decoder_only_and_E_stays_exact(self):
        from etude.g_pi_context import frozen_base_checksum, split_g_context_features
        native, encoder, decoder, interface, config, demo, history, state, language = self.fixture("cuda")
        translator = GTranslator(encoder.native, decoder, config)
        checksum = frozen_base_checksum(encoder.native)
        native_before = {name: value.detach().clone() for name, value in native.state_dict().items()}
        original = {name: value.detach().clone() for name, value in translator.named_parameters()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            before_z = encoder(history[:, :, -1:])
            demo_features, robot_features = split_g_context_features(
                encoder.native, demo, history[:, :, :2], config)
            prediction = decoder(demo_features[1], robot_features[1], state)
            target = {"z": before_z, "goal_poses": torch.eye(4, device="cuda").repeat(1, 1, 1, 1),
                      "goal_gripper": torch.ones(1, 1, device="cuda")}
            loss = g_goal_loss(prediction, target)["total"]
        optimizer = torch.optim.Adam(decoder.parameters(), lr=.01)
        loss.backward()
        optimizer.step()
        changed = {name for name, value in translator.named_parameters() if not torch.equal(value, original[name])}
        self.assertTrue(changed)
        self.assertTrue(all(name.startswith("goal_decoder.") for name in changed))
        self.assertTrue(all(value.dtype == torch.float32 and value.grad.dtype == torch.float32
                            for value in decoder.parameters()))
        self.assertTrue(all(torch.equal(value, native_before[name]) for name, value in native.state_dict().items()))
        self.assertEqual(frozen_base_checksum(encoder.native), checksum)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            torch.testing.assert_close(encoder(history[:, :, -1:]), before_z, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_native_pi_step_changes_only_interface_and_action_expert(self):
        from etude.g_pi_context import frozen_base_checksum, pi_context_features
        from etude.goal_action import action_named_parameters, goal_action_forward
        from etude.g_pi_training import _set_training
        from etude.goal_interface import goal_pose_loss
        native, encoder, decoder, interface, config, demo, history, state, language = self.fixture("cuda")
        _set_training(native, interface, encoder, {"interface_type": "pi_goal",
            "pi_training": {"ablations": {"no_stage1": True, "exact_goal": True}}}, stage="pi")
        action_parameters = dict(action_named_parameters(native))
        stage2_parameters = [value for name, value in interface.named_parameters()
                             if not name.startswith("endpoint_encoder.")]
        self.assertTrue(all(value.requires_grad for value in stage2_parameters))
        original = {name: value.detach().clone() for name, value in native.named_parameters()}
        decoder_before = {name: value.detach().clone() for name, value in decoder.named_parameters()}
        interface_before = {name: value.detach().clone() for name, value in interface.named_parameters()}
        checksum = frozen_base_checksum(encoder.native)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            before_z = encoder(history[:, :, -1:])
            goal = {"z": before_z, "goal_poses": torch.eye(4, device="cuda").repeat(1, 1, 1, 1),
                    "goal_gripper": torch.ones(1, 1, device="cuda")}
            features = pi_context_features(encoder.native, history[:, :, :2], config)
            conditions, endpoint = interface.stage2_conditions(features, state,
                native.condition_embedder_action.text_embedder(language), goal,
                torch.tensor([0., .2]), torch.tensor([[-.5, 0.], [.5, 0.]]))
            predicted = goal_action_forward(native, torch.randn(1, 3, 1, 2, 1, device="cuda"),
                                            torch.full((1, 1), 500., device="cuda"), conditions)
            clean_endpoint = goal["goal_poses"].clone()
            clean_endpoint[..., 0, 3] = .03
            pose = goal_pose_loss(endpoint["goal_poses"], clean_endpoint,
                gripper_prediction=endpoint["goal_gripper"], gripper_target=torch.full_like(goal["goal_gripper"], .25))
            loss = predicted.float().square().mean() + .3 * pose["total"]
        optimizer = torch.optim.Adam([*stage2_parameters, *action_parameters.values()], lr=.01)
        loss.backward()
        optimizer.step()
        changed = {name for name, value in native.named_parameters() if not torch.equal(value, original[name])}
        self.assertTrue(changed)
        self.assertTrue(changed <= action_parameters.keys())
        self.assertTrue(all(value.dtype == torch.float32 and value.grad is not None
                            and value.grad.dtype == torch.float32 for value in action_parameters.values()))
        self.assertTrue(all(value.dtype == torch.float32 and value.grad is not None
                            and value.grad.dtype == torch.float32 for value in stage2_parameters))
        for name, value in interface.named_parameters():
            if name.startswith("endpoint_encoder."):
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad)
                torch.testing.assert_close(value, interface_before[name], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(value, interface_before[name]) for name, value in interface.named_parameters()
                            if name.startswith("pose_decoder.")))
        self.assertTrue(all(torch.equal(value, decoder_before[name]) for name, value in decoder.named_parameters()))
        self.assertTrue(any(not torch.equal(value, interface_before[name]) for name, value in interface.named_parameters()))
        self.assertEqual(frozen_base_checksum(encoder.native), checksum)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            torch.testing.assert_close(encoder(history[:, :, -1:]), before_z, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_predict_cache_and_future_isolation_with_fixed_goal_and_noise(self):
        native, encoder, decoder, interface, config, demo, history, state, language = self.fixture("cuda")
        translator = GTranslator(encoder.native, decoder, config)
        policy = PiGoalPolicy(native, interface, config, action_shape=(1, 3, 1, 2, 1),
                              actions_mask=torch.ones(1, 3, 1, 2, 1, dtype=torch.bool), video_native=encoder.native)
        goal = translator.predict(demo, history, state, current_index=1, use_cache=False)
        cached = translator.predict(demo, history, state, current_index=1)
        for name in goal:
            torch.testing.assert_close(cached[name], goal[name], atol=.015, rtol=.015)
        seed = policy.generator.get_state()
        first = policy.predict(history, state, language, goal, current_index=1)
        history[:, :, 2:] = float("nan")
        policy.generator.set_state(seed)
        second = policy.predict(history, state, language, goal, current_index=1)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        after = translator.predict(demo, history, state, current_index=1)
        for name in goal:
            torch.testing.assert_close(after[name], cached[name], rtol=0, atol=0)
        policy.generator.set_state(seed)
        absent = policy.predict(history, state, goal=goal, current_index=1)
        policy.generator.set_state(seed)
        empty = policy.predict(history, state, native.g_pi_empty_text, goal, current_index=1)
        torch.testing.assert_close(absent, empty, rtol=0, atol=0)
