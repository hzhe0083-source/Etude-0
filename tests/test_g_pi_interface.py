"""G decoding, robot-only LIT semantics, and frozen native ownership."""

import inspect
import unittest
from unittest.mock import patch

import torch

from evo_wam.g_pi_interface import (
    GGoalDecoder, GTranslator, PiGoalInterface, PiGoalPolicy, g_goal_loss, normalize_z, perturb_goal,
)
from evo_wam.goal_interface import _RecurrentGroup, validate_goal_poses


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
        self.assertEqual(self.decoder.queries.shape, (4, 16))
        calls, handles = [], []
        for index, block in enumerate(self.decoder.blocks):
            for name in ("self_attention", "cross_attention", "ffn"):
                handles.append(getattr(block, name).register_forward_hook(
                    lambda _module, _args, _out, label=(index, name): calls.append(label)))
        try:
            prediction = self.decoder(self.features[1], self.state)
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
        prediction = self.decoder(features, state)
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
        before = self.decoder(self.features[1], self.state)
        after = self.decoder(self.features[1], self.state + 10)
        self.assertFalse(torch.allclose(before["z"], after["z"]))
        decoder = GGoalDecoder(12, 12, 4, 2, k_z=3, dim=16, num_heads=4, use_state=False)
        self.assertIsNone(decoder.state_projection)
        first = decoder(self.features[1], self.state)
        second = decoder(self.features[1], self.state + 10)
        for name in first:
            torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)

    def test_pi_semantics_reuse_pose_encoder_and_keep_actions_separate(self):
        semantic = self.interface.goal_semantic(self.language, self.state, self.goal)
        self.assertEqual(semantic.shape, (2, 2 + 1 + 3 + 4, 12))
        torch.testing.assert_close(semantic[:, :2], self.language)
        torch.testing.assert_close(semantic[:, 3:6], self.interface.z_projection(self.goal["z"]))
        pose = self.interface.encode_goal(self.goal["goal_poses"], self.goal["goal_gripper"])
        torch.testing.assert_close(semantic[:, 6:], self.interface.condition_adapter(pose))
        conditions = self.conditions()
        self.assertEqual(len(conditions), 2)
        self.assertTrue(all(value.shape == (2, 2 + 1 + 4, 12) for value in conditions))
        self.assertFalse(hasattr(self.interface, "pose_decoder"))
        self.assertTrue(all(isinstance(group, _RecurrentGroup) for group in self.interface.recurrent_groups))
        for value in conditions:
            torch.testing.assert_close(value[:, :2], self.language)
            torch.testing.assert_close(value[:, 2:3], self.interface.state_encoder(self.state))

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
        for prefix in ("goal_encoder.", "z_projection.", "state_encoder.", "condition_adapter.",
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
        options = dict(z_std=.02, translation_std=.001, rotation_std=.01, gripper_std=.005)
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
            self.decoder(self.features[0], self.state[:, :1])
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
        from evo_wam.zerowam import NativeDependencyError, load_native_class
        try:
            load_native_class()
        except NativeDependencyError as exc:
            raise unittest.SkipTest(str(exc))

    def fixture(self, device="cpu"):
        from test_native_icl import tiny_model
        from evo_wam.g_pi_context import FrozenGoalEncoder, install_empty_text
        from evo_wam.goal_action import action_named_parameters, install_action_interface
        torch.manual_seed(243)
        native = install_action_interface(tiny_model(device).float()).requires_grad_(False).eval()
        install_empty_text(native, torch.randn(1, 2, 8, device=device), "unit-test empty-text fixture")
        encoder = FrozenGoalEncoder(native, layer=1, k_z=3, base_id="native-fixture")
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
        translator = GTranslator(encoder.native, decoder, config)
        translator.train()
        self.assertFalse(translator.native.training)
        self.assertTrue(translator.goal_decoder.training)
        self.assertTrue(all(name.startswith("goal_decoder.") for name, value in translator.named_parameters()
                            if value.requires_grad))
        policy = PiGoalPolicy(native, interface, config, action_shape=(1, 3, 1, 2, 1),
                              actions_mask=torch.ones(1, 3, 1, 2, 1, dtype=torch.bool), video_native=encoder.native)
        policy.train()
        self.assertFalse(policy.video_native.training)
        self.assertTrue(policy.interface.training)
        with self.assertRaisesRegex(ValueError, "frozen"):
            PiGoalPolicy(native, interface, config, action_shape=(1, 3, 1, 2, 1), actions_mask=policy.actions_mask)
        with self.assertRaises(TypeError):
            policy.predict(history, state, language, {}, demonstration=demo)

    def test_policy_physically_truncates_before_native_read(self):
        native, encoder, decoder, interface, config, demo, history, state, language = self.fixture()
        policy = PiGoalPolicy(native, interface, config, action_shape=(1, 3, 1, 2, 1),
                              actions_mask=torch.ones(1, 3, 1, 2, 1, dtype=torch.bool), video_native=encoder.native)
        goal = decoder(torch.randn(1, 4, 36), state)
        history[:, :, 2:] = float("nan")
        features = {index: torch.randn(1, 4, 36) for index in range(2)}
        with patch("evo_wam.g_pi_context.pi_context_features", return_value=features) as read, \
             patch("evo_wam.g_pi_interface.goal_action_sample", return_value=torch.zeros(1, 3, 1, 2, 1)):
            policy.predict(history, state, language, goal, current_index=1)
        observed = read.call_args.args[1]
        self.assertEqual(observed.shape[2], 2)
        self.assertTrue(torch.isfinite(observed).all())
        del policy.config["latent_frame_dt"]
        with patch("evo_wam.g_pi_context.pi_context_features", return_value=features), \
             self.assertRaisesRegex(ValueError, "control_dt"):
            policy.predict(history, state, language, goal, current_index=1)
        interface.control_dt = .2
        with patch("evo_wam.g_pi_context.pi_context_features", return_value=features), \
             patch("evo_wam.g_pi_interface.goal_action_sample", return_value=torch.zeros(1, 3, 1, 2, 1)), \
             patch.object(interface, "conditions", wraps=interface.conditions) as condition:
            policy.predict(history, state, language, goal, current_index=1)
        torch.testing.assert_close(condition.call_args.args[4], torch.tensor([0., .2], dtype=torch.float64))

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_native_g_step_changes_decoder_only_and_E_stays_exact(self):
        from evo_wam.g_pi_context import frozen_base_checksum, g_context_features
        native, encoder, decoder, interface, config, demo, history, state, language = self.fixture("cuda")
        translator = GTranslator(encoder.native, decoder, config)
        checksum = frozen_base_checksum(encoder.native)
        original = {name: value.detach().clone() for name, value in translator.named_parameters()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            before_z = encoder(history[:, :, -1:])
            prediction = decoder(g_context_features(encoder.native, demo, history[:, :, :2], config)[1], state)
            target = {"z": before_z, "goal_poses": torch.eye(4, device="cuda").repeat(1, 1, 1, 1),
                      "goal_gripper": torch.ones(1, 1, device="cuda")}
            loss = g_goal_loss(prediction, target)["total"]
        optimizer = torch.optim.Adam(decoder.parameters(), lr=.01)
        loss.backward()
        optimizer.step()
        changed = {name for name, value in translator.named_parameters() if not torch.equal(value, original[name])}
        self.assertTrue(changed)
        self.assertTrue(all(name.startswith("goal_decoder.") for name in changed))
        self.assertEqual(frozen_base_checksum(encoder.native), checksum)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            torch.testing.assert_close(encoder(history[:, :, -1:]), before_z, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
    def test_native_pi_step_changes_only_interface_and_action_expert(self):
        from evo_wam.g_pi_context import frozen_base_checksum, pi_context_features
        from evo_wam.goal_action import action_named_parameters, goal_action_forward
        native, encoder, decoder, interface, config, demo, history, state, language = self.fixture("cuda")
        action_parameters = dict(action_named_parameters(native))
        original = {name: value.detach().clone() for name, value in native.named_parameters()}
        interface_before = {name: value.detach().clone() for name, value in interface.named_parameters()}
        checksum = frozen_base_checksum(encoder.native)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            before_z = encoder(history[:, :, -1:])
            goal = {"z": before_z, "goal_poses": torch.eye(4, device="cuda").repeat(1, 1, 1, 1),
                    "goal_gripper": torch.ones(1, 1, device="cuda")}
            features = pi_context_features(encoder.native, history[:, :, :2], config)
            conditions = interface.conditions(features, state,
                native.condition_embedder_action.text_embedder(language), goal,
                torch.tensor([0., .2]), torch.tensor([[-.5, 0.], [.5, 0.]]))
            predicted = goal_action_forward(native, torch.randn(1, 3, 1, 2, 1, device="cuda"),
                                            torch.full((1, 1), 500., device="cuda"), conditions)
            loss = predicted.float().square().mean()
        optimizer = torch.optim.Adam([*interface.parameters(), *action_parameters.values()], lr=.01)
        loss.backward()
        optimizer.step()
        changed = {name for name, value in native.named_parameters() if not torch.equal(value, original[name])}
        self.assertTrue(changed)
        self.assertTrue(changed <= action_parameters.keys())
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
