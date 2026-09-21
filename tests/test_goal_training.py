from dataclasses import replace
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from evo_wam.goal_action import action_named_parameters, goal_action_forward, goal_action_sample
from evo_wam.goal_data import load_goal_observation, load_goal_sample
from evo_wam.goal_interface import goal_pose_loss, validate_goal_poses
from evo_wam.icl_training import native_icl_objective
from evo_wam.goal_training import (autocast_for, build_goal_system, export_goal_policy, goal_registry,
    goal_training_loss, load_goal_config, load_goal_policy, masked_action_loss, predict_goal_actions,
    train_goal_interface, visual_goal_tokens, _restore_system, _system_state)
from evo_wam.zerowam import NativeDependencyError, load_native_class
from test_goal_data import write_goal_observation, write_goal_sample
from test_icl_data import save_sample


class GoalLossTest(unittest.TestCase):
    def test_masked_loss_has_no_inactive_value_or_gradient_contribution(self):
        prediction = torch.tensor([1., 2., float('nan')], requires_grad=True)
        target = torch.tensor([3., 3., float('nan')])
        mask = torch.tensor([True, True, False])
        loss = masked_action_loss(prediction, target, mask)
        torch.testing.assert_close(loss, torch.tensor(2.5))
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([-2., -1., 0.]))

    def test_fixed_prediction_losses_preserve_native_mask_and_weight_denominators(self):
        # Already-produced model outputs are held fixed. Altering only teacher
        # targets must affect the objective, without re-noising an input.
        shape = (1, 2, 2, 1, 1)
        mask = torch.ones(shape, dtype=torch.bool)
        mask[:, 1] = False
        stream = {'targets': torch.zeros(shape), 'valid_mask': mask,
                  'training_weight': torch.tensor([[2., 3.]])}
        inputs = {'latent_dict': stream, 'action_dict': stream, 'mcp_latent_dicts': [stream]}
        config = {'ifp': {'loss_weights': [.5]}, 'lambda_human': 3.}
        native = SimpleNamespace(patch_size=(1, 1, 1))
        video = torch.tensor([[[1., 2.], [3., 4.]]], requires_grad=True)
        action = torch.ones(1, 2, 2, requires_grad=True)
        future = torch.full((1, 2, 2), 2., requires_grad=True)
        losses = native_icl_objective(native, inputs, config, video, action, [future], human=False)
        torch.testing.assert_close(losses['video'], torch.tensor(29. / 4))
        torch.testing.assert_close(losses['action'], torch.tensor(5. / 4))
        torch.testing.assert_close(losses['ifp'], torch.tensor(5.))
        torch.testing.assert_close(losses['total'], torch.tensor(13.5))
        video_only = native_icl_objective(native, inputs, config, video, None, [future],
                                         human=False, video_only=True)
        torch.testing.assert_close(video_only['total'], losses['video'] + losses['ifp'])
        changed = {**inputs, 'latent_dict': {**stream, 'targets': torch.ones(shape)}}
        changed_losses = native_icl_objective(native, changed, config, video, action, [future], human=False)
        torch.testing.assert_close(changed_losses['video'], torch.tensor(3.))
        torch.testing.assert_close(changed_losses['action'], losses['action'])
        torch.testing.assert_close(changed_losses['ifp'], losses['ifp'])
        losses['total'].backward()
        self.assertTrue(video.grad.count_nonzero())
        self.assertTrue(action.grad.count_nonzero())
        self.assertTrue(future.grad.count_nonzero())


@unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
class GoalTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            load_native_class()
        except NativeDependencyError as exc:
            raise unittest.SkipTest(str(exc))

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        config = load_goal_config(Path(__file__).parents[1] / "configs/se3/goal_interface.json")
        config.update(sampling_steps=1, action_sampling_steps=1, pose_weight=.3, video_weight=.7,
                      goal_interface={"state_dim": 4, "dim": 16, "num_tokens": 4,
                                      "num_heads": 2, "num_layer_groups": 1, "num_pose_tokens": 2,
                                      "translation_scale": 1.})
        config["training"]["learning_rate"] = .001
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(config))
        self.config = load_goal_config(self.config_path)
        path, metadata, arrays = write_goal_sample(self.root, visual=True)
        metadata.update(current_time=.6, goal_time=1., control_dt=.1)
        metadata["action_space"].update(dimension=3, valid_channels=[True, True, False])
        rng = np.random.default_rng(19)
        arrays["actions"] = rng.normal(size=(3, 2, 2, 1)).astype(np.float32)
        arrays["actions"][2] = np.nan
        arrays["actions_mask"] = np.ones_like(arrays["actions"], dtype=np.bool_)
        arrays["goal_poses"][0, :3, 3] = [.1, -.2, .3]
        path.write_text(json.dumps(metadata))
        np.savez_compressed(self.root / metadata["arrays"], **arrays)
        pair_path = self.root / metadata["visual_pair"]
        pair = json.loads(pair_path.read_text())
        pair["history_frames"] = 4
        pair["target"]["action_space"] = metadata["action_space"]
        demo = {"latent": rng.normal(size=(4, 4, 1, 2)).astype(np.float32),
                "frame_times": np.arange(4, dtype=np.float64) * .2}
        target = {"latent": rng.normal(size=(4, 6, 1, 2)).astype(np.float32),
                  "frame_times": np.arange(6, dtype=np.float64) * .2,
                  "actions": np.concatenate((np.zeros((3, 4, 2, 1), dtype=np.float32), arrays["actions"]), axis=1),
                  "actions_mask": np.ones((3, 6, 2, 1), dtype=np.bool_)}
        save_sample(pair_path, pair, demo, target)
        self.sample_path = path
        self.sample = load_goal_sample(path, visual=True)
        self.registry = goal_registry(self.sample)
        self.index = self.root / "index.json"
        self.index.write_text(json.dumps({"format_version": 2, "kind": "se3_goal_index",
                                          "samples": [{"manifest": path.name, "split": "train"}]}))

    def args(self, output, steps=1, *, stage="goal", resume=None, initialize=None):
        return SimpleNamespace(config=str(self.config_path), index=str(self.index),
            output=str(self.root / output), steps=steps, seed=19, device="cuda", tiny_native=True,
            checkpoint=None, stage=stage, resume=resume, initialize=initialize)

    def generators(self):
        return {name: torch.Generator().manual_seed(19 + index)
                for index, name in enumerate(("action", "future", "video"))}

    def assert_same(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_same(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_same(first, second)
        else:
            self.assertEqual(left, right)

    def visual_conditions(self, native, interface, sample=None):
        sample = sample or self.sample
        pair = sample.visual
        with autocast_for(native):
            return visual_goal_tokens(native, interface, pair.demonstration,
                pair.target[:, :, :pair.history_frames], sample.state.cuda(), sample.language,
                self.config, self.generators()["future"], feature_layers=list(range(len(native.blocks))),
                current_time=sample.metadata["current_time"], control_dt=sample.metadata["control_dt"],
                actions_per_frame=sample.actions.shape[3], demo_times=pair.demonstration_times)

    def test_nonvisual_goal_stage_and_full_visual_updates_without_teacher_leakage(self):
        torch.manual_seed(19)
        native, interface, unused, _, layers = build_goal_system(self.config, self.registry,
            stage="goal", tiny_native=True, device="cuda")
        self.assertIsNone(unused)
        self.assertTrue(all(p.dtype == torch.float32 for m in (native, interface) for p in m.parameters()))
        self.assertFalse(any('.up.' in name or '.down.' in name for name in native.state_dict()))
        action_names = {name for name, _ in action_named_parameters(native)}
        self.assertTrue(action_names)
        self.assertTrue(all(p.requires_grad == (name in action_names) for name, p in native.named_parameters()))
        before = {name: value.detach().clone() for name, value in native.state_dict().items()}
        optimizer = torch.optim.AdamW([p for m in (native, interface) for p in m.parameters() if p.requires_grad], lr=.001)
        with patch.object(native, "forward", side_effect=AssertionError("Stage 1 cannot run video WAM")), \
             patch.object(native.patch_embedding_mlp, "forward", side_effect=AssertionError("Stage 1 cannot embed images")), \
             patch("evo_wam.goal_data.load_icl_sample", side_effect=AssertionError("Stage 1 cannot load images")):
            sample = load_goal_sample(self.sample_path, visual=False)
            losses = goal_training_loss(native, interface, unused, sample, self.config, self.generators(),
                                       stage="goal", feature_layers=layers)
            self.assertEqual(set(losses), {"action", "total"})
            losses["total"].backward()
            self.assertTrue(all(p.grad is None or p.grad.dtype == torch.float32
                                for m in (native, interface) for p in m.parameters()))
            self.assertTrue(any(p.grad is not None and p.grad.count_nonzero() for p in interface.goal_encoder.parameters()))
            optimizer.step()
        changed = {name for name, value in native.state_dict().items() if not torch.equal(value, before[name])}
        self.assertTrue(changed)
        self.assertTrue(changed <= action_names, changed - action_names)
        trained = copy.deepcopy(_system_state(native, interface, True))
        self.assertEqual(trained['native'].keys(), native.state_dict().keys())

        native, interface, unused, _, layers = build_goal_system(self.config, self.registry,
            stage="visual", tiny_native=True, device="cuda")
        _restore_system(native, interface, trained, True)
        self.assertTrue(all(p.requires_grad for _, p in action_named_parameters(native)))
        self.assertTrue(native.patch_embedding_mlp.weight.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in interface.goal_encoder.parameters()))
        before = {name: value.detach().clone() for name, value in native.state_dict().items()}
        optimizer = torch.optim.AdamW([p for m in (native, interface) for p in m.parameters() if p.requires_grad], lr=.001)
        with patch.object(interface, "encode_goal", side_effect=AssertionError("Stage 2 cannot read true poses")):
            losses = goal_training_loss(native, interface, unused, self.sample, self.config, self.generators(),
                                       stage="visual", feature_layers=layers)
        self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))
        losses['total'].backward()
        self.assertTrue(all(p.dtype == torch.float32 and (p.grad is None or p.grad.dtype == torch.float32)
                            for m in (native, interface) for p in m.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.count_nonzero() for _, p in action_named_parameters(native)))
        self.assertTrue(any(p.grad is not None and p.grad.count_nonzero() for n, p in native.named_parameters()
                            if n not in action_names))
        self.assertTrue(any(p.grad is not None and p.grad.count_nonzero() for p in interface.pose_decoder.parameters()))
        self.assertTrue(all(p.grad is None for m in (native, interface) for p in m.parameters() if not p.requires_grad))
        optimizer.step()
        changed = {name for name, value in native.state_dict().items() if not torch.equal(value, before[name])}
        self.assertTrue(changed & action_names)
        self.assertTrue(changed - action_names)

        # Same observed scene, language, demonstration and randomness; every teacher target changes.
        future = self.sample.visual.target.clone()
        future[:, :, self.sample.visual.history_frames:] += 5
        poses = self.sample.goal_poses.clone()
        poses[..., :3, 3] += 3
        changed_sample = replace(self.sample, goal_poses=poses, goal_gripper=torch.zeros_like(self.sample.goal_gripper),
            actions=self.sample.actions + 2, visual=replace(self.sample.visual, target=future))
        native.eval()
        interface.eval()
        with torch.no_grad(), autocast_for(native):
            original = self.visual_conditions(native, interface)
            changed = self.visual_conditions(native, interface, changed_sample)
            original_decoded = interface.decode_goal(original[1])
            changed_decoded = interface.decode_goal(changed[1])
            original_actions = goal_action_sample(native, original[0], self.sample.actions.shape,
                self.sample.actions_mask, torch.Generator().manual_seed(31), steps=2)
            changed_actions = goal_action_sample(native, changed[0], self.sample.actions.shape,
                self.sample.actions_mask, torch.Generator().manual_seed(31), steps=2)
        self.assert_same(original, changed)
        self.assert_same(original_decoded, changed_decoded)
        self.assert_same(original_actions, changed_actions)
        self.assertEqual(len(original[0]), len(native.blocks))
        self.assertTrue({"goal_poses", "goal_gripper", "actions", "target", "future"}.isdisjoint(
            inspect.signature(visual_goal_tokens).parameters))

    def test_exact_resume_fresh_visual_optimizer_and_standalone_observed_only_policy(self):
        with self.assertRaisesRegex(ValueError, "Stage 2 requires"):
            train_goal_interface(self.args("invalid", stage="visual"))
        full_goal = train_goal_interface(self.args("goal-full", 2))
        partial = train_goal_interface(self.args("goal-resume"))
        resumed_goal = train_goal_interface(self.args("goal-resume", resume=partial["artifact"]))
        full = torch.load(full_goal["artifact"], weights_only=True, map_location="cpu")
        resumed = torch.load(resumed_goal["artifact"], weights_only=True, map_location="cpu")
        self.assertEqual(full['format_version'], 2)
        self.assertTrue(all(value.dtype == torch.float32 for values in full['model'].values()
                            for value in values.values() if value.is_floating_point()))
        self.assertTrue(all(value.dtype == torch.float32 for values in full['optimizer']['state'].values()
                            for value in values.values() if isinstance(value, torch.Tensor) and value.is_floating_point()))
        for key in ("model", "optimizer", "rng", "torch_rng", "cuda_rng", "python_rng", "numpy_rng", "scheduler", "data_cursor"):
            self.assert_same(full[key], resumed[key])
        with self.assertRaisesRegex(ValueError, "both.*trained"):
            export_goal_policy(SimpleNamespace(artifact=full_goal["artifact"], output=str(self.root / "early-policy")))
        full_visual = train_goal_interface(self.args("visual-full", 2, stage="visual", initialize=full_goal["artifact"]))
        first_visual = train_goal_interface(self.args("visual-resume", stage="visual", initialize=full_goal["artifact"]))
        first = torch.load(first_visual["artifact"], weights_only=True, map_location="cpu")
        self.assertEqual(first["updates"], 1)
        self.assertEqual(first["goal_stage_updates"], 2)
        self.assertTrue(all(value["step"].item() == 1 for value in first["optimizer"]["state"].values()))
        resumed_visual = train_goal_interface(self.args("visual-resume", stage="visual", resume=first_visual["artifact"]))
        final = torch.load(full_visual["artifact"], weights_only=True, map_location="cpu")
        self.assertTrue(all(value.dtype == torch.float32 for values in final['optimizer']['state'].values()
                            for value in values.values() if isinstance(value, torch.Tensor) and value.is_floating_point()))
        resumed = torch.load(resumed_visual["artifact"], weights_only=True, map_location="cpu")
        for key in ("model", "optimizer", "rng", "torch_rng", "cuda_rng", "python_rng", "numpy_rng", "scheduler", "data_cursor"):
            self.assert_same(final[key], resumed[key])
        changed = {name for name, value in final["model"]["native"].items()
                   if not torch.equal(value, full["model"]["native"][name])}
        self.assertTrue(any('action_' in n for n in changed))
        self.assertTrue(any('action_' not in n for n in changed))
        policy_dir = self.root / "policy"
        exported = subprocess.run([sys.executable, "-m", "evo_wam", "export-goal-policy",
            "--artifact", full_visual["artifact"], "--output", str(policy_dir), "--max-shard-size", "50KB"],
            env={**os.environ, "PYTHONHASHSEED": "54321"}, capture_output=True, text=True, timeout=60)
        self.assertEqual(exported.returncode, 0, exported.stderr)
        native, interface, unused, payload = load_goal_policy(policy_dir, device="cuda")
        self.assertTrue(all(not p.requires_grad for m in (native, interface) for p in m.parameters()))
        self.assertNotIn("optimizer", payload)
        self.assertGreater(len(payload["shards"]), 1)
        self.assertNotIn('model', payload)
        self.assertEqual({name.removeprefix('native.') for name in payload['weight_map']
                          if name.startswith('native.')}, set(native.state_dict()))
        pair = self.sample.visual
        observation_path, metadata, _, _ = write_goal_observation(self.root)
        metadata.update({key: value for key, value in self.registry.items()
                         if key not in {'goal_source', 'language_identity'}},
                        current_time=self.sample.metadata["current_time"])
        metadata['language'] = self.sample.metadata['language']
        observation_path.write_text(json.dumps(metadata))
        np.savez_compressed(self.root / metadata["arrays"], state=self.sample.state[0].numpy(),
            history_latent=pair.target[0, :, :pair.history_frames].numpy(),
            history_times=pair.target_times[:pair.history_frames].numpy())
        np.savez_compressed(self.root / metadata["demonstration"]["arrays"],
            latent=pair.demonstration[0].numpy(), frame_times=pair.demonstration_times.numpy())
        observation = load_goal_observation(observation_path)
        frozen = copy.deepcopy(_system_state(native, interface, True))
        prediction = predict_goal_actions(native, interface, unused, payload, observation, seed=17)
        self.assertTrue(all(torch.isfinite(value).all() for value in prediction.values()))
        self.assertEqual(prediction["actions"].shape, (1, 3, 2, 2, 1))
        self.assertEqual(prediction["actions"][:, 2].count_nonzero().item(), 0)
        validate_goal_poses(prediction["goal_poses"])
        self.assertTrue(((prediction['goal_gripper'] >= 0) & (prediction['goal_gripper'] <= 1)).all())
        self.assert_same(frozen, _system_state(native, interface, True))
        prediction_path = self.root / "prediction.npz"
        predicted = subprocess.run([sys.executable, "-m", "evo_wam", "predict-goal-policy",
            "--policy", str(policy_dir), "--observation", str(observation_path), "--output", str(prediction_path),
            "--device", "cuda", "--seed", "17"], env={**os.environ, "PYTHONHASHSEED": "12345"},
            capture_output=True, text=True, timeout=90)
        self.assertEqual(predicted.returncode, 0, predicted.stderr)
        with np.load(prediction_path) as archive:
            self.assertEqual(set(archive.files), set(prediction))
            for name, expected in prediction.items():
                torch.testing.assert_close(torch.from_numpy(archive[name]), expected.float().cpu(), rtol=.025, atol=.025)
        with self.assertRaisesRegex(ValueError, "observed-only"):
            predict_goal_actions(native, interface, unused, payload, self.sample)
        with self.assertRaisesRegex(TypeError, "goal_poses"):
            predict_goal_actions(native, interface, unused, payload, observation, goal_poses=self.sample.goal_poses)
        manifest_path = policy_dir / 'policy.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['config']['interface_type'] = 'unknown'
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'interface_type'):
            load_goal_policy(policy_dir, device='cuda')
        manifest['config']['interface_type'] = self.config['interface_type']
        shard = next(iter(manifest['shards']))
        manifest['shards'][shard] = 'tampered'
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_goal_policy(policy_dir, device='cuda')
        arrays_path = self.root / self.sample.metadata["arrays"]
        with np.load(arrays_path) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays["state"][0] += .1
        np.savez_compressed(arrays_path, **arrays)
        with self.assertRaisesRegex(ValueError, "consumed.*changed"):
            train_goal_interface(self.args("visual-resume", stage="visual", resume=resumed_visual["artifact"]))

    def test_two_goals_and_two_demonstrations_fit_different_actions_from_identical_inputs(self):
        # Fix scene, language and noisy action input. The only distinguishing
        # information is the goal in Stage 1 or the demonstration in Stage 2.
        # Fitting both targets rules out a random-initialization inequality.
        samples = []
        for sign in (-1., 1.):
            poses = self.sample.goal_poses.clone()
            poses[..., 0, 3] = sign * .3
            samples.append(replace(self.sample, goal_poses=poses,
                goal_gripper=torch.full_like(self.sample.goal_gripper, (sign + 1) / 2),
                visual=replace(self.sample.visual,
                    demonstration=torch.full_like(self.sample.visual.demonstration, sign * 2))))
        noisy = torch.zeros_like(self.sample.actions, device='cuda')
        times = torch.full((1, noisy.shape[2]), 1000., device='cuda')
        mask = self.sample.actions_mask.cuda()
        targets = [torch.full_like(noisy, sign * .5) for sign in (-1., 1.)]
        torch.manual_seed(41)
        native, interface, _, _, _ = build_goal_system(self.config, self.registry,
            stage='goal', tiny_native=True, device='cuda')
        previous = None
        self.fit_metrics = {}
        for stage in ('goal', 'visual'):
            if previous is not None:
                native, interface, _, _, _ = build_goal_system(self.config, self.registry,
                    stage=stage, tiny_native=True, device='cuda')
                _restore_system(native, interface, previous, True)
            optimizer = torch.optim.AdamW([p for m in (native, interface) for p in m.parameters()
                                           if p.requires_grad], lr=.006, weight_decay=0.)

            def prediction(sample):
                with autocast_for(native):
                    if stage == 'goal':
                        language = native.condition_embedder_action.text_embedder(sample.language.cuda())
                        tokens = interface.encode_goal(sample.goal_poses.cuda(), sample.goal_gripper.cuda())
                        condition = interface.condition(tokens, sample.state.cuda(), language)
                        conditions = [condition] * len(native.blocks)
                    else:
                        conditions, tokens, _ = self.visual_conditions(native, interface, sample)
                    actions = goal_action_forward(native, noisy, times, conditions).float()
                    decoded = interface.decode_goal(tokens) if stage == 'visual' else None
                    return actions, decoded

            def pose_loss(decoded, sample):
                return goal_pose_loss(decoded['goal_poses'], sample.goal_poses.cuda(),
                    interface.translation_scale, gripper_prediction=decoded['goal_gripper'],
                    gripper_target=sample.goal_gripper.cuda())

            initial = None
            for step in range(100 if stage == 'goal' else 200):
                optimizer.zero_grad(set_to_none=True)
                errors = []
                for sample, target in zip(samples, targets):
                    actions, decoded = prediction(sample)
                    loss = masked_action_loss(actions, target, mask)
                    if decoded is not None:
                        loss = loss + pose_loss(decoded, sample)['total']
                    (loss / 2).backward()
                    errors.append(loss.item())
                if initial is None:
                    initial = sum(errors) / 2
                torch.nn.utils.clip_grad_norm_([p for m in (native, interface) for p in m.parameters()
                                              if p.requires_grad], 5.)
                optimizer.step()
                if max(errors) < .008:
                    break
            with torch.no_grad():
                outputs = [prediction(sample) for sample in samples]
            final_errors, final_goals = [], []
            for (output, decoded), target, sample in zip(outputs, targets, samples):
                error = masked_action_loss(output, target, mask).item()
                final_errors.append(error)
                self.assertLess(error, .025, (stage, step, initial, error))
                if decoded is not None:
                    pose = pose_loss(decoded, sample)
                    self.assertLess(pose['translation'].item(), .005)
                    self.assertLess(pose['rotation'].item(), .05)
                    self.assertLess(pose['gripper'].item(), .015)
                    torch.testing.assert_close(decoded['goal_poses'][..., :3, 3],
                        sample.goal_poses.cuda()[..., :3, 3], rtol=0, atol=.08)
                    torch.testing.assert_close(decoded['goal_gripper'], sample.goal_gripper.cuda(),
                                               rtol=0, atol=.12)
                    final_goals.append({'x': decoded['goal_poses'][0, 0, 0, 3].item(),
                                        'gripper': decoded['goal_gripper'][0, 0].item()})
            self.assertLess(outputs[0][0][mask].mean().item(), -.3)
            self.assertGreater(outputs[1][0][mask].mean().item(), .3)
            if stage == 'visual':
                self.assertLess(final_goals[0]['x'], -.2)
                self.assertGreater(final_goals[1]['x'], .2)
            self.fit_metrics[stage] = {'steps': step + 1, 'initial_mean_loss': initial,
                                       'final_action_mse': final_errors, 'final_goals': final_goals}
            previous = copy.deepcopy(_system_state(native, interface, True))

    def test_direct_feature_baseline_has_no_pose_decoder_supervision(self):
        config = {**self.config, 'interface_type': 'direct_features'}
        native, interface, unused, _, layers = build_goal_system(config, self.registry,
            stage='visual', tiny_native=True, device='cuda')
        with patch.object(interface, 'encode_goal', side_effect=AssertionError('no teacher conditioning')), \
             patch.object(interface, 'decode_goal', side_effect=AssertionError('baseline has no pose objective')):
            losses = goal_training_loss(native, interface, unused, self.sample, config, self.generators(),
                                       stage='visual', feature_layers=layers)
        self.assertFalse(any(name.startswith('pose') for name in losses))
        losses['total'].backward()
        self.assertTrue(native.patch_embedding_mlp.weight.grad.count_nonzero())
        self.assertTrue(any(p.grad is not None and p.grad.count_nonzero() for _, p in action_named_parameters(native)))


if __name__ == "__main__":
    unittest.main()
