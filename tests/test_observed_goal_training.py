"""Native joint-training integration checks; synthetic fitting is not generalization."""

from contextlib import ExitStack
import copy
from dataclasses import replace
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
from evo_wam.goal_observed_interface import ObservedGoalInterface
from evo_wam.goal_training import (
    ARCHITECTURE, OBSERVED_ARCHITECTURE, _system_state, autocast_for,
    build_goal_system, export_goal_policy, goal_architecture, goal_registry,
    goal_training_loss, load_goal_config, load_goal_policy, masked_action_loss,
    observed_goal_conditions, predict_goal_actions, read_goal_artifact,
    train_goal_interface, validate_goal_config,
)
from evo_wam.zerowam import NativeDependencyError, load_native_class
from test_goal_data import write_goal_observation, write_goal_sample
from test_icl_data import save_sample


CONFIG_DIR = Path(__file__).parents[1] / "configs/se3"


class ObservedGoalConfigTest(unittest.TestCase):
    def test_minimal_joint_config_rejects_legacy_stages_and_video_objectives(self):
        config = load_goal_config(CONFIG_DIR / "observed_dual.json")
        self.assertEqual(goal_architecture(config), OBSERVED_ARCHITECTURE)
        self.assertEqual(OBSERVED_ARCHITECTURE, "observed_goal_dual_v1")
        self.assertEqual(set(config["goal_interface"]),
                         {"state_dim", "dim", "num_heads", "translation_scale"})
        self.assertNotIn("sampling_steps", config)
        self.assertEqual(config["video_weight"], 0)
        self.assertFalse(config["ifp"]["enabled"])
        self.assertFalse(any(config["ifp"]["loss_weights"]))

        invalid = []
        for value in (.1, -1., True, float("nan")):
            invalid.append({**copy.deepcopy(config), "video_weight": value})
        invalid.extend([
            {**copy.deepcopy(config), "sampling_steps": 1},
            {**copy.deepcopy(config), "ifp": {**config["ifp"], "enabled": True}},
            {**copy.deepcopy(config), "ifp": {**config["ifp"], "loss_weights": [.1, 0., 0., 0.]}},
            {**copy.deepcopy(config), "goal_interface": {**config["goal_interface"], "num_tokens": 4}},
            {**copy.deepcopy(config), "interface_type": "unknown"},
        ])
        for changed in invalid:
            with self.subTest(config=changed), self.assertRaises(ValueError):
                validate_goal_config(changed)

        # Reject an incompatible stage before constructing/loading a backbone.
        with patch("evo_wam.goal_training.build_icl_model",
                   side_effect=AssertionError("invalid stages cannot build a model")):
            for stage in ("goal", "visual"):
                with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "requires stage"):
                    build_goal_system(config, {}, stage=stage, tiny_native=True, device="cpu")
            legacy = load_goal_config(CONFIG_DIR / "goal_interface.json")
            self.assertEqual(goal_architecture(legacy), ARCHITECTURE)
            self.assertEqual(ARCHITECTURE, "recurrent_goal_full_v2")
            with self.assertRaisesRegex(ValueError, "requires stage"):
                build_goal_system(legacy, {}, stage="joint", tiny_native=True, device="cpu")


@unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
class ObservedGoalTrainingTest(unittest.TestCase):
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
        config = load_goal_config(CONFIG_DIR / "observed_dual.json")
        config.update(action_sampling_steps=1, pose_weight=.3,
                      goal_interface={"state_dim": 4, "dim": 16,
                                      "num_heads": 2, "translation_scale": 1.})
        config["training"].update(learning_rate=.001, backbone_learning_rate=.0005)
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(config))
        self.config = load_goal_config(self.config_path)

        # Own tiny observed-context fixture, reusing only the data writers.
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
        self.sample = load_goal_sample(path, visual=True)
        self.registry = goal_registry(self.sample)
        self.index = self.root / "index.json"
        self.index.write_text(json.dumps({"format_version": 2, "kind": "se3_goal_index",
                                          "samples": [{"manifest": path.name, "split": "train"}]}))

    def args(self, output, steps=1, *, stage="joint", resume=None, initialize=None):
        return SimpleNamespace(config=str(self.config_path), index=str(self.index),
            output=str(self.root / output), steps=steps, seed=19, device="cuda", tiny_native=True,
            checkpoint=None, stage=stage, resume=resume, initialize=initialize)

    def build(self, seed=19):
        torch.manual_seed(seed)
        return build_goal_system(self.config, self.registry, stage="joint", tiny_native=True, device="cuda")

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

    def forbid_video_generation(self, native):
        stack = ExitStack()
        for name in ("generated_robot_features", "native_icl_loss", "prepare_icl_inputs"):
            stack.enter_context(patch(f"evo_wam.goal_training.{name}",
                                      side_effect=AssertionError(f"joint training cannot call {name}")))
        stack.enter_context(patch.object(native.proj_out, "forward",
                            side_effect=AssertionError("joint training has no video output head")))
        stack.enter_context(patch.object(native, "forward",
                            side_effect=AssertionError("joint training cannot run video/IFP forward")))
        return stack

    def loss(self, native, interface, layers, sample=None):
        # No future/video generators exist: action diffusion is the only sampling path.
        return goal_training_loss(native, interface, None, sample or self.sample, self.config,
            {"action": torch.Generator().manual_seed(19)}, stage="joint", feature_layers=layers)

    def conditions(self, native, interface, layers, sample=None):
        sample = sample or self.sample
        pair = sample.visual
        with autocast_for(native):
            return observed_goal_conditions(native, interface, pair.demonstration,
                pair.target[:, :, :pair.history_frames], sample.state, sample.language, self.config,
                feature_layers=layers, demo_times=pair.demonstration_times)

    def assert_gradient(self, parameters):
        gradients = [p.grad for p in parameters if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    def test_joint_gradients_update_wan_and_both_action_conditions_without_video_forward(self):
        native, interface, unused, _, layers = self.build()
        self.assertIsNone(unused)
        self.assertIsInstance(interface, ObservedGoalInterface)
        self.assertTrue(all(p.requires_grad and p.dtype == torch.float32
                            for module in (native, interface) for p in module.parameters()))
        before = copy.deepcopy(_system_state(native, interface))
        parameters = [p for module in (native, interface) for p in module.parameters()]
        optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=0.)
        with self.forbid_video_generation(native):
            losses = self.loss(native, interface, layers)
            self.assertEqual(set(losses), {"action", "pose_translation", "pose_rotation", "pose_gripper",
                                          "pose_position_error_m", "pose_orientation_error_deg",
                                          "pose_gripper_error", "total"})
            self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))
            losses["action"].backward()
            self.assert_gradient(native.patch_embedding_mlp.parameters())
            self.assert_gradient(p for block in native.blocks for p in block.parameters())
            self.assert_gradient(p for _, p in action_named_parameters(native))
            self.assert_gradient(interface.wan_action_projection.parameters())
            self.assert_gradient(interface.pose_action_projection.parameters())
            self.assert_gradient(interface.pose_decoder.readout.parameters())
            # The numeric pose/gripper head is supervised only by the pose loss.
            self.assertTrue(all(p.grad is None for p in interface.pose_decoder.head.parameters()))
            optimizer.zero_grad(set_to_none=True)
            # Native compiled FlexAttention donates buffers, so use a fresh graph.
            losses = self.loss(native, interface, layers)
            losses["total"].backward()
            self.assert_gradient(interface.pose_decoder.head.parameters())
            self.assertTrue(all(p.grad is None or p.grad.dtype == torch.float32 for p in parameters))
            self.assertTrue(all(p.grad is None for p in native.proj_out.parameters()))
            optimizer.step()
        changed = {name for name, value in native.state_dict().items()
                   if not torch.equal(value.cpu(), before["native"][name])}
        action_names = {name for name, _ in action_named_parameters(native)}
        self.assertTrue(changed & action_names)
        self.assertTrue(changed - action_names)
        for prefix in ("wan_action_projection.", "pose_action_projection.", "pose_decoder.head."):
            self.assertTrue(any(name.startswith(prefix) and not torch.equal(value.cpu(), before["interface"][name])
                                for name, value in interface.state_dict().items()), prefix)

    def test_future_frames_never_affect_losses_or_predictions_and_pose_labels_only_affect_loss(self):
        native, interface, _, _, layers = self.build()
        native.eval()
        interface.eval()
        future = self.sample.visual.target.clone()
        future[:, :, self.sample.visual.history_frames:] += 17.
        changed_future = replace(self.sample, visual=replace(self.sample.visual, target=future))
        poses = self.sample.goal_poses.clone()
        poses[..., :3, 3] += 3.
        poses[..., :3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        changed_labels = replace(self.sample, goal_poses=poses,
                                 goal_gripper=torch.zeros_like(self.sample.goal_gripper))

        def outputs(sample):
            conditions, decoded = self.conditions(native, interface, layers, sample)
            with autocast_for(native):
                actions = goal_action_sample(native, conditions, sample.actions.shape, sample.actions_mask,
                                            torch.Generator().manual_seed(31), steps=2)
            return conditions, decoded, actions

        with torch.no_grad(), self.forbid_video_generation(native):
            baseline_loss = self.loss(native, interface, layers)
            future_loss = self.loss(native, interface, layers, changed_future)
            labels_loss = self.loss(native, interface, layers, changed_labels)
            baseline = outputs(self.sample)
            self.assert_same(baseline, outputs(changed_future))
            self.assert_same(baseline, outputs(changed_labels))
        self.assert_same(baseline_loss, future_loss)
        self.assert_same(baseline_loss["action"], labels_loss["action"])
        for name in ("pose_translation", "pose_rotation", "pose_gripper", "total"):
            self.assertNotEqual(baseline_loss[name].item(), labels_loss[name].item(), name)
        self.assertEqual(len(baseline[0]), len(native.blocks))
        self.assertTrue({"goal_poses", "goal_gripper", "actions", "target", "future", "generator"}.isdisjoint(
                        inspect.signature(observed_goal_conditions).parameters))

    def write_observation(self):
        path, metadata, _, _ = write_goal_observation(self.root)
        metadata.update({key: value for key, value in self.registry.items()
                         if key not in {"goal_source", "language_identity"}},
                        current_time=self.sample.metadata["current_time"],
                        language=self.sample.metadata["language"])
        path.write_text(json.dumps(metadata))
        pair = self.sample.visual
        np.savez_compressed(self.root / metadata["arrays"], state=self.sample.state[0].numpy(),
            history_latent=pair.target[0, :, :pair.history_frames].numpy(),
            history_times=pair.target_times[:pair.history_frames].numpy())
        np.savez_compressed(self.root / metadata["demonstration"]["arrays"],
            latent=pair.demonstration[0].numpy(), frame_times=pair.demonstration_times.numpy())
        return path

    def test_exact_joint_resume_and_standalone_cli_policy_without_stage_one_or_parameter_updates(self):
        full_report = train_goal_interface(self.args("full", 2))
        partial_report = train_goal_interface(self.args("resume"))
        with self.assertRaisesRegex(ValueError, "initialize is only"):
            train_goal_interface(self.args("invalid-initialize", initialize=partial_report["artifact"]))
        resumed_report = train_goal_interface(self.args("resume", resume=partial_report["artifact"]))
        full = read_goal_artifact(full_report["artifact"])
        resumed = read_goal_artifact(resumed_report["artifact"])
        for key in ("model", "optimizer", "rng", "torch_rng", "cuda_rng", "python_rng", "numpy_rng",
                    "scheduler", "data_cursor", "updates", "attempted_steps", "visited_arrays"):
            self.assert_same(full[key], resumed[key])
        self.assertEqual(full["architecture"], "observed_goal_dual_v1")
        self.assertEqual(full["stage"], "joint")
        self.assertEqual(full["updates"], 2)
        self.assertEqual(full["goal_stage_updates"], 0)
        self.assertIsNone(full["stage1_artifact_sha256"])
        for name, offset in (("future", 1), ("video", 2)):
            self.assert_same(full["rng"][name], torch.Generator().manual_seed(19 + offset).get_state())
        self.assertTrue(all(value.dtype == torch.float32 for values in full["model"].values()
                            for value in values.values() if value.is_floating_point()))
        self.assertTrue(all(value.dtype == torch.float32 for values in full["optimizer"]["state"].values()
                            for value in values.values() if isinstance(value, torch.Tensor) and value.is_floating_point()))
        self.assertFalse(full_report["video_generation"])
        self.assertFalse(full_report["video_supervision"])
        metrics = [json.loads(line) for line in (self.root / "full/metrics.jsonl").read_text().splitlines()]
        self.assertEqual(len(metrics), 2)
        self.assertTrue(all(row["goal_input"] == "observed_context" and not row["video_generation"]
                            and not row["video_supervision"] and "video" not in row and "ifp" not in row
                            for row in metrics))

        zero_path = self.root / "untrained.pt"
        torch.save({**full, "updates": 0}, zero_path)
        with self.assertRaisesRegex(ValueError, "joint training updates"):
            export_goal_policy(SimpleNamespace(artifact=str(zero_path), output=str(self.root / "untrained-policy")))
        wrong_path = self.root / "wrong-architecture.pt"
        torch.save({**full, "architecture": ARCHITECTURE}, wrong_path)
        with self.assertRaisesRegex(ValueError, "version-2"):
            read_goal_artifact(wrong_path)
        with self.assertRaisesRegex(ValueError, "version-2"):
            train_goal_interface(self.args("wrong-resume", resume=str(wrong_path)))

        policy_dir = self.root / "policy"
        exported = subprocess.run([sys.executable, "-m", "evo_wam", "export-goal-policy",
            "--artifact", full_report["artifact"], "--output", str(policy_dir), "--max-shard-size", "50KB"],
            env={**os.environ, "PYTHONHASHSEED": "54321"}, capture_output=True, text=True, timeout=60)
        self.assertEqual(exported.returncode, 0, exported.stderr)
        native, interface, unused, payload = load_goal_policy(policy_dir, device="cuda")
        self.assertEqual(payload["architecture"], OBSERVED_ARCHITECTURE)
        self.assertEqual(payload["goal_stage_updates"], 0)
        self.assertNotIn("optimizer", payload)
        self.assertNotIn("model", payload)
        self.assertGreater(len(payload["shards"]), 1)
        self.assertEqual({name.removeprefix("native.") for name in payload["weight_map"]
                          if name.startswith("native.")}, set(native.state_dict()))
        self.assertTrue(all(not p.requires_grad for module in (native, interface) for p in module.parameters()))
        observation_path = self.write_observation()
        observation = load_goal_observation(observation_path)
        frozen = copy.deepcopy(_system_state(native, interface))
        with self.forbid_video_generation(native):
            prediction = predict_goal_actions(native, interface, unused, payload, observation, seed=17)
            repeated = predict_goal_actions(native, interface, unused, payload, observation, seed=17)
        self.assert_same(prediction, repeated)
        self.assert_same(frozen, _system_state(native, interface))
        self.assertTrue(all(p.grad is None for module in (native, interface) for p in module.parameters()))
        self.assertEqual(set(prediction), {"actions", "goal_poses", "goal_gripper"})
        self.assertEqual(prediction["actions"].shape, (1, 3, 2, 2, 1))
        self.assertEqual(prediction["actions"][:, 2].count_nonzero().item(), 0)
        self.assertTrue(all(torch.isfinite(value).all() for value in prediction.values()))
        validate_goal_poses(prediction["goal_poses"])
        self.assertTrue(((prediction["goal_gripper"] >= 0) & (prediction["goal_gripper"] <= 1)).all())
        with self.assertRaisesRegex(ValueError, "observed-only"):
            predict_goal_actions(native, interface, unused, payload, self.sample)

        # The deployment bundle is sufficient after the training artifact disappears.
        Path(full_report["artifact"]).unlink()
        prediction_path = self.root / "prediction.npz"
        predicted = subprocess.run([sys.executable, "-m", "evo_wam", "predict-goal-policy",
            "--policy", str(policy_dir), "--observation", str(observation_path), "--output", str(prediction_path),
            "--device", "cuda", "--seed", "17"], env={**os.environ, "PYTHONHASHSEED": "12345"},
            capture_output=True, text=True, timeout=90)
        self.assertEqual(predicted.returncode, 0, predicted.stderr)
        with np.load(prediction_path) as archive:
            self.assertEqual(set(archive.files), {"actions", "goal_poses", "goal_gripper"})
            for name, expected in prediction.items():
                torch.testing.assert_close(torch.from_numpy(archive[name]), expected.float().cpu(), rtol=.025, atol=.025)

        manifest_path = policy_dir / "policy.json"
        manifest = json.loads(manifest_path.read_text())
        manifest_path.write_text(json.dumps({**manifest, "architecture": ARCHITECTURE}))
        with self.assertRaisesRegex(ValueError, "version-2"):
            load_goal_policy(policy_dir, device="cuda")
        manifest_path.write_text(json.dumps({**manifest, "updates": 0}))
        with self.assertRaisesRegex(ValueError, "joint training updates"):
            load_goal_policy(policy_dir, device="cuda")

    def test_two_demonstrations_fit_distinct_actions_and_poses_with_all_other_inputs_fixed(self):
        # A bounded synthetic capacity check, not evidence of task generalization.
        # Labels never enter the condition; only the demonstration distinguishes inputs.
        samples, targets = [], []
        noisy = torch.zeros_like(self.sample.actions, device="cuda")
        times = torch.full((1, noisy.shape[2]), 1000., device="cuda")
        mask = self.sample.actions_mask.cuda()
        for sign in (-1., 1.):
            poses = self.sample.goal_poses.clone()
            poses[..., :3, 3] = torch.tensor([sign * .3, 0., 0.])
            samples.append(replace(self.sample, goal_poses=poses,
                goal_gripper=torch.full_like(self.sample.goal_gripper, .5 + sign * .3),
                visual=replace(self.sample.visual,
                    demonstration=torch.full_like(self.sample.visual.demonstration, sign * 2))))
            targets.append(torch.full_like(noisy, sign * .5))
        self.assert_same(samples[0].state, samples[1].state)
        self.assert_same(samples[0].language, samples[1].language)
        self.assert_same(samples[0].visual.target, samples[1].visual.target)
        self.assert_same(samples[0].visual.demonstration_times, samples[1].visual.demonstration_times)
        native, interface, _, _, layers = self.build(seed=41)
        parameters = [p for module in (native, interface) for p in module.parameters()]
        optimizer = torch.optim.AdamW(parameters, lr=.006, weight_decay=0.)

        def prediction(sample):
            conditions, decoded = self.conditions(native, interface, layers, sample)
            with autocast_for(native):
                actions = goal_action_forward(native, noisy, times, conditions).float()
            return actions, decoded

        def pose_loss(decoded, sample):
            return goal_pose_loss(decoded["goal_poses"], sample.goal_poses.cuda(),
                interface.translation_scale, gripper_prediction=decoded["goal_gripper"],
                gripper_target=sample.goal_gripper.cuda())

        initial = None
        with self.forbid_video_generation(native):
            for step in range(200):
                optimizer.zero_grad(set_to_none=True)
                errors = []
                for sample, target in zip(samples, targets):
                    actions, decoded = prediction(sample)
                    loss = masked_action_loss(actions, target, mask) + pose_loss(decoded, sample)["total"]
                    (loss / 2).backward()
                    errors.append(loss.item())
                if initial is None:
                    initial = sum(errors) / 2
                torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                optimizer.step()
                if max(errors) < .002:
                    break
            with torch.no_grad():
                outputs = [prediction(sample) for sample in samples]
        final_errors, final_goals = [], []
        for (actions, decoded), target, sample in zip(outputs, targets, samples):
            action_error = masked_action_loss(actions, target, mask).item()
            pose = pose_loss(decoded, sample)
            final_errors.append(action_error + pose["total"].item())
            self.assertLess(action_error, .025, (step, initial, action_error))
            details = {"steps": step + 1, "initial": initial, "action": action_error,
                       **{key: value.item() for key, value in pose.items()}}
            self.assertLess(pose["translation"].item(), .005, details)
            self.assertLess(pose["rotation"].item(), .05, details)
            self.assertLess(pose["gripper"].item(), .015, details)
            torch.testing.assert_close(decoded["goal_poses"][..., :3, 3],
                                       sample.goal_poses.cuda()[..., :3, 3], rtol=0, atol=.08, msg=str(details))
            torch.testing.assert_close(decoded["goal_gripper"], sample.goal_gripper.cuda(),
                                       rtol=0, atol=.12, msg=str(details))
            final_goals.append(decoded["goal_poses"][0, 0, 0, 3].item())
        self.assertLess(sum(final_errors) / 2, initial * .1)
        self.assertLess(outputs[0][0][mask].mean().item(), -.3)
        self.assertGreater(outputs[1][0][mask].mean().item(), .3)
        self.assertLess(final_goals[0], -.2)
        self.assertGreater(final_goals[1], .2)
        self.fit_metrics = {"steps": step + 1, "initial_mean_loss": initial,
                            "final_total_losses": final_errors, "goal_x": final_goals}


if __name__ == "__main__":
    unittest.main()
