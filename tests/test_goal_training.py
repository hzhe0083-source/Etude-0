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

from evo_wam.goal_data import load_goal_observation, load_goal_sample
from evo_wam.goal_interface import validate_goal_poses
from evo_wam.goal_training import (build_goal_system, export_goal_policy, goal_registry,
    goal_training_loss, load_goal_config, load_goal_policy, predict_goal_actions,
    train_goal_interface, visual_goal_tokens, _restore_system, _system_state)
from evo_wam.zerowam import NativeDependencyError, load_native_class
from test_goal_data import write_goal_observation, write_goal_sample
from test_icl_data import save_sample


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
        config.update(sampling_steps=1, lambda_human=3.,
                      action_sampling_steps=1, pose_weight=.3, video_weight=.7,
                      goal_interface={"state_dim": 4, "dim": 16, "num_tokens": 4,
                                      "num_heads": 4, "translation_scale": 1.})
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
        self.index.write_text(json.dumps({"format_version": 1, "kind": "se3_goal_index",
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

    def test_goal_stage_is_nonvisual_and_visual_stage_has_only_generated_goal_input(self):
        torch.manual_seed(19)
        native, interface, null, _, layers = build_goal_system(self.config, self.registry,
            stage="goal", tiny_native=True, device="cuda")
        before = {name: value.clone() for name, value in _system_state(native, interface, True)["native"].items()}
        optimizer = torch.optim.AdamW([p for m in (native, interface) for p in m.parameters() if p.requires_grad], lr=.001)
        with patch.object(native, "forward", side_effect=AssertionError("Stage 1 cannot run video WAM")), \
             patch.object(native.patch_embedding_mlp, "forward", side_effect=AssertionError("Stage 1 cannot embed images")), \
             patch("evo_wam.goal_data.load_icl_sample", side_effect=AssertionError("Stage 1 cannot load images")):
            sample = load_goal_sample(self.sample_path, visual=False)
            losses = goal_training_loss(native, interface, null, sample, self.config, self.generators(),
                                       stage="goal", feature_layers=layers)
            self.assertEqual(set(losses), {"action", "total"})
            losses["total"].backward()
            for prefix in ("goal_encoder.", "state_encoder.", "condition_adapter."):
                self.assertTrue(any(p.grad is not None and p.grad.count_nonzero() for n, p in interface.named_parameters()
                                    if n.startswith(prefix)), prefix)
            optimizer.step()
        changed = {name for name, value in native.state_dict().items() if not torch.equal(value.cpu(), before[name])}
        self.assertTrue(changed)
        self.assertTrue(all("action_" in name and (".up." in name or ".down." in name) for name in changed), changed)
        trained = _system_state(native, interface, True)
        compact = _system_state(native, interface, False)
        self.assertTrue(any("action_" in name for name in compact["native"]))
        self.assertTrue(any("action_" not in name for name in compact["native"]))
        self.assertTrue(all(".up." in name or ".down." in name for name in compact["native"]))
        self.assertEqual(compact["interface"].keys(), interface.state_dict().keys())

        native, interface, null, _, layers = build_goal_system(self.config, self.registry,
            stage="visual", tiny_native=True, device="cuda")
        incomplete = {**compact, "native": dict(compact["native"])}
        del incomplete["native"][next(n for n in incomplete["native"] if "action_" in n)]
        with self.assertRaisesRegex(ValueError, "full action/video adapters"):
            _restore_system(native, interface, incomplete, False)
        _restore_system(native, interface, trained, True)
        self.assertTrue(all(not p.requires_grad for n, p in native.named_parameters() if "action_" in n))
        self.assertTrue(all(not p.requires_grad for n, p in interface.named_parameters()
                            if n.startswith(("goal_encoder.", "state_encoder.", "condition_adapter."))))
        captured = []
        original_read = interface.read_future

        def read(*args, **kwargs):
            tokens = original_read(*args, **kwargs)
            captured.append(tokens.detach().clone())
            return tokens

        with patch.object(interface, "encode_goal", side_effect=AssertionError("Stage 2 cannot read true poses")), \
             patch.object(interface, "read_future", side_effect=read):
            losses = goal_training_loss(native, interface, null, self.sample, self.config, self.generators(),
                                       stage="visual", feature_layers=layers)
        self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))
        self.assertGreater(losses["ifp"].item(), 0)
        torch.testing.assert_close(losses["total"], losses["action"]
            + self.config["pose_weight"] * (losses["pose_translation"] + losses["pose_rotation"])
            + self.config["video_weight"] * (losses["video"] + losses["ifp"]))
        losses["action"].backward()
        self.assertTrue(any(p.grad is not None and p.grad.count_nonzero() for n, p in native.named_parameters()
                            if "action_" not in n and (".up." in n or ".down." in n)))
        self.assertTrue(any(p.grad is not None and p.grad.count_nonzero() for n, p in interface.named_parameters()
                            if n.startswith("visual_")))
        native.zero_grad(set_to_none=True)
        interface.zero_grad(set_to_none=True)
        # Native compiled attention donates buffers; use a fresh graph for the
        # full objective rather than retaining its action-only backward graph.
        losses = goal_training_loss(native, interface, null, self.sample, self.config, self.generators(),
                                   stage="visual", feature_layers=layers)
        losses["total"].backward()
        self.assertTrue(any(p.grad is not None and p.grad.count_nonzero() for n, p in interface.named_parameters()
                            if n.startswith("pose_decoder.")))
        self.assertTrue(all(p.grad is None for m in (native, interface) for p in m.parameters() if not p.requires_grad))

        # Change all teacher targets while leaving A, observed B and randomness fixed.
        future = self.sample.visual.target.clone()
        future[:, :, self.sample.visual.history_frames:] += 5
        poses = self.sample.goal_poses.clone()
        poses[..., :3, 3] += 3
        changed_sample = replace(self.sample, goal_poses=poses, actions=self.sample.actions + 2,
                                visual=replace(self.sample.visual, target=future))
        with torch.no_grad():
            pair = changed_sample.visual
            tokens, _ = visual_goal_tokens(native, interface, pair.demonstration,
                pair.target[:, :, :pair.history_frames], changed_sample.state.cuda(), null,
                self.config, self.generators()["future"], feature_layers=layers,
                current_time=changed_sample.metadata["current_time"], control_dt=changed_sample.metadata["control_dt"],
                actions_per_frame=changed_sample.actions.shape[3], demo_times=pair.demonstration_times)
        torch.testing.assert_close(tokens, captured[0], rtol=0, atol=0)
        self.assertTrue({"goal_poses", "actions", "target", "future"}.isdisjoint(inspect.signature(visual_goal_tokens).parameters))

    def test_exact_stage_resume_fresh_visual_optimizer_and_observed_only_policy(self):
        with self.assertRaisesRegex(ValueError, "Stage 2 requires"):
            train_goal_interface(self.args("invalid", stage="visual"))
        full_goal = train_goal_interface(self.args("goal-full", 2))
        partial = train_goal_interface(self.args("goal-resume"))
        resumed_goal = train_goal_interface(self.args("goal-resume", resume=partial["artifact"]))
        full = torch.load(full_goal["artifact"], weights_only=True, map_location="cpu")
        resumed = torch.load(resumed_goal["artifact"], weights_only=True, map_location="cpu")
        for key in ("model", "optimizer", "rng", "torch_rng", "cuda_rng"):
            self.assert_same(full[key], resumed[key])
        with self.assertRaisesRegex(ValueError, "both trained"):
            export_goal_policy(SimpleNamespace(artifact=full_goal["artifact"], output=str(self.root / "early-policy")))
        full_visual = train_goal_interface(self.args("visual-full", 2, stage="visual", initialize=full_goal["artifact"]))
        first_visual = train_goal_interface(self.args("visual-resume", stage="visual", initialize=full_goal["artifact"]))
        first = torch.load(first_visual["artifact"], weights_only=True, map_location="cpu")
        self.assertEqual(first["updates"], 1)
        self.assertEqual(first["goal_stage_updates"], 2)
        self.assertTrue(all(value["step"].item() == 1 for value in first["optimizer"]["state"].values()))
        resumed_visual = train_goal_interface(self.args("visual-resume", stage="visual", resume=first_visual["artifact"]))
        final = torch.load(full_visual["artifact"], weights_only=True, map_location="cpu")
        resumed = torch.load(resumed_visual["artifact"], weights_only=True, map_location="cpu")
        for key in ("model", "optimizer", "rng", "torch_rng", "cuda_rng"):
            self.assert_same(final[key], resumed[key])
        changed = {name for name, value in final["model"]["native"].items()
                   if not torch.equal(value, full["model"]["native"][name])}
        self.assertTrue(changed)
        self.assertTrue(all("action_" not in n and (".up." in n or ".down." in n) for n in changed), changed)
        changed_interface = {name for name, value in final["model"]["interface"].items()
                             if not torch.equal(value, full["model"]["interface"][name])}
        self.assertTrue(any(name.startswith("visual_") for name in changed_interface))
        self.assertTrue(any(name.startswith("pose_decoder.") for name in changed_interface))
        for name, value in full["model"]["interface"].items():
            if name.startswith(("goal_encoder.", "state_encoder.", "condition_adapter.")):
                self.assert_same(value, final["model"]["interface"][name])
        policy_dir = self.root / "policy"
        exported = subprocess.run([sys.executable, "-m", "evo_wam", "export-goal-policy",
            "--artifact", full_visual["artifact"], "--output", str(policy_dir)],
            env={**os.environ, "PYTHONHASHSEED": "54321"}, capture_output=True, text=True, timeout=60)
        self.assertEqual(exported.returncode, 0, exported.stderr)
        native, interface, null, payload = load_goal_policy(policy_dir, device="cuda")
        self.assertTrue(all(not p.requires_grad for m in (native, interface) for p in m.parameters()))
        self.assertNotIn("optimizer", payload)
        pair = self.sample.visual
        observation_path, metadata, _, _ = write_goal_observation(self.root)
        metadata.update({key: value for key, value in self.registry.items() if key != "goal_source"},
                        current_time=self.sample.metadata["current_time"])
        observation_path.write_text(json.dumps(metadata))
        np.savez_compressed(self.root / metadata["arrays"], state=self.sample.state[0].numpy(),
            history_latent=pair.target[0, :, :pair.history_frames].numpy(),
            history_times=pair.target_times[:pair.history_frames].numpy())
        np.savez_compressed(self.root / metadata["demonstration"]["arrays"],
            latent=pair.demonstration[0].numpy(), frame_times=pair.demonstration_times.numpy())
        observation = load_goal_observation(observation_path)
        frozen = {part: {key: value.clone() for key, value in values.items()}
                  for part, values in _system_state(native, interface, True).items()}
        prediction = predict_goal_actions(native, interface, null, payload, observation, seed=17)
        self.assertTrue(all(torch.isfinite(value).all() for value in prediction.values()))
        self.assertEqual(prediction["actions"].shape, (1, 3, 2, 2, 1))
        self.assertEqual(prediction["actions"][:, 2].count_nonzero().item(), 0)
        validate_goal_poses(prediction["goal_poses"])
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
            predict_goal_actions(native, interface, null, payload, self.sample)
        with self.assertRaisesRegex(TypeError, "goal_poses"):
            predict_goal_actions(native, interface, null, payload, observation, goal_poses=self.sample.goal_poses)
        manifest = json.loads((policy_dir / "policy.json").read_text())
        manifest["policy_sha256"] = "tampered"
        (policy_dir / "policy.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_goal_policy(policy_dir, device="cuda")
        arrays_path = self.root / self.sample.metadata["arrays"]
        with np.load(arrays_path) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays["state"][0] += .1
        np.savez_compressed(arrays_path, **arrays)
        with self.assertRaisesRegex(ValueError, "consumed.*changed"):
            train_goal_interface(self.args("visual-resume", stage="visual", resume=resumed_visual["artifact"]))


if __name__ == "__main__":
    unittest.main()
