from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from evo_wam.g_pi_controller import GoalThresholds
from evo_wam.g_pi_evaluation import (EvaluationCase, _load_replay, delta_metrics, evaluate_g_pi_cli,
    evaluate_records, grouped_statistics, scene_modal_goals, valid_goal_region, write_evaluation_plots)


def goal(position=0., grip=0.):
    pose = torch.eye(4)[None, None]
    pose[0, 0, 0, 3] = position
    return {"z": torch.tensor([[[1., 0.], [0., 1.]]]), "goal_poses": pose,
            "goal_gripper": torch.tensor([[grip]])}


def case(name="a", intent="a", position=1., **kwargs):
    return EvaluationCase(id=name, scene="room", task="transfer", layout="left", intent=intent,
        source_id=f"source-{name}", subgoal=0, demonstration=torch.full((1, 1, 10, 1, 1), position),
        history=torch.zeros(1, 1, 2, 1, 1), state=torch.zeros(1, 2),
        history_times=torch.tensor([0., .4]), current=goal(), truth=goal(position), **kwargs)


def training(position=1., source="train", **kwargs):
    return {"scene": "room", "task": "transfer", "subgoal": 0, "source_id": source,
            "split": "train", "goal": goal(position), **kwargs}


class EvaluationMetricsTest(unittest.TestCase):
    def setUp(self):
        self.thresholds = GoalThresholds(.1, .1, 5., .05)

    def run_records(self, cases, **kwargs):
        return evaluate_records(cases, kwargs.pop("training_goals", [training()]),
            thresholds=self.thresholds,
            g_predict=kwargs.pop("g_predict", lambda demo, history, state: goal(float(demo.flatten()[-1]))),
            **kwargs)

    def test_signed_changes_and_relative_rotations(self):
        result = delta_metrics(goal(-1.), goal(1.), goal(2.))
        self.assertEqual(result["predicted_change"]["translation"][0][0], [-3., 0., 0.])
        self.assertEqual(result["reference_change"]["translation"][0][0], [-1., 0., 0.])
        self.assertEqual(result["error"]["position_m"], 2.)
        self.assertEqual(result["predicted_change"]["rotation"], torch.eye(3)[None, None].tolist())

    def test_region_excludes_overlapping_other_intent(self):
        region = valid_goal_region(goal(1.), [goal(1.)], [goal(1.05)], self.thresholds)
        self.assertTrue(region["near_own"])
        self.assertTrue(region["near_other"])
        self.assertFalse(region["valid"])
        with self.assertRaisesRegex(ValueError, "measured references"):
            valid_goal_region(goal(), [], [], self.thresholds)

    def test_scene_modal_baseline_only_training_independent_sources(self):
        rows = [training(1., "one")] * 8 + [training(2., "two"), training(2., "three")]
        mode = scene_modal_goals(rows, self.thresholds)[("room", 0)]
        self.assertEqual(mode["independent_support"], 2)
        self.assertEqual(float(mode["goal"]["goal_poses"][0, 0, 0, 3]), 2.)
        with self.assertRaisesRegex(ValueError, "training goals only"):
            self.run_records([case()], training_goals=[training(split="test")])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.run_records([case()], training_goals=[training(source="source-a")])
        # Task labels cannot reveal the intended goal to this scene-only baseline.
        other_task = replace(case(), task="different-purpose")
        self.assertEqual(self.run_records([other_task])["cases"][0]["scene_modal_source"], "train")

    def test_swap_requires_both_outputs_closer_to_their_own_truth(self):
        first, second = case(), case("b", "b", 2.)
        pair = [{"kind": "demo_swap", "ids": ["a", "b"]}]
        report = self.run_records([first, second], pairs=pair, g_predict=lambda *args: goal(1.))
        self.assertTrue(report["pairs"][0]["first_correct"])
        self.assertFalse(report["pairs"][0]["second_correct"])
        self.assertFalse(report["pairs"][0]["correct"])
        self.assertEqual(report["pair_summary"]["demo_swap"]["macro_mean"], 0.)
        with self.assertRaisesRegex(ValueError, "observation"):
            self.run_records([first, replace(second, state=torch.ones_like(second.state))], pairs=pair)

    def test_placebo_types_and_subgoal_reports_stay_separate(self):
        first = case()
        view = replace(case("view"), demonstration=torch.full_like(first.demonstration, 2.))
        speed = replace(case("speed"), demonstration=first.demonstration.clone())
        later = replace(case("later", position=3.), subgoal=1)
        report = self.run_records([first, view, speed, later],
            training_goals=[training(), training(3., subgoal=1)], pairs=[
                {"kind": "performer_viewpoint", "ids": ["a", "view"]},
                {"kind": "grasp_speed", "ids": ["a", "speed"]}])
        self.assertEqual(report["pair_summary"]["performer_viewpoint"]["macro_mean"], 1.)
        self.assertEqual(report["pair_summary"]["grasp_speed"]["macro_mean"], 0.)
        self.assertEqual(set(report["summary"]["g"]["per_subgoal"]), {"0", "1"})
        self.assertEqual(report["summary"]["g"]["valid_region"]["independent_scene_tasks"], 1)

    def test_grouped_statistics_dont_count_views_or_seeds_as_units(self):
        rows = [{"scene": "one", "task": "t", "score": 0.}] * 30
        rows += [{"scene": "two", "task": "t", "score": 1.}]
        result = grouped_statistics(rows, "score")
        self.assertEqual(result["independent_scene_tasks"], 2)
        self.assertEqual(result["records"], 31)
        self.assertEqual(result["macro_mean"], .5)

    def test_prefix_physically_cloned_and_recomputed_changes_mind(self):
        item = case()
        item.demonstration[:, :, :5] = 2.
        lengths, storage = [], []
        def predict(demo, history, state):
            lengths.append(demo.shape[2])
            storage.append(demo.untyped_storage().nbytes())
            return goal(float(demo.flatten()[-1]))
        report = self.run_records([item], g_predict=predict)
        self.assertEqual(lengths, [10] + list(range(1, 11)))
        self.assertEqual(storage[1:], [4 * length for length in range(1, 11)])
        self.assertEqual([row["macro_mean"] for row in report["prefix_curve"]], [0.] * 5 + [1.] * 5)
        self.assertEqual([row["latent_frames"] for row in report["prefix_cases"]], list(range(1, 11)))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "curves.svg"
            write_evaluation_plots(report, path)
            self.assertIn("<svg", path.read_text())
            self.assertIn("Demo prefix", path.read_text())
            json.dumps(report, allow_nan=False)

    def test_two_intents_two_layouts_controls_demo_and_observation(self):
        a, b = case(), case("b", "b", 2.)
        c, d = replace(a, id="c", layout="right", truth=goal(2.)), replace(b, id="d", layout="right", truth=goal(3.))
        c, d = replace(c, state=torch.ones(1, 2)), replace(d, state=torch.ones(1, 2))
        report = self.run_records([a, b, c, d], quadruples=[["a", "b", "c", "d"]],
            g_predict=lambda demo, history, state: goal(float(demo.flatten()[-1] + state[0, 0])))
        self.assertTrue(report["quadruples"][0]["all_correct"])
        with self.assertRaisesRegex(ValueError, "demonstration fixed"):
            self.run_records([a, b, replace(c, demonstration=c.demonstration + 1), d],
                             quadruples=[["a", "b", "c", "d"]])

    def test_replay_empty_language_default_and_instruction_goal_2x2(self):
        actions = torch.tensor([[[[[1.], [999.]]]]])
        mask = torch.tensor([[[[[True], [False]]]]])
        item = case(actions=actions, actions_mask=mask, language=torch.ones(1, 3, 2), wrong_goal=goal(2.))
        calls = []
        def pi(history, state, language, target, times):
            calls.append((language is None, float(target["goal_poses"][0, 0, 0, 3])))
            # Deliberately ignores g when language is supplied: diagnostic reports errors, not success.
            value = 1. if language is not None else float(target["goal_poses"][0, 0, 0, 3])
            return torch.full_like(actions, value)
        report = self.run_records([item], pi_predict=pi)
        self.assertEqual(calls, [(True, 1.), (True, 1.), (False, 1.), (False, 2.), (True, 1.), (True, 2.)])
        self.assertEqual([row["masked_mse"] for row in report["action_replays"]], [0., 0., 0., 0., 0., 1.])
        self.assertIsNone(report["policy_success"])
        self.assertEqual(report["replay_language_default"], "empty")
        self.assertEqual(len(report["action_replay_summary"]), 6)
        with self.assertRaisesRegex(ValueError, "correct language"):
            self.run_records([replace(item, language=None)], pi_predict=pi)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.run_records([replace(item, wrong_goal=item.truth)], pi_predict=pi)

    def test_bypass_reports_separate_goal_change_and_truth_errors(self):
        report = self.run_records([case()], diagnose_intent=lambda *args: {
            "baseline": goal(1.), "u_permuted": goal(2.), "robot_without_demo": goal(3.)})
        self.assertEqual([row["change"]["position_m"] for row in report["intent_bypass"]], [1., 2.])
        self.assertEqual([row["truth_error"]["position_m"] for row in report["intent_bypass"]], [1., 2.])
        self.assertEqual(report["intent_bypass_summary"]["u_permuted"]["change"]["z"]["macro_mean"], 0.)
        self.assertTrue(any("not causal necessity" in text for text in report["limitations"]))

    def test_independent_object_evidence_is_reference_only(self):
        evidence = {"source": "visual_object_relation", "provenance": "camera annotation v1",
                    "measurements": [{"object_id": "cup", "relation": "in bowl", "observed": True}]}
        report = self.run_records([case(object_evidence=evidence)])
        self.assertEqual(report["cases"][0]["object_evidence"], evidence)
        self.assertIsNone(report["policy_success"])
        with self.assertRaisesRegex(ValueError, "independent object"):
            self.run_records([case(object_evidence={**evidence, "source": "end_effector"})])

    def test_replay_rejects_actions_at_or_after_switch(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            actions = np.ones((1, 1, 2, 4, 1), dtype=np.float32)
            mask = np.ones_like(actions, dtype=np.bool_)
            np.savez(path.parent / "actions.npz", actions=actions, actions_mask=mask)
            observation = SimpleNamespace(metadata={"current_time": .4, "control_dt": .1})
            spec = {"arrays": "actions.npz", "subgoal_time": .6}
            with self.assertRaisesRegex(ValueError, "at/after"):
                _load_replay(path, spec, observation)
            mask[:, :, 0, 2:] = False
            mask[:, :, 1] = False
            np.savez(path.parent / "actions.npz", actions=actions, actions_mask=mask)
            _, loaded, _ = _load_replay(path, spec, observation)
            self.assertEqual(int(loaded.sum()), 2)

    def test_invalid_contracts_are_errors(self):
        with self.assertRaisesRegex(ValueError, "unique ids"):
            self.run_records([case(), case()])
        with self.assertRaisesRegex(ValueError, "missing training"):
            self.run_records([case()], training_goals=[])
        with self.assertRaisesRegex(ValueError, "nonempty boolean"):
            self.run_records([case(actions=torch.zeros(1, 1, 1, 1, 1),
                                   actions_mask=torch.zeros(1, 1, 1, 1, 1, dtype=torch.bool))])
        with self.assertRaisesRegex(ValueError, "placebo pairs"):
            self.run_records([case(), case("b", "b", 2.)],
                             pairs=[{"kind": "grasp_speed", "ids": ["a", "b"]}])


@unittest.skipUnless(torch.cuda.is_available(), "Native G/π evaluation requires CUDA")
class EvaluationNativeTest(unittest.TestCase):
    def test_real_exported_policies_end_to_end_command(self):
        import test_g_pi_training as training_tests
        from evo_wam.g_pi_data import load_g_pi_sample
        from evo_wam.g_pi_deployment import save_goal_prediction
        from evo_wam.g_pi_training import export_g_pi_policy, load_g_pi_policy, train_g_pi_interface

        fixture = training_tests.GPiNativeTrainingTest()
        fixture.setUpClass()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        root = fixture.root
        policies = {}
        for route in ("g_translator", "pi_goal"):
            args = fixture.args(route, f"{route}-training")
            trained = train_g_pi_interface(args)
            policies[route] = export_g_pi_policy(SimpleNamespace(artifact=trained["artifact"],
                output=root / f"{route}-policy", dtype="float32", stop_thresholds=training_tests.explicit_thresholds()))["policy"]
        _, _, encoder, payload = load_g_pi_policy(policies["g_translator"], device="cuda")
        sample = load_g_pi_sample(fixture.path, route="pi_goal", current_time=.4)
        truth = {"z": encoder(sample.target_frame), "goal_poses": sample.goal_poses, "goal_gripper": sample.goal_gripper}
        wrong = {key: value.clone() for key, value in truth.items()}
        wrong["goal_poses"][..., 0, 3] += 1.
        registry, identity = payload["registry"], encoder.identity
        truth_path = save_goal_prediction(root / "truth.npz", truth, identity, registry)
        wrong_path = save_goal_prediction(root / "wrong.npz", wrong, identity, registry)
        task = json.loads(fixture.path.read_text())
        fields = {"feature_space_id", "latent_normalization", "action_space", "control_dt", "actions_per_frame",
            "state_space_id", "coordinate_frame", "pose_units", "end_effectors", "pose_representation", "tool_frames",
            "gripper_space", "frame_stride", "temporal_down_rate", "alignment", "subgoal_encoding", "demonstration"}
        observation = {key: task[key] for key in fields}
        observation.update(format_version=3, kind="g_pi_observation", arrays="observed.npz", current_time=.4)
        (root / "observed.json").write_text(json.dumps(observation))
        np.savez(root / "observed.npz", state=sample.state[0].numpy(), history_latent=sample.history[0].numpy(),
                 latent_available_times=sample.history_times.numpy())
        np.savez(root / "replay.npz", actions=sample.actions.numpy(), actions_mask=sample.actions_mask.numpy())
        manifest = {"format_version": 1, "kind": "g_pi_evaluation", "split": "test",
            "g_policy": Path(policies["g_translator"]).name, "pi_policy": Path(policies["pi_goal"]).name,
            "encoder_identity": identity, "registry": registry, "thresholds": training_tests.explicit_thresholds(),
            "training_goals": [{"scene": "fixture", "task": "task", "source_id": "train-source", "subgoal": 0,
                                "split": "train", "goal": truth_path.name}],
            "cases": [{"id": "one", "scene": "fixture", "task": "task", "layout": "layout", "intent": "purpose",
                "source_id": "heldout-source", "subgoal": 0, "observation": "observed.json", "truth": truth_path.name,
                "current": truth_path.name, "language": task["language"], "wrong_goal": wrong_path.name,
                "action_replay": {"arrays": "replay.npz", "subgoal_time": sample.subgoal_time}}],
            "pairs": [], "quadruples": []}
        (root / "evaluation.json").write_text(json.dumps(manifest))
        output = root / "metrics.json"
        report = evaluate_g_pi_cli(SimpleNamespace(manifest=root / "evaluation.json", output=output,
            device="cuda", g_checkpoint=None, pi_checkpoint=None))
        result = json.loads(output.read_text())
        self.assertEqual(report["cases"], 1)
        self.assertTrue(result["shared_video_base"])
        self.assertTrue(output.with_suffix(".svg").is_file())
        self.assertEqual(len(result["prefix_cases"]), 10)
        self.assertEqual(len(result["action_replays"]), 6)
        self.assertEqual(len(result["intent_bypass"]), 2)
        self.assertTrue(any(name.endswith("policy.json") for name in result["input_sha256"]))
        self.assertTrue(any(name.endswith(".safetensors") for name in result["input_sha256"]))
        self.assertIsNone(result["policy_success"])
        # Fixed seeds pair the empty oracle and empty correct-goal diagnostic exactly.
        self.assertEqual(result["action_replays"][0]["masked_mse"], result["action_replays"][4]["masked_mse"])
        self.assertEqual({(row["language"], row["goal"]) for row in result["action_replays"]
                          if row["path"] == "instruction_goal_2x2"},
                         {("correct", "correct"), ("correct", "wrong"), ("empty", "correct"), ("empty", "wrong")})
        for mode in ("regression_only", "independent"):
            args = fixture.args("g_translator", f"g-{mode}")
            config_path = Path(args.config)
            config = json.loads(config_path.read_text())
            config["goal_interface"]["intent_mode"] = mode
            config_path.write_text(json.dumps(config))
            trained = train_g_pi_interface(args)
            exported = export_g_pi_policy(SimpleNamespace(artifact=trained["artifact"], output=root / f"policy-{mode}",
                dtype="float32", stop_thresholds=training_tests.explicit_thresholds()))
            manifest["g_policy"] = Path(exported["policy"]).name
            (root / "evaluation.json").write_text(json.dumps(manifest))
            mode_output = root / f"metrics-{mode}.json"
            evaluate_g_pi_cli(SimpleNamespace(manifest=root / "evaluation.json", output=mode_output,
                device="cuda", g_checkpoint=None, pi_checkpoint=None))
            mode_report = json.loads(mode_output.read_text())
            self.assertEqual(mode_report["intent_bypass_status"]["status"], "not_applicable")
            self.assertEqual(mode_report["intent_bypass_status"]["intent_mode"], mode)
            self.assertEqual(len(mode_report["prefix_cases"]), 10)
            self.assertEqual(mode_report["intent_bypass"], [])


if __name__ == "__main__":
    unittest.main()
