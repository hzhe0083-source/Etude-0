"""Action math and fake-environment control checks; these are not simulator tests."""

import ast
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
try:
    from scipy.spatial.transform import Rotation
except ImportError:
    Rotation = None

from evo_wam.evaluation import Candidate, Provenance
from evo_wam.inference import RequirementRejected
from evo_wam.robotwin import (
    RoboTwinBridge, RobotwinActionTransform, USED_CHANNELS,
    run_oracle_requirement,
)


SOURCE = Path(__file__).resolve().parents[1] / "third_party" / "Zero-WAM"
STATS = SOURCE / "wan_va/assets/norm_stats/robotwin_icl.json"


def native_functions(relative_path, names):
    """Exercise actual pinned pure functions without importing simulator/client startup."""
    module = ast.parse((SOURCE / relative_path).read_text())
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    namespace = {"np": np, "R": Rotation, "ROBOTWIN_ACTION_DIM": 16}
    exec(compile(ast.Module(functions, type_ignores=[]), relative_path, "exec"), namespace)
    return namespace


def transform():
    return RobotwinActionTransform.from_stats(STATS, expected_sha256=sha256(STATS.read_bytes()).hexdigest())


class FakeRoboTwin:
    """Only the genuine public method names are emulated; there is no physics."""

    def __init__(self, step_lim=20):
        self.take_action_cnt, self.step_lim = 0, step_lim
        self.pose = np.array([0.3, 0.1, 0.2, 0, 0, 0, 1, 0.5, -0.3, 0.1, 0.2, 0, 0, 0, 1, 0.5])
        self.commands = []

    def get_obs(self):
        return {"endpose": {
            "left_endpose": self.pose[:7].tolist(), "left_gripper": self.pose[7],
            "right_endpose": self.pose[8:15].tolist(), "right_gripper": self.pose[15],
        }}

    def take_action(self, command, *, action_type):
        assert action_type == "ee"
        self.pose = command.copy()
        self.commands.append(command.copy())
        self.take_action_cnt += 1


def bridge(env, *, stop="original", original_at=100, modified_at=100, terminated=None):
    convert = transform()
    space = {"representation": "zero-wam-normalized", "normalization_id": convert.normalization_id,
             "dimension": 30, "valid_channels": np.isin(np.arange(30), USED_CHANNELS).tolist()}
    return RoboTwinBridge(
        env, convert, lambda raw, history: SimpleNamespace(raw=raw, history=history, action_space=deepcopy(space)),
        criteria={"original": lambda env: env.take_action_cnt >= original_at,
                  "zero_wam": lambda env: env.take_action_cnt >= modified_at},
        criterion_ids={"original": "fixture-original-v1", "zero_wam": "fixture-modified-v1"},
        stop_criterion=stop, control_dt=0.02, terminated=terminated,
    )


def provenance():
    return Provenance("toy", "fake-RoboTwin-API-no-physics", "fixture-policy", "fixture-original-v1")


@unittest.skipIf(Rotation is None, "native SciPy extra is unavailable")
class RoboTwinTests(unittest.TestCase):
    def test_native_postprocessing_and_relative_pose_roundtrip(self):
        convert = transform()
        native = native_functions("evaluation/robotwin/eval_policy_client_openpi.py", {"add_eef_pose", "add_init_pose"})
        relative_native = native_functions("wan_va/dataset/robotwin_action.py", {"_to_numpy", "relative_eef_pose", "relative_robotwin_action"})
        initial = FakeRoboTwin().pose
        initial[3:7] = Rotation.from_euler("xyz", [0.3, -0.4, 0.1]).as_quat()
        initial[11:15] = Rotation.from_euler("xyz", [-0.2, 0.2, 0.5]).as_quat()
        command = np.zeros(30)
        command[USED_CHANNELS] = np.linspace(-0.7, 0.8, 16)
        relative = ((command + 1) / 2 * (convert.q99 - convert.q01 + 1e-6) + convert.q01)[USED_CHANNELS]
        expected = native["add_init_pose"](relative, initial)
        expected[3:7] /= np.linalg.norm(expected[3:7])
        expected[11:15] /= np.linalg.norm(expected[11:15])
        actual = convert.to_absolute(command, initial)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        normalized = convert.from_absolute(actual, initial)
        native_relative = relative_native["relative_robotwin_action"](actual[None], initial[None])[0]
        np.testing.assert_allclose(normalized[USED_CHANNELS],
            (native_relative - convert.q01[USED_CHANNELS]) / (convert.q99[USED_CHANNELS] - convert.q01[USED_CHANNELS] + 1e-6) * 2 - 1)
        recovered = convert.to_absolute(normalized, initial)
        for start in (0, 8):
            np.testing.assert_allclose(recovered[start:start + 3], actual[start:start + 3], atol=1e-12)
            np.testing.assert_allclose(Rotation.from_quat(recovered[start + 3:start + 7]).as_matrix(),
                                       Rotation.from_quat(actual[start + 3:start + 7]).as_matrix(), atol=1e-12)
        self.assertTrue(np.all(normalized[14:28] == 0))
        with self.assertRaises(ValueError):
            RobotwinActionTransform.from_stats(STATS, expected_sha256="wrong")
        command[14] = 1
        with self.assertRaises(ValueError):
            convert.to_absolute(command, initial)

    def test_prefix_history_initial_pose_dual_criteria_and_early_stop(self):
        env = FakeRoboTwin()
        wrapped = bridge(env, original_at=3, modified_at=1)
        initial = wrapped.initial_pose.copy()
        command = wrapped.transform.from_absolute(initial, initial)
        command[0] += 0.1
        calls = []

        def policy(observation, requirement, rng):
            calls.append((len(observation.history), requirement))
            return Candidate("test", (command.tolist(),) * 8)  # exactly two per decision

        report = run_oracle_requirement(wrapped, lambda obs: "oracle-effect", policy,
            budget=10, seed=7, provenance=provenance(), requirement_source="fixture-labels")
        self.assertEqual(calls, [(0, "oracle-effect"), (2, "oracle-effect")])
        self.assertEqual(report["steps"], 3)
        self.assertEqual(report["status"], "success")
        self.assertEqual(report["success"], {"original": True, "zero_wam": True})
        self.assertEqual([d["executed_prefix_length"] for d in report["decisions"]], [2, 1])
        self.assertEqual(report["decisions"][1]["label_prefix_valid"], [True] + [False] * 7)
        self.assertEqual([row["time"] for row in report["executed"]], [0.02, 0.04, 0.06])
        observed_history = wrapped.observe().history
        observed_history[0]["normalized"][0] = 999
        self.assertNotEqual(wrapped.executed[0]["normalized"][0], 999)
        # Replanning must not rebase the relative command onto the new current pose.
        for physical in env.commands:
            np.testing.assert_allclose(physical, env.commands[0])
        for row in report["executed"]:
            np.testing.assert_allclose(wrapped.transform.to_absolute(row["normalized"], initial), row["absolute_ee"], atol=1e-12)
        with self.assertRaises(RuntimeError):
            wrapped.step(command)

    def test_termination_budget_and_bad_actions_do_not_invent_history(self):
        for mode, expected_steps in (("budget", 1), ("step_limit", 2), ("terminated", 1), ("initial_success", 0)):
            env = FakeRoboTwin(step_lim=2 if mode == "step_limit" else 10)
            wrapped = bridge(env, original_at=0 if mode == "initial_success" else 100,
                             terminated=(lambda env: env.take_action_cnt == 1) if mode == "terminated" else None)
            command = wrapped.transform.from_absolute(env.pose, env.pose)
            report = run_oracle_requirement(wrapped, lambda obs: None,
                lambda obs, req, rng: Candidate("long", (command.tolist(),) * 12),
                budget=1 if mode == "budget" else 10, seed=0, provenance=provenance(), requirement_source="fixture-labels")
            self.assertEqual(report["steps"], expected_steps)
            self.assertEqual(len(env.commands), expected_steps)
            self.assertEqual(report["status"], {"budget": "budget_exhausted", "step_limit": "environment_step_limit",
                "terminated": "terminated", "initial_success": "success"}[mode])
        wrapped = bridge(FakeRoboTwin())
        with self.assertRaises(ValueError):
            wrapped.step([float("nan")] * 30)
        self.assertEqual(wrapped.executed, [])
        self.assertEqual(wrapped.env.commands, [])
        zero_pose_command = np.zeros(30)
        zero_pose_command[USED_CHANNELS] = -wrapped.transform.q01[USED_CHANNELS] / (
            wrapped.transform.q99[USED_CHANNELS] - wrapped.transform.q01[USED_CHANNELS] + 1e-6
        ) * 2 - 1
        with self.assertRaises(ValueError):
            wrapped.step(zero_pose_command)
        self.assertEqual(wrapped.env.commands, [])
        with self.assertRaises(ValueError):
            bridge(FakeRoboTwin(), stop="missing")
        env = FakeRoboTwin()
        env.take_action_cnt = 1
        with self.assertRaises(ValueError):
            bridge(env)

    def test_observation_space_must_match_transform_before_policy_or_execution(self):
        changes = ({"normalization_id": "another-valid-policy-normalizer"}, {"dimension": 29},
                   {"valid_channels": [True] * 30}, {"representation": "absolute-ee"})
        for change in changes:
            wrapped = bridge(FakeRoboTwin())
            observation = wrapped.observe()
            observation.action_space.update(change)
            wrapped.observation_provider = lambda raw, history: observation
            calls = []
            with self.assertRaises(ValueError):
                run_oracle_requirement(wrapped, lambda obs: None, lambda *args: calls.append(args),
                    budget=5, seed=0, provenance=provenance(), requirement_source="fixture-labels")
            self.assertEqual(calls, [])
            self.assertEqual(wrapped.env.commands, [])
        wrapped = bridge(FakeRoboTwin())
        wrapped.observation_provider = lambda raw, history: {"action_space": {}}
        with self.assertRaises(ValueError):
            wrapped.observe()

    def test_requirement_rejection_stops_and_reports_actual_commands_only(self):
        for allowed_decisions in (0, 1):
            wrapped = bridge(FakeRoboTwin())
            command = wrapped.transform.from_absolute(wrapped.initial_pose, wrapped.initial_pose)
            calls = []

            def policy(observation, requirement, rng):
                calls.append(len(observation.history))
                if len(calls) > allowed_decisions:
                    raise RequirementRejected("unresolved_required_binding")
                return Candidate("test", (command.tolist(),) * 8)

            report = run_oracle_requirement(wrapped, lambda obs: "oracle", policy,
                budget=8, seed=0, provenance=provenance(), requirement_source="fixture-labels")
            self.assertEqual((report["status"], report["reason"]), ("rejected", "unresolved_required_binding"))
            self.assertEqual(report["commands_sent"], 2 * allowed_decisions)
            self.assertEqual(report["steps"], len(wrapped.env.commands))
            self.assertEqual(len(calls), allowed_decisions + 1)
        wrapped = bridge(FakeRoboTwin())

        def broken_policy(*args):
            raise ValueError("implementation error")

        with self.assertRaisesRegex(ValueError, "implementation error"):
            run_oracle_requirement(wrapped, lambda obs: "oracle", broken_policy,
                budget=8, seed=0, provenance=provenance(), requirement_source="fixture-labels")
        self.assertEqual(wrapped.env.commands, [])


if __name__ == "__main__":
    unittest.main()
