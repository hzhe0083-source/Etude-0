"""RoboTwin EE action bridge and single-candidate oracle-requirement diagnostic.

No simulator, asset, policy weight, entity tracker, or oracle label is fabricated.
An episode factory supplies those server-side dependencies explicitly.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib
import json
from pathlib import Path
import random
from typing import Callable, Mapping

import numpy as np

from .evaluation import Candidate, Provenance, prefix_length


USED_CHANNELS = np.array([*range(7), 28, *range(7, 14), 29])


def _finite(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return value


def _pose(value, name):
    value = _finite(value, (16,), name).copy()
    for start in (3, 11):
        norm = np.linalg.norm(value[start:start + 4])
        if norm < 1e-8:
            raise ValueError(f"{name} contains a zero quaternion")
        value[start:start + 4] /= norm
    return value


def endpose_from_observation(raw):
    """Read the same left-pose/gripper/right-pose/gripper fields as Zero-WAM."""
    pose = raw["endpose"]
    return _pose(np.concatenate([
        pose["left_endpose"], [pose["left_gripper"]],
        pose["right_endpose"], [pose["right_gripper"]],
    ]), "initial endpose")


@dataclass(frozen=True)
class RobotwinActionTransform:
    q01: np.ndarray
    q99: np.ndarray
    normalization_id: str

    def __post_init__(self):
        for name in ("q01", "q99"):
            value = _finite(getattr(self, name), (30,), name).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if np.any(self.q99 < self.q01) or not isinstance(self.normalization_id, str) or not self.normalization_id:
            raise ValueError("normalization ranges and their artifact identity are required")

    @classmethod
    def from_stats(cls, path, *, expected_sha256: str):
        """Load pinned Robotwin quantiles; the stats digest must match the policy."""
        content = Path(path).read_bytes()
        digest = sha256(content).hexdigest()
        if digest != expected_sha256:
            raise ValueError("normalization artifact does not match the policy digest")
        stats = json.loads(content)
        if stats.get("method") != "abs":
            raise ValueError("Zero-WAM Robotwin statistics require method=abs")
        hand = stats["norm_stats"]["action.hand.position"]
        effector = stats["norm_stats"]["action.effector.position"]
        # Exact layout used by pinned va_robotwin_cfg.load_robotwin_norm_stat.
        quantiles = [np.array(hand[key] + [0.0] * 14 + effector[key]) for key in ("q01", "q99")]
        return cls(*quantiles, normalization_id=f"sha256:{digest}")

    def to_absolute(self, normalized, initial_pose):
        """30 normalized channels -> 16 absolute EE targets, SciPy xyzw convention.

        Translation is added in world coordinates; orientation is initial * relative.
        This is pinned postprocess_action followed by add_init_pose, not joint control.
        """
        from scipy.spatial.transform import Rotation

        normalized = _finite(normalized, (30,), "normalized command")
        inactive = np.ones(30, dtype=bool)
        inactive[USED_CHANNELS] = False
        if np.any(normalized[inactive] != 0):
            raise ValueError("inactive Zero-WAM action channels must be zero")
        initial = _pose(initial_pose, "initial pose")
        relative = ((normalized + 1) / 2 * (self.q99 - self.q01 + 1e-6) + self.q01)[USED_CHANNELS]
        relative = _pose(relative, "relative command")
        absolute = relative.copy()
        for start in (0, 8):
            absolute[start:start + 3] += initial[start:start + 3]
            absolute[start + 3:start + 7] = (
                Rotation.from_quat(initial[start + 3:start + 7])
                * Rotation.from_quat(relative[start + 3:start + 7])
            ).as_quat()
        return absolute

    def from_absolute(self, absolute, initial_pose):
        """Canonical normalized history of the command actually sent to the simulator.

        Quaternion sign follows the native training helper. Do not clip observed
        commands: clipping would make recorded history describe a different command.
        """
        from scipy.spatial.transform import Rotation

        initial = _pose(initial_pose, "initial pose")
        relative = _pose(absolute, "absolute command")
        for start in (0, 8):
            relative[start:start + 3] -= initial[start:start + 3]
            quaternion = (
                Rotation.from_quat(initial[start + 3:start + 7]).inv()
                * Rotation.from_quat(relative[start + 3:start + 7])
            ).as_quat()
            relative[start + 3:start + 7] = quaternion * (1.0 if quaternion[-1] > 0 else -1.0)
        normalized = np.zeros(30)
        normalized[USED_CHANNELS] = (
            (relative - self.q01[USED_CHANNELS])
            / (self.q99[USED_CHANNELS] - self.q01[USED_CHANNELS] + 1e-6) * 2 - 1
        )
        return normalized


class RoboTwinBridge:
    """One already-set-up episode. Observed history contains submitted commands only.

    Both success callbacks must inspect the same environment without mutating it.
    They are explicit because the patched task's check_success cannot recover the
    original predicate. The observation provider consumes raw current get_obs()
    and the immutable executed-command records; visual tracking stays external.
    """

    def __init__(
        self, env, transform: RobotwinActionTransform, observation_provider: Callable,
        *, criteria: Mapping[str, Callable], criterion_ids: Mapping[str, str],
        stop_criterion: str, control_dt: float, terminated: Callable | None = None,
    ):
        if set(criteria) != {"original", "zero_wam"} or set(criterion_ids) != set(criteria):
            raise ValueError("both original and zero_wam success predicates and source IDs are required")
        if stop_criterion not in criteria or not all(isinstance(value, str) and value for value in criterion_ids.values()):
            raise ValueError("identify both success predicates and the fixed stopping predicate")
        if not all(callable(test) for test in criteria.values()) or not callable(observation_provider):
            raise ValueError("success predicates and observation provider must be callable")
        if terminated is not None and not callable(terminated):
            raise ValueError("terminated must be callable")
        if isinstance(control_dt, bool) or not np.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("control_dt must be the real action control period, not video fps")
        if env.take_action_cnt != 0 or isinstance(env.step_lim, bool) or not isinstance(env.step_lim, (int, np.integer)) or env.step_lim <= 0:
            raise ValueError("create the bridge at the start of a fresh episode with a positive step limit")
        self.env, self.transform = env, transform
        self.observation_provider = observation_provider
        self.criteria, self.criterion_ids = dict(criteria), dict(criterion_ids)
        self.stop_criterion, self.control_dt = stop_criterion, float(control_dt)
        self.terminated = terminated
        self.initial_pose = endpose_from_observation(env.get_obs())
        self.executed = []
        self.success_ever = {name: False for name in criteria}
        self._refresh_success()

    def _refresh_success(self):
        self.success_now = {name: bool(test(self.env)) for name, test in self.criteria.items()}
        for name, success in self.success_now.items():
            self.success_ever[name] |= success

    def status(self):
        if self.success_ever[self.stop_criterion]:
            return "success"
        if self.terminated is not None and self.terminated(self.env):
            return "terminated"
        if self.env.take_action_cnt >= self.env.step_lim:
            return "environment_step_limit"
        return "running"

    def observe(self):
        from .data import _action_space

        observation = self.observation_provider(self.env.get_obs(), deepcopy(tuple(self.executed)))
        space = _action_space(getattr(observation, "action_space", None), 30)
        expected_channels = np.isin(np.arange(30), USED_CHANNELS).tolist()
        if space["normalization_id"] != self.transform.normalization_id or space["valid_channels"] != expected_channels:
            raise ValueError("observation action space differs from the RoboTwin command transform")
        return observation

    def step(self, normalized):
        if self.status() != "running":
            raise RuntimeError("cannot step a terminated episode")
        command = self.transform.to_absolute(normalized, self.initial_pose)
        # Only append after take_action returns; never record the unexecuted tail.
        previous_count = self.env.take_action_cnt
        self.env.take_action(command.copy(), action_type="ee")
        if self.env.take_action_cnt != previous_count + 1:
            raise RuntimeError("RoboTwin did not account for exactly one submitted command; stop rather than invent history")
        step = len(self.executed) + 1
        self.executed.append({
            "step": step, "time": step * self.control_dt,
            "normalized": self.transform.from_absolute(command, self.initial_pose).tolist(),
            "absolute_ee": command.tolist(),
        })
        self._refresh_success()


def run_oracle_requirement(
    bridge: RoboTwinBridge, requirement_provider: Callable, policy: Callable,
    *, budget: int, seed: int, provenance: Provenance, requirement_source: str,
):
    """Single candidate, H//4 prefix, re-observe and rebuild current oracle requirements.

    policy(observation, requirement, random.Random) returns one Candidate. This
    diagnostic intentionally bypasses human task reading and cannot measure ICL.
    """
    from .inference import RequirementRejected

    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if provenance.kind not in {"toy", "simulated"} or not requirement_source:
        raise ValueError("identify simulator/test provenance and the oracle requirement source")
    if provenance.success_criterion != bridge.criterion_ids[bridge.stop_criterion]:
        raise ValueError("provenance must identify the actual stopping success predicate")
    if bridge.executed:
        raise ValueError("run one diagnostic per fresh bridge")
    rng, decisions, rejection = random.Random(seed), [], None
    while len(bridge.executed) < budget and bridge.status() == "running":
        observation = bridge.observe()
        try:
            requirement = requirement_provider(observation)
            candidate = policy(observation, requirement, rng)
        except RequirementRejected as error:
            rejection = str(error)
            break
        if not isinstance(candidate, Candidate):
            raise TypeError("the single-candidate policy must return Candidate")
        planned = min(prefix_length(len(candidate.actions)), budget - len(bridge.executed))
        before = len(bridge.executed)
        for action in candidate.actions[:planned]:
            if bridge.status() != "running":
                break
            bridge.step(action)
        actual = len(bridge.executed) - before
        decisions.append({
            "candidate_id": candidate.candidate_id, "horizon": len(candidate.actions),
            "planned_prefix_length": planned, "executed_prefix_length": actual,
            "label_prefix_valid": [i < actual for i in range(len(candidate.actions))],
        })
    status = "rejected" if rejection is not None else bridge.status()
    return {
        "schema_version": 1, "metric": "oracle_requirement_closed_loop_task_success",
        "provenance": asdict(provenance), "requirement_source": requirement_source,
        "normalization_id": bridge.transform.normalization_id,
        "criterion_ids": bridge.criterion_ids, "stop_criterion": bridge.stop_criterion,
        "success": bridge.success_ever, "final_success": bridge.success_now,
        "status": "budget_exhausted" if status == "running" else status,
        "reason": rejection, "commands_sent": len(bridge.executed),
        "seed": seed, "budget": budget, "steps": len(bridge.executed),
        "control_dt": bridge.control_dt, "decisions": decisions, "executed": bridge.executed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True, help="server-side module:function returning runner keyword arguments")
    parser.add_argument("--config", type=Path, required=True, help="factory JSON; weight and environment paths stay server-side")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module, name = args.factory.split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    kwargs = factory(json.loads(args.config.read_text(encoding="utf-8")))
    report = run_oracle_requirement(**kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
