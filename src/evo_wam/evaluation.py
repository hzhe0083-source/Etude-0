"""Prefix-matched rollout evaluation; no simulator or robot is emulated here.

An environment exposes observe(), step(action), success(), and, for Oracle@4,
snapshot()/restore(snapshot). Observations and actions must be JSON-compatible.
Policies receive (observation, random.Random) and return Candidate sequences.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    actions: tuple[Any, ...]

    def __post_init__(self):
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a nonempty string")
        if not isinstance(self.actions, Sequence) or isinstance(self.actions, (str, bytes)) or not self.actions:
            raise ValueError("candidate actions must be a nonempty sequence")
        object.__setattr__(self, "actions", tuple(deepcopy(self.actions)))


@dataclass(frozen=True)
class Selection:
    candidate_id: str | None
    status: str
    costs: tuple[float | None, ...]


@dataclass(frozen=True)
class Provenance:
    kind: str
    backend: str
    policy_id: str
    success_criterion: str
    checkpoint_id: str = "untrained"

    def __post_init__(self):
        if self.kind not in {"toy", "simulated", "real"}:
            raise ValueError("provenance kind must be toy, simulated, or real")
        if not all((self.backend, self.policy_id, self.success_criterion, self.checkpoint_id)):
            raise ValueError("backend, policy, criterion and checkpoint must be identified")


@dataclass(frozen=True)
class Replay:
    candidate_id: str
    proposed_actions: tuple[Any, ...]
    planned_prefix_length: int
    candidate_prefix_actions: tuple[Any, ...]
    continuation_actions: tuple[Any, ...]
    label_prefix_valid: tuple[bool, ...]
    observations: tuple[Any, ...]
    prefix_observation: Any
    prefix_progress: Any
    success: bool

    @property
    def actual_actions(self):
        """Commands actually submitted, including continuation (not ideal motion)."""
        return self.candidate_prefix_actions + self.continuation_actions


@dataclass(frozen=True)
class Episode:
    provenance: Provenance
    seed: int
    budget: int
    status: str
    success: bool
    steps: int
    decisions: tuple[Replay, ...]


@dataclass(frozen=True)
class LocalOracle:
    provenance: Provenance
    seed: int
    continuation_id: str
    continuation_budget: int
    budget: int
    replays: tuple[Replay, ...]
    local_oracle_at_4: bool
    successful_candidate_ids: tuple[str, ...]


def prefix_length(horizon: int) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    return max(1, horizon // 4)


def _validate_candidates(candidates: Sequence[Candidate], count: int):
    if count not in (1, 4) or len(candidates) != count:
        raise ValueError("evaluation requires exactly one or four candidates")
    if len({c.candidate_id for c in candidates}) != count:
        raise ValueError("candidate IDs must be unique within a decision")
    if len({len(c.actions) for c in candidates}) != 1:
        raise ValueError("a fixed candidate set must have one common horizon")


def rank_candidates(
    candidates: Sequence[Candidate], costs: Sequence[float], *, max_cost: float = math.inf
) -> Selection:
    """Select a fixed-set minimum; nonfinite predictions are rejected, never zeroed."""
    _validate_candidates(candidates, len(candidates))
    if len(costs) != len(candidates):
        raise ValueError("one effect-matching cost is required for every candidate")
    if math.isnan(max_cost) or max_cost < 0:
        raise ValueError("max_cost must be nonnegative")
    values = tuple(float(value) for value in costs)
    if any(math.isfinite(value) and value < 0 for value in values):
        raise ValueError("effect-matching costs must be nonnegative")
    eligible = [i for i, value in enumerate(values) if math.isfinite(value) and value <= max_cost]
    serializable = tuple(value if math.isfinite(value) else None for value in values)
    if not eligible:
        return Selection(None, "rejected", serializable)
    winner = min(eligible, key=lambda i: (values[i], i))
    return Selection(candidates[winner].candidate_id, "selected", serializable)


def select_baseline(
    candidates: Sequence[Candidate], *, seed: int, costs: Sequence[float] | None = None,
    max_cost: float = math.inf,
) -> Selection:
    """Random baseline, or score-shuffle baseline over the very same candidate set.

    Random selection does not get a learned feasibility filter. Supplying costs
    shuffles their correspondence to candidates before applying the same ranker
    and locked rejection threshold. max_cost is unused for the random baseline.
    """
    _validate_candidates(candidates, len(candidates))
    rng = random.Random(seed)
    if costs is None:
        return Selection(rng.choice(candidates).candidate_id, "selected", ())
    shuffled = list(costs)
    rng.shuffle(shuffled)
    return rank_candidates(candidates, shuffled, max_cost=max_cost)


def _budget(value: int):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("step budgets must be nonnegative integers")


def _execute(env, actions, observations):
    executed = []
    for action in actions:
        if env.success():
            break
        command = deepcopy(action)
        env.step(deepcopy(command))
        executed.append(command)
        observations.append(deepcopy(env.observe()))
    return executed


def replay_candidate(
    env,
    candidate: Candidate,
    *,
    seed: int,
    continuation: Callable | None = None,
    continuation_budget: int = 0,
    progress: Callable | None = None,
    budget: int | None = None,
) -> Replay:
    """Execute a candidate prefix and optionally a frozen single-candidate rule.

    Stops early only for observed success or an explicit total step budget. The
    validity vector applies solely to proposed_actions, never continuation. It
    marks execution eligibility, not sensor reliability: intersect it with the
    dataset's per-field label_valid before computing any supervised loss.
    """
    _budget(continuation_budget)
    if continuation_budget and continuation is None:
        raise ValueError("a continuation rule is required for a positive budget")
    planned = prefix_length(len(candidate.actions))
    if budget is None:
        budget = planned + continuation_budget
    _budget(budget)
    planned = min(planned, budget)
    observations = [deepcopy(env.observe())]
    original = _execute(env, candidate.actions[:planned], observations)
    prefix_obs = deepcopy(observations[-1])
    prefix_progress = progress(observations[0], prefix_obs) if progress is not None else None
    continued = []
    rng = random.Random(seed)
    remaining = min(continuation_budget, budget - len(original))
    while len(continued) < remaining and not env.success():
        proposals = tuple(continuation(deepcopy(env.observe()), rng))
        _validate_candidates(proposals, 1)
        chunk = proposals[0].actions[: min(prefix_length(len(proposals[0].actions)), remaining - len(continued))]
        continued.extend(_execute(env, chunk, observations))
    return Replay(
        candidate.candidate_id,
        deepcopy(candidate.actions),
        planned,
        tuple(original),
        tuple(continued),
        tuple(i < len(original) for i in range(len(candidate.actions))),
        tuple(observations),
        prefix_obs,
        prefix_progress,
        bool(env.success()),
    )


def local_oracle_at_4(
    env,
    candidates: Sequence[Candidate],
    continuation: Callable,
    *,
    continuation_id: str,
    continuation_budget: int,
    seed: int,
    provenance: Provenance,
    progress: Callable | None = None,
    budget: int | None = None,
) -> LocalOracle:
    """Branch from one complete snapshot, including simulator RNG and controller state.

    continuation must be the same frozen, stateless rule for every branch; its
    only stochastic input is the supplied RNG. The input environment is restored
    even if a branch raises. Approximate real-world resets are deliberately disallowed.
    """
    _validate_candidates(candidates, 4)
    _budget(continuation_budget)
    if budget is None:
        budget = prefix_length(len(candidates[0].actions)) + continuation_budget
    _budget(budget)
    if provenance.kind == "real":
        raise ValueError("exact snapshot Oracle@4 is not valid for real-world approximate resets")
    if not continuation_id:
        raise ValueError("identify the fixed continuation policy")
    snapshot = deepcopy(env.snapshot())
    replays = []
    try:
        for candidate in candidates:
            env.restore(deepcopy(snapshot))
            replays.append(replay_candidate(
                env, candidate, seed=seed, continuation=continuation,
                continuation_budget=continuation_budget, progress=progress, budget=budget,
            ))
    finally:
        env.restore(snapshot)
    winners = tuple(replay.candidate_id for replay in replays if replay.success)
    return LocalOracle(provenance, seed, continuation_id, continuation_budget, budget, tuple(replays), bool(winners), winners)


def closed_loop(
    env,
    policy: Callable,
    *,
    budget: int,
    seed: int,
    provenance: Provenance,
    candidate_count: int = 1,
    scorer: Callable | None = None,
    max_cost: float = math.inf,
    progress: Callable | None = None,
) -> Episode:
    """Replan after every executed prefix, with a hard total action-step budget."""
    _budget(budget)
    if candidate_count not in (1, 4):
        raise ValueError("candidate_count must be one or four")
    if candidate_count == 1 and scorer is not None:
        raise ValueError("the first-round single-candidate protocol disables F ranking")
    if candidate_count == 4 and scorer is None:
        raise ValueError("four-candidate execution requires an explicit scorer")
    rng = random.Random(seed)
    decisions = []
    steps = 0
    status = "budget_exhausted"
    while steps < budget and not env.success():
        observation = deepcopy(env.observe())
        candidates = tuple(policy(deepcopy(observation), rng))
        _validate_candidates(candidates, candidate_count)
        if scorer is None:
            chosen = candidates[0]
        else:
            selection = rank_candidates(candidates, scorer(observation, candidates), max_cost=max_cost)
            if selection.status == "rejected":
                status = "rejected"
                break
            chosen = next(c for c in candidates if c.candidate_id == selection.candidate_id)
        replay = replay_candidate(env, chosen, seed=seed, budget=budget - steps, progress=progress)
        decisions.append(replay)
        steps += len(replay.candidate_prefix_actions)
    success = bool(env.success())
    return Episode(provenance, seed, budget, "success" if success else status, success, steps, tuple(decisions))


def fingerprint(value: Any) -> str:
    """Hash explicit, JSON-compatible values or precomputed content digests."""
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass
class InferenceCache:
    """Two cache domains; synchronize before every policy query, including task swaps."""
    physical: dict = field(default_factory=dict)
    task: dict = field(default_factory=dict)
    _physical_key: str | None = None
    _task_key: str | None = None

    def synchronize(
        self, *, history, entity_ids, calibration, embodiment, model,
        demo, view, text, binding, progress, noise,
    ) -> tuple[bool, bool]:
        physical_key = fingerprint([history, entity_ids, calibration, embodiment, model])
        task_key = fingerprint([physical_key, demo, view, text, binding, progress, noise])
        physical_changed = physical_key != self._physical_key
        task_changed = task_key != self._task_key
        if physical_changed:
            self.physical.clear()
        if task_changed:
            self.task.clear()
        self._physical_key, self._task_key = physical_key, task_key
        return physical_changed, task_changed


def result_json(result: Episode | LocalOracle, path: str | Path | None = None) -> str:
    """Stable artifact; no wall-clock fields or unlabeled pooled robot/toy scores."""
    payload = asdict(result)
    payload["schema_version"] = 1
    payload["metric"] = "local_oracle_at_4" if isinstance(result, LocalOracle) else "closed_loop_task_success"
    output = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path is not None:
        Path(path).write_text(output, encoding="utf-8")
    return output
