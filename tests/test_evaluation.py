"""Protocol checks only: this small object state machine is not RoboTwin."""

from copy import deepcopy
import json
import math
import unittest

from etude.evaluation import (
    Candidate, InferenceCache, Provenance, closed_loop, local_oracle_at_4,
    prefix_length, rank_candidates, replay_candidate, result_json, select_baseline,
)


class ToyObject:
    """An explicitly toy one-dimensional held object, for control-flow regression."""

    def __init__(self, target=1, require_held=False):
        self.position = 0
        self.held = True
        self.target = target
        self.require_held = require_held
        self.ticks = 0

    def observe(self):
        return {"position": self.position, "held": self.held, "ticks": self.ticks}

    def step(self, action):
        if action["op"] == "move" and self.held:
            self.position += action["dx"]
        elif action["op"] == "release":
            self.held = False
        elif action["op"] == "grasp":
            self.held = True
        elif action["op"] != "wait":
            if action["op"] not in {"move", "release", "grasp"}:
                raise ValueError("unknown action")
        self.ticks += 1

    def success(self):
        return self.position == self.target and self.held == self.require_held

    def snapshot(self):
        return deepcopy(self.__dict__)

    def restore(self, snapshot):
        self.__dict__ = deepcopy(snapshot)


def action(op, dx=0):
    return {"op": op, "dx": dx}


def candidate(name, first, tail=None):
    return Candidate(name, (first,) + (tail or action("wait"),) * 3)


PROVENANCE = Provenance("toy", "deterministic-object-fixture", "fixture-policy-v1", "exact-position-and-grasp")


class EvaluationTests(unittest.TestCase):
    def test_prefix_rank_validation_and_rejection(self):
        candidates = tuple(candidate(str(i), action("wait")) for i in range(4))
        self.assertEqual([prefix_length(h) for h in (1, 3, 4, 7, 8)], [1, 1, 1, 1, 2])
        self.assertEqual(rank_candidates(candidates, [3, 1, 1, 4]).candidate_id, "1")
        self.assertEqual(rank_candidates(candidates, [math.nan, math.inf, 10, 11], max_cost=3).status, "rejected")
        self.assertEqual(rank_candidates(candidates, [-math.inf, math.nan, math.inf, 2]).candidate_id, "3")
        self.assertEqual(rank_candidates(candidates, [-math.inf, math.nan, math.inf, -math.inf]).status, "rejected")
        self.assertEqual(select_baseline(candidates, seed=7), select_baseline(candidates, seed=7))
        self.assertEqual(select_baseline(candidates, seed=7, costs=[3, 1, 4, 2]), select_baseline(candidates, seed=7, costs=[3, 1, 4, 2]))
        self.assertEqual(select_baseline(candidates, seed=7, costs=[3, 1, 4, 2], max_cost=0.5).status, "rejected")
        shuffled = select_baseline(candidates, seed=7, costs=[3, 1, 4, 2], max_cost=1)
        self.assertEqual(shuffled.costs[int(shuffled.candidate_id)], 1)
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                prefix_length(bad)
        with self.assertRaises(ValueError):
            rank_candidates(candidates, [1])
        with self.assertRaises(ValueError):
            rank_candidates(candidates, [1, 2, -1, 3])
        with self.assertRaises(ValueError):
            rank_candidates((candidates[0],) * 4, [1] * 4)
        with self.assertRaises(ValueError):
            rank_candidates((candidates[0], candidates[1], candidates[2], Candidate("other", (action("wait"),))), [1] * 4)

    def test_prefix_labels_never_describe_continuation_or_candidate_tail(self):
        env = ToyObject()
        proposed = candidate("right", action("move", 1), action("move", -8))
        replay = replay_candidate(
            env, proposed, seed=1, continuation_budget=2,
            continuation=lambda obs, rng: [candidate("release", action("release"))],
            progress=lambda before, after: after["position"] - before["position"],
        )
        self.assertEqual(replay.label_prefix_valid, (True, False, False, False))
        self.assertEqual(replay.prefix_progress, 1)
        self.assertTrue(replay.prefix_observation["held"])
        self.assertTrue(replay.success)
        self.assertEqual(replay.actual_actions, (action("move", 1), action("release")))
        self.assertEqual(replay.proposed_actions[1], action("move", -8))
        self.assertEqual(len(replay.observations), len(replay.actual_actions) + 1)

    def test_local_oracle_uses_same_snapshot_seed_budget_and_restores(self):
        env = ToyObject()
        before = env.snapshot()
        candidates = tuple(candidate(str(i), action("move", i - 1)) for i in range(4))
        draws = []

        def release(obs, rng):
            draws.append(rng.random())
            return [candidate("release", action("release"))]

        report = local_oracle_at_4(
            env, candidates, release, continuation_id="fixed-release", continuation_budget=1,
            seed=42, provenance=PROVENANCE,
        )
        self.assertEqual(env.snapshot(), before)
        self.assertEqual(report.successful_candidate_ids, ("2",))
        self.assertTrue(report.local_oracle_at_4)
        self.assertEqual(len(set(draws)), 1)
        self.assertTrue(all(r.observations[0]["ticks"] == 0 for r in report.replays))
        self.assertTrue(all(len(r.continuation_actions) <= 1 for r in report.replays))
        self.assertTrue(all(r.label_prefix_valid == (True, False, False, False) for r in report.replays))
        data = json.loads(result_json(report))
        self.assertEqual(data["provenance"]["kind"], "toy")
        self.assertEqual(data["metric"], "local_oracle_at_4")
        self.assertEqual(result_json(report), result_json(report))

    def test_oracle_restores_on_failure_and_rejects_real_approximate_resets(self):
        env = ToyObject()
        before = env.snapshot()
        candidates = tuple(candidate(str(i), action("move", 2)) for i in range(4))

        def broken(obs, rng):
            raise RuntimeError("test failure")

        with self.assertRaises(RuntimeError):
            local_oracle_at_4(env, candidates, broken, continuation_id="broken", continuation_budget=1, seed=0, provenance=PROVENANCE)
        self.assertEqual(before, env.snapshot())
        with self.assertRaises(ValueError):
            local_oracle_at_4(env, candidates, broken, continuation_id="broken", continuation_budget=1, seed=0,
                              provenance=Provenance("real", "hardware", "policy", "criterion"))

    def test_oracle_matches_remaining_deployment_budget_in_prefix_and_continuation(self):
        def continuation(obs, rng):
            return [candidate("continue", action("move", 1))]

        for horizon in (12, 4):
            env = ToyObject(target=3, require_held=True)
            candidates = tuple(Candidate(str(i), (action("move", 1),) * horizon) for i in range(4))
            oracle = local_oracle_at_4(
                env, candidates, continuation, continuation_id="advance", continuation_budget=5,
                budget=2, seed=0, provenance=PROVENANCE,
            )
            self.assertFalse(oracle.local_oracle_at_4)
            self.assertEqual(oracle.budget, 2)
            self.assertTrue(all(len(r.actual_actions) == 2 for r in oracle.replays))
            actual_prefix = min(prefix_length(horizon), 2)
            self.assertTrue(all(r.planned_prefix_length == actual_prefix for r in oracle.replays))
            self.assertEqual(oracle.replays[0].label_prefix_valid, (True,) * actual_prefix + (False,) * (horizon - actual_prefix))
            episode = closed_loop(env, lambda obs, rng: [candidates[0]], budget=2, seed=0, provenance=PROVENANCE)
            self.assertEqual((episode.success, episode.steps, episode.budget), (False, 2, 2))
            self.assertEqual(json.loads(result_json(oracle))["budget"], 2)

        env = ToyObject(target=3, require_held=True)
        oracle = local_oracle_at_4(env, candidates, continuation, continuation_id="advance", continuation_budget=5,
                                  budget=0, seed=0, provenance=PROVENANCE)
        self.assertTrue(all(not r.actual_actions and not any(r.label_prefix_valid) for r in oracle.replays))

    def test_four_conditions_use_same_prefix_then_task_specific_branch(self):
        prefix_commands = []
        for view in ("front", "side"):
            for target in (-1, 1):
                env = ToyObject(target=target)

                def policy(obs, rng):
                    # The fixture understands a supplied goal; no learned capability is claimed.
                    if obs["ticks"] == 0:
                        command = action("wait")
                    elif obs["position"] != target:
                        command = action("move", target - obs["position"])
                    else:
                        command = action("release")
                    return [candidate(f"{view}-{target}", command)]

                result = closed_loop(env, policy, budget=4, seed=5, provenance=PROVENANCE)
                self.assertTrue(result.success)
                self.assertEqual(result.steps, 3)
                self.assertEqual(env.position, target)
                prefix_commands.append(result.decisions[0].candidate_prefix_actions)
                self.assertEqual(json.loads(result_json(result))["metric"], "closed_loop_task_success")
        self.assertEqual(prefix_commands, [(action("wait"),)] * 4)

    def test_same_geometry_different_control_requirement(self):
        for require_held in (True, False):
            env = ToyObject(target=1, require_held=require_held)

            def policy(obs, rng):
                command = action("move", 1) if obs["position"] == 0 else action("release")
                return [candidate("control", command)]

            result = closed_loop(env, policy, budget=3, seed=0, provenance=PROVENANCE)
            self.assertTrue(result.success)
            self.assertEqual(env.position, 1)
            self.assertEqual(env.held, require_held)
            self.assertEqual(result.steps, 1 if require_held else 2)

    def test_budget_rejection_and_initial_success_do_not_send_actions(self):
        env = ToyObject(target=10)
        proposals = [Candidate("long", (action("move", 1),) * 12)]
        episode = closed_loop(env, lambda obs, rng: proposals, budget=2, seed=0, provenance=PROVENANCE)
        self.assertEqual((episode.status, episode.steps), ("budget_exhausted", 2))
        self.assertEqual(episode.decisions[0].label_prefix_valid, (True,) * 2 + (False,) * 10)
        candidates = tuple(candidate(str(i), action("wait")) for i in range(4))
        episode = closed_loop(env, lambda obs, rng: candidates, budget=4, seed=0, provenance=PROVENANCE,
                              candidate_count=4, scorer=lambda obs, cs: [math.inf] * 4)
        self.assertEqual((episode.status, episode.steps), ("rejected", 0))
        episode = closed_loop(ToyObject(target=0, require_held=True), lambda obs, rng: proposals,
                              budget=4, seed=0, provenance=PROVENANCE)
        self.assertEqual((episode.status, episode.steps), ("success", 0))

    def test_cache_domains_and_task_replacement_match_uncached(self):
        values = dict(history=[0], entity_ids=["cup"], calibration="v1", embodiment="arm1", model="m1",
                      demo="demo-left", view="front", text="", binding={"target": "left"}, progress=0, noise=17)
        cache = InferenceCache()
        self.assertEqual(cache.synchronize(**values), (True, True))
        cache.physical["features"] = "robot"
        cache.task["answer"] = "left"
        self.assertEqual(cache.synchronize(**values), (False, False))
        for field in ("demo", "view", "text", "binding", "progress", "noise"):
            changed = dict(values, **{field: "changed"})
            cache.synchronize(**values)
            cache.physical["features"] = "robot"
            cache.task["answer"] = "old-task"
            self.assertEqual(cache.synchronize(**changed), (False, True))
            self.assertEqual(cache.physical, {"features": "robot"})
            self.assertEqual(cache.task, {})
        for field in ("history", "entity_ids", "calibration", "embodiment", "model"):
            cache.synchronize(**values)
            cache.physical["features"] = "old-state"
            cache.task["answer"] = "old-task"
            self.assertEqual(cache.synchronize(**dict(values, **{field: "changed"})), (True, True))
            self.assertEqual((cache.physical, cache.task), ({}, {}))
        for view in ("front", "side"):
            for target in ("left", "right"):
                values.update(view=view, demo=f"demo-{target}", binding={"target": target})
                cache.synchronize(**values)
                cached = cache.task.setdefault("answer", values["binding"]["target"])
                self.assertEqual(cached, target)


if __name__ == "__main__":
    unittest.main()
