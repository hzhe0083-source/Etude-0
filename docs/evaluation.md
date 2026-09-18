# Prefix-matched evaluation

`src/evo_wam/evaluation.py` contains simulator-independent rollout functions. The
only toy environment lives in the tests; passing those checks proves protocol
properties, not learned task understanding, RoboTwin success, or hardware readiness.

## Backend boundary

An environment provides `observe()`, `step(action)`, and `success()`. `step`
must synchronously execute one supplied action at the documented control period;
backend protections remain in force. Commands in replay records are commands
actually submitted, not a claim that measured motion equaled the command.
Observations must contain the measured signals needed to label physical effects.

Oracle additionally requires `snapshot()` and `restore(snapshot)`. The snapshot
must include the complete simulator, controller, task, elapsed-time, and random
state. Resetting only poses is insufficient. Real robots are explicitly rejected
by this Oracle entry point. Snapshots are restored after every comparison, also
when an evaluation raises. Snapshot failures must be resolved in the real backend,
not papered over with approximate resets.

Actions and observations must be JSON-compatible (convert model tensors to lists
at the policy boundary). Success must be checked from the environment's actual
state. A learned effect-matching threshold is not a hardware safety mechanism.

## Calling the evaluator

```python
from evo_wam.evaluation import Candidate, Provenance, closed_loop, result_json

def policy(observation, rng):
    # Replace with the real deployment policy, converted to atomic action steps.
    actions = deployed_policy(observation, rng)
    return [Candidate("proposal-0", tuple(actions))]

provenance = Provenance(
    kind="simulated", backend="RoboTwin-<exact-revision>",
    policy_id="Evo-WAM-<config-digest>", checkpoint_id="<weight-digest>",
    success_criterion="original-<criterion-revision>",
)
episode = closed_loop(env, policy, budget=500, seed=7, provenance=provenance)
result_json(episode, "episode.json")
```

`closed_loop` defaults to the first-round single-candidate protocol and rejects a
scorer in that mode. Four-candidate execution requires `candidate_count=4` and a
`scorer(observation, candidates)` that returns four nonnegative costs. Candidates
must have unique IDs and the same horizon. The default ranker chooses the lowest
finite cost; an optional locked `max_cost` rejects high-cost predictions. Rejection
sends no new actions. Nonfinite predictions are rejected, not converted to zero.
Ties use candidate order. All methods must receive the exact same saved candidate
set; `select_baseline` supplies seeded random or score-permutation baselines.
Random selection deliberately has no learned feasibility filter.
Pass the same locked `max_cost` to `select_baseline(..., costs=..., max_cost=...)`
for the score-permutation comparison. NaN and either infinity are ineligible;
only finite negative costs raise a validation error.

One control decision executes `max(1, H // 4)` actions and then reobserves and
replans. The total step budget truncates the final prefix. Observed task success
may terminate a prefix early; deployment, replay, and Oracle use the same rule and
record the actual length. A backend requiring separate failure termination must
raise or implement that handling outside these minimal callbacks; do not infer
failure from absence of success.

## Three distinct metrics

1. **Prefix progress:** supply `progress(before, after)` for physical progress
   diagnostics. The raw prefix observation is always saved, even without a scalar.
2. **Local Oracle@4:** call `local_oracle_at_4(env, fixed_candidates,
   frozen_continuation, continuation_id=..., continuation_budget=..., budget=..., seed=...,
   provenance=...)`. Each branch restores the same snapshot, executes the same
   prefix rule, and uses the same continuation budget and seeded RNG. `budget`
   is the deployment episode's remaining total action budget: it truncates the
   prefix and caps continuation to the remaining balance. For example, `H=12`
   with `budget=2` executes at most two prefix actions and no continuation. With
   `H=4`, that budget allows one prefix and at most one continuation action.
   Without an explicit budget, the total is one full prefix plus the requested
   continuation budget. Both Oracle and episode artifacts record the total.
   The
   continuation must be a frozen, stateless single-candidate policy. Its only
   stochastic input is the supplied RNG; closures with counters, global random
   generators, stale KV caches, or online parameter updates invalidate the test.
   Seeded stochastic draws must be consumed in a documented order. This is a local
   decision diagnostic under that continuation rule, not a strict system-wide
   success upper bound.
3. **Closed-loop task success:** measured by `closed_loop` from actual observations
   over the full selection and replanning episode.

`Replay.label_prefix_valid` has the length of the **original** proposed action
sequence. Only original actions actually executed have true entries. Continuation
commands and observations are stored separately. Train on the actual concatenated
commands if using continuation labels; never attach the continuation outcome to
unexecuted original candidate actions. `prefix_observation` and `prefix_progress`
are captured before continuation, even if the continuation later succeeds.
This mask represents execution eligibility only. Intersect it with each field's
data-owned `label_valid`; execution does not make occluded contact labels reliable.

Candidate-distribution adaptation belongs only to the training split. Lock ranking
scales and rejection thresholds on independent validation data. Add collected
interactions to matched-data baselines and budget accounting. Run original and
Zero-WAM-modified RoboTwin criteria separately and preserve the criterion identity
in `Provenance`; do not pool them into an unlabeled success rate.

`result_json` writes deterministic, sorted JSON with schema version, metric type,
seed, checkpoint, backend, policy, and success criterion. `kind` must be `toy`,
`simulated`, or `real`; there is no default that silently turns a toy score into a
robot result. Nonfinite values are prohibited in artifacts.

## Cache synchronization

Call `InferenceCache.synchronize(...)` before every deployment query:

- Physical domain: history content/version, stable entity IDs, calibration,
  embodiment, model checkpoint/version. Changing any clears physical and task caches.
- Task domain: demonstration content/version, view, text, binding, task progress,
  sampling noise/seed. Changing any clears task KV and generated-future caches.

All keys are explicit keyword arguments. Use immutable content hashes for large
arrays rather than filenames or object identity. Update history after every
executed prefix. A task replacement must synchronize before consulting any cached
value. The cache is in-memory only and does not override a model's internal KV;
the deployment integration must store/rebuild its caches in the corresponding
domain. Pure observation cache reuse must never include task-conditioned features.

## Real RoboTwin integration boundary

The pinned upstream entry points remain in
`third_party/Zero-WAM/evaluation/robotwin/`: `launch_server.sh`,
`launch_client.sh`, and `run_icl_eval.sh`. Start with the native client and convert
its measured observations and action chunks at the callback boundary above. Use
the actual robot control frequency and action normalization from that client.

The upstream client is not asserted to provide full simulator snapshots. Before
Oracle@4, implement and validate complete snapshot/restore in the installed
RoboTwin environment, including its controller and RNG state. Until then, run
native closed-loop evaluation and prefix diagnostics only. There is no fabricated
RoboTwin backend here. Consult the pinned `INSTALL.md` for its required environment
revision and the `move_stapler_pad` / `stamp_seal` criterion modifications.

Run protocol checks with:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -p test_evaluation.py -v
```

The toy four-condition test deliberately supplies the target to its fixture
policy. It verifies that the evaluator preserves task-specific later branches and
common initial prefixes; it does not demonstrate video understanding. The
hold/release fixture likewise verifies control-state accounting only.
