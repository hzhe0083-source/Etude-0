# RoboTwin execution bridge

`etude.robotwin` connects normalized model commands to an already initialized
RoboTwin episode. Full weights, RoboTwin, assets and visual entity tracks are
server-side inputs. No weights are downloaded by this module. Local tests use a
fake environment and verify API/control semantics, **not simulator performance**.

## Exact action convention

The bridge follows the pinned Zero-WAM source at
`08e2c4ae41e2b63573a299825cebe6753481407c`:

- `wan_va/configs/va_robotwin_cfg.py`: quantiles and channel mapping.
- `wan_va/wan_va_server.py::postprocess_action`: inverse quantile normalization,
  including `1e-6` in the denominator/range.
- `evaluation/robotwin/eval_policy_client_openpi.py::add_init_pose`: world-frame
  translation addition and `initial_rotation * relative_rotation` using SciPy's
  `xyzw` convention; quaternion normalization before execution.

Each candidate command is 30-dimensional and normalized. Active channels map to
16 EE values via `[0..6, 28, 7..13, 29]`; other channels must be zero. A command
becomes `[left_xyz, left_xyzw, left_gripper, right_xyz, right_xyzw, right_gripper]`
and is submitted through `env.take_action(command, action_type="ee")`.
The reference pose is captured **once at the start of the episode**, never at
each replan. A nonzero initial action count is rejected. The stats file's SHA-256
must equal the policy's normalization artifact digest.
Every observation provider must return an object with an `action_space` mapping
(normally `data.Observation`). Before calling the policy, the bridge checks
`representation="zero-wam-normalized"`, `dimension=30`, the exact active-channel
mask, and `normalization_id="sha256:<stats-file-digest>"` against its transform.
`NativePolicy` independently checks that mapping against the trained artifact.
Passing two independently valid but different normalizers is rejected.

Executed history contains only commands whose `take_action` call completed and
incremented the environment's action count once. Both absolute EE targets and
their canonical normalized equivalents are recorded, with integer control steps
and times from the configured real `control_dt`. Quaternion canonicalization
matches native training; executed history is not clipped to hide out-of-range
commands. These are submitted controller targets, not claims about measured
movement. Unexecuted candidate tails never enter history.

## Oracle-requirement diagnostic

`run_oracle_requirement(bridge, requirement_provider, policy, *, budget, seed,
provenance, requirement_source)` re-observes after every prefix and rebuilds the
oracle requirement from the **current** state. Its policy receives
`(observation, requirement, rng)` and returns one `evaluation.Candidate` whose
actions are normalized 30-D lists. It executes `max(1, H//4)` steps, respecting
the remaining budget, environment step limit, success and an optional terminal
predicate. Pass `NativePolicy.oracle_candidate`: it encodes the supplied
requirement with the trained codec and invokes the same native video/action
sampler used for file prediction. No model implementation is needed in the
episode factory. It must not replace the model with the RoboTwin expert trajectory.
`RequirementRejected` stops the loop with `status="rejected"`, the rejection
reason and the actual `commands_sent` count; no additional action is sent.
Other exceptions propagate so implementation errors are not mislabeled as
ordinary model rejection.

This diagnostic isolates the executable interface and sampler. It bypasses human
demonstration understanding and must not be reported as demonstration transfer,
ICL performance, candidate ranking or Oracle@4. Exact simulator snapshot/replay
is not implemented by this bridge.

Both `original` and `zero_wam` success callbacks and versioned source identifiers
are mandatory. Evaluate both pure predicates on the same state; choose and fix
one as the stopping rule before evaluation. Keep the original predicate before
applying the upstream patches for `move_stapler_pad` and `stamp_seal`: calling
the patched `check_success` twice is not dual reporting. The result stores each
predicate's success at the final state and whether it was satisfied at any
observed control step. Changing stopping rules can change the trajectory and
must be identified separately.

## Server entry point

```bash
python -m etude.robotwin \
  --factory server_episode:build \
  --config /server/configs/oracle_episode.json \
  --output /server/results/oracle_episode.json
```

`server_episode.build(config)` initializes the real environment and returns the
keyword arguments accepted by `run_oracle_requirement`. It supplies these
deployment-specific objects:

1. A `RoboTwinBridge` around an environment with `get_obs`, `take_action`,
   `take_action_cnt` and `step_lim`. Its observation provider receives raw
   `get_obs()` and a copy of the executed history and returns the model's current
   observation. Use observed visual tracks and calibrations; if simulator state
   is used for entity features, explicitly label the result as oracle perception.
2. A requirement provider tied to the current observation, with its source ID.
3. `NativePolicy.from_artifact(config["artifact"], checkpoint=config["checkpoint"],
   device="cuda").oracle_candidate`, loading the Etude artifact and base weights
   from configured local server paths. The bridge neither locates nor downloads
   them; weights may remain path placeholders until server training completes.
4. `Provenance(kind="simulated", ...)` identifying the actual policy, checkpoint,
   backend and stopping predicate. Fake API tests use `kind="toy"` instead.

Environment initialization, task/robot configuration, valid cameras, visual
tracking and pure original/modified success functions depend on the server's
RoboTwin installation. The factory is that explicit boundary; the shared bridge
owns action conversion, prefix execution, history recording and dual reporting.

The model part of the factory is only:

```python
from etude.inference import NativePolicy

native_policy = NativePolicy.from_artifact(
    config["artifact"], checkpoint=config["checkpoint"], device="cuda"
)
# Include this bound method as the runner's policy argument alongside the
# initialized bridge, current-state requirement provider, and provenance.
policy = native_policy.oracle_candidate
```
