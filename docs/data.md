# Single-view data, optional synchronized pairs, and experiment inputs

`etude.data` owns sampling and validates inputs. It does not infer contacts,
task necessity, human/robot alignment, or camera calibration from a filename.
All IDs and annotation provenance stay in metadata; they are never appended
to model tokens or task text.

## A sample is one JSON manifest and one numeric NPZ

```json
{
  "format_version": 2,
  "kind": "training_sample",
  "arrays": "arrays.npz",
  "source_id": "original-human-recording-17",
  "trajectory_id": "robot-execution-42",
  "history_id": "robot-execution-42-step-120",
  "window_start": 120,
  "observation_step": 120,
  "executed_steps": 8,
  "coordinate_frame": "robot_base",
  "control_dt": 0.05,
  "view_ids": ["front", "side"],
  "pair_kind": "synchronized_views",
  "task_annotation": "audited-requirements-v2",
  "actions_per_frame": 2,
  "action_space": {
    "representation": "zero-wam-normalized", "normalization_id": "audited-arm-v1",
    "dimension": 3, "valid_channels": [true, true, true]
  },
  "observed_action_space": {
    "representation": "zero-wam-normalized", "normalization_id": "audited-arm-v1",
    "dimension": 3, "valid_channels": [true, true, true]
  },
  "history_chunks": [
    {"mode": "video", "slice": [0, 2], "frame_id": 0, "rope_offset": 0},
    {"mode": "action", "slice": [0, 3], "frame_id": 1, "rope_offset": 0}
  ]
}
```

Robot-supervised `training_sample` records accept **one or two** recorded
demonstration views. The manifest identifies exactly one robot
history/trajectory/window, so demonstrations cannot accidentally refer to
different robot targets. `window_start`
is the absolute control-step index of the current robot state; all offsets
below are relative to that state. `executed_steps` is the number of actions
actually executed from the stored plan, excluding any replacement continuation.
Frame transforms, entity ID tracking, and `control_dt` must be fixed upstream.
`observation_step` equals `window_start` for training. The array manifest uses
v2 intentionally: missing historical commands or new semantic/evidence labels
in v1 cannot safely be reconstructed, so the loader rejects v1. Migration must
return to recorded commands, timing and annotations; never fill missing evidence
with `True` or copy candidate actions into history.

`pair_kind` is `none` or `synchronized_views`. A synchronized pair needs exactly
two distinct view IDs and an audited common source execution/time interval;
only that explicit declaration permits cross-view consistency. Two video
arrays, the same task label, crops or color augmentations do not establish a
reliable viewpoint pair. A single view uses `pair_kind="none"` and only its
own arrays and evidence masks; it is never copied into a second branch.

Existing v2 two-view files without `pair_kind` still load, conservatively as
`none`: their supervised branches remain available, but CV is disabled until
sync provenance has been audited and declared. The updated paired fixture
explicitly declares synchronization. A two-view `none` record does not imply
matching viewpoints or time; both demonstrations still need the independently
annotated robot task targets for that record. Different original human source
recordings should be separate records with their own `source_id`.

An ordinary unpaired video with no robot trajectory, actions or task labels
is **not** a `training_sample`. It belongs to the separate `video_pretrain`
data kind and pretraining loader. Do not fabricate a robot example or demand
a synchronized partner to admit that video into representation pretraining.

## Demonstration encoding identity

Feature width alone does not distinguish visual features from learned effect
tokens. Both training and observation manifests can carry
`demonstration_encoding`; omitting it preserves v2 behavior as exactly
`{"kind":"raw_features"}`. An explicit raw declaration accepts no other keys.
Learned tokens must declare this exact structure:

```json
{
  "demonstration_encoding": {
    "kind": "video_effect_tokens",
    "encoder_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "feature_space_id": "audited-frozen-visual-feature-space",
    "token_dim": 32,
    "window_frames": 3,
    "num_tokens": 4
  }
}
```

The digest identifies the actual encoder artifact, not a model nickname; the
loader normalizes its hex representation to lowercase. `feature_space_id`
must be nonempty, `token_dim` must match the actual final feature axis,
`window_frames` must be at least 3, and `num_tokens` must be positive. All views
in one record must share their feature width and encoding declaration. Unknown
fields, explicit null metadata or mismatched widths are rejected.

`validate_demo_encoding(manifest, width)` returns the normalized identity.
`LoadedSample.demonstration_encoding` derives it from metadata, while
`Observation.demonstration_encoding` carries it into deployment. Training,
checkpoint loading and policy use must compare this identity with their
registered encoder; equal dimensions alone never permit raw/token substitution.
The data loader validates the declaration but does not load encoder weights or
claim that a digest has been verified against an artifact by itself.

`load_sample(path)` returns a `LoadedSample`, adding batch size 1 to arrays
except time grids. It calls the semantic contracts' `validate()` methods.
Physical labels use scene slots; requirement labels use task roles, bound
to scene slots by `*_binding`. Relations are directed and independently
binary, allowing contact, support, and grasp to coexist.

| NPZ keys | Unbatched shape / dtype |
| --- | --- |
| `entity_ids` | `[N]`, int64 stable IDs; `-1` is padding |
| `step_offsets` | `[T]`, int64 positive, strictly increasing action offsets |
| `robot_history` | `[L,N,entity_dim]`, finite observed scene features in entity-table order |
| `proprio_history` | `[L,proprio_dim]`, finite observed robot state, aligned to history |
| `embodiment` | `[embodiment_dim]`, finite robot-format/capability features |
| `entity_patch_weights` | `[N,S_video_patches]`, nonnegative observed ROI weights, independent of task binding |
| `robot_latent` | `[C,F,H,W]`, observed floating video latent |
| `observed_action_history` | `[K,A]`, normalized commands confirmed executed; explicit `K=0` is allowed |
| `observed_action_step_offsets` | `[K]`, int64, increasing command-end offsets relative to current observation, all `<=0` |
| `observed_video_step_offsets` | `[F]`, int64, increasing observed frame offsets, all `<=0` |
| `actions` | `[H,A]`, finite floating planned actions |
| `demo_view_i` | one finite floating deterministic input per declared view; no `demo_view_1` for a single-view record |
| `outcome_geometry` | `[T,N,Dg]`, floating |
| `outcome_relations`, `outcome_events` | `[T,N,N,C]`, `[T,N,N,E]`, floating binary labels where valid |
| `outcome_{field}_valid` | Boolean, exactly the matching outcome shape |
| `{part}_step_offsets` | `[Tpart]`, int64; current and remaining may differ |
| `{part}_binding`, `{part}_binding_valid` | `[R]`, int64 slot or `-1` unused / `-2` unmatched / `-3` uncertain; Boolean validity |
| `{part}_geometry` | `[Tpart,R,Dg]`, floating |
| `{part}_relations`, `{part}_events` | `[Tpart,R,R,C]`, `[Tpart,R,R,E]` |
| `{part}_{field}_valid`, `{part}_{field}_required` | Boolean, exactly the matching requirement shape |
| `{part}_geometry_tolerance`, `{part}_geometry_tolerance_valid` | floating nonnegative interval half-width and Boolean validity, matching geometry |
| `{part}_event_windows`, `{part}_event_windows_valid` | int64 inclusive `[Tpart,R,R,E,2]` windows; Boolean validity `[Tpart,R,R,E]` |
| `{part}_event_precedence`, `{part}_event_precedence_valid` | int64 `[P,2]` event-node edges and Boolean `[P]` validity |
| `view{v}_{part}_{semantic}_valid` | Boolean shape matching the respective label-valid mask for all seven semantic fields |
| `view{v}_relations_valid`, `view{v}_events_valid` | matching physical relation/event shape, Boolean per-view evidence validity |

Here `part` is `current` or `remaining`, `field` is `geometry`, `relations`,
or `events`, and `v` ranges over the one or two recorded views. Requirements may refer beyond the local plan;
physical outcome offsets may not exceed `H`. Required masks, valid labels,
and model uncertainty are distinct. Actual trajectory events must not simply
be copied into necessary-event annotations.

The seven per-view semantic fields are `binding`, `geometry`, `relations`,
`events`, `geometry_tolerance`, `event_windows`, and `event_precedence`.
Precedence node IDs index flattened `(Tpart,R,R,E)` event entries. A valid
`(-1,-1)` edge slot means known absence of another constraint; an invalid slot
means unknown, not permission to ignore order. Required roles cannot be UNUSED;
UNMATCHED/UNCERTAIN retain their requirements and cannot authorize execution.
The loader requires every new value and validity array explicitly and checks
window consistency and the precedence DAG through `EffectRequirement`.

Present entities must have positive observation patch mass, and padded entity
slots must have zero mass. Patch weights describe pure robot observations,
never task roles or target attention. Their patch axis must match the adapter's
video token layout; that checkpoint-specific check occurs in the adapter.
In a native packed teacher-forcing window, pool weights must be zero after the
first prediction chunk. The interaction head forecasts relative to the current
state and must not read later future tokens to recover its labels. The trainer
checks this causal pooling boundary using the actual patch and chunk sizes.

The loader intersects physical validity with `step_offsets <= executed_steps`.
It does not relabel continuation results as effects of unexecuted planned
actions. Requirement validity is not truncated: task requirements can extend
beyond what has already been executed. An event's absence is valid only when
its entire annotated interval was observed. The producer must mark missing or
partially observed intervals invalid; the loader does not manufacture negatives.

`per_view_valid=(view0,)` or `(view0,view1)` preserves each independent data-owned
evidence map. Keys use dots, e.g. `current.binding`, `remaining.event_windows`; physical
head keys are `relations` and `events`. The loader does not AND the views or
replace evidence with label availability. Training intersects each view's
evidence with that field's `label_valid`; optional CV additionally intersects
the two synchronized views. Prediction confidence never changes these masks. An occluded view's
missing evidence must not delete the other view's valid supervision.

`np.load(..., allow_pickle=False)` rejects objects. Exact NPZ keys, contract
shapes, dtypes and numeric values are validated. NPZ paths must remain within
the manifest directory. Metadata is separate from all model input dataclasses.

## Actual action history and deployment observations

`ObservedActionHistory` stores `commands [1,K,A]`, `step_offsets [K]`, the
absolute `observation_step`, `control_dt`, and its own declared `action_space`.
Its normalization declaration must exactly match
the current action space; unused channels must be zero. Every offset must be
`<=0`, increasing, and not before episode start. Positive future offsets,
nonmonotonic timestamps and normalization mismatches fail before inference.
The training `actions` field remains the separate future candidate/label
window and is never read when building history.

`history_chunks` references only the observed video latent (video slices count
latent frames) or observed commands (action slices count executed commands).
Slices cover each stream once, in order. `frame_id` follows native computation:
video even, action odd, globally increasing. It is not a physical timestamp.
`rope_offset` is an independent latent-frame coordinate; matched video/action
chunks have the same origin and same-modality intervals cannot overlap.
Sequence IDs remain the native episode IDs, never the chunk number.

Both `LoadedSample` and `Observation` expose `native_history(dtype=None)`.
Commands are packed as `[1,A,F,actions_per_frame,1]`. A partially executed tail
is retained with zero rectangle padding and a flat Boolean `token_valid` mask;
the native adapter masks these padding tokens from cache attention. The loader
never pads with unexecuted candidate commands or drops a valid executed tail.
Explicit zero-length action history is permitted, with no action descriptor.

`sampling_position()` returns the next even attention `frame_id` after the
maximum recorded chunk, and `rope_offset` at the greatest covered latent-frame
end. This is independent of the raw tensor length or absolute control step.
Use the same helper for training-time generated futures and deployment.

Deployment uses `kind="observation"`, v2, and `chunk_size` plus the above timing,
history and action-space metadata. Its NPZ contains entity IDs, robot/proprio
history, embodiment, observed robot latent, explicit past commands/offsets and
demonstration inputs. It cannot contain future `actions`, outcomes, true task
requirements or native teacher-forcing streams. `load_observation` enforces the
exact key set; changing the manifest kind cannot bypass that boundary.

## Native Zero-WAM stream arrays

For native integration, a manifest may declare additional numeric arrays:

```json
{
  "native_arrays": {
    "latent_dict": {
      "latent": "native_video_clean",
      "noisy_latents": "native_video_noisy",
      "timesteps": "native_video_times",
      "cond_timesteps": "native_history_times",
      "grid_id": "native_video_grid"
    },
    "action_dict": {
      "latent": "native_action_clean",
      "noisy_latents": "native_action_noisy",
      "timesteps": "native_action_times",
      "cond_timesteps": "native_action_history_times",
      "grid_id": "native_action_grid"
    },
    "mcp_latent_dicts": []
  },
  "native_scalars": {
    "chunk_size": 2,
    "max_frame_chunk_size": 4,
    "window_size": 8
  }
}
```

Native arrays already include their native batch/layout axes; the loader does
not reshape them. Optional stream keys are `targets`, `valid_mask`, and
`actions_mask`. Arrays must use the `native_` prefix and have finite numeric
values. Text, ICL, requirement and arbitrary cache keys are rejected. The
result is returned separately as `sample.native_inputs`; the native adapter
validates model dimensions and controls which training branch may read it.
Clean future training streams must never be reused as inference evidence.
Normally fresh shared noise is generated by the training loop; stored noisy
streams are useful only for deterministic integration/replay checks.

## Shared denoising and condition dropping

`prepare_training_input` calls the history encoder once and each target encoder
once for one or two conditional views; two branches share the resulting
`DenoisingInput` objects.
Every named target has its own sigma table, query offset, Gaussian draw, and
time draw. Pass tables from the pinned upstream `FlowMatchScheduler`; the data
module does not implement another schedule. `time_dim=2` supports native
per-frame video timesteps. Across views the same target's clean/noisy latent,
noise, time and query offset are identical; across targets they are independent
draws using the respective tables. Robot augmentations belong before this
shared encoder call, never independently inside each view branch.

The selector draws a 90% conditional example or one 10% unconditional branch.
Conditional examples retain their actual one or two demonstrations and use
empty task text. Its `pair_kind` argument defaults to `none`; enabled CV still
requires `synchronized_views` and exactly two views. A nonzero global CV
coefficient does not turn a single view or unaudited pair into a consistency
example; those samples keep their supervised losses and skip only CV.
Unconditional branches carry `TaskCondition()` with demonstration, current,
remaining and task cache all `None`, and empty text. Their `loss_enabled` map
allows only next-video and enabled IFP losses. It disables requirement,
execution, interaction and cross-view objectives, so no true-requirement
teacher is called for unconditional samples. The loop must honor these flags
before reading labels or building teacher computation.

`paired_dropout_disabled` disables `nn.Dropout` children within its context and
restores their original state. The adapter must separately ensure functional
dropout and attention dropout probabilities are zero. Identical demonstrations
and shared inputs must yield identical results. Legacy upstream `drop_icl`
and `droptext_target` are zero in all supplied configurations; the unified
selector is the only task-condition drop source.

## Source grouping and fixed comparisons

`connected_components` joins every human `source_id` with its paired robot
`trajectory_id`. All views/augmentations of that source and all windows of that
trajectory must use the same IDs. Multi-to-multi connections are transitive.
`assign_splits` deterministically hashes full components with a seed;
`validate_splits` rejects any component crossing train/validation/test. Inspect
realized counts: fractions are expectations over components, and a giant
component cannot be safely subdivided. Task holdouts can be applied upstream;
component constraints still have to hold after assignment. Do not silently
move individual windows to repair split imbalance.

The seven JSON configurations are matched pilot defaults, not a claimed
optimal hyperparameter search result. They share seeds `[0,1,2]`, 1,000 update
steps, batch size 1, LoRA rank/alpha 8, 8 current and 8 remaining tokens, source
splits, noise configuration, the same recorded-view sampling, single-candidate deployment
and no F ranking. Samples can contain one or two real views; keep the same
mixture across comparisons and report how many examples have eligible CV pairs.
Dimensions are explicitly **synthetic fixture dimensions**;
real robot and checkpoint schemas must be supplied and validated rather than
guessed. Each token count is roles times its part's query count.

T0/T1/T2 enable recent video / plus IFP / plus interaction supervision. T2 and
V0 are identical. V0 and V1 differ only in `lambda_cv` (0 vs pilot 0.1). Both
compute the actual number of supervised view branches and take their mean.
Only explicitly reliable pairs can contribute the extra CV term. `geometry.json` and
`full.json` differ only in interface content; both retain the same IFP,
interaction supervision, CV fields and token capacity. Full-only interface
control labels must not add extra CV fields relative to geometry.

`load_experiment(..., for_test=True)` rejects configurations until
`validation_locked` is true. Lock shared settings using validation results,
record actual update count / GPU time / sampling overhead, then run test once.
Do not claim a compute match solely because the update counts are equal.

Configuration schema v2 records all objective coefficients explicitly in
`training.loss_weights`; its `cv` entry must agree with `lambda_cv`. It also
records `exec_start_step` (pilot: 100), bounded `max_precedence_edges` capacity
(pilot: 4), and validation-locked binding confidence/margin thresholds. Both
copies of the CV coefficient change together in V0/V1. Missing coefficients,
invalid warmup boundaries or unregistered binding refusal thresholds fail
configuration validation. Formal test evaluation requires both experiment
and binding-policy locks.

Run the data checks with:

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_data.py -v
```

These checks use temporary numeric NPZ files and CPU tensors. They verify
contracts and invariants; they are not robot performance measurements.
