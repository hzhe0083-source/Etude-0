# Unpaired video pretraining data

`evo_wam.video_data` accepts one continuous, single-view feature window. Human
video does not need a matching robot execution, actions, task-role labels or a
second camera. Robot replay can use the same schema and feature encoder. The
loader never manufactures a human/robot pair from task names or filenames.

## One manifest and one numeric NPZ

```json
{
  "format_version": 2,
  "kind": "video_pretrain",
  "arrays": "window-001.npz",
  "sample_id": "recording-17-window-001",
  "source_id": "original-recording-17",
  "source_group": "original-recording-17-and-reposts",
  "domain": "human",
  "feature_space_id": "wan-vae-checkpoint-and-preprocessing-v1",
  "feature_kind": "patches",
  "patch_grid": [20, 30],
  "patch_coordinate_system": "normalized_xy_patch_centers",
  "context_frames": 4,
  "provenance": {
    "description": "Recorded source identity, encoder hashes and frame sampling belong here."
  }
}
```

`features` contains both context and the following actually observed frames of
this same window. `context_frames=P` chooses the split, with `1 <= P <= T-2`.
There must be at least three feature frames; at least two are future targets.
This is a feature-frame count, not a raw-video-frame count. For a causal Wan
VAE, record the actual source timestamps of the encoded endpoints.

| NPZ array | Shape and meaning |
| --- | --- |
| `features` | floating `[T,N,D]`; N is patches or stable tracked entities |
| `feature_valid` | Boolean, exactly `[T,N,D]`; data-owned visibility/availability |
| `frame_times` | floating `[T]`, finite strictly increasing seconds on the recorded source clock |
| `patch_coordinates` | only for `patches`: required floating `[N,2]`, actual normalized patch-center `(x,y)` coordinates in feature-slot order |
| `entity_ids` | only for `tracked_entities`: required int64 `[N]`, unique nonnegative stable IDs |

Patch windows declare their actual `patch_grid=[H,W]`, with `N=H*W`, and
`patch_coordinate_system="normalized_xy_patch_centers"`. Every grid center
appears exactly once in `patch_coordinates`; a permutation of the full grid is
allowed when features, masks and coordinates are permuted together. The grid
is never inferred from `N`. Coordinates are static within a window and enter
both the encoder and future-feature predictor before spatial aggregation.
For zero-based row `r` and column `c`, `x=2*(c+0.5)/W-1` and
`y=2*(r+0.5)/H-1`, both strictly between -1 and 1. Canonical storage visits
rows before columns; explicit coordinates also permit a different slot order.

Tracked-entity windows do not supply patch-grid metadata or patch coordinates.
Each slot represents the same entity throughout the window. The encoder first
encodes each slot's temporal trajectory, then pools the entity set; numeric
entity IDs are identity checks, not position features. Reordering the entity
table consistently across frames does not change the operation representation.

`feature_space_id` identifies the fixed encoder and preprocessing convention,
including feature dimensions and token order. Different identities cannot be
silently pooled in one index. All additional source/video/encoder details go
in the metadata-only `provenance` object. They are not appended to model tokens.
`N` may differ across windows, but tracked entity slots must remain stable
throughout an individual context/future window.

`load_video_window(path)` returns `VideoWindow` with batch size 1 on feature,
mask and target tensors; `frame_times` remains `[T]`. Tracked `entity_ids` becomes
`[1,N]`. Patch coordinates retain their explicit slot order. Fully invisible windows are accepted. `has_training_signal` is false
if no context element is observed or no future feature/effect target is valid;
the training caller must skip its optimizer update. An effect label can supply
a future target when its feature is occluded, provided context is observed.

Finite values are checked only where the corresponding data mask is true.
Invalid entries, including NaN/Inf, are returned as zero with their masks still
false. This does not create observed zeros or negative contact labels. Losses
must continue to use the masks. Prediction confidence never changes them.

The NPZ is numeric and opened with `allow_pickle=False`; its key set is strict.
Action arrays, robot targets, forced-pair declarations and unrelated fields are
rejected. The NPZ path must stay inside the manifest directory. Metadata is
also explicit: source details beyond the listed fields go in `provenance`.

## Optional observed effect labels

Any subset of `geometry`, `relations`, and `events` is allowed, but each present
value array must have its matching `*_valid` array. Only `tracked_entities`
windows can supply these labels. Untracked VAE patches do not become objects
merely by assigning patch indices.

| NPZ arrays | Shape |
| --- | --- |
| `geometry`, `geometry_valid` | floating / Boolean `[T-P,N,G]` |
| `relations`, `relations_valid` | floating binary / Boolean `[T-P,N,N,C]` |
| `events`, `events_valid` | floating binary / Boolean `[T-P,N,N,E]` |

Relation channels are independently binary: grasp and support can coexist.
The first target corresponds to `frame_times[P]`. Event labels describe the
interval from the preceding feature frame to that target frame; an absent
event may be marked valid only if the whole interval was observed. Unobserved
events or contacts remain invalid, rather than being filled with zero labels.

When any effect field is present, four additional nonempty metadata strings
are mandatory:

- `effect_schema_id`: fixed channel meanings, directed relation conventions and geometry semantics.
- `geometry_frame`: the declared geometry coordinate convention, e.g. `object-relative-v1`.
- `geometry_units`: the schema's explicit units, e.g. `metres`.
- `evidence_source`: where these independently audited labels came from.

Even a relation-only subset identifies this schema. Geometry conventions and
normalization must be made compatible upstream; the loader does not transform
units, estimate poses, infer contact from hand proximity or infer robot effects
from an unrelated human clip. Without reliable labels, omit the effect fields
and train on observed future features alone.

## Indices and downstream source leakage

```json
{
  "format_version": 1,
  "kind": "video_pretrain_index",
  "samples": [
    {"manifest": "window-001.json", "split": "train"},
    {"manifest": "other-recording-window.json", "split": "test"}
  ],
  "bridge_sources": [
    {"source_id": "another-human-recording", "trajectory_id": "robot-execution-42", "split": "test"}
  ]
}
```

`load_video_index(path, split="train")` returns selected manifest paths and
source-only records for **all** splits, including `bridge_sources`. Records
contain no features or action labels and distinguish `record_kind="video"`
from `record_kind="bridge"`. Save this complete list in the pretrained encoder
artifact, together with feature-space and artifact identities.

`validate_video_sources(records)` is also public. Downstream bridge training
must combine the pretrained artifact's source records with its robot dataset
records marked `record_kind="bridge"`, and run the same check before training
or reporting unseen-demonstration tests. Bridge records require `source_id`,
`trajectory_id`, and `split`; `source_group` is optional. They create provenance
links only, not extra training pairs.

Splits are exactly `train`, `validation`, and `test`. The source graph joins:

- shared `source_id`, even when window IDs or source-group labels differ;
- shared `source_group`, covering reposts, alternate views and adjacent windows;
- robot trajectories explicitly named by `trajectory_id`; for robot video,
  `source_id` also names the trajectory unless `trajectory_id` is supplied;
- the source/trajectory links in audited bridge records.

Every connected component must stay within one split. The entire index is
checked before selecting the requested split, so hidden held-out records are
not excluded from the audit. `sample_id` is unique across the index. One index
uses one `feature_space_id` and at most one declared `effect_schema_id`; robot
replay is permitted under those same identities.

Source IDs should be globally stable; domain alone must not rename the same
recording. The caller remains responsible for recording repost/source aliases
and providing all relevant downstream bridge records. An empty bridge list
does not establish that an unlisted robot benchmark is uncontaminated.

The separate raw single-video preprocessing entry point may produce Wan patch
windows with the four required NPZ arrays: features, validity, frame times and
patch coordinates. No effect annotations or
robot actions are needed for that initial route. These inputs train only the
authorized video encoder/reader path, not a new action policy or planner.

Window manifests use format v2; index manifests remain format v1. Old v1
windows must be regenerated with the actual layout, rather than upgraded by
guessing a grid. Encoder artifacts and exported operation tokens also carry
version 2; v1 artifacts cannot resume this architecture and require fresh
pretraining, demonstration export and downstream reader training.
