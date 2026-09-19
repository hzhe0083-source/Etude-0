# Native human-video ICL

The current experiment extends the existing Zero-WAM training path with one sample type: **watch human demonstration A, then predict the continuation of an independent human execution B from its earlier observations**. Robot samples retain the native video/action objective. The additional human samples do not require robot trajectories, human pose estimates, or action labels.

This is an implementation route for an experiment, not a claim of novelty or measured transfer. Real pretraining data has not yet been collected. The earlier [B/P effect-encoder route](unpaired_video.md), G/Q requirement interface, relation supervision, and F ranking remain optional experiments; none is a prerequisite here.

## Information and gradient path

```text
Human sample: complete A + earlier B chunks -> native video/context attention -> B continuation
Robot sample: demonstration + robot history -> native video and action branches -> recorded targets
                                             |
                                             +-> shared video-attention LoRA updates

Deployment: new human demonstration + robot observations -> original Zero-WAM execution
```

The input is the frozen Wan VAE latent in its native `[C,F,H,W]` layout, normalized once as `(posterior_mode-mean)/std`. A enters through `icl_latent_dict`, not a learned B encoder or a new requirement-token adapter. Trainable LoRA covers video self-attention Q/K/V/output and text cross-attention Q/output in the native blocks. The base model, text K/V, and action expert are frozen. Robot action loss still provides gradients to the video adapters through the existing computation graph; frozen parameters do not imply a detached forward pass.

Human forwarding reuses the native embedding, blocks, position grids, attention masks, output head, and optional MCP/IFP helper. It omits `action_dict` and passes no action hidden states. There are no fabricated zero actions, action masks used as pseudo labels, robot proprioception, geometry/relations/events labels, or human action loss. Robot forwarding remains the native `forward_train` path.

Training uses native **chunkwise teacher forcing**. A noisy B query may read earlier clean B chunks and the complete A context. It cannot read clean B from the same or a later chunk. Thus a late training query can use more recorded history than an early query; the whole rollout does not share one fixed clean prefix. `history_frames` is measured in latent frames and excludes the initial observed prefix from the human target loss. It must agree with the configured native chunk boundaries. Robot samples retain upstream whole-window supervision: video loss uses the weighted full-tensor mean, action loss uses the full-tensor mean after masking, and MCP uses its valid-target count. Human history exclusion does not silently rescale the robot objective.

No target task description is added as an input. Semantic matching evidence remains metadata for audit. The ordinary-human-video control omits A entirely from the model input while predicting the same B targets.

## Data contract

Each `native_icl_sample` JSON has `format_version: 1`, `sample_id`, `feature_space_id`, `latent_normalization`, `history_frames`, `demonstration`, `target`, and `compatibility`. Optional `provenance` is audit metadata only.

Each video record declares `source_id`, `source_group`, `domain` (`human` or `robot`), and a relative `arrays` NPZ path. Robot records may declare `trajectory_id`. Paths must stay inside the containing manifest directory; no remote downloads are performed by this loader.

| Arrays | Demonstration / human target | Robot target |
|---|---|---|
| `latent` | Finite floating `[C,F,H,W]` | Same layout |
| `frame_times` | Strictly increasing floating `[F]`, in seconds | Same layout |
| `actions` | Absent | `[A,F,N,1]`, aligned with target latent frames |
| `actions_mask` | Absent | Boolean array with exactly the actions shape |

A robot target also declares the existing explicit `action_space` contract; native model loading verifies the channel and action dimensions. These dimensions are taken from the selected checkpoint and audited data, not guessed from a test configuration. Target clips must contain future frames after `history_frames`. IFP offsets beyond the clip carry no labels and contribute zero loss, following the native masking convention; each metrics row records `ifp_valid_values` per head so a configured but unsupervised horizon is visible. Compare this coverage between runs rather than assuming every enabled horizon received supervision.

For a human target, A must also be human. A and B must be independent recordings with different source identities, repost groups, and array files. `compatibility` must be an object with `kind: "audited_semantic_task"` and a nonempty `evidence` string. Evidence should identify a reviewed task annotation or reviewed retrieval decision. Sharing an object category does not establish matching intent; putting a cup into a box and taking one out are not interchangeable.

This is human-to-human semantic pairing. It does not require synchronized cameras, frame alignment, or an associated robot trajectory, but it is not zero-correspondence training. An ordinary continuation dataset can include unpaired videos in principle; the matched H0/H1 comparison here deliberately uses the same audited index and disables A only in H0.

A `native_icl_index` JSON has `format_version: 1`, `kind: "native_icl_index"`, and `samples`, whose entries contain exactly `manifest` and `split` (`train`, `validation`, or `test`). Optional `source_aliases` and `bridge_sources` connect reposts and existing human/robot correspondence records for leakage auditing; they do not create model inputs or training pairs. The audit includes both A and B, their transitive source groups, and robot trajectories. Connected recordings cannot cross dataset splits. All records use one fixed `feature_space_id`.

## Preparing continuous video clips

`preprocess-icl-video` caches one screened, continuous local clip with the frozen Wan VAE:

```bash
evo-wam preprocess-icl-video \
  --manifest /server/data/raw-human-a.json \
  --output /server/data/human-a --device cuda
```

The `raw_icl_video` version-1 manifest declares `video`, `vae_path`, `vae_sha256`, `size: [height,width]`, `fps`, source identities, `domain`, `feature_space_id`, and `continuous_segment_verified: true`. `vae_sha256` records the frozen config and weight checksums. Image dimensions must be multiples of 16; video and encoder paths are local. This command does not detect cuts or decide semantic compatibility.

The output contains `clip.npz` (`latent`, `frame_times`) and `clip.json` provenance. A whole-clip causal encoding preserves Wan's temporal normalization and endpoint convention; latent timestamps refer to actual sampled source-frame endpoints. After audit, arrange the A/B caches under their sample manifest directories and construct the sample/index JSONs described above. Preprocessing one clip does not by itself produce an audited training pair. Robot action arrays and their validity come from real robot records, not this RGB-only command.

## Three matched configurations

| Config | Human input | Robot updates | Human updates |
|---|---|---:|---:|
| `configs/icl/R0_robot_only.json` | None | 250 | 0 |
| `configs/icl/H0_video_only.json` | B history, no A | 250 | 750 |
| `configs/icl/H1_cross_video.json` | A demonstration + B history | 250 | 750 |

R0 uses the same local adaptation protocol without extra human updates. It is not a reported result for an untouched published checkpoint; evaluate that checkpoint separately when establishing the reference performance. All runs start from the same base model and use the same robot target order. H0/H1 use the same human index, B target order, loss weights, optimizer settings, and seeds. Every human video used as A must also occur as a B target in the split used for this matched training comparison, referencing the same source identity and clip cache, so the ordinary-video control also receives that clip's prediction supervision. A different window from the same recording is insufficient. Reciprocal A→B and B→A samples are one way to meet this requirement.

H0/H1 repeat the domain schedule `robot, human, human, human` for 1,000 updates; R0 performs 250 robot updates. This matches robot exposure, not total compute. H1 also processes demonstration tokens absent from H0. Report actual updates, video/frame exposure, demonstration lengths, runtime, and peak memory rather than describing equal step counts as equal compute.

The checked-in files retain `synthetic_dimensions_only: true`; the native model supplies its own dimensions. They use LoRA rank/alpha 8, learning rate `1e-4`, gradient clipping 1, `lambda_human=1`, chunk size 2, maximum frame chunk size 4, window size 32, ICL height offset 24, and the configured video/MCP noise shifts. IFP is enabled with four weights `[0.5,0.25,0.15,0.1]` and future chunk stride 2. These are test/starting settings, not validated real-data hyperparameters. Keep IFP and all loss settings matched between H0/H1; validate target coverage and checkpoint MCP compatibility before training.

## Training, recovery, and deployment

Use a reviewed copy of the configuration with the real-data marker and data/checkpoint constraints set for the experiment:

```bash
evo-wam train-native-icl \
  --config /server/configs/H1_cross_video.json \
  --index /server/data/native-icl-index.json \
  --checkpoint /server/zero-wam \
  --steps 1000 --seed 1 --device cuda --output /server/runs/H1
```

`--tiny-native` selects a randomly initialized native small model instead of `--checkpoint` for computation-graph checks. It still requires correctly shaped sample arrays; it is not a trained control policy. Use `--resume` with the saved training artifact to continue within the configured cumulative step budget. Keep the index, consumed data, configuration, and base checkpoint consistent when resuming.

Export merges the learned adapters into the original model's linear layers and writes native checkpoint files:

```bash
evo-wam export-native-icl \
  --artifact /server/runs/H1/native_icl.pt \
  --checkpoint /server/zero-wam \
  --device cpu --output /server/models/H1
```

Use the merged model through the original Zero-WAM demonstration/robot inference interface. No new B/P, effect reader, requirement decoder, ranking network, or test-time parameter update is needed. Training artifacts and small-model exports do not themselves establish safe or successful robot execution.

## Required experimental evidence

Implementation validation and full-checkpoint experiments must be reported separately; see the [validation record](validation.md) for checks actually run. This guide does not inherit the earlier B/P test counts as validation of the native route. Full-weight training and real transfer results remain pending.

The decisive comparison is H1 versus H0 on held-out demonstration views and robot tasks, with R0 checking whether adding human updates changes control performance. Fix the robot scene and vary demonstration intent to test whether behavior changes correctly. Include examples with similar early motions but different later goals; also measure the effect of removing or swapping demonstrations. A lower human-video loss alone cannot show better robot ICL, and IFP cannot compensate for a dataset in which B's recent motion already reveals every answer.
