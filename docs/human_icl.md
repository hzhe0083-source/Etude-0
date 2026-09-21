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

The input is the frozen Wan VAE latent in its native `[C,F,H,W]` layout, normalized once as `(posterior_mode-mean)/std`. In H1, A enters directly through `icl_latent_dict`. In H2/H3, a shared temporal bottleneck compresses A after the frozen native patch embedding and adapts its output to the native context width (see below). Trainable LoRA covers video self-attention Q/K/V/output and text cross-attention Q/output in the native blocks. The base model, text K/V, and action expert are frozen. Robot action loss still provides gradients to the video adapters through the existing computation graph; frozen parameters do not imply a detached forward pass.

Human forwarding reuses the native embedding, blocks, position grids, attention masks, output head, and optional MCP/IFP helper. It omits `action_dict` and passes no action hidden states. There are no fabricated zero actions, action masks used as pseudo labels, robot proprioception, geometry/relations/events labels, or human action loss. Robot forwarding remains the native `forward_train` path; when enabled, the same demonstration interface prepares its A context before that call.

Training uses native **chunkwise teacher forcing**. A noisy B query may read earlier clean B chunks and the complete A context. It cannot read clean B from the same or a later chunk. Thus a late training query can use more recorded history than an early query; the whole rollout does not share one fixed clean prefix. `history_frames` is measured in latent frames and excludes the initial observed prefix from the human target loss. It must agree with the configured native chunk boundaries. Robot samples retain upstream whole-window supervision: video loss uses the weighted full-tensor mean, action loss uses the full-tensor mean after masking, and MCP uses its valid-target count. Human history exclusion does not silently rescale the robot objective.

No target task description is added as an input. Semantic matching evidence remains metadata for audit. The ordinary-human-video control omits A entirely from the model input while predicting the same B targets.

## Data contract

Each `native_icl_sample` JSON has `format_version: 1`, `sample_id`, `feature_space_id`, `latent_normalization`, `history_frames`, `demonstration`, `target`, and `compatibility`. Optional `provenance` is audit metadata only. Optional `appearance_variant` supplies the audited same-demonstration variant used by H3, as specified below.

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

## Temporal demonstration bottleneck: H2 and H3

H1 remains the raw-context baseline. `H2_temporal_bottleneck.json` adds only a `demo_bottleneck` configuration; `H3_appearance_consistency.json` differs from H2 only in `consistency_weight` (`0.0` versus `0.01`). Data, predictive losses, LoRA settings, seeds, and update schedules stay matched. Use the same index, including variant records, for the H2/H3 comparison; H2 performs no additional augmented forward pass.

```text
A latent -> frozen native patch embedding -> content/time/xy MLP
         -> grouped learned queries + cross-attention + feed-forward
         -> group-time encoding + small Transformer -> adapter -> WAM context
B / robot observations ------------------- native observation path -> WAM
```

The compressor binds content to actual latent-frame timestamps and the patch grid's normalized xy centers before pooling. Each consecutive group of `group_frames` latent frames supplies keys and values for `tokens_per_group` learned queries; padding in the final short group is masked. A small Transformer connects the ordered group tokens with their group-time encodings. All output tokens are retained, with no whole-video average. Query slots are learned vectors, not fixed action categories or annotated hand/object roles.

For `T` latent frames, the context contains `S = ceil(T / group_frames) * tokens_per_group` tokens. H2/H3 use candidate width 768, 8 heads, groups of 4 latent frames, 4 tokens per group, and 2 Transformer layers. These are illustrative starting capacities, not validated real-data settings. For example, 32 latent frames yield 32 context tokens; doubling the clip length doubles this context length. Record actual input/output token counts, runtime, and memory when comparing configurations.

Only reference A is compressed. B's history, the robot's current observations, and native action supervision remain on their original paths. Human training, robot training, and inference share one compressor and adapter. Enabled runs replace A with a typed prepared context, rebuild its position/cache metadata for the actual `S`, and use native attention masking; the compressed tokens are not passed off as a video patch grid. The enabled path supplies no parallel raw-A K/V context. Native context coordinates identify time group, demonstration namespace, and query slot.

The prediction objective sends gradients through WAM and the adapter into the compressor. H3 adds mean squared distance between L2-normalized original/variant tokens **before the WAM adapter**. The original demonstration still supplies the prediction loss. This additional constraint encourages stability under reviewed appearance changes; it does not prove background removal, pure interaction semantics, or arbitrary-view transfer.

To enable consistency, each conditioned human and robot sample must include:

```json
"appearance_variant": {
  "arrays": "demo-appearance.npz",
  "derived_from": "same-source-id-as-demonstration",
  "evidence": "Reviewed appearance-only transform; interaction and frame timing preserved."
}
```

The variant NPZ contains only `latent` and `frame_times`, matches A's `[C,F,H,W]` shape and exact timestamps, and uses a different file from A and B. `derived_from` must equal `demonstration.source_id`. Produce the variant by an audited appearance-only change to the source clip with the same frozen visual encoding and sampling convention. Review that task-relevant colors, objects, contact evidence, motion, and timing remain valid. Neither arbitrary latent noise nor a same-task but independent recording constitutes this appearance variant. The loader verifies structural provenance and alignment, not the truth of the semantic audit; that remains a data preparation responsibility. H3 rejects conditioned samples without the required variant.

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

For H1, export merges the learned LoRA adapters into the original model's linear layers and writes native checkpoint files:

```bash
evo-wam export-native-icl \
  --artifact /server/runs/H1/native_icl.pt \
  --checkpoint /server/zero-wam \
  --device cpu --output /server/models/H1
```

Use the H1 merged model through the original Zero-WAM demonstration/robot inference interface. No earlier B/P effect encoder, effect reader, requirement decoder, ranking network, or test-time parameter update is needed. Training artifacts and small-model exports do not themselves establish safe or successful robot execution.

With H2/H3 enabled, training and resume explicitly preserve the compressor, its adapter, and LoRA parameters together with optimizer/RNG state. `export-native-icl` instead writes a dedicated bundle with root `bottleneck.json`, `demo_bottleneck.pt`, and nested merged backbone weights. Loading the bundle root as a stock Zero-WAM checkpoint is intentionally unsupported: dropping the interface would change the trained model.

`evo_wam.icl_deployment.load_bottleneck_deployment(path)` restores the native model and null text embedding. `evo_wam.demo_context.cache_demo_context` prepares a demonstration through the restored compressor. Initialize the demo in an empty native cache before adding robot observations, as the original server reset does; replacing a context without resetting is rejected so previous tasks or observations cannot contaminate its encoding. For the original Zero-WAM server, call `evo_wam.icl_deployment.attach_bottleneck_server(server, bundle)` before the first reset or inference. Construct that server with its transformer path resolving to the bundle's `backbone` directory (for example, using a model-resource symlink); retain the original VAE, tokenizer, and text encoder resources. The helper verifies the loaded transformer's origin, attaches the interface without loading a second backbone, and retains the original video/action sampler. The server must use ICL, cache name `pos`, and the trained `icl_rope_h`.

The attached server's demonstration input is an audited `preprocess-icl-video` cache (`clip.npz` with adjacent `clip.json`), with matching feature-space identity and explicit frame timestamps. It does not fall back to the old raw-video or `.pth` loader. All deployment parameters are frozen; no test-time training occurs. The appearance variant is needed for consistency training, not deployment.

## Required experimental evidence

Implementation validation and full-checkpoint experiments must be reported separately; see the [validation record](validation.md) for checks actually run. This guide does not inherit the earlier B/P test counts as validation of the native route. Full-weight training and real transfer results remain pending. The validation record includes temporal-interface, save/restore, and native-cache checks; these are numerical checks, not real-data transfer results.

The decisive comparison is H1 versus H0 on held-out demonstration views and robot tasks, with R0 checking whether adding human updates changes control performance. Fix the robot scene and vary demonstration intent to test whether behavior changes correctly. Include examples with similar early motions but different later goals; also measure the effect of removing or swapping demonstrations. A lower human-video loss alone cannot show better robot ICL, and IFP cannot compensate for a dataset in which B's recent motion already reveals every answer.

For the bottleneck experiment, compare H2 against H1 to isolate compression, then H3 against H2 to isolate consistency. Check order-sensitive examples (including equal endpoints with different middle steps), held-out appearance and views, and fixed-scene demonstration swaps. Require both preserved interaction distinctions and reduced dependence on irrelevant appearance before describing the representation as an interaction bottleneck. An unchanged prediction after a wrong demonstration is a warning that the model may ignore its context.
