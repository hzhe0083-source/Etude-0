# Full-parameter, goal-supervised video-to-action interface

This experiment adapts Zero-WAM to interpret a compatible demonstration in the **target robot's current scene**. It uses a LIT-style soft latent interface, explicit language, and the target robot's endpoint pose, gripper state, and action labels. It is a modified experiment initialized from pretrained Zero-WAM, not an exact LIT reproduction or evidence of cross-view transfer.

Each query represents exactly the **next action block**, with one group of recurrent interface tokens. The native execution path currently requires batch size 1. This route does not add subgoal segmentation, completion detection, additional action history, or the archived B/P, G/Q, F, and temporal demonstration compressors.

```text
Stage 1 — no visual inputs:
true endpoint pose + gripper -> Goal Encoder -> 100 goal tokens
independent language + current robot state -------------------+
                                                             v
                                            full Action Expert -> action block

Stage 2 — true endpoint is supervision only:
demonstration + observed robot history + language
                      |
             generate future without gradients
                      |
          detach future; recompute context with gradients
                      |
          video layer 1 -> update Z1 -> action layer 1
          video layer 2 -> update Z2 -> action layer 2
                      ...
          video layer L -> update ZL -> action layer L -> action block
                                  |
                        first 8 tokens -> endpoint pose + gripper
```

## Stages, parameters, and losses

| Stage | Action conditions | Updated parameters | Frozen or unused |
| --- | --- | --- | --- |
| `goal` | True goal tokens, independent language features, measured current state | Complete Action Expert, Goal Encoder, state encoder, action-side language and condition projections | Video backbone; unused recurrent visual interface and goal decoder |
| `visual`, `interface_type: "latent"` | The corresponding layer's recurrent latent, independent language features, measured current state | Complete video backbone and Action Expert, recurrent interface, state/condition projections, goal decoder | Stage 1 Goal Encoder is unused |

The external text encoder and visual VAE remain frozen; their cached outputs are inputs. **No LoRA is installed in this route.** Stage 1 loads the pretrained action weights and performs nonvisual adaptation; it does not randomly reinitialize the expert, and input isolation does not prove that previous visual biases have disappeared. Stage 2 continues to update the action expert.

Training keeps model parameters, gradients, and AdamW state in **FP32**, using BF16 autocast on CUDA for forward computation. CPU checks use FP32. Backward executes outside autocast. Defaults are `training.learning_rate: 1e-4` for action/interface parameters and `training.backbone_learning_rate: 1e-5` for Stage 2 video parameters, with constant learning rates and no scheduler. Stage 2 constructs a new optimizer rather than reusing Stage 1 optimizer groups.

Stage 1 minimizes masked native action flow-matching loss. The main Stage 2 objective is:

```text
L_action
  + pose_weight * (L_translation + L_rotation + L_gripper)
  + video_weight * (L_video + configured L_IFP)
```

`L_translation` is MSE after dividing positions by the positive `translation_scale` in meters. `L_rotation` is squared rotation-matrix distance; the decoder constructs valid rotations from a 6D representation. `L_gripper` is MSE in the declared `[0,1]` gripper scale. Logs separate these losses and report mean position distance in meters, angular error in degrees, and gripper absolute error. Default outer weights remain `pose_weight: 1.0` and `video_weight: 1.0`; these losses are not LIT's quantile-normalized vector loss, and its reported `0.3` coefficient is not copied here.

## Recurrent soft interface and visual boundary

The default interface uses **100 tokens of width 768**, 8 attention heads, and 6 groups of shared parameters. Each group serves a contiguous portion of the native layers; the 30-layer model assigns five layers per group. The number of layers must be divisible by the configured number of groups. Small native tests explicitly reduce these settings.

The Goal Encoder maps normalized position, 6D rotation, and gripper values to `[B,100,768]`. At Stage 1, the same goal condition is available to every action layer. Stage 2 begins with learned queries and updates them at **every native coupling layer**, in order: self-attention, language/state cross-attention, then visual cross-attention. Visual content combines the actual patch grid coordinates and relative frame times before aggregation. Every action layer receives its own updated condition.

Only the first 8 final-layer tokens are directly read by the goal decoder. All 100 condition action generation, and all can indirectly receive goal-loss gradients through self-attention. This is a **goal-supervised soft latent interface**: tokens are not explicit SE(3) coordinates, and successful reconstruction does not prove that background information has been removed. Decoded pose/gripper values are diagnostics, not the action expert's input.

The main action path receives only its layer's interface condition, independent language/state features, and this block's noisy actions. `goal_action_forward` supplies `hs_latent=None`, uses a separate action cache, and clears stale action entries. It cannot directly read raw demonstration, observed-video, or generated-future K/V. Action-side cross-attention K/V projections and key normalization are separate from video projections, and the actual forward dispatch uses those copies. The language condition is frozen text-encoder output followed by an action-side projection; the state encoder is independent of video. Neither direct condition is a visually fused hidden state.

The `direct_features` comparison intentionally relaxes this visual boundary, as described below. Its output metadata explicitly reports `raw_video_to_action: true`.

## Three forwards and their gradient boundary

Stage 2 separates these computations, while sharing model parameters:

1. **Future sampling:** use evaluation mode and `torch.no_grad()` to sample from the demonstration, observed history, language, and initial noise. Detach the generated future; do not use `inference_mode`.
2. **Differentiable feature replay:** clear the sampling caches, restore training mode, and recompute demonstration/history conditions and their K/V. Read the detached generated future with autograd enabled and update the per-layer interface. Action and goal losses can update the visual condition path through this replay.
3. **Video/IFP supervision:** execute a separate native video-only training forward with its own teacher-forced/noisy inputs, masks, and cleared caches. True future targets never enter the deployment-style prediction path.

Modes and caches are restored or cleared on exceptional exits too. No-grad sampling disables autocast's weight cache so detached weight casts cannot silently be reused by differentiable replay. The generated sample has no gradient, while recomputed context features retain gradients. **Full-parameter fine-tuning does not mean differentiation through the sampling trajectory.**

## Version-2 supervision and observation contracts

A goal index uses `format_version: 2`, `kind: "se3_goal_index"`, and entries containing exactly `manifest` and `split` (`train`, `validation`, or `test`). Optional source aliases and human/robot bridges retain the [existing split audit](human_icl.md). A single index cannot mix `measured_endpoint` and `controller_target` labels.

Each `se3_goal_sample` manifest uses version 2 and references an NPZ containing exactly:

| Array | Shape before batching | Meaning |
| --- | --- | --- |
| `state` | `[S]` | Finite current measured state in the declared state convention |
| `goal_poses` | `[E,4,4]` | Absolute transforms from robot base to each tool at the block endpoint |
| `goal_gripper` | `[E]` | Endpoint gripper values, normalized with closed = 0 and open = 1 |
| `actions` | `[A,F,N,1]` | Normalized native controls for this one future block |
| `actions_mask` | Same as `actions`, Boolean | Valid labels, intersected with declared valid action channels |

Required metadata includes:

- `sample_id`, local relative `arrays`, and local relative `language` manifest paths.
- `robot_source`: exactly `source_id`, `source_group`, `domain: "robot"`, and `trajectory_id`.
- `state_space_id` and `action_space` (representation, normalization identity, width, valid channels).
- `pose_representation: "absolute_robot_base_tool"`, `coordinate_frame` naming the audited robot base, `pose_units: "m"`, ordered unique `end_effectors`, and one distinct named `tool_frames` entry per end effector.
- `gripper_space`: `normalization_id`, physical `closed` and `open` lists in effector order, and `units`. Bounds must be finite and differ. The normalized target is `(physical_value - closed) / (open - closed)` and must be in `[0,1]`; the loader validates declared targets rather than inventing them from action channels.
- `goal_source`: `measured_endpoint` or `controller_target`, applying consistently to pose and gripper. A close command and a measured gripper opening are different labels. Runs and stage transitions require the same declared convention.
- `current_time`, `goal_time`, and positive `control_dt` in seconds, with `goal_time = current_time + F * N * control_dt`. `F` must equal the configured `chunk_size`; `N` is controls per latent frame.

Optional `provenance` is an audit record, not a model input. Stage 2 also requires `visual_pair`, a local [native ICL sample](human_icl.md) with an audited, operation-compatible human or robot reference and the same target robot execution. A robot reference cannot share the target recording/group/explicit trajectory. A human semantic bridge remains allowed. A shared task name alone does not verify extent, order, or timing.

The visual pair contains observed history plus exactly one future action block. Future action labels, masks, normalization, history endpoint, final endpoint, and every future video time must agree with the goal block: `current_time + (i + 1) * N * control_dt`. Human references need no human pose or action labels. Stage 1 reads goal, action, state, and language arrays only; source-audit metadata can still reference visual records.

Inference instead accepts `se3_goal_observation` version 2. It contains the same state/action, pose/gripper, language, and visual-encoder conventions, plus `current_time`, `control_dt`, and `actions_per_frame`, but **no goal source, endpoint time, or supervision fields**. Its NPZ contains exactly `state [S]`, `history_latent [C,F,H,W]`, and `history_times [F]`; the final history time equals `current_time`. A separate demonstration record references native `latent` and `frame_times` arrays. All paths are local and relative to their containing manifest.

## Frozen instruction cache

Cache each explicitly supplied instruction with the local pretrained model's `text_encoder/` and `tokenizer/` components:

```bash
evo-wam cache-goal-language \
  --text "Operate the drawer as demonstrated." \
  --checkpoint /models/zero-wam --device cuda \
  --output /data/evo/language/drawer
```

The command does not download missing components. It uses frozen UMT5, native `prompt_clean`, a maximum length of 512 with truncation and special tokens, and zero padding. It writes `language.json` and `language.npz`. The separate language format remains version 1; it is not a version-1 goal sample.

The manifest records the original instruction and its hash, cleaned instruction, valid length, array hash, encoder/tokenizer file identities, preprocessing/library versions, encoder precision, and text width. Training/deployment compare that encoder identity independently of the instruction contents, so a new instruction using the same encoder is allowed. Loading validates the full finite FP32 `[1,512,D]` cache and zero padding, then returns only valid tokens `[1,L,D]`. There is no implicit null-text fallback or visual caption generation. Use one consistent cache encoder/precision across a run and its deployment inputs.

## Training, matched comparison, and persistence

[`configs/se3/goal_interface.json`](../configs/se3/goal_interface.json) declares `schema_version: 2` and `interface_type: "latent"`. It still has `synthetic_dimensions_only: true`: `state_dim: 4`, chunk length, sampler steps, and training budget are placeholders. Audit these against collected robot data before formal training. `domain_schedule: ["robot"]` describes supervised targets and does not prohibit human references.

```bash
evo-wam train-goal-interface \
  --config /data/evo/se3-config.json --index /data/evo/goal-index.json \
  --stage goal --checkpoint /models/zero-wam --device cuda \
  --steps 1000 --seed 0 --output outputs/se3-stage1

evo-wam train-goal-interface \
  --config /data/evo/se3-config.json --index /data/evo/visual-goal-index.json \
  --stage visual --checkpoint /models/zero-wam --device cuda \
  --initialize outputs/se3-stage1/goal_interface.pt \
  --steps 1000 --seed 0 --output outputs/se3-stage2
```

For the matched Stage 2 comparison, copy the same configuration and change **only** `interface_type` to `"direct_features"`. Initialize it from the **same Stage 1 artifact**, use the same visual index, seed, budget, language/state inputs, future-sampling settings, and video/IFP objective, and write to a separate output directory. It conditions each action layer on that layer's generated-future features directly, without recurrent latent processing or goal reconstruction loss. This compares the **joint increment of the Stage 2 interface and geometric supervision**; it does not isolate pose loss or establish Stage 1's benefit. Log actual elapsed time, trainable parameter counts, and peak allocated CUDA memory rather than assuming equal cost. The stored Stage 1 artifact hash identifies the common starting point.

Use `--resume` instead of `--initialize` for continuation of the same stage, interface type, index, seed, and cumulative budget. The trainer records deterministic sample order/cursor, successful updates, all random states (including separate action/future/video generators), complete FP32 model/interface state, optimizer state, constant-scheduler declaration, data identities, and hashes of consumed arrays and language caches. Training writes `goal_interface.pt`, `config.json`, `metrics.jsonl`, and `run.json`. Exact continuation depends on the same execution environment and deterministic settings. Stage transitions restore model parameters and construct a fresh optimizer.

A new run/export needs a fresh output directory. `--tiny-native` replaces `--checkpoint` for random small-model checks only; its two native blocks require `num_layer_groups: 1` (the production example has six groups for thirty blocks). It is not validation of trained full weights. Goal v1 data and old LoRA/one-shot interface artifacts are rejected rather than silently migrated.

## Complete deployment export and frozen inference

```bash
evo-wam export-goal-policy \
  --artifact outputs/se3-stage2/goal_interface.pt \
  --output outputs/se3-policy --dtype float32 --max-shard-size 2GB

evo-wam predict-goal-policy \
  --policy outputs/se3-policy \
  --observation /data/evo/current-observation.json --device cuda --seed 0 \
  --output outputs/se3-prediction.npz
```

Export requires successful updates in both stages. It writes complete native/interface **safetensors shards** plus `policy.json`, with construction configuration, dtype, weight map, and shard checksums. `--dtype` selects `float32` (default) or `bfloat16`; this is a deployment choice, not a change to training master precision. Loading reconstructs the separate action K/V modules, strictly restores the full state, validates shard integrity, and freezes all parameters. **No external base checkpoint is needed to load this policy.** Cached language and visual inputs still need their corresponding preprocessing components when new raw inputs are prepared.

Training recovery and deployment export are distinct formats. A deployment bundle has no optimizer/RNG recovery state and cannot replace the full FP32 training artifact. The dedicated `load_goal_policy` and `predict_goal_actions` functions in [`goal_training.py`](../src/evo_wam/goal_training.py) execute the same per-layer conditions as training. An original unmodified Zero-WAM server is not a substitute for this loader.

Prediction outputs `actions` and `generated_future`; the main latent route additionally outputs diagnostic `goal_poses` and `goal_gripper`. The direct-feature baseline has no goal decoder output. All weights stay frozen; changing a demonstration is test-time ICL, not task-specific fine-tuning. Actions are normalized model outputs, not robot commands. The CLI sends zero commands and supplies no closed-loop controller.

## Acceptance evidence and remaining limits

The [validation record](validation.md) separates current checks from earlier versions. Required checks include full parameter updates, layer-specific conditions, the main action branch's independence from substituted visual/stale caches, nonzero gradients in freshly recomputed demonstration/history features, detached sampling, exact resume, and standalone export/loading.

Leakage checks have two different interventions. Deployment tests hold observed inputs and initial noise fixed while changing supervision, and require identical generated futures, predicted goals, and sampled actions. Loss-isolation tests first hold already constructed network inputs—including noisy action/video tensors—fixed, then change loss targets; outputs must remain fixed while losses and gradients may change. Changing clean actions before constructing noisy actions is not such a test.

Small-sample behavior checks must fit different goals/gripper requirements in Stage 1. In Stage 2 they must hold scene, state, **language**, and initial noise fixed while only changing the demonstration's operation requirement, then verify correct goal/action changes after training. Random-initialization differences and attention to latent tokens alone are insufficient evidence.

Real robot data have not yet been collected for this experiment. Full-checkpoint training, distributed execution, actual hardware memory requirements, arbitrary-view transfer, and robot success remain unvalidated. Small native checks do not establish these results. A terminal pose/gripper state may also omit necessary path, contact, force, or timing information; reconstruction cannot prove interaction-only latents. External review, when available, is recorded separately and does not replace these checks.
