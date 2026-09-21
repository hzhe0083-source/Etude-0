# SE(3)-supervised video-to-action interface

This experiment trains a continuous interface between Zero-WAM's predicted robot future and its Action Expert. A compatible human or robot demonstration is interpreted in the **target robot's scene**. Supervision comes from that robot's next action block and its endpoint pose; human videos need no pose or action labels.

The interface tokens `z` are learned features. They are neither SE(3) coordinates nor a complete action block. A pose decoder provides auxiliary supervision and diagnostic outputs; the Action Expert receives `z` and the observed robot state, never the decoded pose.

```text
Stage 1: robot endpoint pose -> Goal Encoder -> goal tokens
                                                    + observed robot state
                                                    -> adapter -> Action Expert -> action block

Stage 2 and deployment:
reference video + observed robot history -> Zero-WAM -> generated robot future features
                                                    + observed robot state
                                                    -> visual goal readout -> z
                                                        |                  |
                                                        v                  v
                                                   Pose Decoder      adapter + state
                                                        |                  |
                                                   endpoint pose     Action Expert
                                                                           |
                                                                      action block
```

This is a modified two-stage experiment initialized from Zero-WAM, not a reproduction of LIT or evidence of cross-view transfer. The independent [H1/H2/H3 ICL experiments](human_icl.md), including temporal demonstration compression, remain available as comparisons. The SE(3) route does not stack that compressor, B/P, G/Q, or F ranking onto this interface.

## Two stages and trainable parameters

| Stage | Inputs to the goal/action path | Trainable | Frozen |
| --- | --- | --- | --- |
| `goal` | Robot endpoint pose, observed state, noisy action block | Goal Encoder, state encoder, condition adapter, Action Expert attention LoRA | Base checkpoint and video LoRA; unused visual readout and pose decoder |
| `visual` | Reference video, observed robot history/state, noisy action block | Video/context attention LoRA, visual goal readout, Pose Decoder | Stage 1 Goal Encoder, state encoder, condition adapter, Action Expert including its learned LoRA, base checkpoint |

Stage 1 loads Zero-WAM action weights and adapts selected action-attention projections. It does not train an Action Expert from scratch. Video arrays are not read for this stage; if a visual pair is listed, its metadata can still be inspected for source/split auditing.

Stage 2 must initialize from a successfully updated Stage 1 artifact, or resume its own Stage 2 artifact. Both stages must use the same interface dimensions, base checkpoint, robot state/action conventions, pose frame and source type. Freezing Stage 1 parameters preserves the learned goal-to-action interface while action-loss gradients still reach the visual readout through the frozen Action Expert.

Stage 1 minimizes masked action flow-matching loss. Stage 2 minimizes:

```text
action loss
  + pose_weight * (normalized translation MSE + squared rotation-matrix distance)
  + video_weight * (native video prediction loss + configured IFP loss)
```

Translation is divided by the positive `translation_scale`, expressed in meters. Rotation uses a squared chordal distance on rotation matrices, not Euler-angle MSE. The decoder constructs valid rotations from a continuous 6D representation. `pose_weight` is a reconstruction weight; it is not a token-energy, KL, or information-rate penalty.

The default interface has 64 tokens of width 768, plus a separate encoded state token before projection into the native action cross-attention width. The visual readout combines collected per-layer future features with patch coordinates and elapsed time before attention pooling, using state-conditioned queries. These dimensions are experimental choices, not measured optimal capacities.

## Preventing a true-future or raw-video shortcut

The goal/action path generates the next robot video block from the reference and observed history only. Native denoising runs without gradients; the resulting latent is detached. A final clean read of this **generated** block, with the observed context replayed, retains gradients during Stage 2. Features from the checkpoint's MCP collection layers are concatenated for the visual goal readout. This trains the feature computation; gradients do not backpropagate through the sampling trajectory.

The true robot future is used only by the separate native video/IFP supervision branch. Its teacher-forced inputs are not passed into the goal readout or action conditioning. That branch calls the video-only objective and does not invoke the original raw-video-conditioned action branch.

`goal_action_forward` supplies no video hidden tokens (`hs_latent=None`). It uses an isolated action cache and conditions action cross-attention only on the interface and current-state tokens. It neither reads raw video keys/values nor falls back to Zero-WAM's original action route. The same action function is used in both training stages and deployment.

## Robot supervision data

An index uses `format_version: 1`, `kind: "se3_goal_index"`, and `samples` with exactly `manifest` and `split` per entry. Splits are `train`, `validation`, and `test`. Optional `source_aliases` and `bridge_sources` use the existing [source-component audit](human_icl.md); robot trajectories and reference sources connected by aliases or correspondences cannot cross splits.

Each `se3_goal_sample` JSON points to one NPZ containing exactly:

| Array | Shape before batching | Meaning |
| --- | --- | --- |
| `state` | `[S]` | Finite, measured current robot state in the declared state convention |
| `goal_poses` | `[E,4,4]` | One valid homogeneous SE(3) transform per ordered end effector, at the end of the next action block |
| `actions` | `[A,F,N,1]` | Normalized native controls for that one future block |
| `actions_mask` | Same as `actions`, Boolean | Valid action labels, intersected with the declared valid channels |

`E` is the number of end effectors, `A` the action width, `F` the configured `chunk_size`, and `N` the number of controls per latent video frame. State width and action width are separate quantities. Neither is inferred from a human video.

Required metadata is:

- `format_version: 1`, `kind: "se3_goal_sample"`, `sample_id`, and local relative `arrays` path.
- `robot_source`: `source_id`, `source_group`, `domain: "robot"`, and `trajectory_id`.
- `state_space_id` identifying the state representation and normalization; `action_space` declaring representation, normalization identity, dimension, and valid channels.
- `coordinate_frame` naming the target robot's pose frame (use the audited robot base frame), `pose_units: "m"`, and ordered unique `end_effectors`.
- `goal_source`: explicitly either `measured_endpoint` or `controller_target`. A measured achieved endpoint and a commanded target are different labels and cannot be silently mixed in one run.
- `current_time`, `goal_time`, and positive `control_dt`, all in seconds. The duration must equal `F * N * control_dt`.

Optional `provenance` records label derivation and audits; it is not a model input. Supply measured endpoint transforms or documented controller targets directly. Normalized action channels alone do not establish an endpoint SE(3), and the loader does not fabricate poses from them.

Stage 2 also requires a local `visual_pair` path to a [native ICL sample](human_icl.md). Its demonstration may be human or robot, but its target must be the same robot execution as `robot_source`. A robot reference must not be the same source, recording group, or explicitly identified trajectory as that target; otherwise it could carry the target future through the reference input. This does not reject a human reference merely because it has a semantic HumanGen/trajectory bridge. The pair must carry audited operation compatibility; sharing a task name does not establish matching movement extent, contact sequence, or timing. Its target future contains exactly one action block after the observed history. Future actions, masks, action normalization, history endpoint time and final time must agree with the goal sample. Every future video endpoint must also follow `current_time + (i + 1) * N * control_dt`, matching the generated feature schedule; observed history cadence is not otherwise constrained. Supervision always uses the **target robot's** poses and actions, never the demonstrator's coordinates.

This route therefore needs compatible reference-to-robot pairs. It does not train SE(3) from unpaired human-only target videos; the separate H1 route can still use human-to-human cross-video prediction.

## Training and export

[`configs/se3/goal_interface.json`](../configs/se3/goal_interface.json) is marked `synthetic_dimensions_only: true`. In particular, `state_dim: 4` is a placeholder, and its chunk length, loss weights, 4-step samplers and budget are not audited production settings. Copy and configure it against collected robot data and the chosen checkpoint before formal experiments. Keeping `domain_schedule: ["robot"]` means all supervised targets are robots; it does not prohibit human reference videos. `demo_bottleneck` is rejected by this route.

With audited data and a matching native checkpoint:

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

evo-wam export-goal-policy \
  --artifact outputs/se3-stage2/goal_interface.pt --output outputs/se3-policy
```

Use `--resume` with the stage's `goal_interface.pt` instead of `--initialize` to continue that same stage, index, seed and cumulative configured budget. A new run or export requires a fresh output directory. `--tiny-native` replaces `--checkpoint` only for randomly initialized small native checks; it is not pretrained-model validation.

Training writes `goal_interface.pt`, the resolved `config.json`, stepwise `metrics.jsonl`, and `run.json`. Export requires successful updates in both stages. It produces `policy.pt` and a checksum manifest `policy.json`, preserving the complete interface and both action/video LoRA states. The original immutable Zero-WAM checkpoint remains separately required, except for tiny-native artifacts, which carry their tiny base weights. This is not an original-format merged Zero-WAM checkpoint.

## Frozen inference

The public loading/prediction entry points are `load_goal_policy` and `predict_goal_actions` in [`goal_training.py`](../src/evo_wam/goal_training.py). The CLI uses those same functions:

```bash
evo-wam predict-goal-policy \
  --policy outputs/se3-policy --checkpoint /models/zero-wam \
  --observation /data/evo/current-observation.json --device cuda --seed 0 \
  --output outputs/se3-prediction.npz
```

Inference accepts only a `GoalObservation`, not a supervised sample. Its `se3_goal_observation` JSON supplies the trained frame, units, effectors, state/action conventions, `current_time`, `control_dt`, `actions_per_frame`, visual `feature_space_id`, and native `latent_normalization`. Its observation NPZ contains exactly `state [S]`, `history_latent [C,F,H,W]`, and strictly increasing `history_times [F]`. A separate `demonstration` source record points to a native video NPZ containing `latent` and `frame_times`. The latest observed time must equal `current_time`; future video, goal poses, and action labels have no input fields.

All parameters are frozen during inference. Output NPZ fields are `actions`, diagnostic `goal_poses`, and `generated_future`. Actions retain the declared normalization and are not robot commands: this CLI sends zero commands and supplies no closed-loop robot controller. Use the dedicated policy loader; running an exported interface through the original raw Zero-WAM server would omit the required action-conditioning change.

## What still needs evidence

Small checks can verify stage freezes, valid SE(3), gradient paths, leakage rejection, and persistence. They do not establish learned target transfer, interaction-only representations, novelty, or robot execution success. Formal training and real cross-view transfer remain unmeasured.

The first behavioral checks should hold robot observations/state and the random seed fixed while changing an operation-compatible reference's movement extent or necessary intermediate steps, and then measure both endpoint error and action/execution behavior. Compare generated-future performance with the raw ICL baseline; do not substitute a result that reads the true future for a deployment result.

One endpoint and the current state may not uniquely specify the path, gripper timing, contact sequence, or success. Action supervision remains necessary, and Stage 1's nonvisual conditions may themselves be insufficient for ambiguous blocks. Pose reconstruction also cannot prevent unused token dimensions from retaining background appearance. These are experimental limitations to test, not guarantees supplied by the interface.
