# Direct observed-context pose and action training

The input order is **robot history, then complete human/robot demonstration**. Wan computes internal hidden features once from this observed context. It does not sample or reconstruct a future video. A pose decoder predicts the target robot's next action-block endpoint position, orientation, and gripper value. The Action Expert reads both Wan features and the pose decoder's hidden features.

```text
frozen video encoder: observed robot history + demonstration
                         |
                 [history tokens, demo tokens]
                         |
                  clean Wan context pass
                         |
               per-layer hidden features ------------+
                         |                            |
                 pose decoder hidden ----------------+--> Action Expert
                         |                            |       |
                 SE(3) + gripper head      measured state/text  actions
                         |
               robot endpoint supervision
```

## Model and information flow

- `goal_context.observed_context_features` embeds each stream at video timestep zero, concatenates history first and demonstration second, and runs the existing native Transformer blocks directly. There is no video sampler, synthetic future query, or video output head. Features are `[1, history_tokens + demo_tokens, native_dim]` at every layer.
- All supplied frames are already available observations/reference. Both streams can attend to each other; padding is excluded. Their latent-frame time coordinates each start from zero, while demonstration height coordinates use the native `icl_rope_h` namespace. `demo_times` are validated, but do not replace frame indices in native rotary positions. Inputs must retain the trained sampling convention; arbitrary irregular sampling is not implied.
- `ObservedGoalInterface` has one learned readout query per robot end effector. The decoder attends to **all** projected context tokens, plus independent language and state. Its hidden features feed a pose/gripper output head and a separate action projection. The second action branch independently projects/normalizes Wan features.
- Action conditions are ordered as language, state, Wan features, pose-decoder features. Each Action Expert layer receives the corresponding video layer's conditions. The final video layer's pose/gripper prediction supplies the auxiliary labels. Position is in meters, rotation is decoded through the existing valid-rotation construction, and target gripper values are closed=0/open=1.
- Wan does not directly predict robot actions. The Action Expert retains its action flow-matching objective and action sampler. Its loss also updates the visual and pose-feature computations. This task gradient is different from a video reconstruction objective.
- Pose-decoder hidden features may contain information beyond the decoded coordinates. This is not a strict geometric bottleneck, and neither valid rotations nor a low pose loss proves immunity to nuisance appearance.

## Supervision and initialization

The `joint` stage starts from a local pretrained Zero-WAM checkpoint. It updates the existing Wan/action branches and the new interface together, keeping external cached visual/text encoders frozen. There is no required Stage 1 warm-up, no CRHR objective, no online parameter update, and no automatic data-flywheel orchestration.

```text
total = action_flow_loss
      + pose_weight * (translation_loss + rotation_loss + gripper_loss)
```

Only target **robot** recordings provide endpoint and action supervision. A complete human demonstration provides context and requires audited operation correspondence, not human 3D/action labels. Ground-truth endpoints and actions never enter the context/pose-condition function. Action labels are used in the normal training-time noisy-action input and flow target.

The new configuration explicitly requires `video_weight: 0`, `ifp.enabled: false`, and zero IFP weights. Supplying video `sampling_steps` is rejected; `action_sampling_steps` remains required. Shared scheduler fields remain in the configuration for the existing native configuration validator but are not used by the observed-context path. Training never calls `generated_robot_features`, `prepare_icl_inputs`, or `native_icl_loss` for this mode.

## Data and commands

Reuse version-2 goal sample/index and observed-only deployment contracts described in [se3_interface.md](se3_interface.md). The initial implementation reads existing compatible visual-pair manifests. Such files still contain robot future frames under the older data contract; the new forward reads only `target[:, :, :history_frames]`. Future video arrays are neither model inputs nor reconstruction targets. Robot future **action and endpoint labels** remain necessary. Future-video metadata consistency is still validated by the shared loader.

The example configuration is deliberately marked synthetic. Before real training, validate the SO101's 12 state values, action-to-native-channel mapping, normalization, tool/base coordinates, endpoint geometry, video/action cadence, language cache, and data splits. Do not obtain different tasks by assigning arbitrary labels to an unchanged recording. Split by original recording/source groups.

With actual audited configuration, paired data and local weights:

```bash
evo-wam train-goal-interface \
  --config /data/evo/observed-so101.json --index /data/evo/goal-index.json \
  --stage joint --checkpoint /models/zero-wam --device cuda \
  --steps 1000 --seed 0 --output outputs/observed-joint

evo-wam export-goal-policy \
  --artifact outputs/observed-joint/goal_interface.pt \
  --output outputs/observed-policy --dtype bfloat16

evo-wam predict-goal-policy \
  --policy outputs/observed-policy --observation /data/evo/current.json \
  --device cuda --seed 0 --output outputs/observed-prediction.npz
```

The example steps/paths are not a validated training budget. `--resume` restores the same joint run with its data identities, optimizer, random states and cursor. Adding new data is not exact resume and still needs an explicitly defined new training round. `--initialize` remains reserved for the retained legacy goal-to-visual transition.

Export requires at least one successful joint update, not legacy goal-stage updates. Its architecture is `observed_goal_dual_v1`; old `recurrent_goal_full_v2` artifacts keep their original routes and cannot silently substitute for the new one. Prediction returns `actions`, `goal_poses`, `goal_gripper`, without `generated_future`. The prediction CLI does not send robot commands.

## Demonstration dependence and limits

[Zero-WAM](https://arxiv.org/abs/2608.26103) uses demonstration-conditioned learning and an in-context future chunk prediction objective to reduce shortcuts. This route retains demonstration conditioning but deliberately removes that video objective; it cannot inherit a claim that its anti-shortcut effect is unchanged.

Training data must make the demonstration informative. Use audited cases with compatible scene/state and underspecified shared instructions, where different demonstrations require different robot outcomes. Test correct versus mismatched demonstrations with model, observation, state, language and noise controlled. Different random-model outputs or nonzero attention/gradients are only wiring checks. A synthetic fixed-context fit checks that the route can learn different responses; held-out task success is still required to establish real dependence and transfer.

The direct Wan branch can still convey background/appearance shortcuts. Compare the two feature branches and assess held-out appearance, task discrimination and success; no invariance is guaranteed by removing video reconstruction. Real data conversion, complete-model memory/throughput, SO101 closed-loop behavior, and data-flywheel integration remain separate, unmeasured work.
