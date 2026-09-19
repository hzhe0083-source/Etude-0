# Training objectives and runtime contract

`EvoTrainer` in `src/evo_wam/training.py` composes the existing codec, reader,
physical predictor, temporal interaction head and `ZeroWAMAdapter`. It does not
replace the upstream transformer or introduce a second model protocol.

## Stages

| Stage | Trainable parameters | Conditional objectives |
| --- | --- | --- |
| `interface` | G/Q, F, native condition projection, video/action conditional LoRA | Requirement reconstruction, actual-outcome prediction, paired native video/action flow targets, configured IFP |
| `reader` | Demonstration projection and task reader | Latent alignment, decoded current/remaining requirements, execution consistency |
| `joint` | Reader, deployed video conditional LoRA, native IFP fusion/heads, shared temporal interaction head | Reader objectives plus next video, IFP, physical interaction supervision and optional paired JS |

The adapter controls its own native parameter selection. The action expert,
codec and F remain frozen in stage two. A stage switch clears old gradients and
creates a fresh AdamW optimizer; it does not carry stage-one moments into a new
optimization problem. Resume the saved optimizer only for the same stage.

`exec_start_step` is the number of **successful optimizer updates in the current
stage** before execution distillation is enabled (default 100). Before that
boundary, the reader still receives requirement/latent supervision, but the
trainer does not invoke the video sampler, execution teacher or execution
student. Merely setting a loss coefficient to zero after performing sampling
would not implement this warmup. `weights.execution=0` is a separate permanent
ablation: it never calls any of those three paths, even beyond the boundary.

Null examples with no trainable path and failed finite checks do not advance
the counter. A stage switch resets `updates` to zero. A same-stage resume must
restore the checkpoint's successful-update counter after construction, together
with optimizer and RNG state; it must not recompute the phase from attempted
batch count. `execution_enabled` exposes the schedule decision for diagnostics.

`LossWeights` names each objective explicitly. T0/T1/T2 use `enable_ifp` and
`enable_interaction`; V0/V1 differ only in `weights.cv`. A zero coefficient does
not imply a different trainable parameter set. Native action label regression
is included **only** in `interface`, with its true paired video/action data.

Each optimizer step checks finite losses, backpropagates, rejects nonfinite
gradient norms, clips the global norm and then steps. `updates` counts actual
updates; the returned dictionary includes every active objective, CV coverage,
gradient norm and `updated`. There is no silent optimizer update after a
failed finite check.

## Batch inputs

The real adapter currently supports one packed robot trajectory (`B=1`).
Accumulate optimizer steps externally if a larger effective batch is required;
do not stack independent histories into an unsupported native batch dimension.

`TrainingBatch` contains:

- Pure, detached `entity_features [1,N,De]`, `entity_history [1,L,N,De]`,
  `proprio_history [1,L,Dp]` and `embodiment [1,Db]`. These cannot be WAM hidden
  states containing task conditions.
- `null_text`: the native empty-text embedding, not a task instruction.
- A single `native_inputs` dictionary shared by both demonstration views. Its
  `latent_dict`, `action_dict` and each `mcp_latent_dicts` entry contain the
  upstream inputs, detached flow `targets [1,C,F,H,W]`, optional data-side
  binary `valid_mask`/`actions_mask`, and optional scheduler
  `training_weight [1,F]`. Weights multiply squared errors before normalization
  by the valid count; absent weights are exactly one. Compute these with the
  pinned upstream schedulers when loss reweighting is enabled.
- `requirements: TaskRequirement` and one or two demonstration token tensors.
  A native pair reuses the robot dictionary, including noisy latent and time
  for every target. Each IFP target retains its independently sampled noise.
- `outcome: PhysicalOutcome` and actually executed `physical_actions [1,H,Da]`
  for stage-one F supervision. The predictor produces stepwise outcomes; its
  loss selects the data's one-based offsets and rejects labels past H.
- `entity_patch_weights [1,N,S]`: fixed observation-derived weights mapping
  noisy robot video tokens to scene entities. Rows for visible entities must
  have support; padding rows may be zero. The trainer normalizes each row and
  pools `Phi [1,S,D]` before the interaction head. These weights must never be
  generated from task bindings, labels or the demonstration.

The trainer rejects raw `icl_latent_dict`, caller-supplied task text embeddings
and unexpected sampling arguments. Tensor provenance still belongs to the data
loader: a runtime shape check cannot certify that an observation feature or
pooling map was not constructed from a future label.

## Two separate gradient graphs

The training forward passes the shared noisy robot input through the deployed
WAM. Next-video, IFP and interaction losses update its video LoRA through Phi.
The auxiliary heads do not receive demonstration tokens or goal tokens directly.

Execution consistency uses `sample_noise`, `noisy_actions`, `action_timestep`
and optional Boolean `execution_valid`. `sample_kwargs` permits only actual
`history`, sampler `steps`/`shift`, `frame_id`, `grid_id` and `rope_offset`. It cannot contain
training dictionaries or teacher-forced caches.

For each view:

1. Inside `torch.no_grad()`, sample a future from fresh noise using predicted
   current and remaining requirements.
2. Inside the same no-gradient scope, evaluate the frozen action expert with
   the true current requirement and that generated future.
3. Evaluate the student with the predicted current requirement, the **same
   future object**, the **same noisy actions**, and the **same action time**.
   The frozen expert's forward retains autograd to the predicted current goal.

The video sampler has no gradient. The objective does not force a generated
future to fit the recorded action target. The action expert's cached future
context is fixed to the sampled future's conditions for both teacher and
student. This is execution consistency, not a claim that the teacher supplies
new physical truth or that the complete sampler is trained end to end.

Module dropout is disabled during an objective and its prior state is restored
afterward. Upstream functional attention dropout must remain configured as
zero. Deterministic preprocessing and sampling once per robot target are data
loader responsibilities.

## Pair loss and null examples

One demonstration view is sufficient for robot bridge training. With one view,
the reader, native training branch and any enabled execution-distillation branch
run once. A positive global CV coefficient does not require a second view and
does not cause duplication: the reported CV loss and coverage are both zero.

`pair_kind` defaults to `"none"`. Only an explicitly verified
`"synchronized_views"` pair with two actual views may receive CV. Merely storing
two views does not establish reliable correspondence; legacy two-view records
without this annotation do not silently activate CV. Claiming synchronization
with only one view is a metadata error. Null examples never run CV.

Each available view's supervised objective is averaged arithmetically; adding a
second view does not double supervision. `per_view_valid=(view1, view2)` retains each
view's evidence separately. It must explicitly include `current.<field>` and
`remaining.<field>` for every requirement `label_valid` field, plus `relations`
and `events` for physical interaction labels. Masks are Boolean, data-owned and
have the corresponding label-validity shape. Single-view training uses a
one-element tuple. Missing fields are rejected rather than assumed observable.
Requirement fields include binding, geometry, relations, events, geometry
tolerance, event windows and precedence slots. Window evidence uses the event
label shape, while precedence evidence is `[B,P]`; a padding slot known to mean
"no edge" is valid, while an unknown slot is not proof that order is irrelevant.

The decoded requirement loss receives each view's own evidence. An unidentified
binding is supervised as UNCERTAIN, not as the globally known object; other
requirement values and requirement-mask targets are supervised only where the
view provides evidence. Interaction-label supervision also intersects that
view's evidence with `label_valid`. This policy does not use privileged hidden
identities to teach a deterministic answer to an unidentifiable demonstration.

Latent alignment and execution consistency require a uniquely identifiable
current **and** remaining condition. The gate uses annotated evidence, resolved
required bindings and reliable required labels, never model confidence. If the
gate fails for one view, both losses are zero for that view and its sampler,
teacher and execution student are not called. Its available decoded labels,
including the UNCERTAIN binding category, still train the reader. A missing
remaining destination blocks distillation even when the immediate grasp is
visible. Geometry controls inspect only their own explicit geometry interface
content (including tolerance) for this gate, not hidden control-relation labels.
Full interfaces also require identifiable event windows and precedence. When no
view qualifies, the trainer never constructs the privileged G target at all.

CV takes the two evidence masks' intersection only when computing the pair
loss. It never substitutes that common mask for either view's individual
supervision. Predicted requirement masks and uncertainty cannot remove labels.

The common CV field set consists of current/remaining **binding** logits and
the auxiliary physical interaction head's relation/event logits. Interface
relation and event decoders are deliberately excluded from CV so full and
geometry interfaces receive the same auxiliary objective. Relations and events
use independent Bernoulli JS; binding uses categorical JS in the same scene
entity order. Optional `category_groups` maps every category into an annotated
equivalence class, retaining illegal-result mass. Continuous geometry stays in
its supervised loss.

Each field is normalized within a sample; available field means are aggregated
with fixed weights, then averaged across valid pairs. `cv_coverage` is the
fraction of pairs with any common valid field. Empty common masks yield zero
CV and preserve supervised losses. Both view branches receive JS gradients.

A batch with `conditional=False` constructs only null task slots and computes
native next-video plus enabled IFP losses. It never encodes true requirements,
calls the reader or interaction head, predicts F, performs action regression,
or invokes the execution teacher—even if labels remain in the dataset object.
During reader-only warmup the generator is frozen, so these null examples are
measured with `updated=False`. During interface/joint training they can update
the allowed native video parameters. The loader owns the common 90/10 sampling
policy; legacy independent `drop_icl` and text-drop entry points must be off.

Unpaired network-video representation learning is a separate upstream objective,
not a reason to fabricate robot actions or task requirements for those videos.
When using the offline effect-encoder route, the demonstrations provided here
are frozen, ordered observed-effect tokens `Z_D`; the reader still produces
robot-state-dependent `g_current` and `g_remaining`. Encoder/representation
identity and feature dimensions must match preprocessing and the checkpoint.
This trainer does not register or update an offline demonstration encoder, and
its pretraining alone is not evidence that deployed WAM parameters improved.

## Checks and limits

Run `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_training.py -v`.
The CPU tests execute small real torch graphs and check stage ownership,
optimizer switching, shared robot inputs, exact teacher/student conditions,
sampler detachment, the direct-goal gradient (and its removal), repeated-view
zero JS, supervision scale, null-branch isolation, empty masks, observation
boundaries and rejection of nonfinite gradients. Additional checks verify zero
sampler/teacher/student calls throughout warmup and the permanent execution-off
ablation; per-view occlusion suppresses latent/distillation targets; changing a
hidden privileged object identity changes neither the uncertain view's loss nor
its gradient; and geometry controls do not depend on relation-evidence gates.
Single-view tests also check one forward per branch and zero CV under a nonzero
CV setting, while unverified two-view data is distinguished from an explicitly
trusted pair without changing the supervised-loss scale.

Those tests are not native Zero-WAM loading, GPU memory measurements, dataset
training or robot execution. The adapter's independent native smoke test and
subsequent simulator/robot experiments remain required. Checkpoints must record
the stage, model and optimizer state, loss/ablation configuration, upstream
revision, random-number-generator state and completed update count; a small
graph test must not be reported as scientific task success.
