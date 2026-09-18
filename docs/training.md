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
`history`, sampler `steps`/`shift`, `frame_id` and `grid_id`. It cannot contain
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

Each view's supervised objective is averaged arithmetically; adding a duplicate
view does not double supervision. CV requires `pair_valid=(view1, view2)`, with
data-side masks for `current.binding`, `remaining.binding`, `relations` and
`events`. Each is intersected with authoritative target `label_valid`, never
with predicted requirement masks or uncertainty.

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

## Checks and limits

Run `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_training.py -v`.
The CPU tests execute small real torch graphs and check stage ownership,
optimizer switching, shared robot inputs, exact teacher/student conditions,
sampler detachment, the direct-goal gradient (and its removal), repeated-view
zero JS, supervision scale, null-branch isolation, empty masks, observation
boundaries and rejection of nonfinite gradients.

Those tests are not native Zero-WAM loading, GPU memory measurements, dataset
training or robot execution. The adapter's independent native smoke test and
subsequent simulator/robot experiments remain required. Checkpoints must record
the stage, model and optimizer state, loss/ablation configuration, upstream
revision, random-number-generator state and completed update count; a small
graph test must not be reported as scientific task success.
