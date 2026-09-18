# Native Zero-WAM integration

`evo_wam.zerowam` calls the upstream transformer at commit
`08e2c4ae41e2b63573a299825cebe6753481407c`. Loading verifies both HEAD and
tracked-file cleanliness. It does not download model weights. Use
`ZeroWAMAdapter.from_checkpoint(local_path, condition_dim=...)` with a local
transformer directory or a checkpoint root containing `transformer/`. For a
random tiny model, `from_config` accepts a native JSON path or configuration dict.

## Conditions and trainable parameters

`TaskConditions(current, remaining, null)` holds `[1,N,D]` codec tokens and a
precomputed native empty-text embedding `[1,N_null,text_dim]`. Current and remaining
token counts are fixed by the adapter constructor. The adapter projects codec
tokens to the native text dimension. It supplies the slots through the actual
native text-conditioning path and intersects the native cross-attention masks:

- Main video queries: null, current and remaining.
- Action queries: null and current.
- MCP queries: null only; task information reaches MCP through backbone Phi.

Two learned type embeddings mark current versus remaining after projection;
otherwise the main cross-attention would see an unordered combined token set.
Type embeddings train with the interface projection in stage one and remain
frozen in reader/joint stages. Within-group event order remains in codec tokens.

The original self-attention/flow/MCP masks and inputs are retained. Raw ICL tokens
are rejected, so they cannot bypass this task interface. Both task groups become
inaccessible together for `conditions.unconditional()`; original task text is
replaced. A null embedding must be provided explicitly; do not substitute a task
text embedding. Inference receives task semantics only through these slots.

One context correction is necessary: native MCP receives the main action hidden
states as well as Phi. Those states would now carry our direct current condition.
A first-block hook therefore supplies the original task-free action embeddings
instead, with the native action timing and attention masks unchanged. This removes
the new task bypass without adding a network; MCP still receives task information
through the main fused video representation.

`set_stage('interface')` trains the condition projection and video/action-specific
LoRA. `reader` freezes the adapter while retaining input gradients. `joint` opens
only video-specific LoRA and existing MCP modules. LoRA is attached to video
`attn2.to_q/to_out.0` and action `attn2.action_to_q/action_to_out.0`. Shared K/V and
the native text projector stay frozen. New LoRA starts at the original model's
output. These newly initialized task adapters require interface training before
checkpoint-level robot performance can be claimed.
Constructor parameters `lora_rank`, `lora_alpha` use the standard alpha/rank scale;
omitted alpha defaults to rank. `lora_dropout` must be zero, as must native module
dropout, for the paired-view path. All three parameters pass through `from_config`
and `from_checkpoint` rather than silently ignoring configuration values.

## Training and generated-future paths

`forward_train(native_input_dict, conditions)` preserves native inputs including
`latent_dict`, `action_dict`, `mcp_latent_dicts`, their timesteps, grids and clean
teacher-forcing streams. It returns native flattened video/action/MCP velocities
and current noisy-video `phi` for the shared time-query head. The wrapper does not
sample new noise or compute losses: paired-view noise, schedules, targets and MCP
tail masks remain the caller's responsibility. Phi uses native multi-layer MCP
fusion where enabled; clean-prefix and raw demonstration tokens are excluded.

`sample_video(initial_noise, conditions, history=..., steps=4, shift=5)` starts
only from the supplied noise and invokes the native flow scheduler and video
forward under `no_grad`. It does not accept the supervised training dictionary.
History is a chronological list of `(video_or_action, actual_observed_tensor,
frame_id)`; use interleaved video/action frame IDs, for example 0/1 followed by
future video/action 2/3. History chunks are real past observations/actions, never
the recorded future. Native video tensors are `[1,C,F,H,W]`; actions are
`[1,A,F,N,1]`. Explicit video grids can be supplied for non-default layouts.
Stream tensors and explicit grids are moved to the native model's device, and
stream values use its parameter dtype; sampling returns native-device tensors.

The returned `GeneratedFuture` holds detached generated latents, their grid/frame
and the fixed task conditions used to generate/encode them. Pass this same object
to both calls of `action_velocity(noisy_action, timestep, conditions, future)`
for execution distillation. Both calls replay the same future context; only their
direct current requirement differs. Teacher execution belongs inside `no_grad`;
the student's action forward must remain tracked. `sample_actions` performs the
corresponding native action sampling (default four steps, shift 1). Sampling steps
and shifts are explicit pilot defaults, not claims of matching released rollout
quality. `unpack_velocity` reverses native patch-token ordering before scheduler
updates, including non-unit video patches.

All independent calls clear and reconstruct native KV history. `update_cache=0`
alone does not disable reads, and upstream metadata is not independent per cache
name. Calls on one adapter must be sequential. Current implementation replays
history for each sampled action step to prioritize cache isolation over latency.
The returned Phi graph never supplies sampling/action caches or the physical F.

## Verification and dependencies

Run `python -m unittest discover -s tests -p test_zerowam.py -v`. CPU tests verify
the pinned source, condition routing, LoRA and native output reshaping. The native
test invokes actual CUDA FlexAttention, native video/action/MCP forwards, sampling
and backward passes; it checks video LoRA and direct condition gradients. It skips
with a specific reason when CUDA/native dependencies are absent, and a skipped
native test is not a successful native integration result. No fake attention is
used. The smoke function can also be called as `tiny_native_smoke()`.
The native smoke additionally holds full fused Phi fixed while changing task
tokens: MCP outputs must remain identical, auxiliary action inputs must remain
identical, and the bypass gradient to task tokens must be zero.

Upstream pins torch 2.9.0, diffusers 0.36.0 and transformers 4.55.2; matching CUDA
torchvision, einops, easydict, imageio and websockets/msgpack are also needed by
upstream imports. Robot data, released
checkpoints and simulator execution remain separate from tiny random-model QA.

The pinned legacy `model.py` imports flash-attn even though `icl_model.py` uses
real PyTorch FlexAttention exclusively. A scoped source loader changes exactly
that import block to an optional ImportError guard, only for the verified absolute
`wan_va/modules/model.py` path. The finder is removed in `finally`; no upstream
source or bytecode is changed. If real flash-attn is unavailable, calling the
legacy `flash_attn_func` raises explicitly: it never executes approximate or
replacement attention. No fake `flash_attn` module is inserted into `sys.modules`.
All model forward and ICL attention code remains upstream code. The guard test
forces missing FlashAttention and checks this failure behavior and module hygiene;
the CUDA smoke separately runs the real native FlexAttention kernels.
