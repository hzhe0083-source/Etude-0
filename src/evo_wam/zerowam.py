"""Thin, commit-pinned integration with the real Zero-WAM transformer.

No attention replacement: both training and sampling execute the upstream model.
The wrapper routes task slots with native FlexAttention masks and keeps sampling
and supervised training graphs separate. Calls on one adapter are sequential.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import subprocess
import sys
import warnings
from typing import Sequence

import torch
from torch import Tensor, nn


ZERO_WAM_COMMIT = "08e2c4ae41e2b63573a299825cebe6753481407c"
DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / "third_party" / "Zero-WAM"


class NativeDependencyError(ImportError):
    """The real upstream runtime cannot be imported; never substitute attention."""


_LEGACY_FLASH_IMPORT = """try:
    from flash_attn_interface import flash_attn_func
except:
    from flash_attn import flash_attn_func
"""
_OPTIONAL_FLASH_IMPORT = """_EVO_OPTIONAL_FLASH_IMPORT = True
_EVO_FLASH_UNAVAILABLE = False
try:
    from flash_attn_interface import flash_attn_func
except ImportError:
    try:
        from flash_attn import flash_attn_func
    except ImportError as _flash_error:
        _EVO_FLASH_UNAVAILABLE = True
        _EVO_FLASH_IMPORT_ERROR = _flash_error
        def flash_attn_func(*args, **kwargs):
            raise ImportError(
                "Legacy FlashAttention execution requires a real flash-attn installation; "
                "the Evo-WAM ICL path uses upstream PyTorch FlexAttention."
            ) from _EVO_FLASH_IMPORT_ERROR
"""


def optional_flash_source(source: str) -> str:
    """Change precisely the legacy import guard, never an attention implementation."""
    if source.count(_LEGACY_FLASH_IMPORT) != 1:
        raise RuntimeError("Pinned Zero-WAM legacy FlashAttention import block changed")
    return source.replace(_LEGACY_FLASH_IMPORT, _OPTIONAL_FLASH_IMPORT, 1)


class _OptionalFlashLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        # Bypass bytecode caches: compatibility must never alter upstream source
        # or create a .pyc that ordinary upstream imports would silently reuse.
        source = self.get_data(self.path).decode("utf-8")
        return compile(optional_flash_source(source), self.path, "exec", dont_inherit=True)


class _OptionalFlashFinder(importlib.abc.MetaPathFinder):
    def __init__(self, source: Path):
        self.path = (source / "wan_va/modules/model.py").resolve()

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "wan_va.modules.model":
            return None
        loader = _OptionalFlashLoader(fullname, str(self.path))
        return importlib.util.spec_from_file_location(fullname, self.path, loader=loader)


def verify_source(source: str | Path = DEFAULT_SOURCE) -> Path:
    source = Path(source).resolve()
    result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True)
    if result.stdout.strip() != ZERO_WAM_COMMIT:
        raise ValueError(f"Zero-WAM must be at {ZERO_WAM_COMMIT}")
    subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"],
                   check=True)
    return source


def load_native_class(source: str | Path = DEFAULT_SOURCE):
    source = verify_source(source)
    previous = sys.modules.get("wan_va")
    if previous and Path(previous.__file__).resolve().parent != source / "wan_va":
        raise RuntimeError("Another wan_va installation is already imported")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    finder = _OptionalFlashFinder(source)
    sys.meta_path.insert(0, finder)
    try:
        return importlib.import_module("wan_va.modules.icl_model").WanICLTransformer3DModel
    except ImportError as exc:
        raise NativeDependencyError(
            "Native Zero-WAM needs its pinned torch/diffusers/transformers stack "
            f"and upstream utility dependencies: {exc}"
        ) from exc
    finally:
        sys.meta_path.remove(finder)


@dataclass(frozen=True)
class TaskConditions:
    current: Tensor | None
    remaining: Tensor | None
    null: Tensor  # Native empty-text embedding, [1, N_null, text_dim].

    def detached(self) -> "TaskConditions":
        return TaskConditions(None if self.current is None else self.current.detach(),
                              None if self.remaining is None else self.remaining.detach(),
                              self.null.detach())

    def unconditional(self) -> "TaskConditions":
        return TaskConditions(None, None, self.null)


@dataclass
class NativeOutput:
    video: Tensor
    action: Tensor
    mcp: list[Tensor]
    phi: Tensor  # Current noisy-video tokens only; contains no clean-prefix tokens.


@dataclass(frozen=True)
class GeneratedFuture:
    latents: Tensor
    conditions: TaskConditions
    grid_id: Tensor
    frame_id: int
    rope_offset: int | None = None


@dataclass(frozen=True)
class NativeHistoryChunk:
    """One observed native chunk; physical timestamps are checked by the loader.

    frame_id is an attention chunk ID (video even, action odd), while rope_offset
    is an independent latent-frame position. token_valid refers to patch tokens,
    not channels: padded/unexecuted action positions are false.
    """
    mode: str
    latent: Tensor
    frame_id: int
    rope_offset: int
    token_valid: Tensor | None = None

    def __post_init__(self):
        if self.mode not in {"video", "action"}:
            raise ValueError("History mode must be video or action")
        if type(self.frame_id) is not int or self.frame_id < 0 or self.frame_id % 2 != (self.mode == "action"):
            raise ValueError("History frame_id must be nonnegative, video even/action odd")
        if type(self.rope_offset) is not int or self.rope_offset < 0:
            raise ValueError("History rope_offset must be a nonnegative integer")
        if not torch.is_tensor(self.latent) or self.latent.ndim != 5 or self.latent.shape[0] != 1 or not self.latent.numel():
            raise ValueError("History latent must be nonempty native [1,C,F,H,W]")
        if self.token_valid is not None and (
            not torch.is_tensor(self.token_valid) or self.token_valid.dtype != torch.bool
            or self.token_valid.ndim != 1 or self.token_valid.requires_grad
        ):
            raise ValueError("History token_valid must be a flat, data-owned boolean mask")


def _history_chunk(value) -> NativeHistoryChunk:
    if isinstance(value, NativeHistoryChunk):
        return value
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("Use NativeHistoryChunk with an explicit RoPE offset")
    warnings.warn("Three-tuple history assumes one-frame chunk positions; use NativeHistoryChunk",
                  DeprecationWarning, stacklevel=3)
    mode, latent, frame_id = value
    if type(frame_id) is not int:
        raise ValueError("Legacy history frame_id must be an integer")
    return NativeHistoryChunk(mode, latent, frame_id, frame_id // 2)


class VideoLoRA(nn.Module):
    """Also used for action-specific projections; never applied to shared K/V."""
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float | None = None):
        super().__init__()
        if not 0 < rank <= min(base.in_features, base.out_features):
            raise ValueError("LoRA rank must fit the linear projection")
        self.base = base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False).to(base.weight)
        self.up = nn.Linear(rank, base.out_features, bias=False).to(base.weight)
        nn.init.zeros_(self.up.weight)
        alpha = float(rank if alpha is None else alpha)
        if not torch.isfinite(torch.tensor(alpha)) or alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        self.scale = alpha / rank

    def forward(self, value: Tensor) -> Tensor:
        return self.base(value) + self.up(self.down(value)) * self.scale

    def enable(self, enabled: bool) -> None:
        self.down.requires_grad_(enabled)
        self.up.requires_grad_(enabled)


def route_allowed(query: Tensor, key: Tensor, *, latent_length: int,
                  action_length: int, null_length: int, current_length: int,
                  conditional: bool, mcp: bool) -> Tensor:
    """Native cross-mask intersection; also directly testable on CPU."""
    real_query = query < latent_length + action_length
    null = key < null_length
    current = (key >= null_length) & (key < null_length + current_length)
    remaining = key >= null_length + current_length
    task = conditional and not mcp
    return real_query & (null | (task & current) |
                         (task & (query < latent_length) & remaining))


def unpack_velocity(value: Tensor, shape: Sequence[int], patch_size=(1, 1, 1)) -> Tensor:
    """Undo upstream token order, including its within-patch video ordering."""
    batch, channels, frames, height, width = shape
    pt, ph, pw = patch_size
    if frames % pt or height % ph or width % pw:
        raise ValueError("Latent shape must be divisible by the patch size")
    expected = (batch, frames * height * width, channels)
    if tuple(value.shape) != expected:
        raise ValueError(f"Expected native velocity {expected}, got {tuple(value.shape)}")
    patches = value.reshape(batch, frames // pt, height // ph, width // pw,
                            pt, ph, pw, channels)
    return patches.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(tuple(shape))


def action_mask_for(mask: Tensor | None, sample: Tensor) -> Tensor:
    """Validate data-owned 0/1 masks and broadcast to native [1,A,F,N,1]."""
    if mask is None:
        return torch.ones_like(sample, dtype=torch.bool)
    mask = torch.as_tensor(mask)
    if mask.requires_grad:
        raise ValueError("action_mask must not require gradients")
    if mask.is_complex() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("action_mask must contain only boolean or 0/1 values")
    try:
        return torch.broadcast_to(mask.to(device=sample.device, dtype=torch.bool), sample.shape)
    except RuntimeError as exc:
        raise ValueError("action_mask must broadcast to native [1,A,F,N,1] shape") from exc


class ZeroWAMAdapter(nn.Module):
    def __init__(self, native: nn.Module, condition_dim: int, *, current_tokens=1,
                 remaining_tokens=1, lora_rank=4, lora_alpha=None, lora_dropout=0.0):
        super().__init__()
        if current_tokens < 1 or remaining_tokens < 1:
            raise ValueError("Both fixed task slot groups must be nonempty")
        if lora_dropout != 0:
            raise ValueError("Paired LoRA forwards require lora_dropout=0")
        self.native = native
        if any(isinstance(module, nn.Dropout) and module.p != 0 for module in native.modules()):
            raise ValueError("Paired native forwards require model dropout=0")
        self.current_tokens, self.remaining_tokens = current_tokens, remaining_tokens
        self.condition_dim = condition_dim
        self.condition_projection = nn.Linear(condition_dim, native.config.text_dim)
        self.condition_projection.to(next(native.parameters()))
        self.condition_types = nn.Parameter(torch.empty(2, native.config.text_dim,
                                                        device=next(native.parameters()).device,
                                                        dtype=next(native.parameters()).dtype))
        nn.init.normal_(self.condition_types, std=native.config.text_dim ** -0.5)
        self._video_loras, self._action_loras = [], []
        for block in native.blocks:
            for name, modules in (("to_q", self._video_loras),
                                  ("action_to_q", self._action_loras)):
                lora = VideoLoRA(getattr(block.attn2, name), lora_rank, lora_alpha)
                setattr(block.attn2, name, lora)
                modules.append(lora)
            for name, modules in (("to_out", self._video_loras),
                                  ("action_to_out", self._action_loras)):
                outputs = getattr(block.attn2, name)
                outputs[0] = VideoLoRA(outputs[0], lora_rank, lora_alpha)
                modules.append(outputs[0])
        self._routing = None
        self._hidden: dict[int, Tensor] = {}
        self._fused: Tensor | None = None
        self._physical_actions: Tensor | None = None
        self._handles = []
        for index, block in enumerate(native.blocks):
            self._handles.append(block.attn2.register_forward_pre_hook(self._mask_hook(False)))
            self._handles.append(block.register_forward_hook(self._hidden_hook(index)))
        for group in getattr(native, "mcp_blocks", []):
            self._handles.append(group[0].register_forward_pre_hook(self._physical_action_hook))
            for block in group:
                self._handles.append(block.attn2.register_forward_pre_hook(self._mask_hook(True)))
        if hasattr(native, "mcp_mlp_hidden"):
            self._handles.append(native.mcp_mlp_hidden.register_forward_hook(self._fused_hook))
        self.set_stage("reader")

    @classmethod
    def from_config(cls, config: str | Path | dict, condition_dim: int, *,
                    source=DEFAULT_SOURCE, device="cpu", dtype=torch.bfloat16, **kwargs):
        if not isinstance(config, dict):
            config = json.loads(Path(config).read_text())
        native = load_native_class(source).from_config(config).to(device=device, dtype=dtype)
        return cls(native, condition_dim, **kwargs)

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, condition_dim: int, *,
                        source=DEFAULT_SOURCE, device="cuda", dtype=torch.bfloat16, **kwargs):
        checkpoint = Path(checkpoint).resolve()
        if (checkpoint / "transformer").is_dir():
            checkpoint = checkpoint / "transformer"
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"Missing native transformer config in {checkpoint}")
        native = load_native_class(source).from_pretrained(
            str(checkpoint), local_files_only=True, torch_dtype=dtype)
        return cls(native.to(device=device, dtype=dtype), condition_dim, **kwargs)

    def set_stage(self, stage: str) -> None:
        if stage not in {"interface", "reader", "joint"}:
            raise ValueError("stage must be interface, reader or joint")
        self.requires_grad_(False)
        self.condition_projection.requires_grad_(stage == "interface")
        self.condition_types.requires_grad_(stage == "interface")
        for lora in self._video_loras:
            lora.enable(stage in {"interface", "joint"})
        for lora in self._action_loras:
            lora.enable(stage == "interface")
        if stage == "joint":
            for name in ("mcp_mlp_hidden", "mcp_projections", "mcp_blocks"):
                module = getattr(self.native, name, None)
                if module is not None:
                    module.requires_grad_(True)
        self.stage = stage

    def _hidden_hook(self, index):
        def capture(module, args, output):
            if output[0] is not None:
                self._hidden[index] = output[0]
        return capture

    def _fused_hook(self, module, args, output):
        self._fused = output

    def _physical_action_hook(self, module, args):
        # Main action hidden states now receive g_current directly. Feeding them
        # to MCP would bypass Phi. Keep the native action embedding and masks,
        # but replace that task-contaminated context before the first MCP block.
        if self._physical_actions is None:
            raise RuntimeError("Missing task-free MCP action context")
        return (args[0], self._physical_actions, *args[2:])

    def _mask_hook(self, mcp):
        def route(module, args):
            from torch.nn.attention.flex_attention import create_block_mask
            if self._routing is None:
                raise RuntimeError("Native model must be called through its adapter")
            null_length, conditional = self._routing
            latent, action, pad, encoder = args[:4]
            nl = 0 if latent is None else latent.shape[1]
            na = 0 if action is None else action.shape[1]
            original = module.cross_block_mask.mask_mod
            # FlexAttention's dynamic lowering cannot capture arbitrary SymInt
            # arithmetic in mask_mod. Like upstream's sequence masks, capture
            # tensor metadata rather than token-count scalars.
            query_roles = torch.cat([
                torch.full((nl,), 2, device=encoder.device, dtype=torch.int),
                torch.full((na,), 1, device=encoder.device, dtype=torch.int),
                torch.zeros(pad.shape[1], device=encoder.device, dtype=torch.int),
            ])
            task = conditional and not mcp
            key_roles = torch.cat([
                torch.zeros(null_length, device=encoder.device, dtype=torch.int),
                torch.full((self.current_tokens,), 1 if task else 3,
                           device=encoder.device, dtype=torch.int),
                torch.full((self.remaining_tokens,), 2 if task else 3,
                           device=encoder.device, dtype=torch.int),
            ])

            def mask(b, h, q, k):
                qr, kr = query_roles[q], key_roles[k]
                return original(b, h, q, k) & (qr > 0) & (
                    (kr == 0) | (kr == 1) | ((kr == 2) & (qr == 2)))

            module.cross_block_mask = create_block_mask(
                mask, 1, 1, nl + na + pad.shape[1], encoder.shape[1],
                device=encoder.device, _compile=False)
        return route

    def _condition_input(self, inputs: dict, conditions: TaskConditions) -> dict:
        if inputs.get("icl_latent_dict") is not None:
            raise ValueError("Raw ICL tokens bypass the Evo-WAM task interface")
        null = conditions.null
        if null.ndim != 3 or null.shape[0] != 1 or null.shape[1] < 1 or null.shape[2] != self.native.config.text_dim:
            raise ValueError("null must be a nonempty native [1,N,text_dim] embedding")
        active = conditions.current is not None
        if active != (conditions.remaining is not None):
            raise ValueError("Drop current and remaining together")
        reference = self.condition_projection.weight
        null = null.to(reference)
        task = []
        for kind, (tokens, length) in enumerate(((conditions.current, self.current_tokens),
                                                (conditions.remaining, self.remaining_tokens))):
            if tokens is None:
                task.append(null.new_zeros(1, length, self.native.config.text_dim))
            else:
                if tuple(tokens.shape) != (1, length, self.condition_dim):
                    raise ValueError("Task tokens do not match their fixed slot shape")
                task.append(self.condition_projection(tokens.to(reference)) + self.condition_types[kind])
        text = torch.cat([null, *task], dim=1)
        if not torch.isfinite(text).all():
            raise ValueError("Condition embeddings must be finite")
        self._routing = (null.shape[1], active)
        return {**inputs, "text_emb": text,
                "encoder_seq_ids": torch.zeros(text.shape[1], device=text.device, dtype=torch.int)}

    def clear_cache(self) -> None:
        self.native.clear_cache()
        self._hidden.clear()
        self._fused = None
        self._physical_actions = None

    def _call_native(self, *args, **kwargs):
        # History, training and sampling have distinct native Flex signatures.
        # Do not silently fall back to a dense score matrix at Dynamo's default
        # eight-specialization limit; a real capacity failure must be visible.
        import torch._dynamo.config as compiler_config
        with compiler_config.patch(recompile_limit=max(32, compiler_config.recompile_limit),
                                   fail_on_recompile_limit_hit=True):
            return self.native(*args, **kwargs)

    def _history_training_inputs(self, inputs: dict, history) -> tuple[dict, Tensor, Tensor]:
        """Rebase future grids after past context without changing targets/noise."""
        chunks = tuple(_history_chunk(item) for item in history)
        frame_start = 2 * (max(chunk.frame_id for chunk in chunks) // 2 + 1)
        rope_start = max(chunk.rope_offset + chunk.latent.shape[2] // (
            self.native.patch_size[0] if chunk.mode == "video" else 1) for chunk in chunks)
        base = inputs["latent_dict"]["grid_id"][0, 0].min()
        if inputs["action_dict"]["grid_id"][0, 0].min() != base:
            raise ValueError("Future video/action training grids must share a window origin")

        def shifted(stream):
            grid = stream["grid_id"].clone()
            grid[:, 0] += rope_start - base
            return {**stream, "grid_id": grid}

        result = {**inputs, "latent_dict": shifted(inputs["latent_dict"]),
                  "action_dict": shifted(inputs["action_dict"]),
                  "mcp_latent_dicts": [shifted(s) for s in inputs.get("mcp_latent_dicts", [])]}
        size = int(inputs["chunk_size"])
        if size < 1:
            raise ValueError("Training chunk_size must be positive")
        video_frames = ((result["latent_dict"]["grid_id"][0, 0] - rope_start) // size * 2 + frame_start).to(torch.int)
        action_frames = ((result["action_dict"]["grid_id"][0, 0] - rope_start) // size * 2 + frame_start + 1).to(torch.int)
        return result, video_frames, action_frames

    def _history_training_masks(self, original, prefix_seq, prefix_frame,
                                video_frames, action_frames):
        """Extend native teacher forcing with clean, observed prefix KV keys."""
        from torch.nn.attention.flex_attention import create_block_mask

        def prepare(seq_ids, frame_ids, noise_ids, type_ids, icl_ids,
                    cross_seq_ids, encoder_seq_ids, window_size, blocks):
            frames = torch.cat([video_frames, video_frames, action_frames, action_frames])
            frames = torch.cat([frames, frames.new_full((frame_ids.numel() - frames.numel(),), -1)])
            original(seq_ids, frames, noise_ids, type_ids, icl_ids,
                     cross_seq_ids, encoder_seq_ids, window_size, blocks)
            # MCP has its own blocks and no prefix KV. Its only task-bearing
            # source remains the native fused main-branch Phi.
            if blocks is not self.native.blocks:
                return
            key_seq = torch.cat([prefix_seq, seq_ids])
            key_frame = torch.cat([prefix_frame, frames])
            key_noise = torch.cat([torch.ones_like(prefix_seq), noise_ids])
            window = torch.tensor(window_size, device=seq_ids.device, dtype=torch.int)

            def mask(b, h, q, k):
                valid = (seq_ids[q] >= 0) & (seq_ids[q] == key_seq[k])
                in_window = (window == -1) | ((frames[q] - key_frame[k]).abs() <= window)
                clean = (noise_ids[q] == 1) & (key_noise[k] == 1) & (key_frame[k] <= frames[q])
                past = (noise_ids[q] == 0) & (key_noise[k] == 1) & (key_frame[k] < frames[q])
                same_noisy = (noise_ids[q] == 0) & (key_noise[k] == 0) & (key_frame[k] == frames[q])
                return valid & in_window & (clean | past | same_noisy)

            self_mask = create_block_mask(mask, 1, 1, seq_ids.numel(), key_seq.numel(),
                                          device=seq_ids.device, _compile=False)
            for block in blocks:
                block.attn1.set_block_masks(self_mask=self_mask)
        return prepare

    def forward_train(self, inputs: dict, conditions: TaskConditions, *, history=()) -> NativeOutput:
        self.clear_cache()
        original_masks = self.native._build_training_masks
        had_override = "_build_training_masks" in self.native.__dict__
        try:
            if history:
                inputs, video_frames, action_frames = self._history_training_inputs(inputs, history)
                # Unlike deployment prefill, training context stays differentiable
                # through native W. Only the observed data tensors are detached.
                self._prefill_history(history, conditions)
                self.native._build_training_masks = self._history_training_masks(
                    original_masks, self.native.seq_ids_cache, self.native.frame_ids_cache,
                    video_frames, action_frames)
            if getattr(self.native, "enable_mcp", False):
                action = inputs["action_dict"]
                self._physical_actions = torch.cat([
                    self.native._training_embed(action["noisy_latents"], action["timesteps"], "action")[0],
                    self.native._training_embed(action["latent"], action["cond_timesteps"], "action")[0],
                ], dim=1)
            result = self._call_native(self._condition_input(inputs, conditions), train_mode=True)
            video, action = result[:2]
            count = video.shape[1] // int(torch.tensor(self.native.patch_size).prod())
            if self._fused is not None:
                phi = self._fused[:, :count]
            elif hasattr(self.native, "mcp_mlp_hidden"):
                phi = self.native.mcp_mlp_hidden(torch.cat(
                    [self._hidden[i][:, :count] for i in self.native.mcp_hidden_collect_layers], -1))
            else:
                phi = self._hidden[len(self.native.blocks) - 1][:, :count]
            return NativeOutput(video, action, list(result[2]) if len(result) == 3 else [], phi)
        finally:
            if had_override:
                self.native._build_training_masks = original_masks
            elif "_build_training_masks" in self.native.__dict__:
                del self.native._build_training_masks
            self.clear_cache()

    def _stream(self, data: Tensor, mode: str, timestep, frame_id: int,
                grid_id: Tensor | None = None, cache_type=1, *,
                rope_offset: int | None = None, token_valid: Tensor | None = None) -> dict:
        from wan_va.utils import get_mesh_id
        if data.ndim != 5 or data.shape[0] != 1:
            raise ValueError("Native streams require [1,C,F,H,W]")
        if mode not in {"video", "action"}:
            raise ValueError("History stream mode must be video or action")
        if type(frame_id) is not int or frame_id < 0 or frame_id % 2 != (mode == "action"):
            raise ValueError("Stream frame_id must be nonnegative, video even/action odd")
        if rope_offset is not None and (type(rope_offset) is not int or rope_offset < 0):
            raise ValueError("rope_offset must be a nonnegative integer")
        data = data.to(self.native.patch_embedding_mlp.weight if mode == "video"
                       else self.native.action_embedder.weight)
        patch = self.native.patch_size if mode == "video" else (1, 1, 1)
        if any(n % p for n, p in zip(data.shape[-3:], patch)):
            raise ValueError("Stream shape is not patch aligned")
        if grid_id is None:
            f, h, w = (n // p for n, p in zip(data.shape[-3:], patch))
            offset = frame_id // 2 if rope_offset is None else rope_offset
            grid_id = get_mesh_id(f, h, w, int(mode == "action"), f_shift=offset).to(data.device)
        if grid_id.ndim == 3:
            grid_id = grid_id[0]
        grid_id = grid_id.to(data.device)
        expected_count = (data.shape[2] // patch[0]) * (data.shape[3] // patch[1]) * (data.shape[4] // patch[2])
        if grid_id.ndim != 2 or grid_id.shape != (4, expected_count):
            raise ValueError("Grid must have shape [4, native patch-token count]")
        if rope_offset is not None and not (grid_id[0].min() == rope_offset):
            raise ValueError("Explicit grid and rope_offset disagree")
        count = grid_id.shape[1]
        if token_valid is None:
            valid = torch.ones(count, device=data.device, dtype=torch.bool)
        else:
            if token_valid.dtype != torch.bool or token_valid.shape != (count,) or token_valid.requires_grad:
                raise ValueError("token_valid must be a flat boolean mask matching native tokens")
            valid = token_valid.to(device=data.device)
            if not valid.any():
                raise ValueError("Omit empty history chunks instead of caching only padding")
            pixels = valid.reshape(*(n // p for n, p in zip(data.shape[-3:], patch)))
            for axis, repeats in enumerate(patch):
                pixels = pixels.repeat_interleave(repeats, dim=axis)
            data = torch.where(pixels[None, None], data, 0)
        t = torch.as_tensor(timestep, device=data.device, dtype=torch.float32).reshape(-1)
        frames = data.shape[2] // patch[0]
        if t.numel() not in (1, frames):
            raise ValueError("Stream needs one timestep or one timestep per frame")
        if t.numel() == 1:
            t = t.expand(frames)
        stream = {"noisy_latents": data, "timesteps": t,
                  "cache_type_ids": torch.full((count,), cache_type, device=data.device, dtype=torch.int)}
        key = "latent" if mode == "video" else "action"
        return {f"{key}_res_lst": stream, f"{key}_grid_id": grid_id,
                "current_seq_ids": torch.zeros(count, device=data.device, dtype=torch.int).masked_fill(~valid, -1),
                "current_frame_ids": torch.full((count,), frame_id, device=data.device, dtype=torch.int).masked_fill(~valid, -1)}

    @torch.no_grad()
    def prefill_history(self, history: Sequence[NativeHistoryChunk | tuple[str, Tensor, int]],
                        conditions: TaskConditions) -> None:
        """Clear and replay only actual observed video/action chunks in time order."""
        self._prefill_history(history, conditions.detached())

    def _prefill_history(self, history, conditions) -> None:
        self.clear_cache()
        previous = -1
        ends = {"video": -1, "action": -1}
        try:
            for item in history:
                chunk = _history_chunk(item)
                if chunk.frame_id <= previous:
                    raise ValueError("History frame IDs must be strictly increasing")
                if chunk.rope_offset < ends[chunk.mode]:
                    raise ValueError("History RoPE intervals overlap within a modality")
                previous = chunk.frame_id
                pt = self.native.patch_size[0] if chunk.mode == "video" else 1
                ends[chunk.mode] = chunk.rope_offset + chunk.latent.shape[2] // pt
                payload = self._stream(chunk.latent.detach(), chunk.mode, 0, chunk.frame_id,
                                       cache_type=0, rope_offset=chunk.rope_offset,
                                       token_valid=chunk.token_valid)
                self._call_native(self._condition_input(payload, conditions),
                            mode=f"forward_{'latent' if chunk.mode == 'video' else 'action'}_only", update_cache=1)
        except Exception:
            self.clear_cache()
            raise

    @torch.no_grad()
    def sample_video(self, initial_noise: Tensor, conditions: TaskConditions, *,
                     history=(), steps=4, shift=5.0, frame_id=2, grid_id=None,
                     rope_offset: int | None = None) -> GeneratedFuture:
        """From noise only; ground-truth future/training dictionaries are not inputs."""
        from wan_va.utils import FlowMatchScheduler
        if steps < 1:
            raise ValueError("Sampling steps must be positive")
        if history and frame_id <= _history_chunk(history[-1]).frame_id:
            raise ValueError("Future must follow the actual history")
        conditions = conditions.detached()
        scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
        scheduler.set_timesteps(steps)
        self.prefill_history(history, conditions)
        sample = initial_noise.detach().to(self.native.patch_embedding_mlp.weight).clone()
        try:
            for timestep in scheduler.timesteps:
                payload = self._stream(sample, "video", timestep, frame_id, grid_id,
                                       rope_offset=rope_offset)
                velocity = self._call_native(self._condition_input(payload, conditions),
                                       mode="forward_latent_only", update_cache=0)
                velocity = unpack_velocity(velocity, sample.shape, self.native.patch_size)
                sample = scheduler.step(velocity, timestep, sample)
            return GeneratedFuture(sample.detach(), conditions,
                                   payload["latent_grid_id"].detach(), frame_id,
                                   int(payload["latent_grid_id"][0].min().item()))
        finally:
            self.clear_cache()

    def action_velocity(self, noisy_action: Tensor, timestep, conditions: TaskConditions,
                        future: GeneratedFuture, *, history=()) -> Tensor:
        """Frozen action parameters still transmit gradients to current task tokens."""
        if not isinstance(future, GeneratedFuture) or future.latents.requires_grad:
            raise ValueError("Action decoding requires a detached GeneratedFuture")
        if history and future.frame_id <= _history_chunk(history[-1]).frame_id:
            raise ValueError("Generated future must follow the actual history")
        self.prefill_history(history, future.conditions)
        try:
            with torch.no_grad():
                video = self._stream(future.latents, "video", 0, future.frame_id,
                                     future.grid_id, cache_type=1, rope_offset=future.rope_offset)
                self._call_native(self._condition_input(video, future.conditions),
                            mode="forward_latent_only", update_cache=1)
            offset = future.rope_offset
            if offset is None:
                offset = int(future.grid_id[0].min().item())
            action = self._stream(noisy_action, "action", timestep, future.frame_id + 1,
                                  rope_offset=offset)
            result = self._call_native(self._condition_input(action, conditions),
                                 mode="forward_action_only", update_cache=0)
            return unpack_velocity(result, noisy_action.shape)
        finally:
            self.clear_cache()

    @torch.no_grad()
    def sample_actions(self, initial_noise: Tensor, conditions: TaskConditions,
                       future: GeneratedFuture, *, history=(), steps=4, shift=1.0,
                       action_mask: Tensor | None = None) -> Tensor:
        """Keep unused action channels zero at initialization and every flow step."""
        from wan_va.utils import FlowMatchScheduler
        if steps < 1:
            raise ValueError("Sampling steps must be positive")
        scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
        scheduler.set_timesteps(steps)
        sample = initial_noise.detach().to(self.native.action_embedder.weight).clone()
        active = action_mask_for(action_mask, sample)
        sample = torch.where(active, sample, 0)
        # ponytail: replay history each step for strict cache isolation; cache snapshots
        # can replace replay only after memory/latency measurements justify them.
        for timestep in scheduler.timesteps:
            velocity = self.action_velocity(sample, timestep, conditions, future, history=history)
            sample = scheduler.step(velocity, timestep, sample)
            sample = torch.where(active, sample, 0)
        return sample


def tiny_native_smoke(source=DEFAULT_SOURCE) -> dict:
    """Real CUDA FlexAttention forward/MCP/backward and generated-video action path."""
    if not torch.cuda.is_available():
        raise NativeDependencyError("Native smoke requires CUDA; CPU structure checks are separate")
    native_cls = load_native_class(source)
    from wan_va.utils import get_mesh_id
    model = native_cls(patch_size=(1, 1, 1), num_attention_heads=2,
                      attention_head_dim=18, in_channels=4, out_channels=4,
                      action_dim=3, text_dim=8, freq_dim=4, ffn_dim=16,
                      num_layers=2, rope_max_seq_len=32, action_inner_dim=36,
                      action_ffn_dim=16, attn_window=8, enable_mcp=True,
                      num_mcp_modules=1, mcp_hidden_collect_layers=(0, 1))
    adapter = ZeroWAMAdapter(model.to(device="cuda", dtype=torch.bfloat16), 8, lora_rank=2)
    adapter.eval().set_stage("joint")
    device, dtype = "cuda", torch.bfloat16
    video = torch.randn(1, 4, 2, 1, 2, device=device, dtype=dtype)
    action = torch.randn(1, 3, 2, 2, 1, device=device, dtype=dtype)

    def stream(data, action=False, shift=0):
        grid = get_mesh_id(*data.shape[-3:], int(action), f_shift=shift).to(device)[None]
        return {"noisy_latents": data, "latent": data.clone(), "grid_id": grid,
                "timesteps": torch.full((1, 2), 500., device=device),
                "cond_timesteps": torch.zeros(1, 2, device=device)}

    inputs = {"latent_dict": stream(video), "action_dict": stream(action, True),
              "mcp_latent_dicts": [stream(torch.randn_like(video), shift=1)],
              "chunk_size": 1, "max_frame_chunk_size": 4, "window_size": 8}
    current = torch.randn(1, 1, 8, device=device, dtype=dtype, requires_grad=True)
    remaining = torch.randn_like(current).requires_grad_()
    conditions = TaskConditions(current, remaining, torch.zeros_like(current))
    captured = {}

    def capture_fusion(module, args, output):
        captured["fusion"] = output.detach().clone()

    def check_physical_context(module, args):
        assert args[1] is adapter._physical_actions
        assert not args[1].requires_grad
        captured["actions"] = args[1].detach().clone()

    fusion_handle = model.mcp_mlp_hidden.register_forward_hook(capture_fusion)
    action_handle = model.mcp_blocks[0][0].register_forward_pre_hook(check_physical_context)
    result = adapter.forward_train(inputs, conditions)
    fusion_handle.remove()
    original_actions = captured["actions"]
    fusion = captured["fusion"]
    swapped = adapter.forward_train(inputs, TaskConditions(remaining, current, conditions.null))
    assert not torch.equal(swapped.video, result.video), "W ignored current/remaining type markers"
    assert result.phi.shape == (1, 4, 36) and len(result.mcp) == 1
    loss = result.video.float().square().mean() + result.mcp[0].float().square().mean()
    loss.backward()
    assert all(torch.isfinite(x).all() for x in [result.video, result.action, result.phi, *result.mcp])
    assert any(x.up.weight.grad is not None and x.up.weight.grad.abs().sum() > 0
               for x in adapter._video_loras)
    assert all(x.up.weight.grad is None for x in adapter._action_loras)
    # Hold the ONLY task-bearing MCP input fixed. Task changes must then have no
    # effect via direct text or the auxiliary action context.
    fixed_handle = model.mcp_mlp_hidden.register_forward_hook(lambda module, args, output: fusion)
    alternate_current = (current.detach() + 0.5).requires_grad_()
    alternate = TaskConditions(alternate_current, conditions.remaining + 0.5, conditions.null)
    blocked = adapter.forward_train(inputs, alternate)
    fixed_handle.remove()
    action_handle.remove()
    torch.testing.assert_close(blocked.mcp[0], result.mcp[0].detach(), rtol=0, atol=0)
    assert torch.equal(captured["actions"], original_actions)
    shortcut = torch.autograd.grad(blocked.mcp[0].float().square().mean(),
                                   alternate_current, allow_unused=True)[0]
    assert shortcut is None or not shortcut.any()

    # A real training-history check, independent of future targets: changing
    # observed commands changes main W/Phi; changing invalid padding does not.
    historical_video = torch.randn_like(video).detach()
    historical_action = torch.randn_like(action).detach()
    history_valid = torch.tensor([True, True, True, False])
    train_history = [NativeHistoryChunk("video", historical_video, 4, 10),
                     NativeHistoryChunk("action", historical_action, 5, 10, history_valid)]
    saved_grids = [inputs["latent_dict"]["grid_id"].clone(), inputs["action_dict"]["grid_id"].clone(),
                   inputs["mcp_latent_dicts"][0]["grid_id"].clone()]
    prefix_keys = []

    def capture_training_prefix(module, args):
        if args[0] is not None and args[1] is not None:
            key = module.attn_caches["pos"]["k"]
            assert key.requires_grad, "Training history was detached/no_grad"
            key.retain_grad()
            prefix_keys.append(key)

    history_hook = model.blocks[1].attn1.register_forward_pre_hook(capture_training_prefix)
    contextual = adapter.forward_train(inputs, conditions, history=train_history)
    history_hook.remove()
    changed = historical_action.clone()
    changed[:, :, 0, 0] += 3
    different = adapter.forward_train(inputs, conditions, history=[train_history[0],
        NativeHistoryChunk("action", changed, 5, 10, history_valid)])
    assert not torch.equal(contextual.video, different.video)
    assert not torch.equal(contextual.phi, different.phi)
    padding_only = historical_action.clone()
    padding_only[:, :, -1, -1] = 999
    padded = adapter.forward_train(inputs, conditions, history=[train_history[0],
        NativeHistoryChunk("action", padding_only, 5, 10, history_valid)])
    replay = adapter.forward_train(inputs, conditions, history=train_history)
    for output in (padded, replay):
        torch.testing.assert_close(contextual.video, output.video, rtol=0, atol=0)
        torch.testing.assert_close(contextual.phi, output.phi, rtol=0, atol=0)
    assert contextual.video.shape == result.video.shape and contextual.phi.shape == result.phi.shape
    for before, after in zip(saved_grids, [inputs["latent_dict"]["grid_id"], inputs["action_dict"]["grid_id"],
                                          inputs["mcp_latent_dicts"][0]["grid_id"]]):
        assert torch.equal(before, after), "A paired view mutated the shared future grid"
    (contextual.video.float().square().mean() + contextual.mcp[0].float().square().mean()).backward()
    assert prefix_keys and all(key.grad is not None and torch.isfinite(key.grad).all()
                               and key.grad.abs().sum() > 0 for key in prefix_keys)
    assert "_build_training_masks" not in model.__dict__ and model.seq_ids_cache is None
    adapter.zero_grad(set_to_none=True)
    current.grad = None
    remaining.grad = None
    # Keep W LoRA trainable here: absence of W gradients must come from the
    # stopped sampling/encoding path, not merely from freezing every parameter.
    adapter.set_stage("joint")
    past_actions = action.detach().clone()
    past_actions[:, :, -1, -1] = 100
    history = [NativeHistoryChunk("video", video.detach(), 0, 0),
               NativeHistoryChunk("action", past_actions, 1, 0,
                                  torch.tensor([True, True, True, False]))]
    adapter.prefill_history(history, conditions)
    assert adapter.native.cache_counts()[0] == 7  # Four video + three executed action tokens.
    assert adapter.native.seq_ids_cache[128 + 3] == -1
    assert adapter.native.frame_ids_cache[128 + 3] == -1
    original_action_keys = model.blocks[0].attn1.attn_caches["pos"]["k"][:, 128:131].clone()
    changed_actions = past_actions.clone()
    changed_actions[:, :, 0, 0] += 3
    adapter.prefill_history([history[0], NativeHistoryChunk("action", changed_actions, 1, 0,
        torch.tensor([True, True, True, False]))], conditions)
    assert not torch.equal(original_action_keys, model.blocks[0].attn1.attn_caches["pos"]["k"][:, 128:131])
    future = adapter.sample_video(torch.randn_like(video[:, :, :1]).float().cpu(), conditions,
                                  history=history, steps=2, frame_id=2, rope_offset=2)
    assert future.rope_offset == 2 and future.grid_id[0].min() == 2
    assert not future.latents.requires_grad and future.latents.grad_fn is None
    velocity = adapter.action_velocity(action[:, :, :1].float().cpu(), torch.tensor([[500.]]), conditions,
                                       future, history=history)
    velocity.float().square().mean().backward()
    assert current.grad is not None and torch.isfinite(current.grad).all() and current.grad.abs().sum() > 0
    assert remaining.grad is None or not remaining.grad.any()
    assert not any(p.grad is not None for p in adapter.parameters())
    seen_actions = []

    def capture_sample_input(module, args):
        # History has four tokens; the sampled one-frame action has two.
        if args[0].shape[1] == 2:
            seen_actions.append(args[0].detach().clone())

    sample_hook = model.action_embedder.register_forward_pre_hook(capture_sample_input)
    action_noise = torch.randn_like(action[:, :, :1]).float().cpu()
    action_noise[:, 1] = 100
    try:
        sampled_action = adapter.sample_actions(action_noise, conditions, future,
            history=history, steps=2, action_mask=torch.tensor([1, 0, 1]).reshape(1, 3, 1, 1, 1))
    finally:
        sample_hook.remove()
    assert len(seen_actions) == 2 and all(not value[..., 1].any() for value in seen_actions)
    assert not sampled_action[:, 1].any()
    assert sampled_action.shape == action[:, :, :1].shape
    assert sampled_action.device.type == "cuda" and torch.isfinite(sampled_action).all()
    assert not sampled_action.requires_grad
    assert adapter.native.type_ids_cache is None
    return {"native": True, "device": device, "mcp_heads": len(result.mcp),
            "phi_shape": list(result.phi.shape), "direct_condition_gradient": True,
            "mcp_phi_only_task_path": True, "current_remaining_types": True,
            "inactive_action_channels_zero_each_step": True,
            "observed_action_history": True, "history_padding_excluded": True,
            "independent_rope_offset": True, "training_observed_history": True,
            "differentiable_history_prefix": True, "paired_training_grids_unchanged": True}
