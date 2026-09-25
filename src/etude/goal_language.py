"""Frozen local Zero-WAM instruction caches; no implicit null text or downloads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .video_data import _local_path, _text
from .vision import sha256


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_map(value, name):
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must identify every local encoder/tokenizer file")
    for filename, digest in value.items():
        if (not isinstance(filename, str) or not filename or Path(filename).is_absolute()
                or ".." in Path(filename).parts or not isinstance(digest, str)
                or len(digest) != 64 or set(digest) - set("0123456789abcdef")):
            raise ValueError(f"invalid {name} file identity")


def _language_metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    required = {"format_version", "kind", "arrays", "arrays_sha256", "text", "text_sha256",
                "cleaned_text", "valid_length", "identity"}
    if (not isinstance(metadata, dict) or set(metadata) != required
            or type(metadata.get("format_version")) is not int or metadata["format_version"] != 1
            or metadata.get("kind") != "goal_language"):
        raise ValueError("expected an explicit version-1 goal_language cache")
    _local_path(path.parent, _text(metadata, "arrays"), ".npz")
    if _text_hash(_text(metadata, "text")) != metadata["text_sha256"]:
        raise ValueError("language original text hash mismatch")
    _text(metadata, "cleaned_text")
    if type(metadata["valid_length"]) is not int or not 1 <= metadata["valid_length"] <= 512:
        raise ValueError("language valid_length must be between 1 and 512")
    identity = metadata["identity"]
    if (not isinstance(identity, dict)
            or set(identity) != {"encoder_sha256", "tokenizer_sha256", "preprocessing", "text_dim"}
            or type(identity["text_dim"]) is not int or identity["text_dim"] < 1):
        raise ValueError("language identity must include encoder, tokenizer, preprocessing and text_dim")
    for name in ("encoder_sha256", "tokenizer_sha256"):
        _hash_map(identity[name], name)
    if "config.json" not in identity["encoder_sha256"] or not any(
            name.endswith(".safetensors") for name in identity["encoder_sha256"]):
        raise ValueError("language encoder identity must include config and safetensors weights")
    policy = identity["preprocessing"]
    expected = {"cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean", "max_length": 512,
                "padding": "max_length", "truncation": True, "add_special_tokens": True,
                "zero_padding": True, "output_dtype": "float32"}
    if (not isinstance(policy, dict) or set(policy) != set(expected) | {
            "diffusers_version", "transformers_version", "encoder_dtype"}
            or any(policy.get(key) != value or type(policy.get(key)) is not type(value)
                   for key, value in expected.items())
            or not isinstance(policy.get("encoder_dtype"), str)
            or policy["encoder_dtype"] not in {"float32", "bfloat16"}):
        raise ValueError("language cache must use the native prompt_clean/max512/zero-padding policy")
    for name in ("diffusers_version", "transformers_version"):
        _text(policy, name)
    if (not isinstance(metadata["arrays_sha256"], str) or len(metadata["arrays_sha256"]) != 64
            or set(metadata["arrays_sha256"]) - set("0123456789abcdef")):
        raise ValueError("invalid language arrays_sha256")
    return metadata


def load_goal_language(manifest_path: str | Path) -> tuple[torch.Tensor, dict]:
    """Validate the full padded cache, then return only valid frozen text tokens."""
    path = Path(manifest_path)
    metadata = _language_metadata(path)
    arrays = _local_path(path.parent, metadata["arrays"], ".npz")
    if sha256(arrays) != metadata["arrays_sha256"]:
        raise ValueError("language cache arrays hash mismatch")
    with np.load(arrays, allow_pickle=False) as archive:
        if archive.files != ["language"]:
            raise ValueError("language cache NPZ needs exactly language")
        language = torch.from_numpy(archive["language"].copy())
    length, dimension = metadata["valid_length"], metadata["identity"]["text_dim"]
    if (language.shape != (1, 512, dimension) or language.dtype != torch.float32
            or not torch.isfinite(language).all() or torch.count_nonzero(language[:, length:])):
        raise ValueError("language must be finite float32 [1,512,D] with exactly zero padding")
    return language[:, :length].contiguous(), metadata["identity"]


def cache_goal_language(args):
    """Cache one explicit instruction using a local native UMT5 checkpoint."""
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    import diffusers
    import transformers
    from transformers import T5TokenizerFast, UMT5EncoderModel

    if not isinstance(args.text, str) or not args.text.strip():
        raise ValueError("text must be an explicitly provided nonempty instruction")
    if "://" in str(args.checkpoint):
        raise ValueError("checkpoint must be a local Zero-WAM model directory")
    checkpoint, output = Path(args.checkpoint).expanduser().resolve(), Path(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("language cache output must be a fresh directory")
    encoder_path, tokenizer_path = checkpoint / "text_encoder", checkpoint / "tokenizer"
    for component in (encoder_path, tokenizer_path):
        if not component.is_dir():
            raise ValueError(f"missing local model component: {component.name}")
    files = lambda folder: {str(path.relative_to(folder)): sha256(path)
                            for path in sorted(folder.rglob("*")) if path.is_file()}
    dtype = torch.bfloat16 if torch.device(args.device).type == "cuda" else torch.float32
    identity = {
        "encoder_sha256": files(encoder_path), "tokenizer_sha256": files(tokenizer_path),
        "preprocessing": {"cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean",
                          "max_length": 512, "padding": "max_length", "truncation": True,
                          "add_special_tokens": True, "zero_padding": True, "output_dtype": "float32",
                          "encoder_dtype": str(dtype).removeprefix("torch."),
                          "diffusers_version": diffusers.__version__,
                          "transformers_version": transformers.__version__},
    }
    if "config.json" not in identity["encoder_sha256"] or not any(
            name.endswith(".safetensors") for name in identity["encoder_sha256"]):
        raise ValueError("local text_encoder requires config.json and safetensors weights")
    for index in encoder_path.glob("*.safetensors.index.json"):
        mapping = json.loads(index.read_text()).get("weight_map")
        if (not isinstance(mapping, dict) or not mapping
                or any(not isinstance(shard, str) or shard not in identity["encoder_sha256"]
                       or not shard.endswith(".safetensors") for shard in mapping.values())):
            raise ValueError("text encoder index references an unverified local weight shard")
    tokenizer = T5TokenizerFast.from_pretrained(str(tokenizer_path), local_files_only=True)
    encoder = UMT5EncoderModel.from_pretrained(str(encoder_path), local_files_only=True,
                                               use_safetensors=True, torch_dtype=dtype)
    encoder.eval().requires_grad_(False).to(args.device)
    cleaned = prompt_clean(args.text)
    if not cleaned.strip():
        raise ValueError("instruction is empty after native prompt_clean")
    tokens = tokenizer([cleaned], padding="max_length", max_length=512, truncation=True,
                       add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    mask = tokens.attention_mask.to(args.device)
    length = int(mask.gt(0).sum().item())
    if mask.shape != (1, 512) or not 1 <= length <= 512:
        raise ValueError("native tokenizer must return a nonempty [1,512] instruction")
    with torch.no_grad():
        encoded = encoder(tokens.input_ids.to(args.device), mask).last_hidden_state
        if encoded.ndim != 3 or encoded.shape[:2] != (1, 512) or not torch.isfinite(encoded).all():
            raise ValueError("native text encoder must return finite [1,512,D] features")
        language = encoded.detach().float().cpu()
        language[:, length:] = 0
    identity["text_dim"] = language.shape[-1]
    output.mkdir(parents=True, exist_ok=True)
    arrays_path = output / "language.npz"
    np.savez_compressed(arrays_path, language=language.numpy())
    metadata = {"format_version": 1, "kind": "goal_language", "arrays": arrays_path.name,
                "arrays_sha256": sha256(arrays_path), "text": args.text, "text_sha256": _text_hash(args.text),
                "cleaned_text": cleaned, "valid_length": length, "identity": identity}
    manifest = output / "language.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    load_goal_language(manifest)
    return {"manifest": str(manifest), "shape": [1, length, identity["text_dim"]], "identity": identity}
