"""Load the complete demo interface and bind it to the original Zero-WAM server."""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType

import torch

from .cli import file_sha256
from .demo_context import cache_demo_context, install_demo_interface
from .icl_data import LATENT_NORMALIZATION, _arrays, _identity
from .video_data import _local_path
from .zerowam import DEFAULT_SOURCE, ZERO_WAM_COMMIT, load_native_class


def _bundle_manifest(folder: Path) -> dict:
    manifest = json.loads((folder / "bottleneck.json").read_text())
    required = {"format_version", "kind", "upstream_commit", "demo_bottleneck",
                "icl_rope_h", "feature_space_id", "tiny_native", "files"}
    if (not isinstance(manifest, dict) or set(manifest) != required
            or type(manifest["format_version"]) is not int or manifest["format_version"] != 1
            or manifest["kind"] != "native_icl_temporal_bottleneck"
            or manifest["upstream_commit"] != ZERO_WAM_COMMIT):
        raise ValueError("expected a complete version-1 bottleneck bundle with the pinned Zero-WAM source")
    config = manifest["demo_bottleneck"]
    if (not isinstance(config, dict)
            or set(config) != {"dim", "num_heads", "group_frames", "tokens_per_group", "layers"}
            or any(type(value) is not int or value < 1 for value in config.values())
            or config["dim"] % config["num_heads"]):
        raise ValueError("demo_bottleneck must contain the exact positive integer module configuration")
    if (type(manifest["icl_rope_h"]) is not int or manifest["icl_rope_h"] < 1
            or type(manifest["tiny_native"]) is not bool
            or not isinstance(manifest["feature_space_id"], str) or not manifest["feature_space_id"].strip()):
        raise ValueError("bundle requires explicit RoPE, feature-space and tiny-native identities")
    files = manifest["files"]
    if not isinstance(files, dict) or not {"backbone/config.json", "demo_bottleneck.pt"} <= files.keys():
        raise ValueError("bundle checksums must cover the native config, weights and bottleneck sidecar")
    for name, digest in files.items():
        parts = Path(name).parts if isinstance(name, str) else ()
        allowed = (name == "demo_bottleneck.pt" or len(parts) == 2 and parts[0] == "backbone"
                   and (parts[1] == "config.json" or parts[1].endswith((".safetensors", ".safetensors.index.json"))))
        if (not allowed or Path(name).as_posix() != name or ".." in parts
                or not isinstance(digest, str) or len(digest) != 64
                or set(digest) - set("0123456789abcdef")):
            raise ValueError("bundle checksum paths must name local native files and SHA256 digests")
    actual = set()
    for path in folder.rglob("*"):
        if path.is_symlink():
            raise ValueError("bundle files must be local regular files, not symlinks")
        if path.is_dir() and path != folder / "backbone":
            raise ValueError("bundle must contain only its backbone directory and sidecar files")
        if path.is_file():
            actual.add(path.relative_to(folder).as_posix())
    if actual != set(files) | {"bottleneck.json"}:
        raise ValueError("bundle checksum map is incomplete: missing or untracked files")
    for name, digest in files.items():
        if file_sha256(folder / name) != digest:
            raise ValueError(f"bundle checksum mismatch: {name}")
    weights = {name for name in files if name.endswith(".safetensors")}
    indices = [name for name in files if name.endswith(".safetensors.index.json")]
    if not weights or len(indices) > 1 or not indices and len(weights) != 1:
        raise ValueError("bundle needs one native safetensors file or one complete shard index")
    if indices:
        index = json.loads((folder / indices[0]).read_text())
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if (not isinstance(weight_map, dict) or not weight_map
                or any(not isinstance(name, str) or not name or not isinstance(shard, str)
                       or Path(shard).name != shard or not shard.endswith(".safetensors")
                       for name, shard in weight_map.items())
                or {"backbone/" + shard for shard in weight_map.values()} != weights):
            raise ValueError("native shard index must reference exactly the local checksummed weights")
    return manifest


def _restore_interface(native, folder, manifest):
    weight = native.patch_embedding_mlp.weight
    device, dtype = weight.device, weight.dtype
    install_demo_interface(native, manifest["demo_bottleneck"])
    state = torch.load(folder / "demo_bottleneck.pt", weights_only=True, map_location=device)
    if (not isinstance(state, dict) or not state
            or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in state.values())):
        raise ValueError("bottleneck sidecar must be a finite full state dictionary")
    native.demo_bottleneck.load_state_dict(state, strict=True)
    if manifest["tiny_native"]:
        null = torch.zeros(1, 2, native.config.text_dim, device=device, dtype=dtype)
    else:
        null = torch.load(DEFAULT_SOURCE / "wan_va/assets/empty_text_emb.pt", weights_only=True,
                          map_location=device)[None].to(dtype=dtype)
    if (null.ndim != 3 or null.shape[0] != 1 or not null.shape[1]
            or null.shape[-1] != native.config.text_dim or not torch.isfinite(null).all()):
        raise ValueError("pinned empty-text embedding does not match native text dimensions")
    native.demo_feature_space_id = manifest["feature_space_id"]
    native.demo_icl_rope_h = manifest["icl_rope_h"]
    native.demo_tiny_native = manifest["tiny_native"]
    return native.eval().requires_grad_(False), null


def load_bottleneck_deployment(path, device="cuda"):
    """Strictly restore both merged native weights and the trained demo interface."""
    if "://" in str(path):
        raise ValueError("bottleneck deployment requires a local bundle directory")
    folder = Path(path).resolve()
    manifest = _bundle_manifest(folder)
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    native, info = load_native_class().from_pretrained(str(folder / "backbone"), local_files_only=True,
        use_safetensors=True, torch_dtype=dtype, output_loading_info=True)
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"native backbone must restore exactly: {info}")
    return _restore_interface(native.to(device), folder, manifest)


def attach_bottleneck_server(server, bundle_dir):
    """Restore the demo interface on an already loaded original native server.

    Construct the original server with the bundle's ``backbone/`` component and
    its usual VAE/tokenizer/text encoder first, then attach before reset/infer.
    Observation caching, native sampling, action generation and protocol stay
    the server's original implementations.
    """
    if not getattr(server, "use_icl_model", False) or getattr(server, "cache_name", None) != "pos":
        raise ValueError("bottleneck binding requires the original ICL server and its pos cache")
    if "://" in str(bundle_dir):
        raise ValueError("bottleneck deployment requires a local bundle directory")
    folder = Path(bundle_dir).resolve()
    manifest = _bundle_manifest(folder)
    native = server.transformer
    if not isinstance(native, load_native_class()):
        raise ValueError("server transformer must be the pinned native ICL model")
    source = native.config.get("_name_or_path")
    if not isinstance(source, str) or Path(source).resolve() != folder / "backbone":
        raise ValueError("construct the native server with model_root/transformer pointing to this bundle/backbone before attaching")
    backbone = json.loads((folder / "backbone/config.json").read_text())
    for key in ("patch_size", "in_channels", "out_channels", "action_dim", "text_dim",
                "num_attention_heads", "attention_head_dim", "action_inner_dim"):
        old, new = native.config[key], backbone[key]
        if (tuple(old) if isinstance(old, (list, tuple)) else old) != (tuple(new) if isinstance(new, (list, tuple)) else new):
            raise ValueError(f"server and bundle disagree on native {key}")
    if (type(server.job_config.icl_rope_h) is not int
            or server.job_config.icl_rope_h != manifest["icl_rope_h"]):
        raise ValueError("server icl_rope_h must match the trained bottleneck namespace")
    native.to(device=server.device, dtype=server.dtype)
    native, null = _restore_interface(native, folder, manifest)

    def cache_context(bound_server, video_path, latent_path):
        if not latent_path or "://" in str(latent_path) or Path(latent_path).suffix != ".npz":
            raise ValueError("bottleneck inference requires preprocess-icl-video NPZ and adjacent clip.json; raw video and legacy PT input are unsupported")
        path = Path(latent_path).resolve()
        metadata_path = path.parent / "clip.json"
        metadata = json.loads(metadata_path.read_text())
        if (not isinstance(metadata, dict) or type(metadata.get("format_version")) is not int
                or metadata["format_version"] != 1 or metadata.get("kind") != "native_icl_clip"
                or metadata.get("feature_space_id") != native.demo_feature_space_id
                or _local_path(path.parent, metadata.get("arrays"), ".npz") != path):
            raise ValueError("clip.json must identify this preprocessed NPZ and the bundle's feature space")
        _identity(metadata)
        provenance = metadata.get("provenance", {})
        if (not isinstance(provenance, dict) or provenance.get("latent_normalization") != LATENT_NORMALIZATION
                or provenance.get("latent_layout") != "channel,time,height,width"
                or provenance.get("continuous_segment_verified") is not True
                or provenance.get("feature_encoder") != "frozen-wan-vae"
                or not native.demo_tiny_native and provenance.get("injected_test_encoder") is not False):
            raise ValueError("clip must retain the frozen Wan preprocessing provenance")
        values = _arrays(metadata_path, metadata, robot_target=False)
        if bound_server.job_config.icl_rope_h != native.demo_icl_rope_h:
            raise ValueError("server icl_rope_h changed after binding")
        return cache_demo_context(bound_server.transformer, values["latent"][None], values["frame_times"],
                                  null, icl_rope_h=native.demo_icl_rope_h)

    server.transformer = native
    server._cache_icl_context = MethodType(cache_context, server)
    return server
