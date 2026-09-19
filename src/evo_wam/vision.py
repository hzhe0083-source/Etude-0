"""Local Wan VAE preprocessing; tracked entity masks remain explicit inputs.

No detector, camera calibration or contact annotations are inferred here. Frozen
Wan features feed video-only pretraining and robot ROIs; an optional frozen B
artifact converts ordered demonstration features into effect tokens.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_vae(path, expected_files, *, device="cuda"):
    """Require local files with recorded identities; never fetch missing weights."""
    from diffusers import AutoencoderKLWan
    path = Path(path).resolve()
    if not path.is_dir() or not isinstance(expected_files, dict) or "config.json" not in expected_files:
        raise ValueError("provide a local VAE directory and SHA256 map including config.json")
    weights = {p.name for p in path.glob("*.safetensors")}
    actual = weights | {p.name for p in path.glob("*.safetensors.index.json")} | {"config.json"}
    if not weights or set(expected_files) != actual:
        raise ValueError("VAE identity must cover config and every safetensors shard")
    for name, digest in expected_files.items():
        if Path(name).name != name or sha256(path / name) != digest:
            raise ValueError(f"VAE file identity mismatch: {name}")
    for name in actual - weights - {"config.json"}:
        mapping = json.loads((path / name).read_text()).get("weight_map")
        if (not isinstance(mapping, dict) or not mapping
                or any(not isinstance(shard, str) or Path(shard).name != shard or shard not in weights
                       for shard in mapping.values())):
            raise ValueError("VAE index references an unverified weight shard")
    model = AutoencoderKLWan.from_pretrained(str(path), local_files_only=True, use_safetensors=True,
                                            torch_dtype=torch.float32)
    return model.eval().requires_grad_(False).to(device)


def read_video(path, target_fps):
    """Deterministic timestamp sampling, RGB, preserving first-to-last order."""
    import cv2
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("demo_fps must be positive")
    video = cv2.VideoCapture(str(path))
    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        if not video.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"unreadable video or missing frame rate: {path}")
        frames, indices, index, next_time = [], [], 0, 0.0
        while True:
            ok, bgr = video.read()
            if not ok:
                break
            if index / fps + 1e-9 >= next_time:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                indices.append(index)
                next_time = (len(frames)) / min(fps, target_fps)
            index += 1
    finally:
        video.release()
    if not frames:
        raise ValueError(f"video has no decoded frames: {path}")
    # Wan's first frame then groups of four: discard only an incomplete tail.
    count = 1 + 4 * ((len(frames) - 1) // 4)
    return np.stack(frames[:count]), {"source_fps": fps, "frame_indices": indices[:count]}


@torch.no_grad()
def encode_rgb(vae, frames, size):
    """RGB uint8 [T,H,W,3] -> normalized Wan [1,C,F,H',W']."""
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("RGB frames must be uint8 [T,H,W,3]")
    if frames.shape[0] % 4 != 1:
        raise ValueError("causal Wan clips must contain 1+4*k frames")
    if len(size) != 2 or any(type(v) is not int or v < 16 or v % 16 for v in size):
        raise ValueError("resize height/width must be positive multiples of 16")
    device = next(vae.parameters()).device
    pixels = torch.from_numpy(frames.copy()).to(device).permute(0, 3, 1, 2).float()
    pixels = F.interpolate(pixels, size=tuple(size), mode="bilinear", align_corners=False)
    pixels = (pixels / 127.5 - 1).permute(1, 0, 2, 3).unsqueeze(0)
    latent = vae.encode(pixels).latent_dist.mode().float()
    mean = torch.as_tensor(vae.config.latents_mean, device=device).reshape(1, -1, 1, 1, 1)
    std = torch.as_tensor(vae.config.latents_std, device=device).reshape(1, -1, 1, 1, 1)
    if (std <= 0).any() or not torch.isfinite(std).all():
        raise ValueError("invalid native VAE latent scale")
    latent = (latent - mean) / std
    if not torch.isfinite(latent).all():
        raise ValueError("non-finite VAE output")
    return latent.cpu()


def robot_features(latents, masks, ids):
    """Pool fixed entity tracks without task input; retain declared camera order.

    masks is [F,N,C,H,W] at latent endpoint times. ROI absence is rejected:
    inventing a feature for a fully unobserved entity would hide a state-estimation
    dependency. Entity IDs must remain stable across the complete clip.
    """
    if (not latents or any(x.ndim != 5 or x.shape[0] != 1 or not x.is_floating_point()
                          or not torch.isfinite(x).all() for x in latents)
            or any(x.shape != latents[0].shape for x in latents)):
        raise ValueError("robot cameras must have identical encoded shapes")
    _, channels, frames, height, width = latents[0].shape
    masks = torch.as_tensor(masks)
    ids = torch.as_tensor(ids)
    if ids.dtype != torch.int64 or ids.ndim != 1 or (ids < 0).any() or ids.unique().numel() != ids.numel():
        raise ValueError("visual tracks require unique nonnegative stable entity IDs")
    if masks.ndim != 5 or masks.shape[:3] != (frames, len(ids), len(latents)):
        raise ValueError("track masks must be [latent_frames,entities,cameras,height,width]")
    if not torch.isfinite(masks).all() or (masks < 0).any() or (masks > 1).any():
        raise ValueError("track masks must be finite values in [0,1]")
    flat = masks.float().reshape(-1, 1, *masks.shape[-2:])
    resized = F.interpolate(flat, size=(height, width), mode="area")
    resized = resized.reshape(frames, len(ids), len(latents), height, width)
    # [F,N,H,C*W] matches the native camera-width concatenation exactly.
    weights = resized.permute(0, 1, 3, 2, 4).flatten(3).flatten(2)
    mass = weights.sum(-1, keepdim=True)
    if (mass <= 0).any():
        raise ValueError("every entity needs observed ROI mass at every encoded history time")
    weights = weights / mass
    packed = torch.cat(latents, dim=-1)
    pixels = packed[0].permute(1, 2, 3, 0).reshape(frames, -1, channels)
    return packed[0].numpy(), torch.einsum("fns,fsc->fnc", weights, pixels).numpy()


def preprocess_video(manifest_path, output, *, device="cuda", vae=None):
    """Encode a screened single video into independent, unpaired feature windows.

    This produces observed patch features, not actions, task roles or inferred
    contact/effect labels. Source-group identity stays the same in every window.
    """
    from .cli import write_json
    from .video_data import load_video_index, load_video_window
    path, output = Path(manifest_path).resolve(), Path(output).resolve()
    spec = json.loads(path.read_text())
    required = {"format_version", "kind", "video", "vae_path", "vae_sha256", "size", "fps",
                "domain", "source_id", "source_group", "feature_space_id", "split",
                "window_frames", "context_frames", "continuous_segment_verified"}
    if not isinstance(spec, dict) or set(spec) - required - {"window_stride_frames", "trajectory_id"} or required - set(spec):
        raise ValueError("raw_video_pretrain must explicitly identify its local continuous clip, encoder and window policy")
    if type(spec["format_version"]) is not int or spec["format_version"] != 1 or spec["kind"] != "raw_video_pretrain":
        raise ValueError("expected raw_video_pretrain version 1")
    if spec["continuous_segment_verified"] is not True:
        raise ValueError("input must be a screened continuous clip; scene cuts are not detected automatically")
    if spec["domain"] not in {"human", "robot"} or spec["split"] not in {"train", "validation", "test"}:
        raise ValueError("declare human/robot domain and train/validation/test split")
    if type(spec["fps"]) not in (int, float) or not np.isfinite(spec["fps"]) or spec["fps"] <= 0:
        raise ValueError("fps must be a positive finite number")
    for name in ("source_id", "source_group", "feature_space_id", "video", "vae_path"):
        if not isinstance(spec[name], str) or not spec[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    if "trajectory_id" in spec and (spec["domain"] != "robot" or not isinstance(spec["trajectory_id"], str) or not spec["trajectory_id"].strip()):
        raise ValueError("trajectory_id is only an explicit nonempty robot trajectory identity")
    window, context = spec["window_frames"], spec["context_frames"]
    stride = spec.get("window_stride_frames", window)
    if (type(window) is not int or window < 3 or type(context) is not int or not 1 <= context <= window - 2
            or type(stride) is not int or not 1 <= stride <= window):
        raise ValueError("windows require >=1 context and >=2 future latent frames; stride must be in [1,window_frames]")
    if not isinstance(spec["vae_sha256"], dict) or not spec["vae_sha256"]:
        raise ValueError("declare the frozen VAE file identities")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("video preprocessing output must be a fresh directory")
    video = (path.parent / spec["video"]).resolve()
    if not video.is_file():
        raise ValueError("video must name an existing local file")
    frames, sampling = read_video(video, spec["fps"])
    injected = vae is not None
    if vae is None:
        vae = load_vae((path.parent / spec["vae_path"]).resolve(), spec["vae_sha256"], device=device)
    encoded = encode_rgb(vae, frames, spec["size"])
    features = encoded[0].permute(1, 2, 3, 0).flatten(1, 2).numpy()  # [T,H*W,C]
    endpoints = np.asarray(sampling["frame_indices"], dtype=np.int64)[::4]
    times = endpoints.astype(np.float64) / sampling["source_fps"]
    if len(features) != len(times) or len(features) < window:
        raise ValueError("continuous clip is too short for one complete context-plus-two-futures window")
    starts = list(range(0, len(features) - window + 1, stride))
    common = {"manifest_sha256": sha256(path), "source_video_sha256": sha256(video),
              "vae_sha256": spec["vae_sha256"], "feature_encoder": "frozen-wan-vae",
              "latent_normalization": "(posterior_mode-mean)/std", "token_order": "time,height,width,channel",
              "source_fps": sampling["source_fps"], "requested_fps": spec["fps"], "size": spec["size"],
              "frame_time_basis": "source_frame_index/source_fps", "encoding_policy": "whole-clip-causal-before-windowing",
              "continuous_segment_verified": True, "injected_test_encoder": injected,
              "window_policy": {"window_frames": window, "context_frames": context, "stride_frames": stride,
                                "incomplete_tail": "discard", "discarded_tail_frames": len(features) - starts[-1] - window}}
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for number, start in enumerate(starts):
        stop = start + window
        name = f"window_{number:06d}"
        metadata = {"format_version": 1, "kind": "video_pretrain", "arrays": f"{name}.npz",
                    "sample_id": f"{spec['source_id']}:{start}:{stop}", "source_id": spec["source_id"],
                    "source_group": spec["source_group"], "domain": spec["domain"],
                    "feature_space_id": spec["feature_space_id"], "feature_kind": "patches", "context_frames": context,
                    "provenance": {**common, "latent_frame_slice": [start, stop],
                                   "latent_endpoint_source_frame_indices": endpoints[start:stop].tolist()}}
        if "trajectory_id" in spec:
            metadata["trajectory_id"] = spec["trajectory_id"]
        np.savez_compressed(output / f"{name}.npz", features=features[start:stop],
                            feature_valid=np.ones_like(features[start:stop], dtype=np.bool_), frame_times=times[start:stop])
        write_json(output / f"{name}.json", metadata)
        load_video_window(output / f"{name}.json")
        records.append({"manifest": f"{name}.json", "split": spec["split"]})
    write_json(output / "index.json", {"format_version": 1, "kind": "video_pretrain_index", "samples": records,
                                        "bridge_sources": []})
    load_video_index(output / "index.json", split=spec["split"])
    return {"index": str(output / "index.json"), "windows": len(records), "source_group": spec["source_group"],
            "feature_space_id": spec["feature_space_id"], "feature_dim": features.shape[-1],
            "tokens_per_frame": features.shape[1], "released_vae_run": not injected,
            "automatic_effect_labels": False, "commands_sent": 0}


def preprocess(manifest_path, output, *, device="cuda", vae=None):
    """Write a v2 observed-only input from raw robot RGB, tracks and human video."""
    from .cli import write_json
    from .data import load_observation
    path, output = Path(manifest_path).resolve(), Path(output)
    spec = json.loads(path.read_text())
    if spec.get("format_version") != 1 or spec.get("kind") != "raw_visual_observation":
        raise ValueError("expected a raw_visual_observation version-1 manifest")
    if output.exists() and any(output.iterdir()):
        raise ValueError("visual preprocessing output must be a fresh directory")
    cameras = spec.get("camera_order")
    if not isinstance(cameras, list) or not cameras or len(set(cameras)) != len(cameras):
        raise ValueError("provide a fixed, unique camera_order")
    if spec.get("entity_source") not in {"visual_tracks", "simulator_oracle"}:
        raise ValueError("entity_source must distinguish visual_tracks and simulator_oracle")
    if not spec.get("tracker_identity") or not spec.get("coordinate_frame"):
        raise ValueError("track provenance and coordinate_frame are required")
    source = (path.parent / spec["arrays"]).resolve()
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    expected = {"entity_ids", "entity_masks", "proprio_history", "embodiment", "observed_action_history",
                "observed_action_step_offsets", "rgb_step_offsets"} | {f"rgb_{c}" for c in cameras}
    if set(arrays) != expected:
        raise ValueError("raw NPZ must contain exactly observed RGB, tracks, proprioception and executed actions")
    offsets = arrays["rgb_step_offsets"]
    if offsets.dtype != np.int64 or offsets.ndim != 1 or not len(offsets) or (offsets > 0).any() or (np.diff(offsets) <= 0).any():
        raise ValueError("RGB timestamps must be strictly increasing nonpositive control-step offsets")
    if offsets[-1] != 0 or any(len(arrays[f"rgb_{c}"]) != len(offsets) for c in cameras):
        raise ValueError("all cameras must be synchronized and end at the current observation")
    # Weights are supplied on the server; tests may inject an explicit small VAE.
    injected = vae is not None
    if vae is None:
        vae = load_vae(spec["vae_path"], spec["vae_sha256"], device=device)
    latents = [encode_rgb(vae, arrays[f"rgb_{c}"], spec["robot_size"]) for c in cameras]
    packed, features = robot_features(latents, arrays["entity_masks"], arrays["entity_ids"])
    encoded_offsets = offsets[::4]
    if features.shape[0] != len(encoded_offsets) or arrays["proprio_history"].shape[0] != len(encoded_offsets):
        raise ValueError("proprioception and track masks must align with causal latent endpoint times")
    result = {key: arrays[key] for key in ("entity_ids", "proprio_history", "embodiment", "observed_action_history", "observed_action_step_offsets")}
    result.update(robot_latent=packed, robot_history=features, observed_video_step_offsets=encoded_offsets)
    demo_records, demo_layouts = [], []
    demo_encoder, demo_payload, demonstration_encoding = None, None, {"kind": "raw_features"}
    artifact_path = spec.get("demo_encoder_artifact")
    if artifact_path is not None:
        from .video_cli import encoding_metadata, load_video_encoder
        if not isinstance(artifact_path, str) or not artifact_path.strip():
            raise ValueError("demo_encoder_artifact must name a local artifact")
        artifact_path = (path.parent / artifact_path).resolve()
        artifact_sha = sha256(artifact_path)
        if spec.get("demo_encoder_sha256", artifact_sha) != artifact_sha:
            raise ValueError("demonstration encoder artifact SHA256 mismatch")
        demo_encoder, demo_payload = load_video_encoder(artifact_path, device=device)
        if not spec.get("demo_feature_space_id") or spec["demo_feature_space_id"] != demo_payload["feature_space_id"]:
            raise ValueError("demonstration feature space must match the video encoder artifact")
        demo_encoder.eval().requires_grad_(False)
        demonstration_encoding = encoding_metadata(demo_payload, artifact_sha)
    for i, demo in enumerate(spec["demonstrations"]):
        video = (path.parent / demo["video"]).resolve()
        frames, sampling = read_video(video, spec["demo_fps"])
        encoded = encode_rgb(vae, frames, spec["demo_size"])
        features = encoded.permute(0, 2, 3, 4, 1).flatten(2, 3)
        times = torch.as_tensor(np.asarray(sampling["frame_indices"])[::4] / sampling["source_fps"], dtype=torch.float64)
        if features.shape[1] != len(times):
            raise ValueError("demonstration latent endpoints and sampled source times differ")
        if demo_encoder is None:
            result[f"demo_view_{i}"] = features.flatten(1, 2)[0].numpy()
            demo_layouts.append({"frames": features.shape[1], "tokens_per_frame": features.shape[2],
                                 "frame_times": times.tolist()})
        else:
            features = features.to(next(demo_encoder.parameters()))
            with torch.no_grad():
                tokens = demo_encoder.encode_demo(features, torch.ones_like(features, dtype=torch.bool),
                                                   times.to(features.device), window_frames=demo_payload["config"]["window_frames"])
            result[f"demo_view_{i}"] = tokens[0].cpu().float().numpy()
        demo_records.append({"view_id": demo["view_id"], "source_sha256": sha256(video), **sampling})
    identity = {"manifest_sha256": sha256(path), "robot_arrays_sha256": sha256(source),
                "vae_sha256": spec["vae_sha256"], "demonstrations": demo_records,
                "camera_order": cameras, "entity_source": spec["entity_source"],
                "tracker_identity": spec["tracker_identity"], "coordinate_frame": spec["coordinate_frame"],
                "latent_normalization": "(posterior_mode-mean)/std", "token_order": "time,height,camera,width",
                "injected_test_encoder": injected, "demonstration_encoding": demonstration_encoding}
    if demo_encoder is not None:
        identity["demonstration_window_policy"] = {"window_frames": demo_payload["config"]["window_frames"],
            "context_frames": demo_payload["config"]["context_frames"], "stride": "nonoverlapping",
            "incomplete_tail": "repeat_last_values_with_invalid_padding", "noise": False}
    identity["cache_identity"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    meta = {key: spec[key] for key in ("chunk_size", "actions_per_frame", "action_space", "observed_action_space",
                                      "observation_step", "control_dt", "history_chunks")}
    meta.update(format_version=2, kind="observation", arrays="observation.npz",
                view_ids=[d["view_id"] for d in spec["demonstrations"]], visual_provenance=identity,
                demonstration_encoding=demonstration_encoding)
    if demo_layouts:
        meta["demonstration_layouts"] = demo_layouts
    if "demo_feature_space_id" in spec:
        meta["demo_feature_space_id"] = spec["demo_feature_space_id"]
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "observation.npz", **result)
    write_json(output / "observation.json", meta)
    load_observation(output / "observation.json")
    return {"manifest": str((output / "observation.json").resolve()), "cache_identity": identity["cache_identity"],
            "entity_source": spec["entity_source"], "released_vae_run": not injected, "commands_sent": 0}
