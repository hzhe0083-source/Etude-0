"""Cache local RGB for native ICL without actions, tracking or effect encoders."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .vision import encode_rgb, load_vae, read_video, sha256


def encode_g_pi_frames(vae, frames, control_times, gripper, *, frame_stride, control_dt,
                       event_rules, size, subgoal_metadata=None, offline_arrays=None):
    """Encode native causal history and each event/terminal image separately.

    RGB contains every control-grid frame, including the incomplete video tail.
    Returns visual NPZ fields and their alignment metadata for a v3 g_pi_task.
    """
    from .g_pi_data import EventRules, subgoal_control_indices, validate_latent_grid

    if (not isinstance(frames, np.ndarray) or frames.dtype != np.uint8 or frames.ndim != 4
            or frames.shape[-1] != 3 or min(frames.shape) < 1 or len(frames) < 2):
        raise ValueError("G/pi RGB must be uint8 [T_c,H,W,3] including every control step")
    times = torch.as_tensor(control_times).detach().cpu()
    measured = torch.as_tensor(gripper).detach().cpu()
    if times.shape != (len(frames),):
        raise ValueError("control_times must identify every supplied RGB frame")
    rules = EventRules.from_metadata(event_rules) if isinstance(event_rules, dict) else event_rules
    if not isinstance(rules, EventRules):
        raise ValueError("event_rules must contain the measured gripper detection rules")
    if type(frame_stride) is not int or frame_stride < 1:
        raise ValueError("frame_stride must be a positive integer")
    metadata = {"frame_stride": frame_stride, "temporal_down_rate": 4, "control_dt": control_dt,
                "actions_per_frame": 4 * frame_stride, "alignment": "zerowam_causal_first_then_four",
                "subgoal_encoding": "wan_vae_single_frame"}
    endpoints = torch.arange(0, len(frames), 4 * frame_stride)
    available = times[endpoints]
    validate_latent_grid(metadata, times, available)
    if subgoal_metadata is None:
        subgoals = subgoal_control_indices(measured, times, rules)
    else:
        from dataclasses import asdict
        from .g_pi_subgoals import resolve_subgoal_indices

        evidence = {name: torch.as_tensor(value) for name, value in (offline_arrays or {}).items()}
        evidence.update(control_times=times, gripper=measured)
        subgoals, audit = resolve_subgoal_indices({**subgoal_metadata, "event_rules": asdict(rules)}, evidence)
        metadata.update(subgoal_source=subgoal_metadata.get("subgoal_source", "gripper"), subgoal_annotation=audit)
    # Discard incomplete groups only from the sequence, never from target RGB.
    sampled = frames[:endpoints[-1].item() + 1:frame_stride]
    latent = encode_rgb(vae, sampled, size)
    targets = [encode_rgb(vae, frames[index:index + 1], size) for index in subgoals.tolist()]
    if (latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[2] != len(endpoints)
            or min(latent.shape) < 1 or not latent.is_floating_point() or not torch.isfinite(latent).all()):
        raise ValueError("Wan history must return finite [1,C,T_l,H,W] on the causal latent grid")
    expected = (1, latent.shape[1], 1, *latent.shape[-2:])
    if any(target.shape != expected or not target.is_floating_point() or not torch.isfinite(target).all()
           for target in targets):
        raise ValueError("each subgoal must be independently encoded as finite [1,C,1,H,W]")
    arrays = {"latent": latent[0].float().numpy(), "latent_available_times": available.numpy(),
              "subgoal_latents": torch.cat(targets, dim=0).float().numpy(),
              "subgoal_times": times[subgoals].numpy()}
    return arrays, metadata


def preprocess_icl_video(manifest_path, output, *, device="cuda", vae=None):
    """Encode one continuous clip; semantic A/B pairing remains an audited input."""
    from .cli import write_json

    path, output = Path(manifest_path).resolve(), Path(output).resolve()
    spec = json.loads(path.read_text())
    required = {"format_version", "kind", "video", "vae_path", "vae_sha256", "size", "fps",
                "source_id", "source_group", "domain", "feature_space_id", "continuous_segment_verified"}
    if not isinstance(spec, dict) or required - set(spec) or set(spec) - required - {"trajectory_id"}:
        raise ValueError("raw_icl_video must identify one local continuous RGB clip and its frozen encoder")
    if type(spec["format_version"]) is not int or spec["format_version"] != 1 or spec["kind"] != "raw_icl_video":
        raise ValueError("expected raw_icl_video version 1")
    if spec["continuous_segment_verified"] is not True:
        raise ValueError("input must be a screened continuous clip; scene cuts are not detected automatically")
    if spec["domain"] not in ("human", "robot"):
        raise ValueError("declare human or robot domain")
    for name in ("video", "vae_path", "source_id", "source_group", "feature_space_id"):
        if not isinstance(spec[name], str) or not spec[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    if any("://" in spec[name] for name in ("video", "vae_path")):
        raise ValueError("video and VAE must use local paths")
    if "trajectory_id" in spec and (spec["domain"] != "robot" or not isinstance(spec["trajectory_id"], str)
                                    or not spec["trajectory_id"].strip()):
        raise ValueError("trajectory_id is only an explicit nonempty robot trajectory identity")
    if type(spec["fps"]) not in (int, float) or not np.isfinite(spec["fps"]) or spec["fps"] <= 0:
        raise ValueError("fps must be a positive finite number")
    if (not isinstance(spec["size"], list) or len(spec["size"]) != 2
            or any(type(v) is not int or v < 16 or v % 16 for v in spec["size"])):
        raise ValueError("resize height/width must be positive multiples of 16")
    identity = spec["vae_sha256"]
    if (not isinstance(identity, dict) or "config.json" not in identity
            or any(not isinstance(name, str) or Path(name).name != name or not isinstance(digest, str)
                   or len(digest) != 64 or set(digest) - set("0123456789abcdef")
                   for name, digest in identity.items())):
        raise ValueError("declare a SHA256 map covering the frozen VAE config and weights")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("ICL preprocessing output must be a fresh directory")
    video = (path.parent / spec["video"]).resolve()
    if not video.is_file():
        raise ValueError("video must name an existing local file")
    injected = vae is not None
    if vae is None:
        vae = load_vae((path.parent / spec["vae_path"]).resolve(), identity, device=device)
    frames, sampling = read_video(video, spec["fps"])
    indices = np.asarray(sampling["frame_indices"])
    source_fps = sampling["source_fps"]
    if (type(source_fps) not in (int, float) or not np.isfinite(source_fps) or source_fps <= 0
            or indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or len(indices) != len(frames)
            or not len(indices) or indices[0] != 0 or (np.diff(indices) <= 0).any()):
        raise ValueError("sampled source frame indices and frame rate must define ordered RGB timestamps")
    # encode_rgb already applies the native per-channel normalization exactly once.
    encoded = encode_rgb(vae, frames, spec["size"])
    times = indices[::4].astype(np.float64) / source_fps
    if (not isinstance(encoded, torch.Tensor) or encoded.ndim != 5 or encoded.shape[0] != 1
            or min(encoded.shape) <= 0 or not encoded.is_floating_point() or not torch.isfinite(encoded).all()
            or encoded.shape[2] != len(times) or not np.isfinite(times).all() or (np.diff(times) <= 0).any()):
        raise ValueError("Wan latent must be finite [1,C,F,H,W] with one time per causal endpoint")
    latent = encoded[0].detach().float().cpu().numpy()
    metadata = {"format_version": 1, "kind": "native_icl_clip", "arrays": "clip.npz",
                **{key: spec[key] for key in ("source_id", "source_group", "domain", "feature_space_id")},
                "provenance": {"manifest_sha256": sha256(path), "source_video_sha256": sha256(video),
                               "vae_sha256": identity, "feature_encoder": "frozen-wan-vae",
                               "latent_normalization": "(posterior_mode-mean)/std", "latent_layout": "channel,time,height,width",
                               "source_fps": source_fps, "requested_fps": spec["fps"], "size": spec["size"],
                               "frame_indices": indices.tolist(), "latent_endpoint_source_frame_indices": indices[::4].tolist(),
                               "frame_time_basis": "source_frame_index/source_fps",
                               "encoding_policy": "whole-clip-causal", "continuous_segment_verified": True,
                               "injected_test_encoder": injected}}
    if "trajectory_id" in spec:
        metadata["trajectory_id"] = spec["trajectory_id"]
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "clip.npz", latent=latent, frame_times=times)
    write_json(output / "clip.json", metadata)
    return {"manifest": str(output / "clip.json"), "arrays": str(output / "clip.npz"), "shape": list(latent.shape),
            "source_group": spec["source_group"], "feature_space_id": spec["feature_space_id"],
            "released_vae_run": not injected, "commands_sent": 0}
