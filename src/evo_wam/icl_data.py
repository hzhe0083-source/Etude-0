"""Native Wan context–target samples; human videos carry no robot labels."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor

from .video_data import SPLITS, _local_path, _masked_values, _source_components, _text


LATENT_NORMALIZATION = "(posterior_mode-mean)/std"
_IDENTITY_FIELDS = {"source_id", "source_group", "domain", "trajectory_id"}


@dataclass(frozen=True)
class NativeICLSample:
    metadata: Mapping
    demonstration: Tensor          # [1,C,F,H,W], complete context video
    target: Tensor                 # [1,C,F,H,W], clean teacher/noising target
    demonstration_times: Tensor     # [F], seconds
    target_times: Tensor            # [F], seconds
    history_frames: int             # target[:history_frames] is observed history
    actions: Tensor | None = None   # robot only: [1,A,F,N,1]
    actions_mask: Tensor | None = None


def _identity(record: Mapping) -> None:
    if not isinstance(record, dict):
        raise ValueError("video identities must be objects")
    for key in ("source_id", "source_group"):
        _text(record, key)
    if record.get("domain") not in {"human", "robot"}:
        raise ValueError("video domain must be human or robot")
    if "trajectory_id" in record:
        _text(record, "trajectory_id")
        if record["domain"] != "robot":
            raise ValueError("human videos cannot declare robot trajectory_id")


def _metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    required = {"format_version", "kind", "sample_id", "feature_space_id", "latent_normalization",
                "history_frames", "demonstration", "target", "compatibility"}
    if (not isinstance(metadata, dict) or required - set(metadata)
            or set(metadata) - required - {"provenance"}
            or type(metadata.get("format_version")) is not int or metadata["format_version"] != 1
            or metadata.get("kind") != "native_icl_sample"):
        raise ValueError("expected an explicit version-1 native_icl_sample schema")
    for key in ("sample_id", "feature_space_id"):
        _text(metadata, key)
    if metadata["latent_normalization"] != LATENT_NORMALIZATION:
        raise ValueError(f"latent_normalization must be {LATENT_NORMALIZATION!r}")
    if type(metadata["history_frames"]) is not int or metadata["history_frames"] < 1:
        raise ValueError("history_frames must be a positive integer in latent frames")
    if "provenance" in metadata and not isinstance(metadata["provenance"], dict):
        raise ValueError("provenance must be an object, never an additional model input")
    compatibility = metadata["compatibility"]
    if (not isinstance(compatibility, dict) or set(compatibility) != {"kind", "evidence"}
            or compatibility["kind"] != "audited_semantic_task"):
        raise ValueError("compatibility requires audited_semantic_task and explicit evidence; videos need not be synchronized")
    _text(compatibility, "evidence")
    for role in ("demonstration", "target"):
        record = metadata[role]
        _identity(record)
        allowed = _IDENTITY_FIELDS | {"arrays"}
        if role == "target" and record["domain"] == "robot":
            from .cli import action_space
            allowed |= {"action_space"}
            space = record.get("action_space")
            action_space(record, space.get("dimension") if isinstance(space, dict) else None)
        if set(record) - allowed:
            raise ValueError("native video records accept only source identity, arrays and robot-target action_space")
        _local_path(path.parent, _text(record, "arrays"), ".npz")
    demo, target = metadata["demonstration"], metadata["target"]
    if target["domain"] == "human":
        if demo["domain"] != "human":
            raise ValueError("human cross-video ICL requires a human demonstration")
        if (demo["source_id"] == target["source_id"] or demo["source_group"] == target["source_group"]
                or _local_path(path.parent, demo["arrays"], ".npz") == _local_path(path.parent, target["arrays"], ".npz")):
            raise ValueError("human demonstration and target must be independent videos, not the same source or repost group")
    return metadata


def _arrays(path: Path, record: Mapping, *, robot_target: bool):
    with np.load(_local_path(path.parent, record["arrays"], ".npz"), allow_pickle=False) as archive:
        expected = {"latent", "frame_times"} | ({"actions", "actions_mask"} if robot_target else set())
        if len(archive.files) != len(expected) or set(archive.files) != expected:
            raise ValueError("video NPZ needs exactly latent/frame_times; only robot targets require actions/actions_mask")
        values = {name: torch.from_numpy(archive[name].copy()) for name in expected}
    latent, times = values["latent"], values["frame_times"]
    if latent.ndim != 4 or min(latent.shape) < 1 or not latent.is_floating_point() or not torch.isfinite(latent).all():
        raise ValueError("native latent must be finite floating nonempty [C,F,H,W]")
    if (not times.is_floating_point() or times.shape != (latent.shape[1],)
            or not torch.isfinite(times).all() or (times[1:] <= times[:-1]).any()):
        raise ValueError("frame_times must be finite floating [F] strictly increasing seconds")
    return values


def load_icl_sample(manifest_path: str | Path) -> NativeICLSample:
    """Load clean videos; native causal masks must keep target future out of history."""
    path = Path(manifest_path)
    metadata = _metadata(path)
    robot = metadata["target"]["domain"] == "robot"
    demo = _arrays(path, metadata["demonstration"], robot_target=False)
    target = _arrays(path, metadata["target"], robot_target=robot)
    context, frames = metadata["history_frames"], target["latent"].shape[1]
    if context >= frames:
        raise ValueError("history_frames must leave at least one future latent frame")
    if demo["latent"].shape[0] != target["latent"].shape[0]:
        raise ValueError("demonstration and target must share the same latent channel dimension")
    actions, mask = None, None
    if robot:
        from .cli import action_space
        actions, mask = target["actions"], target["actions_mask"]
        if (actions.ndim != 4 or min(actions.shape) < 1 or actions.shape[1] != frames
                or actions.shape[-1] != 1):
            raise ValueError("robot actions must be [A,F,N,1] aligned with target latent frames")
        space = action_space(metadata["target"], actions.shape[0])
        if mask.dtype != torch.bool or mask.shape != actions.shape:
            raise ValueError("actions_mask must be Boolean with exactly the actions shape")
        mask = mask & torch.tensor(space["valid_channels"])[:, None, None, None]
        actions = _masked_values(actions, mask, "actions")
        if not mask[:, context:].any():
            raise ValueError("robot targets require valid future action supervision")
        actions, mask = actions.unsqueeze(0), mask.unsqueeze(0)
    return NativeICLSample(metadata, demo["latent"].unsqueeze(0), target["latent"].unsqueeze(0),
                           demo["frame_times"], target["frame_times"], context, actions, mask)


def load_icl_index(index_path: str | Path, split: str = "train") -> tuple[list[Path], list[dict]]:
    """Audit both video roles and transitive source aliases without reading arrays."""
    if split not in SPLITS:
        raise ValueError("split must be train, validation or test")
    path = Path(index_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(document, dict) or type(document.get("format_version")) is not int
            or document["format_version"] != 1 or document.get("kind") != "native_icl_index"
            or set(document) - {"format_version", "kind", "samples", "source_aliases", "bridge_sources"}):
        raise ValueError("expected a version-1 native_icl_index")
    entries, aliases, bridges = (document.get("samples"), document.get("source_aliases", []),
                                  document.get("bridge_sources", []))
    if not isinstance(entries, list) or not entries or not isinstance(aliases, list) or not isinstance(bridges, list):
        raise ValueError("native ICL index needs nonempty samples and optional source_aliases/bridge_sources lists")
    selected, records, sample_ids, spaces, human_pairs = [], [], set(), set(), []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"manifest", "split"}:
            raise ValueError("native ICL index samples require exactly manifest and split")
        manifest = _local_path(path.parent, entry["manifest"], ".json")
        metadata = _metadata(manifest)
        if metadata["sample_id"] in sample_ids:
            raise ValueError("sample_id must be unique across the native ICL index")
        sample_ids.add(metadata["sample_id"])
        spaces.add(metadata["feature_space_id"])
        if metadata["target"]["domain"] == "human":
            human_pairs.append((len(records), len(records) + 1))
        for role in ("demonstration", "target"):
            source = metadata[role]
            record = {key: source[key] for key in _IDENTITY_FIELDS if key in source}
            records.append({**record, "sample_id": metadata["sample_id"], "feature_space_id": metadata["feature_space_id"],
                            "record_kind": "video", "split": entry["split"]})
        if entry["split"] == split:
            selected.append(manifest)
    if len(spaces) != 1:
        raise ValueError("one native ICL index must use one fixed feature_space_id")
    for alias in aliases:
        if not isinstance(alias, dict) or set(alias) - _IDENTITY_FIELDS - {"split"}:
            raise ValueError("source_aliases contain only video source identities and split")
        _identity(alias)
        records.append({**alias, "record_kind": "video"})
    for bridge in bridges:
        if not isinstance(bridge, dict) or set(bridge) - {"source_id", "source_group", "trajectory_id", "split"}:
            raise ValueError("bridge_sources contain only source/trajectory identities and split")
        records.append({**bridge, "record_kind": "bridge"})
    _source_components(records)  # Full provenance closure still governs split isolation.
    videos = [(index, {key: value for key, value in record.items() if key != "trajectory_id"})
              for index, record in enumerate(records) if record["record_kind"] == "video"]
    # Shared robot supervision does not establish that two human clips are reposts.
    components = {videos[index][0]: number for number, component in
                  enumerate(_source_components([record for _, record in videos])) for index in component}
    if any(components[demo] == components[target] for demo, target in human_pairs):
        raise ValueError("human demonstration and target must be independent source components, including repost aliases")
    if not selected:
        raise ValueError(f"native ICL dataset has no {split} samples")
    return selected, records
