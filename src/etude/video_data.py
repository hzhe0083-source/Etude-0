"""Unpaired observed-video windows; no robot action or cross-domain pair is inferred."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


EFFECT_FIELDS = ("geometry", "relations", "events")
EFFECT_METADATA = ("effect_schema_id", "geometry_frame", "geometry_units", "evidence_source")
SPLITS = {"train", "validation", "test"}
PATCH_COORDINATE_SYSTEM = "normalized_xy_patch_centers"
_BASE_METADATA = {
    "format_version", "kind", "arrays", "sample_id", "source_id", "source_group",
    "domain", "feature_space_id", "feature_kind", "context_frames", "patch_grid",
    "patch_coordinate_system",
}


def _patch_grid(grid: Sequence[int]) -> tuple[int, int]:
    if (not isinstance(grid, (list, tuple)) or len(grid) != 2
            or any(type(size) is not int or size < 1 for size in grid)):
        raise ValueError("patch_grid must explicitly provide positive integer [height, width]; regenerate from the visual encoder, never infer it from N")
    return tuple(grid)


def patch_grid_coordinates(grid: Sequence[int]) -> Tensor:
    """Normalized (x,y) patch centers in canonical row-major (y,x) order."""
    height, width = _patch_grid(grid)
    y, x = torch.meshgrid((torch.arange(height, dtype=torch.float32) + 0.5) * (2 / height) - 1,
                          (torch.arange(width, dtype=torch.float32) + 0.5) * (2 / width) - 1,
                          indexing="ij")
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


def validate_patch_coordinates(coordinates: Tensor, grid: Sequence[int], coordinate_system: str) -> Tensor:
    """Validate complete grid coverage while preserving the NPZ's storage order."""
    if coordinate_system != PATCH_COORDINATE_SYSTEM:
        raise ValueError(f"patch_coordinate_system must be {PATCH_COORDINATE_SYSTEM!r}")
    height, width = _patch_grid(grid)
    if (not isinstance(coordinates, Tensor) or not coordinates.is_floating_point()
            or coordinates.shape != (height * width, 2) or not torch.isfinite(coordinates).all()
            or (coordinates.abs() >= 1).any()):
        raise ValueError("patch_coordinates must be finite floating [N,2] centers for the explicit patch_grid")
    coordinates = coordinates.float()
    cells = ((coordinates + 1) * coordinates.new_tensor([width, height]) / 2 - 0.5).round().long()
    indices = cells[:, 1] * width + cells[:, 0]
    expected = patch_grid_coordinates(grid).to(coordinates.device)
    if (indices.min() < 0 or indices.max() >= height * width or indices.unique().numel() != height * width
            or not torch.allclose(coordinates, expected[indices], atol=1e-6, rtol=0)):
        raise ValueError("patch_coordinates must contain each explicit patch_grid center exactly once")
    return coordinates


def _text(record: Mapping, name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be an explicit nonempty string")
    return value


def _local_path(parent: Path, name: str, suffix: str) -> Path:
    if not isinstance(name, str) or not name:
        raise ValueError(f"expected a relative {suffix} path")
    path = (parent / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(parent.resolve()) or path.suffix != suffix:
        raise ValueError(f"{suffix} paths must remain within the containing directory")
    return path


def _metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(metadata, dict) or type(metadata.get("format_version")) is not int
            or metadata["format_version"] != 2 or metadata.get("kind") != "video_pretrain"):
        raise ValueError("expected a version-2 video_pretrain window; regenerate old windows from the visual encoder with explicit patch_grid and patch_coordinates (never infer a grid from N)")
    allowed = _BASE_METADATA | set(EFFECT_METADATA) | {"trajectory_id", "provenance"}
    if set(metadata) - allowed:
        raise ValueError("video windows accept only their explicit unpaired schema; put source details in provenance")
    for name in ("arrays", "sample_id", "source_id", "source_group", "feature_space_id"):
        _text(metadata, name)
    if metadata.get("domain") not in {"human", "robot"}:
        raise ValueError("domain must be human or robot")
    if metadata.get("feature_kind") not in {"patches", "tracked_entities"}:
        raise ValueError("feature_kind must be patches or tracked_entities")
    if metadata["feature_kind"] == "patches":
        _patch_grid(metadata.get("patch_grid"))
        if metadata.get("patch_coordinate_system") != PATCH_COORDINATE_SYSTEM:
            raise ValueError(f"patch_coordinate_system must explicitly be {PATCH_COORDINATE_SYSTEM!r}")
    elif {"patch_grid", "patch_coordinate_system"} & set(metadata):
        raise ValueError("tracked_entities cannot declare patch grid metadata; entity storage IDs are not positions")
    if type(metadata.get("context_frames")) is not int or metadata["context_frames"] < 1:
        raise ValueError("context_frames must be a positive integer")
    if "trajectory_id" in metadata:
        _text(metadata, "trajectory_id")
        if metadata["domain"] != "robot":
            raise ValueError("a human video cannot declare an arbitrary paired robot trajectory")
    if "provenance" in metadata and not isinstance(metadata["provenance"], dict):
        raise ValueError("provenance must be an object, never an additional model input")
    for name in EFFECT_METADATA:
        if name in metadata:
            _text(metadata, name)
    return metadata


@dataclass(frozen=True)
class VideoWindow:
    metadata: Mapping
    features: Tensor                  # float [1,T,N,D], invalid entries replaced with zero
    feature_valid: Tensor             # bool [1,T,N,D], data-owned observation validity
    frame_times: Tensor               # float [T], strictly increasing seconds
    context_frames: int               # 1..T-2, split between context and observed future
    effect_targets: dict[str, Tensor] # optional geometry [1,T-P,N,G], pairs [1,T-P,N,N,C/E]
    effect_valid: dict[str, Tensor]   # bool, matching each supplied effect target
    entity_ids: Tensor | None = None  # tracked entities only: int64 [1,N]
    patch_coordinates: Tensor | None = None  # patches only: float [N,2], in storage order

    @property
    def has_training_signal(self) -> bool:
        """Skip an update if there is no observed context or no valid future target."""
        context = self.feature_valid[:, :self.context_frames].any()
        future = self.feature_valid[:, self.context_frames:].any()
        effect = any(mask.any().item() for mask in self.effect_valid.values())
        return bool(context and (future or effect))


def _masked_values(values: Tensor, valid: Tensor, name: str) -> Tensor:
    if not values.is_floating_point() or valid.dtype != torch.bool or valid.shape != values.shape:
        raise ValueError(f"{name} needs floating values and an exactly matching Boolean validity mask")
    if not torch.isfinite(values[valid]).all():
        raise ValueError(f"valid {name} entries must be finite")
    # Sanitize before model arithmetic; multiplying a computed NaN by zero is unsafe.
    return torch.where(valid, values, 0)


def load_video_window(manifest_path: str | Path) -> VideoWindow:
    """Load one continuous single-view window, with no action labels or forced pair."""
    path = Path(manifest_path)
    metadata = _metadata(path)
    arrays_path = _local_path(path.parent, metadata["arrays"], ".npz")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
    required = {"features", "feature_valid", "frame_times"}
    allowed = required | {"entity_ids", "patch_coordinates"} | {key for field in EFFECT_FIELDS for key in (field, f"{field}_valid")}
    if not required <= set(arrays) or set(arrays) - allowed:
        raise ValueError("video NPZ needs features/feature_valid/frame_times and only optional patch/entity/effect fields")
    features, valid = arrays["features"], arrays["feature_valid"]
    if features.ndim != 3 or min(features.shape) < 1:
        raise ValueError("features must be nonempty [T,N,D]")
    steps, entities, _ = features.shape
    context = metadata["context_frames"]
    if steps < 3 or not 1 <= context <= steps - 2:
        raise ValueError("a window needs at least one context frame and two future frames")
    features = _masked_values(features, valid, "features")
    times = arrays["frame_times"]
    if (not times.is_floating_point() or times.shape != (steps,) or not torch.isfinite(times).all()
            or (times[1:] <= times[:-1]).any()):
        raise ValueError("frame_times must be finite floating [T] strictly increasing seconds")

    ids = arrays.get("entity_ids")
    coordinates = arrays.get("patch_coordinates")
    tracked = metadata["feature_kind"] == "tracked_entities"
    if tracked:
        if coordinates is not None:
            raise ValueError("tracked_entities cannot use patch_coordinates; entity storage IDs are not positions")
        if (ids is None or ids.dtype != torch.int64 or ids.shape != (entities,)
                or (ids < 0).any() or ids.unique().numel() != entities):
            raise ValueError("tracked_entities requires stable unique nonnegative entity_ids [N]")
    else:
        if ids is not None:
            raise ValueError("patch indices cannot be presented as tracked entity IDs")
        coordinates = validate_patch_coordinates(coordinates, metadata["patch_grid"], metadata["patch_coordinate_system"])
        if coordinates.shape[0] != entities:
            raise ValueError("patch_grid and patch_coordinates must match the feature patch axis N")

    targets, masks = {}, {}
    for field in EFFECT_FIELDS:
        if (field in arrays) != (f"{field}_valid" in arrays):
            raise ValueError(f"{field} values and validity must either both be present or both be absent")
        if field not in arrays:
            continue
        if not tracked:
            raise ValueError("effect labels require tracked_entities, never untracked patches")
        target, mask = arrays[field], arrays[f"{field}_valid"]
        prefix = (steps - context, entities) if field == "geometry" else (steps - context, entities, entities)
        if target.ndim != len(prefix) + 1 or tuple(target.shape[:-1]) != prefix or target.shape[-1] < 1:
            raise ValueError(f"{field} must match the future window and stable entity axes")
        clean = _masked_values(target, mask, field)
        if field != "geometry" and ((clean[mask] != 0) & (clean[mask] != 1)).any():
            raise ValueError(f"valid {field} labels must be binary, not inferred negatives")
        targets[field], masks[field] = clean.unsqueeze(0), mask.unsqueeze(0)
    if targets:
        for name in EFFECT_METADATA:
            _text(metadata, name)
    return VideoWindow(metadata, features.unsqueeze(0), valid.unsqueeze(0), times, context,
                       targets, masks, None if ids is None else ids.unsqueeze(0), coordinates)


def _source_components(records: Sequence[Mapping]) -> list[tuple[int, ...]]:
    """Validate and retain transitive provenance links, including bridge records.

    Video records join original source IDs and repost/window source groups.
    Robot source IDs also identify trajectories unless an explicit trajectory_id
    is provided. Bridge records join their human source and robot trajectory;
    they supply provenance links only, never synthetic training pairs.
    """
    owners: dict[tuple[str, str], int] = {}
    neighbors = [set() for _ in records]
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("split") not in SPLITS:
            raise ValueError("every source record needs an explicit train/validation/test split")
        kind = record.get("record_kind", "video")
        if kind not in {"video", "bridge"}:
            raise ValueError("record_kind must be video or bridge")
        source = _text(record, "source_id")
        keys = [("source", source)]
        if kind == "video":
            keys.append(("group", _text(record, "source_group")))
            if record.get("domain") not in {"human", "robot"}:
                raise ValueError("video source records must identify human or robot domain")
            trajectory = record.get("trajectory_id", source if record["domain"] == "robot" else None)
        else:
            trajectory = _text(record, "trajectory_id")
            if "source_group" in record:
                keys.append(("group", _text(record, "source_group")))
        if trajectory is not None:
            if not isinstance(trajectory, str) or not trajectory.strip():
                raise ValueError("trajectory_id must be a nonempty string")
            keys.append(("trajectory", trajectory))
        for key in keys:
            other = owners.setdefault(key, index)
            if other != index:
                neighbors[index].add(other)
                neighbors[other].add(index)
    seen, components = set(), []
    for start in range(len(records)):
        if start in seen:
            continue
        stack, splits, component = [start], set(), []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            splits.add(records[node]["split"])
            stack.extend(neighbors[node] - seen)
        if len(splits) > 1:
            raise ValueError("video/robot bridge source component crosses train/validation/test splits")
        components.append(tuple(sorted(component)))
    return components


def validate_video_sources(records: Sequence[Mapping]) -> None:
    """Reject train/validation/test overlap through any known source or bridge edge."""
    _source_components(records)


def select_training_sources(records: Sequence[Mapping], domains: Sequence[str]) -> list[dict]:
    """Keep the provenance closure of train videos in the selected domains.

    Bridge/alias records in those components remain necessary for later split
    checks even when they were not model inputs. Independent unused domains are
    excluded. Sampling counts, rather than this closure, describe actual updates.
    """
    if isinstance(domains, str) or set(domains) - {"human", "robot"}:
        raise ValueError("training domains must be a sequence of human/robot names")
    domains = set(domains)
    selected = set()
    for component in _source_components(records):
        if any(records[i].get("record_kind", "video") == "video"
               and records[i]["split"] == "train" and records[i]["domain"] in domains for i in component):
            selected.update(component)
    return [dict(record) for i, record in enumerate(records) if i in selected]


def load_video_index(index_path: str | Path, split: str = "train") -> tuple[list[Path], list[dict]]:
    """Return selected window paths and all source-only records for artifact auditing."""
    if split not in SPLITS:
        raise ValueError("split must be train, validation or test")
    path = Path(index_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(document, dict) or document.get("format_version") != 1
            or document.get("kind") != "video_pretrain_index"
            or set(document) - {"format_version", "kind", "samples", "bridge_sources"}):
        raise ValueError("expected a version-1 video_pretrain_index")
    entries = document.get("samples")
    bridges = document.get("bridge_sources", [])
    if not isinstance(entries, list) or not entries or not isinstance(bridges, list):
        raise ValueError("video index needs nonempty samples and an optional bridge_sources list")
    selected, records, sample_ids = [], [], set()
    feature_spaces, effect_schemas = set(), set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"manifest", "split"}:
            raise ValueError("each video index sample must contain only manifest and split")
        manifest = _local_path(path.parent, entry["manifest"], ".json")
        metadata = _metadata(manifest)
        if metadata["sample_id"] in sample_ids:
            raise ValueError("sample_id must be unique across the video index")
        sample_ids.add(metadata["sample_id"])
        feature_spaces.add(metadata["feature_space_id"])
        if "effect_schema_id" in metadata:
            effect_schemas.add(metadata["effect_schema_id"])
        record = {name: metadata[name] for name in ("sample_id", "source_id", "source_group", "domain", "feature_space_id")}
        for name in ("trajectory_id", "effect_schema_id"):
            if name in metadata:
                record[name] = metadata[name]
        record.update(record_kind="video", split=entry["split"])
        records.append(record)
        if entry["split"] == split:
            selected.append(manifest)
    if len(feature_spaces) != 1 or len(effect_schemas) > 1:
        raise ValueError("one index must use one fixed feature space and a compatible effect vocabulary")
    for bridge in bridges:
        if not isinstance(bridge, dict):
            raise ValueError("bridge_sources entries must be source identity records")
        allowed = {"source_id", "source_group", "trajectory_id", "split"}
        if set(bridge) - allowed:
            raise ValueError("bridge_sources contain identities/split only, never paired training targets")
        records.append({**bridge, "record_kind": "bridge"})
    validate_video_sources(records)
    if not selected:
        raise ValueError(f"video dataset has no {split} windows")
    return selected, records
