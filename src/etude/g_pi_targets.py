"""Offline, source-bound E targets for independent G and pi training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .cli import file_sha256, write_json
from .g_pi_context import _json_identity, _tensor_hash, load_target_cache, save_target_cache
from .g_pi_data import load_g_pi_index, load_g_pi_sample
from .video_data import SPLITS, _local_path


def _task_targets(path):
    # Reuse every grid/boundary validation without reading demo or language data.
    sample = load_g_pi_sample(path, read_language=False, generator=torch.Generator().manual_seed(0))
    arrays = _local_path(path.parent, sample.metadata["arrays"], ".npz")
    with np.load(arrays, allow_pickle=False) as archive:
        times = archive["subgoal_times"].tolist()
        frames = torch.from_numpy(archive["subgoal_latents"].copy())
    return sample, arrays, times, frames


def build_target_cache_index(index_path, output_dir, encoder, *, split="train"):
    """Encode each subgoal as its own native frame, then bind targets to sources."""
    source = Path(index_path).resolve()
    paths, _ = load_g_pi_index(source, split)
    identity = _json_identity(encoder.identity)
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("target cache output must be a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    source_hash = file_sha256(source)
    records = []
    for path in paths:
        task_hash = file_sha256(path)
        arrays_path = _local_path(path.parent, json.loads(path.read_text())["arrays"], ".npz")
        arrays_hash = file_sha256(arrays_path)
        sample, arrays, times, frames = _task_targets(path)
        with torch.no_grad():
            z = torch.cat([encoder(frame[None]).detach().float().cpu() for frame in frames])
        if task_hash != file_sha256(path) or arrays_hash != file_sha256(arrays):
            raise ValueError("target source changed during offline encoding")
        relative = str(path.relative_to(source.parent))
        name = hashlib.sha256(relative.encode()).hexdigest() + ".npz"
        cache = output / name
        save_target_cache(cache, z, identity)
        records.append({"task": relative, "sample_id": sample.metadata["sample_id"],
                        "task_sha256": task_hash, "arrays_sha256": arrays_hash,
                        "subgoal_times": times,
                        "subgoal_latent_sha256": [_tensor_hash(frame) for frame in frames],
                        "cache": name, "cache_sha256": file_sha256(cache)})
    if source_hash != file_sha256(source):
        raise ValueError("target dataset index changed during offline encoding")
    document = {"format_version": 1, "kind": "g_pi_target_cache_index", "split": split,
                "source_index": str(source), "source_sha256": source_hash,
                "encoder_identity": identity, "records": records}
    path = output / "target-index.json"
    write_json(path, document)
    return path


@dataclass(frozen=True)
class _TargetRecord:
    sample_id: str
    times: tuple[float, ...]
    frame_hashes: tuple[str, ...]
    cache: Path
    cache_hash: str


class TargetCacheIndex:
    def __init__(self, identity, records, files):
        self.encoder_identity = _json_identity(identity)
        self._records = records
        self.files = tuple(dict.fromkeys(files))

    def goal(self, sample, manifest_path):
        """Select the exact robot boundary; never run E or return mutable cache data."""
        record = self._records.get(Path(manifest_path).resolve())
        if record is None:
            raise ValueError("task is absent from target_cache_index")
        if sample.metadata["sample_id"] != record.sample_id:
            raise ValueError("target cache sample identity differs from the selected task")
        try:
            index = record.times.index(sample.subgoal_time)
        except ValueError as exc:
            raise ValueError("selected subgoal_time is absent from target cache") from exc
        if _tensor_hash(sample.target_frame[0]) != record.frame_hashes[index]:
            raise ValueError("selected single-frame latent differs from the target cache source")
        if file_sha256(record.cache) != record.cache_hash:
            raise ValueError("target cache token archive changed after loading its index")
        # Keep the dataset's tokens on disk rather than accumulating all tasks in RAM.
        z = load_target_cache(record.cache, self.encoder_identity)
        return z[index:index + 1].clone()


def load_target_cache_index(path, expected_identity, *, task_paths=None):
    """Validate E, source contents, boundary order and all token archive hashes."""
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    required = {"format_version", "kind", "split", "source_index", "source_sha256",
                "encoder_identity", "records"}
    if (not isinstance(document, dict) or set(document) != required
            or type(document["format_version"]) is not int or document["format_version"] != 1
            or document["kind"] != "g_pi_target_cache_index" or document["split"] not in SPLITS
            or not isinstance(document["records"], list) or not document["records"]):
        raise ValueError("expected a version-1 g_pi_target_cache_index with nonempty records")
    identity = _json_identity(expected_identity)
    if not isinstance(identity, dict) or document["encoder_identity"] != identity:
        raise ValueError("E identity mismatch in target cache index")
    if not isinstance(document["source_index"], str) or not Path(document["source_index"]).is_absolute():
        raise ValueError("target cache source_index must be an explicit absolute dataset index path")
    source = Path(document["source_index"]).resolve()
    if file_sha256(source) != document["source_sha256"]:
        raise ValueError("target cache source dataset index changed")
    selected, _ = load_g_pi_index(source, document["split"])
    expected_tasks = set(selected)
    if task_paths is not None and not {Path(task).resolve() for task in task_paths} <= expected_tasks:
        raise ValueError("target cache split does not contain every requested training task")
    files, records, caches = [path, source], {}, set()
    fields = {"task", "sample_id", "task_sha256", "arrays_sha256", "subgoal_times",
              "subgoal_latent_sha256", "cache", "cache_sha256"}
    for record in document["records"]:
        if not isinstance(record, dict) or set(record) != fields:
            raise ValueError("target cache record requires exact task, boundary, source and cache identities")
        task = _local_path(source.parent, record["task"], ".json")
        cache = _local_path(path.parent, record["cache"], ".npz")
        if task in records or cache in caches or task not in expected_tasks:
            raise ValueError("target cache task/cache records must be unique and belong to the declared split")
        if file_sha256(task) != record["task_sha256"]:
            raise ValueError("target cache source task manifest changed")
        sample, arrays, times, frames = _task_targets(task)
        if file_sha256(arrays) != record["arrays_sha256"]:
            raise ValueError("target cache source arrays changed")
        if sample.metadata["sample_id"] != record["sample_id"]:
            raise ValueError("target cache sample identity differs from source task")
        if times != record["subgoal_times"]:
            raise ValueError("target cache subgoal_times differ from the exact source boundary order")
        hashes = [_tensor_hash(frame) for frame in frames]
        if hashes != record["subgoal_latent_sha256"]:
            raise ValueError("target cache single-frame latent identity or order changed")
        if file_sha256(cache) != record["cache_sha256"]:
            raise ValueError("target cache token archive changed")
        z = load_target_cache(cache, identity)
        if z.shape[0] != len(times):
            raise ValueError("target cache must contain one z for every subgoal in chronological order")
        records[task] = _TargetRecord(record["sample_id"], tuple(times), tuple(hashes),
                                      cache, record["cache_sha256"])
        caches.add(cache)
        files.extend((task, arrays, cache))
    if set(records) != expected_tasks:
        raise ValueError("target cache records must cover every task in its declared split")
    return TargetCacheIndex(identity, records, files)


def cache_g_pi_targets(args):
    from .g_pi_training import ROUTES, build_g_pi_encoder, load_g_pi_encoder

    artifact, config_path = getattr(args, "artifact", None), getattr(args, "config", None)
    if bool(artifact) == bool(config_path):
        raise ValueError("offline targets require exactly one of artifact or fresh training config")
    device, checkpoint = getattr(args, "device", "cuda"), getattr(args, "checkpoint", None)
    if artifact:
        if getattr(args, "tiny_native", False):
            raise ValueError("artifact already records tiny_native; the flag is only for fresh config")
        encoder, _ = load_g_pi_encoder(artifact, device=device, checkpoint=checkpoint)
    else:
        config = json.loads(Path(config_path).read_text())
        route = config.get("interface_type")
        if route not in ROUTES:
            raise ValueError("offline target config must select g_translator or pi_goal")
        if route == "g_translator" and "intent_training" in config:
            # Purpose groups supervise the decoder, not the fixed image encoder.
            config["intent_training"] = {**config["intent_training"], "manifest": None,
                                         "contrastive_weight": 0.}
        _, encoder, _, _ = build_g_pi_encoder(config, stage=ROUTES[route], device=device,
            checkpoint=checkpoint, tiny_native=getattr(args, "tiny_native", False))
    path = build_target_cache_index(args.index, args.output, encoder, split=getattr(args, "split", "train"))
    load_target_cache_index(path, encoder.identity)
    return {"target_cache_index": str(path), "encoder_identity": encoder.identity}
