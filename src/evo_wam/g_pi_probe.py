"""Offline ordered-feature probes; purpose identities never condition Wan or G."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .g_pi_context import assert_frozen_base, frozen_base_checksum


def _grid(grid):
    if (not isinstance(grid, (list, tuple)) or len(grid) != 3
            or any(type(value) is not int or value < 1 for value in grid)):
        raise ValueError("probe pool_grid must contain positive time, height and width sizes")
    return tuple(grid)


def ordered_probe_features(features, video_grid, pool_grid=(4, 2, 2)):
    """Keep temporal and spatial slots, then flatten in time/row/column order."""
    video_grid, pool_grid = _grid(video_grid), _grid(pool_grid)
    if (not isinstance(features, torch.Tensor) or features.ndim != 3
            or features.shape[0] != 1 or features.shape[1] != math.prod(video_grid)
            or features.shape[2] < 1 or not features.is_floating_point()
            or not torch.isfinite(features).all()):
        raise ValueError("probe features must be finite [1,T*H*W,D] matching the video grid")
    spatial = features.detach().float().transpose(1, 2).reshape(1, features.shape[2], *video_grid)
    return F.adaptive_avg_pool3d(spatial, pool_grid).flatten(2).transpose(1, 2).contiguous()


def validate_probe_split(table):
    """Hold out all three confounds jointly, not merely distinct filenames."""
    train = [entry for entry in table.entries if entry.split == "train"]
    test = [entry for entry in table.entries if entry.split == "test"]
    if not train or not test:
        raise ValueError("intent probe requires nonempty train and test splits")
    for field in ("person_id", "scene_id", "source_id", "source_group", "component"):
        splits = {}
        for entry in table.entries:
            value = getattr(entry, field)
            if splits.setdefault(value, entry.split) != entry.split:
                raise ValueError(f"intent probe train/test {field} groups must be disjoint across all splits")
    train_classes = {entry.purpose_group for entry in train}
    test_classes = {entry.purpose_group for entry in test}
    if len(train_classes) < 2 or not train_classes & test_classes:
        raise ValueError("intent probe needs at least two training purposes and shared train/test purposes")
    return train, test


def _data_identity(table):
    from .cli import file_sha256
    # A probe depends only on demonstrations and offline groups, never paired
    # robot states, future goals or task language archives.
    paths = {table.path} | {entry.arrays for entry in table.entries}
    return [{"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in sorted(paths, key=str)]


@torch.no_grad()
def extract_probe_features(table, native, config, *, layer, pool_grid=(4, 2, 2)):
    """One complete demo at a time, under the pretrained fixed empty prompt."""
    from .g_pi_context import demo_context_features
    from .g_pi_intent import load_intent_demo

    validate_probe_split(table)
    pool_grid = _grid(pool_grid)
    if type(layer) is not int or not 0 <= layer < len(native.blocks):
        raise ValueError("probe layer must select a native video block")
    assert_frozen_base(native)
    before = frozen_base_checksum(native)
    weight = native.patch_embedding_mlp.weight
    features = {}
    for entry in table.entries:
        demo = load_intent_demo(entry).demonstration.to(weight)
        selected = demo_context_features(native, demo, config)[layer]
        _, ph, pw = native.patch_size
        features[entry.demo_id] = ordered_probe_features(selected,
            (demo.shape[2], demo.shape[3] // ph, demo.shape[4] // pw), pool_grid)[0].cpu()
    if frozen_base_checksum(native) != before:
        raise ValueError("frozen video base changed during demonstration-only probe extraction")
    identity = {"kind": "g_pi_demo_features", "layer": layer, "pool_grid": list(pool_grid),
                "token_order": "time_row_column", "conditioning": "pretrained_empty_prompt_only",
                "base_sha256": before, "empty_text_identity": native.g_pi_empty_text_identity,
                "context": {key: config[key] for key in
                            ("chunk_size", "max_frame_chunk_size", "icl_rope_h", "window_size")},
                "data_files": _data_identity(table)}
    return features, identity


def save_probe_features(path, table, features, identity):
    """A portable, detached cache, with input and frozen-base provenance."""
    features = _feature_tensor(table, features)
    _validate_identity(table, identity)
    if features.shape[1] != math.prod(_grid(identity["pool_grid"])):
        raise ValueError("cached probe token count must match ordered pooling grid")
    with open(path, "wb") as stream:
        np.savez(stream, features=features.numpy(),
                 demo_ids=np.asarray([entry.demo_id for entry in table.entries]),
                 feature_identity=np.asarray(json.dumps(identity, sort_keys=True, allow_nan=False)))


def _validate_identity(table, identity, *, layer=None, pool_grid=None):
    required = {"kind", "layer", "pool_grid", "token_order", "conditioning",
                "base_sha256", "empty_text_identity", "context", "data_files"}
    if (not isinstance(identity, dict) or set(identity) != required
            or identity["kind"] != "g_pi_demo_features"
            or identity["token_order"] != "time_row_column"
            or identity["conditioning"] != "pretrained_empty_prompt_only"
            or type(identity["layer"]) is not int or identity["layer"] < 0
            or not isinstance(identity["base_sha256"], str) or len(identity["base_sha256"]) != 64
            or not isinstance(identity["empty_text_identity"], dict)
            or not identity["empty_text_identity"]):
        raise ValueError("probe cache requires frozen-base, layer, ordered pooling and empty-prompt identity")
    _grid(identity["pool_grid"])
    context = identity["context"]
    if (not isinstance(context, dict) or set(context) !=
            {"chunk_size", "max_frame_chunk_size", "icl_rope_h", "window_size"}
            or any(type(value) is not int or value < 1 for value in context.values())
            or context["chunk_size"] > context["max_frame_chunk_size"]):
        raise ValueError("probe identity requires the native demo context and RoPE configuration")
    if identity["data_files"] != _data_identity(table):
        raise ValueError("probe cache input file identity changed")
    if ((layer is not None and layer != identity["layer"])
            or (pool_grid is not None and list(_grid(pool_grid)) != identity["pool_grid"])):
        raise ValueError("probe cache layer or pooling identity mismatch")


def load_probe_features(path, table, *, layer, pool_grid):
    with np.load(path, allow_pickle=False) as values:
        if set(values.files) != {"features", "demo_ids", "feature_identity"}:
            raise ValueError("probe feature cache requires features, demo_ids and feature_identity")
        ids = values["demo_ids"].tolist()
        if ids != [entry.demo_id for entry in table.entries]:
            raise ValueError("probe feature cache must match the exact offline demonstration order")
        identity = json.loads(str(values["feature_identity"].item()))
        _validate_identity(table, identity, layer=layer, pool_grid=pool_grid)
        array = torch.from_numpy(np.array(values["features"], copy=True))
    if array.ndim != 3 or array.shape[0] != len(ids):
        raise ValueError("cached probe features must be [N,K,D]")
    features = dict(zip(ids, array))
    _feature_tensor(table, features)
    if array.shape[1] != math.prod(_grid(pool_grid)):
        raise ValueError("cached probe token count must match ordered pooling grid")
    return features, identity


def _feature_tensor(table, features):
    if not isinstance(features, dict) or set(features) != {entry.demo_id for entry in table.entries}:
        raise ValueError("probe features must cover exactly the offline table demonstration ids")
    rows = [features[entry.demo_id] for entry in table.entries]
    if any(not isinstance(row, torch.Tensor) or row.ndim != 2 or min(row.shape) < 1
           or not row.is_floating_point() or not torch.isfinite(row).all() for row in rows):
        raise ValueError("probe features must be finite nonempty [K,D] tensors")
    if len({tuple(row.shape) for row in rows}) != 1:
        raise ValueError("probe features must share a fixed ordered token grid and dimension")
    return torch.stack([row.detach().float().cpu() for row in rows])


def _group_weights(entries):
    counts = Counter(entry.component for entry in entries)
    weights = torch.tensor([1. / counts[entry.component] for entry in entries], dtype=torch.float64)
    return weights / weights.sum()


def fit_intent_probe(table, features, *, steps=200, learning_rate=.01, seed=0):
    """Fit only a linear readout; normalization is estimated on training data."""
    train, test = validate_probe_split(table)
    if type(steps) is not int or steps < 1 or type(seed) is not int or seed < 0:
        raise ValueError("probe steps must be positive and seed a nonnegative integer")
    if (type(learning_rate) not in (int, float) or not math.isfinite(learning_rate)
            or learning_rate <= 0):
        raise ValueError("probe learning_rate must be finite and positive")
    rows = _feature_tensor(table, features).flatten(1)
    indices = {entry.demo_id: index for index, entry in enumerate(table.entries)}
    classes = sorted({entry.purpose_group for entry in train})
    labels = {purpose: index for index, purpose in enumerate(classes)}
    x = rows[[indices[entry.demo_id] for entry in train]]
    source_weights = _group_weights(train)
    weights = source_weights.to(x)
    mean = (x * weights[:, None]).sum(0)
    scale = ((x - mean).square() * weights[:, None]).sum(0).sqrt().clamp_min(1e-6)
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
        raise ValueError("probe training feature scale overflowed; inspect cached feature magnitudes")
    x = (x - mean) / scale
    target = torch.tensor([labels[entry.purpose_group] for entry in train])
    # Isolate initialization from training's data/noise RNG streams.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        probe = torch.nn.Linear(rows.shape[1], len(classes))
        optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = (F.cross_entropy(probe(x), target, reduction="none") * weights).sum()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite probe objective; inspect features and learning_rate")
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        logits = probe((rows - mean) / scale)
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite probe predictions; inspect held-out feature magnitudes")
    predicted = {entry.demo_id: classes[int(logits[index].argmax())]
                 for index, entry in enumerate(table.entries)}
    class_prior = {name: sum(float(source_weights[index]) for index, entry in enumerate(train)
                            if entry.purpose_group == name) for name in classes}
    majority = max(classes, key=class_prior.get)
    test_weights = _group_weights(test)

    def score(entries):
        return sum(float(weight) * (predicted[entry.demo_id] == entry.purpose_group)
                   for weight, entry in zip(_group_weights(entries), entries))

    known = [entry for entry in test if entry.purpose_group in labels]
    per_class = {name: score([entry for entry in test if entry.purpose_group == name])
                 for name in sorted({entry.purpose_group for entry in test})}
    known_fraction = sum(float(weight) for weight, entry in zip(test_weights, test)
                         if entry.purpose_group in labels)
    return {"format_version": 1, "kind": "g_pi_intent_probe_report",
            "readout": "linear_ordered_slots", "standardization": "training_source_weighted_only",
            "seed": seed, "steps": steps, "learning_rate": float(learning_rate),
            "train_loss": float(loss.detach()), "train_accuracy": score(train),
            "test_accuracy": score(test), "known_purpose_accuracy": score(known),
            "macro_test_accuracy": sum(per_class.values()) / len(per_class),
            "random_accuracy": known_fraction / len(classes),
            "known_purpose_random_accuracy": 1. / len(classes),
            "majority_accuracy": sum(float(weight) * (entry.purpose_group == majority)
                                     for weight, entry in zip(test_weights, test)),
            "train_purposes": classes, "shared_purposes": sorted({entry.purpose_group for entry in known}),
            "unseen_test_purposes": sorted({entry.purpose_group for entry in test} - set(classes)),
            "train_class_prior": class_prior, "per_purpose_accuracy": per_class,
            "split_counts": {name: {"demonstrations": len(entries),
                "independent_sources": len({entry.component for entry in entries}),
                "people": len({entry.person_id for entry in entries}),
                "scenes": len({entry.scene_id for entry in entries})}
                for name, entries in (("train", train), ("test", test))},
            "aggregation": "equal_source_components_then_equal_demonstrations_within_source",
            "predictions": [{"demo_id": entry.demo_id, "purpose_group": entry.purpose_group,
                             "predicted_purpose": predicted[entry.demo_id]}
                            for entry in test]}


def probe_g_pi_intent(args):
    """CLI entry: cache-only fixtures or frozen native full-demo extraction."""
    from .g_pi_intent import load_intent_table

    path = Path(args.manifest).resolve()
    manifest = json.loads(path.read_text())
    required = {"format_version", "kind", "purpose_table", "layer", "pool_grid"}
    allowed = required | {"steps", "learning_rate", "seed", "feature_cache", "save_feature_cache"}
    if (not isinstance(manifest, dict) or required - set(manifest) or set(manifest) - allowed
            or type(manifest["format_version"]) is not int or manifest["format_version"] != 1
            or manifest["kind"] != "g_pi_intent_probe"):
        raise ValueError("expected a version-1 g_pi_intent_probe manifest")
    if type(manifest["layer"]) is not int or manifest["layer"] < 0:
        raise ValueError("probe layer must be a nonnegative block index")
    grid = _grid(manifest["pool_grid"])
    table = load_intent_table(path.parent / manifest["purpose_table"])
    validate_probe_split(table)
    if "feature_cache" in manifest:
        if getattr(args, "artifact", None) is not None:
            raise ValueError("probe chooses exactly one feature_cache or artifact extraction route")
        if "save_feature_cache" in manifest:
            raise ValueError("save_feature_cache is only available with native feature extraction")
        features, identity = load_probe_features(path.parent / manifest["feature_cache"], table,
                                                 layer=manifest["layer"], pool_grid=grid)
    else:
        if getattr(args, "artifact", None) is None:
            raise ValueError("native intent probe requires a G training artifact or a feature cache")
        from .g_pi_training import load_g_pi_encoder

        encoder, payload = load_g_pi_encoder(args.artifact, device=getattr(args, "device", "cuda"),
                                              checkpoint=getattr(args, "checkpoint", None))
        if payload["interface_type"] != "g_translator":
            raise ValueError("intent probe requires the G frozen-video base")
        features, identity = extract_probe_features(table, encoder.native, payload["config"],
                                                    layer=manifest["layer"], pool_grid=grid)
        if "save_feature_cache" in manifest:
            save_probe_features(path.parent / manifest["save_feature_cache"], table, features, identity)
    report = fit_intent_probe(table, features, steps=manifest.get("steps", 200),
                             learning_rate=manifest.get("learning_rate", .01), seed=manifest.get("seed", 0))
    report["feature_identity"] = identity
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return {"report": str(Path(args.output).resolve()), "test_accuracy": report["test_accuracy"],
            "random_accuracy": report["random_accuracy"]}
