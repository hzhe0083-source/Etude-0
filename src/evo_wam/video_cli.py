"""Unpaired video pretraining and frozen demonstration-token export."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .video_effects import VideoEffectEncoder, EffectFeaturePredictor, effect_pretraining_loss


def load_video_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("schema_version") != 1 or config.get("kind") != "video_effect_experiment":
        raise ValueError("expected a version-1 video_effect_experiment config")
    model = config["model"]
    required = {"feature_dim", "latent_dim", "num_tokens", "hidden_dim", "noise_std",
                "geometry_dim", "relation_dim", "event_dim"}
    if set(model) != required:
        raise ValueError("video model config must explicitly specify the encoder and prediction dimensions")
    for name in required - {"noise_std"}:
        minimum = 0 if name in {"geometry_dim", "relation_dim", "event_dim"} else 1
        if type(model[name]) is not int or model[name] < minimum:
            raise ValueError(f"invalid {name}")
    if type(model["noise_std"]) not in (int, float) or not math.isfinite(model["noise_std"]) or model["noise_std"] < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    frames, context = config["window_frames"], config["context_frames"]
    if type(frames) is not int or type(context) is not int or not 1 <= context <= frames - 2:
        raise ValueError("each window needs past observations, an intermediate target and a terminal target")
    weights = config["loss_weights"]
    if set(weights) != {"features", "geometry", "relations", "events", "capacity"}:
        raise ValueError("declare every video pretraining loss weight")
    if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in weights.values()) or weights["features"] <= 0:
        raise ValueError("video losses must be nonnegative with positive feature prediction weight")
    schedule = config["domain_schedule"]
    if not isinstance(schedule, list) or not schedule or set(schedule) - {"human", "robot"}:
        raise ValueError("domain_schedule must list human/robot sampling slots")
    training = config["training"]
    if type(training["max_steps"]) is not int or training["max_steps"] < 1:
        raise ValueError("positive pretraining budget required")
    if type(training["learning_rate"]) not in (int, float) or not math.isfinite(training["learning_rate"]) or training["learning_rate"] <= 0:
        raise ValueError("positive finite learning_rate required")
    return config


def make_capacity_configs(video_config_path, robot_config_path, output):
    """Vary K only; widen B/P and the reader/codec without guessing data dimensions.

    Synthetic markers are preserved. Generating these files does not audit the
    data, run an experiment, or establish that any candidate capacity is enough.
    """
    from .cli import write_json
    from .data import load_experiment

    video = load_video_config(video_config_path)
    robot = load_experiment(robot_config_path)
    data_dimensions = ("entity_dim", "proprio_dim", "embodiment_dim", "geometry_dim",
                       "relation_dim", "event_dim", "action_dim", "roles", "max_precedence_edges")
    if any(type(robot["dimensions"].get(name)) is not int or robot["dimensions"][name] < 1
           for name in data_dimensions):
        raise ValueError("declare positive integer robot data dimensions before generating capacity configs")
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("capacity configs require a fresh output directory")
    video_paths = {}
    for count in (16, 64, 100):
        candidate = deepcopy(video)
        candidate["model"].update(num_tokens=count, latent_dim=768, hidden_dim=512)
        path = output / f"video_K{count}.json"
        write_json(path, candidate)
        video_paths[str(count)] = str(path)
    robot["dimensions"].update(demo_dim=768, token_dim=768)
    # Changed capacity requires fresh validation of predictions and thresholds.
    robot["validation_locked"] = False
    robot["binding_policy"]["validation_locked"] = False
    robot_path = output / "robot_768.json"
    write_json(robot_path, robot)
    return {"video_configs": video_paths, "primary_video_config": video_paths["64"],
            "robot_config": str(robot_path)}


def build_video_models(config, device):
    model = config["model"]
    encoder = VideoEffectEncoder(**{k: model[k] for k in
        ("feature_dim", "latent_dim", "num_tokens", "hidden_dim", "noise_std")}).to(device)
    predictor = EffectFeaturePredictor(**{k: model[k] for k in
        ("feature_dim", "latent_dim", "hidden_dim", "geometry_dim", "relation_dim", "event_dim")}).to(device)
    return encoder, predictor


def _read_artifact(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or type(payload.get("format_version")) is not int
            or payload["format_version"] != 2 or payload.get("kind") != "video_effect_pretrain"):
        raise ValueError("expected a version-2 video-effect artifact; re-pretrain and re-export legacy encoders/tokens")
    counts = payload.get("feature_kind_updates")
    if (not isinstance(counts, dict) or set(counts) != {"patches", "tracked_entities"}
            or any(type(count) is not int or count < 0 for count in counts.values())
            or type(payload.get("updates")) is not int or sum(counts.values()) != payload["updates"]):
        raise ValueError("artifact must record successful feature_kind_updates matching its total updates")
    return payload


def require_feature_kind(payload, feature_kind):
    if feature_kind not in {"patches", "tracked_entities"}:
        raise ValueError("feature_kind must be patches or tracked_entities")
    if payload.get("feature_kind_updates", {}).get(feature_kind, 0) < 1:
        raise ValueError(f"encoder has no successful {feature_kind} updates; train that feature kind before evaluation or export")


def load_video_encoder(path, *, device="cpu"):
    payload = _read_artifact(path)
    if payload.get("updates", 0) < 1:
        raise ValueError("cannot deploy a video encoder without a successful supervised update")
    model = payload["config"]["model"]
    encoder = VideoEffectEncoder(**{k: model[k] for k in
        ("feature_dim", "latent_dim", "num_tokens", "hidden_dim", "noise_std")}).to(device)
    encoder.load_state_dict(payload["encoder"], strict=True)
    return encoder.eval().requires_grad_(False), payload


def encoding_metadata(payload, digest):
    model = payload["config"]["model"]
    return {"kind": "video_effect_tokens", "encoder_version": 2, "encoder_sha256": digest,
            "feature_space_id": payload["feature_space_id"], "token_dim": model["latent_dim"],
            "window_frames": payload["config"]["window_frames"], "num_tokens": model["num_tokens"]}


def train_video(args):
    from .cli import file_sha256, move, write_json
    from .video_data import load_video_index, load_video_window, select_training_sources
    config = load_video_config(args.config)
    paths, sources = load_video_index(args.index)
    if type(args.steps) is not int or not 1 <= args.steps <= config["training"]["max_steps"]:
        raise ValueError("steps must fit the registered pretraining budget")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("use a fresh video run directory or resume explicitly")
    grouped = {domain: [] for domain in set(config["domain_schedule"])}
    feature_spaces, schemas = set(), set()
    identities = {"index": file_sha256(args.index)}
    for path in paths:
        meta = json.loads(path.read_text())
        if meta["domain"] in grouped:
            grouped[meta["domain"]].append(path)
            feature_spaces.add(meta["feature_space_id"])
            if meta.get("effect_schema_id"):
                schemas.add(meta["effect_schema_id"])
            identities[str(path.relative_to(Path(args.index).resolve().parent))] = file_sha256(path)
    if any(not items for items in grouped.values()) or len(feature_spaces) != 1 or len(schemas) > 1:
        raise ValueError("every scheduled domain needs training data in one feature space and effect vocabulary")
    torch.manual_seed(args.seed)
    encoder, predictor = build_video_models(config, args.device)
    parameters = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config["training"]["learning_rate"], weight_decay=0.)
    start, updates = 0, 0
    counts = {domain: 0 for domain in grouped}
    domain_updates = dict(counts)
    feature_kind_updates = {"patches": 0, "tracked_entities": 0}
    visited_arrays = {}
    geometry_basis = None
    if args.resume:
        previous = _read_artifact(args.resume)
        if previous["config"] != config or previous["data_identity"] != identities or previous["seed"] != args.seed:
            raise ValueError("resume requires the same config, data contents and seed")
        encoder.load_state_dict(previous["encoder"], strict=True)
        predictor.load_state_dict(previous["predictor"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        torch.set_rng_state(previous["torch_rng"])
        if torch.cuda.is_available() and previous["cuda_rng"]:
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
        start, updates = previous["attempted_steps"], previous["updates"]
        counts, domain_updates = previous["domain_windows"], previous["domain_updates"]
        feature_kind_updates = previous["feature_kind_updates"]
        visited_arrays = previous["visited_arrays"]
        geometry_basis = previous.get("geometry_basis")
        known_paths = {str(path.relative_to(Path(args.index).resolve().parent)): path for items in grouped.values() for path in items}
        for name, digest in visited_arrays.items():
            if name not in known_paths:
                raise ValueError("resume includes an unregistered video window")
            checked = load_video_window(known_paths[name])
            if file_sha256(known_paths[name].parent / checked.metadata["arrays"]) != digest:
                raise ValueError("a previously consumed video window changed on disk")
    if start + args.steps > config["training"]["max_steps"]:
        raise ValueError("cumulative video pretraining budget exceeded")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    started = time.monotonic()
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
        for step in range(start, start + args.steps):
            domain = config["domain_schedule"][step % len(config["domain_schedule"]) ]
            path = grouped[domain][counts[domain] % len(grouped[domain])]
            counts[domain] += 1
            sample = move(load_video_window(path), args.device)
            name = str(path.relative_to(Path(args.index).resolve().parent))
            if name not in visited_arrays:
                visited_arrays[name] = file_sha256(path.parent / sample.metadata["arrays"])
            if sample.features.shape[1] != config["window_frames"] or sample.context_frames != config["context_frames"]:
                raise ValueError("video window/context length differs from experiment")
            if sample.features.shape[-1] != config["model"]["feature_dim"]:
                raise ValueError("visual feature dimension differs from the frozen target encoder")
            if "geometry" in sample.effect_valid and bool(sample.effect_valid["geometry"].any()):
                basis = [sample.metadata["geometry_frame"], sample.metadata["geometry_units"]]
                if geometry_basis is not None and geometry_basis != basis:
                    raise ValueError("geometry labels must share one audited frame convention and unit system")
                geometry_basis = basis
            optimizer.zero_grad(set_to_none=True)
            losses = {"total": sample.features.new_zeros(()), "valid_count": sample.features.new_zeros(())}
            if sample.has_training_signal:
                tokens = encoder(sample.features, sample.feature_valid, sample.frame_times,
                    feature_kind=sample.metadata["feature_kind"], patch_coordinates=sample.patch_coordinates)
                past = sample.context_frames
                predictions = predictor(sample.features[:, :past], sample.feature_valid[:, :past], tokens,
                    sample.frame_times[past:] - sample.frame_times[past - 1], past_times=sample.frame_times[:past],
                    effect_fields=tuple(sample.effect_targets), feature_kind=sample.metadata["feature_kind"],
                    patch_coordinates=sample.patch_coordinates)
                losses = effect_pretraining_loss(predictions, sample.features[:, past:], sample.feature_valid[:, past:],
                    sample.effect_targets, sample.effect_valid, tokens, config["loss_weights"])
            if not torch.isfinite(losses["total"]):
                raise ValueError("non-finite video objective; no optimizer update performed")
            updated = losses["total"].requires_grad
            if updated:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                domain_updates[domain] += 1
                feature_kind_updates[sample.metadata["feature_kind"]] += 1
            metric = {key: float(value.detach()) for key, value in losses.items()}
            metric.update(step=step, domain=domain, updated=updated, source_id=sample.metadata["source_id"])
            log.write(json.dumps(metric, allow_nan=False) + "\n")
            log.flush()
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    payload = {"format_version": 2, "kind": "video_effect_pretrain", "config": config,
        "encoder": encoder.state_dict(), "predictor": predictor.state_dict(), "optimizer": optimizer.state_dict(),
        "updates": updates, "attempted_steps": start + args.steps, "seed": args.seed,
        "domain_windows": counts, "domain_updates": domain_updates, "source_records": sources,
        "feature_kind_updates": feature_kind_updates,
        "training_source_records": select_training_sources(sources, {domain for domain, count in counts.items() if count}),
        "feature_space_id": next(iter(feature_spaces)), "effect_schema_id": next(iter(schemas), None),
        "geometry_basis": geometry_basis,
        "data_identity": identities, "visited_arrays": visited_arrays, "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "wam_updated_by_video_loss": False}
    temporary = output / "video_encoder.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(output / "video_encoder.pt")
    report = {"artifact": str((output / "video_encoder.pt").resolve()), "updates": updates,
        "domain_windows": counts, "domain_updates": domain_updates, "feature_kind_updates": feature_kind_updates,
        "elapsed_seconds": time.monotonic() - started,
        "wam_updated_by_video_loss": False, "robot_execution_evaluated": False}
    write_json(output / "run.json", report)
    return report


@torch.no_grad()
def evaluate_video(args):
    """Held-out feature diagnostics, including zeroed and shuffled bottlenecks."""
    from .cli import move, write_json, file_sha256
    from .video_data import load_video_index, load_video_window, validate_video_sources
    encoder, payload = load_video_encoder(args.artifact, device=args.device)
    _, predictor = build_video_models(payload["config"], args.device)
    predictor.load_state_dict(payload["predictor"], strict=True)
    predictor.eval().requires_grad_(False)
    paths, sources = load_video_index(args.index, split=args.split)
    trained = payload.get("training_source_records", payload["source_records"])
    validate_video_sources([*trained, *sources])
    if any(record.get("feature_space_id", payload["feature_space_id"]) != payload["feature_space_id"] for record in sources):
        raise ValueError("evaluation feature space differs from the encoder")
    if type(args.max_samples) is not int or args.max_samples < 2:
        raise ValueError("shuffle diagnostics require a budget of at least two windows")
    selected, tokens = [], []
    for path in paths[:args.max_samples]:
        sample = move(load_video_window(path), args.device)
        if not sample.has_training_signal or not sample.feature_valid[:, sample.context_frames:].any():
            continue
        require_feature_kind(payload, sample.metadata["feature_kind"])
        if sample.features.shape[1] != payload["config"]["window_frames"] or sample.context_frames != payload["config"]["context_frames"]:
            raise ValueError("evaluation must retain the registered context and window lengths")
        selected.append(path)
        tokens.append(encoder(sample.features, sample.feature_valid, sample.frame_times, noise=False,
            feature_kind=sample.metadata["feature_kind"], patch_coordinates=sample.patch_coordinates))
    if len(selected) < 2:
        raise ValueError("need at least two held-out windows with valid future features")
    squared_errors = {name: 0. for name in ("correct_z", "zero_z", "shuffled_z")}
    count = 0
    for index, path in enumerate(selected):
        sample = move(load_video_window(path), args.device)
        past = sample.context_frames
        target, valid = sample.features[:, past:], sample.feature_valid[:, past:]
        count += int(valid.sum())
        for name, z in (("correct_z", tokens[index]), ("zero_z", torch.zeros_like(tokens[index])),
                        ("shuffled_z", tokens[(index + 1) % len(tokens)])):
            prediction = predictor(sample.features[:, :past], sample.feature_valid[:, :past], z,
                sample.frame_times[past:] - sample.frame_times[past - 1], past_times=sample.frame_times[:past],
                effect_fields=(), feature_kind=sample.metadata["feature_kind"],
                patch_coordinates=sample.patch_coordinates)["features"]
            squared_errors[name] += float(torch.where(valid, (prediction - target).square(), 0).sum())
    report = {"metric": "heldout_multi_time_feature_mse", "split": args.split,
        "windows": len(selected), "valid_feature_values": count,
        "mse": {name: error / count for name, error in squared_errors.items()},
        "encoder_sha256": file_sha256(args.artifact), "evaluation_index_sha256": file_sha256(args.index),
        "shuffle": "one_window_cyclic_permutation", "robot_execution_evaluated": False,
        "view_invariance_established": False}
    if Path(args.output).exists():
        raise ValueError("diagnostics output already exists")
    write_json(args.output, report)
    return report


@torch.no_grad()
def encode_demonstrations(args):
    """Convert an existing labeled robot sample or observed-only input, retaining its labels."""
    from .cli import file_sha256, write_json
    from .data import load_sample, load_observation
    from .video_data import patch_grid_coordinates, validate_patch_coordinates
    path, output = Path(args.manifest).resolve(), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("demonstration export requires a fresh output directory")
    meta = json.loads(path.read_text())
    loader = load_sample if meta.get("kind") == "training_sample" else load_observation
    sample = loader(path)
    if sample.demonstration_encoding != {"kind": "raw_features"}:
        raise ValueError("demonstrations are already encoded; do not encode tokens twice")
    encoder, payload = load_video_encoder(args.artifact, device=args.device)
    if meta.get("demo_feature_space_id") != payload["feature_space_id"]:
        raise ValueError("declare the exact frozen visual feature space used for encoder pretraining")
    layouts = meta.get("demonstration_layouts")
    if not isinstance(layouts, list) or len(layouts) != len(sample.demonstrations):
        raise ValueError("raw demonstration tokens require audited frame/patch layout and times")
    array_path = path.parent / meta["arrays"]
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for index, (demo, layout) in enumerate(zip(sample.demonstrations, layouts)):
        if not isinstance(layout, dict) or not {"frames", "tokens_per_frame", "frame_times", "feature_kind"} <= layout.keys():
            raise ValueError("raw demonstrations require explicit feature_kind, frame/patch layout and times")
        frames, patches = layout["frames"], layout["tokens_per_frame"]
        if type(frames) is not int or type(patches) is not int or min(frames, patches) < 1 or frames * patches != demo.shape[1]:
            raise ValueError("demo layout does not match the ordered feature array")
        kind, coordinates = layout["feature_kind"], None
        if kind == "patches":
            if layout.get("token_order") != "time,height,width,channel":
                raise ValueError("patch demonstrations require canonical time,height,width,channel token_order")
            coordinates = patch_grid_coordinates(layout.get("patch_grid"))
            validate_patch_coordinates(coordinates, layout.get("patch_grid"), layout.get("patch_coordinate_system"))
            if coordinates.shape[0] != patches:
                raise ValueError("demo patch_grid does not match tokens_per_frame")
            coordinates = coordinates.to(args.device)
        elif kind == "tracked_entities":
            ids = layout.get("entity_ids")
            if (not isinstance(ids, list) or len(ids) != patches
                    or any(type(value) is not int or value < 0 for value in ids) or len(set(ids)) != patches):
                raise ValueError("tracked demonstrations require stable unique nonnegative entity_ids for their columns")
            if layout.get("token_order") != "time,entity,channel":
                raise ValueError("tracked demonstrations require time,entity,channel token_order")
            if any(name in layout for name in ("patch_grid", "patch_coordinate_system")):
                raise ValueError("tracked entity IDs are not patch coordinates")
        else:
            raise ValueError("demo feature_kind must be patches or tracked_entities")
        require_feature_kind(payload, kind)
        features = demo.reshape(1, frames, patches, demo.shape[-1]).to(args.device)
        times = torch.tensor(layout["frame_times"], dtype=torch.float32, device=args.device)
        tokens = encoder.encode_demo(features, torch.ones_like(features, dtype=torch.bool), times,
                                     window_frames=payload["config"]["window_frames"],
                                     feature_kind=kind, patch_coordinates=coordinates)
        arrays[f"demo_view_{index}"] = tokens[0].cpu().numpy()
    meta["raw_demonstration_layouts"] = meta.pop("demonstration_layouts")
    meta["demonstration_encoding"] = encoding_metadata(payload, file_sha256(args.artifact))
    meta["arrays"] = "arrays.npz"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "arrays.npz", **arrays)
    write_json(output / "sample.json", meta)
    loader(output / "sample.json")
    return {"manifest": str((output / "sample.json").resolve()), "demonstration_encoding": meta["demonstration_encoding"],
            "wam_updated": False, "commands_sent": 0}
