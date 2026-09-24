"""Measured offline G/π diagnostics; replay errors are not rollout success."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import html
import json
import math
from pathlib import Path

import numpy as np
import torch

from .cli import file_sha256, write_json
from .g_pi_controller import GoalThresholds, goal_distances, goal_reached
from .g_pi_deployment import load_g_pi_observation, load_goal_prediction
from .goal_language import load_goal_language
from .video_data import _local_path


_METRICS = ("z", "position_m", "rotation_deg", "gripper")
_PAIR_TYPES = {"demo_swap", "performer_viewpoint", "grasp_speed"}


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    scene: str
    task: str
    layout: str
    intent: str
    source_id: str
    subgoal: int
    demonstration: torch.Tensor
    history: torch.Tensor
    state: torch.Tensor
    history_times: torch.Tensor
    current: dict
    truth: dict
    language: torch.Tensor | None = None
    actions: torch.Tensor | None = None
    actions_mask: torch.Tensor | None = None
    wrong_goal: dict | None = None
    object_evidence: dict | None = None


def _labels(record, keys):
    if any(not isinstance(record.get(key), str) or not record[key].strip() for key in keys):
        raise ValueError(f"evaluation identifiers must be nonempty strings: {', '.join(keys)}")
    if type(record.get("subgoal")) is not int or record["subgoal"] < 0:
        raise ValueError("evaluation subgoal must be a nonnegative integer ordinal")


def _score(first, second, thresholds):
    # Dimensionless maximum: the calibrated acceptance rectangle has radius 1.
    distances = goal_distances(first, second)
    return max(distances[name] / max(getattr(thresholds, name), 1e-12) for name in _METRICS)


def _delta(goal, current):
    poses, initial = goal["goal_poses"].double().cpu(), current["goal_poses"].double().cpu()
    return {"z": goal["z"].double().cpu() - current["z"].double().cpu(),
            "translation": poses[..., :3, 3] - initial[..., :3, 3],
            "rotation": initial[..., :3, :3].transpose(-1, -2) @ poses[..., :3, :3],
            "gripper": goal["goal_gripper"].double().cpu() - current["goal_gripper"].double().cpu()}


def delta_metrics(prediction, truth, current):
    """Keep signed changes as well as errors; rotations use relative SO(3)."""
    predicted, reference = _delta(prediction, current), _delta(truth, current)
    distances = goal_distances(prediction, truth)
    return {"error": distances,
            "predicted_change": {key: value.tolist() for key, value in predicted.items()},
            "reference_change": {key: value.tolist() for key, value in reference.items()},
            "predicted_magnitude": goal_distances(prediction, current),
            "reference_magnitude": goal_distances(truth, current)}


def _regions(case, cases):
    peers = [other for other in cases if (other.scene, other.layout, other.subgoal)
             == (case.scene, case.layout, case.subgoal)]
    return ([other.truth for other in peers if other.intent == case.intent],
            [other.truth for other in peers if other.intent != case.intent])


def valid_goal_region(prediction, own, other, thresholds):
    """Accept the own-intent union and exclude every competing intent region."""
    if not own:
        raise ValueError("valid goal regions need measured references for the intended goal")
    near_own = any(goal_reached(prediction, goal, thresholds) for goal in own)
    near_other = any(goal_reached(prediction, goal, thresholds) for goal in other)
    return {"near_own": near_own, "near_other": near_other,
            "valid": near_own and not near_other, "competing_references": len(other)}


def scene_modal_goals(training_goals, thresholds):
    """Choose a training medoid by independent-source support, never test labels."""
    groups = defaultdict(list)
    for row in training_goals:
        _labels(row, ("scene", "task", "source_id"))
        if row.get("split") != "train":
            raise ValueError("scene modal goal baseline accepts training goals only")
        goal_distances(row["goal"], row["goal"])
        groups[(row["scene"], row["subgoal"])].append(row)
    result = {}
    for key, rows in groups.items():
        support = [len({other["source_id"] for other in rows
                        if goal_reached(row["goal"], other["goal"], thresholds)}) for row in rows]
        winner = max(range(len(rows)), key=lambda index: support[index])
        result[key] = {"goal": rows[winner]["goal"], "source_id": rows[winner]["source_id"],
                       "independent_support": support[winner]}
    return result


def grouped_statistics(rows, value_key):
    """Average repeats inside scene/task first; seeds and views add no units."""
    grouped = defaultdict(list)
    for row in rows:
        value = row[value_key]
        if value is not None:
            grouped[(row["scene"], row["task"])].append(float(value))
    units = [{"scene": key[0], "task": key[1], "records": len(values),
              "mean": sum(values) / len(values)} for key, values in sorted(grouped.items())]
    return {"independent_scene_tasks": len(units), "records": sum(row["records"] for row in units),
            "macro_mean": sum(row["mean"] for row in units) / len(units) if units else None,
            "units": units}


def _same_observation(left, right):
    return (left.scene, left.task, left.layout, left.subgoal) == (right.scene, right.task, right.layout, right.subgoal) \
        and all(torch.equal(getattr(left, key), getattr(right, key))
                for key in ("history", "state", "history_times")) \
        and all(torch.equal(left.current[key], right.current[key]) for key in left.current)


def _pair_cases(pair, indexed):
    if (not isinstance(pair, dict) or set(pair) != {"kind", "ids"} or pair["kind"] not in _PAIR_TYPES
            or not isinstance(pair["ids"], list) or len(pair["ids"]) != 2
            or len(set(pair["ids"])) != 2 or any(key not in indexed for key in pair["ids"])):
        raise ValueError("evaluation pairs need a known kind and two distinct case ids")
    left, right = (indexed[key] for key in pair["ids"])
    if not _same_observation(left, right):
        raise ValueError("paired demonstrations must keep the robot observation and current goal fixed")
    same_intent = left.intent == right.intent
    if same_intent == (pair["kind"] == "demo_swap"):
        raise ValueError("demo_swap needs distinct intents; placebo pairs need the same intent")
    return left, right


def _replay_error(prediction, actions, mask):
    if (not isinstance(prediction, torch.Tensor) or prediction.shape != actions.shape
            or not prediction.is_floating_point() or not torch.isfinite(prediction).all()):
        raise ValueError("π replay prediction must match the finite measured action chunk")
    errors = prediction.detach().double().cpu() - actions.double().cpu()
    selected = errors[mask.cpu()]
    return {"masked_mse": float(selected.square().mean()), "masked_mae": float(selected.abs().mean()),
            "valid_values": selected.numel()}


def _object_evidence(value):
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {"source", "provenance", "measurements"}
            or value["source"] not in {"measured_object_state", "visual_object_relation"}
            or not isinstance(value["provenance"], str) or not value["provenance"].strip()
            or not isinstance(value["measurements"], list) or not value["measurements"]):
        raise ValueError("object evidence needs independent object state/relation measurements and provenance")
    for item in value["measurements"]:
        if value["source"] == "measured_object_state":
            if (not isinstance(item, dict) or set(item) != {"object_id", "position_m"}
                    or not isinstance(item["object_id"], str) or not item["object_id"].strip()
                    or not isinstance(item["position_m"], list) or len(item["position_m"]) != 3
                    or any(type(number) not in (int, float) or not math.isfinite(number)
                           for number in item["position_m"])):
                raise ValueError("object state evidence needs object_id and a finite 3D position_m")
        elif (not isinstance(item, dict) or set(item) != {"object_id", "relation", "observed"}
                or any(not isinstance(item[key], str) or not item[key].strip()
                       for key in ("object_id", "relation")) or type(item["observed"]) is not bool):
            raise ValueError("relation evidence needs object_id, relation and measured observed boolean")
    return value


def _validate_cases(cases):
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("evaluation needs nonempty cases with unique ids")
    for case in cases:
        _labels(vars(case), ("id", "scene", "task", "layout", "intent", "source_id"))
        for key in ("demonstration", "history"):
            value = getattr(case, key)
            if (not isinstance(value, torch.Tensor) or value.ndim != 5 or value.shape[0] != 1
                    or min(value.shape) < 1 or not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError("evaluation demo/history must be finite floating [1,C,F,H,W]")
        if case.demonstration.shape[1] != case.history.shape[1]:
            raise ValueError("evaluation demo and history channels differ")
        if (case.state.ndim != 2 or case.state.shape[0] != 1 or not torch.isfinite(case.state).all()
                or case.history_times.shape != (case.history.shape[2],)
                or not torch.isfinite(case.history_times).all()):
            raise ValueError("evaluation state/history times must match a single observed history")
        if case.truth["z"].shape[0] != 1:
            raise ValueError("evaluation goals must describe one case at a time")
        goal_distances(case.truth, case.current)
        if (case.actions is None) != (case.actions_mask is None):
            raise ValueError("action replay requires both measured actions and actions_mask")
        if case.actions is not None:
            if (case.actions.ndim != 5 or case.actions.shape[0] != 1 or case.actions.shape[-1] != 1
                    or not case.actions.is_floating_point() or not torch.isfinite(case.actions).all()
                    or case.actions_mask.shape != case.actions.shape or case.actions_mask.dtype != torch.bool
                    or not case.actions_mask.any()):
                raise ValueError("action replay needs finite [1,A,F,N,1] actions and nonempty boolean mask")
        if case.wrong_goal is not None:
            goal_distances(case.truth, case.wrong_goal)
        _object_evidence(case.object_evidence)


@torch.no_grad()
def evaluate_records(cases, training_goals, *, g_predict, thresholds, pi_predict=None,
                     pairs=(), quadruples=(), diagnose_intent=None):
    """Run model callbacks on observed tensors; offline grouping never reaches G/π.

    g_predict receives (demo, history, state). pi_predict receives
    (history, state, language, goal, history_times); its stochastic seed must be
    reset for each invocation to make counterfactual replay comparisons paired.
    """
    if not isinstance(thresholds, GoalThresholds):
        raise ValueError("evaluation requires explicit calibrated/noise-floor goal thresholds")
    cases = list(cases)
    _validate_cases(cases)
    modal = scene_modal_goals(training_goals, thresholds)
    if {case.source_id for case in cases} & {row["source_id"] for row in training_goals}:
        raise ValueError("training modal-goal sources must be disjoint from evaluation sources")
    predictions, rows, prefixes, replays, bypass = {}, [], [], [], []
    for case in cases:
        key = (case.scene, case.subgoal)
        if key not in modal:
            raise ValueError(f"missing training-only scene modal baseline for {key}")
        own, other = _regions(case, cases)
        prediction = g_predict(case.demonstration, case.history, case.state)
        goal_distances(prediction, case.truth)
        predictions[case.id] = prediction
        if diagnose_intent is not None:
            interventions = diagnose_intent(case.demonstration, case.history, case.state)
            if set(interventions) != {"baseline", "u_permuted", "robot_without_demo"}:
                raise ValueError("intent diagnostic requires baseline, u_permuted and robot_without_demo predictions")
            for name in ("u_permuted", "robot_without_demo"):
                bypass.append({"id": case.id, "scene": case.scene, "task": case.task,
                    "intervention": name, "change": goal_distances(interventions[name], interventions["baseline"]),
                    "truth_error": goal_distances(interventions[name], case.truth),
                    "baseline_truth_error": goal_distances(interventions["baseline"], case.truth)})
        results = {}
        for name, goal in (("g", prediction), ("current", case.current), ("scene_modal", modal[key]["goal"])):
            results[name] = {**delta_metrics(goal, case.truth, case.current),
                             "region": valid_goal_region(goal, own, other, thresholds)}
        row = {key: getattr(case, key) for key in ("id", "scene", "task", "layout", "intent", "source_id", "subgoal")}
        row.update(results=results, scene_modal_source=modal[key]["source_id"],
                   object_evidence=_object_evidence(case.object_evidence))
        rows.append(row)
        for percent in range(10, 101, 10):
            count = max(1, math.ceil(case.demonstration.shape[2] * percent / 100))
            # Clone physically excludes future demo frames, including backing storage.
            truncated = case.demonstration[:, :, :count].clone()
            prefix_goal = g_predict(truncated, case.history, case.state)
            prefixes.append({"id": case.id, "scene": case.scene, "task": case.task,
                "percent": percent, "latent_frames": count,
                "valid": valid_goal_region(prefix_goal, own, other, thresholds)["valid"],
                "error": goal_distances(prefix_goal, case.truth)})
        if case.actions is not None and pi_predict is not None:
            paths = (("oracle", case.truth), ("g_predicted", prediction))
            for name, goal in paths:
                actions = pi_predict(case.history, case.state, None, goal, case.history_times)
                replays.append({"id": case.id, "scene": case.scene, "task": case.task,
                    "path": name, "language": "empty", "goal": "correct" if name == "oracle" else "predicted",
                    **_replay_error(actions, case.actions, case.actions_mask)})
            if case.wrong_goal is not None:
                if case.language is None:
                    raise ValueError("the instruction×goal diagnostic requires the correct language embedding")
                if goal_reached(case.wrong_goal, case.truth, thresholds):
                    raise ValueError("wrong_goal must lie outside the correct goal's noise region")
                for language_name, language in (("correct", case.language), ("empty", None)):
                    for goal_name, goal in (("correct", case.truth), ("wrong", case.wrong_goal)):
                        actions = pi_predict(case.history, case.state, language, goal, case.history_times)
                        replays.append({"id": case.id, "scene": case.scene, "task": case.task,
                            "path": "instruction_goal_2x2", "language": language_name, "goal": goal_name,
                            **_replay_error(actions, case.actions, case.actions_mask)})
    indexed = {case.id: case for case in cases}
    pair_rows = []
    for pair in pairs:
        left, right = _pair_cases(pair, indexed)
        first, second = predictions[left.id], predictions[right.id]
        result = {"kind": pair["kind"], "ids": pair["ids"], "scene": left.scene, "task": left.task,
                  "truth_distance": _score(left.truth, right.truth, thresholds),
                  "truth_distances": goal_distances(left.truth, right.truth)}
        if pair["kind"] == "demo_swap":
            first_correct = _score(first, left.truth, thresholds) < _score(first, right.truth, thresholds)
            second_correct = _score(second, right.truth, thresholds) < _score(second, left.truth, thresholds)
            result.update(correct=first_correct and second_correct, first_correct=first_correct,
                          second_correct=second_correct)
        else:
            if not goal_reached(left.truth, right.truth, thresholds):
                raise ValueError("placebo ground-truth goals must agree within the measured noise floor")
            result["false_positive"] = not goal_reached(first, second, thresholds)
        pair_rows.append(result)
    quad_rows = []
    for ids in quadruples:
        if not isinstance(ids, list) or len(ids) != 4 or len(set(ids)) != 4 or any(key not in indexed for key in ids):
            raise ValueError("intent×layout quadruples require four distinct known case ids")
        selected = [indexed[key] for key in ids]
        intents, layouts = {case.intent for case in selected}, {case.layout for case in selected}
        if (len(intents) != 2 or len(layouts) != 2 or len({(case.intent, case.layout) for case in selected}) != 4
                or len({(case.scene, case.task, case.subgoal) for case in selected}) != 1):
            raise ValueError("quadruples must form two intents × two layouts in one scene/task/subgoal")
        for layout in layouts:
            same = [case for case in selected if case.layout == layout]
            if not _same_observation(*same):
                raise ValueError("quadruple intent changes must keep the layout observation fixed")
        for intent in intents:
            same = [case for case in selected if case.intent == intent]
            if not torch.equal(same[0].demonstration, same[1].demonstration):
                raise ValueError("quadruple layout changes must keep the demonstration fixed")
        valid = [valid_goal_region(predictions[case.id], *_regions(case, cases), thresholds)["valid"]
                 for case in selected]
        quad_rows.append({"ids": ids, "scene": selected[0].scene, "task": selected[0].task,
                          "all_correct": all(valid), "cell_correct": valid})
    summary = {}
    for method in ("g", "current", "scene_modal"):
        values = [{**row, "valid": row["results"][method]["region"]["valid"]} for row in rows]
        summary[method] = {"valid_region": grouped_statistics(values, "valid"),
            "per_subgoal": {str(index): grouped_statistics([row for row in values if row["subgoal"] == index], "valid")
                            for index in sorted({row["subgoal"] for row in values})},
            "errors": {metric: grouped_statistics([{**row, "value": row["results"][method]["error"][metric]}
                                                    for row in rows], "value") for metric in _METRICS}}
    typed = {kind: grouped_statistics([row for row in pair_rows if row["kind"] == kind],
              "correct" if kind == "demo_swap" else "false_positive") for kind in sorted(_PAIR_TYPES)}
    # Fixed noise-relative bins are comparable across runs; no test-fitted bins.
    edges = (0., 1., 2., 4., 8., 16., math.inf)
    swaps = [row for row in pair_rows if row["kind"] == "demo_swap"]
    resolution = [{"lower": lower, "upper": upper if math.isfinite(upper) else None,
                   **grouped_statistics([row for row in swaps if lower <= row["truth_distance"] < upper], "correct")}
                  for lower, upper in zip(edges[:-1], edges[1:])]
    prefix_curve = [{"percent": percent, **grouped_statistics([row for row in prefixes if row["percent"] == percent], "valid")}
                    for percent in range(10, 101, 10)]
    replay_summary = []
    for path, language, goal in sorted({(row["path"], row["language"], row["goal"]) for row in replays}):
        selected = [row for row in replays if (row["path"], row["language"], row["goal"]) == (path, language, goal)]
        replay_summary.append({"path": path, "language": language, "goal": goal,
            "masked_mse": grouped_statistics(selected, "masked_mse"),
            "masked_mae": grouped_statistics(selected, "masked_mae")})
    return {"format_version": 1, "kind": "g_pi_evaluation_result", "thresholds": asdict(thresholds),
            "summary": summary, "cases": rows, "pairs": pair_rows, "pair_summary": typed,
            "quadruples": quad_rows, "quadruple_summary": grouped_statistics(quad_rows, "all_correct"),
            "resolution_curve": resolution, "prefix_curve": prefix_curve, "prefix_cases": prefixes,
            "action_replays": replays, "action_replay_summary": replay_summary,
            "intent_bypass": bypass,
            "intent_bypass_status": {"status": "measured" if diagnose_intent is not None else "not_requested"},
            "intent_bypass_summary": {name: {measurement: {metric: grouped_statistics(
                [{**row, "value": row[measurement][metric]} for row in bypass if row["intervention"] == name], "value")
                for metric in _METRICS} for measurement in ("change", "truth_error", "baseline_truth_error")}
                for name in ("u_permuted", "robot_without_demo")},
            "replay_language_default": "empty", "policy_success": None,
            "limitations": ["Action replay errors do not measure closed-loop task success.",
                "Object evidence describes the recorded reference, never a generated rollout.",
                "Goal-region exclusion covers only the competing intent references provided.",
                "Scene/task macro means treat repeated views and seeds as dependent observations.",
                "Intent interventions measure sensitivity, not causal necessity or pathway usage proof.",
                "u_permuted holds robot features fixed; robot_without_demo recomputes native robot features with u fixed."]}


def write_evaluation_plots(report, output):
    """A dependency-free SVG of both diagnostic curves, beside the JSON report."""
    panels = (("Intent separation / measured noise", report["resolution_curve"]),
              ("Demo prefix (%)", report["prefix_curve"]))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="330" viewBox="0 0 1000 330">',
           '<rect width="1000" height="330" fill="white"/>']
    for panel, (title, rows) in enumerate(panels):
        x0, y0, width, height = 55 + panel * 500, 260, 410, 210
        svg += [f'<path d="M{x0} 50 V{y0} H{x0 + width}" stroke="#444" fill="none"/>',
                f'<text x="{x0}" y="25" font-size="16">{html.escape(title)}</text>']
        points = []
        for index, row in enumerate(rows):
            x = x0 + width * (index + .5) / len(rows)
            label = str(row["percent"]) if panel else f'{row["lower"]:g}–{row["upper"] if row["upper"] is not None else "∞"}'
            svg.append(f'<text x="{x}" y="280" text-anchor="middle" font-size="11">{html.escape(label)}</text>')
            if row["macro_mean"] is not None:
                y = y0 - height * row["macro_mean"]
                points.append(f"{x},{y}")
                svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#176b9c"><title>scene/tasks={row["independent_scene_tasks"]}</title></circle>')
        if points:
            svg.append(f'<polyline points="{" ".join(points)}" stroke="#176b9c" fill="none"/>')
        for value in (0, .5, 1):
            svg.append(f'<text x="{x0 - 8}" y="{y0 - height * value + 4}" text-anchor="end" font-size="11">{value:g}</text>')
    svg.append('<text x="55" y="315" font-size="12">Y: scene/task macro correctness; missing bins are unmeasured, not zero.</text></svg>')
    Path(output).write_text("\n".join(svg))


def _keys(value, required, optional, description):
    if not isinstance(value, dict) or required - value.keys() or value.keys() - required - optional:
        raise ValueError(f"{description} has missing or unknown fields")


def _load_replay(path, metadata, case):
    _keys(metadata, {"arrays", "subgoal_time"}, set(), "action replay")
    target = _local_path(path.parent, metadata["arrays"], ".npz")
    with np.load(target, allow_pickle=False) as archive:
        if set(archive.files) != {"actions", "actions_mask"}:
            raise ValueError("action replay NPZ needs exactly actions and actions_mask")
        actions, mask = (torch.from_numpy(archive[key].copy()) for key in ("actions", "actions_mask"))
    if actions.ndim != 5 or mask.shape != actions.shape or mask.dtype != torch.bool:
        raise ValueError("action replay arrays must share [1,A,F,N,1] shape and boolean mask")
    end, now, dt = metadata["subgoal_time"], case.metadata["current_time"], case.metadata["control_dt"]
    if type(end) not in (int, float) or not math.isfinite(end) or end <= now:
        raise ValueError("action replay subgoal_time must be strictly after current_time")
    if not math.isclose(round(end / dt) * dt, end, abs_tol=min(1e-6, dt * 1e-4), rel_tol=0):
        raise ValueError("action replay subgoal_time must lie on the control grid")
    times = now + torch.arange(actions.shape[2] * actions.shape[3], dtype=torch.float64) * dt
    allowed = (times < end - min(1e-6, dt * 1e-4)).reshape(1, 1, actions.shape[2], actions.shape[3], 1)
    if (mask & ~allowed).any():
        raise ValueError("action replay mask must exclude actions at/after the subgoal switch")
    return actions, mask, target


@torch.no_grad()
def evaluate_g_pi_cli(args):
    from .g_pi_interface import GTranslator, PiGoalPolicy
    from .g_pi_training import load_g_pi_policy

    path, output = Path(args.manifest), Path(args.output)
    if output.suffix != ".json" or output.exists() or output.with_suffix(".svg").exists():
        raise ValueError("evaluation requires fresh .json and .svg output paths")
    manifest = json.loads(path.read_text())
    required = {"format_version", "kind", "split", "g_policy", "encoder_identity", "registry", "thresholds",
                "training_goals", "cases", "pairs", "quadruples"}
    _keys(manifest, required, {"pi_policy", "seed"}, "g_pi_evaluation manifest")
    if type(manifest["format_version"]) is not int or manifest["format_version"] != 1 or manifest["kind"] != "g_pi_evaluation" or manifest["split"] not in {"validation", "test"}:
        raise ValueError("expected version-1 g_pi_evaluation validation/test manifest")
    if any(not isinstance(manifest[key], list) for key in ("training_goals", "cases", "pairs", "quadruples")):
        raise ValueError("evaluation training_goals, cases, pairs and quadruples must be lists")
    if not isinstance(manifest["thresholds"], dict) or set(manifest["thresholds"]) != set(_METRICS):
        raise ValueError("evaluation thresholds must explicitly specify all four noise-floor distances")
    identity, registry = manifest["encoder_identity"], manifest["registry"]
    thresholds = GoalThresholds(**manifest["thresholds"])
    seed = manifest.get("seed", 0)
    if type(seed) is not int or seed < 0:
        raise ValueError("evaluation seed must be a nonnegative integer")
    files = {path}
    def local(value, suffix=".json"):
        resolved = _local_path(path.parent, value, suffix)
        files.add(resolved)
        return resolved
    def goal(value):
        filename = local(value)
        result = load_goal_prediction(filename, encoder_identity=identity, registry=registry)
        files.add(_local_path(filename.parent, json.loads(filename.read_text())["arrays"], ".npz"))
        return result
    def policy_directory(value):
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise ValueError("evaluation policy must be a relative exported policy directory")
        folder = (path.parent / value).resolve()
        if not folder.is_relative_to(path.parent.resolve()) or not folder.is_dir():
            raise ValueError("evaluation policy directory must remain within the manifest directory")
        sidecar = folder / "policy.json"
        policy_metadata = json.loads(sidecar.read_text())
        files.add(sidecar)
        for filename in policy_metadata["shards"]:
            files.add(_local_path(folder, filename, ".safetensors"))
        return folder
    pi, shared_base = None, None
    if "pi_policy" in manifest:
        pi_native, interface, pi_encoder, pi_payload = load_g_pi_policy(policy_directory(manifest["pi_policy"]),
            device=args.device, checkpoint=getattr(args, "pi_checkpoint", None), expected_encoder_identity=identity)
        pi_registry = pi_payload["registry"]
        shared = lambda value: {key: entry for key, entry in value.items() if key != "language_identity"}
        if pi_payload["config"]["interface_type"] != "pi_goal" or shared(pi_registry) != shared(registry):
            raise ValueError("evaluation π policy route/registry differs from the manifest")
        shape = (1, pi_native.config.action_dim, pi_payload["config"]["chunk_size"], registry["actions_per_frame"], 1)
        mask = torch.tensor(registry["action_space"]["valid_channels"], dtype=torch.bool)[None, :, None, None, None]
        pi = PiGoalPolicy(pi_native, interface, pi_payload["config"], action_shape=shape, actions_mask=mask,
                          seed=seed, video_native=pi_encoder.native)
        shared_base = (pi_native, pi_encoder, pi_payload)
    g_path = policy_directory(manifest["g_policy"])
    native, decoder, encoder, payload = load_g_pi_policy(g_path, device=args.device,
        checkpoint=getattr(args, "g_checkpoint", None), expected_encoder_identity=identity,
        **({"shared_base": shared_base} if shared_base is not None else {}))
    if payload["config"]["interface_type"] != "g_translator" or payload["registry"] != registry:
        raise ValueError("evaluation G policy route/registry differs from the manifest")
    if shared_base is not None and (native is not pi_native or encoder is not pi_encoder):
        raise ValueError("evaluation G, π and E must share one native base")
    g = GTranslator(native, decoder, payload["config"], feature_layer=payload["config"]["goal_encoder"]["layer"])
    training = []
    for row in manifest["training_goals"]:
        _keys(row, {"scene", "task", "source_id", "subgoal", "split", "goal"}, set(), "training modal goal")
        training.append({**row, "goal": goal(row["goal"])})
    cases = []
    for row in manifest["cases"]:
        required_case = {"id", "scene", "task", "layout", "intent", "source_id", "subgoal", "observation", "truth", "current"}
        _keys(row, required_case, {"language", "action_replay", "wrong_goal", "object_evidence"}, "evaluation case")
        observation_path = local(row["observation"])
        observation = load_g_pi_observation(observation_path, payload)
        metadata = observation.metadata
        files.add(_local_path(observation_path.parent, metadata["arrays"], ".npz"))
        files.add(_local_path(observation_path.parent, metadata["demonstration"]["arrays"], ".npz"))
        language, actions, mask = None, None, None
        if "language" in row:
            language_path = local(row["language"])
            language, language_identity = load_goal_language(language_path)
            if pi is None or language_identity != pi_registry["language_identity"]:
                raise ValueError("evaluation language encoder differs from the policy")
            files.add(_local_path(language_path.parent, json.loads(language_path.read_text())["arrays"], ".npz"))
        if "action_replay" in row:
            if pi is None:
                raise ValueError("action replay requires an exported π policy")
            actions, mask, replay_path = _load_replay(path, row["action_replay"], observation)
            if actions.shape != shape or actions.shape[3] != registry["actions_per_frame"]:
                raise ValueError("action replay F,N layout differs from the trained π chunk")
            channel_mask = torch.tensor(registry["action_space"]["valid_channels"], dtype=torch.bool)[None, :, None, None, None]
            if (mask & ~channel_mask).any():
                raise ValueError("action replay mask enables invalid action channels")
            files.add(replay_path)
        cases.append(EvaluationCase(**{key: row[key] for key in ("id", "scene", "task", "layout", "intent", "source_id", "subgoal")},
            demonstration=observation.demonstration, history=observation.history, state=observation.state,
            history_times=observation.history_times, current=goal(row["current"]), truth=goal(row["truth"]),
            language=language, actions=actions, actions_mask=mask,
            wrong_goal=goal(row["wrong_goal"]) if "wrong_goal" in row else None,
            object_evidence=row.get("object_evidence")))
    def pi_predict(history, state, language, target, times):
        pi.generator.manual_seed(seed)
        return pi.predict(history, state, language, target, frame_times=times)
    result = evaluate_records(cases, training, g_predict=g.predict, thresholds=thresholds,
        pi_predict=pi_predict if pi is not None else None, pairs=manifest["pairs"], quadruples=manifest["quadruples"],
        diagnose_intent=g.diagnose_intent if decoder.intent_mode == "connected" else None)
    if decoder.intent_mode != "connected":
        result["intent_bypass_status"] = {"status": "not_applicable", "intent_mode": decoder.intent_mode,
                                          "reason": "The goal path has no connected u in this ablation."}
    result.update(encoder_identity=encoder.identity, registry=registry, seed=seed, split=manifest["split"],
                  shared_video_base=pi is not None and native is pi_native and encoder is pi_encoder,
                  input_sha256={str(filename.resolve()): file_sha256(filename) for filename in sorted(files)})
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    write_evaluation_plots(result, output.with_suffix(".svg"))
    return {"output": str(output.resolve()), "plot": str(output.with_suffix(".svg").resolve()),
            "cases": len(cases), "commands_sent": 0, "policy_success": None}
