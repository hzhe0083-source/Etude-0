"""Offline whole-demonstration purpose groups and ordered-slot contrastive loss."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .icl_data import LATENT_NORMALIZATION, _arrays, _identity
from .video_data import SPLITS, _local_path, _source_components, _text


@dataclass(frozen=True)
class IntentEntry:
    demo_id: str
    purpose_group: str
    split: str
    source_id: str
    source_group: str
    person_id: str
    scene_id: str
    view_id: str
    object_ids: tuple[str, ...]
    arrays: Path
    paired_task: Path | None
    component: int
    metadata: Mapping


@dataclass(frozen=True)
class IntentTable:
    path: Path
    metadata: Mapping
    entries: tuple[IntentEntry, ...]
    uncertain_pairs: frozenset[frozenset[str]]
    source_records: tuple[Mapping, ...] = ()


@dataclass(frozen=True)
class IntentDemo:
    demonstration: Tensor
    demonstration_times: Tensor


def load_intent_demo(entry: IntentEntry) -> IntentDemo:
    """Only image latents and capture times cross this model-input boundary."""
    arrays = _arrays(entry.arrays, {"arrays": entry.arrays.name}, robot_target=False)
    return IntentDemo(arrays["latent"].unsqueeze(0), arrays["frame_times"])


def _demo_digest(entry: IntentEntry) -> str:
    demo = load_intent_demo(entry)
    digest = hashlib.sha256()
    for value in (demo.demonstration, demo.demonstration_times):
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_intent_table(manifest_path: str | Path) -> IntentTable:
    """Audit offline supervision, including transitive aliases and copied videos."""
    path = Path(manifest_path).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    required = {"format_version", "kind", "data_version", "feature_space_id",
                "latent_normalization", "grouping_evidence", "entries"}
    if (not isinstance(document, dict) or required - set(document)
            or set(document) - required - {"source_aliases", "uncertain_pairs"}
            or type(document.get("format_version")) is not int or document["format_version"] != 1
            or document["kind"] != "g_pi_intent_groups" or document["data_version"] not in {"v1", "v2"}):
        raise ValueError("expected version-1 g_pi_intent_groups with data_version v1 or v2")
    for name in ("feature_space_id", "grouping_evidence"):
        _text(document, name)
    if document["latent_normalization"] != LATENT_NORMALIZATION:
        raise ValueError("purpose demonstrations must use the native latent_normalization")
    if not isinstance(document["entries"], list) or not document["entries"]:
        raise ValueError("purpose groups require nonempty entries of complete demonstrations")
    entries, records, robot_records, seen = [], [], [], set()
    required_entry = {"demo_id", "purpose_group", "split", "source_id", "source_group",
                      "person_id", "scene_id", "view_id", "object_ids", "arrays", "complete_demo"}
    optional_entry = {"paired_task", "object_family", "role_candidates", "operation",
                      "relation_signature", "role_signature", "order_signature"}
    for record in document["entries"]:
        if (not isinstance(record, dict) or required_entry - set(record)
                or set(record) - required_entry - optional_entry or record["complete_demo"] is not True):
            raise ValueError("intent entries require complete_demo=true and explicit offline source/purpose identities")
        for name in required_entry - {"object_ids", "complete_demo"}:
            _text(record, name)
        if record["demo_id"] in seen or record["split"] not in SPLITS:
            raise ValueError("demo_id must be unique and split must be train, validation or test")
        seen.add(record["demo_id"])
        objects = record["object_ids"]
        if (not isinstance(objects, list) or not objects
                or any(not isinstance(value, str) or not value.strip() for value in objects)
                or len(set(objects)) != len(objects)):
            raise ValueError("object_ids must be a nonempty list of distinct audited object identities")
        for name in {"object_family", "operation", "relation_signature", "role_signature", "order_signature"} & set(record):
            _text(record, name)
        if document["data_version"] == "v1":
            _text(record, "object_family")
        candidates = record.get("role_candidates")
        if document["data_version"] == "v2" or candidates is not None:
            if (not isinstance(candidates, dict) or not candidates
                    or any(not isinstance(role, str) or not role.strip() or type(count) is not int or count < 1
                           or (document["data_version"] == "v2" and record["split"] == "train" and count != 1)
                           for role, count in candidates.items())):
                raise ValueError("v2 training scenes require exactly one candidate per audited role")
        if document["data_version"] == "v2":
            _text(record, "operation")
        arrays = _local_path(path.parent, record["arrays"], ".npz")
        paired = _local_path(path.parent, _text(record, "paired_task"), ".json") if "paired_task" in record else None
        entry = IntentEntry(*(record[name] for name in ("demo_id", "purpose_group", "split", "source_id",
                            "source_group", "person_id", "scene_id", "view_id")),
                            tuple(objects), arrays, paired, -1, dict(record))
        entries.append(entry)
        records.append({"source_id": entry.source_id, "source_group": entry.source_group,
                        "domain": "human", "split": entry.split})
        if paired is not None:
            from .g_pi_data import _metadata
            task = _metadata(paired)
            demo = task.get("demonstration", {})
            if (demo.get("source_id") != entry.source_id or demo.get("source_group") != entry.source_group
                    or _local_path(paired.parent, demo.get("arrays"), ".npz") != arrays):
                raise ValueError("paired_task must reference exactly this complete demonstration and its source identity")
            if task["feature_space_id"] != document["feature_space_id"]:
                raise ValueError("paired_task and purpose table feature_space_id must match")
            robot_records.append({**task["robot_source"], "split": entry.split})
    aliases = document.get("source_aliases", [])
    if not isinstance(aliases, list):
        raise ValueError("source_aliases must be a list of human provenance records")
    for alias in aliases:
        if (not isinstance(alias, dict) or set(alias) != {"source_id", "source_group", "domain", "split"}
                or alias.get("domain") != "human"):
            raise ValueError("source_aliases require human source_id, source_group, domain and split")
        _identity(alias)
        records.append(dict(alias))
    # Copies with renamed provenance cannot become positives or cross a split.
    digests = {}
    for entry in entries:
        duplicate = digests.setdefault(_demo_digest(entry), entry)
        if duplicate is not entry:
            records.append({"source_id": duplicate.source_id, "source_group": entry.source_group,
                            "domain": "human", "split": entry.split})
    components = _source_components(records)
    _source_components(records + robot_records)
    component_ids = {index: number for number, members in enumerate(components) for index in members}
    entries = [replace(entry, component=component_ids[index]) for index, entry in enumerate(entries)]
    purposes = {}
    for entry in entries:
        previous = purposes.setdefault(entry.component, entry.purpose_group)
        if previous != entry.purpose_group:
            raise ValueError("one complete source component cannot have conflicting purpose groups")
    if document["data_version"] == "v1":
        families = {}
        for entry in entries:
            old = families.setdefault(entry.purpose_group, entry.metadata["object_family"])
            if old != entry.metadata["object_family"]:
                raise ValueError("v1 positives require similar objects within one audited object_family")
    uncertain = document.get("uncertain_pairs", [])
    if not isinstance(uncertain, list):
        raise ValueError("uncertain_pairs must be a list of demo_id pairs")
    excluded = set()
    for pair in uncertain:
        if (not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(value, str) for value in pair)
                or pair[0] == pair[1] or not set(pair) <= seen):
            raise ValueError("uncertain_pairs must reference two different known demo_id values")
        if frozenset(pair) in excluded:
            raise ValueError("uncertain_pairs must not contain duplicate pairs")
        excluded.add(frozenset(pair))
    return IntentTable(path, document, tuple(entries), frozenset(excluded), tuple(records + robot_records))


def intent_source_records(table: IntentTable) -> list[dict]:
    """Preserve full alias/copy closure when auditing against robot indices."""
    return [dict(record) for record in table.source_records]


def intent_table_files(table: IntentTable) -> list[Path]:
    """Dependencies whose content identifies contrastive sampling and resume."""
    paths = {table.path}
    for entry in table.entries:
        paths.add(entry.arrays)
        if entry.paired_task is not None:
            from .g_pi_data import _metadata
            path = entry.paired_task
            metadata = _metadata(path)
            paths.update((path, _local_path(path.parent, metadata["arrays"], ".npz")))
    return sorted(paths)


def _independent(a: IntentEntry, b: IntentEntry, uncertain) -> bool:
    return a.component != b.component and frozenset((a.demo_id, b.demo_id)) not in uncertain


def _hardness(a: IntentEntry, b: IntentEntry) -> tuple[int, int, int]:
    same_objects = bool(set(a.object_ids) & set(b.object_ids))
    different_structure = any(a.metadata.get(key) is not None and b.metadata.get(key) is not None
                              and a.metadata[key] != b.metadata[key]
                              for key in ("relation_signature", "role_signature", "order_signature"))
    return int(same_objects and different_structure), int(same_objects), int(a.scene_id == b.scene_id)


def sample_intent_batch(table: IntentTable, generator: torch.Generator, *, groups_per_batch: int = 2,
                        samples_per_group: int = 2, split: str = "train") -> tuple[IntentEntry, ...]:
    """Sample independent positives; prefer confusable and same-scene negatives."""
    if (type(groups_per_batch) is not int or groups_per_batch < 2
            or type(samples_per_group) is not int or samples_per_group < 2 or split not in SPLITS):
        raise ValueError("contrastive batches need >=2 purpose groups and >=2 independent examples per group")
    by_group = {}
    for entry in table.entries:
        if entry.split == split:
            by_group.setdefault(entry.purpose_group, []).append(entry)
    candidates = {}
    for name, values in by_group.items():
        shuffled = [values[index] for index in torch.randperm(len(values), generator=generator).tolist()]
        for first in shuffled:
            chosen = [first]
            others = [value for value in shuffled if value is not first]
            others.sort(key=lambda value: (value.person_id != first.person_id, value.scene_id != first.scene_id,
                                          value.view_id != first.view_id, value.object_ids != first.object_ids), reverse=True)
            for value in others:
                if all(_independent(value, other, table.uncertain_pairs) for other in chosen):
                    chosen.append(value)
                if len(chosen) == samples_per_group:
                    break
            diverse = any((value.person_id, value.scene_id, value.view_id) !=
                          (first.person_id, first.scene_id, first.view_id) for value in chosen[1:])
            if table.metadata["data_version"] == "v2":
                diverse = any(set(value.object_ids) != set(first.object_ids) for value in chosen[1:])
            if len(chosen) == samples_per_group and diverse:
                packs = candidates.setdefault(name, [])
                if not any({value.demo_id for value in chosen} == {value.demo_id for value in pack} for pack in packs):
                    packs.append(tuple(chosen))
    if len(candidates) < groups_per_batch:
        raise ValueError("not enough purpose groups with independent, diverse, non-uncertain positives")
    def scene_negative(pack, others):
        return any(a.scene_id == b.scene_id and _independent(a, b, table.uncertain_pairs)
                   for a in pack for b in others)

    # Keep alternative representatives: choosing one random clip per group can
    # silently discard an available same-scene negative or every valid pair.
    shared = {}
    for name, packs in candidates.items():
        others = [entry for other, values in by_group.items() if other != name and other in candidates
                  for entry in values]
        shared[name] = [pack for pack in packs if scene_negative(pack, others)]
    names = [name for name in candidates if shared[name]] or list(candidates)
    first_index = int(torch.randint(len(names), (), generator=generator))
    names = names[first_index:] + names[:first_index]
    for first in names:
        for initial in shared[first] + [pack for pack in candidates[first] if pack not in shared[first]]:
            selected, selected_groups = list(initial), {first}
            same_scene = False
            for _ in range(groups_per_batch - 1):
                eligible = [(name, pack) for name, packs in candidates.items() if name not in selected_groups
                            for pack in packs
                            if all(any(_independent(a, b, table.uncertain_pairs) for b in selected) for a in pack)
                            and all(any(_independent(a, b, table.uncertain_pairs) for a in pack) for b in selected)]
                if not eligible:
                    break
                eligible = [eligible[index] for index in torch.randperm(len(eligible), generator=generator).tolist()]
                name, pack = max(eligible, key=lambda item: (
                    int(not same_scene and scene_negative(item[1], selected)),
                    max(_hardness(a, b) for a in item[1] for b in selected
                        if _independent(a, b, table.uncertain_pairs))))
                same_scene = same_scene or scene_negative(pack, selected)
                selected.extend(pack)
                selected_groups.add(name)
            if len(selected_groups) == groups_per_batch:
                return tuple(selected)
    raise ValueError("uncertain pairs leave no independent negative for every contrastive anchor")


def ordered_intent_similarity(u: Tensor) -> Tensor:
    """Mean cosine of matching slots; swapping roles/steps changes similarity."""
    if (u.ndim != 3 or min(u.shape) < 1 or not u.is_floating_point() or not torch.isfinite(u).all()):
        raise ValueError("intent u must be finite floating [B,M_u,D] with fixed ordered slots")
    ordered = F.normalize(u.float(), dim=-1).flatten(1) / math.sqrt(u.shape[1])
    return ordered @ ordered.T


def intent_contrastive_loss(u: Tensor, entries: Sequence[IntentEntry], *, temperature: float = .1,
                            uncertain_pairs=frozenset()) -> Tensor:
    """Supervised contrastive loss without source copies or uncertain pairs."""
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("contrastive temperature must be finite and positive")
    similarity = ordered_intent_similarity(u)
    if len(entries) != u.shape[0] or len({entry.demo_id for entry in entries}) != len(entries):
        raise ValueError("intent entries must identify each distinct batch row")
    allowed = torch.tensor([[_independent(a, b, uncertain_pairs) for b in entries] for a in entries],
                           dtype=torch.bool, device=u.device)
    positive = allowed & torch.tensor([[a.purpose_group == b.purpose_group for b in entries] for a in entries],
                                     dtype=torch.bool, device=u.device)
    if not positive.any(1).all() or not (allowed & ~positive).any(1).all():
        raise ValueError("each contrastive anchor needs an independent positive and a non-uncertain negative")
    logits = similarity / temperature
    denominator = torch.logsumexp(logits.masked_fill(~allowed, float("-inf")), dim=1)
    log_probability = logits - denominator[:, None]
    return -(log_probability.masked_fill(~positive, 0).sum(1) / positive.sum(1)).mean()
