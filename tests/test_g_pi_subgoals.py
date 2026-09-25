from copy import deepcopy
from dataclasses import asdict, fields
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from etude.g_pi_data import EventRules, PiGoalSample, load_g_pi_sample
from etude.g_pi_subgoals import (CANDIDATE_THRESHOLDS, RELATION_REGISTRY, SIM_THRESHOLDS, VERSIONS,
                                   generate_candidate_annotation, generate_candidates,
                                   resolve_subgoal_indices, validate_relations)
from etude.icl_preprocess import encode_g_pi_frames
from test_g_pi_data import write_g_pi_task


def relation(predicate, role="cup", occurrence=1, **extra):
    return {"predicate": predicate, "object_role": role, "effector": 0, "occurrence": occurrence, **extra}


def evidence(frames=17):
    poses = torch.eye(4).repeat(frames, 1, 1, 1)
    return dict(control_times=torch.arange(frames, dtype=torch.float64) * .1,
                gripper=torch.ones(frames, 1) * .9, poses=poses)


def annotation(source, relations, **extra):
    return dict(detector_version=VERSIONS[source], evidence="fixture recorded evidence",
                stable_steps=2, thresholds=dict(SIM_THRESHOLDS), relations=relations,
                relation_registry=RELATION_REGISTRY, weak_label=False, **extra)


class SubgoalSourcesTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def resolve(self, source, audit, arrays, check=False):
        return resolve_subgoal_indices(dict(subgoal_source=source, subgoal_annotation=audit,
                                            event_rules=asdict(EventRules())), arrays, check_indices=check)

    def sim(self):
        arrays = evidence()
        arrays["gripper"][5:10] = .3
        arrays["poses"][:, 0, 2, 3] = .03
        arrays["object_positions"] = torch.zeros(17, 2, 3)
        arrays["object_positions"][5:, 0, 2] = .03
        arrays["object_extents"] = torch.tensor([[.01, .01, .01], [.2, .2, .02]])
        audit = annotation("sim_relation", [relation("in_hand"), relation("placed", relation="on", target_role="tray")],
                           object_roles=["cup", "tray"], geometry="axis_aligned_robot_base_boxes")
        return arrays, audit

    def candidate(self):
        arrays = evidence()
        arrays["gripper"][3:8] = .3
        audit = annotation("candidate_match", [relation("in_hand"), relation("placed", relation="in", target_role="bowl")],
                           template_version="pick_place_v1")
        audit.update(weak_label=True, thresholds=dict(CANDIDATE_THRESHOLDS))
        audit["candidates"] = generate_candidates(arrays, EventRules(), audit["thresholds"], audit["stable_steps"])
        return arrays, audit

    def test_default_gripper_source_preserves_events_and_complete_audit(self):
        arrays = evidence()
        arrays["gripper"][:5] = 0.
        indices, audit = resolve_subgoal_indices({"event_rules": asdict(EventRules())}, arrays)
        self.assertEqual(indices.tolist(), [6, 16])
        self.assertEqual(audit["control_indices"], [6, 16])
        self.assertEqual(audit["detector_version"], VERSIONS["gripper"])
        self.assertTrue(audit["weak_label"])
        audit["control_indices"] = [5, 16]
        with self.assertRaisesRegex(ValueError, "recomputed measured"):
            self.resolve("gripper", audit, arrays, check=True)

    def test_command_gripper_cannot_become_measured_grasp_evidence(self):
        arrays, _ = self.candidate()
        arrays["gripper"][3:8] = .1
        with self.assertRaisesRegex(ValueError, "requires measured"):
            generate_candidates(arrays, EventRules(signal_source="command"))
        metadata = {"event_rules": asdict(EventRules(signal_source="command")),
                    "gripper_signal_source": "command",
                    "gripper_source_evidence": "fixture commanded finger target"}
        indices, audit = resolve_subgoal_indices(metadata, arrays)
        self.assertEqual(audit["detector_version"], "command_gripper_v1")
        self.assertEqual(audit["evidence"], "gripper:command")
        self.assertTrue(audit["weak_label"])
        self.assertEqual(indices.tolist(), [4, 9, 16])
        for source in ("sim_relation", "candidate_match"):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "command gripper"):
                resolve_subgoal_indices({**metadata, "subgoal_source": source}, arrays)

    def test_sim_geometry_recomputed_and_confirmed_without_backdating(self):
        arrays, audit = self.sim()
        indices, audit = self.resolve("sim_relation", audit, arrays)
        self.assertEqual(indices.tolist(), [6, 11, 16])
        self.assertEqual(audit["relation_control_indices"], [6, 11])
        self.resolve("sim_relation", audit, arrays, check=True)
        changed = {**arrays, "object_positions": arrays["object_positions"].clone()}
        changed["object_positions"][5, 0, 2] = 0.
        with self.assertRaisesRegex(ValueError, "recomputed"):
            self.resolve("sim_relation", audit, changed, check=True)
        with self.assertRaisesRegex(ValueError, "object_positions"):
            self.resolve("sim_relation", audit, {key: value for key, value in arrays.items() if key != "object_positions"})

    def test_initial_closed_is_not_reused_as_later_close_and_repeated_relations_reset(self):
        arrays = evidence(14)
        arrays.update(object_positions=torch.zeros(14, 1, 3), object_extents=torch.ones(1, 3),
                      object_joint_fractions=torch.tensor([[0.], [0.], [0.], [1.], [1.], [1.], [1.],
                                                          [0.], [0.], [0.], [1.], [1.], [1.], [1.]]))
        audit = annotation("sim_relation", [relation("opened", "door"), relation("closed", "door"),
                                            relation("opened", "door", occurrence=2)],
                           object_roles=["door"], geometry="axis_aligned_robot_base_boxes")
        indices, _ = self.resolve("sim_relation", audit, arrays)
        self.assertEqual(indices.tolist(), [4, 8, 11, 13])
        audit["relations"] = [relation("opened", "door"), relation("opened", "door", occurrence=2)]
        indices, _ = self.resolve("sim_relation", audit, arrays)
        self.assertEqual(indices.tolist(), [4, 11, 13])
        arrays["object_joint_fractions"][7:] = 1.
        with self.assertRaisesRegex(ValueError, "missing or out-of-order"):
            self.resolve("sim_relation", audit, arrays)

    def test_relation_registry_requires_roles_occurrences_and_rejects_pressed(self):
        for program in ([relation("pressed")], [relation("in_hand", occurrence=2)],
                        [relation("placed", relation="near", target_role="bowl")],
                        [relation("in_hand", effector=2)]):
            with self.subTest(program=program), self.assertRaises(ValueError):
                validate_relations(program, 1)
        arrays, audit = self.sim()
        audit["relation_registry"] = "unversioned_custom_rules"
        with self.assertRaisesRegex(ValueError, "fixed registry"):
            self.resolve("sim_relation", audit, arrays)

    def test_sim_inside_geometry_and_online_confirmation_prefixes(self):
        arrays, audit = self.sim()
        audit["relations"][1].update(relation="in")
        arrays["object_extents"][1] = .2
        _, full = self.resolve("sim_relation", audit, arrays)
        self.assertEqual(full["relation_control_indices"], [6, 11])
        for end, count in ((7, 1), (12, 2), (17, 2)):
            prefix = {key: value[:end] if key != "object_extents" else value for key, value in arrays.items()}
            spec = {**audit, "relations": audit["relations"][:count]}
            _, result = self.resolve("sim_relation", spec, prefix)
            self.assertEqual(result["relation_control_indices"], full["relation_control_indices"][:count])

    def test_sim_rejects_hidden_property_metadata_and_invalid_geometry(self):
        arrays, audit = self.sim()
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            self.resolve("sim_relation", {**audit, "force_threshold": 1.}, arrays)
        with self.assertRaisesRegex(ValueError, "positive.*half extents"):
            self.resolve("sim_relation", audit, {**arrays, "object_extents": torch.zeros(2, 3)})
        with self.assertRaisesRegex(ValueError, "finite floating"):
            self.resolve("sim_relation", audit, {**arrays, "object_positions": torch.zeros(17, 2)})

    def test_empty_closure_is_not_a_blocked_grasp(self):
        arrays = evidence()
        arrays["gripper"][3:8] = 0.
        candidates = generate_candidates(arrays, EventRules())
        self.assertEqual([item["kind"] for item in candidates], ["gripper_close", "gripper_open"])
        self.assertNotIn("blocked_close", [item["kind"] for item in candidates])

    def test_pedal_clock_uses_first_available_control_step_and_records_delay(self):
        arrays = evidence()
        arrays["pedal_times"] = torch.tensor([10.51, 11.04], dtype=torch.float64)
        audit = annotation("pedal", [relation("in_hand"), relation("placed", relation="in", target_role="bowl")],
                           clock_map={"scale": 1., "offset": -10.}, annotation_version="pedal_v1", confirmation_delay=.12)
        audit.update(stable_steps=1, thresholds={})
        indices, audit = self.resolve("pedal", audit, arrays)
        self.assertEqual(indices.tolist(), [6, 11, 16])
        self.assertEqual(audit["confirmation_delay"], .12)
        self.resolve("pedal", audit, arrays, check=True)
        audit["control_indices"] = [5, 10, 16]
        with self.assertRaisesRegex(ValueError, "recomputed source"):
            self.resolve("pedal", audit, arrays, check=True)
        audit["confirmation_delay"] = -.1
        with self.assertRaisesRegex(ValueError, "confirmation_delay"):
            self.resolve("pedal", audit, arrays)

    def test_pedal_rejects_missing_log_bad_clock_duplicate_steps_and_order(self):
        arrays = evidence()
        arrays["pedal_times"] = torch.tensor([.51, .52], dtype=torch.float64)
        audit = annotation("pedal", [relation("in_hand"), relation("placed", relation="in", target_role="bowl")],
                           clock_map={"scale": 1., "offset": 0.}, annotation_version="pedal_v1", confirmation_delay=0.)
        audit.update(stable_steps=1, thresholds={})
        with self.assertRaisesRegex(ValueError, "distinct control"):
            self.resolve("pedal", audit, arrays)
        for value in (torch.tensor([.52, .51]), torch.tensor([-.1, 1.]), torch.tensor([1., 2.])):
            arrays["pedal_times"] = value
            with self.assertRaises(ValueError):
                self.resolve("pedal", audit, arrays)
        with self.assertRaisesRegex(ValueError, "pedal_times"):
            self.resolve("pedal", audit, {key: value for key, value in arrays.items() if key != "pedal_times"})

    def test_candidate_blocked_width_and_release_are_causal_and_weak(self):
        arrays, audit = self.candidate()
        indices, audit = self.resolve("candidate_match", audit, arrays)
        self.assertEqual(indices.tolist(), [5, 9, 16])
        self.assertTrue(audit["weak_label"])
        self.assertEqual(audit["candidates"], [{"control_index": 5, "effector": 0, "kind": "blocked_close"},
                                              {"control_index": 9, "effector": 0, "kind": "gripper_open"}])
        full = generate_candidates(arrays, EventRules())
        for count in range(2, 18):
            prefix = {key: value[:count] for key, value in arrays.items()}
            self.assertEqual(generate_candidates(prefix, EventRules()),
                             [candidate for candidate in full if candidate["control_index"] < count])
        audit["weak_label"] = False
        with self.assertRaisesRegex(ValueError, "weak_label=true"):
            self.resolve("candidate_match", audit, arrays)

    def test_candidate_speed_minimum_confirms_after_rise_or_sustained_stop(self):
        arrays = evidence(9)
        arrays["poses"][:, 0, 0, 3] = torch.tensor([0., .01, .015, .0155, .017, .017, .017, .017, .017])
        candidates = generate_candidates(arrays, EventRules())
        minima = [item["control_index"] for item in candidates if item["kind"] == "speed_minimum"]
        self.assertIn(4, minima)
        self.assertNotIn(3, minima)
        for length in range(2, 10):
            self.assertEqual(generate_candidates({key: value[:length] for key, value in arrays.items()}, EventRules()),
                             [item for item in candidates if item["control_index"] < length])

    def test_candidate_matching_rejects_wrong_order_missing_extra_and_duration(self):
        arrays, original = self.candidate()
        for change, message in (({"relations": original["relations"][::-1]}, "order"),
                                ({"candidates": []}, "regenerated"),
                                ({"thresholds": {**CANDIDATE_THRESHOLDS, "max_duration": .2}}, "duration")):
            audit = {**original, **change}
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                self.resolve("candidate_match", audit, arrays)
        missing = deepcopy(original)
        missing["relations"].append(relation("in_hand", occurrence=2))
        with self.assertRaisesRegex(ValueError, "missing"):
            self.resolve("candidate_match", missing, arrays)
        repeated = deepcopy(arrays)
        repeated["gripper"][11:14] = .3
        extra = {**original, "candidates": generate_candidates(repeated, EventRules())}
        with self.assertRaisesRegex(ValueError, "extra relation"):
            self.resolve("candidate_match", extra, repeated)

    def test_loading_alternative_source_uses_precise_target_and_excludes_offline_tensors(self):
        path, metadata, values = write_g_pi_task(self.root)
        arrays, audit = self.sim()
        indices, audit = self.resolve("sim_relation", audit, arrays)
        metadata.update(subgoal_source="sim_relation", subgoal_annotation=audit)
        values.update({key: value.numpy() for key, value in arrays.items()})
        values["subgoal_times"] = values["control_times"][indices.numpy()]
        values["subgoal_latents"] = np.stack([np.full((2, 1, 2, 2), int(index), np.float32) for index in indices])
        path.write_text(json.dumps(metadata))
        np.savez_compressed(self.root / metadata["arrays"], **values)
        sample = load_g_pi_sample(path, current_time=.8)
        self.assertAlmostEqual(sample.subgoal_time, 1.1)
        self.assertTrue(torch.equal(sample.target_frame, torch.full_like(sample.target_frame, 11.)))
        self.assertEqual(sample.history.shape[2], 3)
        torch.testing.assert_close(sample.goal_poses[0], arrays["poses"][11])
        self.assertEqual(sample.actions_mask.sum().item(), 3)
        forbidden = {"relations", "object_positions", "object_extents", "object_joint_fractions", "pedal_times",
                     "subgoal_source", "relation_control_indices", "stage_index"}
        self.assertFalse(forbidden & {field.name for field in fields(PiGoalSample)})
        for key, value in vars(sample).items():
            if key not in {"metadata", "language_identity"}:
                self.assertNotIsInstance(value, dict)
        values["object_positions"][10:, 0, 0] = 1.
        np.savez_compressed(self.root / metadata["arrays"], **values)
        with self.assertRaisesRegex(ValueError, "missing"):
            load_g_pi_sample(path, current_time=.8)

    def test_candidate_offline_tool_regenerates_audited_output(self):
        arrays, audit = self.candidate()
        np.savez_compressed(self.root / "evidence.npz", **{key: value.numpy() for key, value in arrays.items()})
        spec = dict(format_version=1, kind="g_pi_candidate_spec", arrays="evidence.npz", event_rules=asdict(EventRules()),
                    **{key: audit[key] for key in ("relations", "thresholds", "stable_steps", "evidence", "template_version")})
        path, output = self.root / "spec.json", self.root / "annotation.json"
        path.write_text(json.dumps(spec))
        result = generate_candidate_annotation(path, output)
        self.assertEqual(result, json.loads(output.read_text()))
        self.assertEqual(result["subgoal_annotation"]["control_indices"], [5, 9, 16])
        self.assertTrue(result["subgoal_annotation"]["weak_label"])
        self.resolve("candidate_match", result["subgoal_annotation"], arrays, check=True)

    def test_single_frame_preprocessing_accepts_audited_non_gripper_boundaries(self):
        arrays, audit = self.sim()
        _, audit = self.resolve("sim_relation", audit, arrays)
        frames = np.arange(17, dtype=np.uint8)[:, None, None, None] * np.ones((17, 2, 2, 3), dtype=np.uint8)
        def encode(vae, rgb, size):
            return torch.tensor(rgb[::4, 0, 0, 0].copy(), dtype=torch.float32).reshape(1, 1, -1, 1, 1)
        with patch("etude.icl_preprocess.encode_rgb", side_effect=encode) as encoder:
            result, metadata = encode_g_pi_frames(None, frames, arrays["control_times"], arrays["gripper"],
                                                  frame_stride=1, control_dt=.1, event_rules=EventRules(), size=(2, 2),
                                                  subgoal_metadata={"subgoal_source": "sim_relation", "subgoal_annotation": audit},
                                                  offline_arrays=arrays)
        np.testing.assert_array_equal(result["subgoal_times"], arrays["control_times"][[6, 11, 16]].numpy())
        np.testing.assert_array_equal(result["subgoal_latents"].flatten(), [6., 11., 16.])
        self.assertEqual(metadata["subgoal_source"], "sim_relation")
        self.assertEqual([len(call.args[1]) for call in encoder.call_args_list], [17, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
