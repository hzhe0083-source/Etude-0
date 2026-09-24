import copy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from evo_wam.g_pi_context import frozen_base_checksum
from evo_wam.g_pi_data import EventRules, load_g_pi_sample
from evo_wam.g_pi_training import (ROUTES, _training_language, language_drop_probability, _base_location, _base_reference, _frozen_checksums, _optimizer, _restore_system, _set_precision, _system_state,
    build_g_pi_system, export_g_pi_policy, g_pi_architecture, g_pi_artifact_version, g_pi_training_loss, g_intent_training_loss, intent_training_settings, pi_training_settings, pi_stage_contract, _check_stage, _interface_state, _initialization, _initialize_pi, _endpoint_losses,
    load_g_pi_policy, conditioning_mode, load_g_pi_encoder, read_g_pi_artifact, train_g_pi_interface, validate_g_pi_artifact, validate_g_pi_config)
from evo_wam.goal_action import action_named_parameters
from evo_wam.goal_training import goal_registry
from evo_wam.cli import file_sha256
from evo_wam.zerowam import NativeDependencyError, ZERO_WAM_COMMIT, load_native_class
from test_g_pi_data import write_g_pi_task
from test_native_icl import tiny_model


def explicit_thresholds():
    return {"z": .1, "position_m": .01, "rotation_deg": 5., "gripper": .05}


def config_for(route):
    config = json.loads((Path(__file__).parents[1] / "configs/se3/goal_interface.json").read_text())
    config.update(schema_version=3, interface_type=route, video_weight=0., pose_weight=.3, action_sampling_steps=1,
                  window_size=8, goal_encoder={"layer": 1, "grid_size": [4, 4], "camera_layout": [{"name": "head", "token_width": 2}]}, event_rules=asdict(EventRules()))
    config["conditioning_mode"] = conditioning_mode(config)
    config.pop("sampling_steps")
    config["ifp"].update(enabled=False, loss_weights=[0.])
    config["training"].update(learning_rate=.001, max_steps=8)
    config["goal_interface"] = {"state_dim": 4, "dim": 16, "num_heads": 2, "translation_scale": 1.}
    if route == "g_translator":
        config["goal_interface"].update(num_layers=2, use_state=True)
    else:
        config["p_drop"] = .4
        config["pi_training"] = {"ablations": {"no_stage1": True, "exact_goal": True}}
        config["goal_interface"].update(num_tokens=4, num_layer_groups=1, num_pose_tokens=1)
    return config


def contract_payload(route="g_translator"):
    config = config_for(route)
    identity = {"layer": 1, "k_z": 16, "grid_size": [4, 4], "token_order": "camera_then_row_major",
                "camera_layout": [{"name": "head", "token_width": 2}], "num_views": 1, "d_z": 36, "timestep": 0,
                "pooling": "adaptive_avg_pool2d_spatial", "normalization": "l2_last_dim",
                "base_id": {"kind": "fixture"}, "base_sha256": "a" * 64,
                "empty_text_identity": {"source": {"kind": "fixture"}, "sha256": "b" * 64}}
    return {"format_version": g_pi_artifact_version(config), "kind": "g_pi_training", "config": config,
            **({"demo_route": config.get("demo_route", "one_way")} if route == "g_translator" else
               {"pi_training": pi_stage_contract(config, "pi"), "stage1_artifact_sha256": None}),
            "interface_type": route, "conditioning_mode": conditioning_mode(config),
            "p_drop": language_drop_probability(config),
            "architecture": g_pi_architecture(config), "upstream_commit": ZERO_WAM_COMMIT,
            "stage": ROUTES[route], "precision": "float32", "encoder_precision": "float32",
            "base_identity": identity["base_id"], "encoder_identity": identity,
            "empty_text_identity": identity["empty_text_identity"], "k_z": 16, "d_z": 36,
            "event_rules": config["event_rules"], "registry": {"event_rules": config["event_rules"]},
            "tiny_native": True,
            "base_reference": {"kind": "tiny-native", "checkpoint": None, "base_seed": 0,
                               "identity": identity["base_id"], "encoder_identity": identity,
                               "empty_text_identity": identity["empty_text_identity"], "video_precision": "float32"},
            "model": {"interface": {"weight": torch.ones(2, 2)}, "action": {}}}


class GPiTrainingContractTest(unittest.TestCase):
    def test_pi_stages_require_explicit_initialization_and_noise_ablations(self):
        config = config_for("pi_goal")
        config.pop("pi_training")
        self.assertEqual(pi_training_settings(config)["pose_weight"], .3)
        self.assertFalse(any(pi_training_settings(config)["ablations"].values()))
        _check_stage(config, "pi_prior")
        with self.assertRaisesRegex(ValueError, "exact_goal"):
            _check_stage(config, "pi")
        config["goal_noise"] = {"z_std": .03}
        _check_stage(config, "pi")
        with self.assertRaisesRegex(ValueError, "pi_prior.*initialize"):
            _initialization(SimpleNamespace(stage="pi", initialize=None), config, None)
        config["distributed"] = {"enabled": True}
        with self.assertRaisesRegex(ValueError, "single-GPU"):
            _check_stage(config, "pi_prior")
        config.pop("distributed")
        config["pi_training"] = {"goal_source": "crossfit"}
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_g_pi_config(config)
        for settings in ({"pose_weight": 0.}, {"ablations": {"no_stage1": 1}}, {"extra": True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_g_pi_config({**config, "pi_training": settings})
        config["pi_training"] = {"ablations": {"no_lit_pose": True}}
        self.assertEqual(pi_training_settings(config)["pose_weight"], 0.)
        payload = contract_payload("pi_goal")
        with self.assertRaisesRegex(ValueError, "old versions"):
            read_g_pi_artifact(None, payload={**payload, "format_version": 3})
        payload["pi_training"]["pose_weight"] = 9.
        with self.assertRaisesRegex(ValueError, "endpoint loss"):
            read_g_pi_artifact(None, payload=payload)

    def test_intent_config_modes_route_and_artifact_version_are_explicit(self):
        for mode in ("regression_only", "independent", "connected"):
            config = config_for("g_translator")
            config["goal_interface"].update(intent_mode=mode, num_intent_tokens=3, num_intent_layers=1)
            config["intent_training"] = {"manifest": "groups.json", "data_version": "v2",
                "contrastive_weight": 0. if mode == "regression_only" else .2}
            for route in ("one_way", "via_u_only"):
                config["demo_route"] = route
                if route == "via_u_only" and mode != "connected":
                    with self.assertRaisesRegex(ValueError, "via_u_only requires connected"):
                        validate_g_pi_config(config)
                else:
                    validate_g_pi_config(config)
        for settings in ({"contrastive_weight": .5}, {"manifest": "x", "temperature": 0.},
                         {"groups_per_batch": 1}, {"samples_per_group": 1}, {"data_version": "v3"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_g_pi_config({**config_for("g_translator"), "intent_training": settings})
        with self.assertRaisesRegex(ValueError, "never sees"):
            validate_g_pi_config({**config_for("pi_goal"), "intent_training": {}})
        self.assertEqual(g_pi_artifact_version(config_for("g_translator")), 4)
        self.assertEqual(g_pi_artifact_version(config_for("pi_goal")), 4)
        payload = contract_payload()
        with self.assertRaisesRegex(ValueError, "demo_route"):
            read_g_pi_artifact(None, payload={**payload, "demo_route": "via_u_only"})

    def test_language_dropout_endpoints_probability_and_exact_rng_continuation(self):
        native = SimpleNamespace(g_pi_empty_text=torch.arange(24, dtype=torch.float32).reshape(1, 3, 8))
        language = torch.full((1, 4, 8), -2.)
        generator = torch.Generator().manual_seed(19)
        initial = generator.get_state().clone()
        self.assertIs(_training_language(native, language, generator, 0.), language)
        torch.testing.assert_close(generator.get_state(), initial, rtol=0, atol=0)
        config = config_for("pi_goal")
        self.assertEqual(language_drop_probability(config), .4)
        del config["p_drop"]
        self.assertEqual(language_drop_probability(config), 0.)
        self.assertEqual(language_drop_probability(config_for("g_translator")), 0.)
        for _ in range(8):
            selected = _training_language(native, language, generator, 1.)
            torch.testing.assert_close(selected, native.g_pi_empty_text, rtol=0, atol=0)
            self.assertTrue(torch.count_nonzero(selected))
        generator.manual_seed(7)
        decisions = [torch.equal(_training_language(native, language, generator, .4), native.g_pi_empty_text)
                     for _ in range(5000)]
        self.assertLess(abs(sum(decisions) / len(decisions) - .4), .025)
        state = generator.get_state().clone()
        expected = [_training_language(native, language, generator, .4).clone() for _ in range(20)]
        continued = torch.Generator()
        continued.set_state(state)
        for value in expected:
            torch.testing.assert_close(_training_language(native, language, continued, .4), value, rtol=0, atol=0)

    def test_explicit_empty_text_path_overrides_default_and_records_source(self):
        native = torch.nn.Module()
        native.patch_embedding_mlp = torch.nn.Linear(4, 4)
        native.action_embedder = torch.nn.Linear(4, 4)
        native.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])
        native.config = SimpleNamespace(text_dim=8)
        config = config_for("g_translator")
        with TemporaryDirectory() as folder:
            path = Path(folder) / "custom-empty.pt"
            torch.save(torch.arange(24, dtype=torch.float32).reshape(3, 8), path)
            config["empty_emb_path"] = str(path)
            source_identity = {"kind": "fixture", "config": {"patch_size": (1, 1, 1), "mcp_hidden_collect_layers": (0, 1)}}
            with patch("evo_wam.g_pi_training.build_icl_model", return_value=(native, torch.zeros(1, 2, 8), source_identity)) as builder, \
                 patch("evo_wam.g_pi_training.install_action_interface"), \
                 patch("evo_wam.g_pi_training._interface", return_value=torch.nn.Linear(4, 4)), \
                 patch("evo_wam.g_pi_context.FrozenGoalEncoder", return_value=torch.nn.Linear(4, 4)) as encoder:
                _, _, _, identity, _ = build_g_pi_system(config, {}, stage="g", tiny_native=True, device="cpu")
            self.assertEqual(builder.call_args.kwargs["empty_text_path"], str(path))
            self.assertEqual(native.g_pi_empty_text_identity["source"]["path"], str(path))
            torch.testing.assert_close(native.g_pi_empty_text, torch.arange(24, dtype=torch.float32).reshape(1, 3, 8))
            self.assertEqual(identity, {**json.loads(json.dumps(source_identity)), "base_seed": 0})
            self.assertEqual(encoder.call_args.kwargs["base_id"], identity)
            payload = contract_payload()
            payload["base_identity"] = identity
            payload["encoder_identity"]["base_id"] = identity
            payload["base_reference"]["identity"] = identity
            payload.update(updates=1, interface_type="g_translator")
            artifact = Path(folder) / "checkpoint.pt"
            torch.save(payload, artifact)
            export = export_g_pi_policy(SimpleNamespace(stop_thresholds=explicit_thresholds(), artifact=artifact, output=Path(folder) / "policy", dtype="bfloat16"))
            deployed = json.loads((Path(export["policy"]) / "policy.json").read_text())
            validate_g_pi_artifact(deployed, kind="g_pi_policy", expected_encoder_identity=payload["encoder_identity"])
            self.assertEqual(deployed["precision"], "float32")
            self.assertTrue(all(name.startswith("interface.") for name in deployed["weight_map"]))

    def test_config_rejects_joint_video_adaptation_and_invalid_event_contracts(self):
        for route in ROUTES:
            self.assertIsNotNone(validate_g_pi_config(config_for(route)))
        for updates in ({"video_weight": 1.}, {"sampling_steps": 1}, {"lora": {}},
                        {"event_rules": {}}, {"goal_encoder": {"layer": -1}},
                        {"goal_encoder": {"layer": 1, "k_z": 8}},
                        {"goal_encoder": {"layer": 1, "grid_size": [4, 0]}},
                        {"schema_version": 2}, {"conditioning_mode": "unsupported"},
                        {"p_drop": -1}, {"p_drop": 1.1}, {"p_drop": float("nan")},
                        {"goal_noise": {"z_std": -1}}, {"base_seed": True},
                        {"empty_emb_path": "/tmp/a", "empty_text_emb_path": "/tmp/b"}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                validate_g_pi_config({**config_for("g_translator"), **updates})

    def test_old_checkpoint_and_grid_identity_are_rejected(self):
        for version in (1, 2, 3):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "old versions cannot resume"):
                read_g_pi_artifact(None, payload={**contract_payload(), "format_version": version})
        for field, value in (("grid_size", [2, 8]), ("token_order", "column_major"),
                             ("pooling", "adaptive_avg_pool1d_spatial"), ("num_views", 3),
                             ("camera_layout", [{"name": "head", "token_width": 1}])):
            changed = contract_payload()
            changed["encoder_identity"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "E identity"):
                read_g_pi_artifact(None, payload=changed)

    def test_export_requires_explicit_thresholds_and_preserves_calibration(self):
        with TemporaryDirectory() as folder:
            artifact = Path(folder) / "train.pt"
            payload = {**contract_payload(), "updates": 1}
            torch.save(payload, artifact)
            arguments = dict(artifact=artifact, output=Path(folder) / "policy", dtype="float32")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                export_g_pi_policy(SimpleNamespace(**arguments))
            self.assertFalse(Path(arguments["output"]).exists())
            with self.assertRaisesRegex(ValueError, "explicit stop thresholds"):
                export_g_pi_policy(SimpleNamespace(**arguments, stop_thresholds={"z": .1}))
            with self.assertRaisesRegex(ValueError, "exactly one"):
                export_g_pi_policy(SimpleNamespace(**arguments, stop_thresholds=explicit_thresholds(),
                                                     calibration="calibration.json"))
            from evo_wam.g_pi_calibration import calibrate_goal_thresholds

            def goal(position):
                poses = torch.eye(4)[None, None]
                poses[..., 0, 3] = position
                return {"z": torch.nn.functional.normalize(torch.ones(1, 16, 36), dim=-1),
                        "goal_poses": poses, "goal_gripper": torch.zeros(1, 1)}

            records = [{"sample_id": f"take-{index}", "source_group": f"source-{index}",
                        "intent_group": "fixture", "goals": [goal(offset), goal(1 + offset)]}
                       for index, offset in enumerate((0., .01))]
            calibrated = calibrate_goal_thresholds(records, encoder_identity=payload["encoder_identity"],
                                                   registry=payload["registry"])
            calibration_path = Path(folder) / "calibration.json"
            calibration_path.write_text(json.dumps(calibrated))
            result = export_g_pi_policy(SimpleNamespace(**arguments, calibration=calibration_path))
            policy = json.loads((Path(result["policy"]) / "policy.json").read_text())
            self.assertEqual(policy["stop_thresholds"], calibrated["thresholds"])
            self.assertEqual(policy["stopping_calibration"], calibrated)
            self.assertEqual(policy["conditioning_mode"], conditioning_mode(payload["config"]))
            validate_g_pi_artifact(policy, kind="g_pi_policy")
            changed = copy.deepcopy(policy)
            changed["stop_thresholds"]["z"] += .1
            with self.assertRaisesRegex(ValueError, "differ from the recorded calibration"):
                validate_g_pi_artifact(changed, kind="g_pi_policy")
            changed = copy.deepcopy(policy)
            changed["stopping_calibration"]["encoder_identity"]["layer"] += 1
            with self.assertRaisesRegex(ValueError, "E identity"):
                validate_g_pi_artifact(changed, kind="g_pi_policy")
            del policy["stop_thresholds"]
            with self.assertRaisesRegex(ValueError, "explicit stop thresholds"):
                validate_g_pi_artifact(policy, kind="g_pi_policy")

    def test_checkpoint_contract_roundtrip_and_encoder_mismatch(self):
        with TemporaryDirectory() as folder:
            for route in ROUTES:
                payload = contract_payload(route)
                path = Path(folder) / f"{route}.pt"
                torch.save(payload, path)
                self.assertEqual(read_g_pi_artifact(path)["encoder_identity"], payload["encoder_identity"])
                self.assertIs(read_g_pi_artifact(None, payload=payload,
                    expected_encoder_identity=payload["encoder_identity"]), payload)
                with self.assertRaisesRegex(ValueError, "E identity mismatch"):
                    read_g_pi_artifact(path, expected_encoder_identity={**payload["encoder_identity"], "layer": 0})
                for field, value in (("k_z", 9), ("d_z", 17), ("stage", "joint")):
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        read_g_pi_artifact(None, payload={**payload, field: value})
                with self.assertRaisesRegex(ValueError, "route, E identity"):
                    read_g_pi_artifact(None, payload={**payload, "p_drop": .7})
                changed = copy.deepcopy(payload)
                changed["encoder_identity"]["normalization"] = "affine"
                with self.assertRaisesRegex(ValueError, "E identity"):
                    read_g_pi_artifact(None, payload=changed)
                changed = copy.deepcopy(payload)
                changed["model"]["native"] = {"frozen.weight": torch.ones(3)}
                with self.assertRaisesRegex(ValueError, "frozen weights"):
                    read_g_pi_artifact(None, payload=changed)

    def test_restore_has_no_frozen_tensors_and_requires_fp32_trainables(self):
        modules = [torch.nn.Linear(3, 2), torch.nn.Linear(3, 2), torch.nn.Module()]
        state = copy.deepcopy(_system_state(*modules))
        self.assertEqual(set(state), {"interface", "action"})
        self.assertFalse(state["action"])
        frozen = modules[0].weight.detach().clone()
        with torch.no_grad():
            modules[1].weight.add_(3)
        _restore_system(*modules, state)
        torch.testing.assert_close(modules[1].weight, state["interface"]["weight"], rtol=0, atol=0)
        torch.testing.assert_close(modules[0].weight, frozen, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "frozen native"):
            _restore_system(*modules, {"native": {}, **state})
        changed = copy.deepcopy(state)
        changed["interface"]["weight"] = changed["interface"]["weight"].to(torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "precision"):
            _restore_system(*modules, changed)

    def test_base_file_hashes_and_missing_reference_are_rejected_before_loading(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("config.json", "model.safetensors"):
                (root / name).write_text(name)
            reference = {"kind": "local-checkpoint", "checkpoint": str(root),
                         "identity": {"sha256": {name: file_sha256(root / name)
                                                  for name in ("config.json", "model.safetensors")}}}
            self.assertEqual(_base_location(reference), str(root))
            (root / "model.safetensors").write_text("changed")
            with self.assertRaisesRegex(ValueError, "hashes changed"):
                _base_location(reference)
            with self.assertRaisesRegex(ValueError, "missing"):
                _base_location(reference, str(root / "absent"))
            with self.assertRaisesRegex(ValueError, "fixed base_seed"):
                _base_location({"kind": "tiny-native"}, str(root))


class GPiBaseConstructionTest(unittest.TestCase):
    """CPU topology and serialization checks execute no native attention."""

    @classmethod
    def setUpClass(cls):
        try:
            load_native_class()
        except NativeDependencyError as exc:
            raise unittest.SkipTest(str(exc))

    def setUp(self):
        self.registry = {"action_space": {"dimension": 3}, "language_identity": {"text_dim": 8},
                         "end_effectors": ["tool"], "control_dt": .1, "actions_per_frame": 4,
                         "event_rules": asdict(EventRules())}
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())

        def build(config, **kwargs):
            self.assertEqual(kwargs["device"], "cpu")
            return tiny_model("cpu").float(), torch.ones(1, 3, 8), {"kind": "native-test-fixture", "layers": (0, 1)}

        self.patcher = patch("evo_wam.g_pi_training.build_icl_model", side_effect=build)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_training_uses_one_selected_language_for_lit_and_all_action_layers(self):
        config = config_for("pi_goal")
        native, interface, encoder, _, layers = build_g_pi_system(config, self.registry,
            stage="pi", tiny_native=True, device="cpu")
        sample = SimpleNamespace(state=torch.zeros(1, 4), target_frame=torch.zeros(1, 4, 1, 1, 2),
            history=torch.zeros(1, 4, 1, 1, 2), history_times=torch.zeros(1),
            language=torch.full((1, 5, 8), -3.), goal_poses=torch.eye(4)[None, None],
            goal_gripper=torch.zeros(1, 1), actions=torch.zeros(1, 3, 2, 4, 1),
            actions_mask=torch.ones(1, 3, 2, 4, 1, dtype=torch.bool),
            block_end_valid=True, block_end_poses=torch.eye(4)[None, None],
            block_end_gripper=torch.zeros(1, 1), reaches_subgoal=False)
        features = {layer: torch.zeros(1, 2, native.inner_dim) for layer in layers}
        z = torch.nn.functional.normalize(torch.ones(1, encoder.k_z, encoder.d_z), dim=-1)
        projection = native.condition_embedder_action.text_embedder
        for probability in (0., 1.):
            config["p_drop"] = probability
            selected = sample.language if probability == 0 else native.g_pi_empty_text
            expected = projection(selected)
            projection_inputs = []
            hook = projection.register_forward_pre_hook(lambda module, args: projection_inputs.append(args[0].detach().clone()))
            generators = {name: torch.Generator().manual_seed(23) for name in ("action", "goal", "language")}
            initial_rng = generators["language"].get_state().clone()
            try:
                with patch.object(encoder, "forward", return_value=z), \
                     patch("evo_wam.g_pi_context.pi_context_features", return_value=features), \
                     patch.object(interface, "read_layer", wraps=interface.read_layer) as reader, \
                     patch("evo_wam.g_pi_training.goal_action_forward", side_effect=lambda model, noisy, times, conditions: noisy * 0) as action:
                    g_pi_training_loss(native, interface, encoder, sample, config, generators,
                                       stage="pi", feature_layers=layers)
            finally:
                hook.remove()
            self.assertEqual(len(projection_inputs), 1)
            torch.testing.assert_close(projection_inputs[0], selected, rtol=0, atol=0)
            for call in reader.call_args_list:
                torch.testing.assert_close(call.args[2][:, :selected.shape[1]], expected, rtol=0, atol=0)
            for memory in action.call_args.args[3]:
                torch.testing.assert_close(memory[:, :selected.shape[1]], expected, rtol=0, atol=0)
            if probability == 0:
                torch.testing.assert_close(generators["language"].get_state(), initial_rng, rtol=0, atol=0)

    def test_human_only_examples_do_not_dilute_paired_regression(self):
        config = config_for("g_translator")
        config["intent_training"] = {"manifest": "offline.json", "contrastive_weight": .3}
        native, interface, encoder, _, layers = build_g_pi_system(config, self.registry,
            stage="g", tiny_native=True, device="cpu")
        demo = torch.randn(1, 4, 3, 1, 2)
        sample = SimpleNamespace(state=torch.zeros(1, 4), target_frame=torch.zeros(1, 4, 1, 1, 2),
            demonstration=demo, history=torch.zeros(1, 4, 1, 1, 2),
            goal_poses=torch.eye(4)[None, None], goal_gripper=torch.zeros(1, 1))
        demo_features = {layer: torch.randn(1, 6, native.inner_dim) for layer in layers}
        robot_features = {layer: torch.randn(1, 2, native.inner_dim) for layer in layers}
        z = torch.nn.functional.normalize(torch.randn(1, encoder.k_z, encoder.d_z), dim=-1)
        entries = [SimpleNamespace(demo_id=str(i), component=i, purpose_group=str(i // 2)) for i in range(4)]
        with patch.object(encoder, "forward", return_value=z), \
             patch("evo_wam.g_pi_context.split_g_context_features", return_value=(demo_features, robot_features)), \
             patch("evo_wam.g_pi_context.demo_context_features", return_value=demo_features):
            single = g_pi_training_loss(native, interface, encoder, sample, config, {}, stage="g", feature_layers=layers)
            mixed = g_intent_training_loss(native, interface, encoder,
                [(demo, sample), (demo, None), (demo, None), (demo, None)], entries, config)
        torch.testing.assert_close(mixed["regression"], single["total"], rtol=0, atol=0)
        torch.testing.assert_close(mixed["total"], single["total"] + .3 * mixed["contrastive"], rtol=0, atol=0)
        mixed["total"].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in interface.parameters()))
        self.assertFalse(any(p.grad is not None for p in native.parameters()))

    def test_single_shared_base_seed_independence_and_mixed_parameter_precision(self):
        identities = []
        for index, route in enumerate(ROUTES):
            torch.manual_seed(11 + index)
            native, interface, encoder, _, _ = build_g_pi_system(config_for(route), self.registry,
                stage=ROUTES[route], tiny_native=True, device="cpu", video_precision="bfloat16")
            self.assertIs(encoder.native, native)
            self.assertFalse(list(encoder.parameters()))
            self.assertFalse(encoder.state_dict())
            self.assertEqual(native.patch_embedding_mlp.weight.dtype, torch.bfloat16)
            self.assertTrue(all(p.dtype == torch.float32 for p in interface.parameters()))
            if route == "pi_goal":
                self.assertTrue(all(p.dtype == torch.float32 and p.requires_grad for _, p in action_named_parameters(native)))
            identities.append(encoder.identity)
        self.assertEqual(identities[0], identities[1])
        config = {**config_for("g_translator"), "base_seed": 1}
        _, _, other, _, _ = build_g_pi_system(config, self.registry, stage="g", tiny_native=True,
                                             device="cpu", video_precision="bfloat16")
        self.assertNotEqual(other.identity, identities[0])
        native, _, _, _, _ = build_g_pi_system(config_for("pi_goal"), self.registry, stage="pi",
                                              tiny_native=True, device="cpu")
        with torch.no_grad():
            for _, parameter in action_named_parameters(native):
                parameter.copy_(torch.randn_like(parameter))
        masters = {name: value.detach().clone() for name, value in action_named_parameters(native)}
        _set_precision(native, video_precision="bfloat16", route="pi_goal")
        for name, value in action_named_parameters(native):
            torch.testing.assert_close(value, masters[name], rtol=0, atol=0)

    def test_grid_dimensions_follow_configuration_and_calibration_rebuild(self):
        config = config_for("pi_goal")
        config["goal_encoder"]["grid_size"] = [2, 3]
        config["goal_encoder"]["camera_layout"] = [{"name": "head", "token_width": 1},
                                                    {"name": "wrist", "token_width": 1}]
        native, interface, encoder, identity, _ = build_g_pi_system(config, self.registry,
            stage="pi", tiny_native=True, device="cpu")
        self.assertEqual(encoder.identity["grid_size"], [2, 3])
        self.assertEqual(encoder.k_z, 12)
        self.assertEqual(interface.z_position.shape[1], 12)
        payload = contract_payload("pi_goal")
        payload.update(config=config, registry=self.registry, encoder_identity=encoder.identity,
                       base_identity=identity, empty_text_identity=encoder.identity["empty_text_identity"],
                       k_z=12, model=_system_state(native, interface, encoder),
                       base_reference=_base_reference(config, None, True, identity, encoder))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "train.pt"
            torch.save(payload, path)
            restored, recorded = load_g_pi_encoder(path, device="cpu")
        self.assertEqual(restored.identity, encoder.identity)
        self.assertEqual(recorded["config"]["conditioning_mode"], "language_state_goal")
        self.assertFalse(any(p.requires_grad for p in restored.native.parameters()))

    def test_compact_export_rebuilds_base_and_rejects_wrong_seed(self):
        with TemporaryDirectory() as folder:
            for route, stage in ROUTES.items():
                config = config_for(route)
                native, interface, encoder, identity, layers = build_g_pi_system(config, self.registry,
                    stage=stage, tiny_native=True, device="cpu")
                payload = contract_payload(route)
                payload.update(model=_system_state(native, interface, encoder), updates=1,
                    base_identity=identity, encoder_identity=encoder.identity, feature_layers=layers,
                    empty_text_identity=encoder.identity["empty_text_identity"], registry=self.registry,
                    base_reference=_base_reference(config, None, True, identity, encoder))
                artifact = Path(folder) / f"{route}.pt"
                torch.save(payload, artifact)
                exported = export_g_pi_policy(SimpleNamespace(stop_thresholds=explicit_thresholds(), artifact=artifact,
                    output=Path(folder) / route, dtype="bfloat16", max_shard_size="20KB"))
                rebuilt, decoder, target, deployed = load_g_pi_policy(exported["policy"], device="cpu")
                self.assertIs(target.native, rebuilt)
                self.assertEqual(target.identity, encoder.identity)
                self.assertTrue(all(key.startswith(("action.", "interface.")) for key in deployed["weight_map"]))
                self.assertFalse(any(p.requires_grad for p in rebuilt.parameters()))
                actual = _system_state(rebuilt, decoder, target)
                for module, values in payload["model"].items():
                    for name, value in values.items():
                        torch.testing.assert_close(actual[module][name], value, rtol=0, atol=0)
                policy_path = Path(exported["policy"]) / "policy.json"
                changed = json.loads(policy_path.read_text())
                changed["config"]["base_seed"] = 91
                changed["base_reference"]["base_seed"] = 91
                policy_path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "base checkpoint identity"):
                    load_g_pi_policy(exported["policy"], device="cpu")


@unittest.skipUnless(torch.cuda.is_available(), "Native G/pi FlexAttention requires CUDA")
class GPiNativeTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            load_native_class()
        except NativeDependencyError as exc:
            raise unittest.SkipTest(str(exc))

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path, metadata, arrays = write_g_pi_task(self.root)
        metadata["action_space"].update(dimension=3, valid_channels=[True, True, False])
        rng = np.random.default_rng(9)
        arrays["latent"] = rng.normal(size=(4, len(arrays["latent_available_times"]), 1, 2)).astype(np.float32)
        arrays["subgoal_latents"] = rng.normal(size=(len(arrays["subgoal_times"]), 4, 1, 1, 2)).astype(np.float32)
        arrays["actions"] = rng.normal(size=(3, len(arrays["control_times"]))).astype(np.float32)
        arrays["actions_mask"] = np.ones_like(arrays["actions"], dtype=np.bool_)
        arrays["states"] *= .01
        arrays["poses"][:, :, :3, 3] *= .01
        self.path.write_text(json.dumps(metadata))
        np.savez_compressed(self.root / metadata["arrays"], **arrays)
        np.savez_compressed(self.root / metadata["demonstration"]["arrays"],
                            latent=rng.normal(size=(4, 3, 1, 2)).astype(np.float32),
                            frame_times=np.array([0., .3, .9], dtype=np.float64))
        self.index = self.root / "index.json"
        self.index.write_text(json.dumps({"format_version": 1, "kind": "g_pi_index",
                                          "samples": [{"manifest": self.path.name, "split": "train"}]}))

        def build(config, **kwargs):
            native = tiny_model(kwargs.get("device", "cuda")).float()
            null = torch.arange(24, dtype=torch.float32, device=next(native.parameters()).device).reshape(1, 3, 8) / 24
            return native, null, {"kind": "native-test-fixture", "layers": 2}

        self.patcher = patch("evo_wam.g_pi_training.build_icl_model", side_effect=build)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def args(self, route, output, *, steps=1, resume=None):
        path = self.root / f"{route}.json"
        path.write_text(json.dumps(config_for(route)))
        return SimpleNamespace(config=str(path), index=str(self.index), output=str(self.root / output),
            steps=steps, seed=29, device="cuda", tiny_native=True, checkpoint=None,
            stage=ROUTES[route], resume=resume, initialize=None)

    def assert_same(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_same(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_same(first, second)
        else:
            self.assertEqual(left, right)

    def intent_manifest(self):
        task = json.loads(self.path.read_text())
        entries = []
        for index in range(4):
            name = task["demonstration"]["arrays"] if index == 0 else f"human-{index}.npz"
            if index:
                np.savez_compressed(self.root / name,
                    latent=np.random.default_rng(index).normal(size=(4, 3, 1, 2)).astype(np.float32),
                    frame_times=np.array([0., .3, .9], dtype=np.float64))
            source = task["demonstration"]["source_id"] if index == 0 else f"source-{index}"
            entries.append({"demo_id": f"demo-{index}", "purpose_group": f"purpose-{index // 2}",
                "split": "train", "source_id": source, "source_group": source,
                "person_id": f"person-{index}", "scene_id": f"scene-{index}", "view_id": f"view-{index}",
                "object_ids": ["cup"], "object_family": "cup", "arrays": name, "complete_demo": True,
                **({"paired_task": self.path.name} if index == 0 else {})})
        manifest = self.root / "intent.json"
        manifest.write_text(json.dumps({"format_version": 1, "kind": "g_pi_intent_groups", "data_version": "v1",
            "feature_space_id": task["feature_space_id"], "latent_normalization": task["latent_normalization"],
            "grouping_evidence": "fixture offline purpose audit", "entries": entries}))
        return manifest

    def intent_args(self, output, *, steps=1, resume=None):
        args = self.args("g_translator", output, steps=steps, resume=resume)
        config = json.loads(Path(args.config).read_text())
        config["intent_training"] = {"contrastive_weight": .3, "data_version": "v1"}
        Path(args.config).write_text(json.dumps(config))
        args.intent_groups = str(self.root / "intent.json")
        return args

    def test_mixed_human_only_exact_resume_export_and_complete_input_fingerprints(self):
        manifest = self.intent_manifest()
        # No G training input may depend on a language artifact, even for pairs.
        for path in self.root.glob("*language*"):
            path.unlink()
        complete = train_g_pi_interface(self.intent_args("intent-full", steps=2))
        partial = train_g_pi_interface(self.intent_args("intent-part"))
        resumed = train_g_pi_interface(self.intent_args("intent-part", resume=partial["artifact"]))
        full, continued = read_g_pi_artifact(complete["artifact"]), read_g_pi_artifact(resumed["artifact"])
        self.assertEqual(full["format_version"], 4)
        self.assertIn("intent", full["rng"])
        self.assertIsNone(full["registry"]["language_identity"])
        for name in ("model", "optimizer", "rng", "torch_rng", "cuda_rng", "data_identity", "visited_arrays"):
            self.assert_same(full[name], continued[name])
        metrics = [json.loads(line) for line in (self.root / "intent-full/metrics.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["paired_samples"] == 1 and row["human_only_samples"] == 3 for row in metrics))
        self.assertTrue(all(row["contrastive"] > 0 for row in metrics))
        exported = export_g_pi_policy(SimpleNamespace(stop_thresholds=explicit_thresholds(), artifact=complete["artifact"],
            output=self.root / "intent-policy", dtype="float32", max_shard_size="20KB"))
        native, decoder, encoder, policy = load_g_pi_policy(exported["policy"], device="cuda")
        self.assert_same(full["model"], _system_state(native, decoder, encoder))
        self.assertEqual(policy["config"]["intent_training"]["contrastive_weight"], .3)
        self.assertEqual(frozen_base_checksum(native), full["encoder_identity"]["base_sha256"])
        self.assertIn(str(manifest), full["data_identity"])
        human = self.root / "human-3.npz"
        self.assertIn(str(human), full["data_identity"])
        with np.load(human) as source:
            arrays = {key: source[key].copy() for key in source.files}
        arrays["latent"][0, 0, 0, 0] += .1
        np.savez_compressed(human, **arrays)
        with self.assertRaisesRegex(ValueError, "same route, config, data"):
            train_g_pi_interface(self.intent_args("intent-part", resume=resumed["artifact"]))

    def test_intent_sources_are_audited_jointly_with_all_index_splits(self):
        self.intent_manifest()
        document = json.loads(self.index.read_text())
        document["source_aliases"] = [{"source_id": "source-3", "source_group": "different-alias",
                                       "domain": "human", "split": "validation"}]
        self.index.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "split"):
            train_g_pi_interface(self.intent_args("cross-split"))
        self.assertFalse((self.root / "cross-split").exists())

    def test_unpaired_regression_only_batch_is_not_counted_as_an_update(self):
        from evo_wam.g_pi_intent import load_intent_table

        table = load_intent_table(self.intent_manifest())
        args = self.intent_args("unpaired-regression")
        config = json.loads(Path(args.config).read_text())
        config["goal_interface"]["intent_mode"] = "regression_only"
        config["intent_training"]["contrastive_weight"] = 0.
        Path(args.config).write_text(json.dumps(config))
        with patch("evo_wam.g_pi_intent.sample_intent_batch", return_value=table.entries[1:]):
            report = train_g_pi_interface(args)
        payload = read_g_pi_artifact(report["artifact"])
        self.assertEqual(payload["updates"], 0)
        self.assertEqual(payload["attempted_steps"], 1)
        self.assertFalse(payload["optimizer"]["state"])
        metric = json.loads((self.root / "unpaired-regression/metrics.jsonl").read_text())
        self.assertFalse(metric["updated"])
        self.assertEqual(metric["paired_samples"], 0)

    def test_one_step_parameter_ownership_and_exact_frozen_encoder(self):
        for route, stage in ROUTES.items():
            with self.subTest(route=route):
                config = config_for(route)
                sample = load_g_pi_sample(self.path, route=route, current_time=.4)
                native, interface, encoder, _, layers = build_g_pi_system(config, goal_registry(sample),
                    stage=stage, tiny_native=True, device="cuda")
                self.assertGreaterEqual(len(native.blocks), 2)
                self.assertFalse(any(p.requires_grad for p in encoder.parameters()))
                self.assertIs(encoder.native, native)
                self.assertEqual(encoder.native.patch_embedding_mlp.weight.data_ptr(), native.patch_embedding_mlp.weight.data_ptr())
                self.assertFalse(encoder.state_dict())
                self.assertEqual(native.patch_embedding_mlp.weight.dtype, torch.bfloat16)
                self.assertTrue(all(p.dtype == torch.float32 for p in interface.parameters()))
                action_names = {name for name, _ in action_named_parameters(native)}
                self.assertTrue(all(p.requires_grad == (route == "pi_goal" and name in action_names)
                                    for name, p in native.named_parameters()))
                before = copy.deepcopy(_system_state(native, interface, encoder))
                self.assertEqual(set(before), {"interface", "action"})
                self.assertTrue(all(tensor.dtype == torch.float32 for values in before.values() for tensor in values.values()))
                native_before = {name: value.detach().cpu().clone() for name, value in native.state_dict().items()}
                z_before = encoder(sample.target_frame)
                checksums = _frozen_checksums(native, encoder, config)
                optimizer = _optimizer(native, interface, encoder, config)
                generators = {name: torch.Generator().manual_seed(i + 2)
                              for i, name in enumerate(("action", "goal", "language"))}
                loss = g_pi_training_loss(native, interface, encoder, sample, config, generators,
                                         stage=stage, feature_layers=layers)
                loss["total"].backward()
                self.assertTrue(all(p.dtype == torch.float32 and (p.grad is None or p.grad.dtype == torch.float32)
                                    for group in optimizer.param_groups for p in group["params"]))
                optimizer.step()
                self.assertEqual(checksums, _frozen_checksums(native, encoder, config))
                torch.testing.assert_close(z_before, encoder(sample.target_frame), rtol=0, atol=0)
                self.assertTrue(any(not torch.equal(value.cpu(), before["interface"][name])
                                    for name, value in interface.state_dict().items()))
                changed = {name for name, value in native.state_dict().items()
                           if not torch.equal(value.cpu(), native_before[name])}
                if route == "pi_goal":
                    self.assertTrue(changed)
                    self.assertLessEqual(changed, action_names)
                else:
                    self.assertFalse(changed)
                self.assertFalse(encoder.state_dict())

    def stage_args(self, stage, output, *, steps=1, resume=None, initialize=None):
        args = self.args("pi_goal", output, steps=steps, resume=resume)
        args.stage, args.initialize = stage, initialize
        config = json.loads(Path(args.config).read_text())
        config["pi_training"] = {}
        config["goal_noise"] = ({"z_std": .03, "translation_std": .005, "rotation_std": .02,
            "gripper_std": .01, "translation_max_m": .02, "rotation_max_deg": 5.} if stage == "pi" else {})
        if stage == "pi":
            config["candidate_separation_m"] = .1
        Path(args.config).write_text(json.dumps(config))
        return args

    def test_prior_has_no_visual_forward_and_exact_resume_owns_only_prior_parameters(self):
        from evo_wam.g_pi_context import FrozenGoalEncoder

        forbidden = AssertionError("stage 1 accessed a visual path")
        with patch.object(FrozenGoalEncoder, "forward", side_effect=forbidden), \
             patch("evo_wam.g_pi_context.pi_context_features", side_effect=forbidden), \
             patch("evo_wam.g_pi_context.split_g_context_features", side_effect=forbidden):
            complete = train_g_pi_interface(self.stage_args("pi_prior", "prior-full", steps=2))
            partial = train_g_pi_interface(self.stage_args("pi_prior", "prior-part"))
            resumed = train_g_pi_interface(self.stage_args("pi_prior", "prior-part", resume=partial["artifact"]))
        full, continued = read_g_pi_artifact(complete["artifact"]), read_g_pi_artifact(resumed["artifact"])
        for key in ("model", "optimizer", "rng", "torch_rng", "cuda_rng", "frozen_checksums"):
            self.assert_same(full[key], continued[key])
        self.assertEqual(full["stage"], "pi_prior")
        self.assertTrue(all(key.startswith(("endpoint_encoder.", "state_encoder.", "condition_adapter."))
                            for key in full["model"]["interface"]))
        self.assertTrue(any(key.startswith("endpoint_encoder.") for key in full["model"]["interface"]))
        self.assertFalse(full["pi_training"]["goal_noise_enabled"])
        self.assertEqual(full["pi_training"]["pose_weight"], 0.)
        with self.assertRaisesRegex(ValueError, "not goal-policy deployment"):
            export_g_pi_policy(SimpleNamespace(artifact=complete["artifact"], output=self.root / "prior-export",
                                               stop_thresholds=explicit_thresholds()))

    def test_stage2_transfers_only_shared_parameters_and_resumes_noisy_endpoint_training(self):
        prior_report = train_g_pi_interface(self.stage_args("pi_prior", "prior"))
        prior = read_g_pi_artifact(prior_report["artifact"])
        args = self.stage_args("pi", "visual-full", steps=2, initialize=prior_report["artifact"])
        config = json.loads(Path(args.config).read_text())
        sample = load_g_pi_sample(self.path, route="pi_goal", generator=torch.Generator().manual_seed(0))
        native, interface, encoder, identity, layers = build_g_pi_system(config, goal_registry(sample),
            stage="pi", tiny_native=True, device="cuda")
        before = {key: value.detach().clone() for key, value in interface.state_dict().items()}
        _initialize_pi(native, interface, encoder, config, goal_registry(sample), prior, identity, layers)
        for key, value in interface.state_dict().items():
            if key.startswith(("state_encoder.", "condition_adapter.")):
                torch.testing.assert_close(value.cpu(), prior["model"]["interface"][key], rtol=0, atol=0)
            else:
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        for key, value in action_named_parameters(native):
            torch.testing.assert_close(value.cpu(), prior["model"]["action"][key], rtol=0, atol=0)
        self.assertFalse(any(value.requires_grad for value in interface.endpoint_encoder.parameters()))
        self.assertFalse(_optimizer(native, interface, encoder, config).state)
        self.assertFalse(any(key.startswith("endpoint_encoder.") for key in _system_state(native, interface, encoder)["interface"]))
        del native, interface, encoder
        complete = train_g_pi_interface(args)
        partial = train_g_pi_interface(self.stage_args("pi", "visual-part", initialize=prior_report["artifact"]))
        resumed = train_g_pi_interface(self.stage_args("pi", "visual-part", resume=partial["artifact"]))
        full, continued = read_g_pi_artifact(complete["artifact"]), read_g_pi_artifact(resumed["artifact"])
        for key in ("model", "optimizer", "rng", "torch_rng", "cuda_rng", "frozen_checksums", "pi_training"):
            self.assert_same(full[key], continued[key])
        self.assertEqual(full["stage1_artifact_sha256"], file_sha256(prior_report["artifact"]))
        self.assertEqual(full["pi_training"]["pose_weight"], .3)
        self.assertTrue(full["pi_training"]["goal_noise_enabled"])
        self.assertTrue(any(key.startswith("pose_decoder.") for key in full["model"]["interface"]))
        metrics = [json.loads(line) for line in (self.root / "visual-full/metrics.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["endpoint_valid_count"] == 1 for row in metrics))
        self.assertTrue(all(row["lit_pose_before_count"] + row["lit_pose_reaches_count"] == 1 for row in metrics))
        self.assertTrue(all(row["endpoint_position_error_m"] >= 0 and row["endpoint_orientation_error_deg"] >= 0
                            and row["endpoint_gripper_error"] >= 0 for row in metrics))
        exported = export_g_pi_policy(SimpleNamespace(artifact=complete["artifact"], output=self.root / "stage2-policy",
            stop_thresholds=explicit_thresholds(), dtype="bfloat16", max_shard_size="20KB"))
        native, interface, encoder, _ = load_g_pi_policy(exported["policy"], device="cuda")
        self.assert_same(full["model"], _system_state(native, interface, encoder))

    def test_endpoint_labels_never_enter_stage2_conditions_and_invalid_labels_are_masked(self):
        from dataclasses import replace

        args = self.stage_args("pi", "loss-only")
        config = json.loads(Path(args.config).read_text())
        sample = load_g_pi_sample(self.path, route="pi_goal", generator=torch.Generator().manual_seed(0))
        native, interface, encoder, _, layers = build_g_pi_system(config, goal_registry(sample),
            stage="pi", tiny_native=True, device="cuda")
        clean_pose, clean_grip = sample.block_end_poses.clone(), sample.block_end_gripper.clone()
        def loss(current):
            generators = {name: torch.Generator().manual_seed(i) for i, name in enumerate(("sample", "action", "goal", "language"))}
            return g_pi_training_loss(native, interface, encoder, current, config, generators, stage="pi", feature_layers=layers)
        first = loss(sample)
        other_pose = clean_pose.clone()
        other_pose[..., :3, 3] += .8
        second = loss(replace(sample, block_end_poses=other_pose))
        torch.testing.assert_close(first["action"], second["action"], rtol=0, atol=0)
        self.assertNotEqual(float(first["lit_pose"].detach()), float(second["lit_pose"].detach()))
        torch.testing.assert_close(sample.block_end_poses, clean_pose, rtol=0, atol=0)
        torch.testing.assert_close(sample.block_end_gripper, clean_grip, rtol=0, atol=0)
        masked = loss(replace(sample, block_end_valid=False, block_end_poses=None, block_end_gripper=None))
        self.assertEqual(float(masked["lit_pose"]), 0.)
        self.assertEqual(float(masked["endpoint_valid_count"]), 0.)
        torch.testing.assert_close(masked["total"], masked["action"], rtol=0, atol=0)
        native.zero_grad(set_to_none=True)
        interface.zero_grad(set_to_none=True)
        masked["total"].backward()
        self.assertTrue(all(p.grad is not None and not p.grad.any() for p in interface.pose_decoder.parameters()))
        self.assertTrue(all(p.grad is None for p in interface.endpoint_encoder.parameters()))

    def test_cached_targets_preserve_updates_and_resume_without_calling_E(self):
        from evo_wam.g_pi_context import FrozenGoalEncoder
        from evo_wam.g_pi_targets import build_target_cache_index, load_target_cache_index

        for route in ROUTES:
            with self.subTest(route=route):
                config = config_for(route)
                sample = load_g_pi_sample(self.path, route=route, generator=torch.Generator().manual_seed(0))
                native, interface, encoder, _, _ = build_g_pi_system(config, goal_registry(sample),
                    stage=ROUTES[route], tiny_native=True, device="cuda")
                cache_path = build_target_cache_index(self.index, self.root / f"{route}-cache", encoder)
                cache = load_target_cache_index(cache_path, encoder.identity, task_paths=[self.path])
                if route == "g_translator":
                    expected = g_intent_training_loss(native, interface, encoder,
                        [(sample.demonstration, sample)], [None], config)
                    with patch.object(FrozenGoalEncoder, "forward", side_effect=AssertionError("E was called")):
                        actual = g_intent_training_loss(native, interface, encoder,
                            [(sample.demonstration, sample)], [None], config,
                            cached_targets=[cache.goal(sample, self.path)])
                    for key in expected:
                        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
                del native, interface, encoder
                baseline = read_g_pi_artifact(train_g_pi_interface(self.args(route, f"{route}-online", steps=2))["artifact"])

                def arguments(output, steps=1, resume=None):
                    args = self.args(route, output, steps=steps, resume=resume)
                    cached_config = json.loads(Path(args.config).read_text())
                    cached_config["target_cache_index"] = str(cache_path)
                    Path(args.config).write_text(json.dumps(cached_config))
                    return args

                with patch.object(FrozenGoalEncoder, "forward", side_effect=AssertionError("E was called")):
                    partial = train_g_pi_interface(arguments(f"{route}-cached"))
                    continued = train_g_pi_interface(arguments(f"{route}-cached", resume=partial["artifact"]))
                cached = read_g_pi_artifact(continued["artifact"])
                for key in ("model", "optimizer", "rng", "torch_rng", "cuda_rng", "frozen_checksums"):
                    self.assert_same(baseline[key], cached[key])
                self.assertTrue(set(map(lambda path: str(path.resolve()), cache.files)) <= set(cached["visited_arrays"]))
                archive = next(path for path in cache.files if path.suffix == ".npz" and path.parent == cache_path.parent)
                with archive.open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(ValueError, "consumed training inputs changed"):
                    train_g_pi_interface(arguments(f"{route}-cached", resume=continued["artifact"]))

    def test_independent_exact_resume_and_changed_arrays_rejected(self):
        for route in ROUTES:
            with self.subTest(route=route):
                complete = train_g_pi_interface(self.args(route, f"{route}-full", steps=2))
                partial = train_g_pi_interface(self.args(route, f"{route}-part"))
                resumed = train_g_pi_interface(self.args(route, f"{route}-part", resume=partial["artifact"]))
                full, continuation = read_g_pi_artifact(complete["artifact"]), read_g_pi_artifact(resumed["artifact"])
                for name in ("model", "optimizer", "rng", "torch_rng", "cuda_rng", "python_rng", "numpy_rng",
                             "data_cursor", "frozen_checksums", "encoder_identity"):
                    self.assert_same(full[name], continuation[name])
        path = self.root / "task.npz"
        with np.load(path) as source:
            arrays = {name: source[name].copy() for name in source.files}
        arrays["actions"][0, 0] += 1
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, "consumed training inputs changed"):
            train_g_pi_interface(self.args(route, f"{route}-part", resume=resumed["artifact"]))

    def test_sharded_export_rebuilds_shared_base_and_preserves_fp32_trainables(self):
        for route in ROUTES:
            report = train_g_pi_interface(self.args(route, f"{route}-train"))
            payload = read_g_pi_artifact(report["artifact"])
            for dtype in ("float32", "bfloat16"):
                with self.subTest(route=route, dtype=dtype):
                    result = export_g_pi_policy(SimpleNamespace(stop_thresholds=explicit_thresholds(), artifact=report["artifact"],
                        output=str(self.root / f"{route}-{dtype}"), dtype=dtype, max_shard_size="20KB"))
                    native, interface, encoder, policy = load_g_pi_policy(result["policy"], device="cuda",
                        expected_encoder_identity=payload["encoder_identity"])
                    self.assertGreater(len(policy["shards"]), 1)
                    self.assertIs(encoder.native, native)
                    self.assertFalse(encoder.state_dict())
                    self.assertEqual(policy["precision"], "float32")
                    self.assertEqual(set(payload["model"]), {"interface", "action"})
                    self.assertTrue(all(name.startswith(("action.", "interface.")) for name in policy["weight_map"]))
                    self.assert_same(payload["model"], _system_state(native, interface, encoder))
                    self.assertEqual(frozen_base_checksum(encoder.native), payload["encoder_identity"]["base_sha256"])
                    self.assertFalse(any(p.requires_grad for m in (native, interface, encoder) for p in m.parameters()))
                    with self.assertRaisesRegex(ValueError, "E identity mismatch"):
                        load_g_pi_policy(result["policy"], device="cuda", expected_encoder_identity={})

    def test_g_deployment_reuses_pi_base_without_touching_trained_actions(self):
        policies, training = {}, {}
        for route in ROUTES:
            report = train_g_pi_interface(self.args(route, f"share-{route}-train"))
            training[route] = read_g_pi_artifact(report["artifact"])
            exported = export_g_pi_policy(SimpleNamespace(stop_thresholds=explicit_thresholds(), artifact=report["artifact"],
                output=self.root / f"share-{route}-policy", dtype="float32"))
            policies[route] = exported["policy"]
        native, pi, encoder, metadata = load_g_pi_policy(policies["pi_goal"], device="cuda")
        before = {name: value.detach().clone() for name, value in action_named_parameters(native)}
        checksum = frozen_base_checksum(native)
        with patch("evo_wam.g_pi_training.build_g_pi_system", side_effect=AssertionError("must not rebuild shared base")):
            shared, decoder, target, _ = load_g_pi_policy(policies["g_translator"], device="cuda",
                shared_base=(native, encoder, metadata))
        self.assertIs(shared, native)
        self.assertIs(target, encoder)
        self.assert_same(training["g_translator"]["model"], _system_state(shared, decoder, target))
        self.assertEqual(frozen_base_checksum(native), checksum)
        for name, value in action_named_parameters(native):
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assert_same(training["pi_goal"]["model"], _system_state(native, pi, encoder))
        with self.assertRaisesRegex(ValueError, "device"):
            load_g_pi_policy(policies["g_translator"], device="cpu", shared_base=(native, encoder, metadata))
        with self.assertRaisesRegex(ValueError, "when loading G"):
            load_g_pi_policy(policies["pi_goal"], device="cuda", shared_base=(native, encoder, metadata))
        wrong = copy.deepcopy(metadata)
        wrong["encoder_identity"]["base_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            load_g_pi_policy(policies["g_translator"], device="cuda", shared_base=(native, encoder, wrong))
        with torch.no_grad():
            native.patch_embedding_mlp.weight.view(-1)[0].add_(1)
        with self.assertRaisesRegex(ValueError, "shared base checksum"):
            load_g_pi_policy(policies["g_translator"], device="cuda", shared_base=(native, encoder, metadata))


if __name__ == "__main__":
    unittest.main()
