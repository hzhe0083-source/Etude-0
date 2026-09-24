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
from evo_wam.g_pi_training import (ROUTES, _base_location, _base_reference, _frozen_checksums, _optimizer, _restore_system, _set_precision, _system_state,
    build_g_pi_system, export_g_pi_policy, g_pi_architecture, g_pi_training_loss,
    load_g_pi_policy, read_g_pi_artifact, train_g_pi_interface, validate_g_pi_artifact, validate_g_pi_config)
from evo_wam.goal_action import action_named_parameters
from evo_wam.goal_training import goal_registry
from evo_wam.cli import file_sha256
from evo_wam.zerowam import NativeDependencyError, ZERO_WAM_COMMIT, load_native_class
from test_g_pi_data import write_g_pi_task
from test_native_icl import tiny_model


def config_for(route):
    config = json.loads((Path(__file__).parents[1] / "configs/se3/goal_interface.json").read_text())
    config.update(interface_type=route, video_weight=0., pose_weight=.3, action_sampling_steps=1,
                  window_size=8, goal_encoder={"layer": 1, "k_z": 8}, event_rules=asdict(EventRules()))
    config.pop("sampling_steps")
    config["ifp"].update(enabled=False, loss_weights=[0.])
    config["training"].update(learning_rate=.001, max_steps=8)
    config["goal_interface"] = {"state_dim": 4, "dim": 16, "num_heads": 2, "translation_scale": 1.}
    if route == "g_translator":
        config["goal_interface"].update(num_layers=2, use_state=True)
    else:
        config["goal_interface"].update(num_tokens=4, num_layer_groups=1, num_pose_tokens=1)
    return config


def contract_payload(route="g_translator"):
    config = config_for(route)
    identity = {"layer": 1, "k_z": 8, "d_z": 36, "timestep": 0,
                "pooling": "adaptive_avg_pool1d_spatial", "normalization": "l2_last_dim",
                "base_id": {"kind": "fixture"}, "base_sha256": "a" * 64,
                "empty_text_identity": {"source": {"kind": "fixture"}, "sha256": "b" * 64}}
    return {"format_version": 2, "kind": "g_pi_training", "config": config,
            "interface_type": route,
            "architecture": g_pi_architecture(config), "upstream_commit": ZERO_WAM_COMMIT,
            "stage": ROUTES[route], "precision": "float32", "encoder_precision": "float32",
            "base_identity": identity["base_id"], "encoder_identity": identity,
            "empty_text_identity": identity["empty_text_identity"], "k_z": 8, "d_z": 36,
            "event_rules": config["event_rules"], "registry": {"event_rules": config["event_rules"]},
            "tiny_native": True,
            "base_reference": {"kind": "tiny-native", "checkpoint": None, "base_seed": 0,
                               "identity": identity["base_id"], "encoder_identity": identity,
                               "empty_text_identity": identity["empty_text_identity"], "video_precision": "float32"},
            "model": {"interface": {"weight": torch.ones(2, 2)}, "action": {}}}


class GPiTrainingContractTest(unittest.TestCase):
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
            export = export_g_pi_policy(SimpleNamespace(artifact=artifact, output=Path(folder) / "policy", dtype="bfloat16"))
            deployed = json.loads((Path(export["policy"]) / "policy.json").read_text())
            validate_g_pi_artifact(deployed, kind="g_pi_policy", expected_encoder_identity=payload["encoder_identity"])
            self.assertEqual(deployed["precision"], "float32")
            self.assertTrue(all(name.startswith("interface.") for name in deployed["weight_map"]))

    def test_config_rejects_joint_video_adaptation_and_invalid_event_contracts(self):
        for route in ROUTES:
            self.assertIsNotNone(validate_g_pi_config(config_for(route)))
        for updates in ({"video_weight": 1.}, {"sampling_steps": 1}, {"lora": {}},
                        {"event_rules": {}}, {"goal_encoder": {"layer": -1}},
                        {"goal_noise": {"z_std": -1}}, {"base_seed": True},
                        {"empty_emb_path": "/tmp/a", "empty_text_emb_path": "/tmp/b"}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                validate_g_pi_config({**config_for("g_translator"), **updates})

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
                exported = export_g_pi_policy(SimpleNamespace(artifact=artifact,
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
                              for i, name in enumerate(("action", "goal"))}
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
                    result = export_g_pi_policy(SimpleNamespace(artifact=report["artifact"],
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


if __name__ == "__main__":
    unittest.main()
