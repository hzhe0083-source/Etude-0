import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from etude.g_pi_distributed import distributed_settings, process_group
from etude.g_pi_training import (export_g_pi_policy, load_g_pi_policy, read_g_pi_artifact,
                                  train_g_pi_interface, _system_state, validate_g_pi_config)
from test_g_pi_training import config_for, contract_payload, explicit_thresholds


class DistributedPiContractTest(unittest.TestCase):
    def test_disabled_default_and_invalid_routes_or_flags(self):
        self.assertEqual(distributed_settings(config_for("pi_goal")),
                         {"enabled": False, "activation_checkpointing": True})
        for values in ({"enabled": 1}, {"unknown": True}, {"activation_checkpointing": "yes"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_g_pi_config({**config_for("pi_goal"), "distributed": values})
        with self.assertRaisesRegex(ValueError, "independent pi"):
            validate_g_pi_config({**config_for("g_translator"), "distributed": {"enabled": True}})
        with self.assertRaisesRegex(ValueError, "CUDA"):
            with process_group("cpu"):
                pass

    def test_checkpoint_requires_sharding_and_every_rank_rng(self):
        payload = contract_payload("pi_goal")
        payload["config"]["distributed"] = {"enabled": True}
        with self.assertRaisesRegex(ValueError, "FSDP artifact"):
            read_g_pi_artifact(None, payload=payload)
        payload["distributed"] = {"kind": "fsdp2", "world_size": 2, "activation_checkpointing": True,
            "batch_layout": "one_sample_per_rank_round_robin", "optimizer_format": "full_named_v1"}
        with self.assertRaisesRegex(ValueError, "per rank"):
            read_g_pi_artifact(None, payload=payload)
        payload["config"]["distributed"] = {"enabled": False}
        with self.assertRaisesRegex(ValueError, "disabled FSDP"):
            read_g_pi_artifact(None, payload=payload)

    def test_rank_consensus_and_consumed_input_mismatch_are_rejected(self):
        from etude.g_pi_distributed import rank_consensus, merge_input_hashes

        def records(values):
            def gather(output, value):
                output[:] = values
            return gather

        with patch("etude.g_pi_distributed.dist.get_world_size", return_value=2):
            with patch("etude.g_pi_distributed.dist.all_gather_object", side_effect=records([{"base": "a"}, {"base": "b"}])):
                with self.assertRaisesRegex(ValueError, "ranks disagree"):
                    rank_consensus({"base": "a"})
            with patch("etude.g_pi_distributed.dist.all_gather_object", side_effect=records([{"task": "a"}, {"task": "b"}])):
                with self.assertRaisesRegex(ValueError, "inputs differ"):
                    merge_input_hashes({}, {"task": "a"})
            with patch("etude.g_pi_distributed.dist.all_gather_object", side_effect=records([{"task-a": "a"}, {"task-b": "b"}])):
                self.assertEqual(merge_input_hashes({"old": "c"}, {}),
                                 {"task-a": "a", "task-b": "b", "old": "c"})

    @unittest.skipUnless(torch.cuda.is_available(), "Native FSDP2 test requires CUDA")
    def test_torchrun_single_gpu_exact_resume_checkpoint_and_export(self):
        environment = {**os.environ, "OMP_NUM_THREADS": "1"}
        result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node=1", str(Path(__file__).resolve()), "--worker"],
            capture_output=True, text=True, env=environment, timeout=240)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("FSDP2_NATIVE_CHECKS_PASSED", result.stdout)


def worker():
    from test_g_pi_training import GPiNativeTrainingTest

    fixture = GPiNativeTrainingTest("test_independent_exact_resume_and_changed_arrays_rejected")
    fixture.setUp()
    try:
        prior = train_g_pi_interface(fixture.stage_args("pi_prior", "fsdp-prior"))
        def args(output, steps=1, resume=None, ac=True):
            value = fixture.stage_args("pi", output, steps=steps, resume=resume,
                                       initialize=None if resume else prior["artifact"])
            config = json.loads(Path(value.config).read_text())
            config["distributed"] = {"enabled": True, "activation_checkpointing": ac}
            Path(value.config).write_text(json.dumps(config))
            return value

        complete = train_g_pi_interface(args("full", steps=2))
        partial = train_g_pi_interface(args("part"))
        resumed = train_g_pi_interface(args("part", resume=partial["artifact"]))
        full, continued = read_g_pi_artifact(complete["artifact"]), read_g_pi_artifact(resumed["artifact"])
        for key in ("model", "optimizer", "rank_states", "data_cursor", "frozen_checksums", "encoder_identity"):
            fixture.assert_same(full[key], continued[key])
        fixture.assertEqual(full["distributed"]["kind"], "fsdp2")
        fixture.assertEqual(full["distributed"]["world_size"], 1)
        fixture.assertEqual(set(full["model"]), {"interface", "action"})
        fixture.assertTrue(full["model"]["action"])
        fixture.assertTrue(any(key.startswith("pose_decoder.") for key in full["model"]["interface"]))
        fixture.assertFalse(any(key.startswith("endpoint_encoder.") for key in full["model"]["interface"]))
        fixture.assertTrue(full["pi_training"]["goal_noise_enabled"])
        fixture.assertIsNotNone(full["stage1_artifact_sha256"])
        fixture.assertFalse(any("_checkpoint_wrapped_module" in name for values in full["model"].values() for name in values))
        fixture.assertFalse(any("patch_embedding" in name for name in full["optimizer"]["state"]))
        fixture.assertTrue(all(value.dtype == torch.float32 for values in full["model"].values() for value in values.values()))
        exported = export_g_pi_policy(SimpleNamespace(artifact=complete["artifact"], output=fixture.root / "export",
            stop_thresholds=explicit_thresholds(), dtype="bfloat16", max_shard_size="20KB"))
        native, interface, encoder, metadata = load_g_pi_policy(exported["policy"], device="cuda")
        fixture.assertGreater(len(metadata["shards"]), 1)
        fixture.assert_same(full["model"], _system_state(native, interface, encoder))
        fixture.assertNotIn("rank_states", metadata)
        del native, interface, encoder
        # Checkpoint recomputation must preserve the exact action mask. A fresh
        # one-step run without AC produces the same update as the AC run.
        without_ac = read_g_pi_artifact(train_g_pi_interface(args("no-ac", ac=False))["artifact"])
        # partial path was resumed to step two, so compare to a fresh AC step.
        one_ac = read_g_pi_artifact(train_g_pi_interface(args("one-ac"))["artifact"])
        fixture.assert_same(one_ac["model"], without_ac["model"])
        from etude.g_pi_data import load_g_pi_sample
        from etude.g_pi_distributed import process_group, shard_pi, distributed_checksum
        from etude.g_pi_training import build_g_pi_system, _optimizer
        from etude.goal_training import goal_registry
        from torch.distributed.fsdp import FSDPModule
        config = config_for("pi_goal")
        config["distributed"] = {"enabled": True, "activation_checkpointing": True}
        with process_group("cuda"):
            sample = load_g_pi_sample(fixture.path, route="pi_goal", generator=torch.Generator().manual_seed(0))
            native, interface, encoder, _, layers = build_g_pi_system(config, goal_registry(sample),
                stage="pi", tiny_native=True, device="cuda")
            expected = encoder(sample.target_frame).clone()
            checksum = encoder.identity["base_sha256"]
            shard_pi(native, interface, config)
            fixture.assertIsInstance(native, FSDPModule)
            fixture.assertIsInstance(native.blocks[0].attn1, FSDPModule)
            fixture.assertIsInstance(interface, FSDPModule)
            optimizer = _optimizer(native, interface, encoder, config)
            rng = {key: torch.Generator().manual_seed(i) for i, key in enumerate(("sample", "goal", "action", "language"))}
            loss = native.g_pi_loss(encoder, sample, rng, layers, None)
            loss["total"].backward()
            optimizer.step()
            with torch.no_grad():
                actual = native.g_pi_encode(encoder, sample.target_frame)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            fixture.assertEqual(distributed_checksum(native), checksum)
        print("FSDP2_NATIVE_CHECKS_PASSED")
    finally:
        fixture.doCleanups()


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        unittest.main()
