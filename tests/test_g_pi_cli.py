import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from etude.cli import main


class GPiCLIEntryTest(unittest.TestCase):
    def test_intent_table_override_reaches_route_before_config_validation(self):
        config = Path(__file__).parents[1] / "configs/se3/g_translator.json"
        # This preset has a positive contrastive weight but no embedded dataset
        # path. Route validation must occur after --intent-groups is applied.
        with patch("etude.g_pi_training.train_g_pi_interface", return_value={"routed": True}) as train, \
                contextlib.redirect_stdout(io.StringIO()):
            status = main(["train-goal-interface", "--config", str(config), "--index", "index.json",
                           "--intent-groups", "purposes.json", "--stage", "g", "--tiny-native",
                           "--device", "cuda", "--output", "unused"])
        self.assertEqual(status, 0)
        self.assertEqual(train.call_args.args[0].intent_groups, "purposes.json")

    def test_missing_table_still_rejects_nonzero_contrastive_training(self):
        config = Path(__file__).parents[1] / "configs/se3/g_translator.json"
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            status = main(["train-goal-interface", "--config", str(config), "--index", "index.json",
                           "--stage", "g", "--tiny-native", "--output", "unused"])
        self.assertEqual(status, 2)
        self.assertIn("positive contrastive_weight requires an intent manifest", error.getvalue())

    def test_probe_and_evaluation_entry_points_dispatch_all_options(self):
        for command, target, options, attribute in (
            ("probe-g-pi-intent", "etude.g_pi_probe.probe_g_pi_intent",
             ["--artifact", "training.pt", "--checkpoint", "base"], "artifact"),
            ("evaluate-g-pi", "etude.g_pi_evaluation.evaluate_g_pi_cli",
             ["--g-checkpoint", "base", "--pi-checkpoint", "base"], "g_checkpoint"),
        ):
            with self.subTest(command=command), patch(target, return_value={"routed": True}) as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                status = main([command, "--manifest", "manifest.json", "--output", "report.json",
                               "--device", "cpu", *options])
            self.assertEqual(status, 0)
            self.assertEqual(run.call_args.args[0].device, "cpu")
            self.assertTrue(getattr(run.call_args.args[0], attribute))

    def test_pi_prior_stage_and_noise_calibration_entry_points(self):
        config = Path(__file__).parents[1] / "configs/se3/pi_prior.json"
        with patch("etude.g_pi_training.train_g_pi_interface", return_value={}) as train, \
                contextlib.redirect_stdout(io.StringIO()):
            status = main(["train-goal-interface", "--config", str(config), "--index", "index.json",
                           "--stage", "pi_prior", "--tiny-native", "--output", "prior"])
        self.assertEqual(status, 0)
        self.assertEqual(train.call_args.args[0].stage, "pi_prior")
        with patch("etude.g_pi_noise.calibrate_g_pi_noise", return_value={}) as calibrate, \
                contextlib.redirect_stdout(io.StringIO()):
            status = main(["calibrate-g-pi-noise", "--manifest", "validation.json", "--output", "noise.json"])
        self.assertEqual(status, 0)
        self.assertEqual(calibrate.call_args.args[0].manifest, "validation.json")

    def test_target_cache_config_and_humangen_commands_dispatch(self):
        with patch("etude.g_pi_targets.cache_g_pi_targets", return_value={}) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            status = main(["cache-g-pi-targets", "--index", "index.json", "--config", "pi.json",
                           "--tiny-native", "--split", "validation", "--output", "cache"])
        self.assertEqual(status, 0)
        self.assertTrue(run.call_args.args[0].tiny_native)
        self.assertEqual(run.call_args.args[0].split, "validation")
        self.assertIsNone(run.call_args.args[0].artifact)
        for command, name in (("convert-humangen-g-pi", "convert_humangen_g_pi"),
                              ("audit-humangen-g-pi", "audit_humangen_g_pi")):
            with self.subTest(command=command), patch(f"etude.g_pi_humangen.{name}", return_value={}) as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                status = main([command, "--manifest", "conversion.json", "--output", "converted"])
            self.assertEqual(status, 0)
            self.assertEqual(run.call_args.args[0].manifest, "conversion.json")
