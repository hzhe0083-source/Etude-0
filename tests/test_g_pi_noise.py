from copy import deepcopy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from evo_wam.g_pi_deployment import save_goal_prediction
from evo_wam.g_pi_noise import (calibrate_g_pi_noise, calibrate_noise, perturb_goal_bounded,
                               validate_goal_noise)
from evo_wam.goal_interface import validate_goal_poses
from test_g_pi_calibration import encoder_identity, goal


class BoundedGoalNoiseTest(unittest.TestCase):
    def fixture(self, batch=8, *, dtype=torch.float32):
        return {"z": torch.arange(1, batch * 8 + 1, dtype=dtype).reshape(batch, 2, 4).requires_grad_(),
                "goal_poses": torch.eye(4, dtype=dtype).repeat(batch, 2, 1, 1).requires_grad_(),
                "goal_gripper": torch.full((batch, 2), .5, dtype=dtype).requires_grad_()}

    def settings(self):
        return dict(z_std=.03, translation_std=.005, rotation_std=.03, gripper_std=.02,
                    translation_max_m=.02, rotation_max_deg=10., candidate_separation_m=.1)

    def test_disabled_noise_is_exact_detached_and_does_not_advance_rng(self):
        original = self.fixture(dtype=torch.float64)
        generator = torch.Generator().manual_seed(33)
        state = generator.get_state().clone()
        actual = perturb_goal_bounded(original, generator=generator)
        self.assertTrue(torch.equal(state, generator.get_state()))
        self.assertTrue(torch.equal(actual["z"], F.normalize(original["z"].float(), dim=-1)))
        for key in ("goal_poses", "goal_gripper"):
            self.assertTrue(torch.equal(actual[key], original[key]))
            self.assertNotEqual(actual[key].data_ptr(), original[key].data_ptr())
        self.assertTrue(all(not value.requires_grad for value in actual.values()))

    def test_deterministic_resume_all_components_and_clean_goal_unchanged(self):
        original = self.fixture()
        clean = {key: value.detach().clone() for key, value in original.items()}
        first = torch.Generator().manual_seed(8)
        perturb_goal_bounded(original, generator=first, **self.settings())
        saved = first.get_state().clone()
        expected = perturb_goal_bounded(original, generator=first, **self.settings())
        resumed = torch.Generator().set_state(saved)
        actual = perturb_goal_bounded(original, generator=resumed, **self.settings())
        for key in original:
            self.assertTrue(torch.equal(actual[key], expected[key]))
            self.assertTrue(torch.equal(original[key], clean[key]))
            self.assertFalse(actual[key].requires_grad)
        self.assertTrue(torch.equal(first.get_state(), resumed.get_state()))

    def test_z_preserves_normalize_then_add_existing_semantics(self):
        original = self.fixture()
        generator = torch.Generator().manual_seed(90)
        expected = F.normalize(original["z"].float(), dim=-1) + .3 * torch.randn(original["z"].shape, generator=generator)
        actual = perturb_goal_bounded(original, z_std=.3, generator=torch.Generator().manual_seed(90))
        self.assertTrue(torch.equal(actual["z"], expected))
        self.assertFalse(torch.allclose(actual["z"].norm(dim=-1), torch.ones(8, 2)))

    def test_radial_translation_rotation_bounds_and_gripper_range(self):
        original = self.fixture(1000)
        actual = perturb_goal_bounded(original, translation_std=.1, translation_max_m=.01,
                                      rotation_std=2., rotation_max_deg=8., gripper_std=4.,
                                      candidate_separation_m=.03, generator=torch.Generator().manual_seed(20))
        distances = (actual["goal_poses"][..., :3, 3] - original["goal_poses"][..., :3, 3]).norm(dim=-1)
        self.assertTrue((distances < .01).all())
        rotation = actual["goal_poses"][..., :3, :3].double()
        angles = torch.acos(((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
        self.assertLessEqual(angles.max().item(), math.radians(8) + 1e-6)
        self.assertGreater(angles.max().item(), math.radians(7))
        validate_goal_poses(actual["goal_poses"])
        self.assertTrue(((actual["goal_gripper"] >= 0) & (actual["goal_gripper"] <= 1)).all())

    def test_both_truncated_normal_samplers_have_no_clipping_boundary_mass(self):
        # A 3-D Gaussian radius follows Maxwell; conditioning gives F(r)/F(R).
        def maxwell(radius, std):
            ratio = radius / std
            return math.erf(ratio / math.sqrt(2.)) - math.sqrt(2. / math.pi) * ratio * math.exp(-ratio ** 2 / 2.)

        for radius in (.008, .025):
            with self.subTest(radius=radius):
                actual = perturb_goal_bounded(self.fixture(6000), translation_std=.01,
                    translation_max_m=radius, candidate_separation_m=.1, generator=torch.Generator().manual_seed(34))
                norms = actual["goal_poses"][..., :3, 3].norm(dim=-1)
                self.assertEqual((norms >= radius).sum().item(), 0)
                expected = maxwell(radius / 2, .01) / maxwell(radius, .01)
                self.assertAlmostEqual((norms < radius / 2).float().mean().item(), expected, delta=.02)

    def test_translation_bound_survives_adding_to_nonzero_float32_pose(self):
        original = self.fixture(3000)
        original["goal_poses"] = original["goal_poses"].detach()
        original["goal_poses"][..., :3, 3] = torch.tensor([20., -30., 4.])
        actual = perturb_goal_bounded(original, translation_std=.1, translation_max_m=.01,
                                      candidate_separation_m=.020001, generator=torch.Generator().manual_seed(27))
        offset = actual["goal_poses"][..., :3, 3].double() - original["goal_poses"][..., :3, 3].double()
        self.assertTrue((offset.norm(dim=-1) <= .01).all())
        self.assertTrue((offset.norm(dim=-1) < .020001 / 2).all())

    def test_translation_requires_explicit_strict_candidate_separation(self):
        cases = [dict(translation_std=.01), dict(translation_std=.01, translation_max_m=.02),
                 dict(translation_std=.01, candidate_separation_m=.1),
                 dict(translation_std=.01, translation_max_m=.05, candidate_separation_m=.1),
                 dict(translation_std=.01, translation_max_m=.06, candidate_separation_m=.1),
                 dict(translation_std=.01, translation_max_m=0., candidate_separation_m=.1)]
        for settings in cases:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                perturb_goal_bounded(self.fixture(), **settings)
        valid = perturb_goal_bounded(self.fixture(), translation_std=.01, translation_max_m=.049,
                                    candidate_separation_m=.1, generator=torch.Generator().manual_seed(1))
        self.assertTrue((valid["goal_poses"][..., :3, 3].norm(dim=-1) < .05).all())

    def test_rotation_units_and_bounds_are_explicit(self):
        for settings in (dict(rotation_std=.01), dict(rotation_std=.01, rotation_max_deg=0),
                         dict(rotation_std=.01, rotation_max_deg=181)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                perturb_goal_bounded(self.fixture(), **settings)
        settings = validate_goal_noise({"rotation_std": math.pi / 180, "rotation_max_deg": 3})
        self.assertEqual(settings["rotation_std"], math.pi / 180)
        self.assertEqual(settings["rotation_max_deg"], 3.)

    def test_invalid_noise_config_goal_and_generator_fail_clearly(self):
        for value in (True, -1., float("inf"), float("nan")):
            for key in ("z_std", "translation_std", "rotation_std", "gripper_std", "translation_max_m", "rotation_max_deg"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_goal_noise({key: value})
        for separation in (0., -1., float("inf"), True):
            with self.assertRaises(ValueError):
                validate_goal_noise({}, candidate_separation_m=separation)
        with self.assertRaisesRegex(ValueError, "only"):
            validate_goal_noise({"object_pose_noise": 1.})
        with self.assertRaisesRegex(ValueError, "Generator"):
            perturb_goal_bounded(self.fixture(), generator=3)
        invalid = self.fixture()
        invalid["z"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "goal z"):
            perturb_goal_bounded(invalid)


class NoiseCalibrationTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.identity = encoder_identity()
        self.registry = {"coordinate_frame": "robot_base", "pose_units": "m", "pose_representation": "absolute_robot_base_tool",
                         "tool_frames": ["tcp"], "end_effectors": ["gripper"],
                         "gripper_space": {"normalization_id": "fixture", "closed": [0.], "open": [1.], "units": "fraction"}}
        self.manifest = {"format_version": 1, "kind": "g_pi_noise_validation", "split": "validation",
                         "encoder_identity": self.identity, "registry": self.registry, "candidate_separation_m": .2,
                         "records": []}
        self.path = self.root / "validation.json"
        self.output = self.root / "calibrated.json"

    def pair(self, sample_id, source_group, *, position=.01, rotation=5., z_angle=.1, grip=.05):
        for name, value in (("reference", goal()), ("prediction", goal(z_angle, position=position, rotation=rotation, gripper=.5 + grip))):
            save_goal_prediction(self.root / f"{sample_id}-{name}.npz", value, self.identity, self.registry)
        self.manifest["records"].append({"sample_id": sample_id, "source_group": source_group,
                                        "prediction": f"{sample_id}-prediction.json", "reference": f"{sample_id}-reference.json"})

    def calibrate(self):
        self.path.write_text(json.dumps(self.manifest))
        return calibrate_noise(self.path, self.output)

    def test_validation_goal_sidecars_produce_ready_config_with_units_and_hashes(self):
        self.pair("first", "source-a", position=.01, rotation=5.)
        self.pair("second", "source-b", position=.02, rotation=30.)
        result = self.calibrate()
        settings = result["config"]["goal_noise"]
        self.assertAlmostEqual(settings["translation_std"], .02 / math.sqrt(3), places=7)
        self.assertAlmostEqual(settings["translation_max_m"], .02, places=7)
        self.assertAlmostEqual(settings["rotation_std"], math.radians(30) / math.sqrt(3), places=6)
        self.assertAlmostEqual(settings["rotation_max_deg"], 30., places=5)
        self.assertAlmostEqual(settings["z_std"], (2. - 2. * math.cos(.1)) ** .5 / math.sqrt(2), places=6)
        self.assertEqual(result["units"]["rotation_std"], "rad")
        self.assertEqual(result["units"]["rotation_max_deg"], "deg")
        self.assertEqual(len(result["validation_files"]), 9)
        self.assertFalse(result["method"]["confidence_calibration"])
        self.assertEqual(validate_goal_noise(settings, candidate_separation_m=result["config"]["candidate_separation_m"]), settings)

    def test_source_groups_have_equal_mass_despite_many_frames(self):
        for index in range(20):
            self.pair(f"frequent-{index}", "frequent", position=.001)
        self.pair("rare", "rare", position=.04)
        self.manifest["scale_quantile"] = .75
        result = self.calibrate()
        self.assertAlmostEqual(result["config"]["goal_noise"]["translation_std"], .04 / math.sqrt(3), places=7)
        self.assertEqual(result["evidence"]["source_group_counts"], {"frequent": 20, "rare": 1})

    def test_wrong_object_scale_is_rejected_instead_of_clipped(self):
        self.pair("first", "a", position=.001)
        self.pair("second", "b", position=.11)
        self.manifest["scale_quantile"] = .1
        with self.assertRaisesRegex(ValueError, "half.*separation"):
            self.calibrate()
        self.assertFalse(self.output.exists())

    def test_nonzero_translation_requires_candidate_separation_even_for_calibration(self):
        self.pair("first", "a")
        self.pair("second", "b")
        self.manifest.pop("candidate_separation_m")
        with self.assertRaisesRegex(ValueError, "candidate separation"):
            self.calibrate()

    def test_perfect_goals_can_disable_all_noise_without_separation(self):
        self.pair("first", "a", position=0., rotation=0., z_angle=0., grip=0.)
        self.pair("second", "b", position=0., rotation=0., z_angle=0., grip=0.)
        self.manifest.pop("candidate_separation_m")
        result = self.calibrate()
        self.assertFalse(any(result["config"]["goal_noise"].values()))
        self.assertNotIn("candidate_separation_m", result["config"])

    def test_validation_split_group_identity_and_units_are_enforced(self):
        self.pair("first", "a")
        self.pair("second", "b")
        original = deepcopy(self.manifest)
        for field, value, error in (("split", "train", "validation-only"), ("scale_quantile", 0, "scale_quantile")):
            self.manifest = {**deepcopy(original), field: value}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                self.calibrate()
        self.manifest = deepcopy(original)
        self.manifest["registry"]["pose_units"] = "mm"
        with self.assertRaisesRegex(ValueError, "metre"):
            self.calibrate()
        self.manifest = deepcopy(original)
        self.manifest["records"][1]["source_group"] = "a"
        with self.assertRaisesRegex(ValueError, "two independent"):
            self.calibrate()

    def test_E_identity_mismatch_and_tampered_goal_cache_are_rejected(self):
        self.pair("first", "a")
        self.pair("second", "b")
        self.manifest["encoder_identity"]["base_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "E identity"):
            self.calibrate()
        self.manifest["encoder_identity"]["base_sha256"] = "a" * 64
        (self.root / "first-prediction.npz").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.calibrate()

    def test_duplicate_files_and_sample_ids_are_not_independent_evidence(self):
        self.pair("first", "a")
        self.pair("second", "b")
        second = self.manifest["records"][1]
        original = dict(second)
        second["sample_id"] = "first"
        with self.assertRaisesRegex(ValueError, "sample_id"):
            self.calibrate()
        second.update(original, prediction="first-prediction.json", reference="first-reference.json")
        with self.assertRaisesRegex(ValueError, "repeated.*files"):
            self.calibrate()

    def test_cli_wrapper_and_non_overwrite(self):
        self.pair("first", "a")
        self.pair("second", "b")
        self.path.write_text(json.dumps(self.manifest))
        result = calibrate_g_pi_noise(SimpleNamespace(manifest=self.path, output=self.output))
        self.assertEqual(json.loads(self.output.read_text()), result)
        with self.assertRaisesRegex(ValueError, "new file"):
            calibrate_g_pi_noise(SimpleNamespace(manifest=self.path, output=self.output))


if __name__ == "__main__":
    unittest.main()
