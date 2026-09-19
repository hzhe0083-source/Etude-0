"""Capacity plumbing checks with synthetic data, not transfer or execution evidence."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from evo_wam.cli import make_fixture
from evo_wam.data import load_experiment, load_sample
from evo_wam.models import EffectReader, RequirementCodec, decoded_requirement_loss
from evo_wam.video_cli import build_video_models, load_video_config, make_capacity_configs
from evo_wam.video_data import patch_grid_coordinates
from evo_wam.video_effects import VideoEffectEncoder, EffectFeaturePredictor, effect_pretraining_loss


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.video_path = CONFIGS / "video" / "U2_effect_constraints.json"
        self.robot_path = CONFIGS / "full.json"
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_candidates_change_only_capacity_and_preserve_source_files(self):
        source_bytes = [path.read_bytes() for path in (self.video_path, self.robot_path)]
        video = load_video_config(self.video_path)
        robot = load_experiment(self.robot_path)
        result = make_capacity_configs(self.video_path, self.robot_path, self.root / "capacity")
        self.assertEqual(set(result["video_configs"]), {"16", "64", "100"})
        self.assertEqual(result["primary_video_config"], result["video_configs"]["64"])
        for count, path in result["video_configs"].items():
            with self.subTest(count=count):
                expected = copy.deepcopy(video)
                expected["model"].update(num_tokens=int(count), latent_dim=768, hidden_dim=512)
                self.assertEqual(load_video_config(path), expected)
                self.assertEqual(Path(path).name, f"video_K{count}.json")
        expected = copy.deepcopy(robot)
        expected["dimensions"].update(demo_dim=768, token_dim=768)
        self.assertEqual(load_experiment(result["robot_config"]), expected)
        self.assertEqual(Path(result["robot_config"]).name, "robot_768.json")
        self.assertEqual([path.read_bytes() for path in (self.video_path, self.robot_path)], source_bytes)

    def test_rejects_nonempty_output_without_overwriting(self):
        output = self.root / "capacity"
        output.mkdir()
        sentinel = output / "video_K16.json"
        sentinel.write_text("keep this experiment")
        with self.assertRaisesRegex(ValueError, "fresh output directory"):
            make_capacity_configs(self.video_path, self.robot_path, output)
        self.assertEqual(sentinel.read_text(), "keep this experiment")
        self.assertEqual(list(output.iterdir()), [sentinel])
        with self.assertRaisesRegex(ValueError, "fresh output directory"):
            make_capacity_configs(self.video_path, self.robot_path, sentinel)

    def test_invalid_source_is_rejected_before_writing(self):
        video = load_video_config(self.video_path)
        video["model"]["feature_dim"] = 0
        source = self.root / "invalid.json"
        source.write_text(json.dumps(video))
        output = self.root / "capacity"
        with self.assertRaisesRegex(ValueError, "feature_dim"):
            make_capacity_configs(source, self.robot_path, output)
        self.assertFalse(output.exists())

    def test_new_capacity_cannot_inherit_old_validation_locks(self):
        robot = load_experiment(self.robot_path)
        robot["validation_locked"] = True
        robot["binding_policy"]["validation_locked"] = True
        source = self.root / "previously-validated.json"
        source.write_text(json.dumps(robot))
        result = make_capacity_configs(self.video_path, source, self.root / "capacity")
        generated = load_experiment(result["robot_config"])
        self.assertFalse(generated["validation_locked"])
        self.assertFalse(generated["binding_policy"]["validation_locked"])
        self.assertEqual(generated["binding_policy"]["confidence_threshold"],
                         robot["binding_policy"]["confidence_threshold"])
        with self.assertRaisesRegex(ValueError, "lock"):
            load_experiment(result["robot_config"], for_test=True)
        self.assertTrue(json.loads(source.read_text())["validation_locked"])

    def test_768_width_pretraining_export_reader_and_codec_have_gradients(self):
        torch.manual_seed(23)
        result = make_capacity_configs(self.video_path, self.robot_path, self.root / "capacity")
        video = load_video_config(result["primary_video_config"])
        config = load_experiment(result["robot_config"])
        encoder, predictor = build_video_models(video, "cpu")
        encoder.noise_std = 0
        # Constructor defaults and generated configs must agree on capacity.
        default_encoder = VideoEffectEncoder(feature_dim=4)
        default_predictor = EffectFeaturePredictor(feature_dim=4)
        self.assertEqual((default_encoder.num_tokens, default_encoder.latent_dim, default_encoder.hidden_dim),
                         (64, 768, 512))
        self.assertEqual((default_predictor.latent_dim, default_predictor.hidden_dim), (768, 512))
        del default_encoder, default_predictor

        features = torch.randn(1, 6, 2, video["model"]["feature_dim"])
        valid = torch.ones_like(features, dtype=torch.bool)
        times = torch.arange(6).float()
        layout = {"feature_kind": "patches", "patch_coordinates": patch_grid_coordinates([1, 2])}
        tokens = encoder(features[:, :3], valid[:, :3], times[:3], **layout)
        self.assertEqual(tokens.shape, (1, 64, 768))
        prediction = predictor(features[:, :1], valid[:, :1], tokens, times[1:3],
                               past_times=times[:1], effect_fields=(), **layout)
        loss = effect_pretraining_loss(prediction, features[:, 1:3], valid[:, 1:3],
                                       {}, {}, tokens, video["loss_weights"])["total"]
        loss.backward()
        for weight in (encoder.features.weight, predictor.feature_head.weight):
            self.assertTrue(torch.isfinite(weight.grad).all())
            self.assertGreater(weight.grad.abs().sum().item(), 0)

        encoder.zero_grad(set_to_none=True)
        demonstration = encoder.encode_demo(features, valid, times, video["window_frames"], **layout)
        self.assertEqual(demonstration.shape, (1, 128, 768))
        make_fixture(self.root / "robot")
        sample = load_sample(self.root / "robot" / "sample.json")
        dims = config["dimensions"]
        reader = EffectReader(**{key: dims[key] for key in
            ("demo_dim", "entity_dim", "proprio_dim", "embodiment_dim", "roles", "token_dim")})
        codec = RequirementCodec(**{key: dims[key] for key in
            ("entity_dim", "geometry_dim", "relation_dim", "event_dim", "roles", "token_dim")})
        goals = reader(demonstration, sample.robot_history, sample.proprio_history, sample.embodiment,
                       sample.requirement.current.step_offsets, sample.requirement.remaining.step_offsets)
        losses = []
        for part in ("current", "remaining"):
            goal = getattr(goals, part)
            requirement = getattr(sample.requirement, part)
            self.assertEqual(goal.shape, (1, config["tokens"][part], 768))
            decoded = codec.decode(goal, sample.robot_history[:, -1],
                                   requirement.step_offsets, requirement.entity_ids)
            losses.append(decoded_requirement_loss(decoded, requirement))
        sum(losses).backward()
        for weight in (encoder.features.weight, reader.demo.weight, codec.geometry.weight):
            self.assertTrue(torch.isfinite(weight.grad).all())
            self.assertGreater(weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
