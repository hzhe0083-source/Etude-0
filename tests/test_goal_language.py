import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from etude.goal_language import cache_goal_language, load_goal_language
from etude.vision import sha256


def write_goal_language(root, name="language", *, text="Follow the demonstrated operation.", dimension=8):
    language = np.zeros((1, 512, dimension), dtype=np.float32)
    language[:, :3] = np.arange(3 * dimension, dtype=np.float32).reshape(1, 3, dimension) / dimension
    arrays_path = root / f"{name}.npz"
    np.savez_compressed(arrays_path, language=language)
    metadata = {
        "format_version": 1, "kind": "goal_language", "arrays": arrays_path.name,
        "arrays_sha256": sha256(arrays_path), "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "cleaned_text": text, "valid_length": 3,
        "identity": {
            "encoder_sha256": {"config.json": "a" * 64, "model.safetensors": "b" * 64},
            "tokenizer_sha256": {"tokenizer.json": "c" * 64}, "text_dim": dimension,
            "preprocessing": {"cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean",
                              "max_length": 512, "padding": "max_length", "truncation": True,
                              "add_special_tokens": True, "zero_padding": True, "output_dtype": "float32",
                              "encoder_dtype": "float32", "diffusers_version": "fixture",
                              "transformers_version": "fixture"},
        },
    }
    path = root / f"{name}.json"
    path.write_text(json.dumps(metadata))
    return path, metadata, language


class GoalLanguageTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_cache_validation_trimming_and_instruction_independent_identity(self):
        path, metadata, language = write_goal_language(self.root)
        output, identity = load_goal_language(path)
        self.assertEqual(output.shape, (1, 3, 8))
        self.assertFalse(output.requires_grad)
        torch.testing.assert_close(output, torch.from_numpy(language[:, :3]))
        other, _, _ = write_goal_language(self.root, "other", text="Perform another demonstration.")
        self.assertEqual(identity, load_goal_language(other)[1])
        path.write_text(json.dumps({**metadata, "text": "Silently changed instruction"}))
        with self.assertRaisesRegex(ValueError, "text hash"):
            load_goal_language(path)
        path.write_text(json.dumps(metadata))
        language[:, 0] += 1
        np.savez_compressed(self.root / metadata["arrays"], language=language)
        with self.assertRaisesRegex(ValueError, "arrays hash"):
            load_goal_language(path)

    def test_rejects_invalid_length_padding_features_and_policy(self):
        path, metadata, language = write_goal_language(self.root)
        for length in (0, 513, True, 3.5):
            path.write_text(json.dumps({**metadata, "valid_length": length}))
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, "valid_length"):
                load_goal_language(path)
        for bad in (language[:, :3], language.astype(np.float64), language + 1,
                    np.full_like(language, np.nan)):
            np.savez_compressed(self.root / metadata["arrays"], language=bad)
            path.write_text(json.dumps({**metadata, "arrays_sha256": sha256(self.root / metadata["arrays"])}))
            with self.subTest(shape=bad.shape, dtype=bad.dtype), self.assertRaisesRegex(ValueError, "zero padding"):
                load_goal_language(path)
        changed = json.loads(json.dumps(metadata))
        changed["identity"]["preprocessing"]["max_length"] = 256
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "native prompt_clean"):
            load_goal_language(path)
        changed = json.loads(json.dumps(metadata))
        changed["identity"]["tokenizer_sha256"] = {"../outside": "c" * 64}
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "file identity"):
            load_goal_language(path)

    def test_cache_uses_native_frozen_local_encoder_and_zeroes_padding(self):
        try:
            from diffusers.pipelines.wan.pipeline_wan import prompt_clean  # noqa: F401
            from transformers import T5TokenizerFast, UMT5EncoderModel  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"Native text preprocessing dependencies unavailable: {exc}")
        checkpoint = self.root / "checkpoint"
        for folder in ("text_encoder", "tokenizer"):
            (checkpoint / folder).mkdir(parents=True)
        for filename in ("text_encoder/config.json", "text_encoder/model.safetensors", "tokenizer/tokenizer.json"):
            (checkpoint / filename).write_text("fixture")
        encoder = torch.nn.Linear(1, 1)
        observed = []

        def encode(ids, mask):
            observed.append((encoder.training, torch.is_grad_enabled(), encoder.weight.requires_grad))
            return SimpleNamespace(last_hidden_state=torch.ones(1, 512, 8))

        encoder.forward = encode
        tokens = SimpleNamespace(input_ids=torch.zeros(1, 512, dtype=torch.long),
                                 attention_mask=torch.cat((torch.ones(1, 3), torch.zeros(1, 509)), dim=1).long())
        args = SimpleNamespace(text="  Do &amp; follow  ", checkpoint=str(checkpoint),
                               output=str(self.root / "cache"), device="cpu")
        with patch("transformers.UMT5EncoderModel.from_pretrained", return_value=encoder) as load_encoder, \
                patch("transformers.T5TokenizerFast.from_pretrained") as load_tokenizer:
            load_tokenizer.return_value.return_value = tokens
            result = cache_goal_language(args)
        self.assertEqual(observed, [(False, False, False)])
        self.assertTrue(load_encoder.call_args.kwargs["local_files_only"])
        self.assertTrue(load_encoder.call_args.kwargs["use_safetensors"])
        self.assertTrue(load_tokenizer.call_args.kwargs["local_files_only"])
        tokenizer_call = load_tokenizer.return_value.call_args
        self.assertEqual(tokenizer_call.args[0], ["Do & follow"])
        self.assertEqual(tokenizer_call.kwargs["max_length"], 512)
        self.assertTrue(tokenizer_call.kwargs["truncation"])
        cached, _ = load_goal_language(result["manifest"])
        torch.testing.assert_close(cached, torch.ones(1, 3, 8))
        with self.assertRaisesRegex(ValueError, "fresh directory"):
            cache_goal_language(args)
        args.checkpoint = "https://example.invalid/model"
        with self.assertRaisesRegex(ValueError, "local Zero-WAM"):
            cache_goal_language(args)


if __name__ == "__main__":
    unittest.main()
