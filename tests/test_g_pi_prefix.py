import unittest
from unittest.mock import patch

import torch

from etude.g_pi_context import (
    RobotPrefixCache, g_context_features, pi_context_features, split_g_context_features,
)
from etude.zerowam import NativeDependencyError, load_native_class
from test_g_pi_context import CONFIG, context_model


class PrefixContractTests(unittest.TestCase):
    def test_owner_validation_and_clear(self):
        with self.assertRaisesRegex(ValueError, "owner"):
            RobotPrefixCache("shared")
        cache = RobotPrefixCache("g")
        cache.processed_tokens, cache.closed_frames = 12, 4
        cache.clear()
        self.assertEqual((cache.processed_tokens, cache.closed_frames), (0, 0))
        self.assertIsNone(cache.identity)
        history = torch.ones(1, 4, 1, 1, 2)
        with self.assertRaisesRegex(ValueError, "owner mismatch"):
            pi_context_features(None, history, CONFIG, prefix_cache=cache)


@unittest.skipUnless(torch.cuda.is_available(), "Native FlexAttention requires CUDA")
class NativePrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            load_native_class()
        except NativeDependencyError as exc:
            raise unittest.SkipTest(str(exc))

    def setUp(self):
        torch.manual_seed(830)
        self.native = context_model("cuda")
        self.demo = torch.randn(1, 4, 3, 1, 5, device="cuda")
        self.history = torch.randn(1, 4, 9, 4, 5, device="cuda")

    def test_every_prefix_matches_truncation_across_padding_and_open_chunks(self):
        # Twenty tokens/frame crosses the 128-token padding boundary at t=6.
        for dtype in (torch.bfloat16, torch.float32):
            self.native.to(dtype=dtype)
            for owner in ("g", "pi"):
                with self.subTest(dtype=dtype, owner=owner):
                    cache = RobotPrefixCache(owner)
                    def read(video, **kwargs):
                        if owner == "g":
                            return g_context_features(self.native, self.demo, video, CONFIG, **kwargs)
                        return pi_context_features(self.native, video, CONFIG, **kwargs)
                    cached_prefix, cached_kv = None, None
                    projected = []
                    handle = self.native.blocks[0].attn1.to_q.register_forward_pre_hook(
                        lambda module, args: projected.append(args[0].shape[1]))
                    try:
                        for frames in range(1, 10):
                            baseline = read(self.history, current_index=frames - 1)
                            before = len(projected)
                            cached = read(self.history, current_index=frames - 1, prefix_cache=cache)
                            processed = (frames - (frames - 1) // 2 * 2) * 20
                            self.assertEqual(cache.last_processed_tokens, processed)
                            # First G call encodes the demonstration once, then only robot suffixes.
                            self.assertEqual(projected[before:], ([15] if owner == "g" and frames == 1 else []) + [processed])
                            self.assertEqual(cache.closed_frames, frames // 2 * 2)
                            for layer in range(2):
                                torch.testing.assert_close(cached[layer], baseline[layer], rtol=.015, atol=.015)
                                if cached_prefix is not None:
                                    n = cached_prefix[layer].shape[1]
                                    torch.testing.assert_close(cache.features[layer][:, :n], cached_prefix[layer], rtol=0, atol=0)
                                    for axis in range(2):
                                        torch.testing.assert_close(cache.keys_values[layer][axis][:, :n],
                                                                   cached_kv[layer][axis], rtol=0, atol=0)
                            if cache.closed_frames:
                                cached_prefix = {i: v.clone() for i, v in cache.features.items()}
                                cached_kv = tuple(tuple(v.clone() for v in pair) for pair in cache.keys_values)
                    finally:
                        handle.remove()
                    self.assertLess(cache.processed_tokens, sum(range(1, 10)) * 20)
                    self.assertTrue(all(k.shape[1] == 160 for k, v in cache.keys_values))
                    self.assertTrue(all(not v.requires_grad for v in cache.features.values()))

    def test_closed_prefix_reuses_all_features_without_native_work(self):
        cache = RobotPrefixCache("pi")
        first = pi_context_features(self.native, self.history, CONFIG, current_index=3, prefix_cache=cache)
        with patch("etude.g_pi_context._context", side_effect=AssertionError("cached prefix recomputed")):
            repeated = pi_context_features(self.native, self.history, CONFIG, current_index=3, prefix_cache=cache)
        self.assertEqual(cache.last_processed_tokens, 0)
        for layer in range(2):
            torch.testing.assert_close(first[layer], repeated[layer], rtol=0, atol=0)

    def test_future_physical_truncation_is_exact_and_cache_identity_is_strict(self):
        cache = RobotPrefixCache("g")
        read = lambda video, **kwargs: g_context_features(self.native, self.demo, video, CONFIG, **kwargs)
        baseline = read(self.history, current_index=2, prefix_cache=cache)
        future = self.history.clone()
        future[:, :, 3:] = float("nan")
        actual = read(future, current_index=2, prefix_cache=cache)
        for layer in range(2):
            torch.testing.assert_close(actual[layer], baseline[layer], rtol=0, atol=0)
        changed = self.history.clone()
        changed[:, :, 0] += 1
        with self.assertRaisesRegex(ValueError, "history mismatch"):
            read(changed, current_index=3, prefix_cache=cache)
        with self.assertRaisesRegex(ValueError, "history mismatch"):
            read(self.history, current_index=0, prefix_cache=cache)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            g_context_features(self.native, self.demo + 1, self.history, CONFIG, prefix_cache=cache)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            g_context_features(self.native, self.demo, self.history, dict(CONFIG, chunk_size=1), prefix_cache=cache)
        with self.assertRaisesRegex(ValueError, "owner mismatch"):
            pi_context_features(self.native, self.history, CONFIG, prefix_cache=cache)
        with torch.no_grad():
            self.native.blocks[0].attn1.to_k.weight.add_(.1)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            read(self.history, prefix_cache=cache)
        cache.clear()
        read(self.history, current_index=1, prefix_cache=cache)

    def test_wrappers_own_separate_task_caches_and_preserve_action_rng(self):
        from etude.g_pi_interface import GTranslator, PiGoalPolicy
        from test_g_pi_interface import GPiNativeInterfaceTests
        native, encoder, decoder, interface, config, demo, history, state, language = (
            GPiNativeInterfaceTests().fixture("cuda"))
        config["chunk_size"] = 2
        translator = GTranslator(native, decoder, config)
        policy = PiGoalPolicy(native, interface, config, action_shape=(1, 3, 1, 2, 1),
                              actions_mask=torch.ones(1, 3, 1, 2, 1, dtype=torch.bool))
        self.assertIsNot(translator._prefix_cache, policy._prefix_cache)
        for index in range(3):
            plain_goal = translator.predict(demo, history, state, current_index=index)
            cached_goal = translator.predict(demo, history, state, current_index=index, use_prefix_cache=True)
            for name in plain_goal:
                torch.testing.assert_close(cached_goal[name], plain_goal[name], rtol=.015, atol=.015)
            seed = policy.generator.get_state()
            plain = policy.predict(history, state, language, plain_goal, current_index=index)
            policy.generator.set_state(seed)
            cached = policy.predict(history, state, language, plain_goal, current_index=index, use_prefix_cache=True)
            torch.testing.assert_close(cached, plain, rtol=.015, atol=.015)
        self.assertEqual(translator._prefix_cache.closed_frames, 2)
        self.assertEqual(policy._prefix_cache.closed_frames, 2)
        translator.predict(demo + 1, history, state, current_index=0, use_prefix_cache=True)
        self.assertEqual(translator._prefix_cache.observed_frames, 1)
        self.assertEqual(policy._prefix_cache.observed_frames, 3)
        policy.clear_context_cache()
        policy.predict(history + 1, state, goal=plain_goal, current_index=0, use_prefix_cache=True)
        self.assertEqual(policy._prefix_cache.observed_frames, 1)
        translator.clear_demo_cache()
        self.assertIsNone(translator._prefix_cache.identity)
        self.assertFalse(any("cache" in key for key in translator.state_dict()))
        self.assertFalse(any("cache" in key for key in policy.state_dict()))

    def test_via_u_only_cache_keeps_g_identity_and_never_reads_demo_in_robot(self):
        config = dict(CONFIG, demo_route="via_u_only")
        cache = RobotPrefixCache("g")
        split_g_context_features(self.native, self.demo, self.history, config, current_index=1, prefix_cache=cache)
        demo, robot = split_g_context_features(self.native, self.demo, self.history, config,
                                              current_index=4, prefix_cache=cache)
        expected_demo, expected_robot = split_g_context_features(self.native, self.demo, self.history,
                                                                 config, current_index=4)
        pi = pi_context_features(self.native, self.history, config, current_index=4)
        for layer in range(2):
            torch.testing.assert_close(demo[layer], expected_demo[layer], rtol=0, atol=0)
            torch.testing.assert_close(robot[layer], expected_robot[layer], rtol=.015, atol=.015)
            torch.testing.assert_close(robot[layer], pi[layer], rtol=.015, atol=.015)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            split_g_context_features(self.native, self.demo + 1, self.history, config, prefix_cache=cache)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            split_g_context_features(self.native, self.demo, self.history, CONFIG, prefix_cache=cache)


if __name__ == "__main__":
    unittest.main()
