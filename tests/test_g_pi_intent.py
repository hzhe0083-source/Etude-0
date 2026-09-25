from dataclasses import fields
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from etude.g_pi_intent import (IntentDemo, intent_contrastive_loss, intent_table_files,
                                load_intent_demo, load_intent_table, ordered_intent_similarity,
                                sample_intent_batch)
from etude.icl_data import LATENT_NORMALIZATION
from test_g_pi_data import write_g_pi_task


def write_intent_table(root, *, version="v1", groups=2, per_group=2):
    entries = []
    for group in range(groups):
        for sample in range(per_group):
            name = f"demo-{group}-{sample}"
            np.savez_compressed(root / f"{name}.npz",
                                latent=np.random.default_rng(group * per_group + sample).normal(
                                    size=(2, 3, 2, 2)).astype(np.float32),
                                frame_times=np.array([0., .4, .8], dtype=np.float64))
            entry = {"demo_id": name, "purpose_group": f"purpose-{group}", "split": "train",
                     "source_id": name, "source_group": name, "person_id": f"person-{sample}",
                     "scene_id": f"scene-{sample}", "view_id": f"view-{sample}",
                     "object_ids": ["cup" if version == "v1" else f"object-{sample}"],
                     "arrays": f"{name}.npz", "complete_demo": True,
                     "relation_signature": f"result-{group}", "role_signature": f"roles-{group}",
                     "order_signature": f"order-{group}"}
            if version == "v1":
                entry["object_family"] = "tableware"
            else:
                entry.update(operation="move", role_candidates={"acted_on": 1})
            entries.append(entry)
    document = {"format_version": 1, "kind": "g_pi_intent_groups", "data_version": version,
                "feature_space_id": "fixture-wan-v1", "latent_normalization": LATENT_NORMALIZATION,
                "grouping_evidence": "fixture audited purpose equivalence, not a relation vocabulary",
                "entries": entries}
    path = root / "intent.json"
    path.write_text(json.dumps(document))
    return path, document


class IntentDataTest(unittest.TestCase):
    def test_demo_only_model_boundary_has_no_labels_or_text(self):
        with TemporaryDirectory() as tmp:
            path, _ = write_intent_table(Path(tmp))
            table = load_intent_table(path)
            self.assertTrue(all(entry.paired_task is None for entry in table.entries))
            demo = load_intent_demo(table.entries[0])
            self.assertEqual(demo.demonstration.shape, (1, 2, 3, 2, 2))
            self.assertEqual({field.name for field in fields(IntentDemo)},
                             {"demonstration", "demonstration_times"})
            self.assertEqual(set(intent_table_files(table)), {path, *(entry.arrays for entry in table.entries)})

    def test_schema_rejects_partial_clips_text_and_implicit_grouping(self):
        with TemporaryDirectory() as tmp:
            path, document = write_intent_table(Path(tmp))
            for mutate in (lambda d: d["entries"][0].update(complete_demo=False),
                           lambda d: d["entries"][0].update(text="move the cup"),
                           lambda d: d.pop("grouping_evidence"),
                           lambda d: d.update(data_version="v3")):
                data = json.loads(json.dumps(document))
                mutate(data)
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_intent_table(path)

    def test_v1_similar_objects_are_explicit(self):
        with TemporaryDirectory() as tmp:
            path, document = write_intent_table(Path(tmp))
            document["entries"][1]["object_family"] = "tools"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "object_family"):
                load_intent_table(path)

    def test_v2_unique_training_roles_and_different_object_positives(self):
        with TemporaryDirectory() as tmp:
            path, document = write_intent_table(Path(tmp), version="v2")
            for entry in document["entries"]:
                entry.update(person_id="one-person", scene_id="one-scene", view_id="one-view")
            path.write_text(json.dumps(document))
            table = load_intent_table(path)
            batch = sample_intent_batch(table, torch.Generator().manual_seed(2))
            self.assertNotEqual(batch[0].object_ids, batch[1].object_ids)
            document["entries"][0]["role_candidates"]["acted_on"] = 2
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "one candidate"):
                load_intent_table(path)
            document["entries"][0]["split"] = "test"
            path.write_text(json.dumps(document))
            self.assertEqual(load_intent_table(path).entries[0].metadata["role_candidates"]["acted_on"], 2)

    def test_v2_identical_objects_cannot_supply_configured_positive_diversity(self):
        with TemporaryDirectory() as tmp:
            path, document = write_intent_table(Path(tmp), version="v2")
            for entry in document["entries"]:
                entry["object_ids"] = ["same-object"]
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "diverse"):
                sample_intent_batch(load_intent_table(path), torch.Generator().manual_seed(0))

    def test_transitive_aliases_are_not_independent_positives(self):
        with TemporaryDirectory() as tmp:
            path, document = write_intent_table(Path(tmp))
            first, second = document["entries"][:2]
            document["source_aliases"] = [
                {"source_id": first["source_id"], "source_group": "alias", "domain": "human", "split": "train"},
                {"source_id": second["source_id"], "source_group": "alias", "domain": "human", "split": "train"}]
            path.write_text(json.dumps(document))
            table = load_intent_table(path)
            self.assertEqual(table.entries[0].component, table.entries[1].component)
            with self.assertRaisesRegex(ValueError, "independent"):
                sample_intent_batch(table, torch.Generator().manual_seed(0))

    def test_repacked_video_with_renamed_source_is_not_independent(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, document = write_intent_table(root)
            with np.load(root / document["entries"][0]["arrays"]) as archive:
                np.savez(root / document["entries"][1]["arrays"], **{key: archive[key] for key in archive.files})
            table = load_intent_table(path)
            self.assertEqual(table.entries[0].component, table.entries[1].component)
            document["entries"][1]["split"] = "test"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "crosses"):
                load_intent_table(path)

    def test_source_cannot_have_conflicting_complete_purposes(self):
        with TemporaryDirectory() as tmp:
            path, document = write_intent_table(Path(tmp))
            document["entries"][2]["source_group"] = document["entries"][0]["source_group"]
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "conflicting"):
                load_intent_table(path)

    def test_uncertain_positive_not_sampled_and_invalid_pairs_rejected(self):
        with TemporaryDirectory() as tmp:
            path, document = write_intent_table(Path(tmp))
            document["uncertain_pairs"] = [[entry["demo_id"] for entry in document["entries"][:2]]]
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "non-uncertain"):
                sample_intent_batch(load_intent_table(path), torch.Generator().manual_seed(0))
            document["uncertain_pairs"] = [["missing", "demo-0-0"]]
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "known"):
                load_intent_table(path)

    def test_batch_two_independent_per_group_and_exact_rng_resume(self):
        with TemporaryDirectory() as tmp:
            path, _ = write_intent_table(Path(tmp), groups=3, per_group=3)
            table = load_intent_table(path)
            generator = torch.Generator().manual_seed(12)
            sample_intent_batch(table, generator)
            saved = generator.get_state().clone()
            expected = sample_intent_batch(table, generator)
            restored = torch.Generator().set_state(saved)
            self.assertEqual(expected, sample_intent_batch(table, restored))
            for group in {entry.purpose_group for entry in expected}:
                values = [entry for entry in expected if entry.purpose_group == group]
                self.assertEqual(len(values), 2)
                self.assertEqual(len({entry.component for entry in values}), 2)

    def test_same_object_structure_and_scene_negatives_rank_above_easy(self):
        with TemporaryDirectory() as tmp:
            path, document = write_intent_table(Path(tmp), groups=3)
            for entry in document["entries"][4:]:
                entry.update(object_ids=["hammer"], scene_id="elsewhere")
            path.write_text(json.dumps(document))
            table = load_intent_table(path)
            # Fix first group while leaving source shuffles driven by the RNG.
            with patch("etude.g_pi_intent.torch.randint", return_value=torch.tensor(0)):
                batch = sample_intent_batch(table, torch.Generator().manual_seed(4))
            self.assertEqual({entry.purpose_group for entry in batch}, {"purpose-0", "purpose-1"})
            self.assertTrue(any(a.scene_id == b.scene_id and a.purpose_group != b.purpose_group
                                for a in batch for b in batch))

    def test_available_same_scene_pair_survives_representative_sampling(self):
        with TemporaryDirectory() as tmp:
            path, document = write_intent_table(Path(tmp), groups=3, per_group=3)
            for index, entry in enumerate(document["entries"]):
                entry["scene_id"] = f"private-scene-{index}"
            document["entries"][2]["scene_id"] = "shared-scene"
            document["entries"][5]["scene_id"] = "shared-scene"
            path.write_text(json.dumps(document))
            table = load_intent_table(path)
            for seed in range(10):
                batch = sample_intent_batch(table, torch.Generator().manual_seed(seed))
                self.assertEqual(sum(entry.scene_id == "shared-scene" for entry in batch), 2)

    def test_uncertain_isolated_group_does_not_hide_other_valid_batch(self):
        with TemporaryDirectory() as tmp:
            path, document = write_intent_table(Path(tmp), groups=3)
            document["uncertain_pairs"] = [[a["demo_id"], b["demo_id"]]
                                            for a in document["entries"][:2] for b in document["entries"][2:]]
            path.write_text(json.dumps(document))
            table = load_intent_table(path)
            for seed in range(3):
                batch = sample_intent_batch(table, torch.Generator().manual_seed(seed))
                self.assertEqual({entry.purpose_group for entry in batch}, {"purpose-1", "purpose-2"})

    def test_pairing_reads_no_language_and_rejects_wrong_demo(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, document = write_intent_table(root)
            task_path, task, _ = write_g_pi_task(root)
            entry = document["entries"][0]
            task["demonstration"].update(source_id=entry["source_id"], source_group=entry["source_group"], arrays=entry["arrays"])
            task["language"] = "absent-language.json"
            task_path.write_text(json.dumps(task))
            entry["paired_task"] = task_path.name
            path.write_text(json.dumps(document))
            table = load_intent_table(path)
            files = intent_table_files(table)
            self.assertIn(task_path, files)
            self.assertFalse(any("language" in file.name for file in files))
            task["demonstration"]["source_id"] = "other"
            task_path.write_text(json.dumps(task))
            with self.assertRaisesRegex(ValueError, "exactly this"):
                load_intent_table(path)


class OrderedContrastiveTest(unittest.TestCase):
    def entries(self):
        from types import SimpleNamespace
        return [SimpleNamespace(demo_id=str(i), purpose_group=str(i // 2), component=i) for i in range(4)]

    def test_role_and_step_permutation_remain_negative(self):
        forward = torch.eye(3)
        reverse = forward[[2, 1, 0]]
        roles = forward[[1, 0, 2]]
        u = torch.stack((forward, reverse, roles))
        self.assertTrue(torch.equal(forward.mean(0), reverse.mean(0)))
        self.assertTrue(torch.equal(forward.mean(0), roles.mean(0)))
        similarity = ordered_intent_similarity(u)
        torch.testing.assert_close(similarity.diagonal(), torch.ones(3))
        self.assertLess(float(similarity[0, 1]), .34)
        self.assertLess(float(similarity[0, 2]), .34)

    def test_correct_slot_groups_have_lower_loss_and_gradients(self):
        forward, backward = torch.eye(2), torch.eye(2)[[1, 0]]
        correct = torch.stack((forward, forward, backward, backward)).requires_grad_()
        loss = intent_contrastive_loss(correct, self.entries())
        wrong = correct.detach()[[0, 2, 1, 3]]
        self.assertLess(float(loss.detach()), float(intent_contrastive_loss(wrong, self.entries())))
        loss.backward()
        self.assertTrue(torch.isfinite(correct.grad).all())
        self.assertGreater(float(correct.grad.abs().sum()), 0.)

    def test_uncertain_pairs_removed_from_denominator_exactly(self):
        u = torch.randn(4, 2, 4, generator=torch.Generator().manual_seed(4))
        entries = self.entries()
        unknown = frozenset((frozenset(("0", "2")),))
        logits = ordered_intent_similarity(u) / .2
        permitted = [[1, 3], [0, 2, 3], [1, 3], [0, 1, 2]]
        partner = [1, 0, 3, 2]
        manual = torch.stack([torch.logsumexp(logits[i, permitted[i]], 0) - logits[i, partner[i]]
                              for i in range(4)]).mean()
        torch.testing.assert_close(intent_contrastive_loss(u, entries, temperature=.2, uncertain_pairs=unknown), manual)

    def test_copies_and_uncertain_positives_never_supervise(self):
        u = torch.randn(4, 2, 3)
        entries = self.entries()
        entries[1].component = entries[0].component
        with self.assertRaisesRegex(ValueError, "independent positive"):
            intent_contrastive_loss(u, entries)
        with self.assertRaisesRegex(ValueError, "independent positive"):
            intent_contrastive_loss(u, self.entries(), uncertain_pairs={frozenset(("0", "1"))})

    def test_invalid_temperature_and_shapes_rejected(self):
        for value in (0., -1., float("nan")):
            with self.assertRaisesRegex(ValueError, "temperature"):
                intent_contrastive_loss(torch.ones(4, 2, 3), self.entries(), temperature=value)
        with self.assertRaisesRegex(ValueError, "ordered slots"):
            ordered_intent_similarity(torch.ones(4, 3))


if __name__ == "__main__":
    unittest.main()
