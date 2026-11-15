"""CPU checks for scoring, parent-level uncertainty and durable evaluation.

Run from Qwen_vl: python -m unittest scripts.paired_pruning.test_report -v
No checkpoint, model, dataset download, or GPU is needed.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .evaluate import (ResultStore, arm_specs, generation_record, json_hash,
                       official_anls, sha256, strict_match, validate_keep_indices,
                       verify_freeze)
from .report import build_report, load_database, markdown_report, paired_bootstrap


def fixture():
    protocol = {"n": 4, "k": 2, "max_new_tokens": 128, "seeds": [11]}
    specs = arm_specs(protocol)
    question_a = {"item_id": "q1", "question_id": 1, "question": "Which value?",
                  "answers": ["12", "twelve"], "numeric": True}
    question_b = {"item_id": "q2", "question_id": 2, "question": "Who?",
                  "answers": ["Ada"], "numeric": False}
    test = [{"page_id": "p1", "parent_id": "parent1", "questions": [question_a]},
            {"page_id": "p2", "parent_id": "parent2", "questions": [question_b]}]
    dev_q = {"item_id": "devq", "question": "Development question?", "answers": ["x"], "numeric": False}
    manifest = {"splits": {"test": test, "dev": [{"page_id": "dev", "parent_id": "devparent", "questions": [dev_q]}]}}
    frozen = {"freeze_sha256": "frozen", "arms": specs,
              "timing": {"caps": [1, 16], "warmup": 0, "blocks": 1, "scope": "fresh test fixture"}}
    records, timings = [], []
    for spec in specs:
        count = 4 if spec["arm"] == "full" else 2
        source = 2 if spec["arm"] == "rres512" else 4
        for page, question in [(test[0], question_a), (test[1], question_b)]:
            # Full-model errors are deliberately present: evaluation must not filter them.
            text = "wrong" if spec["arm"] == "full" else question["answers"][0]
            output = {"text": text, "token_ids": [10], "truncated": False,
                      "actual_visual_count": count, "keep_idx": list(range(count)),
                      "ds_lengths": [count] * 3, "positions_sha256": "a" * 64,
                      "audit": {"native": True, "ds_streams": 3, "positions_preserved": True,
                                "physical_visual_count": count}}
            records.append(generation_record(output, page, question, spec, source, count, 128, "frozen"))
        for cap in (1, 16):
            timings.append({"item_id": "devq", "page_id": "dev", "parent_id": "devparent",
                            "arm": spec["name"], "cap": cap, "block": 0, "elapsed_ms": 1.0 + cap,
                            "generated_tokens": cap, "token_ids": [1] * cap, "truncated": True,
                            "actual_visual_count": count, "keep_idx": list(range(count)),
                            "ds_lengths": [count] * 3, "positions_sha256": "b" * 64,
                            "audit": {"native": True, "ds_streams": 3, "positions_preserved": True,
                                      "physical_visual_count": count}, "freeze_sha256": "frozen"})
    return protocol, manifest, frozen, records, timings


class ScoreTests(unittest.TestCase):
    def test_official_threshold_alias_case_and_whitespace(self):
        cases = [("cats", ["cat"], .75), ("dog", ["cat"], 0), ("ac", ["ab"], .5),
                 ("axy", ["abc"], 0), ("DOG", ["cat", "dog"], 1),
                 (" a   b ", ["a b"], 1), ("", [""], 1),
                 ("  ac  ", ["ab"], 5 / 6)]
        for prediction, answers, expected in cases:
            with self.subTest(prediction=prediction, answers=answers):
                self.assertAlmostEqual(official_anls(prediction, answers), expected)
        with self.assertRaises(ValueError):
            official_anls("x", [])

    def test_literal_strict_is_not_normalized_exact(self):
        self.assertEqual(strict_match("Ada", ["Ada", "A"]), 1)
        self.assertEqual(strict_match("ada", ["Ada"]), 0)
        self.assertEqual(strict_match(" Ada ", ["Ada"]), 0)
        self.assertEqual(strict_match("12.0", ["12"]), 0)


class BootstrapTests(unittest.TestCase):
    def test_shared_parent_resampling_preserves_constant_difference(self):
        a = [.1, .3, .7, .6]
        b = [x + .1 for x in a]
        result = paired_bootstrap(a, b, ["p", "p", "q", "r"], draws=500, seed=4)
        np.testing.assert_allclose(result["ci95"], [.1, .1])
        self.assertEqual(result["n_parents"], 3)

    def test_parent_denominators_keep_all_questions(self):
        result = paired_bootstrap([0] * 4, [1, 1, 0, 0], ["p", "p", "q", "r"],
                                  draws=500, seed=8, batch=31)
        index = np.random.default_rng(8).integers(0, 3, size=(500, 3))
        expected = np.array([2., 0., 0.])[index].sum(1) / np.array([2., 1., 1.])[index].sum(1)
        np.testing.assert_allclose(result["ci95"], np.percentile(expected, [2.5, 97.5]))
        self.assertEqual(result["delta"], .5)

    def test_invalid_paired_arrays_rejected(self):
        with self.assertRaises(ValueError):
            paired_bootstrap([1], [1, 0], ["p"])
        with self.assertRaises(ValueError):
            paired_bootstrap([float("nan")], [1], ["p"])


class EvaluationAuditTests(unittest.TestCase):
    def test_complete_roster_and_all_seeds_reported(self):
        protocol, manifest, frozen, records, timings = fixture()
        result = build_report(records, timings, manifest, frozen, protocol, draws=50)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["n_questions"], 2)
        self.assertEqual(result["arms"]["full"]["all"]["n_expected"], 2)
        self.assertEqual(result["seed_summary"]["ordered_pair"]["anls"]["mean"], 1)
        self.assertIn("ordered_pair__seed_11_vs_ordered_rank__seed_11", result["paired_comparisons"])
        self.assertIn("not official", markdown_report(result))

    def test_missing_answer_cannot_be_complete_or_shrink_denominator(self):
        protocol, manifest, frozen, records, timings = fixture()
        lost = records.pop()
        result = build_report(records, timings, manifest, frozen, protocol, draws=50)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["qa_status"], "incomplete")
        row = result["arms"][lost["arm"]]["all"]
        self.assertEqual(row["n_expected"], 2)
        self.assertEqual(row["n_observed"], 1)
        self.assertIsNone(row["anls"])
        self.assertEqual(row["observed_anls"], 1)
        self.assertEqual(row["full_denominator_anls_bounds"], [.5, 1])

    def test_missing_timing_is_not_complete_experiment(self):
        protocol, manifest, frozen, records, timings = fixture()
        result = build_report(records, timings[:-1], manifest, frozen, protocol, draws=50)
        self.assertEqual(result["qa_status"], "complete")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["timing"]["status"], "incomplete")

    def test_truncation_stays_in_denominator(self):
        protocol, manifest, frozen, records, timings = fixture()
        records[0]["truncated"] = True
        records[0]["token_ids"] = [1] * 128
        result = build_report(records, timings, manifest, frozen, protocol, draws=50)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["arms"]["full"]["all"]["truncated"], 1)
        self.assertEqual(result["arms"]["full"]["all"]["n_observed"], 2)

    def test_duplicate_unknown_corrupt_score_and_missing_native_audit_rejected(self):
        protocol, manifest, frozen, records, timings = fixture()
        corruptions = [lambda rows: rows.append(copy.deepcopy(rows[0])),
                       lambda rows: rows[0].update(item_id="unknown"),
                       lambda rows: rows[0].update(anls=1),
                       lambda rows: rows[0].update(keep_idx=[0, 0, 1, 2]),
                       lambda rows: rows[0].update(audit={}),
                       lambda rows: rows[0].update(positions_sha256="missing")]
        for corrupt in corruptions:
            copied = copy.deepcopy(records)
            corrupt(copied)
            with self.assertRaises(ValueError):
                build_report(copied, timings, manifest, frozen, protocol, draws=10)

    def test_indices_are_original_unique_ordered_and_not_boolean(self):
        validate_keep_indices([1, 3], 2, 4)
        for indices in ([3, 1], [1, 1], [1, 4], [True, 2], [1]):
            with self.assertRaises(ValueError):
                validate_keep_indices(indices, 2, 4)


class PersistenceTests(unittest.TestCase):
    def test_resume_keeps_exact_indices_and_rejects_overwrite_or_new_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "predictions.sqlite3"
            record = {"item_id": "q", "arm": "b1", "keep_idx": [1, 7], "truncated": True}
            store = ResultStore(database, "freeze1")
            store.save(record)
            store.close()
            resumed = ResultStore(database, "freeze1")
            self.assertEqual(resumed.get("q", "b1"), record)
            resumed.save(record)  # exact repeat is idempotent
            with self.assertRaises(ValueError):
                resumed.save({**record, "keep_idx": [0, 7]})
            resumed.close()
            with self.assertRaises(ValueError):
                ResultStore(database, "freeze2")
            predictions, timings, events = load_database(database, "freeze1")
            self.assertEqual(predictions, [record])

    def test_file_changed_after_freeze_is_rejected(self):
        from .evaluate import HERE
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "evaluation").mkdir()
            artifact = directory / "checkpoint"
            artifact.write_bytes(b"old")
            frozen = {"files": {str(artifact): sha256(artifact)},
                      "local_sources": sorted(str(p.resolve()) for p in HERE.glob("*.py"))}
            frozen["freeze_sha256"] = json_hash(frozen)
            (directory / "evaluation/freeze.json").write_text(json.dumps(frozen))
            verify_freeze(directory)
            artifact.write_bytes(b"new")
            with self.assertRaises(ValueError):
                verify_freeze(directory)


if __name__ == "__main__":
    unittest.main()
