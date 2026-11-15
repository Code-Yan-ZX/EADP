"""CPU tests against original evaluator entry points and durable report data."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from .metrics import METRIC_SOURCES, OCR_CATEGORIES, _pure_functions, score_item, source_metadata
from .report import DEFAULT_ARMS, build_report, paired_bootstrap


def original_vqa_score(dataset, prediction, answers):
    # Execute the *original process_line + hit_calculate* control flow, rather
    # than reconstructing our score_item formula in the test.
    namespace = {"np": np, "istype": lambda value, cls: isinstance(ast.literal_eval(value), cls),
                 "listinstr": lambda needles, haystack: any(needle in haystack for needle in needles)}
    _pure_functions(METRIC_SOURCES["punctuation"], {"process_punctuation"}, namespace)
    _pure_functions(METRIC_SOURCES["vqa_eval"],
                    {"_process_digit_article", "process_answer", "levenshtein_distance", "anls_compute",
                     "process_line", "hit_calculate"}, namespace)
    method = "anls" if dataset.startswith("DocVQA") else "vqa_score"
    result = namespace["process_line"]({"answer": repr(answers), "prediction": prediction}, method=method)
    return float(namespace["hit_calculate"]([result], dataset)[0])


def original_ocr_score(prediction, answers, category):
    captured = {}

    class Frame:
        iloc = [{"prediction": prediction, "answer": repr(answers), "category": category}]

        def __len__(self):
            return 1

    class Logger:
        def info(self, *args):
            pass

    namespace = {"get_logger": lambda _: Logger(), "load": lambda _: Frame(), "tqdm": lambda seq: seq,
                 "get_intermediate_file_path": lambda *args: "unused",
                 "dump": lambda value, _: captured.update(value)}
    _pure_functions(METRIC_SOURCES["ocrbench"], {"OCRBench_eval"}, namespace)
    namespace["OCRBench_eval"]("unused")
    return float(captured["Final Score"])


class OfficialMetricTests(unittest.TestCase):
    def test_docvqa_matches_original_threshold_and_whitespace_denominator(self):
        cases = [("ac", ["ab"]), ("axy", ["abc"]), ("  ac  ", ["ab"]),
                 (" a   b ", ["a b"]), ("DOG", ["cat", "dog"]), ("", [""]),
                 ("STRAẞE", ["straße"]), ("The answer is 12", ["12"])]
        for pred, answers in cases:
            with self.subTest(pred=pred):
                self.assertAlmostEqual(score_item("DocVQA_VAL", pred, answers),
                                       original_vqa_score("DocVQA_VAL", pred, answers))
        self.assertEqual(score_item("DocVQA_VAL", "ac", ["ab"]), .5)
        self.assertEqual(score_item("DocVQA_VAL", "axy", ["abc"]), 0)
        self.assertAlmostEqual(score_item("DocVQA_VAL", "  ac  ", ["ab"]), 5 / 6)

    def test_textvqa_annotator_consensus_is_not_alias_accuracy(self):
        for count, expected in ((0, 0), (1, .3), (2, .6), (3, .9), (4, 1), (10, 1)):
            answers = ["cat"] * count + ["dog"] * (10 - count)
            with self.subTest(count=count):
                actual = score_item("TextVQA_VAL", "cat", answers)
                self.assertAlmostEqual(actual, original_vqa_score("TextVQA_VAL", "cat", answers))
                self.assertAlmostEqual(actual, expected)

    def test_textvqa_original_punctuation_numbers_articles_contractions(self):
        for pred, answer in (("The TWO!", "2"), ("1,000", "1000"), ("dont", "don't"),
                             ("3.14", "3.14"), ("red-blue", "red blue"), ("a\nCAT\t", "cat"),
                             ("it is blue", "blue")):
            answers = [answer] * 10
            with self.subTest(pred=pred):
                self.assertEqual(score_item("TextVQA_VAL", pred, answers),
                                 original_vqa_score("TextVQA_VAL", pred, answers))

    def test_ocrbench_matches_original_for_every_category(self):
        for category in OCR_CATEGORIES:
            for pred, answers in (("The word is HELLO.", ["other", "hello"]),
                                  ("abcdef", ["bcd"]), ("alpha\nbeta", ["alpha beta"]),
                                  ("x + y = z", ["x+y=z"]), ("x", ["X"]), ("", ["word"])):
                with self.subTest(category=category, pred=pred):
                    self.assertEqual(score_item("OCRBench", pred, answers, category),
                                     original_ocr_score(pred, answers, category))
        math = "Handwritten Mathematical Expression Recognition"
        self.assertEqual(score_item("OCRBench", "x + y = z", ["x+y=z"], math), 1)
        self.assertEqual(score_item("OCRBench", "x", ["X"], math), 0)

    def test_wrong_metric_or_malformed_answers_fail_loudly(self):
        for args in (("Other", "x", ["x"]), ("OCRBench_v2", "x", ["x"]),
                     ("TextVQA_VAL", "x", []), ("DocVQA_VAL", None, ["x"]),
                     ("DocVQA_VAL", "x", [3]), ("OCRBench", "x", ["x"], "unknown")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                score_item(*args)

    def test_exact_metric_source_dependencies_are_recorded(self):
        for name, meta in source_metadata().items():
            self.assertEqual(meta["sha256"], hashlib.sha256(METRIC_SOURCES[name].read_bytes()).hexdigest())


def make_fixture(directory):
    arms = DEFAULT_ARMS
    protocol = {"n": 4, "k": 2, "max_new_tokens": 128, "arms": arms, "seeds": [11, 23, 37],
                "timing": {"caps": [1, 16], "blocks": 2,
                           "item_ids_by_dataset": {"DocVQA_VAL": ["q1", "q2"], "TextVQA_VAL": ["q1", "q2"]}}}
    manifest = {"datasets": {}}
    for dataset in ("DocVQA_VAL", "TextVQA_VAL"):
        manifest["datasets"][dataset] = {
            "items": [{"item_id": "q1", "parent_id": "p1", "answers": ["yes"] * 10, "history_exposed": True},
                      {"item_id": "q2", "parent_id": "p2", "answers": ["yes"] * 10, "history_exposed": False}],
            "decontaminated_item_ids": ["q2"], "conservative_item_ids": ["q2"], "novel_item_ids": ["q2"]}
    for name, value in (("manifest.json", manifest), ("protocol.json", protocol),
                        ("freeze.json", {"freeze_sha256": "fixture", "files": {}})):
        (directory / name).write_text(json.dumps(value))
    (directory / "evaluation").mkdir()
    database = sqlite3.connect(directory / "evaluation/predictions.sqlite3")
    database.execute("CREATE TABLE predictions(dataset TEXT,item_id TEXT,arm TEXT,payload TEXT,PRIMARY KEY(dataset,item_id,arm))")
    database.execute("CREATE TABLE timings(dataset TEXT,item_id TEXT,arm TEXT,cap INTEGER,block INTEGER,payload TEXT,PRIMARY KEY(dataset,item_id,arm,cap,block))")
    for dataset, data in manifest["datasets"].items():
        for item in data["items"]:
            for arm in arms:
                count, source = (4 if arm == "full" else 2), (2 if arm == "rres512" else 4)
                text = "wrong" if arm.endswith("seed_23") else "yes"
                native = {"actual_visual_count": count, "source_visual_count": source, "keep_idx": list(range(count)),
                          "ds_lengths": [count] * 3, "positions_sha256": "a" * 64,
                          "audit": {"native": True, "ds_streams": 3, "positions_preserved": True}}
                common = {"dataset": dataset, "item_id": item["item_id"], "parent_id": item["parent_id"],
                          "arm": arm, "freeze_sha256": "fixture", **native}
                row = {**common, "text": text, "token_ids": [1], "truncated": arm.endswith("seed_37"),
                       "max_new_tokens": 128, "score": score_item(dataset, text, item["answers"])}
                database.execute("INSERT INTO predictions VALUES(?,?,?,?)", (dataset, item["item_id"], arm, json.dumps(row)))
                for cap in (1, 16):
                    for block in range(2):
                        timing = {**common, "cap": cap, "block": block, "token_ids": [1] * cap,
                                  "generated_tokens": cap, "elapsed_ms": 100. + cap + block - (10 if "__seed_" in arm else 0)}
                        database.execute("INSERT INTO timings VALUES(?,?,?,?,?,?)", (dataset, item["item_id"], arm, cap, block, json.dumps(timing)))
    database.commit()
    database.close()
    return protocol, manifest


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.protocol, self.manifest = make_fixture(self.directory)

    def tearDown(self):
        self.temp.cleanup()

    def modify_prediction(self, transform):
        with sqlite3.connect(self.directory / "evaluation/predictions.sqlite3") as connection:
            dataset, item_id, arm, payload = connection.execute("SELECT * FROM predictions LIMIT 1").fetchone()
            row = json.loads(payload)
            transform(row)
            connection.execute("UPDATE predictions SET payload=? WHERE dataset=? AND item_id=? AND arm=?",
                               (json.dumps(row), dataset, item_id, arm))

    def test_complete_native_run_writes_all_seeds_and_metrics(self):
        report = build_report(self.directory, draws=100)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["n_observed"], 28)
        self.assertEqual(report["timing"]["n_observed"], 112)
        self.assertEqual(report["datasets"]["DocVQA_VAL"]["metric"], "ANLS")
        self.assertEqual(report["datasets"]["TextVQA_VAL"]["metric"], "VQA consensus")
        summary = report["datasets"]["DocVQA_VAL"]["seed_summary"]["bag_rank"]["all"]
        self.assertEqual(summary["range"], [0, 1])
        self.assertAlmostEqual(summary["mean"], 2 / 3)
        self.assertEqual(len(summary["seeds"]), 3)
        self.assertTrue((self.directory / "evaluation/report.md").is_file())

    def test_missing_output_keeps_full_denominator_and_prevents_completion(self):
        with sqlite3.connect(self.directory / "evaluation/predictions.sqlite3") as connection:
            connection.execute("DELETE FROM predictions WHERE dataset='DocVQA_VAL' AND item_id='q2' AND arm='full'")
        report = build_report(self.directory, draws=20, write=False)
        row = report["datasets"]["DocVQA_VAL"]["arms"]["full"]["all"]
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(row["n_expected"], 2)
        self.assertIsNone(row["score"])
        self.assertEqual(row["observed_score"], 1)
        self.assertEqual(row["full_denominator_score_bounds"], [.5, 1])

    def test_truncations_are_kept_in_primary_denominator(self):
        report = build_report(self.directory, draws=20, write=False)
        row = report["datasets"]["DocVQA_VAL"]["arms"]["bag_rank__seed_37"]["all"]
        self.assertEqual(row["truncated"], 2)
        self.assertEqual(row["score"], 1)
        self.assertEqual(row["n_expected"], 2)

    def test_stored_failure_is_zero_instead_of_dropped(self):
        self.modify_prediction(lambda row: row.update(error="failed generation", score=0))
        report = build_report(self.directory, draws=20, write=False)
        row = report["datasets"]["DocVQA_VAL"]["arms"]["full"]["all"]
        self.assertEqual(row["failed"], 1)
        self.assertEqual(row["score"], .5)
        self.assertEqual(row["n_observed"], 2)

    def test_corrupt_score_budget_or_freeze_are_rejected(self):
        for transform in (lambda row: row.update(score=.1), lambda row: row.update(keep_idx=[1] * 4),
                          lambda row: row.update(ds_lengths=[4, 4]), lambda row: row.update(freeze_sha256="other"),
                          lambda row: row.update(max_new_tokens=129), lambda row: row.update(parent_id="other")):
            with self.subTest(transform=transform):
                before = (self.directory / "evaluation/predictions.sqlite3").read_bytes()
                self.modify_prediction(transform)
                with self.assertRaises(ValueError):
                    build_report(self.directory, draws=10, write=False)
                (self.directory / "evaluation/predictions.sqlite3").write_bytes(before)

    def test_missing_timing_does_not_become_complete(self):
        with sqlite3.connect(self.directory / "evaluation/predictions.sqlite3") as connection:
            connection.execute("DELETE FROM timings WHERE dataset='DocVQA_VAL' AND item_id='q1' AND arm='full' AND cap=1 AND block=0")
        report = build_report(self.directory, draws=20, write=False)
        self.assertEqual(report["qa_status"], "complete")
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(len(report["timing"]["missing_keys"]), 1)

    def test_timing_pairs_bootstrap_pages_instead_of_repeated_blocks(self):
        report = build_report(self.directory, draws=20, write=False)
        timing = report["timing"]["datasets"]["DocVQA_VAL"]
        self.assertEqual(timing["arms"]["full"]["1"]["n_independent_parents"], 2)
        pair = timing["paired"]["bag_rank__seed_11_vs_b1__cap_1"]
        self.assertEqual(pair["n_parents"], 2)
        self.assertEqual(pair["median_delta_ms"], -10)
        self.assertEqual(pair["delta_ci95_ms"], [-10, -10])

    def test_decontaminated_and_history_subsets_are_preserved(self):
        report = build_report(self.directory, draws=20, write=False)
        data = report["datasets"]["DocVQA_VAL"]
        self.assertEqual(data["subgroup_sizes"]["decontaminated"], 1)
        self.assertEqual(data["subgroup_sizes"]["conservative"], 1)
        self.assertEqual(data["subgroup_sizes"]["no_documented_history"], 1)
        self.assertEqual(data["history_exposure_counts"], {"exposed": 1, "not_documented": 1})

    def test_unknown_history_coverage_cannot_be_labeled_unseen(self):
        manifest = copy.deepcopy(self.manifest)
        for data in manifest["datasets"].values():
            data["historical_exposure"] = {"coverage": "unknown", "note": "No identity evidence; flags do not prove no exposure."}
        (self.directory / "manifest.json").write_text(json.dumps(manifest))
        report = build_report(self.directory, draws=20, write=False)
        data = report["datasets"]["DocVQA_VAL"]
        self.assertEqual(data["history_exposure_counts"], {"exposed": 1, "unknown": 1})
        self.assertNotIn("historically_unexposed", data["subgroup_sizes"])

    def test_missing_freeze_cannot_be_a_complete_experiment(self):
        (self.directory / "freeze.json").unlink()
        report = build_report(self.directory, draws=20, write=False)
        self.assertEqual(report["qa_status"], "complete")
        self.assertEqual(report["status"], "incomplete")

    def test_frozen_file_mutation_rejected(self):
        source = self.directory / "source.txt"
        source.write_text("before")
        frozen = {"freeze_sha256": "fixture", "files": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}}
        (self.directory / "freeze.json").write_text(json.dumps(frozen))
        source.write_text("after")
        with self.assertRaises(ValueError):
            build_report(self.directory, draws=10, write=False)


class BootstrapTests(unittest.TestCase):
    def test_parent_resampling_matches_unequal_question_count_reference(self):
        result = paired_bootstrap([0] * 4, [1, 1, 0, 0], ["p", "p", "q", "r"], draws=500, seed=8)
        index = np.random.default_rng(8).integers(0, 3, size=(500, 3))
        samples = np.array([2., 0., 0.])[index].sum(1) / np.array([2., 1., 1.])[index].sum(1)
        np.testing.assert_allclose(result["ci95"], np.percentile(samples, [2.5, 97.5]))
        self.assertEqual(result["delta"], .5)
        self.assertEqual(result["n_parents"], 3)


if __name__ == "__main__":
    unittest.main()
