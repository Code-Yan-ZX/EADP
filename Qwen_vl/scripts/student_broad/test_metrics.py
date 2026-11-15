"""CPU checks against the pinned evaluator control flows and native grouping."""
from __future__ import annotations

import ast
import copy
import string
from types import SimpleNamespace
from typing import Optional
import unittest

import numpy as np
import pandas as pd

from .metrics import (METRIC_SOURCES, _pure_functions, aggregate_metrics,
                      parse_prediction, score_item, source_metadata)


def original_mcq(prediction, answer, options):
    namespace = {"pd": pd, "string": string, "cp": copy,
                 "os": SimpleNamespace(environ={"VERBOSE": 0}),
                 "get_logger": lambda _: None, "build_option_str": lambda _: "",
                 "cn_string": lambda _: False, "build_prompt": lambda *args: ""}
    _pure_functions(METRIC_SOURCES["matching"], {"can_infer_option", "can_infer_text", "can_infer"}, namespace)
    _pure_functions(METRIC_SOURCES["multiple_choice"],
                    {"build_choices", "extract_answer_from_item", "eval_vanilla"}, namespace)
    return namespace["eval_vanilla"](None, {"prediction": prediction, "GT": answer, "question": "q", **options})["hit"]


def original_vqa(dataset, prediction, answers):
    namespace = {"np": np, "Optional": Optional,
                 "istype": lambda value, cls: isinstance(ast.literal_eval(value), cls),
                 "listinstr": lambda needles, haystack: any(needle in haystack for needle in needles)}
    _pure_functions(METRIC_SOURCES["vqa_eval"],
                    {"levenshtein_distance", "anls_compute", "relaxed_correctness", "process_line", "hit_calculate"}, namespace)
    method = "anls" if dataset == "InfoVQA_VAL" else "relaxed_accuracy"
    row = namespace["process_line"]({"answer": repr(answers), "prediction": prediction}, method=method)
    return namespace["hit_calculate"]([row], dataset)[0]


class ItemMetricsTests(unittest.TestCase):
    def test_mcq_original_exact_matching_including_chinese_and_ambiguity(self):
        for options in ({"A": "red", "B": "blue", "C": "green"}, {"A": "红色", "B": "蓝色", "C": "绿色"}):
            for prediction in ("A", "(B)", "Answer: C.", "a", "A or B", "A/B", "答案是B", "答案是 B。",
                               "红色", "blue", "red and blue", "Cannot determine the answer", "",
                               "Sorry, I can't help with images of people yet.", "Failed to obtain answer via API"):
                for answer in options:
                    with self.subTest(options=options, prediction=prediction, answer=answer):
                        self.assertEqual(score_item("AI2D_TEST", prediction, [answer], metadata={"options": options}),
                                         original_mcq(prediction, answer, options))
        for prediction in ("A or B", "A/B", "Cannot determine the answer", ""):
            self.assertIsNone(parse_prediction("MMStar", prediction, {"options": {"A": "red", "B": "blue"}}))

    def test_option_text_fallback_does_not_mutate_manifest(self):
        metadata = {"options": {"A": "Red", "B": "Blue"}}
        before = copy.deepcopy(metadata)
        self.assertEqual(score_item("RealWorldQA", "Blue", ["B"], metadata=metadata), 1.)
        self.assertEqual(metadata, before)

    def test_yes_no_original_parser_unknown_is_wrong_for_both_labels(self):
        namespace = {}
        _pure_functions(METRIC_SOURCES["punctuation"], {"process_punctuation"}, namespace)
        _pure_functions(METRIC_SOURCES["yorn"], {"YOrN_Extraction"}, namespace)
        for prediction in ("Yes", "YES!", "No.", "There is no dog", "yes, no", "unknown", "yes/no", "不是", "yesterday", ""):
            extracted = namespace["YOrN_Extraction"](prediction)
            for answer in ("Yes", "No"):
                self.assertEqual(score_item("POPE", prediction, [answer]), float(extracted == answer))
        self.assertEqual(score_item("MME", "Unknown", ["No"]), 0.)

    def test_chartqa_official_numeric_percent_zero_and_text_rules(self):
        for prediction, answers in (("105", ["100"]), ("105.01", ["100"]), ("95", ["100"]),
                                    ("0.5", ["50%"]), ("0", ["0"]), ("0.0", ["0"]),
                                    ("-105", ["-100"]), ("1,000", ["1000"]), ("  RED ", ["red"]),
                                    ("The answer is 100", ["100"]), ("b", ["a", "B"]), ("nan", ["100"])):
            self.assertEqual(score_item("ChartQA_TEST", prediction, answers), original_vqa("ChartQA_TEST", prediction, answers))
        self.assertEqual(score_item("ChartQA_TEST", "105", ["100"]), 1.)
        self.assertEqual(score_item("ChartQA_TEST", "0.0", ["0"]), 0.)

    def test_infovqa_exact_original_anls_threshold_and_denominator(self):
        for prediction, answers in (("ac", ["ab"]), ("axy", ["abc"]), ("  ac  ", ["ab"]),
                                    ("", [""]), ("DOG", ["cat", "dog"]), (" a   b ", ["a b"])):
            self.assertAlmostEqual(score_item("InfoVQA_VAL", prediction, answers), original_vqa("InfoVQA_VAL", prediction, answers))

    def test_malformed_inputs_fail_before_silent_wrong_scoring(self):
        for args, kwargs in ((("Other", "A", ["A"]), {}), (("MMStar", "A", ["A"]), {}),
                             (("MME", "Yes", ["yes"]), {}), (("InfoVQA_VAL", "x", []), {}),
                             (("POPE", None, ["Yes"]), {}),
                             (("MMStar", "C", ["not a label"]), {"metadata": {"options": {"A": "red", "B": "blue"}}})):
            with self.assertRaises(ValueError):
                score_item(*args, **kwargs)
        self.assertEqual(len(source_metadata()), 8)

    def test_missing_annotated_option_remains_wrong_in_original_denominator(self):
        options = {"B": "blue", "C": "green", "D": "black"}
        self.assertEqual(score_item("MMStar", "A", ["A"], metadata={"options": options}), 0.)
        self.assertEqual(original_mcq("A", "A", options), 0.)


def categorical_item(item_id, answer="Yes", **metadata):
    return {"item_id": str(item_id), "answers": [answer], **metadata}


def records_for(items, wrong=()):
    return {item["item_id"]: {"text": "No" if item["item_id"] in wrong else item["answers"][0]}
            for item in items}


class NativeAggregationTests(unittest.TestCase):
    def test_mmbench_circular_is_all_rollings_not_row_mean(self):
        options = {"A": "red", "B": "blue"}
        items = [categorical_item(index, "A", source_index=index, options=options)
                 for index in (7, 1000007, 9, 1000009)]
        records = {item["item_id"]: {"text": "A"} for item in items}
        records["1000007"] = {"text": "A or B"}
        result = aggregate_metrics("MMBench_DEV_EN_V11", items, records)
        self.assertEqual(result["row_score"], .75)
        self.assertEqual(result["primary_score"], .5)
        self.assertEqual(result["unparsed"], 1)
        self.assertEqual(result["n_circular_groups"], 2)
        self.assertEqual(result["official"][0]["Overall"], 50.)

    def test_missing_row_is_incomplete_and_no_circular_success_claim(self):
        items = [categorical_item(index, "A", source_index=index, options={"A": "red", "B": "blue"})
                 for index in (7, 1000007)]
        result = aggregate_metrics("MMBench_DEV_EN_V11", items, {"7": {"text": "A"}})
        self.assertIsNone(result["primary_score"])
        self.assertIsNone(result["row_score"])
        self.assertEqual(result["n_expected"], 2)

    def test_circular_orphan_is_rejected(self):
        item = categorical_item("1000007", "A", source_index=1000007, options={"A": "red", "B": "blue"})
        with self.assertRaises(ValueError):
            aggregate_metrics("MMBench_DEV_EN_V11", [item], {"1000007": {"text": "A"}})

    def test_mme_original_points_and_question_pairing(self):
        categories = ["OCR", "artwork", "celebrity", "color", "count", "existence", "landmark", "position", "posters", "scene",
                      "code_reasoning", "commonsense_reasoning", "numerical_calculation", "text_translation"]
        items = [categorical_item(f"{category}_{i}", category=category, mme_image_group=f"{category}/image")
                 for category in categories for i in range(2)]
        good = aggregate_metrics("MME", items, records_for(items))
        self.assertEqual(good["primary_score"], 2800.)
        self.assertEqual(good["primary_unit"], "points")
        self.assertEqual(good["official"][0]["perception"], 2000.)
        bad = aggregate_metrics("MME", items, records_for(items, wrong=("OCR_0",)))
        self.assertEqual(bad["primary_score"], 2650.)
        self.assertEqual(bad["official"][0]["OCR"], 50.)
        with self.assertRaises(ValueError):
            aggregate_metrics("MME", items[:-1], records_for(items[:-1]))

    def test_hallusion_original_a_f_q_group_metrics(self):
        items = [categorical_item(f"{figure}_{question}", source_index=f"a_b_c_0_{figure}_{question}", category="VS", **{"l2-category": "illusion"})
                 for figure in range(2) for question in range(2)]
        result = aggregate_metrics("HallusionBench", items, records_for(items, wrong=("1_1",)))
        overall = result["official"][0]
        self.assertEqual((overall["aAcc"], overall["fAcc"], overall["qAcc"]), (75., 50., 50.))
        self.assertEqual(result["primary_score"], .75)

    def test_pope_original_f1_category_explosion_and_unknown_handling(self):
        items = [categorical_item("a", category="random,popular"),
                 categorical_item("b", "No", category="random"), categorical_item("c", "No", category="random")]
        result = aggregate_metrics("POPE", items, {"a": {"text": "Yes"}, "b": {"text": "Yes"}, "c": {"text": "Unknown"}})
        self.assertAlmostEqual(result["row_score"], 1 / 3)
        self.assertAlmostEqual(result["primary_score"], .8)
        self.assertEqual(result["official"][0]["acc"], 50.)
        self.assertEqual(result["unparsed"], 1)

    def test_failure_is_wrong_duplicate_and_corrupt_score_are_rejected(self):
        items = [categorical_item("a", category="random")]
        failed = aggregate_metrics("POPE", items, {"a": {"text": "Yes", "error": "generation failed", "score": 0}})
        self.assertEqual(failed["row_score"], 0.)
        self.assertEqual(failed["failed"], 1)
        with self.assertRaises(ValueError):
            aggregate_metrics("POPE", items, {"a": {"text": "Yes", "score": 0}})
        with self.assertRaises(ValueError):
            aggregate_metrics("POPE", items, [{"item_id": "a", "text": "Yes"}] * 2)


if __name__ == "__main__":
    unittest.main()
