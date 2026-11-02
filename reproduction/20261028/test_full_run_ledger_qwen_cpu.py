"""Synthetic ledger gate checks; never invokes Qwen verify() or new raw files."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import build_full_run_ledger as ledger

SPEC = importlib.util.spec_from_file_location("ledger_test_frozen_reader", ledger.HERE / "qwen/independent_audit/audit_qwen.py")
READER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(READER)


class QwenLedgerTests(unittest.TestCase):
    def test_missing_whole_sequence_report_never_reads_raw(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "qwen/independent_audit"
            directory.mkdir(parents=True)
            (directory / "public_fa2_full_run02.terminal_audit.state.json").write_text(json.dumps({"status": "waiting_whole_public_sequence_terminal"}))
            with (mock.patch.object(ledger, "HERE", root),
                  mock.patch.object(ledger, "raw_rows", side_effect=AssertionError("raw read before terminal proof")),
                  self.assertRaises(ledger.PendingRun)):
                ledger.qwen_full_row(lambda *args: 1., "DocVQA_VAL")

    def test_preserved_terminal_failure_is_failed_even_without_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "qwen/independent_audit"
            directory.mkdir(parents=True)
            (directory / "public_fa2_full_run02.terminal_audit.failure.json").write_text("{}")
            with mock.patch.object(ledger, "HERE", root), self.assertRaisesRegex(AssertionError, "Preserved failure"):
                ledger.qwen_terminal_metadata()

    def terminal_early_fixture(self):
        state = dict(status="complete", audit_terminated=True, report_sha256="sha", paper_reproduction_success_claimed=False,
                     full_prediction_audit_passed=True)
        report = dict(audit_terminated=True, CPU_only=True, GPU_or_API_used=False,
                      generation_or_source_modified=False, generation_retry_performed=False,
                      paper_reproduction_success_claimed=False, completion_is_not_paper_reproduction_success=True,
                      full_prediction_audit_passed=True, coverage_complete=True, generation_valid=True,
                      existing_CPU_verifier_exit_code=0, all_native_stages_observed=True, errors=[])
        prefix = "public_fa2_full_run02.terminal_audit"
        meta = {prefix + ".state.json": state, prefix + ".report.json": report,
                prefix + ".model_stat_guards.json": [], prefix + ".claim.json": {}, prefix + ".exec.json": {}}
        return meta, report

    def test_complete_but_invalid_or_incomplete_is_never_accepted(self):
        for field in ("full_prediction_audit_passed", "coverage_complete", "generation_valid", "all_native_stages_observed"):
            meta, report = self.terminal_early_fixture()
            report[field] = False
            with (self.subTest(field=field), mock.patch.object(ledger, "terminal_files", return_value=meta),
                  mock.patch.object(ledger, "sha", return_value="sha"),
                  mock.patch.object(ledger, "raw_rows", side_effect=AssertionError("raw read before validated terminal")),
                  self.assertRaises(AssertionError)):
                ledger.qwen_terminal_metadata()

    def row_fixture(self, task="DocVQA_VAL"):
        directory = ledger.HERE / "qwen/independent_audit"
        run = ledger.HERE / "qwen/public_fa2_full_run02"
        model = "synthetic_model"
        raw = run / (model + "_" + task + ".xlsx")
        canonical = [dict(index="1", question="synthetic one", answer="['none']" if task == "DocVQA_VAL" else "A", A="red", B="blue"),
                     dict(index="2", question="synthetic two", answer="['blue']" if task == "DocVQA_VAL" else "B", A="red", B="blue")]
        answers = [dict(canonical[0], prediction=""), dict(canonical[1], prediction="blue")]
        validity = READER.validate_predictions(canonical, answers, 2)
        points = [dict(index="1", raw_prediction="", scored_prediction="nan", prediction_class="empty_model_answer",
                       prediction_changed_by_native_na_loader=True, author_score=.5, independent_score=.5, literal_prediction_score=0.),
                  dict(index="2", raw_prediction="blue", scored_prediction="blue", prediction_class="nonempty_model_answer",
                       prediction_changed_by_native_na_loader=False, author_score=1., independent_score=1., literal_prediction_score=1.)]
        score = dict(n=2, pointwise_author_reference_matches=2, raw_predictions=dict(path=str(raw), sha256="sha"),
                     per_question=dict(path=str(directory / (run.name + "_" + task + ".per_question.jsonl")), sha256="sha"),
                     prediction_validity=validity, value_percent=75., auxiliary_empty_answers_zero_score_percent=50.,
                     auxiliary_literal_prediction_ANLS_percent=50., native_na_prediction_conversions=1,
                     author_score_artifacts=[], author_score_status="author_artifact_missing; isolated unchanged author helpers and independent CPU metric agree")
        plan = dict(model_name=model, datasets={task: dict(data_path="synthetic.tsv", data_sha256="sha")})
        terminal = dict(raw_dataset_diagnostics={task: dict(raw_path=str(raw), raw_sha256="sha")},
                        public_launcher_exit_code=1, public_launcher_exited_successfully=False,
                        author_score_artifacts_complete=False, native_error_lines=[dict(text="synthetic API unavailable")], native_logs=[])
        verification = dict(datasets={task: score}, audit_empty_policy=dict(path="synthetic_policy", sha256="sha"))
        metadata = directory, run, plan, terminal, directory / "terminal_report", verification, directory / "verification", []
        manifest = dict(n=2, source_sha256="sha", source_stat=dict(path="synthetic.tsv"), canonical_sha256="sha")
        reader = types.SimpleNamespace(read_table=lambda path: answers, validate_predictions=READER.validate_predictions,
                                       scoring_prediction=READER.scoring_prediction, prediction_class=READER.prediction_class)
        return metadata, manifest, reader, canonical, answers, points, score

    def run_synthetic_row(self, task="DocVQA_VAL", mutate=None):
        metadata, manifest, reader, canonical, answers, points, score = self.row_fixture(task)
        if mutate:
            mutate(canonical, answers, points, score)
        spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda module: None))
        with (mock.patch.object(ledger, "qwen_terminal_metadata", return_value=metadata),
              mock.patch.dict(ledger.QWEN_COUNTS, {task: 2}),
              mock.patch.object(ledger, "artifact", side_effect=lambda record, expected_path=None: Path(record["path"])),
              mock.patch.object(ledger, "sha", return_value="sha"),
              mock.patch.object(ledger, "read", return_value=manifest),
              mock.patch.object(ledger, "stat_record"),
              mock.patch.object(ledger, "raw_rows", side_effect=lambda path, *args, **kw: canonical if path.name.endswith(".canonical.jsonl") else points),
              mock.patch.object(ledger.importlib.util, "spec_from_file_location", return_value=spec),
              mock.patch.object(ledger.importlib.util, "module_from_spec", return_value=reader)):
            return ledger.qwen_full_row(lambda *args: 62.8, task)

    def test_valid_empty_keeps_denominator_and_original_no_empty_flag(self):
        row = self.run_synthetic_row()
        self.assertEqual(row["n"], 2)
        self.assertEqual(row["value"], 75.)
        self.assertFalse(row["original_planned_no_empty_gate_passed"])
        self.assertEqual(row["prediction_validity"]["empty_count"], 1)
        self.assertEqual(row["auxiliary_empty_answers_zero_score_percent"], 50.)
        self.assertEqual(row["auxiliary_literal_prediction_ANLS_percent"], 50.)
        self.assertFalse(row["three_wait_nodes"])
        self.assertFalse(row["three_wait_nodes_applicable"])

    def test_native_api_failure_retained_with_explicit_cpu_matching_scope(self):
        row = self.run_synthetic_row("AI2D_TEST")
        self.assertEqual(row["native_public_launcher_exit_code"], 1)
        self.assertFalse(row["native_public_launcher_exited_successfully"])
        self.assertIsNone(row["native_author_value"])
        self.assertFalse(row["API_judge_reproduction_claimed"])
        self.assertIn("can_infer", row["metric"])
        self.assertIn("no API judge", row["metric"])

    def test_failure_placeholder_or_missing_row_rejected(self):
        def placeholder(canonical, answers, points, score):
            answers[0]["prediction"] = "Failed to obtain answer: synthetic runtime failure"
        def missing(canonical, answers, points, score):
            answers.pop()
        for change in (placeholder, missing):
            with self.subTest(change=change.__name__), self.assertRaises(AssertionError):
                self.run_synthetic_row(mutate=change)

    def test_changed_raw_or_pointwise_score_rejected(self):
        def changed_raw(canonical, answers, points, score):
            points[0]["raw_prediction"] = "different"
        def changed_metric(canonical, answers, points, score):
            points[0]["author_score"] = .75
        def changed_mean(canonical, answers, points, score):
            score["value_percent"] = 90.
        for change in (changed_raw, changed_metric, changed_mean):
            with self.subTest(change=change.__name__), self.assertRaises(AssertionError):
                self.run_synthetic_row(mutate=change)

    def test_inconsistent_native_author_score_rejected(self):
        def bad_native(canonical, answers, points, score):
            path = Path(score["raw_predictions"]["path"])
            score["author_score_artifacts"] = [dict(path=str(path.with_name(path.stem + "_acc.csv")), sha256="sha", raw_value=90., value_percent=90.)]
        with self.assertRaises(AssertionError):
            self.run_synthetic_row(mutate=bad_native)


if __name__ == "__main__":
    unittest.main()
    assert "torch" not in sys.modules
