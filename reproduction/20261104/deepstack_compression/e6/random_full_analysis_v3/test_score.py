import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import admission
import score


def fake_panel():
    anchors, records, e1, scored = {}, {}, {}, {}
    for task in score.TASKS:
        for i in range(128):
            a = {"task": task, "qid": str(i), "group_id": f"{task}-{i}", "message_sha256": "m", "image_sha256": "i", "processor": {"hash": "p"}}
            k = score.key(a); anchors[k] = a
            e1[k] = {**a, "answer": "old", "eos": True, "generation_scalars": {"top_p": .001}, "eos_ids": [42]}
            scored[k] = {"arms": {"P3-D1": {"raw_answer": "old", "eos": True, "score": .5, "independent_score": .5}}}
            for s in score.SEEDS:
                records[(*k, s)] = {**a, "seed": s, "generation_scalars": {"top_p": .001}, "eos_ids": [42], "selection_mode": "e6_random_fixed_seed", "budget": 256, "ds_on": True, "forced": False, "protocol": "natural", "extra_author_greedy": True}
    return anchors, records, e1, scored


class TestScore(unittest.TestCase):
    def test_complete_pairs_and_mutations(self):
        anchors, records, e1, scores = fake_panel()
        self.assertTrue(score.validate_pairs(anchors, records, e1, scores))
        k = next(iter(records)); records[k]["eos_ids"] = [1]
        with self.assertRaises(ValueError): score.validate_pairs(anchors, records, e1, scores)
        records[k]["eos_ids"] = [42]; del records[k]
        with self.assertRaises(ValueError): score.validate_pairs(anchors, records, e1, scores)

    def test_duplicate_and_wrong_metric(self):
        anchors, records, e1, scores = fake_panel()
        a = next(iter(anchors.values()))
        with self.assertRaises(ValueError): score.unique([a, a])
        scores[next(iter(scores))]["arms"]["P3-D1"]["independent_score"] = .2
        with self.assertRaises(ValueError): score.validate_pairs(anchors, records, e1, scores)

    def test_author_native_xlsx_blank_and_seed_key(self):
        rows = [{"task": t, "qid": "1", "group_id": "g", "seed": s, "answer": "" if s == score.SEEDS[0] else "3"} for t in score.TASKS for s in score.SEEDS]
        v = score.native_predictions(rows)
        self.assertEqual(len(v), 9)
        self.assertEqual(v[(score.TASKS[0], "1", "g", score.SEEDS[0])], "nan")

    def test_xlsx_column_inference_is_isolated_by_seed(self):
        task = score.TASKS[0]
        rows = [{"task": task, "qid": "1", "group_id": "g1", "seed": score.SEEDS[0], "answer": "3"},
                {"task": task, "qid": "2", "group_id": "g2", "seed": score.SEEDS[0], "answer": ""},
                {"task": task, "qid": "1", "group_id": "g1", "seed": score.SEEDS[1], "answer": "word"},
                {"task": task, "qid": "2", "group_id": "g2", "seed": score.SEEDS[1], "answer": ""}]
        v = score.native_predictions(rows)
        self.assertEqual(v[(task, "1", "g1", score.SEEDS[0])], "3.0")
        self.assertEqual(v[(task, "2", "g2", score.SEEDS[0])], "nan")
        self.assertEqual(v[(task, "1", "g1", score.SEEDS[1])], "word")

    def test_admission_absent_fails_before_predictions(self):
        with tempfile.TemporaryDirectory() as d, patch.object(admission, "D", Path(d)):
            with self.assertRaises(FileNotFoundError): admission.require()

    def test_tampered_admission_does_not_execute(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d); marker = path / "marker"
            (path / "admission.py").write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n")
            with patch.object(score, "D", path):
                with self.assertRaises(ValueError): score.load_gate()
            self.assertFalse(marker.exists())

    def test_e1_raw_replacement_and_non_exited_supervisor_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            b = Path(d); r = b / "independent"; r.mkdir()
            def put(path, obj):
                path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj))
            def script(path):
                path.parent.mkdir(parents=True, exist_ok=True); path.write_text("# fake\n")
            verifier = b / "e1_analysis/completion.py"; reviewer = r / "review_full.py"; script(verifier); script(reviewer)
            fm = b / "e1/monitorverifye1full001"; rm = r / "monitor_e1full001"
            for mon, executable, pid, out in ((fm, verifier, 1, b / "e1/e1full001_completion.json"), (rm, reviewer, 2, r / "e1full001_review.json")):
                put(mon / "started.json", {"cwd": str(b), "command": [str(executable), "--out", str(out)], "worker_pid": pid})
                put(mon / "terminal.json", {"reason": "completed", "worker_returncode": 0, "remaining_process_group": [], "worker_pid": pid})
            journal = b / "e1/e1full001/raw_predictions.jsonl"; script(journal)
            proof = b / "e1/e1full001_completion.json"
            put(proof, {"status": "passed", "natural_questions": 384, "natural_records": 1536, "GT_opened": False, "scoring_performed": False,
                        "verifier_sha256": admission.sha(verifier), "files": {"journal": {"path": str(journal), "sha256": admission.sha(journal)}}})
            put(r / "e1full001_review.json", {"status": "passed", "n_verified": 384, "errors": [], "completion_proof_sha256": admission.sha(proof),
                "completion_started_sha256": admission.sha(fm / "started.json"), "completion_terminal_sha256": admission.sha(fm / "terminal.json"), "reviewer_sha256": admission.sha(reviewer)})
            with patch.object(admission, "B", b), patch.object(admission, "R", r):
                self.assertTrue(admission.e1_inference())
                journal.write_text("replaced")
                with self.assertRaises(ValueError): admission.e1_inference()
                script(journal)
                put(rm / "terminal.json", {"reason": "running", "worker_returncode": 0, "remaining_process_group": [], "worker_pid": 2})
                with self.assertRaises(ValueError): admission.e1_inference()


if __name__ == "__main__": unittest.main()
