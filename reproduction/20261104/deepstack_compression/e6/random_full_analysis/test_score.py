import tempfile
import unittest
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


if __name__ == "__main__": unittest.main()
