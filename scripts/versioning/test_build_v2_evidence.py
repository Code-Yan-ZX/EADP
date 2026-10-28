"""CPU-only negative provenance gates; no model/evaluator dependencies."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("evidence_exporter", Path(__file__).with_name("build_v2_evidence.py"))
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


class ProvenanceGates(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "answers.jsonl"
        self.raw.write_text('{"question_id": 1, "text": "yes"}\n{"question_id": 2, "text": "no"}\n')
        self.score = self.root / "answers.official.score.json"
        self.doc = {"value": 50.0, "success": True, "n": 2,
                    "prediction_sha256": hashlib.sha256(self.raw.read_bytes()).hexdigest(),
                    "result_file": str(self.raw)}
        self.score.write_text(json.dumps(self.doc))
        self.row = {"model": "fixture", "budget": 128, "method": "EADP"}
        self.cell = {"value": 50.0, "source": str(self.score)}

    def resolve(self, cell=None):
        return EXPORTER.Audit(self.root).resolve(self.row, "FixtureTask", cell or self.cell)

    def test_complete_raw_chain(self):
        result = self.resolve()
        self.assertEqual(result["raw_prediction"]["count"], 2)
        self.assertTrue(result["raw_prediction"]["recorded_sha_checked"])

    def test_supplied_number_cannot_reuse_real_source(self):
        cell = dict(self.cell, value=99.0,
                    source_sha256=hashlib.sha256(self.score.read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, "differs from exact score"):
            self.resolve(cell)

    def test_modified_raw_prediction_is_rejected(self):
        self.raw.write_text(self.raw.read_text().replace("yes", "maybe"))
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.resolve()

    def test_complete_flag_does_not_replace_raw_count(self):
        self.doc["n"] = 3
        self.score.write_text(json.dumps(self.doc))
        with self.assertRaisesRegex(ValueError, "Prediction count"):
            self.resolve()

    def test_score_file_existence_is_insufficient(self):
        self.score.write_text(json.dumps({"value": 50.0, "n": 2}))
        with self.assertRaisesRegex(ValueError, "No explicit raw-prediction link"):
            self.resolve()

    def test_hash_does_not_establish_unrecorded_score(self):
        del self.doc["value"]
        self.score.write_text(json.dumps(self.doc))
        cell = dict(self.cell, source_sha256=hashlib.sha256(self.score.read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, "No exact score value"):
            self.resolve(cell)


if __name__ == "__main__":
    unittest.main()
