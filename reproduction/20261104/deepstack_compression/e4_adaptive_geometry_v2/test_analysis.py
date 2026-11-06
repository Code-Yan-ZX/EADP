import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import analyze
import analysis_gate


class Tests(unittest.TestCase):
    def test_unreviewed_or_preloaded_source_cannot_execute(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp); marker = d / "marker"; file = d / "gate.py"
            file.write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n")
            with self.assertRaises(ValueError): analyze.load("_test_geometry_gate", file, "0" * 64)
            with patch.dict(sys.modules, {"_test_geometry_preloaded": object()}):
                with self.assertRaises(ValueError): analyze.load("_test_geometry_preloaded", file, hashlib.sha256(file.read_bytes()).hexdigest())
            self.assertFalse(marker.exists())

    def test_missing_freeze_rejects_before_scores(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(analysis_gate, "D", Path(temp)):
            with self.assertRaises(FileNotFoundError): analysis_gate.admit()

    def test_unfinished_or_wrong_score_reviewer_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp); reviewer = d / "review.py"; reviewer.touch()
            s = {"cwd": str(d), "command": ["python", str(reviewer)], "worker_pid": 7}
            t = {"reason": "completed", "worker_returncode": 0, "remaining_process_group": [], "worker_pid": 7}
            def put(name, value): (d / name).write_text(json.dumps(value))
            put("started.json", s); put("terminal.json", t); analysis_gate.terminal(d, reviewer)
            put("terminal.json", {**t, "reason": "running"})
            with self.assertRaises(ValueError): analysis_gate.terminal(d, reviewer)
            put("terminal.json", t); put("started.json", {**s, "command": ["python", "unrelated.py"]})
            with self.assertRaises(ValueError): analysis_gate.terminal(d, reviewer)


if __name__ == "__main__": unittest.main()
