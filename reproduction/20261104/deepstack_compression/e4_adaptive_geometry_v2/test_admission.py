import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import admission
import extract


class Tests(unittest.TestCase):
    def test_hash_before_execution_and_preloaded_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp); marker = d / "marker"; code = d / "x.py"
            code.write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n")
            with self.assertRaises(ValueError): extract.load_verified("_geometry_test_bad", code, "0" * 64)
            self.assertFalse(marker.exists())
            with patch.dict(sys.modules, {"_geometry_test_preload": object()}):
                with self.assertRaises(ValueError): extract.load_verified("_geometry_test_preload", code, hashlib.sha256(code.read_bytes()).hexdigest())
            self.assertFalse(marker.exists())

    def test_supervisor_command_exit_and_output(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp); script = d / "review.py"; out = d / "proof.json"; script.touch()
            def put(name, obj): (d / name).write_text(json.dumps(obj))
            start = {"cwd": str(d), "command": ["python", str(script), "--out", str(out)], "worker_pid": 42}
            end = {"reason": "completed", "worker_returncode": 0, "remaining_process_group": [], "worker_pid": 42}
            put("started.json", start); put("terminal.json", end); admission.terminal(d, script, out)
            put("terminal.json", {**end, "reason": "running"})
            with self.assertRaises(ValueError): admission.terminal(d, script, out)
            put("terminal.json", end); put("started.json", {**start, "command": ["python", "other.py", "--out", str(out)]})
            with self.assertRaises(ValueError): admission.terminal(d, script, out)

    def test_missing_preflight_stops_before_record_parse(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(admission, "D", Path(temp)):
            with self.assertRaises(FileNotFoundError): admission.admit()


if __name__ == "__main__": unittest.main()
