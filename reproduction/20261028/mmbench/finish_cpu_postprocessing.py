#!/usr/bin/env python3
"""Complete author XLSX conversion with isolated dependencies, retaining first attempt."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
state_path = OUT / "state.json"
state = json.loads(state_path.read_text())
assert state["exit_code"] == 0 and state["n_answers"] == 4377
assert state["status"] == "scoring_or_conversion_failed"
first = OUT / "conversion.json"
first_record = json.loads(first.read_text())
assert first_record["exit_code"] != 0
assert "No module named 'openpyxl'" in (OUT / "conversion.log").read_text()
assert not Path(first_record["output"]).exists(), "Preserve any existing conversion output"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(OUT / "vendor_scoring"))
with (OUT / "conversion_isolated.log").open("w") as log:
    result = subprocess.run(first_record["command"], env=env, stdout=log, stderr=subprocess.STDOUT)
record = dict(first_record, exit_code=result.returncode, dependency_isolation=str(OUT / "vendor_scoring"),
              package_install_report_sha256=sha(OUT / "vendor_scoring_install_report.json"),
              unchanged_author_converter=True)
record_path = OUT / "conversion_isolated.json"
record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
completion = {"status": "complete" if result.returncode == 0 else "failed", "native_generation_exit_code": state["exit_code"],
              "original_state": {"path": str(state_path), "sha256": sha(state_path)},
              "first_conversion_record": {"path": str(first), "sha256": sha(first)},
              "first_conversion_log": {"path": str(OUT / "conversion.log"), "sha256": sha(OUT / "conversion.log")},
              "successful_conversion_record": {"path": str(record_path), "sha256": sha(record_path)},
              "prediction_sha256": state["prediction_sha256"], "native_rerun": False,
              "shared_environment_modified": False, "postprocessor_sha256": sha(__file__)}
(OUT / "postprocessing_completion.json").write_text(json.dumps(completion, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(completion, ensure_ascii=False, indent=2))
raise SystemExit(result.returncode)
