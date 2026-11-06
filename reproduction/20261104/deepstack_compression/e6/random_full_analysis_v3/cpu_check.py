"""Synthetic tests and source inventory, without reading any panel or truth."""
import hashlib
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

D = Path(__file__).resolve().parent
B = D.parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def sources():
    frozen = json.loads((B / "e1/e1full001_score/frozen.json").read_text())
    paths = {D / x for x in ("PLAN.md", "plan_freeze001.json", "paired.py", "score.py", "admission.py", "test_paired.py", "test_score.py", "test_main_synthetic.py", "cpu_check.py", "freeze_score.py")}
    paths.update(Path(p) for p in frozen["code_sha256"])
    paths.update({B / "e1_analysis/completion.py", B / "e1/e1full001_score/frozen.json", B / "e6/pixelprune_t045_dev384/plan001.json", B / "e6/random_full_exec_v2/runtime_plan001.json", B / "e6/random_full_exec_v2/verification_plan001.json", B / "e6/random_smoke/selection_manifest001.jsonl"})
    return {str(p.resolve()): sha(p) for p in sorted(paths)}


def main():
    before = sources(); output = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromNames(["test_paired", "test_score"])
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    integration = subprocess.run([sys.executable, str(D / "test_main_synthetic.py"), "-v"], capture_output=True, text=True, timeout=120)
    after = sources()
    proof = {"status": "passed" if result.wasSuccessful() and integration.returncode == 0 and before == after else "failed", "tests": result.testsRun, "test_output": output.getvalue(), "integration_returncode": integration.returncode, "integration_stdout": integration.stdout, "integration_stderr": integration.stderr, "source_sha256_before": before, "source_sha256_after": after, "GT_opened": False, "full_panel_predictions_opened": False}
    with (D / "cpu_proof001.json").open("x") as f:
        json.dump(proof, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
    if proof["status"] != "passed": raise SystemExit(1)


if __name__ == "__main__": main()
