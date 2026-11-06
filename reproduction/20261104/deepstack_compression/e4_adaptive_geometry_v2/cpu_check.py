import hashlib
import io
import json
import os
import unittest
from pathlib import Path

D = Path(__file__).resolve().parent; B = D.parent


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources():
    files = [D / x for x in ("admission.py", "extract.py", "features.py", "test_features.py", "test_admission.py", "cpu_check.py", "freeze.py")]
    files += [B / "e4_adaptive_geometry/PLAN.md", B / "e4_adaptive_geometry/plan_freeze002.json"]
    return {str(f.resolve()): sha(f) for f in files}


def main():
    before = sources(); output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(["test_features", "test_admission"]))
    after = sources(); proof = {"status": "passed" if result.wasSuccessful() and before == after else "failed", "tests": result.testsRun, "test_output": output.getvalue(),
                                "source_sha256_before": before, "source_sha256_after": after, "GT_opened": False, "feature_values_opened": False, "score_values_opened": False}
    with (D / "cpu_proof001.json").open("x") as f: json.dump(proof, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
    if proof["status"] != "passed": raise SystemExit(1)


if __name__ == "__main__": main()
