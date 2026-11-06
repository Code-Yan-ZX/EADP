"""Independent pre-run review of full 7,920-job formal timing verifier."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

R = Path(__file__).resolve().parent; B = R.parent; A = B / "e4_c_multibudget_worker192_v6_accept"; D = B / "e4_c_multibudget_worker192_v6"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for x in iter(lambda: f.read(1 << 20), b""): h.update(x)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    out = R / "e4c_worker192_v6_accept_prelaunch_review.json"
    if a.out.resolve() != out or out.exists() or (A / "verification001.json").exists(): raise ValueError("unexpected result")
    p = read(A / "verification_plan001.json"); s = read(A / "monitor_freeze001/started.json"); t = read(A / "monitor_freeze001/terminal.json")
    if (p.get("status") != "prepared" or p.get("run_id") != "001" or p.get("GPU_authorized") is not False or
            t.get("reason") != "completed" or t.get("worker_returncode") != 0 or t.get("remaining_process_group") != [] or s.get("worker_pid") != t.get("worker_pid") or
            [(Path(s["cwd"]) / x).resolve() for x in s["command"] if Path(x).name == "freeze.py"] != [(A / "freeze.py").resolve()]): raise ValueError("acceptance freeze")
    sources = p["source_sha256"]
    if any(sha(path) != value for path, value in sources.items()): raise ValueError("acceptance sources changed")
    required = [A / x for x in ("verify.py", "snapshot.py", "test_verify.py", "freeze.py")] + [D / "gpu_gate.py", B / "e4_c_multibudget_expansion192/proposal002.json", B / "e4_c_multibudget_analysis192_independent/window_detail.py", B / "e4_c_multibudget_environment192_v2/plan001.json"]
    if any(str(x) not in sources for x in required): raise ValueError("missing essential source")
    digest = sha(D / "gpu_gate.py")
    for name in ("verify.py", "snapshot.py"):
        tree = ast.parse((A / name).read_text()); text = ast.unparse(tree)
        if digest not in text or "sha256" not in text or "preloaded" not in text: raise ValueError("verifier gate not pinned before import")
    script = (A / "verify.py").read_text()
    for fragment in ("len(events)!=7920*3", "fixed32_measure_keys')!=768", "worker_complete.json", "environment_job_details", "watcher_health", "manual_health_check", "fixed32 reference mismatch"):
        if fragment not in script: raise ValueError("missing result requirement: " + fragment)
    first = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_verify"], cwd=A, capture_output=True, text=True, timeout=120)
    second = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_window_detail"], cwd=B / "e4_c_multibudget_analysis192_independent", capture_output=True, text=True, timeout=120)
    if first.returncode or second.returncode or "Ran 7 tests" not in first.stderr or "Ran 3 tests" not in second.stderr: raise ValueError("independent formal/window tests")
    result = {"status": "passed", "independent_review": True, "plan_sha256": sha(A / "verification_plan001.json"), "reviewer_sha256": sha(__file__), "source_sha256": sources,
              "errors": [], "GT_opened": False, "target_predictions_opened": False, "GPU_authorized": False, "formal_tests": first.stderr, "window_tests": second.stderr,
              "limitations": ["pre-run formal verifier review, not a timing result", "source snapshots and endpoint weight stats cannot rule out transient replacement", "observer second fsync and unobserved residual activity remain unknown"]}
    with out.open("x") as f: json.dump(result, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())


if __name__ == "__main__": main()
