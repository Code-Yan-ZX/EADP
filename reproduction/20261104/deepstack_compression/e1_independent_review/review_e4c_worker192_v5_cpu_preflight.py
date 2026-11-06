"""Independent static/synthetic review authorizing only the no-GPU preflight."""
import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parent; B = R.parent; D = B / "e4_c_multibudget_worker192_v5"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for x in iter(lambda: f.read(1 << 20), b""): h.update(x)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())


def main():
    a = argparse.ArgumentParser(); a.add_argument("--out", type=Path, required=True); args = a.parse_args()
    out = R / "e4c_worker192_v5_cpu_preflight_review.json"
    if args.out.resolve() != out or out.exists() or (D / "run_preflight001").exists(): raise ValueError("unexpected prior output")
    p, c = read(D / "cpu_plan001.json"), read(D / "cpu_proof001.json")
    s, t = read(D / "monitor_cpu001/started.json"), read(D / "monitor_cpu001/terminal.json")
    if (p.get("status") != "prepared_cpu_preflight" or p.get("inputs") != 222 or p.get("jobs") != 7920 or p.get("GPU_authorized") is not False or
            p.get("GT_opened") is not False or p.get("target_predictions_opened") is not False or c.get("status") != "passed" or c.get("test_returncode") != 0 or
            c.get("plan_sha256") != sha(D / "cpu_plan001.json") or c.get("source_sha256") != p.get("source_sha256") or "Ran 18 tests" not in c.get("test_output", "") or
            t.get("reason") != "completed" or t.get("worker_returncode") != 0 or t.get("remaining_process_group") != [] or s.get("worker_pid") != t.get("worker_pid") or
            [(Path(s["cwd"]) / x).resolve() for x in s["command"] if Path(x).name == "freeze_cpu.py"] != [(D / "freeze_cpu.py").resolve()]):
        raise ValueError("CPU proof identity")
    sources = p["source_sha256"]
    if len(sources) < 240 or any(sha(path) != expected for path, expected in sources.items()): raise ValueError("source changed")
    proposal = B / "e4_c_multibudget_expansion192/proposal002.json"; review = R / "e4c_expansion192_proposal_review.json"; q = read(proposal); pr = read(review)
    if (q.get("GPU_authorized") is not False or q["counts"]["jobs"] != 7920 or q["counts"]["fixed32_reference_arms"] != 888 or
            pr.get("status") != "passed" or pr.get("independent_review") is not True or pr.get("proposal_sha256") != sha(proposal) or pr.get("GPU_authorized") is not False):
        raise ValueError("proposal evidence")
    for filename in ("launch.py", "worker.py"):
        tree = ast.parse((D / filename).read_text()); text = ast.unparse(tree)
        if sha(D / "cpu_gate.py") not in text or sha(D / "gpu_gate.py") not in text or "sha256" not in text: raise ValueError("gate source not pinned in " + filename)
    if (D / "gpu_plan001.json").exists(): raise ValueError("GPU plan unexpectedly exists at CPU preflight review")
    gate_text = (D / "gpu_gate.py").read_text()
    if "gpu_plan001.json" not in gate_text or "fixed_reference_gate.py" not in gate_text or "active_observer" not in gate_text: raise ValueError("future GPU evidence gate missing")
    worker = (D / "worker.py").read_text(); launcher = (D / "launch.py").read_text()
    for fragment in ("cid not in Path('/proc/self/cgroup').read_text()", "require_no_gpu=True", "fixed32_new_references_opened':False"):
        if fragment not in worker: raise ValueError("CPU worker contract missing")
    if "[str(worker),'--out',str(out),'--preflight-only'],False,name)" not in launcher or "run_bounded(argv,name,780,None" not in launcher: raise ValueError("CPU container contract missing")
    # Independent isolated rerun. It exercises the actual CPU CLI with fake
    # components, GPU denial, all 7,920 fake jobs and 888-reference failures.
    run = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_core", "test_lock", "test_cpu_preflight", "test_gpu_gate"], cwd=D, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True, timeout=120)
    if run.returncode != 0 or "Ran 18 tests" not in run.stderr: raise ValueError("independent synthetic rerun")
    fields = {"plan_sha256": D / "cpu_plan001.json", "cpu_proof_sha256": D / "cpu_proof001.json", "cpu_started_sha256": D / "monitor_cpu001/started.json", "cpu_terminal_sha256": D / "monitor_cpu001/terminal.json", "reviewer_sha256": Path(__file__)}
    result = {"status": "passed", "independent_review": True, "scope": "worker192_cpu_preflight_only", "GT_opened": False, "target_predictions_opened": False, "GPU_authorized": False,
              "errors": [], "source_sha256": sources, "test_returncode": run.returncode, "test_output": run.stdout + run.stderr,
              "limitations": ["authorizes only real no-GPU container preflight", "future GPU source and proof require new version and independent review", "past DEV prediction files hashed as existing source, no target output read"]}
    result.update({k: sha(v) for k, v in fields.items()})
    with out.open("x") as f: json.dump(result, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())


if __name__ == "__main__": main()
