"""Independent static preflight review, authorizing only a no-GPU container check."""
import ast
import collections
import hashlib
import json
import os
from pathlib import Path

B = Path(__file__).resolve().parent.parent
D = B / "e6/random_full_exec_v2"
S = B / "e6/random_smoke"
OUT = Path(__file__).with_name("random_full_v2_worker_preflight_review.json")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def supervised(directory, script):
    s, t = read(directory / "started.json"), read(directory / "terminal.json")
    assert t["reason"] == "completed" and t["worker_returncode"] == 0 and t["remaining_process_group"] == []
    assert s["worker_pid"] == t["worker_pid"]
    assert sum((Path(s["cwd"]) / a).resolve() == script.resolve() for a in s["command"]) == 1


def main():
    assert not OUT.exists() and not (D / "run_preflight001").exists() and not (D / "run_001").exists()
    plan, verify = read(D / "runtime_plan001.json"), read(D / "verification_plan001.json")
    assert plan["stage"] == "random_full_dev" and plan["questions"] == 384 and plan["generations"] == 1152
    assert plan["seeds"] == [20261104, 20261105, 20261106] and plan["GT_opened"] is False and plan["scoring_performed"] is False
    assert plan["GPU_authorized"] is False and len(plan["source_sha256"]) >= 200
    assert verify["runtime_plan_sha256"] == sha(D / "runtime_plan001.json")
    assert verify["verifier_sha256"] == sha(D / "verify.py") and verify["contract_sha256"] == sha(D / "verify_contract.py")
    assert verify["snapshot_sha256"] == sha(D / "snapshot.py") and verify["questions"] == 384 and verify["generations"] == 1152
    assert verify["GT_opened"] is False and verify["scoring_performed"] is False and Path(verify["report"]) == D / "verification001.json"
    supervised(D / "monitor_cpu001", D / "cpu_check.py")
    supervised(D / "monitor_freeze001", D / "freeze.py")
    log = (D / "monitor_cpu001/stderr.log").read_text()
    assert "Ran 10 tests" in log and "OK" in log
    required = [D / x for x in ("gate.py", "launch.py", "worker.py", "core.py", "references.py", "snapshot.py", "verify.py", "verify_contract.py", "test_gate.py", "cpu_check.py", "freeze.py")]
    assert all(str(x) in plan["source_sha256"] for x in required)
    assert all(sha(path) == digest for path, digest in plan["source_sha256"].items())
    gate_sha = sha(D / "gate.py")
    for filename in ("launch.py", "worker.py"):
        tree = ast.parse((D / filename).read_text())
        fns = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "load_gate"]
        assert len(fns) == 1
        body = fns[0].body
        assert isinstance(body[0], ast.If) and gate_sha in ast.unparse(body[0]) and "sha256" in ast.unparse(body[0])
        assert isinstance(body[1], ast.Return) and "load(" in ast.unparse(body[1])
    validation = read(S / "manifest_validation001.json")
    assert validation["status"] == "passed" and validation["questions"] == 384 and validation["selections"] == 1152
    assert validation["GT_opened"] is False and validation["selection_manifest_sha256"] == sha(S / "selection_manifest001.jsonl")
    records = [json.loads(x) for x in (S / "selection_manifest001.jsonl").read_text().splitlines()]
    counts, identities = collections.Counter(), {}
    for r in records:
        key = (r["task"], str(r["qid"])); seed = r["seed"]
        assert seed in plan["seeds"] and len(r["spatial_ids"]) == 256 and r["spatial_ids"] == sorted(set(r["spatial_ids"]))
        identity = (r["group_id"], r["image_sha256"], r["message_sha256"])
        assert key not in identities or identities[key] == identity
        identities[key] = identity; counts[(key, seed)] += 1
    assert len(records) == 1152 and len(identities) == 384 and len(counts) == 1152 and set(counts.values()) == {1}
    assert collections.Counter(k[0] for k in identities) == {"DocVQA_VAL": 128, "TextVQA_VAL": 128, "AI2D_TEST": 128}
    assert set(counts) == {(k, seed) for k in identities for seed in plan["seeds"]}
    proof = {"status": "passed", "independent_review": True, "scope": "CPU worker preflight only", "runtime_plan_sha256": sha(D / "runtime_plan001.json"), "verification_plan_sha256": sha(D / "verification_plan001.json"), "reviewer_sha256": sha(__file__), "source_count": len(plan["source_sha256"]), "manifest_sha256": sha(S / "selection_manifest001.jsonl"), "questions": 384, "selections": 1152, "GT_opened": False, "target_predictions_opened": False, "GPU_authorized": False, "errors": [], "limitations": ["static and synthetic review permits only real no-GPU preflight", "no high-dimensional full-run tensors or GPU behavior verified", "execution author independently built the underlying frozen plan"]}
    with OUT.open("x") as f:
        json.dump(proof, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())


if __name__ == "__main__":
    main()
