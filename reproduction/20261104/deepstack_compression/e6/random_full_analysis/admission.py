"""Fail closed before opening target predictions, prior scores, or labels."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

D = Path(__file__).resolve().parent
B = D.parents[1]
R = B / "e1_independent_review"
RUN = B / "e6/random_full_exec_v2"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def terminal(mon, script, out=None):
    s, t = read(mon / "started.json"), read(mon / "terminal.json")
    cwd, cmd = Path(s["cwd"]), s["command"]
    if (t.get("reason") != "completed" or t.get("worker_returncode") != 0 or t.get("remaining_process_group") != [] or
            type(s.get("worker_pid")) is not int or s["worker_pid"] <= 0 or s["worker_pid"] != t.get("worker_pid") or
            [(cwd / x).resolve() for x in cmd if Path(x).name == script.name] != [script.resolve()]):
        raise ValueError("supervised process identity or exit")
    if out is not None and (cmd.count("--out") != 1 or (cwd / cmd[cmd.index("--out") + 1]).resolve() != out.resolve()):
        raise ValueError("supervised output identity")


def require():
    plan = read(D / "score_plan001.json")
    if plan.get("status") != "frozen" or plan.get("full_panel_predictions_opened") is not False or plan.get("GT_opened") is not False:
        raise ValueError("unfrozen scoring source")
    if any(sha(path) != digest for path, digest in plan["source_sha256"].items()):
        raise ValueError("scoring source changed")
    frozen = read(B / "e1/e1full001_score/frozen.json")
    if any(sha(p) != h for p, h in frozen["code_sha256"].items()) or any(importlib.metadata.version(k) != v for k, v in frozen["runtime_versions"].items()):
        raise ValueError("author scorer identity/version")
    e1proof = R / "e1full001_score_review.json"
    terminal(B / "e1/monitorscoree1full001", B / "e1_analysis/score.py", B / "e1/e1full001_score")
    terminal(R / "monitor_e1score001", R / "review_score.py")
    e1 = read(e1proof)
    if (e1.get("status") != "passed" or e1.get("n_verified") != 384 or e1.get("arms_verified") != 1536 or e1.get("errors") != [] or
            e1.get("score_sha256") != sha(B / "e1/e1full001_score/scores.jsonl") or e1.get("score_frozen_sha256") != sha(B / "e1/e1full001_score/frozen.json") or
            e1.get("reviewer_sha256") != sha(R / "review_score.py")):
        raise ValueError("E1 score review")
    cm = D / "monitor_cpu001"; pre = R / "random_full_v2_score_prelaunch_review.json"; reviewer = R / "review_random_full_v2_score_prelaunch.py"; pm = R / "monitor_random_full_v2_score_prelaunch"
    terminal(cm, D / "cpu_check.py"); terminal(pm, reviewer, pre)
    cpu, review = read(D / "cpu_proof001.json"), read(pre)
    required = {"status": "passed", "independent_review": True, "scope": "random_full_v2_score_prelaunch", "errors": [], "GT_opened": False, "target_predictions_opened": False,
                "score_plan_sha256": sha(D / "score_plan001.json"), "cpu_proof_sha256": sha(D / "cpu_proof001.json"), "reviewer_sha256": sha(reviewer)}
    if cpu.get("status") != "passed" or any(review.get(k) != v for k, v in required.items()) or cpu.get("source_sha256_after") != plan["source_sha256"]:
        raise ValueError("scoring CPU or independent prelaunch proof")
    formal = RUN / "verification001.json"; post = R / "random_full_v2_result_review.json"; rr = R / "review_random_full_v2_result.py"; rm = R / "monitor_random_full_v2_result"
    terminal(RUN / "monitor_run001", RUN / "launch.py"); terminal(RUN / "monitor_snapshot001", RUN / "snapshot.py")
    terminal(RUN / "monitor_verify001", RUN / "verify.py", formal); terminal(rm, rr, post)
    f, q = read(formal), read(post)
    if (f.get("status") != "passed" or f.get("questions") != 384 or f.get("generations") != 1152 or f.get("errors") != [] or f.get("GT_opened") is not False or
            f.get("scoring_performed") is not False or f.get("verifier_sha256") != sha(RUN / "verify.py") or f.get("runtime_plan_sha256") != sha(RUN / "runtime_plan001.json")):
        raise ValueError("full panel formal acceptance")
    fields = {"formal_sha256": formal, "formal_started_sha256": RUN / "monitor_verify001/started.json", "formal_terminal_sha256": RUN / "monitor_verify001/terminal.json",
              "run_started_sha256": RUN / "monitor_run001/started.json", "run_terminal_sha256": RUN / "monitor_run001/terminal.json", "reviewer_sha256": rr}
    if (q.get("status") != "passed" or q.get("independent_review") is not True or q.get("questions") != 384 or q.get("generations") != 1152 or
            q.get("errors") != [] or q.get("GT_opened") is not False or q.get("scoring_performed") is not False or any(q.get(k) != sha(p) for k, p in fields.items())):
        raise ValueError("full panel independent acceptance")
    files = f.get("files_sha256", {})
    if str(RUN / "run_001/records.jsonl") not in files or any(sha(p) != h for p, h in files.items()):
        raise ValueError("accepted panel files changed")
    if any(files.get(p) != h for p, h in q.get("files_sha256", {}).items()) or str(RUN / "run_001/records.jsonl") not in q.get("files_sha256", {}):
        raise ValueError("independent records hash differs")
    return {"source_sha256": plan["source_sha256"], "plan_sha256": sha(D / "score_plan001.json"), "formal_sha256": sha(formal), "independent_sha256": sha(post), "e1_score_review_sha256": sha(e1proof)}
