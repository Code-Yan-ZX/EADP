"""Independent preflight and accepted E1/E4 source gate, before record parsing."""
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent; B = D.parent; R = B / "e1_independent_review"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for x in iter(lambda: f.read(1 << 20), b""): h.update(x)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def terminal(mon, script, out=None):
    s, t = read(mon / "started.json"), read(mon / "terminal.json"); cwd = Path(s["cwd"]); cmd = s["command"]
    if (t.get("reason") != "completed" or t.get("worker_returncode") != 0 or t.get("remaining_process_group") != [] or s.get("worker_pid") != t.get("worker_pid") or
            type(s.get("worker_pid")) is not int or s["worker_pid"] <= 0 or [(cwd / x).resolve() for x in cmd if Path(x).name == script.name] != [script.resolve()]):
        raise ValueError("supervisor identity or exit")
    if out is not None and (cmd.count("--out") != 1 or (cwd / cmd[cmd.index("--out") + 1]).resolve() != out.resolve()): raise ValueError("output path")


def prior():
    ef = B / "e1/e1full001_completion.json"; er = R / "e1full001_review.json"; ev = B / "e1_analysis/completion.py"; es = R / "review_full.py"
    em = B / "e1/monitorverifye1full001"; erm = R / "monitor_e1full001"
    terminal(em, ev, ef); terminal(erm, es, er); ep, eq = read(ef), read(er)
    inp = B / "e1/input_preparation_002/inputs.jsonl"
    if (ep.get("status") != "passed" or ep.get("natural_questions") != 384 or ep.get("GT_opened") is not False or ep.get("scoring_performed") is not False or
            ep.get("verifier_sha256") != sha(ev) or ep["files"]["inputs"].get("path") != str(inp) or ep["files"]["inputs"].get("sha256") != sha(inp) or
            eq.get("status") != "passed" or eq.get("n_verified") != 384 or eq.get("errors") != [] or eq.get("completion_proof_sha256") != sha(ef) or
            eq.get("completion_started_sha256") != sha(em / "started.json") or eq.get("completion_terminal_sha256") != sha(em / "terminal.json") or eq.get("reviewer_sha256") != sha(es)):
        raise ValueError("E1 acceptance")
    ff = B / "e4_panel_v2/panel_003_verification.json"; fr = R / "e4panel003_review.json"; fv = B / "e4_panel_v2/panel_verify.py"; fs = R / "review_e4panel.py"
    fm = B / "e4_panel_v2/monitor_verify_panel003"; frm = R / "monitor_e4panel003"
    terminal(fm, fv, ff); terminal(frm, fs, fr); f, q = read(ff), read(fr); records = B / "e4_panel_v2/panel_003/records.jsonl"
    if (f.get("status") != "passed" or f.get("n_questions") != 384 or f.get("all_arms_validated") != 3072 or f.get("GT_opened") is not False or f.get("scoring_performed") is not False or
            f.get("verifier_sha256") != sha(fv) or f.get("files_sha256", {}).get(str(records)) != sha(records) or q.get("status") != "passed" or q.get("n_verified") != 384 or
            q.get("arms_verified") != 3072 or q.get("GT_opened") is not False or q.get("scoring_performed") is not False or q.get("formal_proof_sha256") != sha(ff) or
            q.get("formal_verifier_started_sha256") != sha(fm / "started.json") or q.get("formal_verifier_terminal_sha256") != sha(fm / "terminal.json") or q.get("reviewer_sha256") != sha(fs)):
        raise ValueError("E4 acceptance")
    return {"e1_formal_sha256": sha(ef), "e1_independent_sha256": sha(er), "inputs_sha256": sha(inp), "e4_formal_sha256": sha(ff), "e4_independent_sha256": sha(fr), "records_sha256": sha(records)}


def admit():
    freeze = read(D / "extraction_freeze001.json"); cpu = read(D / "cpu_proof001.json")
    terminal(D / "monitor_cpu001", D / "cpu_check.py")
    if (freeze.get("status") != "frozen_before_feature_values" or freeze.get("GT_opened") is not False or freeze.get("feature_values_opened") is not False or
            cpu.get("status") != "passed" or cpu.get("source_sha256_before") != cpu.get("source_sha256_after") or
            freeze.get("source_sha256") != cpu.get("source_sha256_after") or any(sha(p) != h for p, h in freeze["source_sha256"].items())):
        raise ValueError("source/CPU freeze")
    pre = R / "e4_adaptive_geometry_v2_prelaunch_review.json"; script = R / "review_e4_adaptive_geometry_v2_prelaunch.py"; monitor = R / "monitor_e4_adaptive_geometry_v2_prelaunch"
    terminal(monitor, script, pre); q = read(pre)
    if (q.get("status") != "passed" or q.get("independent_review") is not True or q.get("errors") != [] or q.get("GT_opened") is not False or q.get("feature_values_opened") is not False or
            q.get("extraction_freeze_sha256") != sha(D / "extraction_freeze001.json") or q.get("cpu_proof_sha256") != sha(D / "cpu_proof001.json") or q.get("reviewer_sha256") != sha(script)):
        raise ValueError("independent preflight")
    return {"sources": prior(), "extraction_freeze_sha256": sha(D / "extraction_freeze001.json"), "prelaunch_sha256": sha(pre)}
