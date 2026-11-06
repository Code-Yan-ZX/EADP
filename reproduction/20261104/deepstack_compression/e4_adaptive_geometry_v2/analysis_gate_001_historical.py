"""Reject partial/unreviewed feature extraction before reading scores or features."""
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent; B = D.parent; R = B / "e1_independent_review"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for x in iter(lambda: f.read(1 << 20), b""): h.update(x)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())


def terminal(mon, script, out=None):
    s, t = read(mon / "started.json"), read(mon / "terminal.json"); cwd = Path(s["cwd"]); cmd = s["command"]
    if (t.get("reason") != "completed" or t.get("worker_returncode") != 0 or t.get("remaining_process_group") != [] or s.get("worker_pid") != t.get("worker_pid") or
            [(cwd / x).resolve() for x in cmd if Path(x).name == script.name] != [script.resolve()]): raise ValueError("supervisor command/exit")
    if out is not None and (cmd.count("--out") != 1 or (cwd / cmd[cmd.index("--out") + 1]).resolve() != out.resolve()): raise ValueError("supervisor output")


def admit():
    route = read(D / "route_freeze001.json"); score = read(D / "analysis_freeze001.json")
    if (route.get("status") != "frozen_before_feature_values" or score.get("status") != "frozen_before_feature_values" or
            route.get("GT_opened") is not False or score.get("GT_opened") is not False or
            any(sha(p) != h for p, h in {**route["source_sha256"], **score["source_sha256"]}.items())): raise ValueError("analysis source changed")
    pre = R / "e4_adaptive_geometry_v2_route_prelaunch_review.json"; ps = R / "review_e4_adaptive_geometry_v2_route_prelaunch.py"; pm = R / "monitor_e4_adaptive_geometry_v2_route_prelaunch"
    terminal(pm, ps, pre); p = read(pre)
    if (p.get("status") != "passed" or p.get("independent_review") is not True or p.get("errors") != [] or p.get("GT_opened") is not False or p.get("feature_values_opened") is not False or
            p.get("analysis_freeze_sha256") != sha(D / "analysis_freeze001.json") or p.get("route_freeze_sha256") != sha(D / "route_freeze001.json") or p.get("reviewer_sha256") != sha(ps)):
        raise ValueError("analysis independent prelaunch")
    feature = D / "features001"; monitor = D / "monitor_extract001"; proof = R / "e4_adaptive_geometry_v2_features_review.json"; rs = R / "review_e4_adaptive_geometry_v2_features.py"; rm = R / "monitor_e4_adaptive_geometry_v2_features"
    terminal(monitor, D / "extract.py", feature); terminal(rm, rs, proof); f, r = read(feature / "extraction.json"), read(proof)
    if (f.get("status") != "completed" or f.get("n") != 384 or f.get("GT_opened") is not False or f.get("score_values_opened") is not False or f.get("extractor_sha256") != sha(D / "extract.py") or
            f.get("feature_code_sha256") != sha(D / "features.py") or f.get("features_sha256") != sha(feature / "features.jsonl") or r.get("status") != "passed" or r.get("independent_review") is not True or
            r.get("questions") != 384 or r.get("GT_opened") is not False or r.get("score_values_opened") is not False or r.get("errors") != [] or r.get("features_sha256") != sha(feature / "features.jsonl") or
            r.get("extraction_sha256") != sha(feature / "extraction.json") or r.get("extraction_started_sha256") != sha(monitor / "started.json") or r.get("extraction_terminal_sha256") != sha(monitor / "terminal.json") or r.get("reviewer_sha256") != sha(rs)):
        raise ValueError("complete feature independent acceptance")
    scores = B / "e4_analysis/score_panel003/scores.jsonl"; sr = R / "e4score003_review.json"; sc = B / "e4_analysis/independent_score_review/verification001.json"
    reviewed = read(sr); independent = read(sc)
    if (reviewed.get("status") != "passed" or reviewed.get("questions_verified") != 384 or reviewed.get("arms_verified") != 3072 or reviewed.get("errors") != [] or
            reviewed.get("score_journal_sha256") != sha(scores) or independent.get("status") != "passed"):
        raise ValueError("E4 scores not accepted")
    expected = read(B / "e4_adaptive_geometry/plan_freeze002.json")["source_sha256"]
    if sha(scores) != expected[str(scores)] or sha(sr) != expected[str(sr)] or sha(sc) != expected[str(sc)]: raise ValueError("E4 score provenance drift")
    return {"feature_sha256": sha(feature / "features.jsonl"), "feature_review_sha256": sha(proof), "scores_sha256": sha(scores), "analysis_freeze_sha256": sha(D / "analysis_freeze001.json"), "prelaunch_sha256": sha(pre)}
