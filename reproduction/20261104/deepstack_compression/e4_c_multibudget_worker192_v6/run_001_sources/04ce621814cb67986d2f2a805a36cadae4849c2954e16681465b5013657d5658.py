"""Independent no-label reconstruction of 192-image selection and job order."""
import collections
import hashlib
import json
import os
from pathlib import Path

B = Path(__file__).resolve().parent.parent
D = B / "e4_c_multibudget_expansion192"
R = Path(__file__).resolve().parent
OUT = R / "e4c_expansion192_proposal_review.json"
TASKS = ("DocVQA_VAL", "TextVQA_VAL", "AI2D_TEST")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for x in iter(lambda: f.read(1 << 20), b""): h.update(x)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def lines(path):
    with Path(path).open() as f: return [json.loads(x) for x in f if x.strip()]
def rank(*parts): return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def supervised(mon, script, out=None):
    s, t = read(mon / "started.json"), read(mon / "terminal.json"); cwd = Path(s["cwd"]); cmd = s["command"]
    assert t["reason"] == "completed" and t["worker_returncode"] == 0 and t["remaining_process_group"] == [] and s["worker_pid"] == t["worker_pid"]
    assert [(cwd / x).resolve() for x in cmd if Path(x).name == script.name] == [script.resolve()]
    if out is not None: assert cmd.count("--out") == 1 and (cwd / cmd[cmd.index("--out") + 1]).resolve() == out.resolve()


def main():
    assert not OUT.exists()
    p = read(D / "proposal002.json"); proof = read(B / "e1/e1full001_completion.json"); erev = read(R / "e1full001_review.json")
    supervised(D / "monitor_cpu002", D / "cpu_check.py"); supervised(D / "monitor_freeze002", D / "freeze_proposal.py")
    supervised(B / "e1/monitorverifye1full001", B / "e1_analysis/completion.py", B / "e1/e1full001_completion.json")
    supervised(R / "monitor_e1full001", R / "review_full.py", R / "e1full001_review.json")
    assert p["status"] == "proposal_frozen" and p["GT_opened"] is False and p["predictions_opened"] is False and p["GPU_authorized"] is False
    assert all(sha(path) == digest for path, digest in p["source_sha256"].items())
    assert read(D / "selection_plan_freeze001.json")["selection_plan_sha256"] == sha(D / "SELECTION_PLAN.md") == "7d0a6ea947e1c117bef59bbb8bf7195503867411e2b059d39a498f6efd87cd2d"
    assert proof["status"] == "passed" and proof["natural_questions"] == 384 and proof["GT_opened"] is False and proof["scoring_performed"] is False
    assert erev["status"] == "passed" and erev["errors"] == [] and erev["completion_proof_sha256"] == sha(B / "e1/e1full001_completion.json")
    ip = B / "e1/input_preparation_002/inputs.jsonl"; pp = B / "data_manifest/panel_manifest.jsonl"
    assert proof["files"]["inputs"]["path"] == str(ip) and proof["files"]["inputs"]["sha256"] == sha(ip)
    assert proof["files"]["panel"]["path"] == str(pp) and proof["files"]["panel"]["sha256"] == sha(pp)
    pilot = read(B / "e4_c_multibudget_expansion/pilot_plan001.json")
    assert p["sources"]["pilot_sha256"] == sha(B / "e4_c_multibudget_expansion/pilot_plan001.json")
    inputs = lines(ip); dev = [r for r in lines(pp) if r.get("split") == "DEV"]
    key = lambda r: (r["task"], str(r["qid"]), r["group_id"])
    lookup = {key(r): r for r in inputs}; assert len(inputs) == len(lookup) == len(dev) == 384 and set(lookup) == {key(r) for r in dev}
    anchors = read(B / "e6/pixelprune_t045_dev384/plan001.json")["anchors"]; assert len(anchors) == 384 and p["sources"]["anchors_sha256"] == sha(B / "e6/pixelprune_t045_dev384/plan001.json")
    amap = {key(r): r for r in anchors}; assert set(amap) == set(lookup)
    excluded = {(r["task"], r["group_id"]) for r in pilot["input_identities"]}; assert len(excluded) == 62
    measured, warm = {}, {}
    for task in TASKS:
        candidates = [r for r in anchors if r["task"] == task and (task, r["group_id"]) not in excluded]
        assert len(candidates) == (108 if task == "AI2D_TEST" else 107)
        candidates.sort(key=lambda r: (rank("EADP-E4C-192-v1", task, r["group_id"], str(r["qid"])), r["group_id"], str(r["qid"])))
        selected = candidates[:64]
        assert [r["group_id"] for r in selected] == p["cohort"][task]["measured"]
        for a in selected: measured[key(a)] = a
        for group in pilot["cohort"][task]["warmup"]:
            matches = [r for r in anchors if r["task"] == task and r["group_id"] == group]; assert len(matches) == 1
            warm[key(matches[0])] = matches[0]
    assert len(measured) == 192 and len(warm) == 30 and set(measured).isdisjoint(warm)
    for rows, expected in ((p["measurement_identities"], measured), (p["warmup_identities"], warm)):
        assert len(rows) == len(expected) and {key(r) for r in rows} == set(expected)
        for row in rows:
            inp, anc = lookup[key(row)], expected[key(row)]
            assert row["split"] == "DEV" and all(row[x] == inp[x] == anc[x] for x in ("task", "group_id", "image_sha256", "message_sha256"))
            assert row["image_path"] == inp["image_path"] and row["processor_reference"] == anc["processor"] and row["image_start_reference"] == anc["image_start"]
    expected_jobs = []
    for phase, subset, reps in (("warmup", warm, 1), ("measure", measured, 5)):
        for rep in range(reps):
            image_keys = sorted(subset, key=lambda k: (rank("EADP-E4C-192-SCHEDULE-v1", "image", phase, rep, k[0], k[2], k[1]), k[0], k[2], k[1]))
            for task, qid, group in image_keys:
                arms = sorted(((k, protocol) for k in (128, 256, 512, 1024) for protocol in ("natural", "fixed32")), key=lambda z: (rank("EADP-E4C-192-SCHEDULE-v1", "arm", phase, rep, task, group, qid, z[0], z[1]), z[0], z[1]))
                for budget, protocol in arms:
                    expected_jobs.append({"phase": phase, "replicate": rep, "task": task, "qid": qid, "group_id": group, "budget": budget, "protocol": protocol, "order": len(expected_jobs)})
    assert expected_jobs == p["jobs"] and len(expected_jobs) == 7920
    assert collections.Counter(r["phase"] for r in expected_jobs) == {"warmup": 240, "measure": 7680}
    assert p["counts"] == {"measurement_images": 192, "warmup_images": 30, "measurement_jobs": 7680, "warmup_jobs": 240, "jobs": 7920, "fixed32_reference_arms": 888}
    result = {"status": "passed", "independent_review": True, "proposal_sha256": sha(D / "proposal002.json"), "reviewer_sha256": sha(__file__), "measurement_images": 192, "warmup_images": 30, "jobs": 7920, "fixed32_reference_arms": 888, "GT_opened": False, "expansion_predictions_opened": False, "GPU_authorized": False, "errors": [], "limitations": ["posthoc DEV, not confirmatory", "processor anchor metadata not independently reconstructed", "complete references, timing worker, analysis and all further acceptance remain future work"]}
    with OUT.open("x") as f: json.dump(result, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())


if __name__ == "__main__": main()
