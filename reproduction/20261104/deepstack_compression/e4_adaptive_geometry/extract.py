"""Extract fixed geometry from accepted complete E4 records; never load scores/GT."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from features import descriptors

D = Path(__file__).resolve().parent
B = D.parent


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())
def lines(p):
    with Path(p).open() as f:
        for x in f:
            if x.strip(): yield json.loads(x)


def completed(mon):
    s, t = read(mon / "started.json"), read(mon / "terminal.json")
    if t.get("reason") != "completed" or t.get("worker_returncode") != 0 or t.get("remaining_process_group") != [] or s.get("worker_pid") != t.get("worker_pid"):
        raise ValueError("prior supervisor not complete")


def main():
    a = argparse.ArgumentParser(); a.add_argument("--out", type=Path, required=True); args = a.parse_args()
    if args.out.resolve() != D / "features001" or args.out.exists(): raise ValueError("unexpected output")
    frozen = read(D / "plan_freeze002.json"); path = B / "e4_panel_v2/panel_003/records.jsonl"; inp = B / "e1/input_preparation_002/inputs.jsonl"
    formal_path = B / "e4_panel_v2/panel_003_verification.json"; review_path = B / "e1_independent_review/e4panel003_review.json"
    # Only hash files needed here, never the score artifact listed in the plan.
    for item in (D / "PLAN.md", path, inp, formal_path, review_path, B / "e1/e1full001_completion.json"):
        if sha(item) != frozen["source_sha256"][str(item)]: raise ValueError("accepted source changed")
    formal, review, e1 = read(formal_path), read(review_path), read(B / "e1/e1full001_completion.json")
    completed(B / "e4_panel_v2/monitor_verify_panel003"); completed(B / "e1_independent_review/monitor_e4panel003")
    if (formal.get("status") != "passed" or formal.get("n_questions") != 384 or formal.get("all_arms_validated") != 3072 or formal.get("GT_opened") is not False or
            formal.get("files_sha256", {}).get(str(path)) != sha(path) or review.get("status") != "passed" or review.get("n_verified") != 384 or
            review.get("arms_verified") != 3072 or review.get("formal_proof_sha256") != sha(formal_path) or review.get("GT_opened") is not False or
            e1.get("status") != "passed" or e1["files"]["inputs"]["sha256"] != sha(inp)):
        raise ValueError("prior acceptance mismatch")
    inputs = {(r["task"], str(r["qid"])): r for r in lines(inp)}
    if len(inputs) != 384: raise ValueError("E1 inputs incomplete")
    rows = {}
    for r in lines(path):
        if r.get("budget") != 128 or r.get("ds_on") is not True or r.get("forced") is not False: continue
        k = (r["task"], str(r["qid"]))
        if k in rows or k not in inputs: raise ValueError("ambiguous K128 record")
        x = inputs[k]
        if any(r.get(key) != x.get(key) for key in ("task", "group_id", "image_sha256", "message_sha256")): raise ValueError("input mismatch")
        values = descriptors(r["full_position_ids"], r["image_start"], r["spatial_ids"])
        rows[k] = {"task": k[0], "qid": k[1], "group_id": r["group_id"], "image_sha256": r["image_sha256"], **values}
    if set(rows) != set(inputs) or any(sum(k[0] == t for k in rows) != 128 for t in ("DocVQA_VAL", "TextVQA_VAL", "AI2D_TEST")):
        raise ValueError("feature denominator")
    args.out.mkdir(exist_ok=False)
    with (args.out / "features.jsonl").open("x") as f:
        for k in sorted(rows): f.write(json.dumps(rows[k], sort_keys=True) + "\n")
        f.flush(); os.fsync(f.fileno())
    proof = {"status": "completed", "n": 384, "plan_sha256": sha(D / "PLAN.md"), "plan_freeze_sha256": sha(D / "plan_freeze002.json"), "extractor_sha256": sha(__file__), "feature_code_sha256": sha(D / "features.py"), "features_sha256": sha(args.out / "features.jsonl"), "e4_records_sha256": sha(path), "GT_opened": False, "score_values_opened": False, "prediction_records_parsed_for_selection_and_positions": True}
    with (args.out / "extraction.json").open("x") as f: json.dump(proof, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())


if __name__ == "__main__": main()
