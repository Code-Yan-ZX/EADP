"""Extract fixed geometry from accepted complete E4 records; never load scores/GT."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
B = D.parent
ADMISSION_SHA = "ec47e61d03abd9fb0a4f582df791e08b02bd90d96c902ac8640d8ff2c37bdee4"
FEATURES_SHA = "e3e70ed9779f27ada4ea290b83ff5013b0a9ac775101fdef230b5299406d17ab"


def load_verified(name, path, expected):
    path = Path(path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected or name in sys.modules: raise ValueError("unreviewed or preloaded module")
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    try: spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]; raise
    return module


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
    admission = load_verified("_geometry_v2_admission", D / "admission.py", ADMISSION_SHA); approval = admission.admit()
    descriptors = load_verified("_geometry_v2_features", D / "features.py", FEATURES_SHA).descriptors
    frozen = read(B / "e4_adaptive_geometry/plan_freeze002.json"); path = B / "e4_panel_v2/panel_003/records.jsonl"; inp = B / "e1/input_preparation_002/inputs.jsonl"
    for item in (B / "e4_adaptive_geometry/PLAN.md", path, inp):
        if sha(item) != frozen["source_sha256"][str(item)]: raise ValueError("accepted source changed")
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
    proof = {"status": "completed", "n": 384, "plan_sha256": sha(B / "e4_adaptive_geometry/PLAN.md"), "plan_freeze_sha256": sha(B / "e4_adaptive_geometry/plan_freeze002.json"), "approval": approval, "extractor_sha256": sha(__file__), "feature_code_sha256": sha(D / "features.py"), "features_sha256": sha(args.out / "features.jsonl"), "e4_records_sha256": sha(path), "GT_opened": False, "score_values_opened": False, "prediction_records_parsed_for_selection_and_positions": True}
    with (args.out / "extraction.json").open("x") as f: json.dump(proof, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())


if __name__ == "__main__": main()
