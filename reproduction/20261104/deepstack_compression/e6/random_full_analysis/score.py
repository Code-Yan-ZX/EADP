"""Score only a completely accepted random DEV panel, after fail-closed admission."""
import argparse
import collections
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
B = D.parents[1]
ADMISSION_SHA = "f11b58899d9f5276e657a2f0aefe62eaedc457b1fa075c2b7c24b5bd892ba60b"
SEEDS = (20261104, 20261105, 20261106)
TASKS = ("DocVQA_VAL", "TextVQA_VAL", "AI2D_TEST")


def load_gate():
    p = D / "admission.py"
    if hashlib.sha256(p.read_bytes()).hexdigest() != ADMISSION_SHA or "_random_score_admission" in sys.modules:
        raise ValueError("unreviewed/preloaded admission")
    spec = importlib.util.spec_from_file_location("_random_score_admission", p)
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
    try: spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[spec.name]; raise
    return mod


def key(r):
    return r["task"], str(r["qid"]), r["group_id"]


def unique(rows, seeds=False):
    result = {}
    for r in rows:
        k = key(r) + ((r["seed"],) if seeds else ())
        if k in result: raise ValueError("duplicate question/seed")
        result[k] = r
    return result


def validate_pairs(anchors, records, e1, scores):
    if len(anchors) != 384 or collections.Counter(x[0] for x in anchors) != {x: 128 for x in TASKS}:
        raise ValueError("anchor denominator")
    if set(anchors) != set(e1) or set(anchors) != set(scores) or set(records) != {(*k, s) for k in anchors for s in SEEDS}:
        raise ValueError("paired full panel inventory")
    for k, a in anchors.items():
        e, old = e1[k], scores[k]["arms"]["P3-D1"]
        if e.get("message_sha256") != a["message_sha256"] or e.get("processor") != a["processor"] or e.get("answer") != old["raw_answer"] or e.get("eos") != old["eos"]:
            raise ValueError("reference input/answer mismatch")
        if not math.isfinite(old["score"]) or not math.isfinite(old["independent_score"]) or abs(old["score"] - old["independent_score"]) > 1e-12:
            raise ValueError("accepted E1 score disagreement")
        for seed in SEEDS:
            r = records[(*k, seed)]
            if (r.get("seed") != seed or r.get("image_sha256") != a["image_sha256"] or r.get("message_sha256") != a["message_sha256"] or
                    r.get("processor") != a["processor"] or r.get("generation_scalars") != e["generation_scalars"] or r.get("eos_ids") != e["eos_ids"] or
                    r.get("selection_mode") != "e6_random_fixed_seed" or r.get("budget") != 256 or r.get("ds_on") is not True or
                    r.get("forced") is not False or r.get("protocol") != "natural" or r.get("extra_author_greedy") is not True):
                raise ValueError("random/reference identity or generation settings")
    return True


def native_predictions(rows):
    import pandas as pd
    result = {}
    for task in TASKS:
        sub = [r for r in rows if r["task"] == task]
        stream = io.BytesIO()
        pd.DataFrame({"index": list(range(len(sub))), "prediction": [r["answer"] for r in sub]}).to_excel(stream, index=False, engine="xlsxwriter")
        stream.seek(0); frame = pd.read_excel(stream)
        if len(frame) != len(sub) or list(frame["index"]) != list(range(len(sub))): raise ValueError("native XLSX identity")
        for row, value in zip(sub, frame["prediction"]): result[(*key(row), row["seed"])] = str(value)
    if len(result) != len(rows): raise ValueError("native XLSX duplicate")
    return result


def lines(path):
    with Path(path).open(encoding="utf8") as f: return [json.loads(x) for x in f if x.strip()]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args()
    if args.out.resolve() != D / "score001" or args.out.exists(): raise ValueError("unexpected score destination")
    admission = load_gate(); evidence = admission.require()
    if any(x in sys.modules for x in ("paired", "completion", "metrics", "score")): raise ValueError("preloaded scoring module")
    sys.path.insert(0, str(D)); import paired
    sys.path.insert(0, str(B / "e1_analysis")); import completion, metrics, score as e1score
    if Path(paired.__file__).resolve() != D / "paired.py" or any(Path(m.__file__).resolve() != B / "e1_analysis" / (name + ".py") for name, m in (("completion", completion), ("metrics", metrics), ("score", e1score))):
        raise ValueError("scorer origin")
    anchors = unique(admission.read(B / "e6/pixelprune_t045_dev384/plan001.json")["anchors"])
    records = unique(lines(B / "e6/random_full_exec_v2/run_001/records.jsonl"), seeds=True)
    e1 = unique([r for r in lines(B / "e1/e1full001/raw_predictions.jsonl") if r["arm"] == "P3-D1" and r["forced"] is False])
    scores = unique(lines(B / "e1/e1full001_score/scores.jsonl"))
    validate_pairs(anchors, records, e1, scores)
    proof = admission.read(B / "e1/e1full001_completion.json"); summary = Path(proof["files"]["dataset_summary"]["path"])
    if admission.sha(summary) != proof["files"]["dataset_summary"]["sha256"]: raise ValueError("dataset summary changed")
    meta = admission.read(summary); needed = {(r["task"], str(r["qid"])) for r in anchors.values()}
    if len(needed) != 384 or set(meta["sources"]) != set(TASKS): raise ValueError("DEV truth inventory")
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "frozen.json").open("x") as f:
        json.dump({"scope": "selected DEV qids only", "admission": evidence, "references": meta["sources"]}, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
    truth = e1score.read_truths(meta["sources"], needed); authors = metrics.author_functions(); constants = metrics._author_constants()
    native = native_predictions(list(records.values())); result = []
    for k, anchor in anchors.items():
        arms = {"eadp": scores[k]["arms"]["P3-D1"]}
        for seed in SEEDS:
            r = records[(*k, seed)]
            value = metrics.score(anchor["task"], r["answer"], truth[(anchor["task"], str(anchor["qid"]))], authors, constants, scoring_prediction=native[(*k, seed)])
            if not math.isfinite(value["official"]) or not math.isfinite(value["independent"]) or abs(value["official"] - value["independent"]) > 1e-12: raise ValueError("author/independent metric mismatch")
            arms[str(seed)] = {"answer": r["answer"], "score": value["official"], "independent_score": value["independent"], "eos": r["eos"], "truncated": r["truncated"]}
        result.append({"task": anchor["task"], "qid": str(anchor["qid"]), "group_id": anchor["group_id"], "arms": arms})
    reports = {}
    for task in TASKS:
        subset = [r for r in result if r["task"] == task]
        paired_rows = [{"qid": r["qid"], "group_id": r["group_id"], "scores": {"eadp": r["arms"]["eadp"]["score"], **{s: r["arms"][str(s)]["score"] for s in SEEDS}}} for r in subset]
        reports[task] = paired.analyze(paired_rows)
        reports[task]["terminal"] = {str(s): {"eos": sum(r["arms"][str(s)]["eos"] for r in subset), "truncated": sum(r["arms"][str(s)]["truncated"] for r in subset), "empty": sum(not r["arms"][str(s)]["answer"].strip() for r in subset)} for s in SEEDS}
    if any(admission.sha(p) != h for p, h in evidence["source_sha256"].items()) or any(admission.sha(v["path"]) != v["sha256"] for v in meta["sources"].values()): raise ValueError("score source or truth changed")
    with (args.out / "scores.jsonl").open("x") as f:
        for r in result: f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush(); os.fsync(f.fileno())
    report = {"status": "completed", "questions": 384, "random_generations": 1152, "tasks": reports, "scores_sha256": admission.sha(args.out / "scores.jsonl"), "plan_sha256": evidence["plan_sha256"],
              "limitations": ["posthoc exploratory DEV; intervals uncorrected", "three-seed mean is per question; seed uncertainty is not estimated", "adapter executes discarded author greedy; timings are not comparable", "same K does not establish equal compute", "high-dimensional tensors are archived as fingerprints"]}
    with (args.out / "analysis.json").open("x") as f: json.dump(report, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())


if __name__ == "__main__": main()
