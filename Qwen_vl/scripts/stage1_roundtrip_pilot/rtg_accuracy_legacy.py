"""L_R_MAIN025 full-panel generation + scoring (official-legacy scenario).

User-directed round (2026-10-03): the ONLY arm is the full method
R_MAIN025 (RTG + official facility + Completion 0.25) run under the EADP
paper scenario — the official legacy inference path (DeepStack OFF,
1-D positions, fixed-res 1024, greedy), NOT the native-repaired engine.
Arm name is prefixed L_ everywhere; pipeline tag is "official_legacy".

Deltas vs rtg_accuracy_full.py (frozen file, not modified):
  * acu_run_one(..., deepstack=False, pos="1d")  -> engine legacy config
    (flag plumbing verified bit-exact vs archived official predictions by
    e0 gate N5 and s1_dualpath; acu_run_one defaults keep native bitwise).
  * output root  outputs/stage1_roundtrip_pilot/legacy_full/
  * arm/pipeline tags "L_R_MAIN025" / "official_legacy" in meta+summary.
Banks: reuses the frozen full RTG banks (bank_full_rtg_<ds>.json.gz) —
bank building touches only prepare/encode (ViT) + scorer, never prefill/
decode, so bank records are engine-mode independent by construction
(bitwise re-check of sample records: rtg_legacy_checks.py).

Scoring semantics identical to rtg_accuracy_full._score_one_full (S0
hard-fail, strict perq-headline reproduction, OCRBench/POPE exact
branches) — only paths/arm names differ.

Usage: python rtg_accuracy_legacy.py --mode both
       python rtg_accuracy_legacy.py --datasets TextVQA_VAL,ChartQA_TEST
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time

import rtg_common as RC
import amp_common as AC
from acu_common import common as C
from acu_common import run_one as acu_run_one, degeneracy
from acu_accuracy import _headline100, _per_q_generic, _per_q_ocerbench

ARM = "L_R_MAIN025"
PIPELINE = "official_legacy"
LEGACY_FLAGS = dict(deepstack=False, pos="1d")

# Table-4 datasets with frozen RTG banks already on disk
LEGACY_DS_LIST = ["TextVQA_VAL", "ChartQA_TEST", "DocVQA_VAL", "OCRBench",
                  "MMBench_DEV_EN_V11"]

FULL = os.path.join(RC.OUT_DIR, "legacy_full", "acc", ARM)
os.makedirs(FULL, exist_ok=True)
FAIL_LOG = os.path.join(RC.OUT_DIR, "legacy_full", "score_failures.jsonl")


def shard_path(ds: str) -> str:
    return os.path.join(FULL, f"{ds}.json")


def fail(ds, kind, detail):
    entry = dict(arm=ARM, ds=ds, kind=kind, detail=str(detail))
    with open(FAIL_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[score FAIL] {ARM} {ds}: {kind}: {detail}", flush=True)
    return False


def load_shard(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"records": {}}


def save_shard(path: str, shard: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(shard, f, indent=1)
    os.replace(tmp, path)


def run_gen(datasets=None):
    model = AC.common.load_model(AC.common.BASELINE_MODEL,
                                 max_new_tokens=2048)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)
    cfg = RC.arm_cfg("R_MAIN025")
    for ds in (datasets or LEGACY_DS_LIST):
        bank_path = os.path.join(RC.OUT_DIR, "full",
                                 f"bank_full_rtg_{ds}.json.gz")
        with gzip.open(bank_path, "rt") as f:
            bank = json.load(f)
        dataset = AC.common.build_dataset(ds)
        eng.vlm.set_dump_image(dataset.dump_image)
        path = shard_path(ds)
        shard = load_shard(path)
        if "meta" not in shard:
            shard["meta"] = dict(arm=ARM, pipeline=PIPELINE, panel="full",
                                 ds=ds, K=RC.K, legacy=LEGACY_FLAGS,
                                 bank_sha256=RC.sha256_file(bank_path),
                                 bank_reused_from="full/bank_full_rtg",
                                 base_commit=RC.git_commit(),
                                 max_new_tokens=2048)
            save_shard(path, shard)
        t0 = time.time()
        n_rows = len(dataset.data)
        for n, i in enumerate(range(n_rows)):
            key = str(i)
            if key in shard["records"]:
                continue
            row = dataset.data.iloc[i]
            msg = C.build_message(eng.vlm, dataset, ds, row)
            timings = {}
            out = acu_run_one(eng, msg, ds, bank["samples"][key], cfg,
                              max_new_tokens=2048, timings=timings,
                              **LEGACY_FLAGS)
            # legacy engine reports st.n_vis = vmask.sum() and vmask is only
            # built when deepstack=True -> always 0 there; record the real
            # kept count from the compacted keep_idx instead
            n_kept = int(out["state"].keep_idx.numel())
            shard["records"][key] = dict(
                prediction=out["text"],
                truncated=len(out["gen_ids"]) >= 2048,
                n_vis_kept=n_kept,
                ttft_ms=timings.get("ttft_ms"),
                merge_ms=timings.get("merge_ms"),
                layer_calls_ok=bool(out["meta"]["layer_calls_ok"]),
                degeneracy=degeneracy(out["text"]))
            if len(shard["records"]) % 25 == 0:
                save_shard(path, shard)
            if n % 50 == 0:
                el = time.time() - t0
                print(f"[{ARM} full {ds}] {n}/{n_rows} "
                      f"({el/max(1, n+1):.2f}s/q)", flush=True)
        save_shard(path, shard)
        print(f"[done] {ARM} full {ds}: {len(shard['records'])} records",
              flush=True)


def _score_one(ds):
    """Mirror of rtg_accuracy_full._score_one_full — frozen semantics,
    legacy paths/arm name."""
    from vlmeval.dataset import build_dataset as vlmeval_build
    path = shard_path(ds)
    shard = load_shard(path)
    done = sorted(int(k) for k in shard["records"])
    if not done:
        return fail(ds, "no_records", path)
    dataset = vlmeval_build(ds)
    data = dataset.data
    sub = data.iloc[done].copy()
    for col in ("image",):
        if col in sub.columns:
            sub = sub.drop(columns=[col])
    sub["prediction"] = [shard["records"][str(i)]["prediction"] for i in done]
    sub["truncated"] = [shard["records"][str(i)]["truncated"] for i in done]
    tsv = path.replace(".json", "_pred.tsv")
    sub.to_csv(tsv, sep="\t", index=False)
    try:
        res = dataset.evaluate(tsv)
    except Exception as e:
        return fail(ds, "evaluate_exception", e)
    if hasattr(res, "to_dict"):
        res = res.to_dict()
    if not res:
        return fail(ds, "empty_official", f"tsv={tsv}")
    per_q = (_per_q_ocerbench(shard, ds) if ds == "OCRBench"
             else _per_q_generic(tsv, ds, shard))
    chk, diff, m, h = None, None, None, None
    if per_q:
        m = 100.0 * sum(float(v) for v in per_q.values()) / len(per_q)
        if ds == "OCRBench" and "Final Score" in res:
            h = 100.0 * float(res["Final Score"]) / len(done)
            diff = abs(m - h)
            chk = diff <= 0.05
        elif ds == "POPE":
            from statistics import mean as _mean
            cats = [str(dataset.data.iloc[int(k)]["category"]).split(",")
                    for k in done]
            exploded = [float(per_q[str(k)]) for k, cs in zip(done, cats)
                        for _ in cs]
            h = _headline100(res, ds)
            diff = abs(100.0 * float(_mean(exploded)) - h)
            chk = diff <= 1e-6
            m = 100.0 * float(_mean([float(v) for v in per_q.values()]))
        else:
            h = _headline100(res, ds)
            if h is not None:
                diff = abs(m - h)
                chk = diff <= 0.05
    if chk is False:
        return fail(ds, "perq_headline_mismatch",
                    f"perq={m:.6f} headline={h:.6f} diff={diff:.6f}")
    summary = dict(arm=ARM, pipeline=PIPELINE, panel="full", ds=ds,
                   n=len(done),
                   official={k: (float(v) if isinstance(v, (int, float))
                                 else str(v)) for k, v in res.items()},
                   per_question=per_q, perq_vs_headline_diff=diff,
                   perq_reproduces_headline=chk,
                   base_commit=RC.git_commit())
    with open(path.replace(".json", "_score.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[scored] {ARM} full {ds}: n={len(done)} diff={diff}", flush=True)
    return True


def run_score(datasets=None):
    ok = True
    for ds in (datasets or LEGACY_DS_LIST):
        ok &= _score_one(ds)
    if not ok:
        raise SystemExit("legacy full-panel scoring failures recorded")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["gen", "score",
                                                       "both"])
    ap.add_argument("--datasets", default=None,
                    help="comma list; default = Table-4 datasets with banks")
    args = ap.parse_args()
    dsets = args.datasets.split(",") if args.datasets else None
    if args.mode in ("gen", "both"):
        run_gen(dsets)
    if args.mode in ("score", "both"):
        run_score(dsets)


if __name__ == "__main__":
    main()
