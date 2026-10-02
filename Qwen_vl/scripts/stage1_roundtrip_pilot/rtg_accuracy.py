"""RTG pilot — generation + scoring driver (protocol §4/§6).

New arms R_GATHER / R_MAIN025 / F_MAIN025 / S_MAIN025: bank-based
generation on the DEV-300 manifest, resumable shards, then official
scoring with per-question extraction (imported verbatim from
acu_accuracy; D-4 OCRBench rules; S0 hard-fail semantics).

E arms E_GATHER / E_MAIN025: REUSE the verified cross_stream-round
predictions (they were gate-verified bitwise against round 1) after
checking provenance (config/keys/manifest); they are re-scored through
THIS round's scoring path and must reproduce 81.1574 / 83.2673 macro.

Usage:
  python rtg_accuracy.py --arms R_GATHER,R_MAIN025,F_MAIN025,S_MAIN025 --mode both
  python rtg_accuracy.py --arms E_GATHER,E_MAIN025 --mode reuse
  python rtg_accuracy.py --arms R_MAIN025 --mode score
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import torch

import rtg_common as RC
import amp_common as AC
from acu_common import common as C
from acu_common import run_one as acu_run_one, degeneracy
from acu_accuracy import (_headline100, _per_q_generic, _per_q_ocerbench)

FAIL_LOG = os.path.join(RC.OUT_DIR, "score_failures.jsonl")
XSP_ACC = os.path.join(RC._XSP_OUT, "acc")


def fail(arm, ds, kind, detail):
    entry = dict(arm=arm, ds=ds, kind=kind, detail=str(detail))
    with open(FAIL_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[score FAIL] {arm} {ds}: {kind}: {detail}", flush=True)
    return entry


# ---------------------------------------------------------------------------
def run_gen(args):
    model = AC.common.load_model(AC.common.BASELINE_MODEL,
                                 max_new_tokens=args.max_new_tokens)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)
    manifest = json.load(open(os.path.join(AC.OUT_DIR, "manifest.json")))
    for ds in RC.DS_LIST:
        items = manifest["datasets"][ds]["dev"]
        dataset = AC.common.build_dataset(ds)
        eng.vlm.set_dump_image(dataset.dump_image)
        t0 = time.time()
        for arm in args.arms.split(","):
            cfg = RC.arm_cfg(arm)
            scorer = RC.ARMS[arm]["scorer"]
            bank = RC.load_bank(scorer, ds)
            path = RC.shard_path(arm, ds)
            shard = RC.load_shard(path)
            if "meta" not in shard:
                shard["meta"] = dict(arm=arm, ds=ds, scorer=scorer, K=RC.K,
                                     bank_sha256=RC.sha256_file(
                                         RC.bank_path(scorer, ds)),
                                     base_commit=RC.git_commit(),
                                     max_new_tokens=args.max_new_tokens,
                                     manifest_sha256=RC.sha256_file(
                                         os.path.join(AC.OUT_DIR,
                                                      "manifest.json")))
            for n, it in enumerate(items):
                key = str(it["idx"])
                if key in shard["records"]:
                    continue
                row = dataset.data.iloc[it["idx"]]
                msg = C.build_message(eng.vlm, dataset, ds, row)
                timings = {}
                out = acu_run_one(eng, msg, ds, bank["samples"][key], cfg,
                                  max_new_tokens=args.max_new_tokens,
                                  timings=timings)
                shard["records"][key] = dict(
                    prediction=out["text"],
                    truncated=len(out["gen_ids"]) >= args.max_new_tokens,
                    n_vis_kept=int(out["meta"]["n_vis_kept"]),
                    ttft_ms=timings.get("ttft_ms"),
                    merge_ms=timings.get("merge_ms"),
                    layer_calls_ok=bool(out["meta"]["layer_calls_ok"]),
                    degeneracy=degeneracy(out["text"]))
                if len(shard["records"]) % 20 == 0:
                    RC.save_shard(path, shard)
                if n % 20 == 0:
                    el = time.time() - t0
                    print(f"[{arm} {ds}] {n}/{len(items)} "
                          f"({el/max(1, n+1):.2f}s/q)", flush=True)
            RC.save_shard(path, shard)
            print(f"[done] {arm} {ds}: {len(shard['records'])} records",
                  flush=True)


# ---------------------------------------------------------------------------
def run_reuse(args):
    """Copy verified E-arm predictions into this round's shards with a
    provenance record; scoring happens in run_score (reproduces macro)."""
    for ds in RC.DS_LIST:
        for arm in args.arms.split(","):
            src = os.path.join(XSP_ACC, arm, f"{ds}.json")
            src_score = os.path.join(XSP_ACC, arm, f"{ds}_score.json")
            dst = RC.shard_path(arm, ds)
            srcs = json.load(open(src))
            manifest = json.load(open(os.path.join(AC.OUT_DIR,
                                                   "manifest.json")))
            want = {str(it["idx"]) for it in
                    manifest["datasets"][ds]["dev"]}
            got = set(srcs["records"])
            if got != want:
                fail(arm, ds, "reuse_key_mismatch",
                     f"missing={sorted(want - got)[:5]} "
                     f"extra={sorted(got - want)[:5]}")
                continue
            prov = dict(
                reused_from=os.path.relpath(src, RC.QWEN_ROOT),
                src_score=os.path.relpath(src_score, RC.QWEN_ROOT),
                src_score_sha256=RC.sha256_file(src_score),
                src_meta=srcs.get("meta", {}),
                verification="cross_stream round G4: E arms bitwise vs "
                             "round-1 (commit 60a40c3); this round re-checks "
                             "keys + re-scores through the same path",
                base_commit=RC.git_commit())
            shard = dict(meta=prov, records=srcs["records"])
            RC.save_shard(dst, shard)
            print(f"[reuse] {arm} {ds}: {len(got)} records "
                  f"(provenance recorded)", flush=True)


# ---------------------------------------------------------------------------
def _score_one(arm, ds, eng_needed=False):
    from vlmeval.dataset import build_dataset as vlmeval_build
    path = RC.shard_path(arm, ds)
    shard = RC.load_shard(path)
    done = sorted(int(k) for k in shard["records"])
    if not done:
        return False
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
        fail(arm, ds, "evaluate_exception", e)
        return False
    if hasattr(res, "to_dict"):
        res = res.to_dict()
    if not res:
        fail(arm, ds, "empty_official", f"tsv={tsv}")
        return False
    if ds == "OCRBench":
        per_q = _per_q_ocerbench(shard, ds)
    else:
        per_q = _per_q_generic(tsv, ds, shard)
    chk, diff = None, None
    if per_q:
        m = 100.0 * sum(float(v) for v in per_q.values()) / len(per_q)
        if ds == "OCRBench" and "Final Score" in res:
            h = 100.0 * float(res["Final Score"]) / len(done)
            diff = abs(m - h)
            chk = diff <= 0.05
        else:
            h = _headline100(res, ds)
            if h is not None:
                diff = abs(m - h)
                chk = diff <= 0.05
    if chk is False:
        fail(arm, ds, "perq_headline_mismatch",
             f"perq={m:.6f} headline={h:.6f} diff={diff:.6f}")
        return False
    summary = dict(arm=arm, ds=ds, n=len(done),
                   official={k: (float(v) if isinstance(v, (int, float))
                                 else str(v)) for k, v in res.items()},
                   per_question=per_q, perq_vs_headline_diff=diff,
                   perq_reproduces_headline=chk,
                   base_commit=RC.git_commit())
    with open(path.replace(".json", "_score.json"), "w") as f:
        json.dump(summary, f, indent=1)
    h_show = None
    if ds == "OCRBench" and "Final Score" in res:
        h_show = 100.0 * float(res["Final Score"]) / len(done)
    else:
        h_show = _headline100(res, ds)
    print(f"[scored] {arm} {ds}: n={len(done)} "
          f"diff={None if diff is None else round(diff, 6)} "
          f"headline={h_show}", flush=True)
    return True


def run_score(args):
    ok = True
    for ds in RC.DS_LIST:
        for arm in args.arms.split(","):
            ok &= _score_one(arm, ds)
    if not ok:
        raise SystemExit(f"run_score: failures recorded in {FAIL_LOG}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--mode", default="both",
                    choices=["gen", "score", "both", "reuse"])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()
    if args.mode == "reuse":
        run_reuse(args)
        run_score(args)
        return
    if args.mode in ("gen", "both"):
        run_gen(args)
    if args.mode in ("score", "both"):
        run_score(args)


if __name__ == "__main__":
    main()
