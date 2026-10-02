"""Stage-1 Cross-Stream Pilot — generation + official scoring (protocol §6).

Resumable per-(arm, ds) shards under outputs/stage1_cross_stream_pilot/acc.
Resume is STRICT: a shard's frozen meta (scorer, params, K, scope, lambda,
manifest sha, code commit, bank sha, max_new_tokens) must match the current
run or the driver aborts — never skip-by-id alone.

Per-question scores reuse the amp/m12 rules verbatim (DocVQA ANLS-correct;
OCRBench D-4-corrected official branch structure, actual-count denominator).
Self-check is STRICT: the per-question mean must reproduce the official
headline to float precision (the evaluators return np.mean(hit)*100 unrounded;
OCRBench Final Score is an exact count).

Usage:
  python xsp_accuracy.py --arms X_GATHER,X_MAIN025 --mode both
  python xsp_accuracy.py --arms E_GATHER --smoke 10
  python xsp_accuracy.py --arms X_GATHER --mode score
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

import amp_common as AC
import amp_accuracy as AA
import xsp_common as XC

ACC_DIR = os.path.join(XC.OUT_DIR, "acc")


def shard_path(arm: str, ds: str) -> str:
    d = os.path.join(ACC_DIR, arm)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{ds}.json")


def shard_meta(arm: str, ds: str, args) -> dict:
    a = XC.ARMS[arm]
    scorer = a["scorer"]
    return dict(
        round="stage1_cross_stream_pilot", split="dev", arm=arm,
        scorer=scorer, K=XC.K, scope=a["scope"], lam=a["lam"],
        stage1_params=dict(m=XC.M_NB, eps=XC.EPS, floor=XC.FLOOR,
                           alpha=0.5, beta=2.0),
        manifest_sha256=XC.sha256_file(os.path.join(AC.OUT_DIR,
                                                    "manifest.json")),
        bank=dict(path=XC.bank_path(scorer, ds),
                  sha256=XC.sha256_file(XC.bank_path(scorer, ds))),
        code_commit=XC.git_commit(), base_model=AC.common.BASELINE_MODEL,
        env=XC.env_meta(), max_new_tokens=args.max_new_tokens)
    # note: `--smoke` only limits the item range; it does not change any
    # generation config, so correctly-generated smoke shards are resumable
    # into the full run under the identical meta (protocol §5)


def check_meta(shard: dict, want: dict, arm: str, ds: str) -> None:
    got = shard.get("meta")
    if got is None:
        return                                    # fresh shard
    for k, v in want.items():
        if isinstance(v, dict) and k == "env":
            v = {kk: str(vv) for kk, vv in v.items()}
            g = {kk: str(vv) for kk, vv in got.get(k, {}).items()}
            if g != v:
                raise RuntimeError(f"shard meta mismatch [{arm}/{ds}] env: "
                                   f"{g} != {v}")
        elif got.get(k) != v:
            raise RuntimeError(f"shard meta mismatch [{arm}/{ds}] {k}: "
                               f"{got.get(k)!r} != {v!r}")


def load_shard(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"records": {}}


def save_shard(path, shard):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(shard, f, indent=1)
    os.replace(tmp, path)


def degeneracy(text: str) -> dict:
    return AA.degeneracy(text)


def run_gen(args, eng, manifest):
    for ds in XC.DS_LIST:
        items = manifest["datasets"][ds]["dev"]
        if args.smoke:
            items = items[:args.smoke]
        want = shard_meta(args.arm, ds, args)
        bank = XC.load_bank(want["scorer"], ds)
        XC.verify_bank_meta(bank, want["scorer"], ds)
        dataset = AC.common.build_dataset(ds)
        eng.vlm.set_dump_image(dataset.dump_image)
        cfg = XC.arm_run_cfg(args.arm)
        path = shard_path(args.arm, ds)
        shard = load_shard(path)
        check_meta(shard, want, args.arm, ds)
        shard.setdefault("meta", want)
        t0 = time.time()
        for n, it in enumerate(items):
            key = str(it["idx"])
            if key in shard["records"]:
                continue
            rec = bank["samples"][key]
            assert rec["keep"], f"empty bank record {ds}/{key}"
            row = dataset.data.iloc[it["idx"]]
            msg = AC.common.build_message(eng.vlm, dataset, ds, row)
            timings = {}
            out = AC.run_one(eng, msg, ds, rec, cfg,
                             max_new_tokens=args.max_new_tokens,
                             timings=timings)
            shard["records"][key] = dict(
                prediction=out["text"],
                truncated=len(out["gen_ids"]) >= args.max_new_tokens,
                n_gen=len(out["gen_ids"]),
                image_key=rec.get("image_key"),
                n_vis_kept=out["meta"]["n_vis_kept"],
                ttft_ms=timings.get("ttft_ms"),
                merge_ms=timings.get("merge_ms"),
                vision_ms=timings.get("vision_ms"),
                llm_prefill_ms=timings.get("llm_prefill_ms"),
                layer_calls_ok=out["meta"]["layer_calls_ok"],
                degeneracy=degeneracy(out["text"]))
            if len(shard["records"]) % 10 == 0 \
                    or len(shard["records"]) == len(items):
                save_shard(path, shard)
            if n % 10 == 0:
                el = time.time() - t0
                print(f"[dev/{args.arm} {ds}] {n}/{len(items)} "
                      f"({el / max(1, n + 1):.2f}s/q)", flush=True)
        save_shard(path, shard)
        print(f"[done] dev/{args.arm} {ds}: {len(shard['records'])} records",
              flush=True)


def _strict_check(ds: str, per_q: dict, res: dict, n: int) -> bool:
    """STRICT self-check: per-question mean must reproduce the official
    headline to float precision (evaluators return unrounded
    np.mean(hit)*100; OCRBench Final Score is an exact count)."""
    m = 100.0 * sum(float(v) for v in per_q.values()) / len(per_q)
    if ds == "OCRBench":
        if "Final Score" not in res:
            return False
        h = 100.0 * float(res["Final Score"]) / n
    else:
        h = AA._headline(res)
        if h is None:
            return False
    return abs(m - h) <= 1e-6


def run_score(args, manifest):
    from vlmeval.dataset import build_dataset as vlmeval_build
    for ds in XC.DS_LIST:
        path = shard_path(args.arm, ds)
        shard = load_shard(path)
        done = sorted(int(k) for k in shard["records"])
        if not done:
            continue
        dataset = vlmeval_build(ds)
        data = dataset.data
        sub = data.iloc[done].copy()
        for col in ("image",):
            if col in sub.columns:
                sub = sub.drop(columns=[col])
        sub["prediction"] = [shard["records"][str(i)]["prediction"]
                             for i in done]
        sub["truncated"] = [shard["records"][str(i)]["truncated"] for i in done]
        tsv = path.replace(".json", "_pred.tsv")
        sub.to_csv(tsv, sep="\t", index=False)
        try:
            res = dataset.evaluate(tsv)
        except Exception as e:
            print(f"[score FAIL] dev/{args.arm} {ds}: {e}", flush=True)
            raise
        if hasattr(res, "to_dict"):
            res = res.to_dict()
        per_q = AA._per_q_from_detail_score_shim(path, ds)
        if not per_q:
            raise RuntimeError(f"per-question scores unavailable "
                               f"[{args.arm}/{ds}]")
        if set(per_q) != {str(i) for i in done}:
            raise RuntimeError(f"per-question key set != shard keys "
                               f"[{args.arm}/{ds}]")
        chk = _strict_check(ds, per_q, res, len(done))
        if not chk:
            raise RuntimeError(
                f"STRICT headline reproduction FAILED [{args.arm}/{ds}] "
                f"perq_mean vs official {res}")
        summary = dict(arm=args.arm, split="dev", K=XC.K,
                       scorer=XC.ARMS[args.arm]["scorer"], ds=ds,
                       n=len(done), official={
                           k: (float(v) if isinstance(v, (int, float)) else str(v))
                           for k, v in res.items()},
                       per_question=per_q,
                       perq_reproduces_headline=chk)
        with open(path.replace(".json", "_score.json"), "w") as f:
            json.dump(summary, f, indent=1)
        print(f"[scored] dev/{args.arm} {ds}: n={len(done)} STRICT_OK",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--mode", default="both", choices=["gen", "score", "both"])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--smoke", type=int, default=None,
                    help="limit each ds to the first N items (part of DEV)")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(AC.OUT_DIR, "manifest.json")))
    model = AC.common.load_model(AC.common.BASELINE_MODEL,
                                 max_new_tokens=args.max_new_tokens)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)

    for arm in args.arms.split(","):
        assert arm in XC.ARMS, f"unknown arm {arm}"
        args.arm = arm
        if args.mode in ("gen", "both"):
            run_gen(args, eng, manifest)
        if args.mode in ("score", "both"):
            run_score(args, manifest)


if __name__ == "__main__":
    main()
