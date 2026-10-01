"""Stage-1 grounding discovery — generation + official scoring driver.

Resumable per-(split, arm, ds) shards under outputs/stage1_grounding/acc.
Selection is ONLINE (engine.generate, selector = the arm's variant);
scoring reuses the amp_accuracy official-evaluator machinery verbatim
(DocVQA ANLS-correct, OCRBench recomputed over the ACTUAL evaluated count).

Usage:
  python s1g_accuracy.py --split dev --arms B_G1 --lam 0.5 --smoke 10
  python s1g_accuracy.py --split dev --arms A_G1,A_G2,A_G3,B_G1,B_G2,B_G3,C_G1 --lam 0.5 --mode both
  python s1g_accuracy.py --split dev --arms B_G1 --mode score
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

import s1g_common as SC
from amp_accuracy import (degeneracy as _deg, run_score as amp_run_score,
                          shard_path as amp_shard_path, _per_q_from_detail_score_shim)  # noqa: F401


def run_gen(args, eng, manifest):
    SC.ensure_selectors(args.lam)
    smoke = args.smoke
    for ds in SC.DS_LIST:
        items = manifest["datasets"][ds][args.split]
        if smoke:
            items = items[:smoke]
        dataset = SC.common.build_dataset(ds)
        eng.vlm.set_dump_image(dataset.dump_image)
        path = SC.shard_path(args.split, args.arm, ds, args.tag)
        shard = SC.load_shard(path)
        diag_path = path.replace(".json", "_diag.jsonl")
        t0 = time.time()
        for n, it in enumerate(items):
            key = str(it["idx"])
            if key in shard["records"]:
                continue
            row = dataset.data.iloc[it["idx"]]
            msg = SC.common.build_message(eng.vlm, dataset, ds, row)
            timings = {}
            out, diag = SC.run_one(eng, msg, ds, args.arm,
                                   max_new_tokens=args.max_new_tokens,
                                   timings=timings, want_diag=args.diag)
            rec = dict(prediction=out["text"],
                       truncated=len(out["gen_ids"]) >= args.max_new_tokens,
                       n_vis_kept=out["meta"]["n_vis_kept"],
                       ttft_ms=timings.get("ttft_ms"),
                       selector_ms=timings.get("selector_ms"),
                       vision_ms=timings.get("vision_ms"),
                       llm_prefill_ms=timings.get("llm_prefill_ms"),
                       layer_calls_ok=out["meta"]["layer_calls_ok"],
                       degeneracy=SC.degeneracy(out["text"]))
            shard["records"][key] = rec
            shard.setdefault("meta", dict(
                split=args.split, arm=args.arm, ds=ds, K=SC.K,
                selector=SC.arm_selector_name(args.arm), lam=args.lam,
                base_commit=manifest["base_commit"],
                max_new_tokens=args.max_new_tokens))
            if args.diag and diag is not None and diag.get("last"):
                tokens = SC.instruction_tokens(eng, msg, ds)
                d = dict(idx=key, ds=ds, arm=args.arm, tokens=tokens,
                         **diag["last"])
                with open(diag_path, "a") as f:
                    f.write(json.dumps(d) + "\n")
            if len(shard["records"]) % 10 == 0 or \
                    len(shard["records"]) == len(items):
                SC.save_shard(path, shard)
            if n % 10 == 0:
                el = time.time() - t0
                print(f"[{args.split}/{args.arm} {ds}] {n}/{len(items)} "
                      f"({el/max(1, n+1):.2f}s/q)", flush=True)
        SC.save_shard(path, shard)
        print(f"[done] {args.split}/{args.arm} {ds}: "
              f"{len(shard['records'])} records", flush=True)


def run_score(args):
    """amp_accuracy.run_score pointed at THIS round's shard paths.
    (amp run_score loops all datasets internally — call it once.)"""
    import amp_accuracy as AA

    orig = AA.shard_path

    def patched(split, arm, ds, K=256):
        return SC.shard_path(split, arm, ds, args.tag)

    AA.shard_path = patched
    try:
        class _Args:
            pass
        a = _Args()
        a.split, a.arm, a.K = args.split, args.arm, SC.K
        a.selector = SC.arm_selector_name(args.arm)
        amp_run_score(a, None)
    finally:
        AA.shard_path = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "confirm"])
    ap.add_argument("--arms", required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--tag", default="",
                    help="shard dir suffix, e.g. lam05 or smoke")
    ap.add_argument("--mode", default="both", choices=["gen", "score", "both"])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--smoke", type=int, default=None)
    ap.add_argument("--diag", action="store_true",
                    help="append per-sample text-token diagnostics jsonl")
    args = ap.parse_args()

    manifest = SC.load_manifest()
    if args.mode in ("gen", "both"):
        model = SC.common.load_model(SC.common.BASELINE_MODEL,
                                     max_new_tokens=args.max_new_tokens)
        from model.native_qwen3 import NativeEngine
        eng = NativeEngine(model)
        for arm in args.arms.split(","):
            args.arm = arm
            run_gen(args, eng, manifest)
    if args.mode in ("score", "both"):
        for arm in args.arms.split(","):
            args.arm = arm
            run_score(args)


if __name__ == "__main__":
    main()
