"""Anchor-Merge Pilot — generation + official scoring driver (protocol §6/§7).

Resumable per-(split, arm, ds) shards.  Scoring reuses the official VLMEvalKit
evaluators; per-question scores come from m12_analyze._per_q_from_detail
(DocVQA ANLS-correct, OCRBench recomputed over the ACTUAL evaluated count).

Usage:
  python amp_accuracy.py --split dev --arms BASE,U025,U050,U100,S025 --mode both
  python amp_accuracy.py --split dev --arms U025 --mode gen
  python amp_accuracy.py --split dev --arms U025 --mode score
  python amp_accuracy.py --split dev --arms BASE,U025 --smoke 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch

import amp_common as AC

ACC_DIR = os.path.join(AC.OUT_DIR, "acc")


def shard_path(split: str, arm: str, ds: str, K: int = 256) -> str:
    d = os.path.join(ACC_DIR, split, arm, f"K{K}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{ds}.json")


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
    t = text.strip()
    toks = t.split()
    rep = 0
    if len(toks) >= 8:
        grams = [" ".join(toks[i:i + 4]) for i in range(len(toks) - 3)]
        rep = max((grams.count(g) for g in set(grams)), default=0)
    return dict(empty=len(t) == 0, n_chars=len(t), repeat4=rep)


def run_gen(args, eng, manifest):
    smoke = args.smoke
    for ds in AC.DS_LIST:
        items = manifest["datasets"][ds][args.split]
        if smoke:
            items = items[:smoke]
        bank = AC.load_bank(args.split, ds, args.selector, args.K)
        dataset = AC.common.build_dataset(ds)
        eng.vlm.set_dump_image(dataset.dump_image)
        cfg = AC.arm_cfg(args.arm, args.winner_lam, args.winner_kind)
        cfg["selector"], cfg["K"] = args.selector, args.K
        t0 = time.time()
        path = shard_path(args.split, args.arm, ds, args.K)
        shard = load_shard(path)
        for n, it in enumerate(items):
            key = str(it["idx"])
            if key in shard["records"]:
                continue
            row = dataset.data.iloc[it["idx"]]
            msg = AC.common.build_message(eng.vlm, dataset, ds, row)
            timings = {}
            out = AC.run_one(eng, msg, ds, bank[key], cfg,
                             max_new_tokens=args.max_new_tokens,
                             timings=timings)
            rec = dict(prediction=out["text"],
                       truncated=len(out["gen_ids"]) >= args.max_new_tokens,
                       n_vis_kept=out["meta"]["n_vis_kept"],
                       ttft_ms=timings.get("ttft_ms"),
                       merge_ms=timings.get("merge_ms"),
                       vision_ms=timings.get("vision_ms"),
                       llm_prefill_ms=timings.get("llm_prefill_ms"),
                       layer_calls_ok=out["meta"]["layer_calls_ok"],
                       degeneracy=degeneracy(out["text"]))
            shard["records"][key] = rec
            shard.setdefault("meta", dict(
                split=args.split, arm=args.arm, ds=ds, K=args.K,
                selector=args.selector,
                bank_sha256=AC.bank_sha256(args.split, ds,
                                           args.selector, args.K),
                base_commit=manifest["base_commit"],
                max_new_tokens=args.max_new_tokens))
            if len(shard["records"]) % 10 == 0 or len(shard["records"]) == len(items):
                save_shard(path, shard)
            if n % 10 == 0:
                el = time.time() - t0
                print(f"[{args.split}/{args.arm} {ds}] {n}/{len(items)} "
                      f"({el/max(1, n+1):.2f}s/q)", flush=True)
        save_shard(path, shard)
        print(f"[done] {args.split}/{args.arm} {ds}: "
              f"{len(shard['records'])} records", flush=True)


def _headline(official):
    """First numeric value, unwrapping one level of nested dicts (the
    TextVQA/DocVQA evaluators here return {'Overall': {0: 86.0}})."""
    for v in official.values():
        if isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, (int, float)):
                    return float(vv)
        elif isinstance(v, (int, float)):
            return float(v)
    return None


def run_score(args, manifest):
    from vlmeval.dataset import build_dataset as vlmeval_build

    for ds in AC.DS_LIST:
        path = shard_path(args.split, args.arm, ds, args.K)
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
            print(f"[score FAIL] {args.split}/{args.arm} {ds}: {e}", flush=True)
            continue
        if hasattr(res, "to_dict"):
            res = res.to_dict()
        # per-question scores: verbatim rule from m12_analyze._per_q_from_detail
        # (DocVQA ANLS-correct; TextVQA mean-of-matches; OCRBench recomputed
        # over the ACTUAL evaluated count), path shape adapted
        per_q = _per_q_from_detail_score_shim(path, ds)
        # self-check: per-question mean must reproduce the headline
        chk = None
        if per_q:
            m = 100.0 * sum(float(v) for v in per_q.values()) / len(per_q)
            if ds == "OCRBench" and "Final Score" in res:
                chk = abs(m - 100.0 * float(res["Final Score"]) / len(done)) < 0.5
            else:
                h = _headline(res)
                chk = None if h is None else abs(m - h) < 1.0
        summary = dict(arm=args.arm, split=args.split, K=args.K,
                       selector=args.selector, ds=ds,
                       n=len(done), official={
                           k: (float(v) if isinstance(v, (int, float)) else str(v))
                           for k, v in res.items()},
                       per_question=per_q,
                       perq_reproduces_headline=chk)
        with open(path.replace(".json", "_score.json"), "w") as f:
            json.dump(summary, f, indent=1)
        print(f"[scored] {args.split}/{args.arm} {ds}: n={len(done)} "
              f"headline={res} perq_check={chk}", flush=True)


def _per_q_from_detail_score_shim(score_path, ds):
    """Same as m12_analyze._per_q_from_detail but tolerating a missing
    _score.json (reads the shard json directly)."""
    import ast as _ast
    import numpy as np
    import pandas as pd
    tsv = score_path.replace(".json", "_pred.tsv")
    shard_p = score_path
    if ds == "OCRBench":
        shard = json.load(open(shard_p))
        from vlmeval.dataset import build_dataset as _bd
        dso = _bd(ds)
        per_q = {}
        for k, rec in shard["records"].items():
            row = dso.data.iloc[int(k)]
            # verbatim structure of the official OCRBench_eval (vlmeval
            # dataset/utils/ocrbench.py): the math branch does NOT lowercase;
            # the other branches lowercase BOTH sides; strip + '\n'->' '
            # applied to both prediction and answer before containment
            predict = str(rec["prediction"]).strip()
            answers = _ast.literal_eval(row["answer"])
            hit = 0.0
            if row["category"] == "Handwritten Mathematical Expression Recognition":
                p2 = predict.replace("\n", " ").replace(" ", "")
                if any(a.strip().replace("\n", " ").replace(" ", "") in p2
                       for a in answers):
                    hit = 1.0
            else:
                pr = predict.lower().strip().replace("\n", " ")
                if any(a.lower().strip().replace("\n", " ") in pr
                       for a in answers):
                    hit = 1.0
            per_q[str(k)] = hit
        return per_q
    for det in (tsv.replace(".tsv", "_results.xlsx"),
                tsv.replace(".tsv", "_results.tsv")):
        if not os.path.exists(det):
            continue
        d = pd.read_excel(det) if det.endswith("xlsx") \
            else pd.read_csv(det, sep="\t")
        if "eval_match" not in d.columns:
            continue
        shard = json.load(open(shard_p))
        positions = sorted(int(k) for k in shard["records"])
        if len(positions) != len(d):
            return None
        per_q = {}
        for pos, (_, r) in zip(positions, d.iterrows()):
            m = r["eval_match"]
            if isinstance(m, str):
                m = _ast.literal_eval(m)
            m = list(m)
            if ds == "TextVQA_VAL":
                hit = float(np.mean(m))
            elif ds == "DocVQA_VAL":
                md = float(np.min(m))
                hit = 0.0 if 1 - md < 0.5 else 1 - md
            else:
                hit = float(np.mean(m))
            per_q[str(pos)] = hit
        return per_q
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "confirm"])
    ap.add_argument("--arms", required=True)
    ap.add_argument("--mode", default="both", choices=["gen", "score", "both"])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--smoke", type=int, default=None,
                    help="limit each ds to N items and tag shards with smoke")
    ap.add_argument("--winner-lam", type=float, default=0.0)
    ap.add_argument("--winner-kind", default="uniform")
    ap.add_argument("--selector", default="b1", choices=["b1", "b2"])
    ap.add_argument("--K", type=int, default=256)
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(AC.OUT_DIR, "manifest.json")))
    model = AC.common.load_model(AC.common.BASELINE_MODEL,
                                 max_new_tokens=args.max_new_tokens)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)

    for arm in args.arms.split(","):
        args.arm = arm
        if args.mode in ("gen", "both"):
            run_gen(args, eng, manifest)
        if args.mode in ("score", "both"):
            run_score(args, manifest)


if __name__ == "__main__":
    main()
