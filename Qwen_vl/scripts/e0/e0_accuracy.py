"""E0 M4/M5-accuracy: all arms x K x DEV datasets (prereg §5.1).

Generation runs the native engine (prereg §3.2); predictions land in
per-(arm,K,dataset) shard files that are resumable.  Scoring reuses the
official VLMEvalKit rule paths (no external LLM judge anywhere).

Usage:
  python e0_accuracy.py --arm b2 --K 256 --ds TextVQA_VAL            # gen+score
  python e0_accuracy.py --arm b2 --K 256 --ds TextVQA_VAL --score-only
  python e0_accuracy.py --arm a1 --ds OCRBench                       # K ignored
  python e0_accuracy.py --arm rres --K 128 --ds ChartQA_TEST         # 352 & 384
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

DISC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "discovery")
sys.path.insert(0, DISC_DIR)
import common  # noqa: E402

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "e0")
ACC_DIR = os.path.join(OUT_DIR, "acc")

OCR_PANEL = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench", "ChartQA_TEST"]
GENERAL_PANEL = ["MMBench_DEV_EN_V11", "MMStar", "RealWorldQA", "POPE"]
ALL_DATASETS = OCR_PANEL + GENERAL_PANEL

# R-res: K -> square fixed resolution(s), side lengths multiple of 32
RRES = {256: [512], 64: [256], 128: [352, 384]}


def arm_spec(arm: str, K: int):
    """Returns (selector, deepstack, pos, K_eff)."""
    if arm == "b0":
        return "identity", True, "mrope3d", 1024
    if arm in ("b1", "b2", "divprune", "cdpruner", "hiprune",
               "visionzip", "fastv", "pdrop", "sparsevlm",
               "sparsevlm_norecycle", "pace"):
        return arm, True, "mrope3d", K
    if arm == "a1":      # B2, legacy: deepstack off + 1-D positions
        return "b2", False, "1d", 256
    if arm == "a2":      # B2, deepstack on + 1-D positions
        return "b2", True, "1d", 256
    if arm == "rres":    # resolution control, no pruning
        return "identity", True, "mrope3d", 1024
    raise KeyError(arm)


def shard_path(arm_id: str, K: int, ds: str) -> str:
    d = os.path.join(ACC_DIR, arm_id, f"K{K}")
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


def dev_items(plan, ds):
    return plan["datasets"][ds]["dev_rows"]


def build_messages(model, ds, rows):
    dataset = common.build_dataset(ds)
    if hasattr(model, "set_dump_image"):
        model.set_dump_image(dataset.dump_image)
    out = []
    for idx in rows:
        row = dataset.data.iloc[idx]
        out.append((int(idx), row, common.build_message(model, dataset, ds, row)))
    return out, dataset


def set_rres_resolution(side):
    import vlmeval.vlm.qwen3_vl.model_fixed_res as mfr
    mfr.QWEN3_FIXED_RESOLUTION = side


def run_gen(eng, args, plan):
    arm, K = args.arm, args.K
    ds_list = args.ds.split(",") if args.ds else ALL_DATASETS
    if arm in ("a1", "a2"):
        ds_list = [d for d in ds_list if d in ("TextVQA_VAL", "DocVQA_VAL",
                                               "OCRBench")]
    selector, deepstack, pos, K_eff = arm_spec(arm, K)
    if arm == "rres":
        sides = RRES[K]
    else:
        sides = [None]

    for ds in ds_list:
        rows = dev_items(plan, ds)
        msg_items, _ = build_messages(eng.vlm, ds, rows)
        for side in sides:
            arm_id = f"{arm}{side}" if side else arm
            path = shard_path(arm_id, K_eff, ds)
            shard = load_shard(path)
            if side:
                set_rres_resolution(side)
            t0 = time.time()
            for n, (idx, row, msg) in enumerate(msg_items):
                if str(idx) in shard["records"]:
                    continue
                timings = {}
                out = eng.generate(msg, ds, K=K_eff, selector=selector,
                                   deepstack=deepstack, pos=pos,
                                   max_new_tokens=args.max_new_tokens,
                                   timings=timings)
                rec = dict(prediction=out["text"],
                           truncated=len(out["gen_ids"]) >= args.max_new_tokens,
                           n_vis_kept=out["meta"]["n_vis_kept"],
                           ttft_ms=timings.get("ttft_ms"),
                           layer_calls_ok=out["meta"]["layer_calls_ok"])
                shard["records"][str(idx)] = rec
                if (len(shard["records"]) % 10 == 0
                        or len(shard["records"]) == len(rows)):
                    save_shard(path, shard)
                if n % 20 == 0:
                    el = time.time() - t0
                    print(f"[{arm_id} K={K_eff} {ds}] {n}/{len(rows)} "
                          f"({el/max(1, n+1):.2f}s/q)", flush=True)
            save_shard(path, shard)
            print(f"[done] {arm_id} K={K_eff} {ds}: "
                  f"{len(shard['records'])} records", flush=True)
    if arm == "rres":
        set_rres_resolution(1024)


def run_score(args, plan):
    """S0 hardening (2026-10-02, after the 09-30 BLOCKED incident):
    an empty official dict or an evaluate() exception is a hard failure,
    never a silently-written/partial _score.json.  Failures are appended to
    outputs/e0/score_failures.jsonl and the process exits non-zero."""
    from vlmeval.dataset import build_dataset as vlmeval_build
    import pandas as pd

    fail_log = os.path.join(OUT_DIR, "score_failures.jsonl")
    failures = []

    def record_failure(arm_id, K, ds, kind, detail):
        entry = dict(arm=arm_id, K=K, ds=ds, kind=kind, detail=str(detail))
        failures.append(entry)
        with open(fail_log, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[score FAIL] {arm_id} K={K} {ds}: {kind}: {detail}",
              flush=True)

    arm, K = args.arm, args.K
    ds_list = args.ds.split(",") if args.ds else ALL_DATASETS
    if arm in ("a1", "a2"):
        ds_list = [d for d in ds_list if d in ("TextVQA_VAL", "DocVQA_VAL",
                                               "OCRBench")]
    selector, deepstack, pos, K_eff = arm_spec(arm, K)
    sides = RRES[K] if arm == "rres" else [None]

    for ds in ds_list:
        for side in sides:
            arm_id = f"{arm}{side}" if side else arm
            path = shard_path(arm_id, K_eff, ds)
            shard = load_shard(path)
            rows = dev_items(plan, ds)
            done = [int(k) for k in shard["records"]]
            if not done:
                continue
            dataset = vlmeval_build(ds)
            data = dataset.data
            sub = data.iloc[sorted(done)].copy()
            for col in ("image",):
                if col in sub.columns:
                    sub = sub.drop(columns=[col])
            sub["prediction"] = [shard["records"][str(i)]["prediction"]
                                 for i in sorted(done)]
            sub["truncated"] = [shard["records"][str(i)]["truncated"]
                                for i in sorted(done)]
            tsv = path.replace(".json", "_pred.tsv")
            sub.to_csv(tsv, sep="\t", index=False)
            try:
                res = dataset.evaluate(tsv)
            except Exception as e:
                record_failure(arm_id, K_eff, ds, "evaluate_exception", e)
                continue
            if hasattr(res, "to_dict"):
                res = res.to_dict()
            if not res:
                # never write an _score.json with empty official: that is the
                # silent mode that produced the 09-30 BLOCKED macros
                record_failure(arm_id, K_eff, ds, "empty_official",
                               f"evaluate returned nothing for tsv={tsv}")
                continue
            per_q = None
            det = path.replace(".json", "_pred_results.tsv")
            if os.path.exists(det):
                d = pd.read_csv(det, sep="\t")
                if "eval_score" in d.columns:
                    per_q = {str(int(r["index"])): float(r["eval_score"])
                             for _, r in d.iterrows() if "index" in d.columns}
            summary = dict(arm=arm_id, K=K_eff, ds=ds, n=len(done),
                           official={k: (float(v) if isinstance(v, (int, float))
                                         else str(v)) for k, v in res.items()},
                           per_question=per_q)
            with open(path.replace(".json", "_score.json"), "w") as f:
                json.dump(summary, f, indent=1)
            print(f"[scored] {arm_id} K={K_eff} {ds}: {res}", flush=True)

    if failures:
        raise SystemExit(
            f"run_score: {len(failures)} scoring failure(s) recorded in "
            f"{fail_log}; _score.json for the affected cells was NOT written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--ds", default=None)
    ap.add_argument("--mode", default="both",
                    choices=["gen", "score", "both"])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    plan = json.load(open(os.path.join(OUT_DIR, "e0_plan.json")))
    model = common.load_model(common.BASELINE_MODEL,
                              max_new_tokens=args.max_new_tokens)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)

    if args.limit:
        for ds in plan["datasets"]:
            plan["datasets"][ds]["dev_rows"] = \
                plan["datasets"][ds]["dev_rows"][:args.limit]

    if args.mode in ("gen", "both"):
        run_gen(eng, args, plan)
    if args.mode in ("score", "both"):
        run_score(args, plan)


if __name__ == "__main__":
    main()
