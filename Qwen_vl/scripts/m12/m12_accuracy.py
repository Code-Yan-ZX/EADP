"""M12 accuracy screening: pre-ViT selectors x budgets x DEV OCR panel.

Arms (all structured selectors share the EXACT merged-token budget K):
  random / uniform / variance / sobel / graydog / event / retinagate
  rres (resolution control, side chosen to match K)
Baselines b0 (K=1024) and b2 (post-encoder EADP, K=256) come from the E0
shards when present; everything here is generated through the SAME native
engine path, one sample at a time, per-sample predictions resumable in
shards, scored by the official VLMEvalKit rules (m12_score reuses
e0_accuracy.run_score logic via shard layout compatibility).

Usage:
  python m12_accuracy.py --arm retinagate --K 512 --ds TextVQA_VAL,DocVQA_VAL,OCRBench
  python m12_accuracy.py --arm retinagate --K 512 --mode score
  python m12_accuracy.py --arm rres --K 512 ...
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

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "m12")
ACC_DIR = os.path.join(OUT_DIR, "acc")
OCR_PANEL = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]

PRE_VIT_ARMS = ["random", "uniform", "variance", "sobel", "graydog",
                "event", "retinagate"]

# R-res: budget K (merged tokens) -> square side (multiple of 32, nearest)
RRES_SIDE = {768: 896, 512: 704, 384: 640, 256: 512, 1024: 1024}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--ds", default=",".join(OCR_PANEL))
    ap.add_argument("--mode", default="both", choices=["gen", "score", "both"])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--base-frac", type=float, default=0.32)
    args = ap.parse_args()

    plan = json.load(open(os.path.join(common.QWEN_ROOT, "outputs", "e0",
                                       "e0_plan.json")))
    ds_list = args.ds.split(",")
    model = common.load_model(common.BASELINE_MODEL,
                              max_new_tokens=args.max_new_tokens)
    from model.native_qwen3 import NativeEngine
    from model import retinagate as rg
    import vlmeval.vlm.qwen3_vl.model_fixed_res as mfr
    eng = NativeEngine(model)

    arm = args.arm
    K = args.K
    is_rres = arm == "rres"
    is_b0 = arm == "b0"

    if args.mode in ("gen", "both"):
        for ds in ds_list:
            rows = plan["datasets"][ds]["dev_rows"]
            dataset = common.build_dataset(ds)
            model.set_dump_image(dataset.dump_image)
            path = shard_path(arm, K, ds)
            shard = load_shard(path)
            if is_rres:
                side = RRES_SIDE[K]
                mfr.QWEN3_FIXED_RESOLUTION = side
                model.min_pixels = side * side
                model.max_pixels = side * side
            t0 = time.time()
            try:
                for n, idx in enumerate(rows):
                    if str(idx) in shard["records"]:
                        continue
                    row = dataset.data.iloc[idx]
                    msg = common.build_message(model, dataset, ds, row)
                    timings = {}
                    if is_b0:
                        out = eng.generate(msg, ds, K=1024,
                                           selector="identity",
                                           max_new_tokens=args.max_new_tokens,
                                           timings=timings)
                    elif is_rres:
                        out = eng.generate(msg, ds, K=1024,
                                           selector="identity",
                                           max_new_tokens=args.max_new_tokens,
                                           timings=timings)
                    else:
                        out = rg.rg_generate(
                            eng, msg, ds, k=K, mode=arm,
                            max_new_tokens=args.max_new_tokens,
                            timings=timings, seed=args.seed + 1000 * idx,
                            base_frac=args.base_frac)
                    rec = dict(prediction=out["text"],
                               truncated=len(out["gen_ids"])
                               >= args.max_new_tokens,
                               n_vis_kept=out["meta"]["n_vis_kept"],
                               ttft_ms=timings.get("ttft_ms"),
                               gate_ms=timings.get("gate_ms"),
                               layer_calls_ok=True)
                    shard["records"][str(idx)] = rec
                    if len(shard["records"]) % 20 == 0:
                        save_shard(path, shard)
                    if n % 20 == 0:
                        el = time.time() - t0
                        print(f"[{arm} K={K} {ds}] {n}/{len(rows)} "
                              f"({el/max(1, n+1):.2f}s/q)", flush=True)
            finally:
                save_shard(path, shard)
                if is_rres:
                    mfr.QWEN3_FIXED_RESOLUTION = 1024
                    model.min_pixels = 1024 * 1024
                    model.max_pixels = 1024 * 1024
            print(f"[done] {arm} K={K} {ds}: {len(shard['records'])} records",
                  flush=True)

    if args.mode in ("score", "both"):
        from vlmeval.dataset import build_dataset as vlmeval_build
        import pandas as pd
        for ds in ds_list:
            path = shard_path(arm, K, ds)
            shard = load_shard(path)
            rows = plan["datasets"][ds]["dev_rows"]
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
                print(f"[score FAIL] {arm} K={K} {ds}: {e}", flush=True)
                continue
            if hasattr(res, "to_dict"):
                res = res.to_dict()
            per_q = None
            det = path.replace(".json", "_pred_results.tsv")
            if os.path.exists(det):
                d = pd.read_csv(det, sep="\t")
                if "eval_score" in d.columns:
                    per_q = {str(int(r["index"])): float(r["eval_score"])
                             for _, r in d.iterrows() if "index" in d.columns}
            summary = dict(arm=arm, K=K, ds=ds, n=len(done),
                           official={k: (float(v) if isinstance(v, (int, float))
                                         else str(v)) for k, v in res.items()},
                           per_question=per_q)
            with open(path.replace(".json", "_score.json"), "w") as f:
                json.dump(summary, f, indent=1)
            print(f"[scored] {arm} K={K} {ds}: {res}", flush=True)


if __name__ == "__main__":
    main()
