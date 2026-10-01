"""E0 S1 root-cause: b0 TextVQA dual-configuration attribution.

The 09-30 BLOCKED note flagged "b0 TextVQA 85.03 vs historical 73.6".  S0
diff showed the message path is byte-identical to the archived SAGE-conf
pipeline (same common.build_message; N5 32/32).  The remaining candidate
explanation is the configuration: the historical harness ran with DeepStack
OFF + 1-D positions (research_reset M2 §1.4), native E0 runs DeepStack ON +
3-D mRoPE.  This script runs b0 (identity, K=1024) on the SAME DEV-300
TextVQA rows in the legacy configuration, scores it with the same official
path, and reports the paired native-vs-legacy delta.

Output: outputs/e0/acc/b0legacy/K1024/TextVQA_VAL.json (+ _score.json),
analysis appended to outputs/e0/s1_dualpath.json.

Usage:
  python s1_textvqa_dualpath.py            # gen + score + analyze
  python s1_textvqa_dualpath.py --analyze  # skip gen, redo score+analyze
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "e0")
ACC_DIR = os.path.join(OUT_DIR, "acc")
DS = "TextVQA_VAL"
ARM_ID = "b0legacy"
K = 1024


def shard_path() -> str:
    d = os.path.join(ACC_DIR, ARM_ID, f"K{K}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{DS}.json")


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


def run_gen(eng, plan, max_new_tokens):
    rows = plan["datasets"][DS]["dev_rows"]
    dataset = common.build_dataset(DS)
    if hasattr(eng.vlm, "set_dump_image"):
        eng.vlm.set_dump_image(dataset.dump_image)
    path = shard_path()
    shard = load_shard(path)
    t0 = time.time()
    for n, idx in enumerate(rows):
        if str(idx) in shard["records"]:
            continue
        row = dataset.data.iloc[idx]
        msg = common.build_message(eng.vlm, dataset, DS, row)
        timings = {}
        out = eng.generate(msg, DS, K=K, selector="identity",
                           deepstack=False, pos="1d",       # legacy config
                           max_new_tokens=max_new_tokens, timings=timings)
        shard["records"][str(idx)] = dict(
            prediction=out["text"],
            truncated=len(out["gen_ids"]) >= max_new_tokens,
            n_vis_kept=out["meta"]["n_vis_kept"],
            ttft_ms=timings.get("ttft_ms"),
            layer_calls_ok=out["meta"]["layer_calls_ok"])
        if len(shard["records"]) % 10 == 0 or len(shard["records"]) == len(rows):
            save_shard(path, shard)
        if n % 20 == 0:
            el = time.time() - t0
            print(f"[{ARM_ID} {DS}] {n}/{len(rows)} ({el/max(1,n+1):.2f}s/q)",
                  flush=True)
    save_shard(path, shard)
    print(f"[done] {ARM_ID} {DS}: {len(shard['records'])} records", flush=True)


def run_score():
    """Same official path as e0_accuracy.run_score, with the S0 guards."""
    from vlmeval.dataset import build_dataset as vlmeval_build
    import pandas as pd

    path = shard_path()
    shard = load_shard(path)
    done = sorted(int(k) for k in shard["records"])
    if not done:
        raise SystemExit("no records to score")
    dataset = vlmeval_build(DS)
    sub = dataset.data.iloc[done].copy()
    if "image" in sub.columns:
        sub = sub.drop(columns=["image"])
    sub["prediction"] = [shard["records"][str(i)]["prediction"] for i in done]
    sub["truncated"] = [shard["records"][str(i)]["truncated"] for i in done]
    tsv = path.replace(".json", "_pred.tsv")
    sub.to_csv(tsv, sep="\t", index=False)
    res = dataset.evaluate(tsv)
    if hasattr(res, "to_dict"):
        res = res.to_dict()
    if not res:
        raise SystemExit("S0: evaluate returned nothing; _score.json NOT written")
    per_q = None
    det = path.replace(".json", "_pred_results.tsv")
    if os.path.exists(det):
        d = pd.read_csv(det, sep="\t")
        if "eval_score" in d.columns and "index" in d.columns:
            per_q = {str(int(r["index"])): float(r["eval_score"])
                     for _, r in d.iterrows()}
    summary = dict(arm=ARM_ID, K=K, ds=DS, n=len(done), official=res,
                   per_question=per_q,
                   note="S1 legacy-config b0 (deepstack off, 1d pos)")
    with open(path.replace(".json", "_score.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[scored] {ARM_ID}: {res}", flush=True)


def run_analyze():
    import numpy as np
    import pandas as pd
    import e0_analyze

    def per_q_of(arm_id):
        """Per-question eval_score from the vlmeval detail file, keyed by the
        dataset's original row id (the detail `index` column)."""
        p = os.path.join(ACC_DIR, arm_id, f"K{K}", f"{DS}_pred_results.xlsx")
        d = pd.read_excel(p)
        s = {str(int(r["index"])): float(r["eval_score"])
             for _, r in d.iterrows()}
        off = json.load(open(os.path.join(ACC_DIR, arm_id, f"K{K}",
                                          f"{DS}_score.json")))["official"]
        return s, off

    pq_leg, off_leg = per_q_of(ARM_ID)
    pq_nat, off_nat = per_q_of("b0")
    common_idx = sorted(set(pq_leg) & set(pq_nat), key=int)
    if len(common_idx) < 300:
        print(f"[analyze] WARNING: only {len(common_idx)} common rows")
    print(f"[analyze] n_common={len(common_idx)} "
          f"legacy_official={off_leg} native_official={off_nat}")
    keys = e0_analyze.image_keys(DS, common_idx)
    delta, lo, hi = e0_analyze.paired_bootstrap(
        {i: pq_nat[i] for i in common_idx},
        {i: pq_leg[i] for i in common_idx}, keys, keys)
    rescued = sum(1 for i in common_idx if pq_nat[i] > 0.5 and pq_leg[i] <= 0.5)
    broken = sum(1 for i in common_idx if pq_nat[i] <= 0.5 and pq_leg[i] > 0.5)
    out = dict(ds=DS, n=len(common_idx),
               native_b0_official=off_nat,
               legacy_b0_official=off_leg,
               per_q_mean_native=float(np.mean([pq_nat[i] for i in common_idx])),
               per_q_mean_legacy=float(np.mean([pq_leg[i] for i in common_idx])),
               paired_native_minus_legacy=dict(delta=delta, ci=[lo, hi],
                                               rescued=rescued, broken=broken),
               historical_reference="sage_conf_B2 TextVQA 74.21 (n=240, legacy "
                                    "config, different 240-row panel); "
                                    "research_reset M2 §1.4 legacy harness",
               verdict=None)
    print(f"[analyze] native - legacy = {delta:+.2f} [{lo:+.2f}, {hi:+.2f}] "
          f"rescue/break={rescued}/{broken}")
    print(f"[analyze] legacy per-q mean {out['per_q_mean_legacy']:.2f} vs "
          f"native {out['per_q_mean_native']:.2f}")
    with open(os.path.join(OUT_DIR, "s1_dualpath.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("[saved] s1_dualpath.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true",
                    help="skip generation, redo score+analyze")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    if not args.analyze:
        plan = json.load(open(os.path.join(OUT_DIR, "e0_plan.json")))
        model = common.load_model(common.BASELINE_MODEL,
                                  max_new_tokens=args.max_new_tokens)
        from model.native_qwen3 import NativeEngine
        eng = NativeEngine(model)
        run_gen(eng, plan, args.max_new_tokens)
        del eng, model
        torch.cuda.empty_cache()
    run_score()
    run_analyze()


if __name__ == "__main__":
    main()
