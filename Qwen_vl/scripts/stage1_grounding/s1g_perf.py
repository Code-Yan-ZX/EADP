"""Stage-1 grounding discovery — paired latency.

Measures the ONLINE b1 selector latency (the BASE reference; gate G2
proves online b1 == the frozen-bank BASE pipeline) on DEV smoke samples
and compares with each arm's recorded selector_ms from its shards.

Usage: python s1g_perf.py --arms A_G1,...,C_G1 --lam <frozen> --samples 10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

import s1g_common as SC


def measure_b1(eng, dataset_cache, manifest, n):
    from model.e0_selectors import run_selector
    out = []
    for ds in SC.DS_LIST:
        dataset = dataset_cache[ds]
        for it in manifest["datasets"][ds]["dev"][:n]:
            row = dataset.data.iloc[it["idx"]]
            msg = SC.common.build_message(eng.vlm, dataset, ds, row)
            prep = eng.prepare(msg, ds)
            V, DS = eng.encode(prep)
            text_mean, text_seq = eng.instruction_embeds(msg, ds)
            if prep["n_vis"] <= SC.K:
                continue
            ctx = dict(prep=prep, V=V, DS=DS, K=SC.K, engine=eng,
                       text_mean=text_mean, text_seq=text_seq,
                       attn_list=None, vz=None, seed=None)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_selector("b1", SC.K, ctx)
            torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) * 1e3)
    return out


def arm_selector_ms(arm, split="dev", tag=""):
    hits = sorted(glob.glob(os.path.join(
        SC.ACC_DIR, split, arm + (f"_{tag}" if tag else ""), "K*", "*.json")))
    vals = []
    for h in hits:
        if h.endswith("_score.json") or h.endswith("_diag.jsonl"):
            continue
        shard = json.load(open(h))
        vals += [r["selector_ms"] for r in shard["records"].values()
                 if r.get("selector_ms") is not None]
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--tag", default="",
                    help="shard dir suffix where arm selector_ms live")
    args = ap.parse_args()

    manifest = SC.load_manifest()
    model = SC.common.load_model(SC.common.BASELINE_MODEL, max_new_tokens=2048)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)
    dataset_cache = {ds: SC.common.build_dataset(ds) for ds in SC.DS_LIST}
    for ds in SC.DS_LIST:
        eng.vlm.set_dump_image(dataset_cache[ds].dump_image)

    b1 = measure_b1(eng, dataset_cache, manifest, args.samples)
    result = dict(n_samples=len(b1),
                  BASE_online_b1_ms=dict(mean=float(np.mean(b1)),
                                         p50=float(np.percentile(b1, 50))))
    print(f"BASE (online b1): mean {np.mean(b1):.1f} ms  "
          f"p50 {np.percentile(b1, 50):.1f} ms  (n={len(b1)})")
    for arm in args.arms.split(","):
        v = arm_selector_ms(arm, tag=args.tag)
        if not v:
            print(f"{arm}: no shard timings yet")
            continue
        m = float(np.mean(v))
        result[arm] = dict(mean_ms=m, p50_ms=float(np.percentile(v, 50)),
                           n=len(v),
                           overhead_ms=m - float(np.mean(b1)))
        print(f"{arm}: mean {m:.1f} ms  overhead {m - np.mean(b1):+.1f} ms "
              f"(n={len(v)})")

    p = os.path.join(SC.OUT_DIR, "perf_selector.json")
    with open(p, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
