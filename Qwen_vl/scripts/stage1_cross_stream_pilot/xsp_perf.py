"""Stage-1 Cross-Stream Pilot — paired live efficiency (protocol §7).

Four arms (E_GATHER, E_MAIN025, X_GATHER, X_MAIN025) on the same 30 DEV
samples (10 per dataset), FULLY LIVE: encode -> [similarity] -> Stage-1
scoring -> official facility -> assignment/completion -> prefill -> decode.
No bank lookups.  Main-path similarity is computed once per request and
reused by scoring + selector (new scorers).  15 warm-up paired blocks,
interleaved arm order with alternating direction, CUDA-synced, per-block
peak-VRAM reset, ignore_eos=True fixed 64-token decode (asserted).

TTFT boundary: from the start of `prepare` (message/processor) through
encode, scoring/selection, completion and prefill, up to the first-token
logits — i.e. full end-to-end pre-first-token, NOT a pure prefill number.
The EADP path's instruction embedding is timed inside its own path (the new
scorers genuinely skip it).

Usage: python xsp_perf.py --samples 30
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import amp_common as AC
import xsp_common as XC

N_WARMUP = 15
FIXED_TOKENS = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(XC.PERF_ARMS))
    ap.add_argument("--samples", type=int, default=30,
                    help="total; 10 per dataset (frozen)")
    ap.add_argument("--fixed-tokens", type=int, default=FIXED_TOKENS)
    args = ap.parse_args()
    arms = args.arms.split(",")

    manifest = json.load(open(os.path.join(AC.OUT_DIR, "manifest.json")))
    model = AC.common.load_model(AC.common.BASELINE_MODEL, max_new_tokens=64)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)

    items = []
    for ds in XC.DS_LIST:
        dataset = AC.common.build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        for it in manifest["datasets"][ds]["dev"][:args.samples // 3]:
            row = dataset.data.iloc[it["idx"]]
            msg = AC.common.build_message(model, dataset, ds, row)
            items.append(dict(ds=ds, idx=it["idx"], message=msg))

    # warm-up paired blocks
    for i in range(N_WARMUP):
        it = items[i % len(items)]
        for arm in arms:
            XC.run_one_live(eng, it["message"], it["ds"],
                            XC.ARMS[arm]["scorer"], arm, args.fixed_tokens,
                            ignore_eos=True)
    torch.cuda.synchronize()

    records = {a: [] for a in arms}
    vram = {a: [] for a in arms}
    for i, it in enumerate(items):
        order = arms if i % 2 == 0 else list(reversed(arms))
        for arm in order:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timings = {}
            out = XC.run_one_live(eng, it["message"], it["ds"],
                                  XC.ARMS[arm]["scorer"], arm,
                                  args.fixed_tokens, ignore_eos=True,
                                  timings=timings)
            torch.cuda.synchronize()
            vram[arm].append(
                int(torch.cuda.max_memory_allocated()) / 2 ** 20)
            assert len(out["gen_ids"]) == args.fixed_tokens
            records[arm].append(dict(ds=it["ds"], idx=it["idx"],
                                     ttft_ms=timings["ttft_ms"],
                                     vision_ms=timings["vision_ms"],
                                     text_embeds_ms=timings.get("text_embeds_ms"),
                                     eadp_stage1_facility_ms=timings.get(
                                         "eadp_stage1_facility_ms"),
                                     similarity_ms=timings.get("similarity_ms"),
                                     stage1_ms=timings.get("stage1_ms"),
                                     facility_ms=timings.get("facility_ms"),
                                     completion_ms=timings.get("completion_ms"),
                                     llm_prefill_ms=timings["llm_prefill_ms"],
                                     decode_fixed_ms=timings["decode_wall_ms"],
                                     n_vis_kept=out["meta"]["n_vis_kept"]))

    def med(x):
        return float(np.median(x)) if len(x) else None

    summary = dict(
        arms=arms, n_blocks={a: len(records[a]) for a in arms},
        warmup=N_WARMUP, fixed_tokens=args.fixed_tokens,
        live=True, ttft_boundary=("prepare -> encode -> scoring/selection -> "
                                  "completion -> prefill -> first token"),
        stats={a: dict(
            ttft_med_ms=med([r["ttft_ms"] for r in records[a]]),
            ttft_mean_ms=float(np.mean([r["ttft_ms"] for r in records[a]])),
            vision_med_ms=med([r["vision_ms"] for r in records[a]]),
            text_embeds_med_ms=med([r["text_embeds_ms"] or 0.0
                                    for r in records[a]]),
            eadp_stage1_facility_med_ms=med(
                [r["eadp_stage1_facility_ms"] or 0.0 for r in records[a]]),
            similarity_med_ms=med([r["similarity_ms"] or 0.0
                                   for r in records[a]]),
            stage1_med_ms=med([r["stage1_ms"] or 0.0 for r in records[a]]),
            facility_med_ms=med([r["facility_ms"] or 0.0
                                 for r in records[a]]),
            completion_med_ms=med([r["completion_ms"] or 0.0
                                   for r in records[a]]),
            prefill_med_ms=med([r["llm_prefill_ms"] for r in records[a]]),
            decode_fixed_med_ms=med([r["decode_fixed_ms"]
                                     for r in records[a]]),
            total_gen_med_ms=med([r["ttft_ms"] + r["decode_fixed_ms"]
                                  for r in records[a]]),
            peak_vram_p50_mb=float(np.median(vram[a])),
            peak_vram_max_mb=float(np.max(vram[a])),
            n_vis=int(np.mean([r["n_vis_kept"] for r in records[a]])),
        ) for a in arms},
        blocks={a: records[a] for a in arms})

    out_path = os.path.join(XC.OUT_DIR, "xsp_perf.json")
    with open(out_path + ".tmp", "w") as f:
        json.dump(summary, f, indent=1)
    os.replace(out_path + ".tmp", out_path)
    for a in arms:
        s = summary["stats"][a]
        print(f"[{a}] ttft={s['ttft_med_ms']:.1f}ms "
              f"sim={s['similarity_med_ms']:.2f} "
              f"stage1={s['stage1_med_ms']:.2f} "
              f"facility={s['facility_med_ms']:.2f} "
              f"eadp_s1+fac={s['eadp_stage1_facility_med_ms']:.2f} "
              f"completion={s['completion_med_ms']:.2f} "
              f"prefill={s['prefill_med_ms']:.1f} "
              f"decode64={s['decode_fixed_med_ms']:.0f}ms "
              f"vram_p50={s['peak_vram_p50_mb']:.0f}MB")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
