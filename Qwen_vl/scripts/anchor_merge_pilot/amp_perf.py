"""Anchor-Merge Pilot — paired efficiency (protocol §8).

Measures BASE and the winner (+ controls' merge stage cost) on the same
samples, interleaved order, 15 warm-up blocks, CUDA-synced.  Every block runs
a FULL fresh generate (prepare -> encode -> [merge] -> prefill -> decode), so
no measurement ever reuses a decoded state/cache.  Fixed-length generation:
ignore_eos + 64 tokens.  Peak VRAM per arm is read after a dedicated fresh
block with reset peak stats.

Stages: image_preprocess, vision, selector(bank lookup), merge (grouping +
pooling), llm_prefill, ttft (end-to-end to first token), decode64 (fixed
64-token wall).

Usage: python amp_perf.py --arms BASE,U025 --samples 30 --split dev
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import amp_common as AC

N_WARMUP = 15
FIXED_TOKENS = 64


def block(eng, msg, ds, rec, cfg, max_new_tokens, ignore_eos):
    timings = {}
    out = AC.run_one(eng, msg, ds, rec, cfg, max_new_tokens=max_new_tokens,
                     ignore_eos=ignore_eos, timings=timings)
    return timings, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="BASE,U025")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--fixed-tokens", type=int, default=FIXED_TOKENS)
    ap.add_argument("--winner-lam", type=float, default=0.0)
    ap.add_argument("--winner-kind", default="uniform")
    args = ap.parse_args()
    arms = args.arms.split(",")

    manifest = json.load(open(os.path.join(AC.OUT_DIR, "manifest.json")))
    model = AC.common.load_model(AC.common.BASELINE_MODEL, max_new_tokens=64)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)

    # stratified samples: 10 per dataset
    items = []
    for ds in AC.DS_LIST:
        bank = AC.load_bank(args.split, ds)
        dataset = AC.common.build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        for it in manifest["datasets"][ds][args.split][:args.samples // 3]:
            row = dataset.data.iloc[it["idx"]]
            msg = AC.common.build_message(model, dataset, ds, row)
            items.append(dict(ds=ds, idx=it["idx"], message=msg,
                              rec=bank[str(it["idx"])]))

    cfgs = {a: AC.arm_cfg(a, args.winner_lam, args.winner_kind) for a in arms}

    # warm-up
    for i in range(N_WARMUP):
        it = items[i % len(items)]
        for a in arms:
            block(eng, it["message"], it["ds"], it["rec"], cfgs[a],
                  args.fixed_tokens, True)
    torch.cuda.synchronize()

    # paired blocks, interleaved arm order, alternating sequence direction
    records = {a: [] for a in arms}
    vram = {a: [] for a in arms}
    for i, it in enumerate(items):
        order = arms if i % 2 == 0 else list(reversed(arms))
        for a in order:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t, out = block(eng, it["message"], it["ds"], it["rec"], cfgs[a],
                           args.fixed_tokens, True)
            torch.cuda.synchronize()
            vram[a].append(int(torch.cuda.max_memory_allocated()) / 2**20)
            records[a].append(dict(
                ds=it["ds"], idx=it["idx"],
                ttft_ms=t["ttft_ms"], vision_ms=t["vision_ms"],
                merge_ms=t.get("merge_ms"),
                llm_prefill_ms=t["llm_prefill_ms"],
                decode_fixed_ms=t["decode_wall_ms"],
                n_vis_kept=out["meta"]["n_vis_kept"]))

    # end-to-end TTFT at the accuracy setting (2048 cap) for context
    e2e = {a: [] for a in arms}
    for it in items[:10]:
        for a in arms:
            torch.cuda.synchronize()
            t, _ = block(eng, it["message"], it["ds"], it["rec"], cfgs[a],
                         2048, False)
            torch.cuda.synchronize()
            e2e[a].append(dict(ds=it["ds"], idx=it["idx"],
                               ttft_ms=t["ttft_ms"]))

    def med(x):
        return float(np.median(x)) if len(x) else None

    summary = dict(arms=arms, n_blocks={a: len(records[a]) for a in arms},
                   warmup=N_WARMUP, fixed_tokens=args.fixed_tokens,
                   stats={a: dict(
                       ttft_med_ms=med([r["ttft_ms"] for r in records[a]]),
                       ttft_mean_ms=float(np.mean([r["ttft_ms"] for r in records[a]])),
                       vision_med_ms=med([r["vision_ms"] for r in records[a]]),
                       merge_med_ms=med([r["merge_ms"] or 0.0 for r in records[a]]),
                       prefill_med_ms=med([r["llm_prefill_ms"] for r in records[a]]),
                       decode_fixed_med_ms=med([r["decode_fixed_ms"] for r in records[a]]),
                       peak_vram_p50_mb=float(np.median(vram[a])),
                       peak_vram_max_mb=float(np.max(vram[a])),
                       e2e_ttft_med_ms=med([r["ttft_ms"] for r in e2e[a]]),
                       n_vis=int(np.mean([r["n_vis_kept"] for r in records[a]])),
                   ) for a in arms},
                   blocks={a: records[a] for a in arms},
                   e2e={a: e2e[a] for a in arms})

    out_path = os.path.join(AC.OUT_DIR, "amp_perf.json")
    with open(out_path + ".tmp", "w") as f:
        json.dump(summary, f, indent=1)
    os.replace(out_path + ".tmp", out_path)
    for a in arms:
        s = summary["stats"][a]
        print(f"[{a}] ttft={s['ttft_med_ms']:.1f}ms prefill={s['prefill_med_ms']:.1f} "
              f"merge={s['merge_med_ms']:.2f} decode{args.fixed_tokens}="
              f"{s['decode_fixed_med_ms']:.0f}ms vram_p50={s['peak_vram_p50_mb']:.0f}MB")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
