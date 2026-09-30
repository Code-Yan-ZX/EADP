"""M13 PART A step 1: DeepStack profiling -- what does each DS branch and
each vision stage really cost in wall-clock on the A40?

Protocol (m12_speed_oracle style): 12 stratified OCR-panel samples,
15 warm-up blocks + >= 30 measured blocks; per-stage sync'd wall clock
(patch_embed+pos / block segments / each DS merger / main merger),
CUDA-event vision total, LLM prefill, TTFT; mean / median / std.

Usage: python m13_profile.py [--blocks 30] [--reps-vit 30]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

DISC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "discovery")
M13_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DISC_DIR)
sys.path.insert(0, M13_DIR)
import common  # noqa: E402
import m13_common as mc  # noqa: E402

OUT_DIR = mc.OUT_DIR
WARMUP = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "m13_ds_profile.json"))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    plan = json.load(open(os.path.join(common.QWEN_ROOT, "outputs", "e0",
                                       "e0_plan.json")))
    model = common.load_model(common.BASELINE_MODEL, max_new_tokens=8)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)
    mc.install_ds_skip_patch(eng)
    visual = eng.inner.visual

    # stratified samples, m12_speed_oracle convention
    samples = []
    for ds in ("TextVQA_VAL", "DocVQA_VAL", "OCRBench", "ChartQA_TEST"):
        rows = plan["datasets"][ds]["dev_rows"]
        step = max(1, len(rows) // 3)
        for idx in rows[::step][:3]:
            dataset = common.build_dataset(ds)
            model.set_dump_image(dataset.dump_image)
            row = dataset.data.iloc[idx]
            samples.append((ds, common.build_message(model, dataset, ds, row)))

    preps = []
    for ds, msg in samples:
        preps.append(eng.prepare(msg, ds))

    # ---------------------------------------------------------- correctness
    # all-on custom forward must be bit-identical to the stock visual pass
    prep = preps[0]
    V_stock, DS_stock = eng.encode(prep)
    V_cust, DS_cust = mc.visual_forward_ds(visual, prep["pv"], prep["gthw"],
                                           ds_on=(1, 1, 1))
    ok_v = bool(torch.equal(V_stock, V_cust))
    ok_ds = all(torch.equal(a, b) for a, b in zip(DS_stock, DS_cust))
    print(f"[gate] all-on bitwise: V={ok_v} DS={ok_ds}", flush=True)
    assert ok_v and ok_ds, "custom forward is not bit-identical to stock"

    # ------------------------------------------------------------ profiling
    print(f"[profile] warm-up {WARMUP} blocks", flush=True)
    for b in range(WARMUP):
        prep = preps[b % len(preps)]
        Vw, DSw = mc.visual_forward_ds(visual, prep["pv"], prep["gthw"])
        eng.prefill(prep, Vw, DSw, None, deepstack=True, pos="mrope3d")

    stage_keys = ["vit_patch_embed_pos_ms", "vit_blocks_0_7_ms",
                  "vit_ds_merger_8_ms", "vit_blocks_8_15_ms",
                  "vit_ds_merger_16_ms", "vit_blocks_16_23_ms",
                  "vit_ds_merger_24_ms", "vit_blocks_24_26_ms",
                  "vit_main_merger_ms"]
    records = []
    for b in range(args.blocks):
        prep = preps[b % len(preps)]
        t = {}
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ev[0].record()
        V, DS = mc.visual_forward_ds(visual, prep["pv"], prep["gthw"],
                                     ds_on=(1, 1, 1), timings=t)
        ev[1].record()
        torch.cuda.synchronize()
        t["vision_ms"] = ev[0].elapsed_time(ev[1])
        evp = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        evp[0].record()
        eng.prefill(prep, V, DS, None, deepstack=True, pos="mrope3d")
        evp[1].record()
        torch.cuda.synchronize()
        t["llm_prefill_ms"] = evp[0].elapsed_time(evp[1])
        records.append(t)
        print(f"[profile] block {b+1}/{args.blocks} vision={t['vision_ms']:.1f}ms",
              flush=True)

    summary = {}
    for key in stage_keys + ["vision_ms", "llm_prefill_ms"]:
        vals = [r[key] for r in records]
        summary[key] = dict(median=statistics.median(vals),
                            mean=sum(vals) / len(vals),
                            std=statistics.pstdev(vals), n=len(vals))
    ds_total = sum(summary[f"vit_ds_merger_{i}_ms"]["median"] for i in mc.DS_INDEXES)
    summary["ds_mergers_total_median"] = ds_total
    summary["vision_median"] = summary["vision_ms"]["median"]

    out = dict(model=common.BASELINE_MODEL, n_blocks=args.blocks,
               stage_keys=stage_keys, summary=summary, records=records)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[saved] {args.out}", flush=True)

    print(f"\n{'stage':28s} {'median':>8s} {'mean':>8s} {'std':>7s}")
    for key in stage_keys + ["vision_ms", "llm_prefill_ms"]:
        s = summary[key]
        print(f"{key:28s} {s['median']:8.2f} {s['mean']:8.2f} {s['std']:7.2f}")
    print(f"\nDS mergers total (median): {ds_total:.2f} ms  "
          f"({ds_total / summary['vision_median'] * 100:.1f}% of vision "
          f"{summary['vision_median']:.1f} ms)")


if __name__ == "__main__":
    main()
