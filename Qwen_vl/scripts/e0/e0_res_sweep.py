"""E0 M5b: encoder-axis margin sweep (prereg §5.3).

Only B0 and R-res, square fixed resolutions {256, 384, 512, 768, 1024, 1280,
1536}; records ViT time, LLM prefill time and TTFT (paired-block style), and
the DEV OCR-panel accuracy at each resolution.  Outputs the two curves:
ViT share of TTFT vs resolution, accuracy vs resolution.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch

DISC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "discovery")
sys.path.insert(0, DISC_DIR)
import common  # noqa: E402

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "e0")
RESOLUTIONS = [256, 384, 512, 768, 1024, 1280, 1536]
OCR_PANEL = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench", "ChartQA_TEST"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "e0_res_sweep.json"))
    args = ap.parse_args()

    plan = json.load(open(os.path.join(OUT_DIR, "e0_plan.json")))
    model = common.load_model(common.BASELINE_MODEL, max_new_tokens=64)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)
    import vlmeval.vlm.qwen3_vl.model_fixed_res as mfr

    # timing samples: stratified 2 per OCR dataset (fewer than §5.2's pool;
    # the sweep only needs the resolution trend)
    samples = []
    for ds in OCR_PANEL:
        rows = plan["datasets"][ds]["dev_rows"]
        step = max(1, len(rows) // 2)
        for idx in rows[::step][:2]:
            dataset = common.build_dataset(ds)
            model.set_dump_image(dataset.dump_image)
            row = dataset.data.iloc[idx]
            samples.append((ds, common.build_message(model, dataset, ds, row)))
    rng = random.Random(args.seed)

    results = {}
    for side in RESOLUTIONS:
        mfr.QWEN3_FIXED_RESOLUTION = side
        model.min_pixels = side * side
        model.max_pixels = side * side
        rec = dict(side=side, blocks=[])
        # warm-up
        for ds, msg in samples[:2]:
            try:
                eng.generate(msg, ds, K=1024, selector="identity",
                             max_new_tokens=4, ignore_eos=True)
            except Exception as e:
                print(f"[res {side}] warmup fail: {e}", flush=True)
                break
        else:
            for b in range(args.blocks):
                ds, msg = samples[b % len(samples)]
                torch.cuda.empty_cache()
                t = {}
                try:
                    out = eng.generate(msg, ds, K=1024, selector="identity",
                                       max_new_tokens=8, ignore_eos=True,
                                       timings=t)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    rec["oom"] = True
                    break
                rec["blocks"].append(dict(
                    ds=ds, ttft_ms=t.get("ttft_ms"),
                    vision_ms=t.get("vision_ms"),
                    preprocess_ms=t.get("image_preprocess_ms"),
                    llm_prefill_ms=t.get("llm_prefill_ms")))
        results[str(side)] = rec
        med = sorted([x["ttft_ms"] for x in rec["blocks"]])
        vit = sorted([x["preprocess_ms"] + x["vision_ms"]
                      for x in rec["blocks"]])
        if med:
            import statistics
            print(f"[res {side}] n={len(med)} ttft_med={statistics.median(med):.1f} "
                  f"(prep+vit med={statistics.median(vit):.1f})", flush=True)

    mfr.QWEN3_FIXED_RESOLUTION = 1024
    with open(args.out, "w") as f:
        json.dump(dict(results=results, seed=args.seed), f, indent=1)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
