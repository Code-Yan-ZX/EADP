#!/usr/bin/env python3
"""
MosaicPrune efficiency profile (phase 1).

Mirrors the measurement protocol of profile_efficiency.py on the fixed-res
Qwen3-VL pipeline: per dataset, on a deterministic sample,

  * vision-tower forward            (shared by every method)
  * mosaic compression              (partition + pooling wall-clock)
  * pruned LLM-only prefill         (replayed on the exact decoder inputs)
  * end-to-end generate             (greedy, max_new_tokens=2048)
  * analytic prefill FLOPs, peak GPU memory

Baseline (no pruning) numbers already exist in
outputs/discovery/efficiency_profile.json from the same protocol; this script
only profiles the mosaic arms.

Usage:
  python scripts/discovery/mosaic_efficiency.py \
      --datasets TextVQA_VAL DocVQA_VAL OCRBench --per-dataset 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
from common import (  # noqa: E402
    analytic_prefill_flops_g,
    collect_samples,
    free_model,
    load_model,
    time_callable,
)
from model.mosaic import mosaic_compress  # noqa: E402
from profile_efficiency import llm_forward  # noqa: E402


def profile_mosaic_sample(model, message, dataset_name, cfg):
    inputs = model._processor_inputs(model._build_messages(message, dataset=dataset_name))
    gthw = inputs["image_grid_thw"]
    pv = inputs["pixel_values"].type(model.model.visual.dtype)

    for _ in range(cfg["warmup"]):
        with torch.no_grad():
            model.model.visual(pv, grid_thw=gthw)

    vision_mean, vision_std = time_callable(
        lambda: model.model.visual(pv, grid_thw=gthw), cfg["repeat"], 1)

    def compress():
        with torch.no_grad():
            feats = model.model.visual(pv, grid_thw=gthw)
            return mosaic_compress(feats, gthw, model.visual_token_num,
                                   model.mode, seed=model.seed, stats_file=None)

    compress_mean, compress_std = time_callable(compress, cfg["repeat"], 1)
    with torch.no_grad():
        feats = model.model.visual(pv, grid_thw=gthw)
        pruned_embeds, sizes = mosaic_compress(
            feats, gthw, model.visual_token_num, model.mode,
            seed=model.seed, stats_file=None)

    inputs_embeds, attention_mask = model._build_pruned_inputs(
        inputs["input_ids"], inputs["attention_mask"], pruned_embeds, sizes)
    seq_len = int(inputs_embeds.shape[1])
    n_kept = int(sum(sizes))

    prefill_mean, prefill_std = time_callable(
        lambda: llm_forward(model, inputs_embeds, attention_mask),
        cfg["repeat"], cfg["warmup"])

    gen_lens, gen_times = [], []
    for _ in range(cfg["gen_repeat"]):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            out = model.model.generate(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                do_sample=False, **model.generate_kwargs)
        e.record()
        torch.cuda.synchronize()
        gen_times.append(s.elapsed_time(e))
        gen_lens.append(int(out.shape[1]))

    torch.cuda.reset_peak_memory_stats()
    llm_forward(model, inputs_embeds, attention_mask)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2

    return {
        "visual_tokens_in": int(gthw.prod(-1).sum().item()) // 4,
        "visual_tokens_kept": n_kept,
        "seq_len": seq_len,
        "vision_ms_mean": vision_mean,
        "vision_ms_std": vision_std,
        "compress_ms_mean": compress_mean,
        "compress_ms_std": compress_std,
        "prefill_ms_mean": prefill_mean,
        "prefill_ms_std": prefill_std,
        "prefill_total_ms_mean": prefill_mean + vision_mean + compress_mean,
        "e2e_ms_mean": float(np.mean(gen_times)),
        "e2e_ms_std": float(np.std(gen_times)),
        "gen_tokens": float(np.mean(gen_lens)),
        "flops_g": analytic_prefill_flops_g(model.model, seq_len),
        "peak_mem_mb": peak,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "Qwen3-VL-8B-Mosaic-256-uniform",
        "Qwen3-VL-8B-Mosaic-256-random",
        "Qwen3-VL-8B-Mosaic-256-dispersion",
        "Qwen3-VL-8B-Mosaic-128-uniform",
        "Qwen3-VL-8B-Mosaic-128-random",
        "Qwen3-VL-8B-Mosaic-128-dispersion",
    ])
    ap.add_argument("--datasets", nargs="+",
                    default=["TextVQA_VAL", "DocVQA_VAL", "OCRBench"])
    ap.add_argument("--per-dataset", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--gen-repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = {"warmup": args.warmup, "repeat": args.repeat,
           "gen_repeat": args.gen_repeat, "max_new_tokens": args.max_new_tokens}
    out_path = args.out or os.path.join(common.OUTPUT_DIR, "mosaic",
                                        "efficiency_mosaic.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results = {}
    for model_name in args.models:
        model = load_model(model_name, max_new_tokens=args.max_new_tokens)
        per_model = {}
        for dataset_name in args.datasets:
            dataset, idxs, rows, messages = collect_samples(
                model, dataset_name, args.per_dataset, offset=args.offset)
            samples = []
            for row, message in zip(rows, messages):
                try:
                    samples.append(profile_mosaic_sample(
                        model, message, dataset_name, cfg))
                except Exception as err:  # noqa: BLE001
                    print(f"  sample failed on {model_name}/{dataset_name}: {err}")
            agg = {}
            for k in samples[0]:
                vals = [s[k] for s in samples if not np.isnan(s[k])]
                agg[k] = {"mean": float(np.mean(vals)),
                          "std": float(np.std(vals)) if len(vals) > 1 else 0.0}
            agg["n"] = len(samples)
            per_model[dataset_name] = agg
            print(f"{model_name} {dataset_name}: "
                  f"kept={agg['visual_tokens_kept']['mean']:.0f} "
                  f"compress={agg['compress_ms_mean']['mean']:.1f}ms "
                  f"prefill={agg['prefill_total_ms_mean']['mean']:.1f}ms "
                  f"e2e={agg['e2e_ms_mean']['mean']:.1f}ms")
        results[model_name] = per_model
        free_model(model)

    with open(out_path, "w") as f:
        json.dump({"config": cfg, "models": results}, f, indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
