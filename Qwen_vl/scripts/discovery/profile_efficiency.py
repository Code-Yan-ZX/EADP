#!/usr/bin/env python3
"""
Part 1 -- reproduce the EADP efficiency profile on Qwen3-VL-8B.

Measures, on a deterministic sample of TextVQA_VAL / DocVQA_VAL / OCRBench:

  * Unpruned baseline  : input visual tokens, prefill, end-to-end latency,
                         analytic FLOPs, peak GPU memory
  * EADP-{128,256,512}: same, plus retained tokens and a per-stage pruning
                         breakdown matching Table 14 of the paper
                         (global guidance / dense guidance / score fusion /
                          smoothing / polarization / facility location).

Timing hygiene: CUDA events on the default stream, warmup runs discarded, N
repeats, mean +/- std reported, model init and dataset loading excluded.

Usage:
  python scripts/discovery/profile_efficiency.py \
      --datasets TextVQA_VAL DocVQA_VAL OCRBench --per-dataset 50 \
      --budgets 128 256 512 --repeat 5 --gen-repeat 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
from common import (  # noqa: E402
    BASELINE_MODEL,
    analytic_prefill_flops_g,
    build_dataset,
    eadp_model_name,
    ensure_out_dir,
    free_model,
    load_model,
    sample_indices,
    time_callable,
)
from instrumented import ALL_STAGE_ORDER, attach_pruner  # noqa: E402
from vlmeval.vlm.qwen3_vl.model_fixed_res import unwrap_visual_output  # noqa: E402


def base_kwargs(model, inputs):
    return {k: v for k, v in inputs.items() if v is not None}


def simple_position_ids(inputs_embeds):
    b, s, _ = inputs_embeds.shape
    pos = torch.arange(s, device=inputs_embeds.device).view(1, 1, -1).expand(3, b, -1)
    return pos


def llm_forward(model, inputs_embeds, attention_mask):
    try:
        return model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
    except Exception:
        lm = model.model.model.language_model
        return lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=simple_position_ids(inputs_embeds),
            use_cache=False,
        )


def vision_forward(model, pixel_values, grid_thw):
    with torch.no_grad():
        return model.model.visual(pixel_values, grid_thw=grid_thw)


def capture_llm_inputs(model, kw):
    """Run the full model once and grab the exact inputs the decoder stack sees."""
    captured = []

    def hook(_m, args):
        if args and isinstance(args[0], torch.Tensor):
            captured.append(args[0].detach())

    lm = model.model.model.language_model
    handle = lm.layers[0].register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model.model(**kw, use_cache=False)
    finally:
        handle.remove()
    if not captured:
        return None, None
    return captured[0], kw.get("attention_mask")


def do_generate(model, **kwargs):
    """Mirror the official generate call, returning raw ids so we can count tokens."""
    with torch.no_grad():
        return model.model.generate(do_sample=False, **model.generate_kwargs, **kwargs)


def run_baseline_sample(model, message, dataset_name, cfg):
    inputs = model._processor_inputs(model._build_messages(message, dataset=dataset_name))
    kw = base_kwargs(model, inputs)
    gthw = inputs["image_grid_thw"]
    pv = inputs["pixel_values"].type(model.model.visual.dtype)
    n_vis = int(gthw.prod(-1).sum().item()) // 4
    seq_len = int(inputs["input_ids"].shape[1])

    for _ in range(cfg["warmup"]):
        with torch.no_grad():
            model.model(**kw, use_cache=False)

    vision_mean, vision_std = time_callable(
        lambda: vision_forward(model, pv, gthw), cfg["repeat"], 1
    )
    total_mean, total_std = time_callable(
        lambda: model.model(**kw, use_cache=False), cfg["repeat"], 1
    )

    # LLM-only prefill: replay the decoder on the exact embeddings it received.
    embeds, mask = capture_llm_inputs(model, kw)
    if embeds is not None:
        llm_mean, llm_std = time_callable(
            lambda: llm_forward(model, embeds, mask), cfg["repeat"], 1
        )
    else:
        llm_mean, llm_std = float("nan"), float("nan")

    gen_lens, gen_times = [], []
    for _ in range(cfg["gen_repeat"]):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = do_generate(model, **kw)
        e.record()
        torch.cuda.synchronize()
        gen_times.append(s.elapsed_time(e))
        gen_lens.append(int(out.shape[1] - seq_len))

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model.model(**kw, use_cache=False)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2

    return {
        "visual_tokens_in": n_vis,
        "visual_tokens_kept": n_vis,
        "seq_len": seq_len,
        "prefill_ms_mean": llm_mean,
        "prefill_ms_std": llm_std,
        "prefill_total_ms_mean": total_mean,
        "prefill_total_ms_std": total_std,
        "vision_ms_mean": vision_mean,
        "vision_ms_std": vision_std,
        "e2e_ms_mean": float(np.mean(gen_times)),
        "e2e_ms_std": float(np.std(gen_times)),
        "gen_tokens": float(np.mean(gen_lens)),
        "flops_g": analytic_prefill_flops_g(model.model, seq_len),
        "peak_mem_mb": peak,
        "stages": {},
        "prune_ms_mean": 0.0,
        "prune_ms_std": 0.0,
    }


def run_eadp_sample(model, message, dataset_name, cfg):
    pruner = model.pruner
    instr = model._get_instruction_sequence_embedding(message, dataset=dataset_name)

    inputs = model._processor_inputs(model._build_messages(message, dataset=dataset_name))
    pv = inputs["pixel_values"].type(model.model.visual.dtype)
    gthw = inputs["image_grid_thw"]
    n_vis_in = int(gthw.prod(-1).sum().item()) // 4

    # Vision tower is shared by every method; hoist it out so `prune_ms`
    # measures the pruning overhead alone (this is the paper's convention).
    vision_mean, vision_std = time_callable(
        lambda: vision_forward(model, pv, gthw), cfg["repeat"], cfg["warmup"]
    )
    with torch.no_grad():
        image_embeds = unwrap_visual_output(vision_forward(model, pv, gthw))
    num_images = gthw.shape[0]
    text_embeds_llm = instr.mean(dim=1).expand(num_images, -1)
    text_embeds_seq_llm = instr.expand(num_images, -1, -1)

    def prune():
        return pruner(image_embeds, text_embeds_llm, text_embeds_seq_llm, gthw)

    for _ in range(cfg["warmup"]):
        prune()
    torch.cuda.synchronize()

    # ---- pruning: total time + per-stage CUDA-event breakdown ------------
    prune_times = []
    stage_acc = {k: [] for k in ALL_STAGE_ORDER}
    pruned_embeds = sizes = None
    for _ in range(cfg["repeat"]):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        pruned_embeds, sizes = prune()
        e.record()
        torch.cuda.synchronize()
        prune_times.append(s.elapsed_time(e))
        for k, v in pruner.last_timing.items():
            if k in stage_acc:
                stage_acc[k].append(v)

    stages = {}
    for k, vals in stage_acc.items():
        stages[k] = {
            "mean": float(np.mean(vals)) if vals else 0.0,
            "std": float(np.std(vals)) if vals else 0.0,
        }
    stages["total"] = {
        "mean": float(np.mean(prune_times)),
        "std": float(np.std(prune_times)),
    }

    # ---- pruned LLM prefill ---------------------------------------------
    inputs_embeds, attention_mask = model._build_pruned_inputs(
        inputs["input_ids"], inputs["attention_mask"], pruned_embeds, sizes
    )
    seq_len = int(inputs_embeds.shape[1])
    n_kept = int(sum(sizes))

    prefill_mean, prefill_std = time_callable(
        lambda: llm_forward(model, inputs_embeds, attention_mask),
        cfg["repeat"],
        cfg["warmup"],
    )

    # ---- end-to-end generate --------------------------------------------
    gen_lens, gen_times = [], []
    for _ in range(cfg["gen_repeat"]):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = do_generate(
            model, inputs_embeds=inputs_embeds, attention_mask=attention_mask
        )
        e.record()
        torch.cuda.synchronize()
        gen_times.append(s.elapsed_time(e))
        # generate() with inputs_embeds returns ONLY the new tokens.
        gen_lens.append(int(out.shape[1]))

    torch.cuda.reset_peak_memory_stats()
    llm_forward(model, inputs_embeds, attention_mask)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2

    return {
        "visual_tokens_in": n_vis_in,
        "visual_tokens_kept": n_kept,
        "seq_len": seq_len,
        "prefill_ms_mean": prefill_mean,
        "prefill_ms_std": prefill_std,
        "prefill_total_ms_mean": prefill_mean + vision_mean + stages["total"]["mean"],
        "prefill_total_ms_std": float("nan"),
        "vision_ms_mean": vision_mean,
        "vision_ms_std": vision_std,
        "e2e_ms_mean": float(np.mean(gen_times)),
        "e2e_ms_std": float(np.std(gen_times)),
        "gen_tokens": float(np.mean(gen_lens)),
        "flops_g": analytic_prefill_flops_g(model.model, seq_len),
        "peak_mem_mb": peak,
        "stages": stages,
        "prune_ms_mean": float(np.mean(prune_times)),
        "prune_ms_std": float(np.std(prune_times)),
    }


def profile_model(model_name, datasets, cfg, is_eadp):
    model = load_model(model_name, max_new_tokens=cfg["max_new_tokens"])
    if is_eadp:
        attach_pruner(model, selector="facility", capture=False)

    results = {}
    for ds_name in datasets:
        dataset = build_dataset(ds_name)
        model.set_dump_image(dataset.dump_image)
        idx = sample_indices(len(dataset.data), cfg["per_dataset"], offset=cfg["offset"])
        messages = [
            common.build_message(model, dataset, ds_name, dataset.data.iloc[i]) for i in idx
        ]
        # warmup at dataset granularity (excluded from stats)
        for m in messages[: min(3, len(messages))]:
            with torch.no_grad():
                model.generate(message=m, dataset=ds_name)
        torch.cuda.synchronize()

        per_sample = []
        for m in tqdm(messages, desc=f"{model_name} | {ds_name}", leave=False):
            try:
                fn = run_eadp_sample if is_eadp else run_baseline_sample
                per_sample.append(fn(model, m, ds_name, cfg))
            except Exception:
                traceback.print_exc()

        if not per_sample:
            continue

        agg = {"n": len(per_sample), "dataset": ds_name, "model": model_name}
        for key in (
            "visual_tokens_in",
            "visual_tokens_kept",
            "seq_len",
            "prefill_ms_mean",
            "prefill_total_ms_mean",
            "vision_ms_mean",
            "e2e_ms_mean",
            "gen_tokens",
            "flops_g",
            "peak_mem_mb",
            "prune_ms_mean",
        ):
            vals = [p[key] for p in per_sample]
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
        # per-stage pruning breakdown (mean over samples of the per-sample mean)
        stage_names = set()
        for p in per_sample:
            stage_names.update(p["stages"])
        agg["stages"] = {}
        for st in sorted(stage_names, key=lambda x: (x not in ALL_STAGE_ORDER, x)):
            means = [p["stages"][st]["mean"] for p in per_sample if st in p["stages"]]
            stds = [p["stages"][st]["std"] for p in per_sample if st in p["stages"]]
            if means:
                agg["stages"][st] = {
                    "mean": float(np.mean(means)),
                    "std": float(np.mean(stds)),
                }
        results[ds_name] = agg
        print(json.dumps(agg, indent=2))

    free_model(model)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["TextVQA_VAL", "DocVQA_VAL", "OCRBench"])
    ap.add_argument("--per-dataset", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--budgets", nargs="+", type=int, default=[128, 256, 512])
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--gen-repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = {
        "per_dataset": args.per_dataset,
        "offset": args.offset,
        "repeat": args.repeat,
        "gen_repeat": args.gen_repeat,
        "warmup": args.warmup,
        "max_new_tokens": args.max_new_tokens,
    }
    out_path = args.out or os.path.join(ensure_out_dir(), "efficiency_profile.json")

    all_out = {"config": cfg, "alpha": args.alpha, "beta": args.beta, "models": {}}

    jobs = [("baseline", BASELINE_MODEL, False)] + [
        (f"eadp{t}", eadp_model_name(t, args.alpha, args.beta), True) for t in args.budgets
    ]
    for label, model_name, is_eadp in jobs:
        print(f"\n{'='*70}\n### {label}: {model_name}\n{'='*70}", flush=True)
        try:
            all_out["models"][label] = profile_model(
                model_name, args.datasets, cfg, is_eadp
            )
        except Exception:
            traceback.print_exc()
            all_out["models"][label] = {"error": traceback.format_exc()}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_out, f, indent=2)
        print(f"[saved] {out_path}", flush=True)

    print(f"\nDone -> {out_path}")


if __name__ == "__main__":
    main()
