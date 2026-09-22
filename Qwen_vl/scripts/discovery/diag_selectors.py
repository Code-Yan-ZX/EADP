#!/usr/bin/env python3
"""
Part 2 -- what does the facility-location selector actually buy us?

The importance-scoring half of the pipeline is held **fixed and identical**
across every arm (the instrumented pruner delegates scoring to the official
``model.pruner`` helpers). Only the final selection operator changes:

  facility       official EADP facility-location greedy          (A)
  facility_fast  same objective, vectorised gain computation      (A')
  lazy_greedy    same objective, CELF lazy evaluation             (A'')
  stochastic     same objective, stochastic greedy                (C3)
  topk           plain Top-K on the very same importance map      (B)
  topk_nms       Top-K + hard spatial exclusion radius            (C1)
  farthest       diversity-only farthest-point sampling           (C2)

Every selector sees the exact same samples, so the comparison is paired.

Usage:
  python scripts/discovery/diag_selectors.py \
      --datasets TextVQA_VAL DocVQA_VAL OCRBench --per-dataset 200 \
      --budgets 256 --selectors facility topk ...
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
    build_dataset,
    build_message,
    eadp_model_name,
    ensure_out_dir,
    load_model,
    sample_indices,
)
from instrumented import SELECTORS, attach_pruner, STAGE_SELECT  # noqa: E402
from scoring import per_sample_hits  # noqa: E402
from vlmeval.vlm.qwen3_vl.model_fixed_res import unwrap_visual_output  # noqa: E402


def generate_prediction(model, message, dataset_name):
    """Run the EADP path and return the decoded response."""
    instr = model._get_instruction_sequence_embedding(message, dataset=dataset_name)
    inputs = model._processor_inputs(model._build_messages(message, dataset=dataset_name))
    pv = inputs["pixel_values"].type(model.model.visual.dtype)
    gthw = inputs["image_grid_thw"]

    with torch.no_grad():
        image_embeds = unwrap_visual_output(model.model.visual(pv, grid_thw=gthw))
    num_images = gthw.shape[0]
    text_embeds_llm = instr.mean(dim=1).expand(num_images, -1)
    text_embeds_seq_llm = instr.expand(num_images, -1, -1)

    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    with torch.no_grad():
        pruned_embeds, sizes = model.pruner(
            image_embeds, text_embeds_llm, text_embeds_seq_llm, gthw
        )
    e.record()
    torch.cuda.synchronize()
    prune_ms = s.elapsed_time(e)
    select_ms = model.pruner.last_timing.get(STAGE_SELECT, float("nan"))

    inputs_embeds, attention_mask = model._build_pruned_inputs(
        inputs["input_ids"], inputs["attention_mask"], pruned_embeds, sizes
    )
    with torch.no_grad():
        out = model.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=False,
            **model.generate_kwargs,
        )
    text = model.processor.tokenizer.batch_decode(
        out, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    response = model._post_process_response(text)
    return {
        "prediction": response,
        "n_kept": int(sum(sizes)),
        "prune_ms": prune_ms,
        "select_ms": select_ms,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["TextVQA_VAL", "DocVQA_VAL", "OCRBench"])
    ap.add_argument("--per-dataset", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--budgets", nargs="+", type=int, default=[256])
    ap.add_argument("--selectors", nargs="+", default=["facility", "topk"])
    ap.add_argument("--sim-mode", default="rebound",
                    help="visual-visual similarity kernel for the selector: "
                         "rebound (official 0.5*(cos+1)) | clamp | exp | softmax")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="diag_selectors")
    args = ap.parse_args()

    for sel in args.selectors:
        if sel not in SELECTORS:
            raise SystemExit(f"unknown selector {sel!r}; have {sorted(SELECTORS)}")

    out_path = args.out or os.path.join(
        ensure_out_dir(),
        f"{args.tag}_b{'-'.join(map(str, args.budgets))}_{args.sim_mode}.json",
    )

    model_name = eadp_model_name(args.budgets[0], args.alpha, args.beta)
    model = load_model(model_name, max_new_tokens=args.max_new_tokens)
    pruner = attach_pruner(model, selector=args.selectors[0], capture=False)
    pruner.sim_mode = args.sim_mode

    # Pre-build the identical sample set every arm will see.
    sample_bank = {}
    for ds_name in args.datasets:
        dataset = build_dataset(ds_name)
        model.set_dump_image(dataset.dump_image)
        idx = sample_indices(len(dataset.data), args.per_dataset, offset=args.offset)
        rows = [dataset.data.iloc[i] for i in idx]
        msgs = [build_message(model, dataset, ds_name, r) for r in rows]
        sample_bank[ds_name] = {"dataset": dataset, "idx": idx, "rows": rows, "msgs": msgs}
        print(f"[samples] {ds_name}: {len(idx)} of {len(dataset.data)}")

    results = {"config": vars(args), "runs": {}}
    for budget in args.budgets:
        pruner.visual_token_num = budget
        for sel in args.selectors:
            pruner.selector_name = sel
            for ds_name in args.datasets:
                bank = sample_bank[ds_name]
                preds, meta = [], []
                for m in tqdm(
                    bank["msgs"], desc=f"b{budget} {sel} {ds_name}", leave=False
                ):
                    try:
                        r = generate_prediction(model, m, ds_name)
                    except Exception:
                        traceback.print_exc()
                        r = {"prediction": "", "n_kept": 0, "prune_ms": float("nan"),
                             "select_ms": float("nan")}
                    preds.append(r["prediction"])
                    meta.append(r)

                hits = per_sample_hits(ds_name, bank["rows"], preds)
                key = f"b{budget}|{sel}|{args.sim_mode}|{ds_name}"
                rec = {
                    "budget": budget,
                    "selector": sel,
                    "sim_mode": args.sim_mode,
                    "dataset": ds_name,
                    "n": int(len(hits)),
                    "acc_pct": float(np.mean(hits) * 100),
                    "sum_hits": float(np.sum(hits)),
                    "n_kept_mean": float(np.mean([m["n_kept"] for m in meta])),
                    "prune_ms_mean": float(np.nanmean([m["prune_ms"] for m in meta])),
                    "select_ms_mean": float(np.nanmean([m["select_ms"] for m in meta])),
                    "predictions": preds,
                    "hits": [float(h) for h in hits],
                    "idx": [int(i) for i in bank["idx"]],
                }
                results["runs"][key] = rec
                print(
                    f"{key:38s} acc={rec['acc_pct']:7.3f}  "
                    f"select={rec['select_ms_mean']:8.2f}ms  prune={rec['prune_ms_mean']:8.2f}ms"
                )
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

    print(f"\nDone -> {out_path}")


if __name__ == "__main__":
    main()
