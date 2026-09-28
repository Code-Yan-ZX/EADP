#!/usr/bin/env python3
"""
Paired diagnostic: is the MosaicPrune accuracy drop caused by POOLING
(off-manifold mean-pooled embeddings) or by the PARTITION signal?

Same N DocVQA samples, same wrapper/pathway, four arms:

  full-1024      no compression (upper reference on this pathway)
  rand-sel-256   random 256-token SELECTION (original embeddings, EADP-style)
  uniform-256    2x2 mean POOLING
  disp-256       dispersion quadtree POOLING

Scores with the exact DocVQA aggregation VLMEvalKit uses
(process_line 'anls' + hit_calculate). Runs alongside the main sweep; results
to outputs/discovery/mosaic/diag_pooling.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
from common import collect_samples, free_model, load_model  # noqa: E402
from model.mosaic import build_partition, pool_leaves  # noqa: E402
from vlmeval.dataset.utils.vqa_eval import hit_calculate, process_line  # noqa: E402


def build_pruned(model, inputs, pruned_embeds, sizes):
    return model._build_pruned_inputs(
        inputs["input_ids"], inputs["attention_mask"],
        pruned_embeds, sizes)


@torch.no_grad()
def generate(model, inputs_embeds, attention_mask):
    out = model.model.generate(
        inputs_embeds=inputs_embeds, attention_mask=attention_mask,
        do_sample=False, **model.generate_kwargs)
    return model.processor.tokenizer.batch_decode(
        out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--dataset", default="DocVQA_VAL")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = load_model("Qwen3-VL-8B-Mosaic-256-uniform", max_new_tokens=64)
    dataset, idxs, rows, messages = collect_samples(
        model, args.dataset, args.n, offset=args.offset)

    results = {arm: [] for arm in ("full-1024", "rand-sel-256",
                                   "uniform-256", "disp-256")}
    gts = []

    for row_i, (row, message) in enumerate(zip(rows, messages)):
        try:
            msgs = model._build_messages(message, dataset=args.dataset)
            inputs = model._processor_inputs(msgs)
            gthw = inputs["image_grid_thw"]
            pv = inputs["pixel_values"].type(model.model.visual.dtype)
            with torch.no_grad():
                from vlmeval.vlm.qwen3_vl.model_fixed_res import (
                    unwrap_visual_output)
                feats = unwrap_visual_output(
                    model.model.visual(pv, grid_thw=gthw))
            gh = int(gthw[0, 1]) // 2
            gw = int(gthw[0, 2]) // 2

            arms = {}
            arms["full-1024"] = (feats, [feats.shape[0]])

            g = torch.Generator(device="cpu").manual_seed(args.seed + row_i)
            sel = torch.randperm(feats.shape[0], generator=g)[:256].sort().values
            arms["rand-sel-256"] = (feats[sel], [256])

            leaves_u = build_partition(feats, gh, gw, 256, "uniform")
            arms["uniform-256"] = (pool_leaves(feats, gh, gw, leaves_u), [256])
            leaves_d = build_partition(feats, gh, gw, 256, "dispersion", seed=0)
            arms["disp-256"] = (pool_leaves(feats, gh, gw, leaves_d), [256])

            for arm, (emb, sizes) in arms.items():
                ie, am = build_pruned(model, inputs, emb, sizes)
                pred = generate(model, ie, am)
                results[arm].append(pred)
            gt = row["answer"]
            gts.append(gt if isinstance(gt, str) else str(gt))
            if (row_i + 1) % 10 == 0:
                print(f"  {row_i+1}/{len(rows)} samples done", flush=True)
        except Exception as err:  # noqa: BLE001
            print(f"  sample {row_i} failed: {err}", flush=True)

    scores = {}
    per_sample = {}
    for arm, preds in results.items():
        lines = [{"answer": gts[i], "prediction": preds[i]}
                 for i in range(len(preds))]
        parsed = [process_line(l, method="anls") for l in lines]
        hits = hit_calculate(parsed, "DocVQA")
        scores[arm] = float(100.0 * np_mean(hits))
        per_sample[arm] = [float(x) for x in hits]

    out_path = os.path.join(common.OUTPUT_DIR, "mosaic", "diag_pooling.json")
    with open(out_path, "w") as f:
        json.dump({"dataset": args.dataset, "n": len(gts), "offset": args.offset,
                   "scores": scores, "per_sample": per_sample,
                   "answers": gts,
                   "samples": {arm: results[arm][:10] for arm in results}}, f,
                  indent=1, ensure_ascii=False)
    print(json.dumps(scores, indent=1))
    print(f"wrote {out_path}")
    free_model(model)


def np_mean(x):
    import numpy as np
    return np.mean(x)


if __name__ == "__main__":
    main()
