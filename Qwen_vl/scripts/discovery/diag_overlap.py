#!/usr/bin/env python3
"""
Cheap structural diagnostic: how *different* are the candidate selectors?

No generation involved -- one vision-tower forward per instance, then every
selector is applied to the identical importance map. Reports:

  * pairwise IoU between the selected token sets
  * the facility-location objective value F(S) each selection achieves
    (so we can see what the extra latency is actually buying in-objective)
  * importance-rank statistics of the selected tokens
  * the distribution of the rebound similarity matrix, to quantify how much
    dynamic range 0.5*(cos+1) leaves for the coverage gain

Usage:
  python scripts/discovery/diag_overlap.py --per-dataset 200 --budget 256
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

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
    free_model,
    load_model,
    sample_indices,
)
from instrumented import SELECTORS, TimedEADPPruner  # noqa: E402
from vlmeval.vlm.qwen3_vl.model_fixed_res import unwrap_visual_output  # noqa: E402


def fl_objective(importance, sim, sel_idx):
    """F(S) = sum_j s_j * max_{i in S} Sim(i, j)  -- the paper's Eq. for F(S)."""
    sub = sim[sel_idx]                      # (T, N)
    return float((importance * sub.max(dim=0).values).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["TextVQA_VAL", "DocVQA_VAL", "OCRBench"])
    ap.add_argument("--per-dataset", type=int, default=200)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--selectors", nargs="+", default=list(SELECTORS.keys()))
    ap.add_argument("--sim-mode", default="rebound")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        ensure_out_dir(), f"diag_overlap_b{args.budget}_{args.sim_mode}.json"
    )

    model = load_model(eadp_model_name(args.budget, args.alpha, args.beta))
    pruner = TimedEADPPruner(
        visual_token_num=args.budget,
        alpha=args.alpha,
        beta=args.beta,
        visual_dim=3584,  # Qwen3-VL-8B LLM hidden size; unused by the selection path
        spatial_merge_size=2,
        selector="facility",
        sim_mode=args.sim_mode,
        capture=True,
    ).to(next(model.model.parameters()).device)
    pruner.eval()

    per_ds = {}
    for ds_name in args.datasets:
        dataset = build_dataset(ds_name)
        model.set_dump_image(dataset.dump_image)
        idx = sample_indices(len(dataset.data), args.per_dataset)
        records = []

        for i in tqdm(idx, desc=f"overlap {ds_name}", leave=False):
            msg = build_message(model, dataset, ds_name, dataset.data.iloc[i])
            instr = model._get_instruction_sequence_embedding(msg, dataset=ds_name)
            inputs = model._processor_inputs(model._build_messages(msg, dataset=ds_name))
            pv = inputs["pixel_values"].type(model.model.visual.dtype)
            gthw = inputs["image_grid_thw"]
            with torch.no_grad():
                feats = unwrap_visual_output(model.model.visual(pv, grid_thw=gthw))
                pruner(
                    feats,
                    instr.mean(dim=1).expand(gthw.shape[0], -1),
                    instr.expand(gthw.shape[0], -1, -1),
                    gthw,
                )
            cap = pruner.last_capture
            imp = cap["importance"][0]
            sim = cap["sim_matrix"][0]
            N = imp.shape[0]

            sels, objs = {}, {}
            for name in args.selectors:
                si, _ = SELECTORS[name](imp.unsqueeze(0), sim.unsqueeze(0), args.budget)
                si = si[0]
                sels[name] = si
                objs[name] = fl_objective(imp, sim, si)

            # pairwise IoU
            iou = {}
            for a, b in itertools.combinations(args.selectors, 2):
                sa = set(sels[a].tolist())
                sb = set(sels[b].tolist())
                iou[f"{a}|{b}"] = len(sa & sb) / len(sa | sb)

            order = torch.argsort(imp, descending=True)
            rank = torch.empty(N, dtype=torch.long)
            rank[order] = torch.arange(N)
            pct = {name: float((rank[sels[name]].float() / N).mean()) for name in args.selectors}

            off = sim[~torch.eye(N, dtype=torch.bool, device=sim.device)]
            records.append({
                "index": int(i),
                "iou": iou,
                "objective": objs,
                "mean_rank_pct": pct,
                "sim": {
                    "min": float(off.min()), "max": float(off.max()),
                    "mean": float(off.mean()), "std": float(off.std()),
                    "p05": float(torch.quantile(off.float(), 0.05)),
                    "p95": float(torch.quantile(off.float(), 0.95)),
                },
            })

        keys = [f"{a}|{b}" for a, b in itertools.combinations(args.selectors, 2)]
        per_ds[ds_name] = {
            "n": len(records),
            "iou_mean": {k: float(np.mean([r["iou"][k] for r in records])) for k in keys},
            "objective_mean": {
                s: float(np.mean([r["objective"][s] for r in records])) for s in args.selectors
            },
            "mean_rank_pct": {
                s: float(np.mean([r["mean_rank_pct"][s] for r in records])) for s in args.selectors
            },
            "sim_stats": {
                k: float(np.mean([r["sim"][k] for r in records]))
                for k in ("min", "max", "mean", "std", "p05", "p95")
            },
            "records": records,
        }
        print(f"\n### {ds_name}  (n={len(records)})")
        print("  IoU:")
        for k in keys:
            print(f"    {k:34s} {per_ds[ds_name]['iou_mean'][k]:.3f}")
        print("  F(S) objective:")
        for s in args.selectors:
            print(f"    {s:16s} {per_ds[ds_name]['objective_mean'][s]:10.4f}")
        print(f"  sim stats: {per_ds[ds_name]['sim_stats']}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "datasets": per_ds}, f, indent=2)
    print(f"\n[saved] {out_path}")
    free_model(model)


if __name__ == "__main__":
    main()
