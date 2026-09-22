#!/usr/bin/env python3
"""
Verify that the `clamp` similarity kernel actually changes the selection.

The clamp run produced accuracies identical to `rebound` on all three
benchmarks (64.38 / 66.00 / 70.27), which is a strong enough coincidence that
it needs checking rather than trusting: either the kernel genuinely does not
matter downstream, or the sim_mode flag silently had no effect.

This compares the selected token sets directly, with no generation involved.
"""

from __future__ import annotations

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


def main():
    datasets = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]
    per_ds = 60
    budget = 256

    model = load_model(eadp_model_name(budget))
    pruner = TimedEADPPruner(
        visual_token_num=budget, alpha=0.5, beta=2.0, visual_dim=3584,
        spatial_merge_size=2, selector="facility", sim_mode="rebound", capture=True,
    ).to(next(model.model.parameters()).device)
    pruner.eval()

    report = {}
    for ds_name in datasets:
        dataset = build_dataset(ds_name)
        model.set_dump_image(dataset.dump_image)
        idx = sample_indices(len(dataset.data), per_ds)
        ious, same_imp, ranges = [], [], []

        for i in tqdm(idx, desc=f"clamp-verify {ds_name}", leave=False):
            msg = build_message(model, dataset, ds_name, dataset.data.iloc[i])
            instr = model._get_instruction_sequence_embedding(msg, dataset=ds_name)
            inputs = model._processor_inputs(model._build_messages(msg, dataset=ds_name))
            pv = inputs["pixel_values"].type(model.model.visual.dtype)
            gthw = inputs["image_grid_thw"]
            with torch.no_grad():
                feats = unwrap_visual_output(model.model.visual(pv, grid_thw=gthw))

            sels = {}
            imp_ref = None
            for mode in ("rebound", "clamp"):
                pruner.sim_mode = mode
                with torch.no_grad():
                    pruner(
                        feats,
                        instr.mean(dim=1).expand(gthw.shape[0], -1),
                        instr.expand(gthw.shape[0], -1, -1),
                        gthw,
                    )
                cap = pruner.last_capture
                # capture stores CPU copies for dumping; selectors need GPU
                imp = cap["importance"][0].to(feats.device)
                sim = pruner._similarity(feats.unsqueeze(0))[0]
                si, _ = SELECTORS["facility"](imp.unsqueeze(0), sim.unsqueeze(0), budget)
                sels[mode] = set(si[0].tolist())
                if imp_ref is None:
                    imp_ref = imp
                else:
                    same_imp.append(bool(torch.equal(imp_ref, imp)))
                off = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]
                ranges.append((mode, float(off.mean()), float(off.std())))

            a, b = sels["rebound"], sels["clamp"]
            ious.append(len(a & b) / len(a | b))

        rng = {}
        for mode, m, s in ranges:
            rng.setdefault(mode, []).append((m, s))
        report[ds_name] = {
            "n": len(ious),
            "mean_iou_rebound_vs_clamp": float(np.mean(ious)),
            "min_iou": float(np.min(ious)),
            "max_iou": float(np.max(ious)),
            "identical_selection_pct": float(100 * np.mean([x == 1.0 for x in ious])),
            "importance_map_identical_pct": float(100 * np.mean(same_imp)),
            "sim_rebound": [float(np.mean([x[0] for x in rng["rebound"]])),
                            float(np.mean([x[1] for x in rng["rebound"]]))],
            "sim_clamp": [float(np.mean([x[0] for x in rng["clamp"]])),
                          float(np.mean([x[1] for x in rng["clamp"]]))],
        }
        print(f"\n### {ds_name} (n={len(ious)})")
        print(f"  importance map identical to rebound : {report[ds_name]['importance_map_identical_pct']:.1f}%")
        print(f"  selection IoU rebound vs clamp      : {report[ds_name]['mean_iou_rebound_vs_clamp']:.4f} "
              f"(min {report[ds_name]['min_iou']:.3f}, max {report[ds_name]['max_iou']:.3f})")
        print(f"  selections exactly identical        : {report[ds_name]['identical_selection_pct']:.1f}%")
        print(f"  similarity mean/std rebound         : {report[ds_name]['sim_rebound']}")
        print(f"  similarity mean/std clamp           : {report[ds_name]['sim_clamp']}")

    out = os.path.join(ensure_out_dir(), "clamp_verification.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[saved] {out}")
    free_model(model)


if __name__ == "__main__":
    main()
