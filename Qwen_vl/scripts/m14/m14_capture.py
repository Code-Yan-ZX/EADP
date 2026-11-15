#!/usr/bin/env python3
"""
M14 PART 0 -- freeze the per-sample inputs.

One pass over the exact sample bank diag_selectors.py used for the +19.84-point
facility-vs-topk measurement (150 evenly spaced instances per benchmark,
offset 0).  For every sample we capture, with the model and the official
instrumented scoring path (alpha=0.5, beta=2.0, q=0.2, T=100):

    importance  post-smoothing, post-polarization score  (N,)      -- Stage 1
    sim_matrix  rebound kernel 0.5*(cos+1)               (N, N)    -- selector input
    vis         post-merger visual features              (N, 4096)

After this pass every PART 1-5 experiment is model-free: solvers are compared
on bit-identical inputs, and the reference selection can be recomputed offline.

Outputs (outputs/m14/):
    m14_capture.npz      key, ds, idx, importance (450, 1024) f32, meta
    m14_sim.f32.npy      (450, 1024, 1024) float32 memmap  (~1.9 GB)
    m14_vis.u16.npy      (450, 1024, 4096) uint16 == bf16 memmap (~3.8 GB)

Usage:
    python scripts/m14/m14_capture.py            # all 450
    python scripts/m14/m14_capture.py --limit 4  # smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "discovery"))

import m14_common as C  # noqa: E402
import common  # noqa: E402
from common import (build_dataset, build_message, eadp_model_name,  # noqa: E402
                    load_model, sample_indices)
from instrumented import attach_pruner  # noqa: E402
from vlmeval.vlm.qwen3_vl.model_fixed_res import unwrap_visual_output  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-dataset", type=int, default=150)
    args = ap.parse_args()

    model = load_model(eadp_model_name(C.BUDGET, 0.5, 2.0), max_new_tokens=64)
    model.model.eval()
    torch.set_grad_enabled(False)
    pruner = attach_pruner(model, selector="facility", capture=False)
    pruner.visual_token_num = C.BUDGET
    pruner.sim_mode = C.SIM_MODE
    pruner.keep_gpu = True

    keys, dss, idxs, imps = [], [], [], []
    n_total = args.per_dataset * len(C.DS_ORDER)
    if args.limit:
        n_total = min(n_total, args.limit)
    sim_mm = np.lib.format.open_memmap(
        os.path.join(C.M14_OUT, C.CAPTURE_SIM), mode="w+",
        dtype=np.float32, shape=(n_total, C.N_VIS, C.N_VIS))
    vis_mm = np.lib.format.open_memmap(
        os.path.join(C.M14_OUT, C.CAPTURE_VIS), mode="w+",
        dtype=np.uint16, shape=(n_total, C.N_VIS, 4096))

    i = 0
    for ds_name in C.DS_ORDER:
        dataset = build_dataset(ds_name)
        model.set_dump_image(dataset.dump_image)
        idx = sample_indices(len(dataset.data), args.per_dataset, offset=0)
        rows = [dataset.data.iloc[j] for j in idx]
        if args.limit:
            rows = rows[:max(0, n_total - i)]
        for per_ds_i, row in enumerate(tqdm(rows, desc=ds_name)):
            message = build_message(model, dataset, ds_name, row)
            instr = model._get_instruction_sequence_embedding(message, dataset=ds_name)
            inputs = model._processor_inputs(
                model._build_messages(message, dataset=ds_name))
            pv = inputs["pixel_values"].type(model.model.visual.dtype)
            gthw = inputs["image_grid_thw"]
            image_embeds = unwrap_visual_output(
                model.model.visual(pv, grid_thw=gthw))
            num_images = gthw.shape[0]
            text_embeds_llm = instr.mean(dim=1).expand(num_images, -1)
            text_embeds_seq_llm = instr.expand(num_images, -1, -1)

            pruner.last_gpu.clear()
            model.pruner(image_embeds, text_embeds_llm, text_embeds_seq_llm, gthw)
            imp = pruner.last_gpu["importance"][0].float().cpu().numpy()
            sim = pruner.last_gpu["sim_matrix"][0].float().cpu().numpy()
            vis_u16 = pruner.last_gpu["image_features"].view(torch.uint16).cpu().numpy()

            assert imp.shape == (C.N_VIS,) and sim.shape == (C.N_VIS, C.N_VIS)
            keys.append(f"{ds_name}:{idx[per_ds_i]}")
            dss.append(ds_name)
            idxs.append(int(idx[per_ds_i]))
            imps.append(imp.astype(np.float32))
            sim_mm[i] = sim.astype(np.float32)
            vis_mm[i] = vis_u16
            i += 1
        if i >= n_total:
            break

    sim_mm.flush()
    vis_mm.flush()
    np.savez(os.path.join(C.M14_OUT, C.CAPTURE_NPY),
             key=np.array(keys), ds=np.array(dss), idx=np.array(idxs, dtype=np.int64),
             importance=np.stack(imps),
             meta=np.array(json.dumps({
                 "budget": C.BUDGET, "sim_mode": C.SIM_MODE,
                 "alpha": 0.5, "beta": 2.0, "per_dataset": args.per_dataset,
                 "sample_rule": "evenly spaced, offset 0 (diag_selectors bank)",
                 "model": eadp_model_name(C.BUDGET, 0.5, 2.0),
             })))
    print(f"[saved] {i} samples -> {C.M14_OUT}")


if __name__ == "__main__":
    main()
