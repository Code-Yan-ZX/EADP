#!/usr/bin/env python3
"""
MosaicPrune token statistics, offline.

The eval driver truncates stats_{mode}_{K}.jsonl at every model init, so
per-dataset partition statistics are collected here instead: run the
compressor on a deterministic sample of images per dataset (vision tower
forward only, no generation) and aggregate the leaf-size distributions per
mode / budget / dataset.

Usage:
  python scripts/discovery/mosaic_tokenstats.py \
      --datasets DocVQA_VAL OCRBench TextVQA_VAL --per-dataset 150
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
from common import collect_samples, free_model, load_model  # noqa: E402
from model.mosaic import leaf_statistics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["DocVQA_VAL", "OCRBench", "TextVQA_VAL"])
    ap.add_argument("--per-dataset", type=int, default=150)
    ap.add_argument("--budgets", nargs="+", type=int, default=[256, 128])
    ap.add_argument("--modes", nargs="+",
                    default=["uniform", "random", "dispersion"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or os.path.join(common.OUTPUT_DIR, "mosaic",
                                        "tokenstats.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # one model instance is enough; the compressor only needs the vision tower
    model = load_model("Qwen3-VL-8B-Mosaic-256-dispersion", max_new_tokens=8)
    visual = model.model.visual

    agg = defaultdict(lambda: defaultdict(list))
    for dataset_name in args.datasets:
        dataset, idxs, rows, messages = collect_samples(
            model, dataset_name, args.per_dataset, offset=0)
        for row, message in zip(rows, messages):
            try:
                inputs = model._processor_inputs(
                    model._build_messages(message, dataset=dataset_name))
            except Exception as err:  # noqa: BLE001
                print(f"  preprocess failed: {err}")
                continue
            gthw = inputs["image_grid_thw"]
            pv = inputs["pixel_values"].type(visual.dtype)
            with torch.no_grad():
                feats = visual(pv, grid_thw=gthw)
            for K in args.budgets:
                for mode in args.modes:
                    from model.mosaic import build_partition
                    split = (gthw.prod(-1) // 4).tolist()
                    rec_count = {}
                    t0 = time.perf_counter()
                    for f_i, f in enumerate(torch.split(feats, split, dim=0)):
                        t_i, h_i, w_i = gthw[f_i].tolist()
                        gh, gw = int(h_i) // 2, int(w_i) // 2
                        leaves = build_partition(f, gh, gw, min(K, gh * gw),
                                                 mode, seed=0)
                        rec_count = leaf_statistics(leaves)
                    ms = (time.perf_counter() - t0) * 1000.0
                    key = f"{dataset_name}|K{K}|{mode}"
                    for side, pct in rec_count["pct_by_side"].items():
                        agg[key]["side_" + side].append(pct)
                    for area, pct in rec_count["pct_by_area"].items():
                        agg[key][f"area_{area}"].append(pct)
                    agg[key]["compress_ms"].append(ms)

    out = {}
    for key, d in agg.items():
        out[key] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                    for k, v in d.items()}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"wrote {out_path}")
    free_model(model)


if __name__ == "__main__":
    main()
