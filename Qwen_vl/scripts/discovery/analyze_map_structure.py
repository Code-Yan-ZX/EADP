#!/usr/bin/env python3
"""
Is there real spatial structure in EADP's relevance maps, or does the
smoothing/polarization stage manufacture it?

For each saved EADP-256 map we compute the lag-1 spatial autocorrelation
(horizontal and vertical) of the 32x32 relevance grid:

    r = corr( map[:, :-1], map[:, 1:] )

White noise gives r ~ 0. A genuinely localised relevance map gives r >> 0. If the
raw fused map has r ~ 0 while the post-smoothing map has r ~ 1, then the apparent
structure was introduced by the 3x3 Gaussian kernel and the subsequent min-max
renormalisation + polarization, not by the text-visual similarity itself.

Also reports the dynamic range of each stage, which shows how much of the
[0, 1] output range is an artefact of per-image min-max normalisation.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GRID = 32


def lag1_autocorr(vec):
    g = vec.reshape(GRID, GRID).astype(np.float64)
    out = []
    for a, b in ((g[:, :-1], g[:, 1:]), (g[:-1, :], g[1:, :])):
        x = a.ravel() - a.mean()
        y = b.ravel() - b.mean()
        denom = np.sqrt((x ** 2).sum() * (y ** 2).sum())
        out.append(float((x * y).sum() / denom) if denom > 0 else 0.0)
    return out


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


STAGES = [
    ("global_sim", "global relevance (raw)"),
    ("local_sim", "dense relevance (raw)"),
    ("fused", "fused (pre-smoothing)"),
    ("importance_post_smooth", "post-smoothing"),
    ("importance", "post-polarization"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps-dir",
                    default=os.path.join(common.OUTPUT_DIR, "fa_maps"))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.maps_dir, "*_b256.npz")))
    if not files:
        print(f"no maps in {args.maps_dir}")
        return

    acc = {name: {"ac": [], "rng": [], "std": []} for name, _ in STAGES}
    shuffled_ac = []

    for f in files:
        z = np.load(f)
        for name, _ in STAGES:
            if name not in z:
                continue
            v = np.asarray(z[name]).reshape(-1)[: GRID * GRID]
            if v.size != GRID * GRID:
                continue
            ac = lag1_autocorr(v)
            acc[name]["ac"].append(np.mean(ac))
            acc[name]["rng"].append(float(v.max() - v.min()))
            acc[name]["std"].append(float(v.std()))
            if name == "fused":
                shuffled_ac.append(np.mean(lag1_autocorr(np.random.permutation(v))))

    n = len(files)
    print(f"\n## Spatial structure of EADP relevance maps (n = {n} instances)\n")
    rows = []
    for name, label in STAGES:
        a = acc[name]
        if not a["ac"]:
            continue
        rows.append([
            label,
            f"{np.mean(a['ac']):+.4f}",
            f"{np.mean(a['rng']):.4f}",
            f"{np.mean(a['std']):.4f}",
        ])
    print(md_table(["stage", "lag-1 spatial autocorr",
                    "value range", "spatial std"], rows))
    print(f"\nShuffled null (same values, randomised positions): "
          f"lag-1 autocorr = {np.mean(shuffled_ac):+.4f}")

    # how much of the post-smoothing pattern is explained by the raw signal?
    raw = np.array(acc["fused"]["std"])
    sm = np.array(acc["importance_post_smooth"]["std"])
    print(f"\nmean spatial std: raw fused = {raw.mean():.4f}, "
          f"post-smoothing = {sm.mean():.4f}")

    out = os.path.join(common.ensure_out_dir(), "map_structure.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"n": n,
                   "stages": {k: {"ac_mean": float(np.mean(v["ac"])),
                                  "range_mean": float(np.mean(v["rng"])),
                                  "std_mean": float(np.mean(v["std"]))}
                              for k, v in acc.items() if v["ac"]},
                   "shuffled_null_ac": float(np.mean(shuffled_ac))}, fh, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
