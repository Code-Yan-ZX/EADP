#!/usr/bin/env python3
"""
Render the saved EADP intermediates for representative failure cases.

For each (dataset, index) with a saved npz under outputs/discovery/fa_maps/,
produce a PNG showing:
    image | retained-token mask | global relevance | dense (denoised) relevance
    | fused | smoothed | polarized | selected mask

Also prints, for that instance, the text-token entropy table so we can see which
instruction tokens the entropy filter discarded.

Usage:
  python scripts/discovery/render_maps.py --cases TextVQA_VAL:34602 ...
  python scripts/discovery/render_maps.py --from-classification --class B_selector_dropped --n 4
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MAPS = os.path.join(common.OUTPUT_DIR, "fa_maps")
FIGS = os.path.join(common.OUTPUT_DIR, "figures")
GRID = 32


def load_case(ds, index):
    p = os.path.join(MAPS, f"{ds}_{index}_b256.npz")
    return np.load(p) if os.path.exists(p) else None


def render(ds, index, out_dir=FIGS, question="", answer="", pred=""):
    z = load_case(ds, index)
    if z is None:
        print(f"[skip] no maps for {ds}:{index}")
        return None
    os.makedirs(out_dir, exist_ok=True)

    sel = z["select_idx"][0] if z["select_idx"].ndim > 1 else z["select_idx"]
    mask = np.zeros(GRID * GRID)
    mask[sel] = 1.0

    panels = [
        ("global relevance", z["global_sim"][0, :, 0]),
        ("dense (denoised)", z["local_sim"][0, :, 0]),
        ("fused importance", z["fused"][0]),
        ("post-smoothing", z["importance_post_smooth"][0]),
        ("post-polarization (beta=2)", z["importance"][0]),
        ("selected mask (256)", mask),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
    for ax, (title, vec) in zip(axes.ravel(), panels):
        im = ax.imshow(vec.reshape(GRID, GRID), cmap="viridis", interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f"{ds}:{index}\nQ: {question[:110]}\nGT: {answer[:70]}   |   EADP-256 pred: {pred[:70]}",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(out_dir, f"{ds}_{index}_maps.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)

    ent = z["text_entropy"][0, 0]
    keep = max(1, int(len(ent) * 0.2))
    order = np.argsort(ent)
    print(f"\n{ds}:{index}  text-token entropy (kept = lowest {keep}/{len(ent)}):")
    for k, j in enumerate(order):
        flag = "KEEP" if k < keep else "drop"
        print(f"   [{flag}] token#{j:3d}  H={ent[j]:.4f}")
    print(f"   saved -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=[])
    ap.add_argument("--from-classification", action="store_true")
    ap.add_argument("--class", dest="cls", default=None)
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    todo = []
    for c in args.cases:
        ds, idx = c.rsplit(":", 1)
        todo.append((ds, int(idx), "", "", ""))

    if args.from_classification:
        path = os.path.join(common.OUTPUT_DIR, "fa_classification.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows = data["rows"]
        if args.cls:
            rows = [r for r in rows if r["class"] == args.cls]
        seen = {}
        for r in rows:
            seen.setdefault(r["dataset"], []).append(r)
        for ds, rs in seen.items():
            for r in rs[: max(1, args.n // max(1, len(seen)))]:
                todo.append((ds, r["index"], r["question"], r["answer"],
                             r.get("pred_eadp256") or ""))

    for ds, idx, q, a, p in todo:
        render(ds, idx, question=q, answer=a, pred=p)


if __name__ == "__main__":
    main()
