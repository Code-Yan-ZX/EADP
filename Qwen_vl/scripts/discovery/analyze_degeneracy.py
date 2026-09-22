#!/usr/bin/env python3
"""
Quantify generation degeneracy under aggressive pruning (observation O-C).

Part 1 showed EADP-128's DocVQA end-to-end latency exceeding the *unpruned*
baseline, driven by a generation-length mean of 45.8 tokens with std 286. The
selector runs save every raw prediction, so we can measure the degeneracy rate
directly instead of inferring it from a mean and a std.

A prediction is called degenerate if its word count is far above the norm for
that (dataset, budget) - the practical signature of a repetition loop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

FILES = [
    "diag_selectors_b256_rebound.json",
    "diag_selectors_b128_rebound.json",
    "diag_selectors_b256_clamp.json",
    "diag_selectors_block.json",
]


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def word_count(s):
    return len(str(s).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long-words", type=int, default=60,
                    help="word count above which a prediction counts as degenerate")
    args = ap.parse_args()

    rows = []
    for fname in FILES:
        p = os.path.join(common.OUTPUT_DIR, fname)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        for v in data["runs"].values():
            preds = v.get("predictions") or []
            if not preds:
                continue
            lens = np.array([word_count(x) for x in preds], dtype=float)
            deg = int((lens > args.long_words).sum())
            rows.append({
                "budget": v["budget"],
                "sim": v.get("sim_mode", "rebound"),
                "selector": v["selector"],
                "dataset": v["dataset"],
                "n": len(lens),
                "acc": v["acc_pct"],
                "mean_words": float(lens.mean()),
                "median_words": float(np.median(lens)),
                "p95_words": float(np.percentile(lens, 95)),
                "max_words": float(lens.max()),
                "n_degen": deg,
                "degen_pct": 100.0 * deg / len(lens),
                "acc_excl_degen": float(
                    np.mean([h for h, l in zip(v["hits"], lens)
                             if l <= args.long_words]) * 100
                ) if deg else v["acc_pct"],
            })

    rows.sort(key=lambda r: (r["dataset"], r["budget"], r["selector"]))
    print(f"\n## Generation degeneracy (> {args.long_words} words = degenerate)\n")
    print(md_table(
        ["dataset", "budget", "sim", "selector", "n", "acc", "median w",
         "p95 w", "max w", "% degen", "acc excl degen"],
        [[r["dataset"], r["budget"], r["sim"], r["selector"], r["n"],
          f"{r['acc']:.2f}", f"{r['median_words']:.0f}", f"{r['p95_words']:.0f}",
          f"{r['max_words']:.0f}", f"{r['degen_pct']:.1f}", f"{r['acc_excl_degen']:.2f}"]
         for r in rows]))

    out = os.path.join(common.ensure_out_dir(), "degeneracy.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
