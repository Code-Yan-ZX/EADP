#!/usr/bin/env python3
"""
Sanity check: does the discovery scorer reproduce the published EADP baselines
when run on the official full-split prediction files?

Expected (from the official runs):
    TextVQA_VAL  Overall = 71.04
    DocVQA_VAL   Overall = 61.14  (ANLS)
    OCRBench     Final Score = 623
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
from scoring import per_sample_hits  # noqa: E402

from vlmeval.smp import load  # noqa: E402

EADP_DIR = os.path.join(
    common.QWEN_ROOT, "outputs", "eadp", "Qwen3-VL-8B-EADP-256-a0.5-b2.0"
)

TARGETS = {
    "TextVQA_VAL": (
        os.path.join(EADP_DIR, "T20260921_Ge1a08801",
                     "Qwen3-VL-8B-EADP-256-a0.5-b2.0_TextVQA_VAL.xlsx"),
        71.042,
    ),
    "DocVQA_VAL": (
        os.path.join(EADP_DIR, "T20260922_Ge1a08801",
                     "Qwen3-VL-8B-EADP-256-a0.5-b2.0_DocVQA_VAL.xlsx"),
        61.1356,
    ),
    "OCRBench": (
        os.path.join(EADP_DIR, "T20260921_Ge1a08801",
                     "Qwen3-VL-8B-EADP-256-a0.5-b2.0_OCRBench.xlsx"),
        623.0,
    ),
}


def main():
    ok = True
    for name, (path, expected) in TARGETS.items():
        if not os.path.exists(path):
            print(f"[MISS] {name}: {path}")
            ok = False
            continue
        data = load(path)
        rows = [data.iloc[i] for i in range(len(data))]
        preds = [r["prediction"] for r in rows]
        hits = per_sample_hits(name, rows, preds)

        if "OCRBench" in name:
            got = float(np.sum(hits))
            print(f"{name:14s} n={len(hits):5d}  Final Score = {got:.0f}  (official {expected:.0f})")
        else:
            got = float(np.mean(hits) * 100)
            print(f"{name:14s} n={len(hits):5d}  Overall = {got:.2f}  (official {expected:.2f})")
        if abs(got - expected) > 1.0:
            print(f"   -> MISMATCH")
            ok = False
    print("\nSCORER VALIDATED" if ok else "\nSCORER MISMATCH -- do not trust downstream numbers")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
