#!/usr/bin/env python3
"""
CPU-only: where is the EADP-256 error mass concentrated?

Uses the official full-split prediction files (no GPU). Breaks the error rate
down by OCRBench category and DocVQA question type, so Part 4 can target the
capability that pruning actually damages rather than a generic "accuracy".
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
from scoring import per_sample_hits  # noqa: E402
from failure_analysis import OFFICIAL_PREDS  # noqa: E402
from vlmeval.smp import load  # noqa: E402


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    results = {}

    for ds, path in OFFICIAL_PREDS.items():
        if not os.path.exists(path):
            print(f"[miss] {ds} {path}")
            continue
        data = load(path)
        rows = [data.iloc[i] for i in range(len(data))]
        preds = [r["prediction"] for r in rows]
        hits = per_sample_hits(ds, rows, preds)
        df = pd.DataFrame(data)
        df["hit"] = hits
        df["err"] = (hits < 1.0).astype(int)
        df["hard_err"] = (hits < 0.5).astype(int)

        print(f"\n### {ds}  n={len(df)}  mean_hit={hits.mean()*100:.2f}")
        print(f"    not-fully-correct (hit<1.0): {df['err'].mean()*100:.1f}%   "
              f"hard errors (hit<0.5): {df['hard_err'].mean()*100:.1f}%")

        for col in ("category", "question_types"):
            if col not in df.columns:
                continue
            print(f"\n  by {col}:")
            g = df.groupby(col).agg(n=("hit", "size"), mean_hit=("hit", "mean"),
                                    hard_err=("hard_err", "mean"))
            g = g.sort_values("n", ascending=False)
            rows_out = [
                [str(idx), int(r.n), f"{r.mean_hit*100:.1f}", f"{r.hard_err*100:.1f}"]
                for idx, r in g.iterrows()
            ]
            print(md_table([col, "n", "mean hit x100", "% hard errors"], rows_out))
            results[f"{ds}:{col}"] = g.reset_index().to_dict("records")

        # answer-length bucket for TextVQA-style short answers
        if "answer" in df.columns:
            def first_len(a):
                try:
                    import ast
                    v = a if isinstance(a, list) else ast.literal_eval(str(a))
                    return len(str(v[0])) if v else 0
                except Exception:
                    return len(str(a))
            df["ans_len"] = df["answer"].map(first_len)
            df["len_bucket"] = pd.cut(df["ans_len"], [-1, 3, 8, 20, 1000],
                                      labels=["1-3", "4-8", "9-20", "21+"])
            g = df.groupby("len_bucket", observed=True).agg(
                n=("hit", "size"), mean_hit=("hit", "mean"), hard_err=("hard_err", "mean"))
            print(f"\n  by ground-truth answer length:")
            print(md_table(["chars", "n", "mean hit x100", "% hard errors"],
                           [[str(i), int(r.n), f"{r.mean_hit*100:.1f}", f"{r.hard_err*100:.1f}"]
                            for i, r in g.iterrows()]))

        # question length
        if "question" in df.columns:
            df["q_len"] = df["question"].astype(str).str.split().str.len()
            df["q_bucket"] = pd.cut(df["q_len"], [0, 6, 10, 16, 1000],
                                    labels=["<=6", "7-10", "11-16", "17+"])
            g = df.groupby("q_bucket", observed=True).agg(
                n=("hit", "size"), mean_hit=("hit", "mean"), hard_err=("hard_err", "mean"))
            print(f"\n  by question length (words):")
            print(md_table(["words", "n", "mean hit x100", "% hard errors"],
                           [[str(i), int(r.n), f"{r.mean_hit*100:.1f}", f"{r.hard_err*100:.1f}"]
                            for i, r in g.iterrows()]))

    out = os.path.join(common.ensure_out_dir(), "error_composition.json")
    import json
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
