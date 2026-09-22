"""
S2-B: break the error transitions down by the Stage-1 metadata categories.

Reuses the exact Stage-1 bucketing (`error_composition.py`):
  * OCRBench ``category`` column
  * DocVQA ``question_types`` column
  * ground-truth answer length -> [-1,3,8,20,1000] = 1-3 / 4-8 / 9-20 / 21+

Indirect mechanism check (brief §9): if gains concentrate on KIE / doc-oriented
VQA / digit strings, DocVQA table-list, and long answers, that is consistent with
the gradient sensitivity map finding dense information-bearing regions. Recorded
as an observation only — no mechanism is claimed.
"""
import argparse
import ast
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import sample_indices
from s1_audit import OUT

DS = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]
CORRECT = 0.5


def first_len(a):
    try:
        v = a if isinstance(a, list) else ast.literal_eval(str(a))
        return len(str(v[0])) if v else 0
    except Exception:
        return len(str(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="s2b_full.json")
    ap.add_argument("--arm", default=None, help="arm key; default = best by macro delta")
    ap.add_argument("--tag", default="s2b_full")
    args = ap.parse_args()

    off = {v["dataset"]: v for v in
           json.load(open(os.path.join(OUT, "diag_selectors_b256.json")))["runs"].values()
           if v["selector"] == "facility"}
    A = json.load(open(os.path.join(OUT, args.arms)))["runs"]
    recs = {}
    for k, v in A.items():          # run keys are "b<budget>|<sel>|<cal>|<dataset>"
        recs.setdefault(f"{v['selector']}|{v['calibration']}", {})[v["dataset"]] = v

    ana = json.load(open(os.path.join(OUT, f"{args.tag}_analysis.json")))
    arm = args.arm or max(ana["summary"], key=lambda k: ana["summary"][k]["macro_delta"])
    print(f"arm = {arm}\n")

    out = {}
    for ds in DS:
        dataset = common.build_dataset(ds)
        idx = sample_indices(len(dataset.data), 150, offset=0)
        rowmap = {int(i): dataset.data.iloc[i] for i in idx}

        v = recs[arm][ds]
        oi = {int(i): float(h) for i, h in zip(off[ds]["idx"], off[ds]["hits"])}
        rows = []
        for i, h in zip(v["idx"], v["hits"]):
            i = int(i)
            if i not in oi:
                continue
            r = rowmap[i]
            rows.append(dict(index=i, off=oi[i], arm=float(h),
                             category=str(r.get("category", "")),
                             qtypes=str(r.get("question_types", "")),
                             ans_len=first_len(r.get("answer", ""))))
        df = pd.DataFrame(rows)
        df["off_c"] = df["off"] >= CORRECT
        df["arm_c"] = df["arm"] >= CORRECT
        df["len_bucket"] = pd.cut(df["ans_len"], [-1, 3, 8, 20, 1000],
                                  labels=["1-3", "4-8", "9-20", "21+"])
        out[ds] = {}
        print(f"### {ds}  n={len(df)}")

        for col in ("category", "qtypes", "len_bucket"):
            if df[col].nunique() <= 1 and col != "len_bucket":
                continue
            print(f"\n  by {col}:")
            print(f"    {'bucket':26s}{'n':>5s}{'fixed':>7s}{'broken':>8s}"
                  f"{'net':>6s}{'Δacc':>9s}{'off_acc':>9s}{'arm_acc':>9s}")
            for b, g in df.groupby(col, observed=True):
                fixed = int((~g.off_c & g.arm_c).sum())
                broken = int((g.off_c & ~g.arm_c).sum())
                rec = dict(n=int(len(g)), fixed=fixed, broken=broken,
                           net=fixed - broken,
                           delta=float((g.arm - g.off).mean() * 100),
                           off_acc=float(g.off_c.mean() * 100),
                           arm_acc=float(g.arm_c.mean() * 100))
                out[ds][f"{col}:{b}"] = rec
                print(f"    {str(b):26s}{rec['n']:5d}{fixed:7d}{broken:8d}"
                      f"{rec['net']:+6d}{rec['delta']:+9.2f}{rec['off_acc']:9.1f}"
                      f"{rec['arm_acc']:9.1f}")
        print()

    json.dump({"arm": arm, "breakdown": out},
              open(os.path.join(OUT, f"{args.tag}_transitions.json"), "w"), indent=1)
    print(f"[saved] {os.path.join(OUT, f'{args.tag}_transitions.json')}")


if __name__ == "__main__":
    main()
