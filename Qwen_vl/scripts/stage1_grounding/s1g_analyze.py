"""Stage-1 grounding discovery — analysis (mirrors amp_analyze rules).

Accuracy = 100 x mean(per-question score) over the ACTUAL evaluated count;
macro = equal-weight mean over the three datasets; paired cluster bootstrap
(by image identity, within dataset) on the macro delta vs BASE, 95% CI.

BASE is read from the anchor-merge-pilot shards (same frozen manifest, same
pipeline: bank keep == online b1, identity gather — gate G2 must be PASS).

Usage:
  python s1g_analyze.py --split dev --arms A_G1,...,C_G1 --lam 0.5
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import s1g_common as SC
import amp_common as AC
import amp_analyze as AA
from amp_analyze import (acc100, macro_of, paired_macro_bootstrap,
                         rescue_break)


def load_scores(split, arms, tag=""):
    out = {}
    for arm in arms:
        for ds in SC.DS_LIST:
            if arm == "BASE":
                hits = sorted(glob.glob(os.path.join(
                    AC.OUT_DIR, "acc", split, "BASE", "K*",
                    f"{ds}_score.json")))
            else:
                hits = sorted(glob.glob(os.path.join(
                    SC.ACC_DIR, split, arm + (f"_{tag}" if tag else ""),
                    "K*", f"{ds}_score.json")))
            if not hits:
                continue
            s = json.load(open(hits[-1]))
            if not s.get("per_question"):
                continue
            out[(arm, ds)] = s
    return out


def run_quality(scores, arm, split, tag=""):
    out = {}
    for ds in SC.DS_LIST:
        if (arm, ds) not in scores:
            continue
        if arm == "BASE":
            hits = sorted(glob.glob(os.path.join(
                AC.OUT_DIR, "acc", split, "BASE", "K*", f"{ds}.json")))
        else:
            hits = sorted(glob.glob(os.path.join(
                SC.ACC_DIR, split, arm + (f"_{tag}" if tag else ""),
                "K*", f"{ds}.json")))
        if not hits:
            continue
        recs = list(json.load(open(hits[-1]))["records"].values())
        out[ds] = dict(
            n=len(recs),
            empty=sum(1 for r in recs if r.get("degeneracy", {}).get("empty")),
            truncated=sum(1 for r in recs if r.get("truncated")),
            degenerate_repeat=sum(1 for r in recs
                                  if r.get("degeneracy", {}).get("repeat4", 0)
                                  >= 10))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "confirm"])
    ap.add_argument("--arms", required=True)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    arms = args.arms.split(",")
    manifest = SC.load_manifest()
    scores = load_scores(args.split, arms, args.tag)

    table = {}
    for arm in arms:
        per = {ds: (acc100(scores[(arm, ds)]) if (arm, ds) in scores else None)
               for ds in SC.DS_LIST}
        table[arm] = dict(per_ds=per, macro=macro_of(scores, arm),
                          quality=run_quality(scores, arm, args.split, args.tag),
                          selector_ms=None)

    # paired bootstrap vs BASE + latency overhead (selector stage only)
    contrasts = {}
    for arm in arms:
        if arm == "BASE":
            continue
        c = paired_macro_bootstrap(scores, manifest, args.split, arm, "BASE")
        if c:
            rb = rescue_break(scores, arm, "BASE")
            c["rescued"] = sum(v["rescued"] for v in rb.values())
            c["broken"] = sum(v["broken"] for v in rb.values())
            c["rescue_break_per_ds"] = rb
            contrasts[f"{arm}_vs_BASE"] = c

    result = dict(split=args.split, arms=arms, lam=args.lam, tag=args.tag,
                  table=table, contrasts=contrasts)
    out = os.path.join(SC.OUT_DIR,
                       f"analysis_{args.split}" + (f"_{args.tag}" if args.tag else "")
                       + ".json")
    with open(out + ".tmp", "w") as f:
        json.dump(result, f, indent=1)
    os.replace(out + ".tmp", out)

    print(f"\n== {args.split} panel (lam={args.lam}{', tag=' + args.tag if args.tag else ''}) ==")
    print("| arm | TextVQA | DocVQA | OCRBench | macro |")
    print("|---|---|---|---|---|")
    for arm in arms:
        t = table[arm]
        row = " | ".join(f"{t['per_ds'][ds]:.2f}"
                         if t["per_ds"][ds] is not None else "—"
                         for ds in SC.DS_LIST)
        mac = f"{t['macro']:.3f}" if t["macro"] is not None else "—"
        print(f"| {arm} | {row} | {mac} |")
    for k, c in contrasts.items():
        print(f"{k}: delta={c['delta']:+.3f} CI=[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]"
              f" rescued={c.get('rescued')} broken={c.get('broken')}"
              f" per_ds={ {d: round(v, 2) for d, v in c['per_ds'].items()} }")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
