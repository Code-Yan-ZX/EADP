"""Stage-1 visual-calibration discovery — analysis (mirrors amp_analyze).

Accuracy = 100 x mean(per-question score) over the ACTUAL evaluated count;
macro = equal-weight mean over the three datasets; paired cluster bootstrap
(by image identity, within dataset) on the macro delta vs BASE, 95% CI.

BASE is read from the anchor-merge-pilot shards (same frozen manifest, same
pipeline: bank keep == online b1, identity gather — gates G1/G2 must PASS).
Adds selected-set overlap vs BASE (from per-record keep lists) and the
GO/KILL gate readout (user brief, frozen thresholds).

Usage:
  python s1n_analyze.py --split dev --arms A,B,C,D
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import s1n_common as SC
import amp_common as AC
from amp_analyze import (acc100, macro_of, paired_macro_bootstrap,
                         rescue_break)


def load_scores(split, arms):
    out = {}
    for arm in arms:
        for ds in SC.DS_LIST:
            if arm == "BASE":
                hits = sorted(glob.glob(os.path.join(
                    AC.OUT_DIR, "acc", split, "BASE", "K*",
                    f"{ds}_score.json")))
            else:
                hits = sorted(glob.glob(os.path.join(
                    SC.ACC_DIR, split, arm, "K*", f"{ds}_score.json")))
            if not hits:
                continue
            s = json.load(open(hits[-1]))
            if not s.get("per_question"):
                continue
            out[(arm, ds)] = s
    return out


def run_quality(scores, arm, split):
    out = {}
    for ds in SC.DS_LIST:
        if (arm, ds) not in scores:
            continue
        if arm == "BASE":
            hits = sorted(glob.glob(os.path.join(
                AC.OUT_DIR, "acc", split, "BASE", "K*", f"{ds}.json")))
        else:
            hits = sorted(glob.glob(os.path.join(
                SC.ACC_DIR, split, arm, "K*", f"{ds}.json")))
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


def overlap_vs_base(split, arms, manifest):
    """Selected-set overlap of each arm vs the frozen BASE bank (b1)."""
    banks = {ds: AC.load_bank(split, ds) for ds in SC.DS_LIST}
    out = {}
    for arm in arms:
        if arm == "BASE":
            continue
        per_ds = {}
        for ds in SC.DS_LIST:
            path = SC.shard_path(split, arm, ds)
            if not os.path.exists(path):
                continue
            records = json.load(open(path))["records"]
            fracs, changed = [], 0
            for key, rec in records.items():
                base_keep = banks[ds].get(key, {}).get("keep")
                arm_keep = rec.get("keep")
                if not base_keep or not arm_keep:
                    continue
                inter = len(set(base_keep) & set(arm_keep))
                fracs.append(inter / max(1, len(base_keep)))
                changed += int(inter != len(base_keep))
            if fracs:
                per_ds[ds] = dict(
                    overlap=float(sum(fracs) / len(fracs)),
                    n_changed=changed, n=len(fracs))
        out[arm] = per_ds
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "confirm"])
    ap.add_argument("--arms", required=True)
    args = ap.parse_args()
    arms = args.arms.split(",")
    manifest = SC.load_manifest()
    scores = load_scores(args.split, arms)

    table = {}
    for arm in arms:
        per = {ds: (acc100(scores[(arm, ds)]) if (arm, ds) in scores else None)
               for ds in SC.DS_LIST}
        table[arm] = dict(per_ds=per, macro=macro_of(scores, arm),
                          quality=run_quality(scores, arm, args.split))

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

    overlap = overlap_vs_base(args.split, arms, manifest)

    result = dict(split=args.split, arms=arms, table=table,
                  contrasts=contrasts, overlap_vs_base=overlap)
    out = os.path.join(SC.OUT_DIR, f"analysis_{args.split}.json")
    with open(out + ".tmp", "w") as f:
        json.dump(result, f, indent=1)
    os.replace(out + ".tmp", out)

    print(f"\n== {args.split} panel ==")
    print("| arm | TextVQA | DocVQA | OCRBench | macro |")
    print("|---|---|---|---|---|")
    for arm in arms:
        t = table[arm]
        row = " | ".join(f"{t['per_ds'][ds]:.2f}"
                         if t['per_ds'][ds] is not None else "—"
                         for ds in SC.DS_LIST)
        mac = f"{t['macro']:.3f}" if t["macro"] is not None else "—"
        print(f"| {arm} | {row} | {mac} |")
    base_mac = table.get("BASE", {}).get("macro")
    best_arm, best_delta = None, None
    for k, c in contrasts.items():
        arm = k.split("_vs_")[0]
        print(f"{k}: delta={c['delta']:+.3f} CI=[{c['ci'][0]:+.3f}, "
              f"{c['ci'][1]:+.3f}] rescued={c.get('rescued')} "
              f"broken={c.get('broken')} per_ds="
              f"{ {d: round(v, 2) for d, v in c['per_ds'].items()} }")
        if base_mac is not None and \
                (best_delta is None or c["delta"] > best_delta):
            best_arm, best_delta = arm, c["delta"]
    for arm, per_ds in overlap.items():
        for ds, o in per_ds.items():
            print(f"overlap {arm} vs BASE {ds}: {o['overlap']:.3f} "
                  f"({o['n_changed']}/{o['n']} samples changed)")
    # frozen GO/KILL readout
    verdict = "NO_SIGNAL"
    if best_delta is not None:
        verdict = "GO" if best_delta >= 0.3 else "KILL"
    print(f"\n[GO/KILL] best={best_arm} delta={best_delta} -> {verdict} "
          f"(frozen rule: GO iff best macro delta >= +0.3 and no benchmark "
          f"collapses and paired trend positive)")
    result["go_kill"] = dict(best_arm=best_arm, best_delta=best_delta,
                             verdict=verdict)
    with open(out + ".tmp", "w") as f:
        json.dump(result, f, indent=1)
    os.replace(out + ".tmp", out)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
