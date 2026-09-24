"""
M3-v2 step 5 -- tables and the verdict, from the proxy JSONs alone.

Reads `m3v2_proxy.json` (the primary grid) and, when present,
`m3v2_proxy_reg.json` (the regularised re-run), and prints the tables the
report is built from.  No number in the report is typed by hand.

Usage
    python scripts/discovery/m3v2_analyze.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUT_DIR                                       # noqa: E402

COLS = ("top4_recall@8", "top8_recall@8", "top8_recall@16", "top16_recall@16",
        "mean_rank@16", "median_rank@16", "frac_in_top8@16", "frac_in_top16@16",
        "hw@16", "ndcg@16")
SHORT = ("T4R@8", "T8R@8", "T8R@16", "T16R@16", "meanRk", "medRk",
         "fT8", "fT16", "hw@16", "ndcg")


def row(name, m):
    return f"| {name} | " + " | ".join(
        f"{m[c]:.4f}" if c != "mean_rank@16" and c != "median_rank@16"
        else f"{m[c]:.1f}" for c in COLS) + " |"


def load(name):
    p = os.path.join(OUTPUT_DIR, name)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    # Drop the per-instance metric vectors: they are 17 MB per grid and are
    # already in `m3v2_proxy*.json`, which this file is derived from.  What is
    # kept here is the aggregate table plus the gate, which is what the report
    # cites.
    for group in ("scorers",):
        for r in d.get(group, {}).values():
            for k in [k for k in r if k.endswith("_per_instance")]:
                del r[k]
    for r in d.get("v2", {}).values():
        for k in [k for k in r if k.endswith("_per_instance")]:
            del r[k]
    return d


def main():
    out = {}
    for tag, fn in (("primary", "m3v2_proxy.json"),
                    ("reg", "m3v2_proxy_reg.json")):
        d = load(fn)
        if d is None:
            print(f"[skip] {fn} not present")
            continue
        out[tag] = d
        print(f"\n{'='*100}\n## {tag}  ({fn})\n{'='*100}")
        print("[xcheck] bank-rebuilt v0 == stored live v0 rescue:",
              d.get("xcheck_v0", {}).get("passed"),
              d.get("xcheck_v0", {}).get("exact"), "/",
              d.get("xcheck_v0", {}).get("n"))

        print("\n### baselines and reference scorers\n")
        print("| scorer | split | " + " | ".join(SHORT) + " |")
        print("|---" * (len(SHORT) + 2) + "|")
        for nm, r in d["scorers"].items():
            for sp in ("fit", "val", "test"):
                print(row(f"{nm} ({sp})", r[sp]))

        v2 = d.get("v2", {})
        if not v2:
            print("\n[no v2 checkpoints in this proxy run]")
            continue
        grids = sorted({v["meta"]["grid"] for v in v2.values()})
        for grid in grids:
            print(f"\n### v2 grid `{grid}` -- val\n")
            print("| cfg | seed | ep | " + " | ".join(SHORT) + " |")
            print("|---" * (len(SHORT) + 3) + "|")
            for nm, r in sorted(v2.items()):
                if r["meta"]["grid"] != grid:
                    continue
                print(f"| {nm} | {r['meta']['seed']} | {r['meta']['best_epoch']} | "
                      + " | ".join(
                          f"{r['val'][c]:.4f}" if c not in ("mean_rank@16", "median_rank@16")
                          else f"{r['val'][c]:.1f}" for c in COLS) + " |")

            print(f"\n### v2 grid `{grid}` -- seed means, and the fit/val gap\n")
            print("| cfg | n | val T16R@16 | val meanRk | val hw@16 | test T16R@16 "
                  "| fit T16R@16 | init T16R@16 | params |")
            print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            agg = {}
            for nm, r in sorted(v2.items()):
                k = nm.split("|")[0]
                agg.setdefault(k, []).append(r)
            for k, rs in sorted(agg.items()):
                f = lambda c, s="val": float(np.mean([r[s][c] for r in rs]))  # noqa: E731
                fit = [r["meta"]["fit"] for r in rs if r["meta"].get("fit")]
                ini = [r["meta"]["init_val"] for r in rs if r["meta"].get("init_val")]
                print(f"| {k} | {len(rs)} | {f('top16_recall@16'):.4f} | "
                      f"{f('mean_rank@16'):.1f} | {f('hw@16'):.4f} | "
                      f"{f('top16_recall@16','test'):.4f} | "
                      f"{(np.mean([x['top16_recall@16'] for x in fit]) if fit else float('nan')):.4f} | "
                      f"{(np.mean([x['top16_recall@16'] for x in ini]) if ini else float('nan')):.4f} | "
                      f"{rs[0]['meta']['n_params']/1000:.0f}k |")
                out.setdefault("seed_means", {}).setdefault(grid, {})[k] = dict(
                    n=len(rs), val_top16_recall_at_16=f("top16_recall@16"),
                    val_mean_rank_at_16=f("mean_rank@16"), val_hw_at_16=f("hw@16"),
                    val_ndcg_at_16=f("ndcg@16"),
                    test_top16_recall_at_16=f("top16_recall@16", "test"),
                    fit_top16_recall_at_16=(float(np.mean(
                        [x["top16_recall@16"] for x in fit])) if fit else None),
                    init_top16_recall_at_16=(float(np.mean(
                        [x["top16_recall@16"] for x in ini])) if ini else None),
                    n_params=rs[0]["meta"]["n_params"])

        print(f"\n### gate ({tag})")
        g = d["gate"]
        print("v0 reference:", g.get("v0_reference"))
        for cfg, v in sorted(g.get("per_config", {}).items()):
            print(f"  {cfg:8s} seeds pass {v['n_pass']}/{v['n_seeds']}  "
                  f"mean rec {v['seed_mean_recall']:.4f}  mean rank "
                  f"{v['seed_mean_rank']:.2f}  mean hw {v['seed_mean_hw']:.4f}  "
                  f"-> {'PASS' if v['gate_pass'] else 'FAIL'}")
        print("  HEAD-GATE:", "PASS" if g.get("any_pass") else "FAIL")
        out.setdefault("gate", {})[tag] = g

    with open(os.path.join(OUTPUT_DIR, "m3v2_analysis.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\n[saved] m3v2_analysis.json")


if __name__ == "__main__":
    main()
