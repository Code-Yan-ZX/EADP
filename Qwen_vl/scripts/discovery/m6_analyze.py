"""
M6 step 3 -- tables for the report, regenerated from the saved grid.

Everything printed here is recomputed from `m6_phase1.json` and the bank, so no
number in `reports/m6_disagreement_rescue.md` is transcribed by hand.

The headroom block is the one piece of Phase-1 evidence that does not come from
the rule grid: it measures where the teacher's head sits inside EADP's own
ranking, which is what makes EADP-next a weak baseline rather than a strong one.

Usage
    python scripts/discovery/m6_analyze.py > outputs/discovery/m6_tables.md
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m6_common import (AUX, BUDGET, Dropped, KS, N_VIS, OUTPUT_DIR, PRIMARY,
                       fit_orientations, load_bank, load_eapd_order)  # noqa: E402

TITLES = {
    "R0-random": "R0 random (mean of 20 seeds)",
    "R1-EADP-next": "R1 EADP-next (greedy 256..256+k-1)",
    "R3-union-top2": "R3 union, 2 strongest proxies",
    "R3-union-top3": "R3 union, 3 strongest proxies",
    "R3-union-top6": "R3 union, all 6 declared proxies",
    "R4-D1": "R4 D1 max_aux_pct - EADP_pct",
    "R4-D1-top2": "R4 D1 restricted to 2 strongest auxiliaries",
    "R4-D1-top3": "R4 D1 restricted to 3 strongest auxiliaries",
    "R4-D2": "R4 D2 EADP_rank - best_aux_rank",
    "R4-D3": "R4 D3 rank range across all proxies",
    "R4-D4": "R4 D4 binary: aux in top-10%, EADP not",
    "R5-maxfusion": "R5 max over oriented aux percentiles",
    "R5-minfusion": "R5 min over oriented aux percentiles (agreement)",
    "R6-ORACLE-union": "R6 ORACLE union (not deployable) ceiling",
}


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def main():
    bank = load_bank()
    dropped = Dropped(bank)
    grid = json.load(open(os.path.join(OUTPUT_DIR, "m6_phase1.json")))
    n = bank["n"]
    orient = fit_orientations(bank, dropped)

    print("# M6 tables (regenerated)\n")
    print(f"bank: {n} instances, budget {BUDGET}, chance recall@k = k/768\n")

    # ---------------------------------------------------------------- audit --
    print("## T1 -- orientation, fitted on the fit split (240) only\n")
    print("| proxy | fitted sign | fit recall@16 | reading |")
    print("|---|---:|---:|---|")
    READ = {
        "imp": "incumbent EADP score; negative sign = teacher head is EADP-unimportant",
        "cos_s0c": "cosine to retained-set centroid; low = representationally novel",
        "nn4_recon": "reconstruction residual from 4 nearest retained; high = not explained by S0",
        "red_s0_top8": "redundancy inside S0; low = not covered by S0",
        "loc_std": "spread of EADP local match across instruction tokens",
        "vis_norm": "feature magnitude",
    }
    for nm in PRIMARY:
        o = orient[nm]
        print(f"| `{nm}` | {o['sign']:+d} | {fmt(o['fit_recall'])} | {READ[nm]} |")

    # ------------------------------------------------------------- headroom --
    print("\n## T2 -- where the teacher's head sits inside EADP's own ranking\n")
    print("Per instance, restricted to the 768 dropped tokens; percentile 100 = "
          "EADP's most important dropped token.\n")
    print("| teacher head | median EADP percentile | frac >= 90th | frac >= 75th | "
          "frac inside EADP-next-64 | chance |")
    print("|---|---:|---:|---:|---:|---:|")
    eo = load_eapd_order(bank)
    loc_cache = []
    for i in range(n):
        loc = {int(t): j for j, t in enumerate(dropped.drop[i])}
        loc_cache.append(loc)
    for q in KS:
        med, f90, f75, fnext = [], [], [], []
        for i in range(n):
            d = dropped.drop[i]
            head = np.where(dropped.tr[i] < q)[0]
            imp = bank["X"][i][d, bank["fi"]["imp"]]
            pct = np.argsort(np.argsort(imp, kind="stable"), kind="stable") / (d.size - 1.0) * 100
            med.append(np.median(pct[head]))
            f90.append((pct[head] >= 90).mean())
            f75.append((pct[head] >= 75).mean())
            nxt = [loc_cache[i][int(t)] for t in eo[i][BUDGET:BUDGET + 64]
                   if int(t) in loc_cache[i]]
            fnext.append(np.isin(head, nxt).mean())
        print(f"| teacher top-{q} dropped | {np.mean(med):.1f} | {fmt(np.mean(f90))} | "
              f"{fmt(np.mean(f75))} | {fmt(np.mean(fnext))} | {fmt(64/768)} |")

    # ---------------------------------------------------- stratified control --
    print("\n## T3 -- does each proxy survive the others as a control?\n")
    print("5 equal-count strata per instance; `ratio` = observed recall@8 over the "
          "exact random-within-stratum expectation. A ratio near 1 means the proxy "
          "is a restatement of the stratifying variable.\n")
    print("| proxy | dir | stratified by | observed | expected | ratio |")
    print("|---|---|---|---:|---:|---:|")
    vi = bank["fi"]["vis_norm"]

    def strat(j, invert, ctrl, nb=5, k=8, q=8):
        obs, exp = [], []
        for i in range(n):
            d = dropped.drop[i]
            head = np.where(dropped.tr[i] < q)[0]
            key = bank["X"][i][d, ctrl]
            strata = np.array_split(np.argsort(key, kind="stable"), nb)
            sc = bank["X"][i][d, j] * (-1 if invert else 1)
            for st in strata:
                if st.size < k:
                    continue
                o = st[np.argsort(-sc[st], kind="stable")[:k]]
                obs.append(len(np.intersect1d(o, head)) / q)
                exp.append(k * len(np.intersect1d(st, head)) / (st.size * q))
        return np.mean(obs), np.mean(exp)

    for nm, invert in (("cos_s0c", True), ("red_s0_top8", True), ("imp", True),
                       ("nn4_recon", False), ("loc_std", False), ("vis_norm", True)):
        j = bank["fi"][nm]
        for cname, ctrl in (("vis_norm", vi), ("cos_s0c", bank["fi"]["cos_s0c"]),
                            ("imp", bank["fi"]["imp"])):
            o, e = strat(j, invert, ctrl)
            print(f"| `{nm}` | {'low' if invert else 'high'} | `{cname}` | "
                  f"{fmt(o)} | {fmt(e)} | {o/e:.2f} |")

    # -------------------------------------------------------------- the grid --
    for k in KS:
        c = grid["grid"][str(k)]
        bs = c["best_single"]
        print(f"\n## T4.k{k} -- critical-miss enrichment, k = {k} (q = {k})\n")
        print("`recall` is the mean over instances of |pick ∩ teacher-top-k dropped| / k "
              "on the held-out 210 (val 60 + test 150). `enrich` is over the k/768 "
              "chance rate. `meanTR` is the mean teacher rank of the picked tokens "
              "(0 = the teacher's best dropped token). The two delta columns are "
              "paired bootstrap on the same 210 instances.\n")
        print("| rule | recall | enrich | meanTR | medTR | Δ vs random | Δ vs EADP-next "
              "| Δ vs best single |")
        print("|---|---:|---:|---:|---:|---|---|---|")
        seeds = [e for nm, e in c["rules"].items() if nm.startswith("R0-random-s")]
        rm = np.mean([e["holdout210"]["recall"] for e in seeds])
        print(f"| R0 random (mean of 20 seeds) | {fmt(rm)} | — | — | — | — | — | — |")

        order = ["R1-EADP-next", bs]
        order += [f"R2-{m}" for m in PRIMARY if f"R2-{m}" != bs]
        order += ["R3-union-top2", "R3-union-top3", "R3-union-top6",
                  "R4-D1", "R4-D1-top2", "R4-D1-top3", "R4-D2", "R4-D3", "R4-D4",
                  "R5-maxfusion", "R5-minfusion", "R6-ORACLE-union"]
        for nm in order:
            e = c["rules"][nm]
            a = e["holdout210"]
            v = e.get("vs", {})

            def d(rn):
                b = v.get(rn)
                if not b:
                    return "—"
                star = "*" if (b["lo"] > 0 or b["hi"] < 0) else ""
                return f"{b['delta']:+.4f}{star}"
            label = TITLES.get(nm, f"R2 single proxy `{nm[3:]}`" if nm.startswith("R2-") else nm)
            print(f"| {label} | {fmt(a['recall'])} | {a['enrichment']:.2f} | "
                  f"{a['mean_teacher_rank']:.1f} | {a['median_teacher_rank']:.1f} | "
                  f"{d('R0-random')} | {d('R1-EADP-next')} | {d(bs)} |")
        seeds5 = [e["holdout210"]["recall"] for nm, e in c["rules"].items()
                  if nm.startswith("R6b-random-in-pool-s")]
        print(f"| R6b random draw from the union pool | {np.mean(seeds5):.4f} | — | — | — "
              f"| — | — | — |")
        print("\n`*` marks a paired bootstrap CI on the 210 that excludes zero. "
              "Chance for recall@k is k/768.\n")

    # ---------------------------------------------------------- per-benchmark --
    print("\n## T5 -- per benchmark (all 450 instances, 150 per benchmark)\n")
    print("Reported on the full bank rather than the held-out 210 so each "
          "benchmark keeps 150 instances; the headline panel for T4/T6 is the "
          "held-out 210.\n")
    for k in KS:
        c = grid["grid"][str(k)]
        bs = c["best_single"]
        print(f"\n**k = {k}**\n")
        print("| rule | TextVQA | DocVQA | OCRBench | macro |")
        print("|---|---:|---:|---:|---:|")
        for nm in ["R1-EADP-next", bs, "R3-union-top6", "R4-D1", "R4-D4",
                   "R5-maxfusion", "R6-ORACLE-union"]:
            pb = c["rules"][nm]["all450"]["per_benchmark"]
            label = TITLES.get(nm, f"R2 single proxy `{nm[3:]}`" if nm.startswith("R2-") else nm)
            print(f"| {label} | {fmt(pb['TextVQA'])} | {fmt(pb['DocVQA'])} | "
                  f"{fmt(pb['OCRBench'])} | {fmt(pb['macro'])} |")

    # ------------------------------------------------------------- holdout vs --
    print("\n## T6 -- paired bootstrap vs the best single proxy, held-out 210\n")
    print("| k | rule | Δ | 95% CI | wins | losses | ties |")
    print("|---:|---|---:|---|---:|---:|---:|")
    for k in KS:
        c = grid["grid"][str(k)]
        bs = c["best_single"]
        for nm in ["R3-union-top2", "R3-union-top6", "R4-D1", "R4-D1-top2", "R4-D2",
                   "R4-D3", "R4-D4", "R5-maxfusion", "R6-ORACLE-union"]:
            b = c["rules"][nm]["vs"][bs]
            print(f"| {k} | {TITLES.get(nm, nm)} | {b['delta']:+.4f} | "
                  f"[{b['lo']:+.4f}, {b['hi']:+.4f}] | {b['n_win']} | {b['n_loss']} | "
                  f"{b['n_tie']} |")

    # --------------------------------------------------------- panels / repro --
    print("\n## T7 -- panels and reproducibility\n")
    print("| panel | n | what it is |")
    print("|---|---:|---|")
    print("| all450 | 450 | the whole frozen bank (fit 240 / val 60 / test 150) |")
    print("| holdout210 | 210 | val + test — every headline number above |")
    print("| test150 | 150 | the historical held-out panel of M2/M3/M4/M5 |")
    print(f"\nrandom-control seed spread over 20 seeds: "
          f"k=8 [{grid['grid']['8']['random_seed_spread']['lo']:.4f}, "
          f"{grid['grid']['8']['random_seed_spread']['hi']:.4f}], "
          f"k=32 [{grid['grid']['32']['random_seed_spread']['lo']:.4f}, "
          f"{grid['grid']['32']['random_seed_spread']['hi']:.4f}]")


if __name__ == "__main__":
    main()
