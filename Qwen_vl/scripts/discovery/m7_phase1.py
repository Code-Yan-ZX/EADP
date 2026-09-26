"""
M7 step 1 -- Phase 1, the candidate-union budget audit.  No generation.

The question, and the only one:

    At a realistic slot cost (|U| ~ 20-50), does keeping the WHOLE union buy
    materially wider catastrophic-miss coverage than simply asking the best
    single proxy for the same number of slots?

Everything is measured per instance and then averaged.  `coverage@q` is
|pick ∩ teacher-top-q dropped| / q, the same quantity M6 called head recall and
M3-v0 called overlap@r.  `yield` is the same intersection divided by the number
of slots spent -- the quantity that decides whether the union is worth its
budget.

The decisive comparison is `union(m)` vs `cos_s0c top-|U|`: same slots, same
panel, paired.  A union that cannot beat the single proxy at equal cost is not
hedging anything; it is just spending budget on a worse ranking.

Usage
    python scripts/discovery/m7_phase1.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m6_common import (Dropped, HOLDOUT, dump_json, fit_orientations,
                       load_bank, load_eapd_order, paired_bootstrap)   # noqa: E402
from m7_common import (HEADS, M_GRID, ORIENT, RANDOM_SEEDS, SUBSETS,
                       by_benchmark, build_union, cos_topn, coverage,
                       coverage_per_slot, eadp_next_n, mean_teacher_rank,
                       oracle_from_union, pairwise_overlap, random_n,
                       size_stats, union_all)                          # noqa: E402


def main():
    bank = load_bank()
    dropped = Dropped(bank)
    eapd_order = load_eapd_order(bank)
    n = bank["n"]

    # --- gate: the inherited orientations must still be what the fit split says
    fitted = fit_orientations(bank, dropped)
    bad = [nm for nm in ORIENT
           if fitted[nm]["sign"] != ORIENT[nm]]
    print(f"[orientation] inherited signs re-checked on fit: "
          f"{'AGREE' if not bad else 'DISAGREE ' + str(bad)}")

    hold = np.array([i for i in range(n) if bank["split"][i] in HOLDOUT])
    test = np.array([i for i in range(n) if bank["split"][i] == "test"])
    panels = {"all450": np.arange(n), "holdout210": hold, "test150": test}

    out = dict(orientation_check=dict(agree=not bad, disagree=bad),
               m_grid=list(M_GRID), heads=list(HEADS),
               panels={k: int(v.size) for k, v in panels.items()},
               configs={})

    for lab, members in SUBSETS.items():
        for m in M_GRID:
            key = f"{lab}|m{m}"
            U = union_all(bank, dropped, m, members)
            sizes = np.array([u.size for u in U])
            # matched-slot controls, built per instance to |U_i| exactly
            cos_m = [cos_topn(bank, dropped, i, int(sizes[i])) for i in range(n)]
            eadp_m = [eadp_next_n(eapd_order, dropped, i, int(sizes[i]))
                      for i in range(n)]
            rnd = {s: [random_n(dropped, i, int(sizes[i]), s) for i in range(n)]
                   for s in RANDOM_SEEDS}
            ceil = [oracle_from_union(bank, dropped, i, U[i], int(sizes[i]))
                    for i in range(n)]

            cell = dict(subset=lab, members=list(members), m=m,
                        size=size_stats(U),
                        overlap=pairwise_overlap(bank, dropped, m, members),
                        panels={})
            for pname, idx in panels.items():
                e = {}
                for q in HEADS:
                    u_cov = coverage(U, dropped, q)
                    e[f"union_cov{q}"] = float(u_cov[idx].mean())
                    e[f"union_yield{q}"] = float(
                        coverage_per_slot(U, dropped, q)[idx].mean())
                    e[f"cos_cov{q}"] = float(coverage(cos_m, dropped, q)[idx].mean())
                    e[f"cos_yield{q}"] = float(
                        coverage_per_slot(cos_m, dropped, q)[idx].mean())
                    e[f"eadpnext_cov{q}"] = float(
                        coverage(eadp_m, dropped, q)[idx].mean())
                    e[f"ceil_cov{q}"] = float(coverage(ceil, dropped, q)[idx].mean())
                    r = np.mean([coverage(rnd[s], dropped, q)[idx].mean()
                                 for s in RANDOM_SEEDS])
                    e[f"random_cov{q}"] = float(r)
                e["union_meanTR"] = mean_teacher_rank([U[i] for i in idx], dropped)
                e["cos_meanTR"] = mean_teacher_rank([cos_m[i] for i in idx], dropped)
                e["union_per_benchmark"] = by_benchmark(coverage(U, dropped, 8), bank)
                e["cos_per_benchmark"] = by_benchmark(coverage(cos_m, dropped, 8), bank)
                cell["panels"][pname] = e
            # paired tests on the held-out 210: union vs each control, at q=8/16/32
            cell["paired_vs_cos"] = {
                f"q{q}": paired_bootstrap(coverage(U, dropped, q)[hold],
                                          coverage(cos_m, dropped, q)[hold])
                for q in HEADS}
            cell["paired_vs_random"] = {
                f"q{q}": paired_bootstrap(
                    coverage(U, dropped, q)[hold],
                    np.mean([coverage(rnd[s], dropped, q)[hold]
                             for s in RANDOM_SEEDS], axis=0))
                for q in HEADS}
            out["configs"][key] = cell
            e = cell["panels"]["holdout210"]
            print(f"[{key:20s}] |U| {cell['size']['mean']:5.1f}  "
                  f"h8 {e['union_cov8']:.4f} vs cos {e['cos_cov8']:.4f} "
                  f"({e['union_cov8']-e['cos_cov8']:+.4f})  "
                  f"h16 {e['union_cov16']:.4f} vs {e['cos_cov16']:.4f}  "
                  f"h32 {e['union_cov32']:.4f} vs {e['cos_cov32']:.4f}")

    dump_json("m7_phase1.json", out)
    print("[saved] m7_phase1.json")


if __name__ == "__main__":
    main()
