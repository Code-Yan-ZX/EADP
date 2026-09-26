"""
M7 step 5 -- tables for the report, regenerated from the saved artefacts.

Nothing in `reports/m7_candidate_union_hedging.md` is transcribed by hand.

Usage
    python scripts/discovery/m7_analyze.py > outputs/discovery/m7_tables.md
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m6_common import (AUX, DS_ORDER, Dropped, OUTPUT_DIR, SHORT, load_bank,
                       load_eapd_order)                              # noqa: E402
from m7_common import (M_GRID, SUBSETS, coverage, cos_topn, mean_teacher_rank,
                       oracle_from_union, random_n, union_all)       # noqa: E402
from m6_common import (fit_orientations, r_disagree, r_maxfusion,
                       paired_bootstrap)                             # noqa: E402

SEED = 20260927


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def arm_table(rec, arms, ref="B2"):
    """macro / per-benchmark / deltas with a paired bootstrap vs `ref`."""
    A = rec["arms"]
    if ref not in A or "error" in A.get(ref, {}):
        return None
    rng = np.random.default_rng(SEED)

    def hits(a):
        return {ds: np.array(A[a]["per_benchmark"][ds]["hits"]) for ds in DS_ORDER}

    def macro(h):
        return float(np.mean([h[ds].mean() for ds in DS_ORDER]))

    base = hits(ref)
    rows = []
    for a in arms:
        if a not in A or "error" in A[a]:
            continue
        h = hits(a)
        per = np.concatenate([[h[ds][i] - base[ds][i] for i in range(len(h[ds]))]
                              for ds in DS_ORDER])
        bs = per[rng.integers(0, per.size, size=(4000, per.size))].mean(1)
        rows.append(dict(
            arm=a, macro=macro(h) * 100,
            tv=h["TextVQA_VAL"].mean() * 100, dv=h["DocVQA_VAL"].mean() * 100,
            oc=h["OCRBench"].mean() * 100,
            d=per.mean() * 100, lo=float(np.percentile(bs, 2.5)) * 100,
            hi=float(np.percentile(bs, 97.5)) * 100,
            win=int((per > 0).sum()), loss=int((per < 0).sum()),
            ms=A[a].get("hedge_ms_median", 0.0)))
    return rows


def print_arm_table(title, rec, arms, ref="B2"):
    rows = arm_table(rec, arms, ref)
    if rows is None:
        print(f"\n## {title}\n\n_({ref} missing -- not run)_\n")
        return
    print(f"\n## {title}\n")
    print(f"| arm | TextVQA | DocVQA | OCRBench | macro | Δ vs {ref} | 95% CI | "
          f"win/loss | hedge ms |")
    print("|---|---:|---:|---:|---:|---:|---|---|---:|")
    for r in rows:
        print(f"| {r['arm']} | {r['tv']:.3f} | {r['dv']:.3f} | {r['oc']:.3f} | "
              f"**{r['macro']:.3f}** | {r['d']:+.3f} | "
              f"[{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['win']}/{r['loss']} | "
              f"{r['ms']:.3f} |")


def main():
    p1 = json.load(open(os.path.join(OUTPUT_DIR, "m7_phase1.json")))
    bank = load_bank()
    dropped = Dropped(bank)
    order = load_eapd_order(bank)
    n = bank["n"]
    hold = np.array([i for i in range(n) if bank["split"][i] in ("val", "test")])
    orient = fit_orientations(bank, dropped)

    print("# M7 tables (regenerated)\n")
    print("`coverage@q` = |pick ∩ teacher-top-q dropped| / q, per instance then "
          "averaged. `yield@q` = the same intersection over the SLOTS SPENT. "
          "`meanTR` = mean teacher rank of the picked tokens (0 = the teacher's "
          "best dropped token). Chance coverage = |pick|/768.\n")

    # ------------------------------------------------------- phase 1 audit --
    print("## T1 -- candidate-union budget audit (held-out 210)\n")
    print("| proxies | m | mean \\|U\\| | p10 | p90 | cov@8 | cov@16 | cov@32 | "
          "yield@8 | meanTR |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, c in p1["configs"].items():
        s, e = c["size"], c["panels"]["holdout210"]
        print(f"| {c['subset']} | {c['m']} | {s['mean']:.1f} | {s['p10']:.0f} | "
              f"{s['p90']:.0f} | {fmt(e['union_cov8'])} | {fmt(e['union_cov16'])} | "
              f"{fmt(e['union_cov32'])} | {fmt(e['union_yield8'])} | "
              f"{e['union_meanTR']:.1f} |")

    # --------------------------------------------- matched-slot comparison --
    print("\n## T2 -- the decisive comparison: union vs matched-slot controls "
          "(held-out 210)\n")
    print("Every row spends the SAME number of slots on the same instance.\n")
    print("| proxies | m | \\|U\\| | rule | cov@8 | cov@16 | cov@32 | meanTR |")
    print("|---|---:|---:|---|---:|---:|---:|---:|")
    for key, c in p1["configs"].items():
        if c["size"]["mean"] < 18:
            continue
        e = c["panels"]["holdout210"]
        for lab, pre in (("M7 complete union", "union"), ("single `cos_s0c`", "cos"),
                         ("EADP-next", "eadpnext"), ("random matched", "random")):
            tr = e.get(f"{pre}_meanTR")
            tr_s = f"{tr:.1f}" if tr is not None else "—"
            print(f"| {c['subset']} | {c['m']} | {c['size']['mean']:.1f} | {lab} | "
                  f"{fmt(e[f'{pre}_cov8'])} | {fmt(e[f'{pre}_cov16'])} | "
                  f"{fmt(e[f'{pre}_cov32'])} | {tr_s} |")

    # -------------------------------------------------- the mechanism claim --
    print("\n## T3 -- the mechanism claim, at matched slots\n")
    print("M6's best disagreement rule and its max-fusion, re-run at the SAME "
          "slot count as the complete union.\n")
    print("| subset | m | slots | rule | cov@8 | cov@16 | cov@32 | meanTR |")
    print("|---|---:|---:|---|---:|---:|---:|---:|")
    for sub in ("E-all6", "D-cos+red+imp+nn4"):
        for m in (8, 12, 16):
            members = SUBSETS[sub]
            U = union_all(bank, dropped, m, members)
            sz = np.array([u.size for u in U])
            cand = {
                "M7 complete union": U,
                "single `cos_s0c`": [cos_topn(bank, dropped, i, int(sz[i]))
                                     for i in range(n)],
                "M6 D4 disagreement": [r_disagree(bank, dropped, orient, i,
                                                  int(sz[i]), "D4") for i in range(n)],
                "M6 max-fusion": [r_maxfusion(bank, dropped, orient, i,
                                              int(sz[i]), AUX) for i in range(n)],
                "random matched": [random_n(dropped, i, int(sz[i]), 0)
                                   for i in range(n)],
            }
            for lab, p in cand.items():
                cov = [coverage(p, dropped, q)[hold].mean() for q in (8, 16, 32)]
                tr = mean_teacher_rank([p[i] for i in hold], dropped)
                print(f"| {sub} | {m} | {sz.mean():.1f} | {lab} | {fmt(cov[0])} | "
                      f"{fmt(cov[1])} | {fmt(cov[2])} | {tr:.1f} |")

    # --------------------------------------------------------- generation --
    for panel, title in (("bank", "T4 -- Phase 3, screening panel: held-out test 150"),
                         ("ext", "T5 -- Phase 3, independent confirmation panel: "
                                 "M1 extension, 720 instances (240 per benchmark)")):
        f = os.path.join(OUTPUT_DIR, f"m7_accuracy_{panel}.json")
        if not os.path.exists(f):
            print(f"\n## {title}\n\n_(not run)_\n")
            continue
        rec = json.load(open(f))
        print_arm_table(title, rec,
                        ["B2", "U8", "U12", "U16", "COS16", "COS12",
                         "RND16", "RND12", "CORE16", "U16R", "U16I"])

    # ------------------------------------ the contrasts that decide it -----
    f = os.path.join(OUTPUT_DIR, "m7_accuracy_ext.json")
    if os.path.exists(f):
        A = json.load(open(f))["arms"]
        rng = np.random.default_rng(SEED)

        def hh(a):
            return {ds: np.array(A[a]["per_benchmark"][ds]["hits"]) for ds in DS_ORDER}

        H = {a: hh(a) for a in A}
        print("\n## T5b -- matched-slot contrasts on the confirmation panel (720)\n")
        print("| contrast | Δ macro | 95 % CI | significant | win/loss |")
        print("|---|---:|---|---|---|")
        for a, b in (("U12", "B2"), ("U12", "COS12"), ("U12", "RND12"),
                     ("RND12", "COS12"), ("COS12", "B2")):
            per = np.concatenate([[H[a][ds][i] - H[b][ds][i]
                                   for i in range(len(H[a][ds]))] for ds in DS_ORDER])
            bs = per[rng.integers(0, per.size, size=(4000, per.size))].mean(1)
            lo, hi = float(np.percentile(bs, 2.5)) * 100, float(np.percentile(bs, 97.5)) * 100
            sig = "**yes**" if (lo > 0 or hi < 0) else "no"
            print(f"| {a} − {b} | {per.mean()*100:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                  f"{sig} | {int((per>0).sum())}/{int((per<0).sum())} |")

        # the decomposition: same core, same slots, only the fill changes
        print("\n## T5c -- decomposition on the confirmation panel\n")
        print("Every row has the same core construction and the same slot count; "
              "only what fills the rescue slots changes. `content` is the macro "
              "recovered above the shrink-only arm.\n")
        for tag, core_arm, fills in (("m=12", "CORE12",
                                      [("single `cos_s0c`", "COS12"),
                                       ("random", "RND12"),
                                       ("M7 complete union", "U12")]),):
            if core_arm not in A:
                continue
            base = A["B2"]["macro_pct"]
            c = A[core_arm]["macro_pct"]
            print(f"**{tag}: the core is 256 − |U| ≈ 197 tokens**\n")
            print("| step | macro | effect |")
            print("|---|---:|---:|")
            print(f"| B2, 256 tokens | {base:.3f} | — |")
            print(f"| {core_arm}, shrink only | {c:.3f} | **shrink {c-base:+.3f}** |")
            for lab, a in fills:
                print(f"| {a} = shrink + {lab} | {A[a]['macro_pct']:.3f} | "
                      f"content **{A[a]['macro_pct']-c:+.3f}** |")

    # ------------------------------------------------------- efficiency ----
    print("\n## T6 -- efficiency\n")
    f = os.path.join(OUTPUT_DIR, "m7_accuracy_bank.json")
    if os.path.exists(f):
        rec = json.load(open(f))
        print("| arm | tokens | hedge median ms | wall s (150) |")
        print("|---|---:|---:|---:|")
        for a, r in rec["arms"].items():
            if "error" in r:
                continue
            tk = 178 if a.startswith("CORE") else 256
            print(f"| {a} | {tk} | {r.get('hedge_ms_median', 0):.3f} | "
                  f"{r['wall_seconds']:.0f} |")

    print("\n## T7 -- provenance\n")
    print("| panel | n | source |")
    print("|---|---:|---|")
    print("| bank test150 | 150 | the frozen 450's test split; the historical panel |")
    print("| ext | 720 | `m1_plan.json` extension rows, disjoint from the bank |")


if __name__ == "__main__":
    main()
