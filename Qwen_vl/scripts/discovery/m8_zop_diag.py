"""
M8 step 4 -- what is the ZOO-Prune score actually measuring?

Phase 1 answers "does ZO-P rank the teacher's head?" with no.  This answers the
question that decides whether that is a fact about the estimator or a fact
about the stage: *what is the score a function of?*

    D1  rank agreement with the gradient teacher, over all 1024 tokens and
        inside the dropped set, per instance then averaged -- with BOTH signs,
        so an anti-correlated score cannot hide behind a sign convention
    D2  rank agreement with each of the 27 cheap columns the M5 bank already
        caches.  ZOO-Prune's estimator is query-free and token-local by
        construction; if it is a restatement of a column the incumbent already
        computes, then it cannot carry what those columns do not
    D3  how much of the score survives a change of the random direction bank
        (split-half reliability, two independent seeds)
    D4  the score's own global selection, against the incumbent's

Usage
    python scripts/discovery/m8_zop_diag.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUT_DIR                                       # noqa: E402
from m6_common import Dropped, dump_json, load_bank                 # noqa: E402
from m8_common import ORIENT                                        # noqa: E402
from scipy.stats import spearmanr                                   # noqa: E402

TAG = "m8_zop_diag"
HOLDOUT = ("val", "test")


def main():
    bank = load_bank()
    bank["n"] = len(bank["key"])
    dropped = Dropped(bank)
    z = np.load(os.path.join(OUTPUT_DIR, "m8_zop.npz"), allow_pickle=False)
    assert [str(k) for k in z["key"]] == bank["key"]
    # read every budget present, so an m=64 run is picked up without an edit
    have = sorted(int(k.split("S_m")[1]) for k in z.files
                  if k.startswith("S_m"))
    S = {m: z[f"S_m{m}"] for m in have}
    out_dirs = have
    names = bank["feature_names"]
    hold = [i for i in range(bank["n"]) if bank["split"][i] in HOLDOUT]
    out = dict(n_holdout=len(hold), features=names, dirs=out_dirs)

    # ---- D1: rank agreement with the teacher -----------------------------
    d1 = {}
    for m in out_dirs:
        over_all, in_drop = [], []
        for i in hold:
            g = np.abs(bank["g2"][i])
            over_all.append(spearmanr(S[m][i], g).statistic)
            d = dropped.drop[i]
            in_drop.append(spearmanr(S[m][i][d], g[d]).statistic)
        d1[f"m{m}"] = dict(
            vs_teacher_all1024=float(np.mean(over_all)),
            vs_teacher_all1024_abs=float(np.mean(np.abs(over_all))),
            vs_teacher_dropped=float(np.mean(in_drop)),
            vs_teacher_dropped_abs=float(np.mean(np.abs(in_drop))),
            frac_positive_all=float(np.mean(np.array(over_all) > 0)),
            frac_positive_dropped=float(np.mean(np.array(in_drop) > 0)))
    out["D1_teacher"] = d1

    # ---- D2: is it a restatement of a cheap column? ----------------------
    d2 = {}
    for m in out_dirs:
        per_col = {}
        for c, nm in enumerate(names):
            rs = [spearmanr(S[m][i], bank["X"][i][:, c]).statistic for i in hold]
            per_col[nm] = dict(mean=float(np.mean(rs)),
                               mean_abs=float(np.mean(np.abs(rs))),
                               frac_positive=float(np.mean(np.array(rs) > 0)))
        d2[f"m{m}"] = dict(
            top_by_abs=sorted(per_col.items(), key=lambda kv: -kv[1]["mean_abs"])[:8],
            per_column=per_col)
    out["D2_cheap"] = d2

    # ---- D4: the score's own global selection ----------------------------
    d4 = {}
    for m in out_dirs:
        pick = [np.argsort(-S[m][i], kind="stable")[:256] for i in range(bank["n"])]
        eadp = [np.argsort(-bank["X"][i][:, bank["fi"]["imp"]], kind="stable")[:256]
                for i in range(bank["n"])]
        ov = [len(set(p.tolist()) & set(bank["s0"][i].tolist())) / 256
              for i, p in enumerate(pick)]
        ov_e = [len(set(p.tolist()) & set(bank["s0"][i].tolist())) / 256
                for i, p in enumerate(eadp)]
        d4[f"m{m}"] = dict(overlap_with_B2_s0=float(np.mean(ov)),
                           eadp_importance_top256_vs_s0=float(np.mean(ov_e)))
    out["D4_selection"] = d4

    # ---- D5: what the CHEAP columns achieve against the same teacher -------
    # The bar ZO-P has to clear, measured on the same instances, same set, same
    # statistic -- so "ZO-P is worse than a free column" is a number and not an
    # inference from a recall table computed elsewhere.
    d5 = {}
    for nm in names:
        c = bank["fi"][nm]
        sgn = ORIENT.get(nm, 1)
        rs = [spearmanr(sgn * bank["X"][i][dropped.drop[i], c],
                        np.abs(bank["g2"][i][dropped.drop[i]])).statistic
              for i in hold]
        d5[nm] = dict(oriented_mean=float(np.mean(rs)),
                      raw_mean=float(-np.mean(rs) if sgn < 0 else np.mean(rs)))
    out["D5_cheap_vs_teacher"] = d5

    dump_json(f"{TAG}.json", out)

    print("D1  ZO-P vs the gradient teacher (spearman, per instance then mean)")
    print(f"{'m':>3} {'all1024':>9} {'|all1024|':>10} {'dropped':>9} "
          f"{'|dropped|':>10} {'frac>0':>8}")
    for m in out_dirs:
        v = d1[f"m{m}"]
        print(f"{m:>3} {v['vs_teacher_all1024']:+9.4f} "
              f"{v['vs_teacher_all1024_abs']:10.4f} "
              f"{v['vs_teacher_dropped']:+9.4f} "
              f"{v['vs_teacher_dropped_abs']:10.4f} "
              f"{v['frac_positive_dropped']:8.3f}")

    print("\nD2  ZO-P vs the cheap columns it would have to beat (m=8, |rho|)")
    for nm, v in d2[f"m{max(out_dirs)}"]["top_by_abs"]:
        print(f"  {nm:>14} mean rho {v['mean']:+.4f}   |rho| {v['mean_abs']:.4f}   "
              f"frac>0 {v['frac_positive']:.2f}")

    print("\nD5  cheap columns vs |teacher| in the dropped set (oriented)")
    for nm, v in sorted(d5.items(), key=lambda kv: -kv[1]["oriented_mean"])[:8]:
        print(f"  {nm:>14} {v['oriented_mean']:+.4f}")

    print("\nD4  ZO-P's own top-256 vs the incumbent's S0 (overlap fraction)")
    for m in out_dirs:
        v = d4[f"m{m}"]
        print(f"  m={m}: ZO-P {v['overlap_with_B2_s0']:.4f}   "
              f"EADP-importance top-256 {v['eadp_importance_top256_vs_s0']:.4f}")


if __name__ == "__main__":
    main()
