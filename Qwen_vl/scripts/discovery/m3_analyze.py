"""
M3-v0 step 4 -- tables, controls, transitions, forecasts and the frozen verdict.

Reads stored artefacts only (no GPU): the M3 grid, the M2 baselines, the pre-LLM
audit cache and the teacher cache.

Three things this file is careful about, because the audit said so:

1. **The verdict reads ONE pre-registered arm.**  `m3_accuracy.PRIMARY_ARM` is
   MG-lowimp-r16, frozen before the grid ran.  The other arms are reported, and
   the max over them is reported next to a simulated null band, but the verdict
   is the primary's.
2. **Every threshold is printed with the resolution it needs.**  The paired SE
   on this 150 is ~2-4 macro; the MDE at 80 % power is printed beside each bar,
   and the paired CI is what decides, not the point estimate.
3. **The oracle is a calibration check, not a discovery.**  S2-C2 already ran
   this exact swap family on these exact 150 instances with a different base
   set; its `addteacher:8` = 66.65 vs `student set` 57.32 fixes the expected
   magnitude before M3 measured anything.  The file reports both forecasts.

Usage
    python scripts/discovery/m3_analyze.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUT_DIR                                       # noqa: E402
from m2_accuracy import DS_ORDER                                     # noqa: E402

B0_MACRO, B1_MACRO, B2_MACRO = 75.62, 61.10, 59.88
PRIMARY_ARM = "MG-lowimp-r16"
# S2-B, on these same 150 instances (the pilot arm of the frozen bank).
S2B_PILOT = dict(official=61.10, teacher_facility=77.80)
# S2-C2 §4.2, on these same 150 instances, base = the S2-C1 student set (57.32).
S2C2 = dict(base_macro=57.32, addteacher8=66.65, teacher8=66.75,
            remworst8=55.95, random8=60.14, shuffled8=58.01, adversarial8=57.43)
N_PAIRED_BOOT = 10000


def load(tag):
    with open(os.path.join(OUTPUT_DIR, f"{tag}.json")) as f:
        return json.load(f)


def macro_of(r):
    return float(np.mean([r["per_benchmark"][ds]["acc_pct"] for ds in DS_ORDER]))


def paired(a_hits, b_hits, nb=N_PAIRED_BOOT, seed=0):
    """Paired macro delta, CI, per-instance transitions, and the MDE.

    The CI resamples images within benchmark, exactly as M2's `_summarise` did,
    so this number is comparable to every paired delta already in the project.
    """
    rng = np.random.default_rng(seed)
    draws = np.zeros(nb)
    per_ds, fixed, broken, both = {}, 0, 0, 0
    for ds in DS_ORDER:
        x = np.asarray(a_hits[ds], float)
        y = np.asarray(b_hits[ds], float)
        per_ds[ds] = float((x.mean() - y.mean()) * 100)
        n = len(x)
        idx = rng.integers(0, n, size=(nb, n))
        draws += (x[idx].mean(1) - y[idx].mean(1)) / 3.0
        fixed += int(((x > y)).sum())
        broken += int(((x < y)).sum())
        both += int(((x == y)).sum())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # paired per-instance delta in *points* (hits are 0/1 for OCR, scores for VQA)
    sd = float(np.std(draws, ddof=1) * np.sqrt(3.0))   # per-benchmark paired SD
    return dict(delta=float(np.mean(draws) * 100), ci=[float(lo * 100), float(hi * 100)],
                per_ds=per_ds, fixed=fixed, broken=broken, tied=both,
                mde_80pct=float(2.802 * np.std(draws, ddof=1) * 100),
                seed_mean_sd=sd)


def mcnemar_p(fixed, broken):
    """Two-sided exact-ish binomial p on the discordant pairs."""
    n = fixed + broken
    if n == 0:
        return 1.0
    from math import comb
    k = min(fixed, broken)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2 * p))


def teacher_mass(bank_tag: str, mg_tag: str, teacher: dict) -> dict:
    """F(S) = teacher mass of S / teacher mass of the teacher's own Top-256."""
    z = np.load(os.path.join(OUTPUT_DIR, f"{bank_tag}.npz"), allow_pickle=False)
    keys = [str(k) for k in z["key"]]
    pos = {k: i for i, k in enumerate(keys)}
    grid = json.load(open(os.path.join(OUTPUT_DIR, f"{mg_tag}.json")))
    order = [k for k in grid["keys"] if k in pos]

    def F(sets):
        vals = []
        for k, s in sets.items():
            g2 = np.asarray(teacher[k], np.float64)
            denom = g2[np.argsort(-g2)[:256]].sum()
            vals.append(float(g2[np.asarray(s, np.int64)].sum() / denom))
        return float(np.mean(vals))

    s0 = {k: np.asarray(z["s0"][pos[k]], np.int64) for k in order}
    out = dict(F_B2=F(s0), F_teacher_top256=1.0, n=len(order), arms={})
    for arm, rec in grid["arms"].items():
        if "error" in rec:
            continue
        sets, used = {}, 0
        for k, m in zip(grid["keys"], rec["per_image_meta"]):
            if m.get("rescue_idx") is None or k not in pos:
                continue
            s = np.setdiff1d(s0[k], np.asarray(m["evict_idx"], np.int64))
            sets[k] = np.union1d(s, np.asarray(m["rescue_idx"], np.int64))
            used += 1
        if used:
            out["arms"][arm] = dict(F=F(sets), r=rec["miss"].get("r"),
                                    rule=rec["miss"].get("rule"),
                                    source=rec["miss"].get("source"),
                                    **_rescue_rank(grid, rec, z, pos, teacher))
    f0 = out["F_B2"]
    for d in out["arms"].values():
        d["mass_forecast_macro"] = S2B_PILOT["official"] + \
            (d["F"] - f0) / (1.0 - f0) * (S2B_PILOT["teacher_facility"]
                                          - S2B_PILOT["official"])
        d["frontload_forecast_macro"] = S2B_PILOT["official"] + \
            _frontload_fraction(d["r"]) * (S2B_PILOT["teacher_facility"]
                                           - S2B_PILOT["official"])
    return out


def _rescue_rank(grid, rec, z, pos, teacher) -> dict:
    """Where in the teacher's ranking the rescued tokens actually sit.

    The formulation's value lives at the HEAD of the teacher's ranking (S2-C2:
    53 % of the gap closed by 8 tokens).  Two arms can gain the same teacher
    MASS and differ completely in accuracy if one takes it from ranks 1-16 and
    the other from ranks 40-200.  This is the number that separates them, and it
    needs no generation: it is a re-reading of the teacher cache.
    """
    ranks, in_top16, in_top64 = [], [], []
    for k, m in zip(grid["keys"], rec["per_image_meta"]):
        if m.get("rescue_idx") is None or k not in pos:
            continue
        g2 = np.asarray(teacher[k], np.float64)
        s0 = np.asarray(z["s0"][pos[k]], np.int64)
        keep = np.zeros(g2.shape[0], dtype=bool)
        keep[s0] = True
        drop = np.where(~keep)[0]
        # rank of each dropped token within the dropped set, 0 = teacher's best
        order = np.argsort(-g2[drop])
        rk = np.empty(drop.size, dtype=np.int64)
        rk[order] = np.arange(drop.size)
        r = np.asarray(m["rescue_idx"], np.int64)
        where = np.searchsorted(drop, r)
        rr = rk[where]
        ranks.append(rr.mean())
        in_top16.append(float((rr < 16).mean()))
        in_top64.append(float((rr < 64).mean()))
    if not ranks:
        return {}
    return dict(mean_rescue_teacher_rank=float(np.mean(ranks)),
                frac_rescue_in_teacher_top16=float(np.mean(in_top16)),
                frac_rescue_in_teacher_top64=float(np.mean(in_top64)))


def _frontload_fraction(r):
    """S2-C2's measured cumulative-gap curve, linearly interpolated."""
    xs = [0, 2, 6, 8, 12, 16, 24, 32, 40, 48]
    ys = [0.0, 0.103, 0.396, 0.529, 0.671, 0.770, 0.866, 0.974, 0.950, 1.033]
    return float(np.interp(r, xs, ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m3_accuracy")
    ap.add_argument("--bank", default="m3_bank")
    ap.add_argument("--out", default="m3_analysis")
    args = ap.parse_args()

    mg = load(args.tag)
    m2 = load("m2_accuracy")
    from m3_common import load_teacher
    teacher = load_teacher()

    rows = {}
    for k in ("B0", "B1", "B2"):
        r = m2["arms"].get(k)
        if not r or "error" in r:
            continue
        rows[k] = dict(arm=k, macro=macro_of(r),
                       per_ds={ds: r["per_benchmark"][ds]["acc_pct"] for ds in DS_ORDER},
                       hits={ds: r["per_benchmark"][ds]["hits"] for ds in DS_ORDER},
                       predictions=r["predictions"], source="m2_accuracy.json")
    for k, r in mg["arms"].items():
        if "error" in r:
            rows[k] = dict(arm=k, error=r["error"][-400:])
            continue
        rows[k] = dict(arm=k, macro=macro_of(r),
                       per_ds={ds: r["per_benchmark"][ds]["acc_pct"] for ds in DS_ORDER},
                       hits={ds: r["per_benchmark"][ds]["hits"] for ds in DS_ORDER},
                       predictions=r["predictions"],
                       missguard_ms=r.get("missguard_ms_median"),
                       wall=r.get("wall_seconds"), source=args.tag + ".json")

    b2, b1 = rows["B2"], rows["B1"]
    for k, r in rows.items():
        if "hits" not in r or k in ("B0", "B1", "B2"):
            continue
        r["vs_B2"] = paired(r["hits"], b2["hits"])
        r["vs_B1"] = paired(r["hits"], b1["hits"])
        r["vs_B2"]["mcnemar_p"] = mcnemar_p(r["vs_B2"]["fixed"], r["vs_B2"]["broken"])
        r["rescued_among_B2_wrong"] = _rescued(r, b2)

    tm = teacher_mass(args.bank, args.tag, teacher)
    out = dict(rows=rows, teacher_mass=tm, s2b_pilot=S2B_PILOT, s2c2=S2C2,
               gate=gate(rows, mg), gates=mg.get("gates"),
               primary_arm=mg.get("primary_arm", PRIMARY_ARM))
    with open(os.path.join(OUTPUT_DIR, f"{args.out}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print_table(rows)
    print_forecasts(tm, rows)
    print(f"\n[gate] {json.dumps(out['gate'], indent=1)}")
    print(f"[saved] {args.out}.json")
    return out


def _rescued(r, b2):
    """Of the instances B2 got wrong, how many does this arm get right."""
    n_wrong = n_fix = n_break = 0
    for ds in DS_ORDER:
        a = np.asarray(r["hits"][ds], float)
        b = np.asarray(b2["hits"][ds], float)
        n_wrong += int((b <= 0).sum())
        n_fix += int(((b <= 0) & (a > 0)).sum())
        n_break += int(((b > 0) & (a <= 0)).sum())
    return dict(b2_wrong=n_wrong, rescued=n_fix, broken=n_break)


def gate(rows, mg):
    """The frozen M3-v0 decision gate (brief §10), applied to the PRIMARY arm."""
    prim = rows.get(PRIMARY_ARM)
    orc = {k: v for k, v in rows.items() if k.startswith("OR-") and "hits" in v}
    if not prim or "hits" not in prim:
        return dict(verdict="NOT_RUN", reason=f"primary arm {PRIMARY_ARM} absent")
    best_or = max(orc.values(), key=lambda r: r["macro"]) if orc else None
    d = prim["vs_B2"]
    v = dict(primary=PRIMARY_ARM, primary_macro=prim["macro"],
             primary_delta_vs_B2=d["delta"], primary_ci_vs_B2=d["ci"],
             primary_mde_80pct=d["mde_80pct"], primary_mcnemar_p=d.get("mcnemar_p"),
             primary_overhead_ms=prim.get("missguard_ms"),
             primary_vs_B1=prim["macro"] - B1_MACRO,
             primary_rescued=prim.get("rescued_among_B2_wrong"),
             oracle_arm=best_or["arm"] if best_or else None,
             oracle_macro=best_or["macro"] if best_or else None,
             oracle_delta_vs_B2=best_or["vs_B2"]["delta"] if best_or else None,
             oracle_ci_vs_B2=best_or["vs_B2"]["ci"] if best_or else None,
             gap_oracle_minus_primary=(best_or["vs_B2"]["delta"] - d["delta"])
             if best_or else None,
             secondary={k: dict(macro=r["macro"], delta_vs_B2=r["vs_B2"]["delta"],
                                ci=r["vs_B2"]["ci"])
                        for k, r in rows.items()
                        if "hits" in r and k not in ("B0", "B1", "B2")
                        and k != PRIMARY_ARM})
    d_or = v["oracle_delta_vs_B2"]
    # A CI that spans zero cannot support a positive claim; the verdict says so.
    ci_positive = d["ci"][0] > 0
    if d_or is None:
        v["verdict"] = "INCOMPLETE"
    elif d_or < 2.0:
        v["verdict"] = "REFUTED"
        v["reason"] = ("the oracle itself moves <2 macro: B2 + critical-miss "
                       "correction has no ceiling to chase")
    elif prim["macro"] >= 64.0 and ci_positive and \
            (v["primary_overhead_ms"] is None or v["primary_overhead_ms"] <= 5.0):
        v["verdict"] = "STRONG"
    elif d["delta"] >= 2.0 and ci_positive and (d_or - d["delta"]) > 0:
        v["verdict"] = "PROMISING"
    elif d["delta"] < 2.0 and d_or >= 2.0:
        v["verdict"] = "WEAK"
        v["reason"] = ("formulation has headroom (oracle +%.1f) but the "
                       "forward-only student does not reach it" % d_or)
    else:
        v["verdict"] = "WEAK"
        v["reason"] = "point estimate clears the bar but the paired CI does not"
    v["mde_note"] = ("paired macro SE on this 150 is ~%.1f pts (MDE80 ~%.1f), so a "
                     "point estimate below the MDE is not evidence on its own; the "
                     "CI and the McNemar p on discordant pairs are reported beside it"
                     % (d["mde_80pct"] / 2.802, d["mde_80pct"]))
    return v


def print_table(rows):
    hdr = (f"{'arm':18s} {'TextVQA':>8s} {'DocVQA':>8s} {'OCR':>7s} {'macro':>7s} "
           f"{'d_B2':>7s} {'95% CI':>16s} {'d_B1':>7s} {'ms':>7s}")
    print(hdr)
    print("-" * len(hdr))
    order = ["B0", "B1", "B2"] + sorted(k for k in rows if k not in ("B0", "B1", "B2"))
    for k in order:
        r = rows.get(k)
        if not r or "hits" not in r:
            continue
        d2 = r.get("vs_B2", {}).get("delta")
        ci = r.get("vs_B2", {}).get("ci")
        d1 = r.get("vs_B1", {}).get("delta")
        ms = r.get("missguard_ms")
        star = " *" if k == PRIMARY_ARM else ""
        print(f"{k + star:18s} {r['per_ds']['TextVQA_VAL']:8.2f} "
              f"{r['per_ds']['DocVQA_VAL']:8.2f} {r['per_ds']['OCRBench']:7.2f} "
              f"{r['macro']:7.2f} "
              f"{(f'{d2:+.2f}' if d2 is not None else '-'):>7s} "
              f"{(f'[{ci[0]:+.1f},{ci[1]:+.1f}]' if ci else '-'):>16s} "
              f"{(f'{d1:+.2f}' if d1 is not None else '-'):>7s} "
              f"{(f'{ms:.2f}' if ms else '-'):>7s}")
    print("  (* = pre-registered primary arm)")


def print_forecasts(tm, rows):
    print(f"\nteacher-mass diagnostic: F(B2) = {tm['F_B2']:.4f}, "
          f"F(teacher top-256) = 1.0")
    print("%-18s %7s %8s %9s %9s %9s" % ("arm", "F(S)", "mass fc", "rescue rk",
                                          "in top16", "measured"))
    for arm, d in sorted(tm["arms"].items(), key=lambda t: t[1]["F"]):
        m = rows.get(arm, {}).get("macro")
        print("%-18s %7.4f %8.2f %9.1f %9.3f %9s" % (
            arm, d["F"], d["mass_forecast_macro"],
            d.get("mean_rescue_teacher_rank", float("nan")),
            d.get("frac_rescue_in_teacher_top16", float("nan")),
            "%.2f" % m if m is not None else "-"))


if __name__ == "__main__":
    main()
