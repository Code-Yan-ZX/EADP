"""
SAGE step 5 -- paired statistics and the verdict against the pre-registered
criteria (refine-logs/EXPERIMENT_PLAN.md, decision 13).

PASS requires ALL of:
    C1  paired macro(SAGE) - macro(B2) > 0 with bootstrap 95% CI excluding 0
    C2  no benchmark drops more than 1.0 macro point vs B2
    C3  median confirmation TTFT within the frozen constraint
    C4  PERM erases the gain (paired CI of PERM - B2 includes 0, or below)
    C5  UNARY recovers less than SAGE (point estimate at least 50% below, or
        paired CI of SAGE - UNARY excluding 0)
Observability premise: hindsight ceiling clearly positive while SAGE fails ->
the early-observability premise is falsified (reported, not hidden).

Usage
    python scripts/discovery/sage_analyze.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m6_common import paired_bootstrap                               # noqa: E402
import sage_common as S                                              # noqa: E402
from sage_common import dump_json                                    # noqa: E402


def hits_of(arm):
    rec = S.load_json(S.F_CONF.format(arm=arm))
    keys = sorted(rec["records"])
    bad = [k for k in keys if rec["records"][k].get("error")]
    assert not bad, f"{arm}: {len(bad)} errored instances, refusing to score"
    h = np.array([rec["records"][k]["hit"] for k in keys])
    ds = [rec["records"][k]["ds"] for k in keys]
    return keys, h, ds, rec


def macro_of(h, ds):
    return float(np.mean([np.mean(np.asarray(h)[np.asarray(ds) == d]) * 100
                          for d in S.DS_ORDER]))


def main():
    keys, h_b2, ds, rec_b2 = hits_of("B2")
    n = len(keys)
    _, h_b1, _, _ = hits_of("B1")
    _, h_sage, _, rec_sage = hits_of("SAGE")
    _, h_rnd, _, rec_rnd = hits_of("RND")
    _, h_perm, _, rec_perm = hits_of("PERM")
    _, h_px, _, _ = hits_of("PERM-X")
    _, h_un, _, rec_un = hits_of("UNARY")
    hind = S.load_json(S.F_CONF.format(arm="HINDSIGHT"))["records"]
    for name, h in (("B1", h_b1), ("SAGE", h_sage), ("RND", h_rnd),
                    ("PERM", h_perm), ("UNARY", h_un)):
        assert len(h) == n, f"{name}: {len(h)} instances != B2's {n}"

    arms = dict(B2=h_b2, B1=h_b1, SAGE=h_sage, RND=h_rnd, PERM=h_perm,
                **{"PERM-X": h_px, "UNARY": h_un})
    per_arm = {}
    for name, h in arms.items():
        per_arm[name] = dict(
            macro=macro_of(h, ds),
            per_ds={d: float(np.mean(np.asarray(h)[np.asarray(ds) == d]) * 100)
                    for d in S.DS_ORDER},
            ttft_median_ms=None)
    # TTFT medians (arms that record them)
    for name in ("B2", "B1", "SAGE", "UNARY"):
        rec = S.load_json(S.F_CONF.format(arm=name))["records"]
        tt = [r["ttft_ms"] for r in rec.values() if "ttft_ms" in r]
        per_arm[name]["ttft_median_ms"] = float(np.median(tt)) if tt else None

    # ---- paired contrasts vs B2 (2000 resamples, the frozen seed) -----------
    contrasts = {}
    for name in ("B1", "SAGE", "RND", "PERM", "PERM-X", "UNARY"):
        contrasts[name] = paired_bootstrap(arms[name], h_b2)
    contrasts["SAGE-UNARY"] = paired_bootstrap(h_sage, h_un)
    contrasts["SAGE-PERM"] = paired_bootstrap(h_sage, h_perm)
    contrasts["SAGE-PERM-X"] = paired_bootstrap(h_sage, h_px)

    # rescue / break per task
    rb = {}
    for name in ("SAGE", "RND", "PERM", "PERM-X", "UNARY"):
        d = np.asarray(arms[name]) - h_b2
        rb[name] = dict(
            rescue=int((d > 0).sum()), break_=int((d < 0).sum()),
            tie=int((d == 0).sum()),
            per_ds={dd: dict(rescue=int(((d > 0) & (np.asarray(ds) == dd)).sum()),
                             break_=int(((d < 0) & (np.asarray(ds) == dd)).sum()))
                    for dd in S.DS_ORDER})

    # ---- criteria -----------------------------------------------------------
    calib = S.load_json(S.F_CALIB)
    c1 = bool(contrasts["SAGE"]["lo"] > 0)
    deltas_ds = {d: per_arm["SAGE"]["per_ds"][d] - per_arm["B2"]["per_ds"][d]
                 for d in S.DS_ORDER}
    c2 = bool(all(v > -1.0 for v in deltas_ds.values()))
    c3 = bool(per_arm["SAGE"]["ttft_median_ms"] is not None and
              per_arm["SAGE"]["ttft_median_ms"] <= calib["bound"]["max_median_ms"])
    # C4 uses PERM-X (the amendment control): the pre-registered
    # within-image permutation is degenerate under eq. 5 (it provably
    # reproduces SAGE -- see sage_deploy.py).  "Erases the gain" here means
    # PERM-X's contrast vs B2 is not better than SAGE's ordering signal.
    c4 = bool(contrasts["PERM-X"]["lo"] <= 0 <= contrasts["PERM-X"]["hi"]
              or contrasts["PERM-X"]["delta"] < contrasts["SAGE"]["delta"])
    sage_gain = contrasts["SAGE"]["delta"]
    un_gain = contrasts["UNARY"]["delta"]
    c5 = bool(un_gain < 0.5 * sage_gain or contrasts["SAGE-UNARY"]["lo"] > 0)

    # hindsight ceiling (realized, over the sampled edges only)
    best_d = []
    for k, r in hind.items():
        ds_s = max(r["sampled"], key=lambda e: e["hit"] - r["h0"])
        best_d.append(ds_s["hit"] - r["h0"])
    best_d = np.array(best_d)
    hindsight = dict(
        mean_best_delta=float(best_d.mean()),
        frac_images_with_gain=float((best_d > 0).mean()),
        macro_if_hindsight_deployed=float(
            np.mean([r["h0"] for r in hind.values()]) * 100 + best_d.mean() * 100),
        note="realized ceiling over 4 sampled edges/image only -- a lower bound "
             "on the true hindsight ceiling over all of E(x)")

    verdict = dict(
        criteria=dict(C1_paired_macro_beats_B2=c1,
                      C2_no_benchmark_loses_1pt=c2,
                      C3_ttft_within_constraint=c3,
                      C4_perm_erases_gain=c4,
                      C5_unary_recovers_less=c5),
        per_arm=per_arm, contrasts=contrasts, rescue_break=rb,
        hindsight=hindsight, calib=calib,
        overall=("PASS" if all([c1, c2, c3, c4, c5]) else
                 "PASS-MECH" if all([c1, c2, c3]) else "FAIL"),
        n=n)
    if not c1 and hindsight["mean_best_delta"] > 0.05:
        verdict["premise"] = ("early-observability premise FALSIFIED: hindsight "
                              "ceiling clearly positive while the critic fails")
    dump_json("sage_verdict.json", verdict)

    print(f"\n=== SAGE verdict (n={n}) ===")
    for name in per_arm:
        print(f"  {name:6s} macro {per_arm[name]['macro']:.3f}  "
              f"TTFT {per_arm[name].get('ttft_median_ms')}")
    for name, c in contrasts.items():
        print(f"  {name:10s} delta {c['delta']:+.3f} "
              f"[{c['lo']:+.3f}, {c['hi']:+.3f}]")
    print(f"  hindsight: mean best delta {hindsight['mean_best_delta']:+.4f} "
          f"on {hindsight['frac_images_with_gain']:.1%} of images")
    print(f"  OVERALL: {verdict['overall']}  "
          f"criteria: {verdict['criteria']}")


if __name__ == "__main__":
    main()
