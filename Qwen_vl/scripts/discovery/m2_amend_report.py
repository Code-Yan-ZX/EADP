"""
M2 — implementation-defect amendment: the valid-arms report and the frozen
decision reading.

Reads m2_accuracy.json (excluding every record marked INVALID_*), the paired
performance file when it exists, and emits exactly the quantities the
amendment §2/§5 require:

    per-benchmark accuracy · macro · seed mean/range · deltas vs B1/B2 with
    paired bootstrap CIs · EOS rate · max-length hit rate · rescue /
    newly-broken counts · accuracy-performance Pareto position · and the
    frozen decision A/B/C/D over {corrected C1-P, C1-R}.

Bootstrap rule identical to m2_consolidate (stratified over the three
benchmarks, 10 000 draws, seed 0), so amendment numbers are comparable with
the pre-registration's.

Usage
    python scripts/discovery/m2_amend_report.py
    python scripts/discovery/m2_amend_report.py --keys C1-R C1-P
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR                                        # noqa: E402
from m2_gdep import dump_json                                        # noqa: E402

DS_ORDER = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]
SEEDS = (0, 1, 2)
NBOOT = 10000
MAX_NEW = 2048
CAP_CHARS = 2000          # consolidation's degeneracy proxy, kept for old rows


def load(name):
    p = os.path.join(OUTPUT_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else None


def valid(rec):
    return rec is not None and "error" not in rec and not str(
        rec.get("validity", "")).startswith("INVALID")


def hits_of(rec):
    return {ds: np.asarray(rec["per_benchmark"][ds]["hits"], float)
            for ds in DS_ORDER}


def macro(h):
    return float(np.mean([h[ds].mean() for ds in DS_ORDER])) * 100.0


def boot(a, b, seed=0):
    rng = np.random.default_rng(seed)
    draws = np.zeros(NBOOT)
    for ds in DS_ORDER:
        n = len(a[ds])
        idx = rng.integers(0, n, size=(NBOOT, n))
        draws += (a[ds][idx].mean(1) - b[ds][idx].mean(1)) / 3.0
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [float(lo * 100), float(hi * 100)]


def term_stats(rec):
    """EOS rate and max-length hit rate. Exact where n_decode was recorded
    (the amendment's runners do), char-proxy otherwise."""
    meta = rec.get("per_image_meta", [])
    nd = [m.get("n_decode") for m in meta]
    if all(x is not None for x in nd):
        eos = [1 if x < MAX_NEW else 0 for x in nd]
        capped = [1 if x >= MAX_NEW else 0 for x in nd]
        basis = "n_decode"
    else:
        preds = rec.get("predictions", [])
        capped = [1 if len(p) > CAP_CHARS else 0 for p in preds]
        eos = [1 - c for c in capped]
        basis = "char-length proxy"
    return dict(eos_rate=float(np.mean(eos)), capped_rate=float(np.mean(capped)),
                n_capped=int(sum(capped)), basis=basis)


def transitions(a, b, cut=0.5):
    A = np.concatenate([a[ds] for ds in DS_ORDER])
    B = np.concatenate([b[ds] for ds in DS_ORDER])
    ac, bc = A >= cut, B >= cut
    return dict(rescued=int((~bc & ac).sum()),
                newly_broken=int((bc & ~ac).sum()),
                both_correct=int((bc & ac).sum()),
                both_wrong=int((~bc & ~ac).sum()), n=int(len(A)))


def arm_block(A, arm, ref_recs, label):
    """Everything the amendment asks for, for one arm across its seeds."""
    seeds_present = [s for s in SEEDS if valid(A.get(f"{arm}|s{s}"))]
    per_seed = []
    for s in seeds_present:
        r = A[f"{arm}|s{s}"]
        h = hits_of(r)
        e = dict(seed=s, macro_pct=macro(h),
                 per_benchmark={ds: float(h[ds].mean() * 100) for ds in DS_ORDER},
                 n_kept_mean=r.get("n_kept_mean"),
                 **{f"vs_{lab}": dict(
                        delta_pts=float(macro(h) - macro(hits_of(ref))),
                        ci=boot(h, hits_of(ref)),
                        transitions=transitions(h, hits_of(ref)))
                    for lab, ref in ref_recs.items()})
        e.update(term_stats(r))
        per_seed.append(e)
    if not per_seed:
        return None
    macros = [x["macro_pct"] for x in per_seed]
    out = dict(arm=arm, n_seeds=len(per_seed), seeds=seeds_present,
               per_seed=per_seed,
               macro_seed_mean=float(np.mean(macros)),
               macro_seed_range=[float(min(macros)), float(max(macros))])
    # seed-averaged hits, paired against each reference
    avg = {ds: np.mean([hits_of(A[f"{arm}|s{x['seed']}"])[ds] for x in per_seed],
                       axis=0) for ds in DS_ORDER}
    out["seed_avg_hits"] = {ds: [float(v) for v in avg[ds]] for ds in DS_ORDER}
    for lab, ref in ref_recs.items():
        out[f"vs_{lab}_seedavg"] = dict(
            delta_pts=float(macro(avg) - macro(hits_of(ref))),
            ci=boot(avg, hits_of(ref)),
            transitions=transitions(avg, hits_of(ref)))
    out["eos_rate_mean"] = float(np.mean([x["eos_rate"] for x in per_seed]))
    out["capped_rate_mean"] = float(np.mean([x["capped_rate"] for x in per_seed]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", default="m2_accuracy.json")
    ap.add_argument("--perf-paired", default="m2_perf_paired.json")
    ap.add_argument("--perf-first", default="m2_perf.json")
    ap.add_argument("--arms", nargs="+", default=["C1-R", "C1-P"])
    ap.add_argument("--proxy-baseline-macro", type=float, default=None,
                    help="old cached-proxy expectation for decision B; interim "
                         "§6 used ~57 (S2-B/C1 translation of the 78.8 vs 66.9 "
                         "gradient result onto the held-out scale)")
    ap.add_argument("--out", default="m2_amend_report.json")
    args = ap.parse_args()

    acc = load(args.acc)
    A = acc["arms"]
    invalid_runs = sorted(k for k, r in A.items()
                          if str(r.get("validity", "")).startswith("INVALID"))
    assert valid(A.get("B1")), "B1 must be a valid reference"
    refs = {"B1": A["B1"]}
    if valid(A.get("B2")):
        refs["B2"] = A["B2"]

    rep = dict(stage="M2 amendment report",
               excluded_invalid_runs=invalid_runs,
               note=("Records carrying a validity marker are excluded from every "
                     "number here, per docs/scoring_search_m2_system_amendment.md."),
               baselines={t: dict(macro_pct=macro(hits_of(A[t])),
                                  per_benchmark={ds: float(
                                      hits_of(A[t])[ds].mean() * 100)
                                      for ds in DS_ORDER})
                          for t in ("B0", "B1", "B2") if valid(A.get(t))})

    blocks = {}
    for arm in args.arms:
        b = arm_block(A, arm, refs, arm)
        if b:
            blocks[arm] = b
    rep["arms"] = blocks

    # P' vs R head-to-head (frozen decision D)
    if valid(A.get("C1-P|s0")) and valid(A.get("C1-R|s0")):
        P = {ds: np.mean([hits_of(A[f"C1-P|s{s}"])[ds] for s in SEEDS], axis=0)
             for ds in DS_ORDER}
        R = {ds: np.mean([hits_of(A[f"C1-R|s{s}"])[ds] for s in SEEDS], axis=0)
             for ds in DS_ORDER}
        rep["P_minus_R"] = dict(delta_pts=float(macro(P) - macro(R)),
                                ci=boot(P, R),
                                transitions=transitions(P, R))

    # ---- performance position ----------------------------------------------
    perf = load(args.perf_paired)
    if perf:
        rep["performance"] = {
            t: dict(ttft_median_ms=s["ttft"]["median"],
                    ttft_p10_p90=[s["ttft"]["p10"], s["ttft"]["p90"]],
                    model_only_median_ms=s["model_only"]["median"],
                    decode32_median_ms=s["decode32"]["median"],
                    selector_and_prune_ms=float(
                        sum(s["stages"].get(k, {"median": 0})["median"]
                            for k in ("eadp_scoring_ms", "selector_ms",
                                      "L0_L4_ms", "scorer_ms",
                                      "token_compaction_ms"))))
            for t, s in perf["summary"].items()}
        rep["performance_basis"] = "paired interleaved (m2_perf_paired.json)"
    else:
        first = load(args.perf_first)
        if first:
            rep["performance"] = {
                t: dict(ttft_median_ms=r["prefill"]["wall"]["median"],
                        ttft_p10_p90=[None, r["prefill"]["wall"]["p90"]],
                        model_only_median_ms=None,
                        decode32_median_ms=r["decode"]["total_cuda"]["median"],
                        selector_and_prune_ms=None)
                for t, r in first["arms"].items() if "prefill" in r}
            rep["performance_basis"] = ("PRELIMINARY sequential grid "
                                        "(m2_perf.json), selector window "
                                        "double-counted for B1/B2; awaiting "
                                        "the paired benchmark")

    # ---- frozen decision reading (amendment §5) ----------------------------
    b1_macro = macro(hits_of(A["B1"]))
    pm = blocks.get("C1-P", {}).get("macro_seed_mean")
    rm = blocks.get("C1-R", {}).get("macro_seed_mean")
    d = dict(b1_macro=b1_macro, P_mean=pm, R_mean=rm,
             P_minus_B1=None if pm is None else pm - b1_macro,
             R_minus_B1=None if rm is None else rm - b1_macro,
             threshold_pts=1.0)
    readings = []
    if pm is not None or rm is not None:
        okP = pm is not None and pm >= b1_macro - 1.0
        okR = rm is not None and rm >= b1_macro - 1.0
        if okP or okR:
            readings.append(("A", "position policy(ies) at or within 1.0 pt of B1 "
                             "enter the selector comparison (block8/facility)"))
        else:
            proxy = args.proxy_baseline_macro
            if proxy is not None:
                above = [t for t, m in (("P′", pm), ("R", rm))
                         if m is not None and m > proxy + 1.0]
                if above:
                    readings.append(("B", "both trail B1 by > 1.0 but "
                                     f"{', '.join(above)} sit clearly above the "
                                     "cached-proxy expectation: early-pruning "
                                     "graph/objective mismatch; do not run the "
                                     "selector grid"))
                else:
                    readings.append(("C", "both far below B1 and not above the "
                                     "cached-proxy expectation: stop the GDEP "
                                     "L4 method; no SHUF/selector sweep"))
            else:
                readings.append(("B-or-C", "both trail B1 by > 1.0; supply "
                                 "--proxy-baseline-macro to separate B from C"))
    if "P_minus_R" in rep:
        d2 = abs(rep["P_minus_R"]["delta_pts"])
        d["P_minus_R"] = rep["P_minus_R"]["delta_pts"]
        if d2 > 1.0:
            readings.append(("D", f"|P′−R| = {d2:.2f} pt > 1.0: the position "
                             "policy is a method design variable and must be "
                             "analysed as its own object; do not report the "
                             "higher of the two as the sole result"))
    d["readings"] = readings
    rep["decision"] = d

    dump_json(args.out, rep)

    # ---- human-readable -----------------------------------------------------
    L = [f"# M2 amendment report — valid arms only\n",
         f"Excluded INVALID records: {', '.join(invalid_runs) or '(none)'}\n",
         f"Baselines: " + "  ".join(
             f"{t} macro {v['macro_pct']:.2f}" for t, v in rep["baselines"].items()) + "\n"]
    for arm, b in blocks.items():
        L.append(f"\n## {arm}  ({b['n_seeds']} seeds: {b['seeds']})\n")
        L.append("| seed | TextVQA | DocVQA | OCRBench | macro | Δ vs B1 [CI] | EOS | capped |")
        L.append("|---|---|---|---|---|---|---|---|")
        for x in b["per_seed"]:
            v = x["vs_B1"]
            L.append(f"| {x['seed']} | {x['per_benchmark']['TextVQA_VAL']:.2f} "
                     f"| {x['per_benchmark']['DocVQA_VAL']:.2f} "
                     f"| {x['per_benchmark']['OCRBench']:.2f} | {x['macro_pct']:.2f} "
                     f"| {v['delta_pts']:+.2f} [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}] "
                     f"| {x['eos_rate']*100:.1f}% | {x['n_capped']} |")
        L.append(f"\nseed mean macro **{b['macro_seed_mean']:.2f}** "
                 f"(range {b['macro_seed_range'][0]:.2f}–{b['macro_seed_range'][1]:.2f})")
        for lab in refs:
            k = f"vs_{lab}_seedavg"
            if k in b:
                t = b[k]["transitions"]
                L.append(f"vs {lab} (seed-avg): {b[k]['delta_pts']:+.2f} pt "
                         f"[{b[k]['ci'][0]:+.2f},{b[k]['ci'][1]:+.2f}], "
                         f"rescued {t['rescued']}, newly broken {t['newly_broken']}")
    if "P_minus_R" in rep:
        t = rep["P_minus_R"]["transitions"]
        L.append(f"\nP′−R: {rep['P_minus_R']['delta_pts']:+.2f} pt "
                 f"[{rep['P_minus_R']['ci'][0]:+.2f},{rep['P_minus_R']['ci'][1]:+.2f}] "
                 f"(rescued {t['rescued']}, newly broken {t['newly_broken']})")
    if "performance" in rep:
        L.append(f"\n## Accuracy–performance position  ({rep['performance_basis']})\n")
        L.append("| arm | macro (seed avg) | TTFT med (ms) | decode32 (ms) |")
        L.append("|---|---|---|---|")
        macs = {t: rep["baselines"][t]["macro_pct"] for t in rep["baselines"]}
        for arm, b in blocks.items():
            macs[arm] = b["macro_seed_mean"]
        for t, p in rep["performance"].items():
            if t in macs:
                L.append(f"| {t} | {macs[t]:.2f} | {p['ttft_median_ms']:.1f} "
                         f"| {p['decode32_median_ms']:.1f} |")
    L.append("\n## Frozen decision reading\n")
    L.append(json.dumps(rep["decision"], indent=1))
    md = "\n".join(L) + "\n"
    with open(os.path.join(OUTPUT_DIR, "m2_amend_report.md"), "w") as f:
        f.write(md)
    print(md)
    print(f"[saved] {args.out} + m2_amend_report.md")


if __name__ == "__main__":
    main()
