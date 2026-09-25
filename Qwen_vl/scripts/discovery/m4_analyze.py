"""
M4-v0 step 3 -- tables, controls and the frozen verdict.

Reads stored artefacts only (no GPU).  Three things it is careful about:

1. **The verdict reads ONE pre-registered arm**, `m4_accuracy.PRIMARY_ARM`
   (REC-r16), frozen before the grid ran.  Every other arm is reported beside a
   control, never in place of one.
2. **Every threshold is printed with the resolution it needs.**  M3-v0 measured
   the paired macro SE on this 150 at ~2-4 points and the MDE at 80 % power at
   7.0, so a +2 point estimate is NOT evidence on its own.  The paired CI and
   the McNemar p on discordant pairs are printed next to every delta, and the
   brief's literal gate is reported alongside the statistical reading of it.
3. **The eviction is accounted for separately.**  `EVICT-r16` is the same
   eviction with nothing put back (240 tokens).  It is not a candidate -- it
   breaks the budget -- but `REC - EVICT` is what the capsules add and
   `MEAN - EVICT` is what any merge adds, so the two questions never blur.

Usage
    python scripts/discovery/m4_analyze.py
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
from m3_analyze import mcnemar_p, paired                             # noqa: E402
from m4_accuracy import PRIMARY_ARM, TAU, TAU_IMP                    # noqa: E402

B2_MACRO, B1_MACRO = 59.88, 61.10
# The brief's §10 gates, verbatim.
GATE_STRONG_MACRO = 64.0
GATE_STRONG_MARGIN_OVER_MEAN = 2.0
GATE_PROMISING_DELTA = 2.0
# A "stable" win over a control needs both a margin and a CI that excludes 0.
STABLE_MARGIN = 2.0

CONTROLS = ("MEAN-r16", "IMP-r16", "SHUF-r16", "ANCH-r16", "NORM-r16", "FPS-r16")


def load(tag):
    with open(os.path.join(OUTPUT_DIR, f"{tag}.json")) as f:
        return json.load(f)


def macro_of(r):
    return float(np.mean([r["per_benchmark"][ds]["acc_pct"] for ds in DS_ORDER]))


def rows_of(m4, m2):
    rows = {}
    for k in ("B0", "B1", "B2"):
        r = m2["arms"].get(k)
        if not r or "error" in r:
            continue
        rows[k] = dict(arm=k, macro=macro_of(r), kind="baseline",
                       per_ds={ds: r["per_benchmark"][ds]["acc_pct"] for ds in DS_ORDER},
                       hits={ds: r["per_benchmark"][ds]["hits"] for ds in DS_ORDER},
                       predictions=r["predictions"])
    for k, r in m4["arms"].items():
        if "error" in r:
            rows[k] = dict(arm=k, error=r["error"][-400:])
            continue
        rows[k] = dict(arm=k, macro=macro_of(r),
                       kind=("diagnostic" if r["rec"]["mode"] == "evict" else "arm"),
                       rec=r["rec"], rec_ms=r.get("rec_ms_median"),
                       wall=r.get("wall_seconds"),
                       per_ds={ds: r["per_benchmark"][ds]["acc_pct"] for ds in DS_ORDER},
                       hits={ds: r["per_benchmark"][ds]["hits"] for ds in DS_ORDER},
                       predictions=r["predictions"],
                       # carried so `structure()` can pair each instance's
                       # covariate with its own response; the per-image record
                       # is in `items` order while `hits` is grouped by
                       # benchmark, and the two must never be concatenated.
                       per_image_meta=r.get("per_image_meta"))
    return rows


def add_contrasts(rows):
    refs = [k for k in ("B2", "B1", "EVICT-r16") if k in rows and "hits" in rows[k]]
    for k, r in rows.items():
        if "hits" not in r or k in ("B0", "B1", "B2"):
            continue
        for ref in refs:
            if ref == k:
                continue
            d = paired(r["hits"], rows[ref]["hits"])
            d["mcnemar_p"] = mcnemar_p(d["fixed"], d["broken"])
            r[f"vs_{ref}"] = d
        r["rescued_among_B2_wrong"] = _rescued(r, rows["B2"])
        r["n_pred_changed_vs_B2"] = int(sum(
            1 for x, y in zip(r["predictions"], rows["B2"]["predictions"])
            if x.strip() != y.strip()))
        r["n_identical_vs_B2"] = int(sum(
            1 for x, y in zip(r["predictions"], rows["B2"]["predictions"])
            if x.strip() == y.strip()))


def _rescued(r, b2):
    """Of the instances B2 got wrong, how many does this arm get right."""
    n_wrong = n_fix = n_break = 0
    for ds in DS_ORDER:
        a = np.asarray(r["hits"][ds], float)
        b = np.asarray(b2["hits"][ds], float)
        n_wrong += int((b == 0).sum())
        n_fix += int(((a > 0) & (b == 0)).sum())
        n_break += int(((a == 0) & (b > 0)).sum())
    return dict(b2_wrong=n_wrong, fixed=n_fix, broken=n_break,
                net=n_fix - n_break,
                frac_fixed=float(n_fix / n_wrong) if n_wrong else None)


def trim_grid(src: str, dst: str):
    """Reduce a stored grid to the fields `reproducibility()` reads.

    The two pre-fix grids exist for ONE claim -- that the same arm scored
    differently in two fresh processes -- and that claim needs a macro, the
    per-benchmark hits and the prediction strings.  Keeping the full per-image
    records would commit ~12 MB to substantiate a 4-row table, so they are
    trimmed and the trim is itself reproducible through this function.
    """
    d = json.load(open(os.path.join(OUTPUT_DIR, src)))
    keep = {}
    for k, r in d["arms"].items():
        if "error" in r:
            keep[k] = dict(arm=k, error=r["error"][-400:])
            continue
        keep[k] = dict(arm=k, macro_pct=r["macro_pct"],
                       predictions=r["predictions"],
                       per_benchmark={ds: dict(hits=r["per_benchmark"][ds]["hits"],
                                               acc_pct=r["per_benchmark"][ds]["acc_pct"])
                                      for ds in DS_ORDER})
    out = dict(arms=keep, keys=d.get("keys"), trimmed_from=src,
               note="trimmed to the fields m4_analyze.reproducibility() reads")
    with open(os.path.join(OUTPUT_DIR, dst), "w") as f:
        json.dump(out, f)
    a = os.path.getsize(os.path.join(OUTPUT_DIR, src)) / 1e6
    b = os.path.getsize(os.path.join(OUTPUT_DIR, dst)) / 1e6
    print(f"  {src} {a:.1f} MB -> {dst} {b:.2f} MB")


def reproducibility():
    """Run-to-run spread of the SAME arm, same code, two fresh processes.

    The first M4 implementation pooled capsules with `Tensor.index_add_`, which
    reduces with atomicAdd on CUDA: the summation order varies between launches,
    the capsules differ by ~2e-7 in fp32, and a bf16 rounding (relative eps
    ~8e-3) occasionally flips a greedy token.  That grid was archived rather
    than reported and the pooling was rewritten as a deterministic segment sum
    (m4_common._group_sum); this function measures how large the defect was, so
    the number is on the record instead of being a claim in prose.

    It reads `m4_accuracy_nondet.json` (11 arms) against
    `m4_accuracy_rep_nondet.json` (3 of them re-run) -- both frozen artefacts.
    """
    a = os.path.join(OUTPUT_DIR, "m4_accuracy_nondet.json")
    b = os.path.join(OUTPUT_DIR, "m4_accuracy_rep_nondet.json")
    if not (os.path.exists(a) and os.path.exists(b)):
        return dict(checked=False, reason="archived non-deterministic grids absent")
    A = json.load(open(a))["arms"]
    B = json.load(open(b))["arms"]
    out = {}
    for k in sorted(set(A) & set(B)):
        ra, rb = A[k], B[k]
        if "error" in ra or "error" in rb:
            continue
        hits_same = sum(int((np.asarray(ra["per_benchmark"][ds]["hits"], float)
                             == np.asarray(rb["per_benchmark"][ds]["hits"], float)).sum())
                        for ds in DS_ORDER)
        pred_same = sum(1 for x, y in zip(ra["predictions"], rb["predictions"])
                        if x.strip() == y.strip())
        out[k] = dict(macro_run1=macro_of(ra), macro_run2=macro_of(rb),
                      macro_delta=macro_of(rb) - macro_of(ra),
                      hit_agreement=f"{hits_same}/150",
                      prediction_agreement=f"{pred_same}/150")
    out["note"] = ("the identity arm is bit-reproducible across processes; the "
                   "capsule arms were not, by up to %.1f macro -- which is why "
                   "the pooling was rewritten and the grid re-run"
                   % max((abs(v["macro_delta"]) for k, v in out.items()
                          if isinstance(v, dict)), default=0.0))
    return out


def structure(rows, tag):
    """Where does the capsule effect live, if it lives anywhere?

    Two per-instance covariates, both computed from artefacts already stored:
      * mean u over the r tokens the arm EVICTED.  A high value means the
        eviction removed tokens the retained set could not explain -- exactly
        the case where a capsule should matter most.
      * the residual spread inside the cells (max u - min u per cell, averaged).
    The response is the instance's own hit difference against B2, summed over
    the three benchmarks, so the split reads on the same 150 the table does.
    """
    path = os.path.join(OUTPUT_DIR, "m4_offline_umaps.npz")
    if not os.path.exists(path):
        return dict(checked=False, reason="no u-maps")
    z = np.load(path, allow_pickle=False)
    umap = {str(k): np.asarray(z["u_r16"][i], np.float32)
            for i, k in enumerate(z["key"])}
    prim = rows.get(PRIMARY_ARM)
    if not prim or "hits" not in prim:
        return dict(checked=False, reason="no primary arm")
    b2 = rows["B2"]
    # `per_benchmark[ds]["hits"]` is grouped BY BENCHMARK while `per_image_meta`
    # is in `items` order, and the benchmarks are interleaved in `items`.  A
    # per-benchmark running counter is therefore the only correct way to line
    # the two up; concatenating the benchmark arrays would silently pair an
    # instance's covariate with another benchmark's response.
    delta = {ds: (np.asarray(prim["hits"][ds], float)
                  - np.asarray(b2["hits"][ds], float)) for ds in DS_ORDER}
    seen = {ds: 0 for ds in DS_ORDER}
    diffs, cov_ev = [], []
    for m in prim.get("per_image_meta", []):
        ds = m["ds"]
        u = umap.get(m["key"])
        if u is None or m.get("evict_idx") is None:
            return dict(checked=False, reason=f"{m['key']} missing from u-maps")
        i = seen[ds]
        seen[ds] += 1
        diffs.append(float(delta[ds][i]))
        cov_ev.append(float(u[np.asarray(m["evict_idx"])].mean()))
    diffs = np.asarray(diffs)
    cov_ev = np.asarray(cov_ev)
    if len(diffs) != len(cov_ev) or len(diffs) != sum(len(v) for v in delta.values()):
        return dict(checked=False, reason="instance count mismatch")
    med = float(np.median(cov_ev))
    hi = cov_ev > med
    return dict(checked=True, n=len(diffs),
                mean_u_evicted=dict(median=med,
                                    p10=float(np.percentile(cov_ev, 10)),
                                    p90=float(np.percentile(cov_ev, 90))),
                delta_hi_u=float(diffs[hi].mean() * 100),
                delta_lo_u=float(diffs[~hi].mean() * 100),
                n_hi=int(hi.sum()), n_lo=int((~hi).sum()),
                pearson_r=float(np.corrcoef(cov_ev, diffs)[0, 1]),
                note="response = the instance's own (arm - B2) hit difference "
                     "summed over the three benchmarks; covariate = mean "
                     "unexplainedness of the r evicted tokens")


def _stable(delta, ci):
    """The operational reading of the brief's 'stably better'."""
    return bool(delta >= STABLE_MARGIN and ci[0] > 0)


def gate(rows):
    """The brief's §10 success gate, applied to the PRIMARY arm, plus the
    statistical reading of the same numbers."""
    prim = rows.get(PRIMARY_ARM)
    if not prim or "hits" not in prim:
        return dict(verdict="NOT_RUN", reason=f"primary arm {PRIMARY_ARM} absent")
    d = prim["vs_B2"]
    mean_ = rows.get("MEAN-r16", {})
    imp_ = rows.get("IMP-r16", {})
    shuf = rows.get("SHUF-r16", {})
    ev = rows.get("EVICT-r16", {})

    v = dict(
        primary=PRIMARY_ARM, primary_macro=prim["macro"],
        primary_delta_vs_B2=d["delta"], primary_ci_vs_B2=d["ci"],
        primary_mde_80pct=d["mde_80pct"], primary_mcnemar_p=d["mcnemar_p"],
        primary_rec_ms=prim.get("rec_ms"),
        primary_vs_B1=prim["macro"] - B1_MACRO,
        primary_rescued=prim.get("rescued_among_B2_wrong"),
        per_benchmark={ds: dict(macro=prim["per_ds"][ds],
                                b2=rows["B2"]["per_ds"][ds],
                                delta=prim["per_ds"][ds] - rows["B2"]["per_ds"][ds])
                       for ds in DS_ORDER},
        controls={k: dict(macro=rows[k]["macro"],
                          delta_vs_B2=rows[k]["vs_B2"]["delta"],
                          ci_vs_B2=rows[k]["vs_B2"]["ci"],
                          delta_vs_primary=rows[k]["vs_" + PRIMARY_ARM]["delta"]
                          if ("vs_" + PRIMARY_ARM) in rows[k] else None)
                  for k in CONTROLS if k in rows and "hits" in rows[k]},
        secondary={k: dict(macro=r["macro"], delta_vs_B2=r["vs_B2"]["delta"],
                           ci_vs_B2=r["vs_B2"]["ci"], rec=r.get("rec"))
                   for k, r in rows.items()
                   if "hits" in r and r.get("kind") == "arm" and k != PRIMARY_ARM},
    )
    if "hits" in mean_:
        v["primary_minus_mean"] = paired(prim["hits"], mean_["hits"])
        v["primary_minus_mean"]["mcnemar_p"] = mcnemar_p(
            v["primary_minus_mean"]["fixed"], v["primary_minus_mean"]["broken"])
    if "hits" in imp_:
        v["primary_minus_imp"] = paired(prim["hits"], imp_["hits"])
    if "hits" in shuf:
        v["primary_minus_shuf"] = paired(prim["hits"], shuf["hits"])
    if "hits" in ev:
        v["capsule_gain_over_evict"] = paired(prim["hits"], ev["hits"])
        v["mean_gain_over_evict"] = paired(mean_["hits"], ev["hits"])
        v["evict_delta_vs_B2"] = ev["vs_B2"]["delta"]
        v["evict_ci_vs_B2"] = ev["vs_B2"]["ci"]
        v["evict_note"] = ("EVICT-r16 is 240 tokens, not 256: it is an accounting "
                           "diagnostic, never a candidate")

    # ---- the brief's literal gate -----------------------------------------
    beats_b2 = d["delta"] >= GATE_PROMISING_DELTA
    beats_b2_stable = _stable(d["delta"], d["ci"])
    mean_d = v.get("primary_minus_mean")
    imp_d = v.get("primary_minus_imp")
    resid_beats_mean = bool(mean_d and _stable(mean_d["delta"], mean_d["ci"]))
    resid_beats_imp = bool(imp_d and _stable(imp_d["delta"], imp_d["ci"]))
    any_merge_beats_b2 = any(
        rows[k]["vs_B2"]["delta"] >= GATE_PROMISING_DELTA and
        rows[k]["vs_B2"]["ci"][0] > 0
        for k in CONTROLS + (PRIMARY_ARM,) if k in rows and "hits" in rows[k])

    if prim["macro"] >= GATE_STRONG_MACRO and \
            mean_d and mean_d["delta"] >= GATE_STRONG_MARGIN_OVER_MEAN:
        verdict = "STRONG"
    elif beats_b2 and (resid_beats_mean or resid_beats_imp):
        verdict = "PROMISING"
    elif any_merge_beats_b2:
        verdict = "MERGING-ONLY"
    else:
        verdict = "REFUTED"
    v["verdict"] = verdict
    v["reading"] = dict(
        primary_macro_over_64=bool(prim["macro"] >= GATE_STRONG_MACRO),
        primary_over_mean_by_2=bool(mean_d and mean_d["delta"] >= GATE_STRONG_MARGIN_OVER_MEAN),
        primary_beats_b2_by_2=bool(beats_b2),
        primary_beats_b2_stably=beats_b2_stable,
        residual_beats_mean_stably=resid_beats_mean,
        residual_beats_imp_stably=resid_beats_imp,
        any_merge_beats_b2_stably=bool(any_merge_beats_b2),
        mde_note=("paired macro SE on this 150 is ~%.1f pts (MDE80 ~%.1f), so a "
                  "point estimate below the MDE is not evidence on its own; the "
                  "paired CI and the McNemar p on discordant pairs decide"
                  % (d["mde_80pct"] / 2.802, d["mde_80pct"])))
    return v


def print_table(rows):
    print(f"\n{'arm':14s} {'TVQA':>7s} {'DocVQA':>7s} {'OCR':>7s} {'macro':>7s} "
          f"{'dB2':>7s} {'CI vs B2':>16s} {'p':>6s} {'fix/brk':>9s} "
          f"{'newpred':>7s} {'ms':>6s}")
    order = ["B0", "B1", "B2"] + sorted(k for k in rows if k not in ("B0", "B1", "B2"))
    for k in order:
        r = rows.get(k)
        if not r or "hits" not in r:
            continue
        d = r.get("vs_B2")
        ds = r["per_ds"]
        delta = f"{d['delta']:+7.2f}" if d else "   -   "
        ci = f"[{d['ci'][0]:+5.1f},{d['ci'][1]:+5.1f}]" if d else " " * 16
        p = f"{d['mcnemar_p']:.3f}" if d and d.get("mcnemar_p") is not None else "  -  "
        fb = f"{d['fixed']:3d}/{d['broken']:<3d}" if d else "   -/-  "
        ms = f"{r['rec_ms']:.2f}" if r.get("rec_ms") is not None else "  -  "
        np_ = r.get("n_pred_changed_vs_B2")
        nps = f"{np_:3d}/150" if np_ is not None else "   -   "
        print(f"{k:14s} {ds['TextVQA_VAL']:7.3f} {ds['DocVQA_VAL']:7.3f} "
              f"{ds['OCRBench']:7.3f} {r['macro']:7.3f} {delta} "
              f"{ci} {p} {fb} {nps} {ms}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m4_accuracy")
    ap.add_argument("--out", default="m4_analysis")
    ap.add_argument("--trim", action="store_true",
                    help="rewrite the archived pre-fix grids in place, trimmed")
    args = ap.parse_args()

    if args.trim:
        for n in ("m4_accuracy_nondet", "m4_accuracy_rep_nondet"):
            p = os.path.join(OUTPUT_DIR, f"{n}.json")
            if os.path.exists(p):
                trim_grid(f"{n}.json", f"{n}.json.tmp")
                os.replace(os.path.join(OUTPUT_DIR, f"{n}.json.tmp"), p)
        return None

    m4 = load(args.tag)
    m2 = load("m2_accuracy")
    rows = rows_of(m4, m2)
    add_contrasts(rows)
    struct = structure(rows, args.tag)
    repro = reproducibility()
    # The per-image records belong to the grid file, not here; keeping them in
    # both would double an 11 MB artefact to say the same thing twice.
    for r in rows.values():
        r.pop("per_image_meta", None)
    out = dict(rows=rows, gate=gate(rows), gates=m4.get("gates"),
               primary_arm=PRIMARY_ARM, tau=TAU, tau_imp=TAU_IMP,
               structure=struct, reproducibility=repro,
               offline=load("m4_offline") if
               os.path.exists(os.path.join(OUTPUT_DIR, "m4_offline.json")) else None)
    with open(os.path.join(OUTPUT_DIR, f"{args.out}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print_table(rows)
    if "reading" in out["gate"]:
        print(f"\n[gate] {json.dumps(out['gate']['reading'], indent=1)}")
    else:
        print(f"\n[gate] not evaluated: {out['gate'].get('reason')}")
    print(f"[verdict] {out['gate']['verdict']}")
    if out.get("structure"):
        print(f"[structure] {json.dumps(out['structure'], indent=1)}")
    if out.get("reproducibility"):
        print(f"[reproducibility] {json.dumps(out['reproducibility'], indent=1)}")
    print(f"[saved] {args.out}.json")
    return out


if __name__ == "__main__":
    main()
