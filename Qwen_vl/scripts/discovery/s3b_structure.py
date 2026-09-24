"""
S3-B step 5 (CPU): Part C and the verdict.

Reads the frozen case bank, the base points, Part A's sweep, Part B's arm grid,
the non-rescued null stage and the crop probe, and answers the one question this
stage was asked about: do the teacher's accuracy-critical small sets have
repeatable, predictable structure?

Three tests, deliberately separated because they answer different things:

  T1  WITHIN INSTANCE, NULL VS NULL. Among the size-matched, window-matched,
      removal-matched random sets drawn for one instance, do the ones that
      happen to rescue differ from the ones that do not? Teacher rank is NOT
      confounded here (both groups are random draws from the same pool), so a
      positive result means inspectable structure predicts success. Primary
      test.
  T2  BUNDLE VS ITS OWN NULLS. Does the teacher's bundle differ structurally
      from same-window alternatives? Rank IS confounded by construction (the
      bundle is the lowest ranks of the window), so T2 is description, never
      evidence for STRUCTURED-BUNDLE.
  T3  ACROSS INSTANCES, PREFIX VS PREFIX AT MATCHED k. Do the k-token prefixes
      that rescue differ structurally from the k-token prefixes that do not?
      Both groups are the teacher's own first k misses, so rank window is
      matched; leave-one-instance-out single-threshold AUC, so no instance
      contributes to its own rule. This is the "could a selector find bundles"
      test.

The verdict is the mechanical reading of docs/s3b_prereg_rules.md section 5;
every clause is printed with the number that decided it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, fisher_exact, mannwhitneyu, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                         # noqa: E402
import s3b_common as B                                           # noqa: E402

NULL_KINDS = ("null_window", "null_uni", "null_out")
# pre-LLM-observable properties eligible to be "candidate structure variables"
CANDIDATE_FEATURES = [
    "sp_mean_cheb", "sp_mean_euc", "sp_min_cheb", "sp_frac_adjacent",
    "sp_bbox_over_k", "sp_bbox_aspect", "sp_ncomp8", "sp_comp_over_k",
    "sp_frac_same_row", "sp_frac_same_col", "sp_centroid_r",
    "sp_dist_backbone", "sp_nn_retained",
    "ft_mean_pair_cos_l4", "ft_cos_to_all_mean", "ft_l4_norm", "ft_l2_norm",
    "ft_dnorm", "ft_cos_l2_l4", "ft_sim_sel_S", "ft_sim_sel_T", "ft_nb_sim",
    "ft_uniq",
    "ct_lum_mean", "ct_lum_std", "ct_edge_mean", "ct_edge_hi_frac",
    "ct_same_line",
]
# reported for completeness; rank/score position is matched or confounded by
# construction in every comparison, so these never qualify as structure
RANK_FEATURES = ["rk_teacher_rank_mean", "rk_teacher_rank_span",
                 "rk_tonly_pos_mean", "rk_tonly_contig",
                 "sc_g_mean", "sc_g_min", "sc_g_span", "sc_s_mean",
                 "sc_s_vs_S", "ft_g2pct", "ft_linpct"]
FEATS = CANDIDATE_FEATURES + RANK_FEATURES


# ---------------------------------------------------------------------------
def cp_ci(k, n, alpha=0.05):
    if n == 0:
        return [float("nan"), float("nan")]
    ci = binomtest(k, n).proportion_ci(1 - alpha)
    return [float(ci.low), float(ci.high)]


def boot_ci(vals, n_boot=10000, seed=0):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(1)
    return (float(v.mean()), float(np.percentile(b, 2.5)),
            float(np.percentile(b, 97.5)))


def auc_of(scores, labels):
    """Mann-Whitney AUC (ties counted as 0.5) via average ranks."""
    from scipy.stats import rankdata
    sc = np.asarray(scores, dtype=float)
    l = np.asarray(labels).astype(int)
    n1, n0 = int(l.sum()), int((1 - l).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(sc)
    return float((r[l == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def best_threshold(v, y):
    """the single threshold that maximises Youden's J, over both comparison
    directions -- "a single-threshold classifier" in the frozen clause has to
    be free to choose its direction, or every lower-is-better property
    (bbox area per token, connected components, ...) reads as a null result.
    AUC of a hard prediction = 0.5 + J/2, so this is the max-AUC cut."""
    v = np.asarray(v, dtype=float)
    ys = np.asarray(y).astype(int)
    pos, neg = ys == 1, ys == 0
    if not pos.any() or not neg.any():
        return (None, 1.0, -1.0)
    best = (None, 1.0, -1.0)
    for sign in (1.0, -1.0):
        vs = sign * v
        for t in np.unique(vs):
            pred = vs >= t
            jj = float(pred[pos].mean() - pred[neg].mean())
            if jj > best[2]:
                best = (float(t), sign, jj)
    return best


def rank_biserial(diff):
    d = np.asarray([x for x in diff if np.isfinite(x) and abs(x) > 0], dtype=float)
    if d.size == 0:
        return float("nan"), float("nan")
    a = np.abs(d)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(d.size, dtype=float)
    ranks[order] = np.arange(1, d.size + 1)
    uniq, inv = np.unique(a, return_inverse=True)
    ranks = np.array([ranks[inv == i].mean() for i in range(uniq.size)])[inv]
    rb = float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())
    try:
        p = float(wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:
        p = float("nan")
    return rb, p


def loo_cv_auc(v, y, inst):
    """leave-one-INSTANCE-out cross-validated AUC: at a fixed k every instance
    contributes exactly one row, so the fold score is pooled over held-out
    instances (one out-of-fold prediction each), not computed inside a fold."""
    pred = np.zeros(len(v), dtype=int)
    ok = np.ones(len(v), dtype=bool)
    for u in np.unique(inst):
        te = inst == u
        tr = ~te
        if len(set(y[tr])) < 2:
            ok[te] = False
            continue
        t_, sign, _ = best_threshold(v[tr], y[tr])
        pred[te] = (sign * v[te] >= t_).astype(int)
    if ok.sum() < 6 or len(set(y[ok])) < 2:
        return float("nan"), int(ok.sum())
    a = auc_of(pred[ok].astype(float), y[ok])
    # balanced accuracy of the same out-of-fold predictions
    pos, neg = y[ok] == 1, y[ok] == 0
    bal = float(np.mean([(pred[ok][pos] == 1).mean(), (pred[ok][neg] == 0).mean()]))
    return a, bal


def oriented(auc_signed, direction):
    """AUC in the direction the pooled data picked (>= 0.5 by construction)."""
    if not np.isfinite(auc_signed):
        return auc_signed
    return auc_signed if direction >= 0 else 1.0 - auc_signed


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-crop", action="store_true")
    args = ap.parse_args()

    def rd(name, required=True):
        p = os.path.join(OUT, name)
        if not os.path.exists(p):
            if required:
                raise SystemExit(f"missing {name}")
            return None
        return json.load(open(p))

    cases = rd("s3b_cases.json")
    base = rd("s3b_base.json")
    sweep = rd("s3b_sweep.json")
    ablate = rd("s3b_ablate.json")
    nulls = rd("s3b_nulls.json", required=False)
    subst = rd("s3b_subst.json", required=False)
    crop = None if args.no_crop else rd("s3b_crop.json", required=False)
    # the JSON bank records carry the sets; the feature machinery also needs
    # the score vectors, which are recomputed from the frozen caches for the
    # instances actually involved.
    need = {b: set() for b in ("G", "L")}
    for ent in ablate["runs"].values():
        need[ent["bank"]].add(ent["key"])
    if subst:
        for ent in subst["runs"].values():
            need[ent["bank"]].add(ent["key"])
    if nulls:
        need["G"].update(nulls["runs"])
    need["G"].update(sweep["G"])
    need["L"].update(k for k, v in cases["bundles_L"].items() if v["curve"])
    recs = {b: {r["key"]: r for r in B.load_bank(b, keys=sorted(ks))}
            for b, ks in need.items() if ks}
    for b in ("G", "L"):
        recs.setdefault(b, {})
    for r in cases["banks"]["G"]:
        recs["G"].setdefault(r["key"], r)
    for r in cases["banks"]["L"]:
        recs["L"].setdefault(r["key"], r)
    if B.load_cellstats() is None:
        B.build_cellstats()
    cells = B.load_cellstats()

    out = dict(config=dict(cases=cases["config"], thr=B.THR, ks_max=B.KS_MAX,
                           candidate_features=CANDIDATE_FEATURES))

    # ================================================================= PART A
    out["partA"] = {"G": partA_G(base, sweep), "L": partA_L(cases)}

    # ================================================================= PART B
    rows = []                                  # one dict per (instance, set)
    for bkey, ent in sorted(ablate["runs"].items()):
        bank, key = ent["bank"], ent["key"]
        rec = recs[bank][key]
        for name, a in ent["arms"].items():
            f = B.set_features(rec, a["adds"], cells=cells.get(key))
            rows.append(dict(bank=bank, key=key, ds=key.rsplit("_", 1)[0],
                             arm=name, kind=a["kind"], k=len(a["adds"]),
                             sig=tuple(sorted(int(t) for t in a["adds"])),
                             hit=a["hit"], L=a["L"], kstar=ent["kstar"],
                             floor=floor_of(sweep, base, bank, key), **f))
    # a k*=1 bundle's `single0` arm is literally the bundle itself: counting it
    # as "one member alone works" would double-report the same measurement and
    # inflate the single-token rate (6 of 27 bank-G bundles are monadic).
    for bkey, ent in ablate["runs"].items():
        bsig = tuple(sorted(ent["arms"]["full"]["adds"]))
        for name, a in ent["arms"].items():
            if a["kind"] == "single" and \
                    tuple(sorted(int(t) for t in a["adds"])) == bsig:
                for r in rows:
                    if r["bank"] == ent["bank"] and r["key"] == ent["key"] \
                            and r["arm"] == name:
                        r["kind"] = "single_trivial"
    # the substitution arms (constant depth, one member traded; amendment
    # stage -- see the module docstring of s3b_subst.py)
    if subst:
        for bkey, ent in sorted(subst["runs"].items()):
            rec = recs[ent["bank"]][ent["key"]]
            for name, a in ent["arms"].items():
                f = B.set_features(rec, a["adds"], cells=cells.get(ent["key"]))
                rows.append(dict(bank=ent["bank"], key=ent["key"],
                                 ds=ent["ds"], arm=name, kind="substitute",
                                 k=len(a["adds"]),
                                 sig=tuple(sorted(int(t) for t in a["adds"])),
                                 hit=a["hit"], L=a["L"],
                                 kstar=ent["kstar"],
                                 floor=floor_of(sweep, base, ent["bank"],
                                                ent["key"]), **f))
    # the non-rescued comparison class (same null families, k = 8 and 16)
    if nulls:
        for key, ent in sorted(nulls["runs"].items()):
            rec = recs["G"][key]
            for k, arms in ent["ks"].items():
                for name, a in arms.items():
                    f = B.set_features(rec, a["adds"], cells=cells.get(key))
                    rows.append(dict(bank="G", key=key, ds=key.rsplit("_", 1)[0],
                                     arm=name, kind=a["kind"], k=int(k),
                                     sig=tuple(sorted(int(t) for t in a["adds"])),
                                     hit=a["hit"], L=a["L"], kstar=None,
                                     floor=ent["floor"], **f))
    for r in rows:
        r["has_bundle"] = bool(r["kstar"])
    out["rows_n"] = len(rows)
    # pooled arm rates count each distinct added set once (see dedup())
    rows_d, n_dup = dedup(rows)
    out["dedup"] = dict(rows=len(rows), rows_distinct=len(rows_d),
                        duplicates_removed=n_dup)
    out["window_degeneracy"] = window_degeneracy(ablate)

    det = []
    for bkey, ent in ablate["runs"].items():
        src = sweep["G"].get(ent["key"]) if ent["bank"] == "G" \
            else sweep["L"].get(ent["key"])
        ref = ((src or {}).get("curve", {}) or {}).get(str(ent["kstar"]))
        if ref and "full" in ent["arms"]:
            det.append(abs(ent["arms"]["full"]["L"] - ref["L"]))
    baseL = {(b, k): e["banks"][b]["L"] for k, e in base["recs"].items()
             for b in ("G", "L")}
    out["partB"] = partB(rows_d, det, baseL)

    # ============================================================ the tests
    out["T1_null_vs_null"] = T1(rows_d)
    out["T2_bundle_vs_nulls"] = T2(rows_d)
    prefix_rows = add_prefix_rows(rows_d, cases, sweep, recs, cells)
    out["T3_prefix_hit_vs_miss"] = T3(prefix_rows)
    out["T3_partial_on_strength"] = T3_partial(prefix_rows)
    out["T4_depth_from_shape"] = T4_depth_from_shape(sweep, recs["G"], cells)
    out["base_rates"] = base_rates(rows_d)
    out["rescuability_covariates"] = rescuability(sweep, base, cases)
    bands = rd("s3b_bands.json", required=False)
    if bands:
        out["band_ladder"] = band_table(bands)
    if crop:
        out["crop"] = crop_table(crop)
    out["cross_bank_bundle_jaccard"] = cross_bank(cases, sweep, recs)
    out["verdict"] = decide(out)

    path = os.path.join(OUT, "s3b_analysis.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    report(out)
    print(f"[saved] {path}")


def floor_of(sweep, base, bank, key):
    e = (sweep[bank] or {}).get(key) or {}
    if e.get("floor") is not None:
        return float(e["floor"])
    return float(base["recs"].get(key, {}).get("floor", float("nan")))


# ---------------------------------------------------------------------------
def partA_G(base, sweep):
    wrong = [k for k, e in base["recs"].items() if e["banks"]["G"]["hit"] < B.THR]
    swept = sweep["G"]
    ks = [v["kstar"] for v in swept.values()]
    curve = {}
    for k in B.KS_SWEEP:
        rr = [v for v in swept.values() if str(k) in v["curve"]]
        curve[str(k)] = dict(
            n=len(rr),
            hit_rate=float(np.mean([v["curve"][str(k)]["hit"] >= B.THR
                                    for v in rr])) if rr else None,
            mean_dL=float(np.mean([v["L0"] - v["curve"][str(k)]["L"]
                                   for v in rr])) if rr else None)
    curve_ds = {}
    for ds in B.DS_ALL:
        rr = [v for v in sweep["G"].values() if v["ds"] == ds]
        curve_ds[ds] = {str(k): (float(np.mean(
            [v["curve"][str(k)]["hit"] >= B.THR for v in rr
             if str(k) in v["curve"]]))
            if any(str(k) in v["curve"] for v in rr) else None)
            for k in B.KS_SWEEP}
    return dict(n_base_wrong=len(wrong), n_swept=len(swept),
                curve_by_ds=curve_ds,
                n_rescued=sum(1 for x in ks if x),
                rescue_rate=sum(1 for x in ks if x) / max(1, len(wrong)),
                kstars=sorted(x for x in ks if x),
                median_kstar=float(np.median([x for x in ks if x]))
                if any(ks) else None,
                per_ds={ds: dict(
                    n=sum(1 for v in swept.values() if v["ds"] == ds),
                    rescued=sum(1 for v in swept.values()
                                if v["ds"] == ds and v["kstar"]),
                    kstars=sorted(v["kstar"] for v in swept.values()
                                  if v["ds"] == ds and v["kstar"]))
                    for ds in B.DS_ALL},
                curve=curve,
                n_ext_measured=sum(1 for v in swept.values() if v.get("ext")),
                ext_kstars=sorted(v["kstar_ext"] for v in swept.values()
                                  if v.get("kstar_ext")))


def partA_L(cases):
    bl = cases["bundles_L"]
    got = {k: v for k, v in bl.items() if v["kstar"]}
    ks = sorted(v["kstar"] for v in got.values())
    curve = {}
    for k in B.KS_SWEEP:
        hh = [v["curve"][str(k)] for v in bl.values() if str(k) in v["curve"]]
        curve[str(k)] = dict(n=len(hh),
                            hit_rate=float(np.mean([h >= B.THR for h in hh]))
                            if hh else None)
    curve_ds = {}
    for ds in B.DS_ALL:
        rr = [v for v in bl.values() if v["ds"] == ds]
        curve_ds[ds] = {str(k): (float(np.mean(
            [v["curve"][str(k)] >= B.THR for v in rr if str(k) in v["curve"]]))
            if any(str(k) in v["curve"] for v in rr) else None)
            for k in B.KS_SWEEP}
    return dict(n_class=len(bl), n_rescued=len(ks), kstars=ks,
                curve_by_ds=curve_ds,
                median_kstar=float(np.median(ks)) if ks else None,
                rescue_rate=len(ks) / max(1, len(bl)),
                per_ds={ds: dict(
                    n=sum(1 for v in bl.values() if v["ds"] == ds),
                    rescued=sum(1 for v in got.values() if v["ds"] == ds),
                    kstars=sorted(v["kstar"] for v in got.values()
                                  if v["ds"] == ds)) for ds in B.DS_ALL},
                curve=curve)


# ---------------------------------------------------------------------------
def partB(rows, det, baseL):
    per = defaultdict(dict)
    for r in rows:
        if r["kstar"]:
            per[(r["bank"], r["key"])][r["arm"]] = r
    ent = []
    for (bank, key), arms in sorted(per.items()):
        if "full" not in arms:
            continue
        ks = arms["full"]["kstar"]
        sing = [r["hit"] >= B.THR for n, r in arms.items()
                if r["kind"] in ("single", "single_trivial")]
        sing_hard = [r["hit"] >= B.THR for n, r in arms.items()
                     if r["kind"] == "single"]
        loo = [r["hit"] >= B.THR for n, r in arms.items() if r["kind"] == "loo"]
        nul = [r["hit"] >= B.THR for n, r in arms.items()
               if r["kind"] in NULL_KINDS]
        win = [r["hit"] >= B.THR for n, r in arms.items()
               if r["kind"] == "null_window"]
        sub = [r["hit"] >= B.THR for n, r in arms.items()
               if r["kind"] == "substitute"]
        bL = baseL.get((bank, key))
        dL = (lambda r: float("nan") if bL is None or not np.isfinite(r["L"])
              else bL - r["L"])
        singles_dL = [dL(r) for n, r in arms.items() if r["kind"] == "single"]
        ent.append(dict(
            bank=bank, key=key, ds=key.rsplit("_", 1)[0], kstar=ks,
            n_single=len(sing), n_suff=int(sum(sing)),
            n_single_nontrivial=len(sing_hard),
            n_suff_nontrivial=int(sum(sing_hard)),
            n_loo=len(loo), n_nec=int(len(loo) - sum(loo)),
            group_only=bool(sing) and sum(sing) == 0 and
            (len(loo) - sum(loo)) == ks,
            monadic=bool(ks == 1),
            n_null=len(nul), n_null_hit=int(sum(nul)),
            n_sub=len(sub), n_sub_hit=int(sum(sub)),
            n_win=len(win), n_win_hit=int(sum(win)),
            floor=arms["full"]["floor"],
            dL_full=dL(arms["full"]),
            dL_single_max=float(max(singles_dL)) if singles_dL else None,
            dL_loo_min=float(min(dL(r) for n, r in arms.items()
                                 if r["kind"] == "loo"))
            if any(r["kind"] == "loo" for r in arms.values()) else None,
            dL_win_max=float(max(dL(r) for n, r in arms.items()
                                 if r["kind"] == "null_window"))
            if any(r["kind"] == "null_window" for r in arms.values()) else None))
    fam = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["kind"] in NULL_KINDS + ("shift", "blockperm", "removal_control",
                                      "single", "single_trivial", "loo",
                                      "substitute"):
            fam[(r["bank"], r["kind"])][0] += int(r["hit"] >= B.THR)
            fam[(r["bank"], r["kind"])][1] += 1
    fam_ds = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["kind"] in NULL_KINDS + ("bundle",):
            fam_ds[(r["kind"], r["ds"])][0] += int(r["hit"] >= B.THR)
            fam_ds[(r["kind"], r["ds"])][1] += 1
    # `single` rows exclude the k*=1 self-comparisons (re-labelled
    # single_trivial above), so this is the honest "one member alone" rate
    nt_ = sum(v[1] for kk, v in fam.items() if kk[1] == "single")
    nt_h = sum(v[0] for kk, v in fam.items() if kk[1] == "single")
    return dict(
        n_bundles=len(ent), bundles=ent,
        single_rate_nontrivial=float(nt_h / nt_) if nt_ else None,
        family_hit_rates={f"{b}|{kind}": dict(
            hits=v[0], n=v[1], rate=v[0] / v[1], ci=cp_ci(v[0], v[1]))
            for (b, kind), v in sorted(fam.items())},
        family_hit_rates_by_ds={f"{kind}|{ds}": dict(
            hits=v[0], n=v[1], rate=v[0] / v[1])
            for (kind, ds), v in sorted(fam_ds.items())},
        group_only_rate=float(np.mean([e["group_only"] for e in ent])),
        monadic_rate=float(np.mean([e["monadic"] for e in ent])),
        mean_frac_sufficient=float(np.mean([e["n_suff"] / max(1, e["n_single"])
                                            for e in ent])),
        mean_frac_necessary=float(np.mean([e["n_nec"] / max(1, e["kstar"])
                                           for e in ent])),
        window_null_hit_rate=float(np.mean([e["n_win_hit"] / max(1, e["n_win"])
                                            for e in ent])),
        subst_hit_rate=float(np.mean([e["n_sub_hit"] / max(1, e["n_sub"])
                                      for e in ent if e["n_sub"]])),
        n_bundles_with_subst=sum(1 for e in ent if e["n_sub"]),
        any_null_hit_rate=float(np.mean([e["n_null_hit"] / max(1, e["n_null"])
                                         for e in ent])),
        determinism_max_abs_dL=float(np.max(det)) if det else None,
        determinism_n=len(det))


# ---------------------------------------------------------------------------
def T1(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["kind"] in NULL_KINDS:
            groups[(r["bank"], r["key"], r["k"])].append(r)
    res = {}
    for f in FEATS:
        aucs, diffs, per_ds = [], [], defaultdict(list)
        for kk, rr in groups.items():
            rr = [r for r in rr if np.isfinite(r[f])]
            lab = [int(r["hit"] >= B.THR) for r in rr]
            if not (0 < sum(lab) < len(lab)):
                continue
            a = auc_of([r[f] for r in rr], lab)
            aucs.append(a)
            pos = [r[f] for r in rr if r["hit"] >= B.THR]
            neg = [r[f] for r in rr if r["hit"] < B.THR]
            diffs.append(float(np.mean(pos) - np.mean(neg)))
            ds = rr[0]["ds"]
            bank = rr[0]["bank"]
            per_ds[f"{ds}|{bank}"].append(a)
        m, lo, hi = boot_ci(aucs)
        rb, p = rank_biserial(diffs)
        # direction is chosen ONCE from the pooled sign of the within-group
        # differences, never per group (a per-group best-of-both would be
        # upward-biased by construction); the frozen clause speaks of "a
        # single-threshold classifier", which may pick either direction.
        d = 1 if (np.isfinite(rb) and rb >= 0) else (-1 if np.isfinite(rb) else 0)
        ao = [oriented(a, d) for a in aucs]
        mo, loo_, hii = boot_ci(ao)
        agree = (float(np.mean([a > 0.5 for a in aucs])) if aucs else None)
        if d < 0 and agree is not None:
            agree = 1.0 - agree
        by_ds = {pan: (float(np.nanmean([oriented(x, d) for x in vv]))
                       if vv else None) for pan, vv in per_ds.items()}
        n_ds_support = sum(1 for pan, vv in by_ds.items()
                           if vv is not None and abs(vv - 0.5) >= 0.05)
        res[f] = dict(n_groups=len(aucs), mean_auc=m, auc_ci=[lo, hi],
                      auc_by_ds=by_ds, n_ds_beyond_chance=n_ds_support,
                      direction=("higher" if d > 0 else
                                 "lower" if d < 0 else "none"),
                      mean_auc_oriented=mo, auc_oriented_ci=[loo_, hii],
                      frac_groups_on_pooled_side=agree,
                      rank_biserial=rb, wilcoxon_p=p)
    cand = {k: v for k, v in res.items() if k in CANDIDATE_FEATURES}
    top = sorted(cand.items(),
                 key=lambda x: -abs(x[1]["mean_auc"] - 0.5)
                 if np.isfinite(x[1]["mean_auc"]) else 0)
    return dict(rows=top + [(k, res[k]) for k in RANK_FEATURES],
                pooled=dict(n_null_rows=sum(len(v) for v in groups.values()),
                            n_null_hits=sum(1 for v in groups.values()
                                            for r in v
                                            if r["hit"] >= B.THR)))


def T2(rows):
    bundles = {(r["bank"], r["key"]): r for r in rows if r["kind"] == "bundle"}
    res = {}
    for f in FEATS:
        diffs = []
        for bk, br in bundles.items():
            nn = [r[f] for r in rows if (r["bank"], r["key"]) == bk
                  and r["kind"] == "null_window" and np.isfinite(r[f])]
            if nn and np.isfinite(br[f]):
                diffs.append(float(br[f] - np.mean(nn)))
        rb, p = rank_biserial(diffs)
        m, lo, hi = boot_ci(diffs)
        res[f] = dict(n=len(diffs), mean_diff=m, ci=[lo, hi],
                      rank_biserial=rb, wilcoxon_p=p)
    return res


def add_prefix_rows(rows, cases, sweep, recs, cells):
    """the teacher prefix at every swept k, for the cross-instance test."""
    have = {(r["bank"], r["key"], r["k"], r["kind"]) for r in rows}
    extra = []
    for key, e in sorted(sweep["G"].items()):
        rec = recs["G"][key]
        for k, v in e["curve"].items():
            f = B.set_features(rec, v["added"], cells=cells.get(key))
            extra.append(dict(bank="G", key=key, ds=e["ds"], arm=f"prefix{k}",
                              kind="prefix_hit" if v["hit"] >= B.THR
                              else "prefix_miss",
                              k=int(k), hit=v["hit"], L=v["L"],
                              kstar=e["kstar"], floor=e["floor"], **f))
    for key, b in sorted(cases["bundles_L"].items()):
        rec = recs["L"][key]
        for k, h in b["curve"].items():
            if int(k) > B.KS_MAX:
                continue
            adds, _ = B.prefix_arms(rec, int(k))
            f = B.set_features(rec, adds, cells=cells.get(key))
            extra.append(dict(bank="L", key=key, ds=b["ds"], arm=f"prefix{k}",
                              kind="prefix_hit" if h >= B.THR else "prefix_miss",
                              k=int(k), hit=h, L=float("nan"),
                              kstar=b["kstar"],
                              floor=float("nan"), **f))
    return rows + extra


def T3(rows):
    res = {}
    for k in (2, 4, 8, 16, 32):
        rr = [r for r in rows if r["k"] == k
              and r["kind"] in ("prefix_hit", "prefix_miss")]
        lab = np.array([int(r["hit"] >= B.THR) for r in rr])
        if not (0 < lab.sum() < len(rr)):
            continue
        ent = {}
        for f in FEATS:
            v = np.array([r[f] for r in rr], dtype=float)
            ok = np.isfinite(v)
            if ok.sum() < 12 or len(set(lab[ok])) < 2:
                continue
            inst = np.array([f'{r["bank"]}|{r["key"]}' for r, o in
                             zip(rr, ok) if o])
            cv_auc, cv_bal = loo_cv_auc(v[ok], lab[ok], inst)
            # the frozen clause-A second half is a separation test at p < 0.05
            mw = mannwhitneyu(v[ok][lab[ok] == 1], v[ok][lab[ok] == 0],
                              alternative="two-sided")
            ent[f] = dict(pooled_auc=auc_of(v[ok], lab[ok]),
                          loo_auc=cv_auc, loo_balanced=cv_bal,
                          mw_p=float(mw.pvalue), n=int(ok.sum()),
                          n_instances=len(set(inst)))
        cand = {kk: vv for kk, vv in ent.items() if kk in CANDIDATE_FEATURES}
        top = sorted(cand.items(),
                     key=lambda x: -abs(x[1]["pooled_auc"] - 0.5)
                     if np.isfinite(x[1]["pooled_auc"]) else 0)
        res[str(k)] = dict(n_rows=len(rr), n_hit=int(lab.sum()), features=ent,
                           top=top[:8])
    return res


def dedup(rows):
    """the zero-overlap window null is FORCED to be unique once k* >= 16: with
    W = max(2k, k+16) the pool T_only[:W] \\ G holds exactly k tokens at k >= 16,
    so all N_RANDWIN draws are the same adjacent block. Counting it six times
    would let one accidental set stand in for six. Pooled rates therefore use
    each distinct (instance, kind, set) once; per-instance rates are unaffected
    (they average the distinct sets)."""
    seen, out, dup = set(), [], 0
    for r in rows:
        if r.get("sig") is None:          # prefix rows: one per k, never duped
            out.append(r)
            continue
        kk = (r["bank"], r["key"], r["kind"], r["sig"])
        if kk in seen:
            dup += 1
            continue
        seen.add(kk)
        out.append(r)
    return out, dup


def base_rates(rows):
    """the honest denominators: P(hit) for each matched-set family, split by
    whether the instance is rescuable by the prefix at all (bundle within k* <=
    32) or not. The non-rescued column is what a "rescue by chance" baseline
    looks like with no bundle anywhere."""
    tab = {}
    for kind in NULL_KINDS + ("shift", "blockperm", "substitute", "single",
                              "loo", "bundle", "prefix_hit", "prefix_miss",
                              "prefix_unsuccess"):
        for grp, name in ((True, "rescuable"), (False, "non_rescued")):
            rr = [r for r in rows if r["kind"] == kind
                  and r["has_bundle"] == grp]
            if not rr:
                continue
            k = sum(1 for r in rr if r["hit"] >= B.THR)
            tab[f"{kind}|{name}"] = dict(
                hits=k, n=len(rr), rate=k / len(rr), ci=cp_ci(k, len(rr)),
                n_instances=len({r["key"] for r in rr}))
    return tab


DEEP_BANDS = ["pos_16_32", "pos_32_64", "pos_64_128", "outside"]


def band_table(bands):
    """P(hit) for a size-k* set drawn from each teacher-rank band, all under the
    F(k*) removal rule. Two views: pooled over every draw, and restricted to the
    instances where ALL deep bands were measurable (common support) -- the
    shallow bands are only available for small-k* bundles, so pooling across
    bands silently changes the instance panel."""
    sup = [k for k, e in bands["runs"].items()
           if all(bb in [a["band"] for a in e["arms"].values()]
                  for bb in DEEP_BANDS)]
    cs = defaultdict(lambda: [0, 0])
    for k in sup:
        for a in bands["runs"][k]["arms"].values():
            cs[a["band"]][0] += int(a["hit"] >= B.THR)
            cs[a["band"]][1] += 1
    common = {bb: dict(hits=cs[bb][0], n=cs[bb][1],
                       rate=cs[bb][0] / max(1, cs[bb][1]),
                       ci=cp_ci(cs[bb][0], cs[bb][1]))
              for bb in DEEP_BANDS if cs[bb][1]}
    tab = defaultdict(lambda: [0, 0, []])
    for e in bands["runs"].values():
        for a in e["arms"].values():
            t = tab[a["band"]]
            t[0] += int(a["hit"] >= B.THR)
            t[1] += 1
            t[2].append(a["rank_mean"])
    pooled = {}
    for name, (k, n, rk) in sorted(tab.items(),
                                   key=lambda x: (len(x[0]), x[0])):
        pooled[name] = dict(hits=k, n=n, rate=k / n if n else None,
                            mean_rank=(float(np.nanmean(rk))
                                       if any(np.isfinite(rk)) else None),
                            ci=cp_ci(k, n))
    return dict(pooled=pooled,
                common_support=dict(n_instances=len(sup), bands=common))


def window_degeneracy(ablate):
    """how many bundles have a forced-unique window null."""
    n_deg = n_free = 0
    for ent in ablate["runs"].values():
        k = ent["kstar"]
        rec_m = ent["m"]
        W = min(rec_m, max(2 * k, k + 16))
        if W - k <= k:
            n_deg += 1
        else:
            n_free += 1
    return dict(n_forced_unique=n_deg, n_free=n_free,
                note="pool T_only[:W] minus the bundle holds W-k tokens; "
                     "the null is unique when W-k <= k (k* >= 16 here)")


def rescuability(sweep, base, cases):
    """what distinguishes an instance the teacher queue CAN flip from one it
    cannot? Descriptive only -- it is the instance-level half of the same
    question, and S3-A showed the answer lives in set composition."""
    import ast
    tab = defaultdict(lambda: defaultdict(list))
    for key, e in sweep["G"].items():
        grp = "rescuable_le32" if e["kstar"] else (
            "rescuable_33_64" if e.get("kstar_ext") else "not_rescuable")
        b = base["recs"][key]
        tab[grp]["L_base"].append(e["L0"])
        tab[grp]["floor"].append(e["floor"])
        tab[grp]["n_golds"].append(b["n_golds"])
        tab[grp]["m"].append(e["m"])
        tab[grp]["ds"].append(e["ds"])
    out = {}
    for grp, d in tab.items():
        out[grp] = dict(
            n=len(d["L_base"]),
            L_base=float(np.mean(d["L_base"])),
            floor=float(np.mean(d["floor"])),
            n_golds=float(np.mean(d["n_golds"])),
            m_T_only=float(np.mean(d["m"])),
            by_ds={ds: int(np.sum([x == ds for x in d["ds"]]))
                   for ds in B.DS_ALL})
    return out


PROXIES = ("sc_g_mean", "sc_mass_frac")
# "how strong / how concentrated is this prefix, period" — the competing
# explanation for any spatial signal: a teacher map that is hot in one place
# gives a compact top-k AND an easy answer, with no geometry involved.


def auc_stratified(scores, labels, strata):
    """Mann-Whitney AUC restricted to pairs inside the same stratum (here: same
    benchmark AND same base), so a spatial effect that is really 'DocVQA has
    line structure and rescues easier', or 'bank L has more hits', cannot pass
    for one."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    num = den = 0.0
    for u in np.unique(strata):
        m = strata == u
        p, q = scores[m & (labels == 1)], scores[m & (labels == 0)]
        if p.size == 0 or q.size == 0:
            continue
        num += sum((x > y) + 0.5 * (x == y) for x in p for y in q)
        den += p.size * q.size
    return float(num / den) if den else float("nan")


def T3_partial(rows):
    """is the shape signal just strength? At a fixed k every candidate set is
    the teacher's own top-k prefix, so the only thing that varies between a
    rescuing and a non-rescuing instance is *where those ranks sit in the image*
    and *how much saliency they carry*. If a spatial property stops mattering
    once the set's mean teacher score is controlled, the finding is evidence
    CONCENTRATION (one number), not evidence GEOMETRY (a selection rule).

    Method: within each k, rank-residualise the spatial property on the proxy
    (both ranks, within k), then recompute the pooled and leave-one-instance-out
    AUC on the residual. Reported next to the unadjusted numbers."""
    out = {}
    for k in (4, 8, 16, 32):
        rr = [r for r in rows if r["k"] == k
              and r["kind"] in ("prefix_hit", "prefix_miss")]
        lab = np.array([int(r["hit"] >= B.THR) for r in rr])
        if not (0 < lab.sum() < len(rr)):
            out[str(k)] = None
            continue
        inst = np.array([f'{r["bank"]}|{r["key"]}' for r in rr])
        # stratify by benchmark AND base: pooling banks or datasets would let
        # "DocVQA rescues easier" or "bank L has more hits" pass as geometry.
        strata = np.array([f'{r["ds"]}|{r["bank"]}' for r in rr])
        ent = {}
        for f in CANDIDATE_FEATURES:
            v = np.array([r[f] for r in rr], dtype=float)
            ok = np.isfinite(v)
            if ok.sum() < 12 or len(set(lab[ok])) < 2:
                continue
            e = dict(raw_auc=auc_of(v[ok], lab[ok]),
                     raw_auc_within_ds=auc_stratified(v[ok], lab[ok],
                                                      strata[ok]),
                     raw_loo=loo_cv_auc(v[ok], lab[ok], inst[ok])[0])
            for px in PROXIES:
                pr = np.array([r[px] for r in rr], dtype=float)
                both = ok & np.isfinite(pr)
                if both.sum() < 12 or len(set(lab[both])) < 2:
                    continue
                resv = _rank_resid(v[both], pr[both])
                e[f"resid_on_{px}"] = auc_of(resv, lab[both])
                e[f"resid_on_{px}_within_ds"] = auc_stratified(
                    resv, lab[both], strata[both])
                e[f"resid_on_{px}_loo"] = loo_cv_auc(resv, lab[both],
                                                     inst[both])[0]
                e[f"proxy_{px}"] = auc_of(pr[both], lab[both])
            ent[f] = e
        out[str(k)] = ent
    return out


def _rank_resid(y, x):
    """y with the monotone part of x removed (ranks against ranks)."""
    from scipy.stats import rankdata
    ry = rankdata(y)
    rx = rankdata(x)
    if np.allclose(rx, rx[0]):
        return ry
    b = np.polyfit(rx, ry, 1)
    return ry - (b[0] * rx + b[1])


def T4_depth_from_shape(sweep, recsG, cells):
    """the predictive form of the same question, which uses every swept
    instance instead of only the ones the sweep happened to reach: how DEEP the
    teacher has to go (k*) is itself the bundle-size variable, and a structure
    claim worth anything should predict it from a fixed-size probe set.

    For each bank-G base-wrong instance the reference set is the teacher's own
    first r missed tokens (r = 4, 8, 16) — a set that exists for every instance,
    measured before any intervention — and it is correlated with the depth k*
    that the sweep needed (instances with no rescue take k* = 65, their rank is
    still meaningful; a second column reports the same correlation among the 34
    instances that were rescued at all, where no censoring is involved).

    Spearman rho, pooled and within (benchmark), plus the partial correlation
    after rank-residualising on the set's share of the teacher mass."""
    from scipy.stats import spearmanr
    out = {}
    keys = sorted(sweep["G"])
    for r in (4, 8, 16):
        rows = []
        for key in keys:
            rec = recsG[key]
            if r > rec["m"]:
                continue
            adds, _ = B.prefix_arms(rec, r)
            f = B.set_features(rec, adds, cells=cells.get(key))
            e = sweep["G"][key]
            kst = e["kstar"] or e.get("kstar_ext") or 65
            rows.append(dict(key=key, ds=rec["ds"], kstar=kst,
                             rescued=bool(e["kstar"] or e.get("kstar_ext")),
                             **f))
        ent = {}
        for f in CANDIDATE_FEATURES:
            v = np.array([x[f] for x in rows], dtype=float)
            y = np.array([x["kstar"] for x in rows], dtype=float)
            ok = np.isfinite(v)
            if ok.sum() < 20:
                continue
            res = dict(n=int(ok.sum()),
                       rho=spearmanr(v[ok], y[ok]).statistic,
                       p=spearmanr(v[ok], y[ok]).pvalue)
            # uncensored subset
            sub = np.array([x["rescued"] for x in rows]) & ok
            if sub.sum() >= 15:
                res["rho_rescued_only"] = spearmanr(v[sub], y[sub]).statistic
                res["p_rescued_only"] = spearmanr(v[sub], y[sub]).pvalue
            # within benchmark
            byds = {}
            for ds in B.DS_ALL:
                m_ = np.array([x["ds"] == ds for x in rows]) & ok
                if m_.sum() >= 10:
                    byds[ds] = float(spearmanr(v[m_], y[m_]).statistic)
            res["rho_by_ds"] = byds
            # partial on teacher-mass concentration
            pr = np.array([x["sc_mass_frac"] for x in rows], dtype=float)
            if np.isfinite(pr).all():
                rv = _rank_resid(v[ok], pr[ok])
                res["rho_partial_mass"] = float(spearmanr(rv, y[ok]).statistic)
                rp = _rank_resid(pr[ok], v[ok])
                res["rho_mass_only_partial"] = float(spearmanr(rp, y[ok]).statistic)
            ent[f] = {kk: (None if not np.isfinite(vv) else float(vv))
                      for kk, vv in res.items() if not isinstance(vv, dict)}
            ent[f]["rho_by_ds"] = byds
            top = sorted(ent.items(),
                         key=lambda x: -abs(x[1].get("rho") or 0))[:6]
        out[str(r)] = dict(n_instances=len(rows),
                           top=sorted(
                               ((f, e) for f, e in ent.items()),
                               key=lambda x: -abs(x[1].get("rho") or 0))[:8],
                           all=ent)
    return out


def crop_table(crop):
    per = defaultdict(dict)
    for bkey, e in crop["runs"].items():
        for name, a in e["arms"].items():
            kind = ("bundle" if name == "bundle" else
                    "null" if name.startswith("randwin") else
                    "matched" if name.startswith("matched") else "whole")
            per[bkey].setdefault(kind, False)
            per[bkey][kind] = per[bkey][kind] or (a["hit"] >= B.THR)
    tab = {}
    for kind in ("whole", "bundle", "null", "matched"):
        col = [v.get(kind, False) for v in per.values()]
        n = len([v for v in per.values() if kind in v])
        k = sum(col)
        tab[kind] = dict(n=n, rate=k / n if n else None, ci=cp_ci(k, n))
    tab["bundle_only_vs_null_only"] = dict(
        bundle_not_null=sum(1 for v in per.values()
                            if v.get("bundle") and not v.get("null")),
        null_not_bundle=sum(1 for v in per.values()
                            if v.get("null") and not v.get("bundle")))
    return tab


def cross_bank(cases, sweep, recs):
    vals = []
    for key in sorted(set(sweep["G"]) & set(cases["bundles_L"])):
        g, l = sweep["G"][key], cases["bundles_L"][key]
        if not g.get("kstar") or not l.get("kstar"):
            continue
        a = set(recs["G"][key]["T_only"][:g["kstar"]])
        b = set(recs["L"][key]["T_only"][:l["kstar"]])
        vals.append(len(a & b) / max(1, len(a | b)))
    return dict(n=len(vals), mean=float(np.mean(vals)) if vals else None,
                values=sorted(round(v, 3) for v in vals))


# ---------------------------------------------------------------------------
def decide(out):
    v = {}
    A = out["partA"]
    ks = A["G"]["kstars"] + A["L"]["kstars"]
    v["gate0_median_kstar"] = float(np.median(ks)) if ks else None
    pB = out["partB"]
    nw_g = pB["family_hit_rates"].get("G|null_window", {"hits": 0, "n": 0})
    nw_l = pB["family_hit_rates"].get("L|null_window", {"hits": 0, "n": 0})
    hits, n = nw_g["hits"] + nw_l["hits"], nw_g["n"] + nw_l["n"]
    v["gate0_window_null"] = dict(hits=hits, n=n, rate=hits / n if n else None,
                                  ci=cp_ci(hits, n))
    v["gate0_bundles"] = pB["n_bundles"]
    v["gate0_fisher_p"] = float(fisher_exact([[pB["n_bundles"], 0],
                                              [hits, n - hits]])[1]) if n else None
    v["gate0_pass"] = bool(v["gate0_median_kstar"] is not None
                           and v["gate0_median_kstar"] <= 16
                           and v["gate0_fisher_p"] is not None
                           and v["gate0_fisher_p"] < 0.05)
    if not v["gate0_pass"]:
        v["label"] = "C. CHAOTIC-SET — Gate 0 fails"
        return v
    t1rows = dict(out["T1_null_vs_null"]["rows"])
    cands = []
    for f in CANDIDATE_FEATURES:
        d = t1rows.get(f)
        if not d or not np.isfinite(d["mean_auc"]) or d["n_groups"] < 8:
            continue
        # frozen clause: AUC >= 0.65, CI above 0.5, Wilcoxon p < 0.05, and the
        # sign consistent on >= 2 of the 3 benchmarks (x base) panels
        if (d["mean_auc_oriented"] >= 0.65
                and d["auc_oriented_ci"][0] > 0.5
                and d["wilcoxon_p"] < 0.05
                and d.get("n_ds_beyond_chance", 0) >= 2):
            cands.append((f, d["mean_auc_oriented"], d["auc_oriented_ci"][0],
                          d["direction"], d.get("auc_by_ds"),
                          d.get("n_ds_beyond_chance")))
    v["T1_candidates"] = cands
    both = []
    for f, a, ci_lo, dirn, byds, nds in cands:
        best = None
        for k, blk in out["T3_prefix_hit_vs_miss"].items():
            e = blk["features"].get(f)
            if not e:
                continue
            # frozen wording: the same feature must separate successful from
            # unsuccessful size-matched sets across instances at p < 0.05
            if not np.isfinite(e["mw_p"]):
                continue
            same_dir = ((e["pooled_auc"] > 0.5) == (a > 0.5))
            if e["mw_p"] < 0.05 and same_dir:
                if best is None or e["mw_p"] < best[2]:
                    best = (k, e["loo_auc"], e["mw_p"], e["pooled_auc"])
        if best:
            both.append((f, dirn, a, best, byds, nds))
    v["A_features_T1_and_T3"] = both
    sh = [pB["family_hit_rates"].get(f"{b}|shift", {"rate": None, "n": 0})
          for b in ("G", "L")]
    bl = [pB["family_hit_rates"].get(f"{b}|blockperm", {"rate": None, "n": 0})
          for b in ("G", "L")]
    v["shift_rates"] = [s["rate"] for s in sh]
    v["blockperm_rates"] = [b["rate"] for b in bl]
    v["order_degrades"] = bool(
        (v["shift_rates"][0] is not None and v["shift_rates"][0] < 0.5)
        or (v["blockperm_rates"][0] is not None and v["blockperm_rates"][0] < 0.5))
    v["cross_bank_jaccard"] = out["cross_bank_bundle_jaccard"]["mean"]
    if both:
        v["label"] = "A. STRUCTURED-BUNDLE"
    elif v["order_degrades"]:
        v["label"] = "B. SCORE-ORDER-ONLY"
    else:
        v["label"] = "C. CHAOTIC-SET"
    return v


def report(out):
    print("\n===== S3-B =====")
    for b in ("G", "L"):
        d = out["partA"][b]
        n_class = (d["n_base_wrong"] if b == "G" else d["n_class"])
        print(f"[Part A {b}] class n={n_class} "
              f"rescued<=32 {d['n_rescued']} rate {d['rescue_rate']:.3f} "
              f"median k*={d['median_kstar']}")
        print("    per ds:", {ds: f"{v['rescued']}/{v['n']} k*{v['kstars']}"
                              for ds, v in d["per_ds"].items()})
        print("    prefix hit-rate by k:", " ".join(
            f"{k}:{v['hit_rate']:.2f}" for k, v in sorted(
                d["curve"].items(), key=lambda x: int(x[0]))))
    pB = out["partB"]
    print(f"[Part B] bundles {pB['n_bundles']}  group-only "
          f"{pB['group_only_rate']:.3f}  monadic {pB['monadic_rate']:.3f}  "
          f"frac members sufficient-alone {pB['mean_frac_sufficient']:.3f}  "
          f"frac members necessary {pB['mean_frac_necessary']:.3f}  "
          f"window-null hit {pB['window_null_hit_rate']:.3f}  "
          f"determinism max |dL| {pB['determinism_max_abs_dL']}")
    for kk, v in pB["family_hit_rates"].items():
        print(f"    {kk:26s} {v['hits']:4d}/{v['n']:4d} = {v['rate']:.3f} "
              f"CI[{v['ci'][0]:.3f},{v['ci'][1]:.3f}]")
    print("[T1] within-instance null-vs-null (candidate structure only):")
    for f, d in out["T1_null_vs_null"]["rows"]:
        if f not in CANDIDATE_FEATURES or not np.isfinite(d["mean_auc"]):
            continue
        print(f"    {f:20s} n={d['n_groups']:3d} AUC {d['mean_auc']:.3f} "
              f"oriented {d['mean_auc_oriented']:.3f} "
              f"CI[{d['auc_oriented_ci'][0]:.3f},{d['auc_oriented_ci'][1]:.3f}] "
              f"({d['direction']}) agree {d['frac_groups_on_pooled_side']:.2f} "
              f"rb {d['rank_biserial']:+.2f} p {d['wilcoxon_p']:.3g}")
    print("[T3] cross-instance prefix hit-vs-miss at matched k:")
    for k, blk in sorted(out["T3_prefix_hit_vs_miss"].items(),
                         key=lambda x: int(x[0])):
        print(f"    k={k:>2s} {blk['n_hit']}/{blk['n_rows']} hits:", "  ".join(
            f"{f} AUC {e['pooled_auc']:.2f} (CV {e['loo_auc']:.2f}) p {e['mw_p']:.3g}"
            for f, e in blk["top"][:4]))
    print("[T3 partial] does the shape signal survive the prefix's own strength?")
    for k, ent in sorted(out["T3_partial_on_strength"].items(), key=lambda x: int(x[0])):
        if not ent:
            continue
        px = {q: next(iter(ent.values())).get(f"proxy_{q}") for q in PROXIES}
        best = sorted(((f, e) for f, e in ent.items()
                       if np.isfinite(e.get("resid_on_sc_mass_frac", np.nan))),
                      key=lambda x: -abs(x[1]["raw_auc"] - 0.5))[:4]
        print(f"    k={k:>2s} proxies alone: " +
              " ".join(f"{q} {v:.3f}" for q, v in px.items() if v) + " |")
        for f, e in best:
            print(f"        {f:20s} raw {e['raw_auc']:.3f} "
                  f"(within-ds {e['raw_auc_within_ds']:.3f}, CV {e['raw_loo']:.3f})"
                  f" -> resid mass {e['resid_on_sc_mass_frac']:.3f}"
                  f" (wd {e['resid_on_sc_mass_frac_within_ds']:.3f},"
                  f" CV {e['resid_on_sc_mass_frac_loo']:.3f})"
                  f" | resid score {e['resid_on_sc_g_mean']:.3f}")
    print("[T4] does a fixed-size probe set predict HOW DEEP the queue must go?")
    for r, blk in sorted(out["T4_depth_from_shape"].items(), key=lambda x: int(x[0])):
        print(f"    reference size r={r} (n={blk['n_instances']} instances, "
              f"Spearman rho vs k*):")
        for f, e in blk["top"][:6]:
            if abs(e.get("rho") or 0) < 0.15:
                continue
            print(f"        {f:20s} rho {e['rho']:+.3f} (p {e['p']:.3g}) "
                  f"uncensored {e.get('rho_rescued_only')} "
                  f"| partial-mass {e.get('rho_partial_mass')} "
                  f"mass-only-partial {e.get('rho_mass_only_partial')}")
    if "crop" in out:
        print("[crop]", json.dumps(out["crop"], default=float))
    print("[base rates] kind | rescuable vs non-rescued:")
    for kk, v in out["base_rates"].items():
        print(f"    {kk:28s} {v['hits']:4d}/{v['n']:4d} = {v['rate']:.3f} "
              f"(inst {v['n_instances']})")
    print("[rescuability covariates]",
          json.dumps(out["rescuability_covariates"], indent=1, default=float))
    if "band_ladder" in out:
        print("[bands, pooled]  mean T_only position -> P(hit):")
        for kk, v in out["band_ladder"]["pooled"].items():
            print(f"    {kk:12s} rank~{str(v['mean_rank']):>6s}  "
                  f"{v['hits']:4d}/{v['n']:4d} = {v['rate']:.3f}")
        bl = out["band_ladder"]["common_support"]
        print(f"[bands, common support n={bl['n_instances']}]")
        for kk, v in bl["bands"].items():
            print(f"    {kk:12s} {v['hits']:3d}/{v['n']:3d} = {v['rate']:.3f}")
    print("[cross-bank bundle Jaccard]", out["cross_bank_bundle_jaccard"])
    print("[verdict]", json.dumps(out["verdict"], indent=1, default=float))


if __name__ == "__main__":
    main()
