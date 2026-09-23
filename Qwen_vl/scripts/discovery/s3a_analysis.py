"""
S3-A step 2 (CPU): the conditionality statistics.

Part A  does u(i | S) really depend on S?
    Every add measurement gives a per-instance 12-candidate x 5-context
    matrix  M[i, S] = L(y|S) - L(y|S + {i} - {r}).
    Each entry already subtracts its OWN context reference, so a context-
    level shift cannot leak into the comparison. The two-way decomposition
        M[i,S] = a_i + b_S + c_{iS}
    isolates the token-by-context interaction c_{iS}: that is THE conditional
    effect. Reported per benchmark: interaction variance share, within-token
    spread vs delta scale, cross-context rank correlation / rank-reversal
    fraction / top-3 stability, sign flips (with named cases), plus the same
    on the leave-marginal side for held tokens.

Part B  TextVQA vs DocVQA vs OCRBench: every Part-A statistic split by
    benchmark with bootstrap CIs on the TextVQA-vs-OCRBench contrast.

Part C  direct (unary) vs conditional utility: tokens with LOW teacher rank
    whose marginal utility nonetheless beats every high-rank candidate under
    some context; their share, and their spatial relation to the strongly
    evidenced (high-teacher-rank) retained region.

Reads s3a_nll.json + s3a_cases.json; writes s3a_analysis.json.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                    # noqa: E402
import s3a_common as C                                      # noqa: E402

CTX = ["base", "weak", "strong", "rand", "rand2"]
NB = 4000                                   # bootstrap replicates


def bootstrap_ci(vals, seed=7):
    rng = np.random.default_rng(seed)
    v = np.asarray(list(vals), dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    boot = v[rng.integers(0, len(v), size=(NB, len(v)))].mean(1)
    return dict(mean=float(v.mean()),
                lo=float(np.percentile(boot, 2.5)),
                hi=float(np.percentile(boot, 97.5)),
                n=int(len(v)))


def boot_diff(a, b, seed=11):
    """CI of mean(a) - mean(b); independent resampling of each side."""
    rng = np.random.default_rng(seed)
    a = np.asarray([x for x in a if np.isfinite(x)], float)
    b = np.asarray([x for x in b if np.isfinite(x)], float)
    if len(a) == 0 or len(b) == 0:
        return dict(diff=float("nan"), lo=float("nan"), hi=float("nan"))
    da = a[rng.integers(0, len(a), size=(NB, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), size=(NB, len(b)))].mean(1)
    d = da - db
    return dict(diff=float(a.mean() - b.mean()),
                lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)))


def spearman(x, y):
    rx = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    ry = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall_dist(x, y):
    """fraction of ordered pairs reversed between two rankings."""
    n = len(x)
    tot = rev = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = np.sign(x[i] - x[j])
            b = np.sign(y[i] - y[j])
            if a == 0 or b == 0:
                continue
            tot += 1
            rev += int(a != b)
    return rev / tot if tot else float("nan")


def interaction_share(M):
    """var(c_{iS}) / var(M) with row and column means projected out."""
    ok = np.isfinite(M)
    if ok.sum() < 6:
        return float("nan")
    A = np.where(ok, M, np.nan)
    rm = np.nanmean(A, axis=1, keepdims=True)
    cm = np.nanmean(A, axis=0, keepdims=True)
    gm = np.nanmean(A)
    R = A - rm - cm + gm
    tot = np.nansum((A - gm) ** 2)
    return float(np.nansum(np.where(ok, R, 0) ** 2) / tot) if tot > 0 \
        else float("nan")


# ---------------------------------------------------------------------------
def per_instance_matrices(nll, cases):
    by_key = {c["key"]: c for c in cases}
    out = {}
    for key, recs in nll["meas"].items():
        c = by_key[key]
        P = [int(t) for t in c["P"]]
        J = [int(t) for t in c["J"]]
        ref = {r["ctx"]: r["L"] for r in recs if r["kind"] == "ref"}
        Ma = np.full((len(P), len(CTX)), np.nan)
        Mr = np.full((len(P), len(CTX)), np.nan)
        Ml = np.full((len(J), len(CTX)), np.nan)
        removed = {}
        for r in recs:
            if r["ctx"] not in CTX or r["token"] is None:
                continue
            ci = CTX.index(r["ctx"])
            if r["kind"] in ("add", "add_norem", "leave"):
                pool = P if r["kind"] != "leave" else J
                if r["token"] not in pool:
                    continue
                pi = pool.index(r["token"])
                val = ref[r["ctx"]] - r["L"]
                if r["kind"] == "add":
                    Ma[pi, ci] = val
                    removed[(r["token"], r["ctx"])] = r["removed"]
                elif r["kind"] == "add_norem":
                    Mr[pi, ci] = val
                else:
                    Ml[pi, ci] = val
        floor = next((r["L"] for r in recs if r["kind"] == "floor"), np.nan)
        out[key] = dict(P=P, J=J, M=Ma, Mrem=Mr, Ml=Ml, ref=ref,
                        removed=removed, ds=c["ds"], floor=floor)
    return out


def resolved_mask(m):
    """candidates whose cross-context |delta| spread clears 3x the instance
    bf16 noise floor (the prereg resolution rule)."""
    span = np.nanmax(m["M"], 1) - np.nanmin(m["M"], 1)
    base = np.abs(m["M"][:, 0])
    return np.isfinite(span) & ((np.nan_to_num(span) >= 3 * m["floor"]) |
                                (base >= 3 * m["floor"]))


def _matrix_stats(rows, restrict=None):
    """Part-A moments over the per-instance delta matrices.

    restrict: None = every candidate; else per-instance boolean masks keeping
    only noise-resolved candidates (prereg rule: >= 3x the instance floor).
    """
    shares, spans, base_abs = [], [], []
    pair_stats = {f"{a}~{b}": dict(rho=[], kend=[], jacc3=[], flip=[])
                  for ia, a in enumerate(CTX) for b in CTX[ia + 1:]}
    for j, m in enumerate(rows):
        M = m["M"]
        keep = np.ones(M.shape[0], bool) if restrict is None else restrict[j]
        M = M[keep]
        if M.shape[0] < 4:
            continue
        shares.append(interaction_share(M))
        spans.append(np.nanmax(M, 1) - np.nanmin(M, 1))
        base_abs.append(np.abs(M[:, 0]))
        for key in pair_stats:
            a, b = key.split("~")
            xa, xb = M[:, CTX.index(a)], M[:, CTX.index(b)]
            ok = np.isfinite(xa) & np.isfinite(xb)
            if ok.sum() >= 4:
                d = pair_stats[key]
                d["rho"].append(spearman(xa[ok], xb[ok]))
                d["kend"].append(kendall_dist(xa[ok], xb[ok]))
                ia3 = np.argsort(-xa[ok], kind="stable")[:3]
                ib3 = np.argsort(-xb[ok], kind="stable")[:3]
                ta = {int(np.where(ok)[0][k]) for k in ia3}
                tb = {int(np.where(ok)[0][k]) for k in ib3}
                d["jacc3"].append(len(ta & tb) / len(ta | tb))
                d["flip"].append(int(np.sum((xa[ok] > 0) != (xb[ok] > 0))))
    out = {}
    out["n_used"] = len(shares)
    out["interaction_share"] = bootstrap_ci(shares)
    out["within_token_range_nats"] = bootstrap_ci(
        np.concatenate(spans)) if spans else bootstrap_ci([])
    out["abs_delta_base_nats"] = bootstrap_ci(
        np.concatenate(base_abs)) if base_abs else bootstrap_ci([])
    mb = float(np.nanmean(np.concatenate(base_abs))) if base_abs else float("nan")
    ms = float(np.nanmean(np.concatenate(spans))) if spans else float("nan")
    out["span_over_absbase"] = ms / mb if mb and np.isfinite(mb) and mb > 0 \
        else float("nan")
    out["pairs"] = {k: dict(rho=bootstrap_ci(v["rho"]),
                            kend=bootstrap_ci(v["kend"]),
                            jacc3=bootstrap_ci(v["jacc3"]),
                            flips_per_inst=bootstrap_ci(v["flip"]))
                    for k, v in pair_stats.items()}
    return out


# ---------------------------------------------------------------------------
def part_a_ds(mats, ds):
    rows = [m for m in mats.values() if m["ds"] == ds]
    out = dict(n=len(rows))
    res_masks = [resolved_mask(m) for m in rows]
    out["resolved_frac"] = float(np.mean([r.mean() for r in res_masks])) \
        if res_masks else float("nan")
    out["noise_floor_nats"] = bootstrap_ci([m["floor"] for m in rows])
    out["all"] = _matrix_stats(rows)
    out["resolved"] = _matrix_stats(rows, res_masks)
    # leave side
    lshares, lranges = [], []
    for m in rows:
        Ml = m["Ml"]
        ok = np.isfinite(Ml).sum(1) >= 3
        if not ok.any():
            continue
        lshares.append(interaction_share(Ml[ok]))
        lranges.append(np.nanmax(Ml[ok], 1) - np.nanmin(Ml[ok], 1))
    out["leave_interaction_share"] = bootstrap_ci(lshares)
    out["leave_within_token_range_nats"] = bootstrap_ci(
        np.concatenate(lranges)) if lranges else bootstrap_ci([])
    return out


def part_a(mats):
    return {ds: part_a_ds(mats, ds) for ds in C.DS_ALL}


def sign_flip_cases(mats, cases):
    """prereg threshold: both deltas resolved (>= 3x the instance noise floor)."""
    by_key = {c["key"]: c for c in cases}
    out = []
    for key, m in mats.items():
        c = by_key[key]
        thr = 3.0 * m["floor"]
        for pi, t in enumerate(m["P"]):
            b = m["M"][pi, 0]
            for ci, cn in ((1, "weak"), (2, "strong")):
                o = m["M"][pi, ci]
                if not (np.isfinite(b) and np.isfinite(o)):
                    continue
                if abs(b) >= thr and abs(o) >= thr and (b > 0) != (o > 0):
                    out.append(dict(key=key, ds=m["ds"], token=int(t),
                                    grank=int(c["grank"][str(t)]),
                                    delta_base=float(b), delta_ctx=float(o),
                                    ctx=cn, stratum=c["stratum"],
                                    gap=float(abs(b - o))))
    return sorted(out, key=lambda r: -r["gap"])


def biggest_shifts(mats, cases, n=10):
    by_key = {c["key"]: c for c in cases}
    rows = []
    for key, m in mats.items():
        c = by_key[key]
        for pi, t in enumerate(m["P"]):
            b = m["M"][pi, 0]
            for ci, cn in enumerate(CTX[1:], 1):
                o = m["M"][pi, ci]
                if not (np.isfinite(b) and np.isfinite(o)):
                    continue
                rows.append(dict(key=key, ds=m["ds"], token=int(t),
                                 grank=int(c["grank"][str(t)]), ctx=cn,
                                 d_base=float(b), d_ctx=float(o),
                                 shift=float(o - b),
                                 removed_ctx=m["removed"].get((int(t), cn))))
    rows.sort(key=lambda r: -abs(r["shift"]))
    return rows[:n], rows[-n:]


def part_c(mats, cases):
    by_key = {c["key"]: c for c in cases}
    per = {ds: dict(n_low=0, n=0, dist_all=[], dist_hi=[], dist_lowhigh=[],
                    grank_lowhigh=[]) for ds in C.DS_ALL}
    detail = []
    rho_uc, rho_uc_w = [], []
    for key, m in mats.items():
        c = by_key[key]
        ds = m["ds"]
        gr = np.array([c["grank"][str(t)] for t in m["P"]], float)
        bands = {t: b for b, ts in c["P_bands"].items() for t in ts}
        bandv = np.array([bands[int(t)] for t in m["P"]])
        best = np.nanmax(m["M"], axis=1)          # best marginal over contexts
        hi_best = [best[j] for j in range(len(m["P"])) if bandv[j] == "hi"
                   and np.isfinite(best[j])]
        hi_best_max = max(hi_best) if hi_best else np.inf
        S = [int(t) for t in c["S"]]
        gmap = {int(k): v for k, v in c["grank"].items()}
        # strong evidence retained tokens: teacher rank < 32 and in S_base
        # (grank map covers P and J only; recompute membership via case lists)
        strong_ev = [t for t in S if t in set(c["drop_weak"])]  # weak drops =
        # exactly the top-W by teacher score among S -- the strongest evidence
        for pi, t in enumerate(m["P"]):
            t = int(t)
            if not (np.isfinite(best[pi])):
                continue
            per[ds]["n"] += 1
            low_unary = gr[pi] >= 256                       # outside teacher top-256
            high_cond = best[pi] > 0 and best[pi] > hi_best_max
            if low_unary and high_cond:
                per[ds]["n_low"] += 1
                if strong_ev:
                    d = min(C.grid_dist(t, e) for e in strong_ev)
                    per[ds]["dist_lowhigh"].append(float(d))
                    per[ds]["grank_lowhigh"].append(float(gr[pi]))
                detail.append(dict(key=key, ds=ds, token=t, band=str(bandv[pi]),
                                   grank=int(gr[pi]), best_delta=float(best[pi]),
                                   delta_base=float(m["M"][pi, 0]),
                                   dist_strong=(float(min(C.grid_dist(t, e)
                                        for e in strong_ev))
                                        if strong_ev else None)))
            if strong_ev:
                d = min(C.grid_dist(t, e) for e in strong_ev)
                per[ds]["dist_all"].append(float(d))
                if bandv[pi] == "hi":
                    per[ds]["dist_hi"].append(float(d))
        db = m["M"][:, 0]
        ok = np.isfinite(db)
        if ok.sum() >= 6:
            rho_uc.append(spearman(-gr[ok], db[ok]))
            dw = m["M"][:, 1]
            ok2 = ok & np.isfinite(dw)
            if ok2.sum() >= 6:
                rho_uc_w.append(spearman(-gr[ok2], dw[ok2]))
    stats = {ds: dict(
        low_unary_high_cond=dict(frac=v["n_low"] / max(v["n"], 1),
                                 n_low=v["n_low"], n=v["n"]),
        dist_strong_hi_band=bootstrap_ci(v["dist_hi"]),
        dist_strong_low_unary_high_cond=bootstrap_ci(v["dist_lowhigh"]),
        grank_lowhigh=bootstrap_ci(v["grank_lowhigh"])) for ds, v in per.items()}
    return dict(per_ds=stats,
                rho_unary_vs_cond_base=bootstrap_ci(rho_uc),
                rho_unary_vs_cond_weak=bootstrap_ci(rho_uc_w),
                notable=detail[:80],
                n_detail=len(detail))


def removal_identity_check(nll):
    base_add = {}
    ri = defaultdict(list)
    for key, recs in nll["meas"].items():
        for r in recs:
            if r["ctx"] != "base":
                continue
            if r["kind"] == "add":
                base_add[(key, r["token"])] = r
            elif r["kind"] == "ri":
                ri[(key, r["token"])].append(r)
    shifts, rrank_gap = [], []
    for k, arr in ri.items():
        if k not in base_add:
            continue
        L0 = base_add[k]["L"]
        for r in arr:
            shifts.append(abs(r["L"] - L0))
            g0 = base_add[k]["removed_grank"]
            g1 = r["removed_grank"]
            if g0 is not None and g1 is not None:
                rrank_gap.append(abs(g0 - g1))
    return dict(n=len(shifts),
                abs_L_shift_nats_alt_removal=bootstrap_ci(shifts),
                removed_grank_gap_mean=float(np.mean(rrank_gap))
                if rrank_gap else float("nan"))


def budget_check(mats):
    dif = []
    for m in mats.values():
        d = np.nanmax(np.abs(m["M"] - m["Mrem"]))
        if np.isfinite(d):
            dif.append(float(d))
    return dict(max_abs_add_minus_addnorem=bootstrap_ci(dif))


def part_b(mats):
    """benchmark contrast on the headline conditionality numbers
    (resolved candidates only -- unresolved entries are noise)."""
    per = {ds: part_a_ds(mats, ds) for ds in C.DS_ALL}
    res = {ds: [m for m in mats.values() if m["ds"] == ds] for ds in C.DS_ALL}
    shares, ranges = {}, {}
    for ds in C.DS_ALL:
        masks = [resolved_mask(m) for m in res[ds]]
        sh, rg = [], []
        for m, mk in zip(res[ds], masks):
            M = m["M"][mk]
            if M.shape[0] >= 4:
                sh.append(interaction_share(M))
                rg.append(float(np.nanmax(np.nanmax(M, 1) - np.nanmin(M, 1))))
        shares[ds], ranges[ds] = sh, rg
    return dict(
        textvqa_vs_ocrbench__interaction_share=boot_diff(
            shares["TextVQA_VAL"], shares["OCRBench"]),
        textvqa_vs_ocrbench__within_token_range=boot_diff(
            ranges["TextVQA_VAL"], ranges["OCRBench"]),
        docvqa_vs_ocrbench__interaction_share=boot_diff(
            shares["DocVQA_VAL"], shares["OCRBench"]),
        per_ds=per)


def part_d(rescue_path):
    """Rescue table: per arm x k, per benchmark: mean dL, rescue rate,
    break-free improvement, plus cond-vs-unary paired contrast."""
    if not os.path.exists(rescue_path):
        return None
    r = json.load(open(rescue_path))["runs"]
    arms = [f"{a}_k{k}" for a in ("unary", "cond", "spatial", "random")
            for k in (8, 16)]
    out = {}
    for ds in C.DS_ALL:
        keys = [k for k, v in r.items() if v["ds"] == ds]
        t = dict(n=len(keys))
        for arm in arms:
            recs = [r[k][arm] for k in keys if arm in r[k]]
            if not recs:
                continue
            t[arm] = dict(n=len(recs),
                          dL=float(np.mean([x["dL"] for x in recs])),
                          hit=float(np.mean([x["hit"] for x in recs])),
                          rescued=sum(1 for x in recs if x["rescued"]),
                          rescue_rate=float(np.mean(
                              [1.0 if x["rescued"] else 0.0 for x in recs])))
        # paired per-instance contrasts (same base, both arms present)
        for pair in (("cond_k8", "unary_k8"), ("cond_k16", "unary_k16")):
            a, b = pair
            both = [k for k in keys if a in r[k] and b in r[k]]
            if both:
                dd = np.array([r[k][a]["dL"] - r[k][b]["dL"] for k in both])
                rr = np.array([float(r[k][a]["rescued"]) -
                               float(r[k][b]["rescued"]) for k in both])
                t[f"{a}_minus_{b}"] = dict(
                    dL=bootstrap_ci(dd), rescue=bootstrap_ci(rr))
        # added-token band composition
        bandmix = defaultdict(list)
        for arm in arms:
            mix = defaultdict(int)
            tot = 0
            for k in keys:
                if arm not in r[k]:
                    continue
                for t_ in r[k][arm]["added"]:
                    idx = r[k]["pool"].index(t_) if t_ in r[k]["pool"] else None
                    if idx is not None:
                        mix[r[k]["bands"][idx]] += 1
                        tot += 1
            bandmix[arm] = {b: v / max(tot, 1) for b, v in mix.items()}
        out[ds] = t
    out["_band_mix_global"] = {}
    for arm in arms:
        mix = defaultdict(int)
        tot = 0
        for k, v in r.items():
            if arm not in v:
                continue
            for t_ in v[arm]["added"]:
                idx = v["pool"].index(t_) if t_ in v["pool"] else None
                if idx is not None:
                    mix[v["bands"][idx]] += 1
                    tot += 1
        out["_band_mix_global"][arm] = {b: round(vv / max(tot, 1), 3)
                                        for b, vv in sorted(mix.items())}
    return dict(per_ds=out, n_instances=len(r),
                base_wrong_rescuable_keys=sorted(r.keys()))


def main():
    cases = json.load(open(os.path.join(OUT, C.CASES_JSON)))["cases"]
    nll = json.load(open(os.path.join(OUT, "s3a_nll.json")))
    mats = per_instance_matrices(nll, cases)

    rep = {}
    rep["part_a"] = part_a(mats)
    rep["part_b"] = part_b(mats)
    sf = sign_flip_cases(mats, cases)
    rep["sign_flips"] = dict(n=len(sf),
                             by_ds={ds: sum(1 for r in sf if r["ds"] == ds)
                                    for ds in C.DS_ALL},
                             examples=sf[:24])
    big, small = biggest_shifts(mats, cases)
    rep["biggest_shifts"] = big
    rep["part_c"] = part_c(mats, cases)
    rep["removal_identity"] = removal_identity_check(nll)
    rep["part_d"] = part_d(os.path.join(OUT, "s3a_rescue.json"))
    rep["budget_check"] = budget_check(mats)
    # correlation of add-delta with the removed token's teacher rank
    # (base context): tests "the effect is really about what got dropped"
    grs, dels = [], []
    for key, recs in nll["meas"].items():
        ref = next(r["L"] for r in recs if r["kind"] == "ref"
                   and r["ctx"] == "base")
        for r in recs:
            if r["kind"] == "add" and r["ctx"] == "base" \
                    and r["removed_grank"] is not None:
                grs.append(r["removed_grank"])
                dels.append(ref - r["L"])
    rep["delta_vs_removedrank_base"] = dict(
        spearman=spearman(np.array(grs, float), np.array(dels, float)),
        n=len(grs))
    preds = nll["preds"]
    rep["base_pred_injection_regime"] = dict(
        n=len(preds),
        hit_mean=float(np.mean([p["hit"] for p in preds.values()])),
        by_ds={ds: float(np.mean([p["hit"] for k, p in preds.items()
                                  if k.rsplit("_", 1)[0] == ds]))
               for ds in C.DS_ALL})
    # agreement of the injection-regime verdict with the frozen m2 outcomes
    # (informational: the two paths differ by construction -- GDEP runs
    # L0-L4 at full length, the injection path does not)
    by_key = {c["key"]: c for c in cases}
    ag = {}
    for fld in ("hit_G", "hit_B1", "hit_B0"):
        tot = ok = 0
        for k2, p in preds.items():
            if k2 in by_key:
                tot += 1
                ok += int((p["hit"] >= C.THR) ==
                          (by_key[k2][fld] >= C.THR))
        ag[fld] = ok / max(tot, 1)
    rep["base_pred_injection_regime"]["agreement"] = ag
    path = os.path.join(OUT, "s3a_analysis.json")
    json.dump(rep, open(path, "w"), indent=1)
    print(f"[saved] {path}")
    _digest(rep)


def _digest(rep):
    for ds in C.DS_ALL:
        a = rep["part_a"][ds]
        print(f"\n== {ds} (n={a['n']}, resolved_frac={a['resolved_frac']:.2f}, "
              f"floor={a['noise_floor_nats']['mean']:.3f}) ==")
        for tag in ("all", "resolved"):
            g = a[tag]
            if not g["n_used"]:
                continue
            i = g["interaction_share"]
            print(f"  [{tag:8s} n={g['n_used']:2d}] share "
                  f"{i['mean']:.3f} [{i['lo']:.3f},{i['hi']:.3f}]  "
                  f"range {g['within_token_range_nats']['mean']:.3f} "
                  f"vs |d|base {g['abs_delta_base_nats']['mean']:.3f} "
                  f"ratio {g['span_over_absbase']:.2f}")
            for pair in ("base~weak", "base~strong", "rand~rand2"):
                p = g["pairs"].get(pair)
                if p and p["rho"]["n"]:
                    print(f"    rho({pair}) {p['rho']['mean']:+.3f} "
                          f"[{p['rho']['lo']:.2f},{p['rho']['hi']:.2f}]  "
                          f"kend {p['kend']['mean']:.3f}  "
                          f"jac3 {p['jacc3']['mean']:.3f}")
        l = a["leave_interaction_share"]
        print(f"  leave share {l['mean']:.3f} [{l['lo']:.3f},{l['hi']:.3f}]")
    print("\nPart B contrast:")
    for k, v in rep["part_b"].items():
        if k.startswith("textvqa") or k.startswith("docvqa"):
            print(f"  {k}: {v['diff']:+.3f} [{v['lo']:.3f},{v['hi']:.3f}]")
    print("\nsign flips:", rep["sign_flips"]["n"], rep["sign_flips"]["by_ds"])
    print("removal identity |shift| nats:", json.dumps(
        rep["removal_identity"]["abs_L_shift_nats_alt_removal"]))
    print("budget (add vs add_norem):", json.dumps(rep["budget_check"][
        "max_abs_add_minus_addnorem"]))
    print("delta vs removed-rank rho:", rep.get("delta_vs_removedrank_base"))
    print("unary vs cond rho:", {k: round(rep["part_c"][k]["mean"], 3)
                                 for k in ("rho_unary_vs_cond_base",
                                           "rho_unary_vs_cond_weak")})
    print("low-unary/high-cond:", {ds: rep["part_c"]["per_ds"][ds][
        "low_unary_high_cond"] for ds in C.DS_ALL})
    if rep.get("part_d"):
        print("\nPart D (k=16):")
        for ds in C.DS_ALL:
            t = rep["part_d"]["per_ds"][ds]
            print(f"  {ds} n={t['n']}")
            for arm in ("unary_k16", "cond_k16", "spatial_k16", "random_k16"):
                if arm in t:
                    print(f"    {arm:11s} dL={t[arm]['dL']:+.3f} "
                          f"hit={t[arm]['hit']:.2f} "
                          f"rescued={t[arm]['rescued']}/{t[arm]['n']}")


if __name__ == "__main__":
    main()
