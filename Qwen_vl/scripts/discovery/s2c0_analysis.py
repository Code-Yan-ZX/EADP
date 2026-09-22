"""
S2-C0 step 2: teacher agreement + causal evaluation for the forward proxies.

Two questions, kept separate:
  1. *diagnostic* -- how well does each forward proxy rank-agree with the S2-B
     P1-G2 gradient teacher (per image Spearman, Top-K overlap)?
  2. *gate* -- does the proxy localize the occlusively necessary regions as well
     as the teacher does, on the same 15 causal cases and the same metrics?

Only the second one gates. Metrics are the S2-A ones, reused verbatim via
``scoring_search_s1.evaluate``; the official / S1-best / P1-G2 reference numbers
are read from the S2-A deliverable rather than recomputed.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import BLOCK, OUT
from scoring_search_s1 import evaluate

VARIANTS = {"A": "attention only", "B": "attention x ||V||", "C": "exact output contribution"}
REF = ["official", "s1_best", "P1_G2", "P2_G2"]
TIERS = {"loc256": lambda n: n <= 256, "all": lambda n: True}
KS = (128, 256, 512)


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def overlap(a, b, k):
    ta = set(np.argsort(-a)[:k].tolist())
    tb = set(np.argsort(-b)[:k].tolist())
    return len(ta & tb) / float(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c0_proxy_analysis")
    args = ap.parse_args()

    Z = np.load(os.path.join(OUT, "s2c0_forward_proxy.npz"))
    meta = json.load(open(os.path.join(OUT, "s2c0_forward_proxy_meta.json")))["meta"]
    G = np.load(os.path.join(OUT, "s2b_gradient_scores.npz"))
    S2A = json.load(open(os.path.join(OUT, "s2a_gradient_viability.json")))["cases"]

    layers = json.load(open(os.path.join(OUT, "s2c0_forward_proxy_meta.json")))["layers"]
    arms = [f"{v}_L{L}" for L in layers for v in ("A", "B", "C")]

    keys = sorted(meta)
    pilot = [k for k in keys if meta[k]["in_pilot"]]
    causal = [k for k in keys if meta[k]["in_causal"]]
    print(f"[sets] {len(keys)} instances scored: {len(pilot)} pilot, {len(causal)} causal "
          f"({len(set(pilot) & set(causal))} overlap)")

    # ------------------------------------------------------------------ 1. agreement
    # The teacher map is P1-G2 from s2b_gradient_scores.npz (the frozen 150, which
    # contains the pilot). The 15 causal cases sit *outside* the frozen bank and
    # S2-A's raw maps for them were overwritten by its own consolidation step, so
    # causal-set agreement is not available without re-running the teacher
    # backward; the pilot is also where the accuracy gate is measured, so it is
    # the set that matters here.
    teacher_maps = {k: G[k].astype(np.float64) for k in keys if k in G.files}
    print(f"[agreement set] {len(teacher_maps)} instances have a P1-G2 teacher map")

    agree = {}
    for arm in arms:
        sp, ov = [], {kk: [] for kk in KS}
        for k in keys:
            if f"{arm}__{k}" not in Z.files or k not in teacher_maps:
                continue
            a, t = Z[f"{arm}__{k}"].astype(np.float64), teacher_maps[k]
            sp.append(spearman(a, t))
            for kk in KS:
                ov[kk].append(overlap(a, t, kk))
        agree[arm] = dict(
            n=len(sp),
            spearman_mean=float(np.mean(sp)),
            spearman_median=float(np.median(sp)),
            spearman_p10=float(np.percentile(sp, 10)),
            spearman_p90=float(np.percentile(sp, 90)),
            **{f"top{kk}_overlap": float(np.mean(ov[kk])) for kk in KS},
        )

    print("\n" + "=" * 112)
    print(f"TEACHER AGREEMENT vs P1-G2 (per image, then aggregated; n={agree[arms[0]]['n']} "
          f"= pilot {len(pilot)} + causal, chance TopK = k/1024)")
    print(f"{'arm':10s}{'spearman':>10s}{'median':>9s}{'p10':>8s}{'p90':>8s}"
          f"{'Top128':>9s}{'Top256':>9s}{'Top512':>9s}")
    for arm in arms:
        a = agree[arm]
        print(f"{arm:10s}{a['spearman_mean']:10.3f}{a['spearman_median']:9.3f}"
              f"{a['spearman_p10']:8.3f}{a['spearman_p90']:8.3f}"
              f"{a['top128_overlap']:9.3f}{a['top256_overlap']:9.3f}{a['top512_overlap']:9.3f}")

    # ------------------------------------------------------------------ 2. causal
    needed = {k: np.array(S2A[k]["needed"]) for k in causal}
    nnec = {k: int(len(needed[k]) * BLOCK * BLOCK) for k in causal}
    metrics = {}
    for k in causal:
        rec = dict(needed=needed[k])
        metrics[k] = {}
        for arm in arms:
            metrics[k][arm] = evaluate(rec, Z[f"{arm}__{k}"].astype(np.float64))
        for r in REF:
            metrics[k][r] = S2A[k]["metrics"]["P1_G2" if r == "P1_G2" else r]

    def agg(method, fn, field):
        ks = [k for k in causal if fn(nnec[k])]
        v = np.array([metrics[k][method][field] for k in ks], dtype=float)
        o = np.array([metrics[k]["official"][field] for k in ks], dtype=float)
        d = v - o
        return dict(n=len(ks), mean=float(np.nanmean(v)), official=float(np.nanmean(o)),
                    delta=float(np.nanmean(d)), improved=int(np.sum(d < -1e-9)),
                    worsened=int(np.sum(d > 1e-9)))

    print("\n" + "=" * 112)
    print("CAUSAL EVALUATION -- PRIMARY TIER nNec<=256, n=8  (negatives in rank/delta are better)")
    print(f"{'method':22s}{'rank':>9s}{'delta':>9s}{'impr':>6s}{'wors':>6s}"
          f"{'AUROC':>8s}{'AP':>7s}{'R@128':>8s}{'R@256':>8s}{'R@512':>8s}")
    rows = {}
    for m in REF + arms:
        a = agg(m, TIERS["loc256"], "mean_rank_pct")
        au = agg(m, TIERS["loc256"], "block_auroc")
        ap = agg(m, TIERS["loc256"], "block_ap")
        rs = [agg(m, TIERS["loc256"], f"recall@{kk}")["mean"] for kk in KS]
        rows[m] = dict(rank=a, auroc=au, ap=ap,
                       recall={kk: agg(m, TIERS["loc256"], f"recall@{kk}") for kk in KS})
        print(f"{m:22s}{a['mean']:9.4f}{a['delta']:+9.4f}{a['improved']:6d}{a['worsened']:6d}"
              f"{au['mean']:8.3f}{ap['mean']:7.3f}"
              + "".join(f"{r:8.3f}" for r in rs))

    # ---- secondary tier: all 15 causal ----
    print("\n" + "=" * 112)
    print("CAUSAL EVALUATION -- ALL 15 CAUSAL CASES (reference only; the primary gate is nNec<=256)")
    print(f"{'method':22s}{'rank':>9s}{'delta':>9s}{'impr':>6s}{'wors':>6s}{'AUROC':>8s}")
    allrows = {}
    for m in REF + arms:
        a = agg(m, TIERS["all"], "mean_rank_pct")
        au = agg(m, TIERS["all"], "block_auroc")
        allrows[m] = dict(rank=a, auroc=au)
        print(f"{m:22s}{a['mean']:9.4f}{a['delta']:+9.4f}{a['improved']:6d}{a['worsened']:6d}"
              f"{au['mean']:8.3f}")

    # ---- per-case, primary tier ----
    prim = sorted([k for k in causal if nnec[k] <= 256], key=lambda k: (meta[k]["dataset"], k))
    print("\n" + "=" * 112)
    print("PER-CASE mean necessary-token rank percentile, primary tier nNec<=256 (lower = better)")
    print(f"{'case':20s}{'nNec':>5s}{'official':>10s}{'P1_G2':>9s}" +
          "".join(f"{a:>9s}" for a in arms))
    per_case = {}
    for k in prim:
        per_case[k] = dict(nnec=nnec[k],
                           official=metrics[k]["official"]["mean_rank_pct"],
                           P1_G2=metrics[k]["P1_G2"]["mean_rank_pct"],
                           **{a: metrics[k][a]["mean_rank_pct"] for a in arms})
        print(f"{k:20s}{nnec[k]:5d}{metrics[k]['official']['mean_rank_pct']:10.4f}"
              f"{metrics[k]['P1_G2']['mean_rank_pct']:9.4f}"
              + "".join(f"{metrics[k][a]['mean_rank_pct']:9.4f}" for a in arms))

    json.dump({"arms": arms, "variants": VARIANTS, "agreement": agree,
               "causal_primary": rows, "causal_all": allrows, "per_case_primary": per_case,
               "n_pilot": len(pilot), "n_causal": len(causal)},
              open(os.path.join(OUT, f"{args.tag}.json"), "w"), indent=1)
    print(f"\n[saved] {os.path.join(OUT, f'{args.tag}.json')}")


if __name__ == "__main__":
    main()
