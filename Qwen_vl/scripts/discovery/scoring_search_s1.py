"""
S1: offline, rank-changing scoring sweep on EADP's existing relevance signal.

Zero model calls. Everything is recomputed from the branch maps saved in
`outputs/discovery/fa_maps/*_b256.npz` (global_sim, local_sim) joined with the
occlusion-derived necessary-block labels in `probe_*.npz`.

Search axes (all downstream of the two branches, hence all zero-cost):
  branch_norm : how each branch is put on a common scale *before* fusion
  alpha       : fusion weight, S = alpha * global' + (1 - alpha) * dense'
  lambda      : smoothing strength, S <- (1 - lambda) * S + lambda * smooth3x3(S)

Metrics are computed per instance and then aggregated across instances; tokens
are never pooled as independent samples. Block-level AUROC/AP use one score per
8x8 block (mean of its tokens), matching the occlusion granularity.
"""
import glob
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import (GRID, BLOCK, MAPS, OUT, minmax, rank_pct,
                      spatial_smooth, tokens_of_blocks)

TIERS = [
    ("loc64", "nNec == 64 (primary)", lambda n: n == 64),
    ("loc128", "nNec <= 128", lambda n: n <= 128),
    ("loc256", "nNec <= 256", lambda n: n <= 256),
    ("all", "all causal cases", lambda n: True),
]
RECALL_KS = (128, 256, 512)


# --------------------------------------------------------------------------
# branch scaling
# --------------------------------------------------------------------------
def pool_stats():
    """Dataset-level branch statistics over all 150 saved maps."""
    G, D = [], []
    for p in sorted(glob.glob(os.path.join(MAPS, "*_b256.npz"))):
        z = np.load(p)
        G.append(z["global_sim"].reshape(-1))
        D.append(z["local_sim"].reshape(-1))
    G, D = np.concatenate(G), np.concatenate(D)
    return {
        "g_med": np.median(G), "g_iqr": np.percentile(G, 75) - np.percentile(G, 25),
        "g_mu": G.mean(), "g_sd": G.std(),
        "d_med": np.median(D), "d_iqr": np.percentile(D, 75) - np.percentile(D, 25),
        "d_mu": D.mean(), "d_sd": D.std(),
    }


def scale_branches(g, d, mode, S):
    """Return the two branches after the chosen normalization."""
    if mode == "none":                      # official: branches fused raw
        return g, d
    if mode == "img_mm":                    # per-image min-max, per branch
        return minmax(g), minmax(d)
    if mode == "img_z":                     # per-image z-score, per branch
        return (g - g.mean()) / (g.std() + 1e-8), (d - d.mean()) / (d.std() + 1e-8)
    if mode == "ds_robust":                 # shared median/IQR scaling
        return ((g - S["g_med"]) / S["g_iqr"], (d - S["d_med"]) / S["d_iqr"])
    if mode == "ds_z":                      # shared z-score
        return (g - S["g_mu"]) / S["g_sd"], (d - S["d_mu"]) / S["d_sd"]
    if mode == "img_rank":                  # per-image rank normalize (non-affine)
        f = lambda x: np.argsort(np.argsort(x)) / (len(x) - 1.0)
        return f(g), f(d)
    raise ValueError(mode)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def auroc_ap(scores, labels):
    """Mann-Whitney AUROC + average precision, with tie handling."""
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, np.nan
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            ranks[inv == i] = ranks[inv == i].mean()
    rp = ranks[:len(pos)].sum()
    auroc = (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    # AP
    ordv = np.argsort(-allv, kind="stable")
    lab = labels[ordv]
    tp = np.cumsum(lab)
    prec = tp / np.arange(1, len(lab) + 1)
    ap = float(np.sum(prec * lab) / lab.sum()) if lab.sum() else np.nan
    return float(auroc), ap


def evaluate(rec, score):
    """Per-instance metrics for one candidate score vector."""
    nec = tokens_of_blocks(rec["needed"])
    p = rank_pct(score)
    order = np.argsort(-score)

    # block-level: one score per 8x8 block
    bs = score.reshape(GRID // BLOCK, BLOCK, GRID // BLOCK, BLOCK).mean(axis=(1, 3)).reshape(-1)
    lab = np.zeros(len(bs), dtype=bool)
    lab[rec["needed"]] = True
    au, ap = auroc_ap(bs, lab)

    out = {
        "n_nec_tokens": int(len(nec)),
        "mean_rank_pct": float(np.mean(p[nec])),
        "median_rank_pct": float(np.median(p[nec])),
        "block_auroc": au,
        "block_ap": ap,
    }
    for k in RECALL_KS:
        out[f"recall@{k}"] = float(len(set(nec.tolist()) & set(order[:k].tolist())) / len(nec))
    return out


# --------------------------------------------------------------------------
def main():
    recs = []
    for p in sorted(glob.glob(os.path.join(MAPS, "probe_*.npz"))):
        base = os.path.basename(p)[len("probe_"):-len(".npz")]
        z = np.load(p)
        if len(z["needed_blocks"]) == 0:
            continue
        b = np.load(os.path.join(MAPS, f"{base}_b256.npz"))
        recs.append(dict(
            key=base, ds=base.rsplit("_", 1)[0],
            needed=z["needed_blocks"],
            g=b["global_sim"].reshape(-1).astype(np.float64),
            d=b["local_sim"].reshape(-1).astype(np.float64),
        ))
    S = pool_stats()

    # ---- official anchor -------------------------------------------------
    official = {}
    for r in recs:
        official[r["key"]] = evaluate(r, spatial_smooth(0.5 * r["g"] + 0.5 * r["d"]))

    # ---- grid ------------------------------------------------------------
    NORMS = ["none", "img_mm", "img_z", "ds_robust", "ds_z"]
    EXTRA = ["img_rank"]
    ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
    LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]

    results = []
    for norm, alpha, lam in itertools.product(NORMS, ALPHAS, LAMBDAS):
        if alpha in (0.0, 1.0) and norm != "none" and norm not in ("img_rank",):
            continue                        # single-branch: branch scaling is a no-op
        per = {}
        for r in recs:
            g, d = scale_branches(r["g"], r["d"], norm, S)
            fused = alpha * g + (1 - alpha) * d
            score = (1 - lam) * fused + lam * spatial_smooth(fused)
            per[r["key"]] = evaluate(r, score)
        results.append(dict(norm=norm, alpha=alpha, lam=lam, per=per))

    # extra non-affine arm, reported separately
    extra = []
    for alpha, lam in itertools.product(ALPHAS, LAMBDAS):
        per = {}
        for r in recs:
            g, d = scale_branches(r["g"], r["d"], "img_rank", S)
            fused = alpha * g + (1 - alpha) * d
            score = (1 - lam) * fused + lam * spatial_smooth(fused)
            per[r["key"]] = evaluate(r, score)
        extra.append(dict(norm="img_rank", alpha=alpha, lam=lam, per=per))

    json.dump({"official": official,
               "results": results + extra,
               "rec_keys": [r["key"] for r in recs],
               "rec_nnec": [int(len(r["needed"]) * BLOCK * BLOCK) for r in recs]},
              open(os.path.join(OUT, "s1_scoring_search.json"), "w"), indent=1)

    # ---- reporting -------------------------------------------------------
    def agg(res, mask):
        keys = [r["key"] for r in recs
                if mask(len(r["needed"]) * BLOCK * BLOCK)]
        m = [res["per"][k]["mean_rank_pct"] for k in keys]
        d = [res["per"][k]["mean_rank_pct"] - official[k]["mean_rank_pct"] for k in keys]
        return dict(n=len(keys), mean=float(np.mean(m)), median=float(np.median(m)),
                    delta=float(np.mean(d)),
                    improved=int(sum(x < -1e-9 for x in d)),
                    worsened=int(sum(x > 1e-9 for x in d)))

    print("=" * 100)
    print("official anchor (rank pct, 0 = most important)")
    for r in recs:
        o = official[r["key"]]
        n = len(r["needed"]) * BLOCK * BLOCK
        print(f"  {r['key']:22s} nNec={n:4d}  mean={o['mean_rank_pct']:.4f} "
              f"med={o['median_rank_pct']:.4f}  AUROC={o['block_auroc']:.3f} "
              f"AP={o['block_ap']:.3f}  R@128={o['recall@128']:.3f} "
              f"R@256={o['recall@256']:.3f} R@512={o['recall@512']:.3f}")
    print("=" * 100)

    for tier_key, tier_name, fn in TIERS:
        print(f"\n### {tier_name}")
        rows = sorted(results, key=lambda x: agg(x, fn)["delta"])
        print(f"{'norm':10s} {'alpha':>5s} {'lam':>5s} {'n':>2s} {'mean':>7s} "
              f"{'med':>7s} {'delta':>8s} {'impr':>5s} {'wors':>5s}")
        for res in rows[:8]:
            a = agg(res, fn)
            if a["n"] == 0:
                continue
            print(f"{res['norm']:10s} {res['alpha']:5.2f} {res['lam']:5.2f} {a['n']:2d} "
                  f"{a['mean']:7.4f} {a['median']:7.4f} {a['delta']:+8.4f} "
                  f"{a['improved']:5d} {a['worsened']:5d}")
        print("  ... worst 3:")
        for res in rows[-3:]:
            a = agg(res, fn)
            print(f"{res['norm']:10s} {res['alpha']:5.2f} {res['lam']:5.2f} {a['n']:2d} "
                  f"{a['mean']:7.4f} {a['median']:7.4f} {a['delta']:+8.4f} "
                  f"{a['improved']:5d} {a['worsened']:5d}")
    print(f"\n[saved] {os.path.join(OUT, 's1_scoring_search.json')}")


if __name__ == "__main__":
    main()
