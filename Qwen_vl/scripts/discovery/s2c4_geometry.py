"""
S2-C4 step 3: the geometry of the per-image oracle directions.

If the 450 oracle directions w_i (unit vectors in R^4096, mean-difference centroid
directions of each image's teacher Top-32) occupy a low-dimensional subspace, an
image-adaptive scorer is at least *conceivable*: a cheap descriptor could pick a
point in that subspace. If they are near-isotropic, no low-dimensional adaptive
mechanism can approximate them and the per-image differences are noise from the
point of view of any shared or routed read-out.

Reported, with no claim beyond what the numbers show:

  * pairwise cosine similarity, within and across benchmarks
  * cosine to the global mean direction (how much of each w_i is shared)
  * PCA spectrum: explained variance, participation-ratio effective rank, and the
    number of components needed for 50/80/90/95 % of the variance
  * whether the leading components align with benchmark identity, tested by a
    contingency table of a k-means partition against the benchmark label, and by
    the benchmark means of the leading coefficients

The PCA itself is fitted on the **fit split only** (240 directions); the held-out
150 are projected onto it, never used to build it. That is the same basis
``s2c4_pca.py`` then uses for the reconstruction bound, so the two agree.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                        # noqa: E402
from s2c4_common import K_HEAD, dump, load_cache, unit          # noqa: E402
from s2c2_common import DS_ALL                                  # noqa: E402


def pca_fit(W, k_max=None):
    """PCA of a direction matrix (n, d). Returns mean, components, eigenvalues."""
    mu = W.mean(axis=0)
    X = W - mu
    # economical SVD: n << d
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    lam = (s ** 2) / max(len(W) - 1, 1)
    if k_max:
        Vt, lam = Vt[:k_max], lam[:k_max]
    return mu.astype(np.float32), Vt.astype(np.float32), lam


def pca_project(W, mu, Vt):
    return (W - mu) @ Vt.T


def effective_rank(lam):
    lam = np.asarray(lam, dtype=np.float64)
    lam = lam[lam > 0]
    pr = float(lam.sum() ** 2 / np.square(lam).sum())            # participation ratio
    cum = np.cumsum(lam) / lam.sum()
    dims = {f"n_for_{int(t*100)}pct": int(np.searchsorted(cum, t) + 1)
            for t in (0.50, 0.80, 0.90, 0.95, 0.99)}
    return pr, cum, dims


def kmeans(X, k, seed=0, iters=100):
    """Tiny deterministic k-means++ (no sklearn dependency)."""
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d = np.min(((X[:, None, :] - np.array(C)[None]) ** 2).sum(-1), axis=1)
        p = d / d.sum() if d.sum() > 0 else np.ones(len(X)) / len(X)
        C.append(X[rng.choice(len(X), p=p)])
    C = np.array(C)
    for _ in range(iters):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j]
                         for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c4")
    args = ap.parse_args()

    C = load_cache()
    W32, split, ds = C["W32"], C["split"], C["ds"]
    fit = np.nonzero(split == "fit")[0]
    val = np.nonzero(split == "val")[0]
    te = np.nonzero(split == "test")[0]

    Wf, Wv, Wt = W32[fit], W32[val], W32[te]
    dsf, dst = ds[fit], ds[te]

    # ---- pairwise cosine similarity -----------------------------------------
    def cos_stats(A, lab=None, name=""):
        S = A @ A.T
        n = len(A)
        iu = np.triu_indices(n, 1)
        off = S[iu]
        out = dict(name=name, n=n, mean=float(off.mean()), sd=float(off.std()),
                   p05=float(np.percentile(off, 5)),
                   p50=float(np.percentile(off, 50)),
                   p95=float(np.percentile(off, 95)),
                   frac_negative=float((off < 0).mean()),
                   frac_above_0p5=float((off > 0.5).mean()))
        if lab is not None:
            same = np.array([lab[i] == lab[j] for i, j in zip(*iu)])
            out["within_benchmark_mean"] = float(off[same].mean())
            out["across_benchmark_mean"] = float(off[~same].mean())
            out["within"] = {b: float(off[same & (np.array(
                [lab[i] == b for i, _ in zip(*iu)]))].mean()) for b in DS_ALL}
        return out, off

    fit_cos, off_fit = cos_stats(Wf, dsf, "fit directions (240)")
    test_cos, off_te = cos_stats(Wt, dst, "held-out directions (150)")

    # cosine to the global mean direction
    g = unit(Wf.mean(axis=0))
    cos_to_global = dict(
        fit=dict(mean=float((Wf @ g).mean()), sd=float((Wf @ g).std()),
                 p05=float(np.percentile(Wf @ g, 5)),
                 p95=float(np.percentile(Wf @ g, 95))),
        test=dict(mean=float((Wt @ g).mean()), sd=float((Wt @ g).std())),
        by_benchmark={b: float((Wt[dst == b] @ g).mean()) for b in DS_ALL},
    )

    # ---- PCA on the fit split ------------------------------------------------
    mu_d, Vt, lam = pca_fit(Wf)
    pr, cum, dims = effective_rank(lam)
    evr = (lam / lam.sum())
    print(f"PCA on {len(Wf)} fit directions (d={Wf.shape[1]}):")
    print(f"  effective rank (participation ratio) = {pr:.2f}")
    print(f"  dims for 50/80/90/95/99% variance   = "
          f"{dims['n_for_50pct']}/{dims['n_for_80pct']}/{dims['n_for_90pct']}/"
          f"{dims['n_for_95pct']}/{dims['n_for_99pct']}")
    print(f"  top-10 explained-variance ratios     = "
          f"{np.round(evr[:10], 4).tolist()}")
    print(f"  top-10 cumulative                    = "
          f"{np.round(cum[:10], 4).tolist()}")
    print(f"  mean cosine between a fit direction and the global mean = "
          f"{cos_to_global['fit']['mean']:.4f}")

    # ---- do the leading components carry benchmark identity? ----------------
    Zf = pca_project(Wf, mu_d, Vt[:32])
    Zt = pca_project(Wt, mu_d, Vt[:32])
    cluster = {}
    for k in (3, 8):
        lab = kmeans(Zf, k, seed=0)
        cont = np.zeros((k, len(DS_ALL)), int)
        for a, b in zip(lab, dsf):
            cont[a, DS_ALL.index(b)] += 1
        # purity: fraction of the fit set in the majority benchmark of its cluster
        purity = float(cont.max(axis=1).sum() / cont.sum())
        cluster[f"k={k}"] = dict(
            contingency={f"c{j}": {b: int(cont[j, i]) for i, b in enumerate(DS_ALL)}
                         for j in range(k)},
            purity_wrt_benchmark=purity,
            sizes=[int((lab == j).sum()) for j in range(k)],
            means_by_benchmark_shape="see contingency",
        )
    # how much of a leading coefficient is explained by benchmark identity alone
    bm_effect = {}
    for c in range(6):
        z = Zf[:, c]
        grand = z.mean()
        between = np.mean([(z[dsf == b].mean() - grand) ** 2 for b in DS_ALL])
        bm_effect[f"pc{c+1}"] = dict(
            var_total=float(z.var()),
            var_between_benchmark=float(between),
            frac_explained_by_benchmark=float(between / z.var()) if z.var() > 0 else 0.0,
            means={b: float(z[dsf == b].mean()) for b in DS_ALL})
    print("\n  fraction of each leading coefficient's variance explained by "
          "benchmark identity alone:")
    for c in range(6):
        e = bm_effect[f"pc{c+1}"]
        print(f"    PC{c+1}: {e['frac_explained_by_benchmark']:.4f}   "
              f"means {np.round(list(e['means'].values()), 3).tolist()}")

    np.savez_compressed(
        os.path.join(OUT, f"{args.tag}_geometry.npz"),
        pca_mean=mu_d, pca_components=Vt[:256], pca_eigenvalues=lam[:256],
        cos_offdiag_fit=off_fit, cos_offdiag_test=off_te,
        pca_coords_fit=Zf, pca_coords_test=Zt,
        fit_rows=fit, val_rows=val, test_rows=te,
    )
    print(f"\n[saved] {args.tag}_geometry.npz")

    dump(f"{args.tag}_geometry.json", dict(
        note=__doc__,
        k_head=K_HEAD,
        pairwise_cosine={"fit": fit_cos, "test": test_cos},
        cosine_to_global=cos_to_global,
        pca=dict(
            n_fit=int(len(Wf)), effective_rank_participation_ratio=pr,
            dims_for_variance=dims,
            evr_top20=[float(x) for x in evr[:20]],
            cumulative_top20=[float(x) for x in cum[:20]],
            evr_at=[dict(k=int(k), cumulative=float(cum[k - 1]))
                    for k in (1, 2, 4, 8, 16, 32, 64, 128, 240) if k <= len(cum)],
        ),
        clusters_wrt_benchmark=cluster,
        leading_coefficients_by_benchmark=bm_effect,
    ))


if __name__ == "__main__":
    main()
