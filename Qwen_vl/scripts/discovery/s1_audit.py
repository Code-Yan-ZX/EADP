"""
S1 correctness audit: verify the EADP scoring chain offline from saved artifacts.

Checks
  A. the saved `<ds>_<idx>_b256.npz` branch maps and the `probe_<ds>_<idx>.npz`
     causal labels refer to the same forward pass (importance must match);
  B. `fused == alpha*global + (1-alpha)*local` on the raw saved branches;
  C. an independent numpy reimplementation of the official chain
     (min-max -> 3x3 reflect-pad Gaussian -> **beta) reproduces the saved
     `importance_post_smooth` / `importance` exactly;
  D. the per-image min-max and the beta exponent are rank no-ops, i.e. the
     rank order of `smooth(raw fused)` equals that of the saved `importance`;
  E. the official necessary-token mean rank percentile reproduces 0.525.

Nothing here calls the model. Run with the qwen3vl_clean env.
"""
import glob
import json
import os

import numpy as np

OUT = "/media/disk2/YZX/research/EADP/Qwen_vl/outputs/discovery"
MAPS = os.path.join(OUT, "fa_maps")
GRID, BLOCK = 32, 8
N_BLOCKS = (GRID // BLOCK) ** 2


# --- official operators, reimplemented in numpy -----------------------------
def gaussian_kernel(kernel_size=3, sigma=1.0):
    x = np.arange(kernel_size) - (kernel_size - 1) / 2
    xg, yg = np.meshgrid(x, x, indexing="ij")
    r = np.sqrt(xg ** 2 + yg ** 2)
    k = np.exp(-(r ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def spatial_smooth(imp, grid_h=GRID, grid_w=GRID, kernel_size=3, sigma=1.0):
    """Matches pruner._spatial_smoothing_impl (reflect padding)."""
    k = gaussian_kernel(kernel_size, sigma)
    pad = kernel_size // 2
    img = imp.reshape(grid_h, grid_w)
    p = np.pad(img, pad, mode="reflect")
    out = np.zeros_like(img)
    for i in range(kernel_size):
        for j in range(kernel_size):
            out += k[i, j] * p[i:i + grid_h, j:j + grid_w]
    return out.reshape(-1)


def minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-6)


def rank_pct(score):
    """Rank percentile, 0 = most important. Matches failure_analysis.py."""
    return np.argsort(np.argsort(-score)) / len(score)


def load_causal():
    """The 15 instances that carry occlusion-derived necessary labels."""
    recs = []
    for p in sorted(glob.glob(os.path.join(MAPS, "probe_*.npz"))):
        base = os.path.basename(p)[len("probe_"):-len(".npz")]
        ds, idx = base.rsplit("_", 1)
        z = np.load(p)
        nb = z["needed_blocks"]
        if len(nb) == 0:
            continue                      # probe found no necessary block
        b = np.load(os.path.join(MAPS, f"{base}_b256.npz"))
        recs.append(dict(
            key=base, ds=ds, idx=int(idx),
            probe_imp=z["importance"], probe_sel=z["select_idx"], needed=nb,
            g=b["global_sim"].reshape(-1), d=b["local_sim"].reshape(-1),
            fused=b["fused"].reshape(-1),
            imp_raw=b["importance_pre_smooth"].reshape(-1),
            imp_post=b["importance_post_smooth"].reshape(-1),
            imp=b["importance"].reshape(-1),
        ))
    return recs


def tokens_of_blocks(blocks):
    tok = []
    for b in blocks:
        r, c = divmod(int(b), GRID // BLOCK)
        for i in range(r * BLOCK, r * BLOCK + BLOCK):
            for j in range(c * BLOCK, c * BLOCK + BLOCK):
                tok.append(i * GRID + j)
    return np.array(sorted(tok))


def main():
    recs = load_causal()
    print(f"[A] causal instances loaded: {len(recs)}")
    print(f"    by dataset: "
          f"{ {d: sum(r['ds'] == d for r in recs) for d in ('DocVQA_VAL','OCRBench','TextVQA_VAL')} }")

    # A. same forward pass?
    dmax = max(np.abs(r["probe_imp"] - r["imp"]).max() for r in recs)
    print(f"[A] max |probe importance - b256 importance| over 15 = {dmax:.3e}")

    # B. fusion identity on raw branches
    devs = []
    for r in recs:
        ref = 0.5 * r["g"] + 0.5 * r["d"]
        devs.append(np.abs(ref - r["fused"]).max())
    print(f"[B] max |0.5*global + 0.5*local - fused| = {max(devs):.3e}")

    # C. reproduce the official chain from raw fused
    err_post, err_final, rank_eq = [], [], []
    for r in recs:
        sm = spatial_smooth(minmax(r["fused"]))
        err_post.append(np.abs(sm - r["imp_post"]).max())
        err_final.append(np.abs(sm ** 2.0 - r["imp"]).max())
        rank_eq.append(np.array_equal(np.argsort(-spatial_smooth(r["fused"])),
                                      np.argsort(-r["imp"])))
    print(f"[C] max |reimpl post_smooth - saved| = {max(err_post):.3e}")
    print(f"[C] max |reimpl **2.0 - saved importance| = {max(err_final):.3e}")

    # D. rank no-ops
    print(f"[D] rank(smooth(raw)) == rank(saved importance) on "
          f"{sum(rank_eq)}/{len(rank_eq)} instances")
    # min-max applied to the *final* score is trivially rank neutral; also check
    # that min-max before the linear smoother is rank neutral
    mm_neutral = all(np.array_equal(np.argsort(-minmax(spatial_smooth(r["fused"]))),
                                    np.argsort(-spatial_smooth(r["fused"])))
                     for r in recs)
    beta_neutral = all(np.array_equal(np.argsort(-r["imp_post"] ** 2.0),
                                      np.argsort(-r["imp_post"])) for r in recs)
    print(f"[D] min-max after smoothing rank neutral: {mm_neutral}")
    print(f"[D] beta exponent rank neutral:          {beta_neutral}")

    # E. reproduce the reported 0.525
    pcts = []
    for r in recs:
        p = rank_pct(r["imp"])
        pcts.append(float(np.mean([p[t] for t in tokens_of_blocks(r["needed"])])))
    print(f"[E] official mean necessary-token rank pct = {np.mean(pcts):.4f} "
          f"(report: 0.5251)  per-instance n={len(pcts)}")

    cls = json.load(open(os.path.join(OUT, "fa_classification.json")))
    stored = {r["dataset"] + "_" + str(r["index"]): r["probe_necessary_mean_rank_pct"]
              for r in cls["rows"] if "probe_necessary_mean_rank_pct" in r}
    mism = [(r["key"], abs(stored[r["key"]] - pcts[i]))
            for i, r in enumerate(recs) if r["key"] in stored]
    print(f"[E] max |recomputed - fa_classification| = {max(m for _, m in mism):.3e}")

    # tier sizes under several readings of "localized"
    nnec = np.array([len(r["needed"]) * BLOCK * BLOCK for r in recs])
    for lo, hi, name in [(64, 64, "nNec == 64 (brief's primary)"),
                         (64, 128, "nNec <= 128"),
                         (64, 256, "nNec <= 256"),
                         (64, 100000, "all causal")]:
        m = (nnec >= lo) & (nnec <= hi)
        print(f"    {name:28s} n={m.sum():2d}   "
              f"{[(r['key'], int(n)) for r, n in zip(recs, nnec) if (n >= lo and n <= hi)]}")


if __name__ == "__main__":
    main()
