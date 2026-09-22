"""
S2-C4 step 2: the cross-image transfer matrix, and the deployable arms built on it.

The experimental object is the full matrix

    M[j, i] = head_recall@8 of  w_i  scored on image j,     j in held-out 150,
                                                            i in fit 240

so the stage reports a *distribution* of cross-image transfer, not one mean. Every
deployable arm below is a rule for picking one column of that matrix (or a linear
combination of columns) using only image j's own cheap descriptors.

Arms, in the order the decision rule reads them:

  SELF          w_j scored on image j                         OPTIMISTIC / upper
                                                              bound -- reads j's
                                                              teacher map
  RANDOM_ANY    the mean over all 240 fit directions          the transfer floor
  RANDOM_SAME   the mean over fit directions of image j's own the floor with the
                benchmark only                                benchmark held fixed
  GLOBAL        mean of all 240 fit directions                no adaptation at all
  GLOBAL_DS     mean of the fit directions in j's benchmark   benchmark adaptation,
                                                              no image adaptation
  TRAINED_*     the S2-C3 trained shared directions           the thing to beat
  NN_<desc>     the single nearest fit image by descriptor    image adaptation
  KNN{k}_<desc> mean of the k nearest fit directions          image adaptation

Two controls carry most of the interpretation. RANDOM_SAME and GLOBAL_DS both hold
the benchmark fixed, so an image-adaptive arm that does not beat them has found
*benchmark* structure and nothing image-specific. And GLOBAL is the honest
"no adaptation" floor: if NN cannot beat the plain mean direction, the per-image
variation in w_i is not exploitable by a nearest-neighbour rule.

All the fit-derived arms are evaluated from the single matrix M by column
averaging: score = (sum_{i in S} h . w_i) / ||sum_{i in S} w_i||, which is exactly
the score of the averaged-then-normalised direction, so no arm needs a second pass
over the features.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                            # noqa: E402
from s2c4_common import (BUDGET, DESCRIPTORS, FEATS, HEAD_KS, K_HEAD,  # noqa: E402
                         KS_RETRIEVAL, LAYER, N_VIS, dump, load_cache,
                         metrics_matrix, metrics_single, paired,
                         teacher_orders, trained_direction_matrix, unit)
from s2c2_common import DS_ALL                                      # noqa: E402

TEACHER = "s2b_gradient_scores.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c4")
    args = ap.parse_args()

    C = load_cache()
    W32, W8 = C["W32"], C["W8"]
    split, ds, keys = C["split"], C["ds"], C["keys"]
    mu, sd = C["mu"], C["sd"]
    fit = np.nonzero(split == "fit")[0]
    te = np.nonzero(split == "test")[0]
    n_te, n_fit = len(te), len(fit)
    print(f"fit pool {n_fit} directions | held-out {n_te} images")

    G = np.load(os.path.join(OUT, TEACHER))
    orders_all = teacher_orders(G, [str(k) for k in keys])
    torders = np.stack([orders_all[str(keys[i])] for i in te])       # (150, 1024)

    # ---- the transfer matrix -------------------------------------------------
    H = np.load(os.path.join(OUT, FEATS), mmap_mode="r")
    Wf = W32[fit]                                                    # (240, 4096)
    S = np.empty((n_te, N_VIS, n_fit), np.float32)
    for a in range(0, n_te, 15):
        b = min(a + 15, n_te)
        h = (H[te[a:b]].astype(np.float32) - mu) / sd                # (b-a, 1024, 4096)
        S[a:b] = np.einsum("jtd,di->jti", h, Wf.T, optimize=True)
        print(f"  scored {b}/{n_te} test images x {n_fit} fit directions")

    M = metrics_matrix(S, torders)
    r8, r32, a8 = M["head_recall8"], M["head_recall32"], M["head_agree8"]
    print(f"\ntransfer matrix M: {r8.shape}")
    print(f"  mean cross-image head_recall@8 = {r8.mean():.4f} "
          f"(min {r8.min():.4f}, max {r8.max():.4f})")

    # ---- arms from the matrix ------------------------------------------------
    per_inst, rows = {}, []

    def add(name, scores_or_cols, note="", optimistic=False):
        """score vector per test image -> metrics + per-instance head_recall@8.

        A 3-D input is a stack of per-direction score vectors to be column-averaged.
        The sum is left unnormalised: dividing by ||sum of directions|| is a
        positive rescaling, and Top-K reads only the ranking, so the selection is
        identical (S2-B: calibration is a monotone no-op for Top-K).
        """
        sc = scores_or_cols.sum(axis=2) if scores_or_cols.ndim == 3 else scores_or_cols
        mets = [metrics_single(sc[j], torders[j]) for j in range(n_te)]
        rec = {f"head_recall{k}": np.array([m[f"head_recall{k}"] for m in mets])
               for k in HEAD_KS}
        rec["head_agree8"] = np.array([m["head_agree8"] for m in mets])
        rec["overlap256"] = np.array([m["overlap256"] for m in mets])
        per_inst[name] = rec
        rows.append(dict(
            arm=name, note=note, optimistic=optimistic,
            head_recall8=float(rec["head_recall8"].mean()),
            head_recall16=float(rec["head_recall16"].mean()),
            head_recall32=float(rec["head_recall32"].mean()),
            head_agree8=float(rec["head_agree8"].mean()),
            overlap256=float(rec["overlap256"].mean()),
            **{f"head_recall{k}_sd": float(rec[f"head_recall{k}"].std())
               for k in HEAD_KS},
        ))
        return rec

    # SELF -- reads image j's own teacher map. Upper bound, not a method.
    Wt = np.stack([W32[i] for i in te])
    S_self = np.einsum("jtd,jd->jt", (H[te].astype(np.float32) - mu) / sd, Wt,
                       optimize=True)
    add("SELF", S_self, "w_j on image j, in-sample (OPTIMISTIC upper bound)", True)

    # SELF32 vs SELF8 are the same object at two target scales; report Top-8 dir too
    Wt8 = np.stack([W8[i] for i in te])
    S_self8 = np.einsum("jtd,jd->jt", (H[te].astype(np.float32) - mu) / sd, Wt8,
                        optimize=True)
    add("SELF_TOP8DIR", S_self8, "top-8 oracle direction (OPTIMISTIC)", True)

    # The "random other-image direction" floor is the transfer matrix itself: its
    # column mean IS the expected score of a randomly chosen fit direction, so that
    # arm is the GLOBAL arm and needs no separate row (see transfer_distribution).

    # GLOBAL / GLOBAL_DS / NN / KNN all come from column averaging
    def cols_to_score(cols, wsum_norm=None):
        """Sum selected score columns and divide by ||sum of those directions||."""
        return cols.sum(axis=2)

    # `fit` holds row indices into the full 450-instance cache; S's third axis is
    # position *within* the fit pool, so map one to the other
    pos_of = -np.ones(len(split), int)
    pos_of[fit] = np.arange(n_fit)

    arms_cfg = []
    arms_cfg.append(("GLOBAL", np.arange(n_fit), "mean of all 240 fit directions"))
    for b in DS_ALL:
        idx = pos_of[fit[ds[fit] == b]]
        arms_cfg.append((f"GLOBAL_DS__{b}", idx, f"mean of fit directions in {b}"))

    # descriptor-standardised nearest neighbours
    Z = {k: C[f"desc::{k}"] for k in DESCRIPTORS}
    dfit = {k: v[fit] for k, v in Z.items()}
    dte = {k: v[te] for k, v in Z.items()}
    fd = {k: (v - v.mean(0)) / (v.std(0) + 1e-8) for k, v in dfit.items()}
    td = {k: (v - dfit[k].mean(0)) / (dfit[k].std(0) + 1e-8) for k, v in dte.items()}

    nn_idx = {}
    for dname in DESCRIPTORS:
        D = ((td[dname][:, None, :] - fd[dname][None, :, :]) ** 2).sum(-1)  # (150,240)
        nn_idx[dname] = np.argsort(D, axis=1, kind="stable")
        for k in KS_RETRIEVAL:
            nb = nn_idx[dname][:, :k]                       # (150, k) positions in fit
            cols = np.take_along_axis(S, nb[:, None, :], axis=2)  # (150,1024,k)
            add(f"KNN{k}__{dname}" if k > 1 else f"NN__{dname}", cols,
                f"{k} nearest fit image(s) by {dname} descriptor")

    for name, idx, note in arms_cfg:
        cols = np.take_along_axis(S, np.tile(idx, (n_te, 1))[:, None, :], axis=2)
        add(name, cols, note)

    # trained shared directions (S2-C3), for the comparison the brief cares about
    Hte_std = (H[te].astype(np.float32) - mu) / sd
    tr_rows = []
    for name in ("S2C1_LIN_L4", "HEAD_BIN", "HEAD_RANK"):
        Wt = trained_direction_matrix(name)
        # average the directions (unit vectors), not their score vectors -- the
        # consensus direction is the object that can be compared to an oracle w_i
        sc = Hte_std @ unit(Wt.mean(axis=0))
        add(f"{name}_seedavg", sc, f"S2-C3 trained {name}, 3-seed mean direction")
        for m, s in enumerate(Wt):
            sc1 = Hte_std @ s
            mm = [metrics_single(sc1[j], torders[j]) for j in range(n_te)]
            tr_rows.append(dict(arm=f"{name}_s{m}",
                                head_recall8=float(np.mean(
                                    [x["head_recall8"] for x in mm])),
                                head_agree8=float(np.mean(
                                    [x["head_agree8"] for x in mm]))))
    del Hte_std

    # ---- summarise ------------------------------------------------------------
    r8_self = per_inst["SELF"]["head_recall8"]
    r8_global = per_inst["GLOBAL"]["head_recall8"]
    best_trained = max((r for r in rows if r["arm"].startswith("HEAD_")),
                       key=lambda r: r["head_recall8"])
    ref = per_inst[f"{best_trained['arm']}"]["head_recall8"]

    summary = []
    for r in rows:
        a = r["arm"]
        if a == "RANDOM_ANY":
            continue
        pi = per_inst[a]["head_recall8"]
        d_g = paired(pi, r8_global)
        d_t = paired(pi, ref)
        summary.append(dict(
            **r,
            d_vs_GLOBAL=d_g[0], ci_vs_GLOBAL=[d_g[1], d_g[2]],
            clears_GLOBAL=bool(d_g[1] > 0 or d_g[2] < 0),
            d_vs_best_trained=d_t[0], ci_vs_best_trained=[d_t[1], d_t[2]],
            clears_best_trained=bool(d_t[1] > 0),
        ))
    summary.sort(key=lambda r: -r["head_recall8"])

    print("\n" + "=" * 100)
    print(f"{'arm':28s} {'R@8':>7s} {'R@16':>7s} {'R@32':>7s} {'A@8':>7s} "
          f"{'ov256':>7s}  {'vs GLOBAL':>20s} {'vs best trained':>18s}")
    print("-" * 100)
    for r in summary:
        print(f"{r['arm']:28s} {r['head_recall8']:7.4f} {r['head_recall16']:7.4f} "
              f"{r['head_recall32']:7.4f} {r['head_agree8']:7.4f} "
              f"{r['overlap256']:7.4f}  "
              f"{r['d_vs_GLOBAL']:+6.4f}[{r['ci_vs_GLOBAL'][0]:+.4f},"
              f"{r['ci_vs_GLOBAL'][1]:+.4f}] {r['d_vs_best_trained']:+.4f}")
    print("=" * 100)
    print(f"\nrandom-other-image floor (mean over all 240 columns): "
          f"{r8.mean():.4f}   SELF (optimistic): {r8_self.mean():.4f}")

    # cross-image transfer distribution, recorded as a distribution
    dist = dict(
        mean=float(r8.mean()), median=float(np.median(r8)),
        p05=float(np.percentile(r8, 5)), p95=float(np.percentile(r8, 95)),
        min=float(r8.min()), max=float(r8.max()), sd=float(r8.std()),
        per_image_mean=r8.mean(axis=1),
        per_image_best=r8.max(axis=1),
        per_image_worst=r8.min(axis=1),
    )
    bench_cond = {}
    for b in DS_ALL:
        m = np.array([str(ds[i]) == b for i in te])
        if m.sum():
            bench_cond[b] = float(r8[m].mean())

    np.savez_compressed(
        os.path.join(OUT, f"{args.tag}_transfer.npz"),
        M_recall8=r8, M_recall32=r32, M_agree8=a8,
        self_recall8=r8_self, self_top8dir=per_inst["SELF_TOP8DIR"]["head_recall8"],
        nn_idx=np.stack([nn_idx[k] for k in DESCRIPTORS]),
        descriptors=np.array(DESCRIPTORS),
        fit_rows=fit, test_rows=te,
        ds_test=ds[te], keys_test=keys[te],
        **{f"pi::{a}::{k}": v for a, rec in per_inst.items()
           for k, v in rec.items()},
    )
    print(f"[saved] {args.tag}_transfer.npz")

    dump(f"{args.tag}_transfer.json", dict(
        note=__doc__,
        n_fit=n_fit, n_test=n_te,
        transfer_distribution=dist,
        transfer_by_benchmark=bench_cond,
        arms=summary,
        trained_per_seed=tr_rows,
        best_trained_arm=best_trained["arm"],
        random_any_column_mean=float(r8.mean()),
        self_mean=float(r8_self.mean()),
        self_head_agree8=float(per_inst["SELF"]["head_agree8"].mean()),
        gap_self_minus_global=float(r8_self.mean() - r8_global.mean()),
    ))


if __name__ == "__main__":
    main()
