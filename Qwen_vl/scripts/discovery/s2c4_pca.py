"""
S2-C4 step 4: is the per-image direction approximable by a low-dimensional
adaptive mechanism?

Two levels, deliberately separated, because they answer different questions:

  PCA_ORACLE(k)   project the held-out image's OWN oracle direction onto the
                  fit-set PCA basis truncated to k components, reconstruct, and
                  score. This reads image j's teacher map, so it is an UPPER
                  BOUND: it says how much of w_j's usefulness survives being
                  squeezed into a k-dimensional subspace fitted on other images.
                  It is not a method and is labelled optimistic everywhere.

  PCA_RIDGE(k)    predict those k coefficients from image j's cheap descriptors
                  with a ridge regression fitted on the fit split. Nothing about
                  image j's teacher map enters. This is the deployable arm.

The gap between the two is the stage's central quantity: if PCA_ORACLE(k) is high
and PCA_RIDGE(k) is not, the low-dimensional structure exists but is not
predictable from cheap forward statistics -- which is a different failure from
"there is no low-dimensional structure", and the two point at different next
stages. If PCA_ORACLE(k) itself is low, then no low-dimensional adaptive scorer
can work at any level of descriptor quality.

Hyperparameters (k, ridge alpha) are chosen on the **validation** split, never on
the held-out 150. The full k-curve is reported regardless, so the choice can be
inspected rather than trusted.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                            # noqa: E402
from s2c4_common import (DESCRIPTORS, HEAD_KS, K_HEAD, dump,        # noqa: E402
                         load_cache, metrics_single, paired, teacher_orders,
                         unit)
from s2c2_common import DS_ALL                                      # noqa: E402

TEACHER = "s2b_gradient_scores.npz"
K_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 240)
ALPHAS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)


def ridge_fit(X, Y, alpha):
    """Closed-form ridge with an unpenalised intercept, in dual (kernel) form.

    The primal solution B = (X'X + aI)^-1 X'Y costs d^3, and the largest
    descriptor here is d = 12 288 -- 54 such solves is hours. The dual identity
    B = X'(XX' + aI)^-1 Y costs n^3 with n = 240 fit images, and is exact whenever
    d >= n, which holds for every descriptor in this stage.
    """
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    G = Xc @ Xc.T
    B = Xc.T @ np.linalg.solve(G + alpha * np.eye(len(Xc), dtype=np.float64), Yc)
    return B.astype(np.float32), mx.astype(np.float32), my.astype(np.float32)


def ridge_pred(X, B, mx, my):
    return (X - mx) @ B + my


def metrics_for_scores(scores, torders):
    mets = [metrics_single(scores[j], torders[j]) for j in range(len(scores))]
    return {f"head_recall{k}": np.array([m[f"head_recall{k}"] for m in mets])
            for k in HEAD_KS} | {
        "head_agree8": np.array([m["head_agree8"] for m in mets]),
        "overlap256": np.array([m["overlap256"] for m in mets])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c4")
    args = ap.parse_args()

    C = load_cache()
    W32, split, ds, keys = C["W32"], C["split"], C["ds"], C["keys"]
    fit = np.nonzero(split == "fit")[0]
    val = np.nonzero(split == "val")[0]
    te = np.nonzero(split == "test")[0]
    Wf, Wv, Wt = W32[fit], W32[val], W32[te]

    G = np.load(os.path.join(OUT, TEACHER))
    oa = teacher_orders(G, [str(k) for k in keys])
    tord_te = np.stack([oa[str(keys[i])] for i in te])
    tord_va = np.stack([oa[str(keys[i])] for i in val])

    # ---- features are at test time: descriptors only (no tokens re-read) -----
    # The PCA basis is fit on the fit split's directions, exactly as in
    # s2c4_geometry.py, so the two stages describe one basis.
    mu_d = Wf.mean(axis=0)
    Xf = (Wf - mu_d).astype(np.float64)
    _, _, Vt = np.linalg.svd(Xf, full_matrices=False)
    Vt = Vt.astype(np.float32)                                   # (240, 4096)

    # for scoring we still need the tokens; read them once
    H = np.load(os.path.join(OUT, "s2c1_feats_L4.npy"), mmap_mode="r")
    mu, sd = C["mu"], C["sd"]

    def h_of(rows):
        return (H[rows].astype(np.float32) - mu) / sd

    Hte, Hva = h_of(te), h_of(val)
    Wte = np.stack([W32[i] for i in te])

    rows = []
    per_inst = {}

    def record(name, scores, note, optimistic):
        rec = metrics_for_scores(scores, tord_te)
        per_inst[name] = rec
        txt = dict(arm=name, note=note, optimistic=optimistic)
        for k in HEAD_KS:
            txt[f"head_recall{k}"] = float(rec[f"head_recall{k}"].mean())
        txt["head_agree8"] = float(rec["head_agree8"].mean())
        txt["overlap256"] = float(rec["overlap256"].mean())
        rows.append(txt)
        return rec

    # reference points, recomputed through the same path
    record("SELF", np.einsum("jtd,jd->jt", Hte, Wte, optimize=True),
           "held-out image's own oracle direction (OPTIMISTIC upper bound)", True)

    # ---- level 1: PCA oracle bound ------------------------------------------
    print("PCA_ORACLE: w_j projected onto the fit-set basis truncated to k\n")
    print(f"{'k':>5s} {'R@8':>8s} {'R@16':>8s} {'R@32':>8s} {'A@8':>8s} "
          f"{'ov256':>8s}")
    for k in K_GRID:
        Vk = Vt[:k]
        c = (Wte - mu_d) @ Vk.T                                  # (150, k) oracle coeff
        w_hat = c @ Vk                                           # (150, 4096)
        sc = np.einsum("jtd,jd->jt", Hte, unit_rows(w_hat), optimize=True)
        rec = record(f"PCA_ORACLE_k{k}", sc,
                     f"oracle coefficients in a {k}-component fit-PCA basis "
                     f"(OPTIMISTIC)", True)
        print(f"{k:5d} {rec['head_recall8'].mean():8.4f} "
              f"{rec['head_recall16'].mean():8.4f} "
              f"{rec['head_recall32'].mean():8.4f} "
              f"{rec['head_agree8'].mean():8.4f} "
              f"{rec['overlap256'].mean():8.4f}")

    # ---- level 2: deployable ridge on cheap descriptors ---------------------
    print("\nPCA_RIDGE: coefficients predicted from cheap descriptors "
          "(fit on fit, k and alpha chosen on val)\n")
    out_ridge = {}
    for dname in DESCRIPTORS:
        Xf_d = C[f"desc::{dname}"][fit].astype(np.float64)
        Xv_d = C[f"desc::{dname}"][val].astype(np.float64)
        Xt_d = C[f"desc::{dname}"][te].astype(np.float64)
        xm, xs = Xf_d.mean(0), Xf_d.std(0) + 1e-8
        Xf_z, Xv_z, Xt_z = (Xf_d - xm) / xs, (Xv_d - xm) / xs, (Xt_d - xm) / xs

        curve = {}
        for k in K_GRID:
            Vk = Vt[:k]
            Yf = (Wf - mu_d) @ Vk.T
            Yv = (Wv - mu_d) @ Vk.T
            best = None
            for a in ALPHAS:
                B, mx, my = ridge_fit(Xf_z, Yf, a)
                cp = ridge_pred(Xv_z, B, mx, my)
                wv = unit_rows(cp @ Vk)
                scv = np.einsum("jtd,jd->jt", Hva, wv, optimize=True)
                m = metrics_for_scores(scv, tord_va)["head_recall8"].mean()
                if best is None or m > best[0]:
                    best = (m, a, B, mx, my)
            _, a, B, mx, my = best
            ct = ridge_pred(Xt_z, B, mx, my)
            wt = unit_rows(ct @ Vk)
            sct = np.einsum("jtd,jd->jt", Hte, wt, optimize=True)
            curve[k] = dict(alpha=float(a),
                            val_head_recall8=float(best[0]),
                            test=metrics_for_scores(sct, tord_te))
            print(f"  {dname:16s} k={k:4d}  alpha={a:7.1f}  "
                  f"val R@8={best[0]:.4f}  test R@8={curve[k]['test']['head_recall8'].mean():.4f}")
        out_ridge[dname] = curve
        # the arm at the k chosen on val, as a single headline row
        kbest = max(K_GRID, key=lambda k: curve[k]["val_head_recall8"])
        Vk = Vt[:kbest]
        Yf = (Wf - mu_d) @ Vk.T
        a = curve[kbest]["alpha"]
        B, mx, my = ridge_fit(Xf_z, Yf, a)
        ct = ridge_pred(Xt_z, B, mx, my)
        sc = np.einsum("jtd,jd->jt", Hte, unit_rows(ct @ Vk), optimize=True)
        record(f"PCA_RIDGE__{dname}", sc,
               f"ridge from {dname} descriptor to {kbest} PCA coefficients "
               f"(k and alpha chosen on val)", False)
        out_ridge[dname]["chosen_k"] = int(kbest)

    # ---- who wins -----------------------------------------------------------
    heads = [r for r in rows if not r["optimistic"]]
    best_dep = max(heads, key=lambda r: r["head_recall8"])
    oracle_at_32 = next(r for r in rows if r["arm"] == "PCA_ORACLE_k32")
    oracle_at_240 = next(r for r in rows if r["arm"] == "PCA_ORACLE_k240")

    ref = per_inst["PCA_RIDGE__mean+std"]["head_recall8"]
    summary = []
    for r in rows:
        pi = per_inst[r["arm"]]["head_recall8"]
        d = paired(pi, ref)
        summary.append(dict(**r, d_vs_best_ridge=d[0],
                            ci_vs_best_ridge=[d[1], d[2]]))
    summary.sort(key=lambda r: -r["head_recall8"])

    print("\n" + "=" * 96)
    print(f"{'arm':24s} {'R@8':>8s} {'R@16':>8s} {'R@32':>8s} {'A@8':>8s} {'ov256':>8s}")
    print("-" * 96)
    for r in summary:
        star = "*" if r["optimistic"] else " "
        print(f"{r['arm']:24s}{star}{r['head_recall8']:7.4f} {r['head_recall16']:8.4f} "
              f"{r['head_recall32']:8.4f} {r['head_agree8']:8.4f} {r['overlap256']:8.4f}")
    print("=" * 96)
    print("  * = optimistic (reads the held-out image's own teacher map)")
    print(f"\nbest deployable arm: {best_dep['arm']}  R@8 = "
          f"{best_dep['head_recall8']:.4f}")
    print(f"oracle at k=32 basis: {oracle_at_32['head_recall8']:.4f} | "
          f"k=240 (full basis): {oracle_at_240['head_recall8']:.4f}")

    np.savez_compressed(
        os.path.join(OUT, f"{args.tag}_pca.npz"),
        **{f"pi::{a}::{k}": v for a, rec in per_inst.items() for k, v in rec.items()},
        hte_rows=te, val_rows=val,
    )
    dump(f"{args.tag}_pca.json", dict(
        note=__doc__, k_grid=list(K_GRID), alphas=list(ALPHAS),
        ridge_curves=out_ridge, arms=summary,
        best_deployable=best_dep["arm"],
        best_deployable_recall8=best_dep["head_recall8"],
        oracle_k32=oracle_at_32["head_recall8"],
        oracle_k240=oracle_at_240["head_recall8"],
        descriptor_dims={k: int(C[f"desc::{k}"].shape[1]) for k in DESCRIPTORS},
    ))
    print(f"[saved] {args.tag}_pca.npz / {args.tag}_pca.json")


def unit_rows(W):
    n = np.linalg.norm(W, axis=1, keepdims=True)
    return (W / np.maximum(n, 1e-12)).astype(np.float32)


if __name__ == "__main__":
    main()
