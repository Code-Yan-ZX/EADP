"""
S2-C4 identity check: the vectorised metric path must equal brute force.

``s2c4_common.metrics_matrix`` computes every (held-out image, fit direction)
cell of the transfer matrix with array indexing over a (150, 1024, 240) block.
That is the measurement the whole stage rests on and it is easy to get subtly
wrong -- an early version scored ``head_agree@k`` against the teacher's Top-256
instead of its Top-k, which looks plausible and is a different quantity.

This script recomputes a random sample of cells with a completely separate,
literal implementation (argsort, python sets) and also pins the two S2-C3 §6
cross-checks the stage is calibrated against:

    SELF scored with the Top-32 direction -> recall@32 = 0.9025   (S2-C3 A32)
    SELF scored with the Top-8  direction -> recall@8  = 0.9817   (S2-C3 A)

Exits non-zero if any cell disagrees.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c4")
    ap.add_argument("--cells", type=int, default=25)
    args = ap.parse_args()

    Z = np.load(os.path.join(OUT, f"{args.tag}_transfer.npz"))
    C = {k: v for k, v in np.load(os.path.join(OUT, f"{args.tag}_dirs.npz")).items()}
    W32, keys, mu, sd = C["W32"], C["keys"], C["mu"], C["sd"]
    fit, te = Z["fit_rows"], Z["test_rows"]
    M, M32, A8 = Z["M_recall8"], Z["M_recall32"], Z["M_agree8"]

    G = np.load(os.path.join(OUT, "s2b_gradient_scores.npz"))
    H = np.load(os.path.join(OUT, "s2c1_feats_L4.npy"), mmap_mode="r")

    rng = np.random.default_rng(7)
    bad = []
    for _ in range(args.cells):
        a = int(rng.integers(len(te)))
        b = int(rng.integers(len(fit)))
        j, i = te[a], fit[b]
        h = (H[j].astype(np.float32) - mu) / sd
        sc = (h @ W32[i]).astype(np.float64)
        torder = np.argsort(-G[str(keys[j])].astype(np.float64), kind="stable")
        sorder = np.argsort(-sc, kind="stable")

        sel = set(sorder[:256].tolist())
        r8 = len(sel & set(torder[:8].tolist())) / 8
        r32 = len(sel & set(torder[:32].tolist())) / 32
        ag8 = len(set(sorder[:8].tolist()) & set(torder[:8].tolist())) / 8

        for name, got, want in (("R@8", M[a, b], r8), ("R@32", M32[a, b], r32),
                                ("A@8", A8[a, b], ag8)):
            if abs(got - want) > 1e-9:
                bad.append((a, b, name, float(got), float(want)))

    # S2-C3 §6 calibration points
    self32 = float(Z["pi::SELF::head_recall32"].mean())
    self8 = float(Z["self_top8dir"].mean())
    checks = [("SELF recall@32 vs S2-C3 A32", self32, 0.9025, 2e-3),
              ("SELF_TOP8DIR recall@8 vs S2-C3 A", self8, 0.9817, 2e-3)]
    for name, got, want, tol in checks:
        ok = abs(got - want) <= tol
        if not ok:
            bad.append((name, got, want))
        print(f"  {name:38s} {got:.4f}  (S2-C3 {want:.4f})  "
              f"{'OK' if ok else 'MISMATCH'}")

    print(f"\n  {args.cells} random matrix cells checked against brute force: "
          f"{args.cells*3 - len([x for x in bad if len(x) == 5])} / {args.cells*3} OK")
    if bad:
        for x in bad:
            print("   MISMATCH", x)
        sys.exit(1)
    print("  all checks passed")


if __name__ == "__main__":
    main()
