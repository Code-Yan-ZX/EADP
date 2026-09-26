"""
M7 step 2 -- Phase 2, fixed-budget set construction.

    S_final = core(256 - |U|)  UNION  U          |S_final| = 256 exactly

Core constructions, all cheap and none of them trained:

    C0  B2 prefix      keep the first 256-|U| of the incumbent's OWN selection
                       order, then add U.  The evicted tokens are therefore the
                       last |U| the greedy picked -- the ones its own marginal
                       gain rated lowest.  This is the primary arm.
    C1  redundancy     evict the |U| most redundant retained tokens (highest
                       cosine to the rest of S0).  Auxiliary only.
    C2  importance     evict the |U| lowest-importance retained tokens.

Controls, all with the SAME per-instance replacement budget and the SAME core
construction as the union arm they are matched to:

    COS   `cos_s0c` nominating |U_i| tokens -- the matched-slot single proxy.
          This is the arm the whole stage turns on.
    RND   |U_i| uniformly random dropped tokens.

`CORE` is a deliberately budget-breaking diagnostic: the C0 core with nothing
added, at 256-|U| tokens.  It measures what the shrink costs on its own, which
is the one confound a 256-token-only grid cannot separate.

Two panels:
    bank  the frozen 450's test 150 -- the screening panel
    ext   M1's 720-row extension set -- the independent confirmation panel

Nothing here reads the teacher.  The teacher is used afterwards, in
`m7_analyze.py`, and never to build a set.

Usage
    python scripts/discovery/m7_sets.py --panel bank
    python scripts/discovery/m7_sets.py --panel ext
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m6_common import (BUDGET, OUTPUT_DIR, Dropped, dump_json, load_bank,
                       load_eapd_order)                                  # noqa: E402
from m7_common import DropOnly, SUBSETS, build_union, cos_topn, random_n      # noqa: E402

# (arm, subset, m, core, rescue-source)  -- frozen before generation ran
ARMS = [
    ("U16",  "E-all6", 16, "C0", "union"),
    ("U12",  "E-all6", 12, "C0", "union"),
    ("U8",   "E-all6", 8,  "C0", "union"),
    ("COS16", "E-all6", 16, "C0", "cos"),
    ("COS12", "E-all6", 12, "C0", "cos"),
    ("RND16", "E-all6", 16, "C0", "random"),
    ("RND12", "E-all6", 12, "C0", "random"),
    ("CORE16", "E-all6", 16, "C0", "none"),
    ("CORE12", "E-all6", 12, "C0", "none"),
    ("U16R", "E-all6", 16, "C1", "union"),
    ("U16I", "E-all6", 16, "C2", "union"),
]
RNG_SEED = 20260927


def core_c0(order, n_keep):
    """First `n_keep` picks of the incumbent's own greedy order."""
    return set(order[:n_keep].tolist())


def core_c1(X, fi, s0, n_evict):
    """Evict the `n_evict` most redundant retained tokens (max cosine to the
    rest of S0) -- the incumbent's own redundancy column, not a new rule."""
    red = X[:, fi["red_s0"]]
    return set(s0[np.argsort(-red[s0], kind="stable")[n_evict:]].tolist())


def core_c2(X, fi, s0, n_evict):
    """Evict the `n_evict` lowest-importance retained tokens."""
    imp = X[:, fi["imp"]]
    return set(s0[np.argsort(imp[s0], kind="stable")[n_evict:]].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=("bank", "ext"), default="bank")
    ap.add_argument("--arms", nargs="+", default=None)
    args = ap.parse_args()

    if args.panel == "bank":
        bank = load_bank()
        dropped = Dropped(bank)
        order = load_eapd_order(bank)
        keys, s0_all, X = bank["key"], bank["s0"], bank["X"]
        fi = bank["fi"]
        want = [i for i in range(bank["n"]) if bank["split"][i] == "test"]
    else:
        z = np.load(os.path.join(OUTPUT_DIR, "m7_ext.npz"), allow_pickle=False)
        keys = [str(k) for k in z["key"]]
        s0_all, X, order = z["s0"], z["X"], z["order"]
        fi = {str(n): i for i, n in enumerate(z["feature_names"])}
        dropped = DropOnly(s0_all)
        want = list(range(len(keys)))

    # `build_union`/`cos_topn` only ever read X and the feature index, so a
    # two-key wrapper lets both panels share the identical nomination code
    XW = {"X": X, "fi": fi}
    arms = [a for a in ARMS if args.arms is None or a[0] in args.arms]
    print(f"[panel] {args.panel}: {len(want)} instances; arms {[a[0] for a in arms]}")

    # the greedy's first 256 picks must be exactly B2's set -- without this the
    # "prefix" is not a prefix of anything the incumbent actually does
    bad = [keys[i] for i in want
           if not np.array_equal(np.sort(order[i][:BUDGET]),
                                 np.sort(np.asarray(s0_all[i], dtype=np.int64)))]
    assert not bad, f"greedy order is not B2's order: {bad[:3]}"
    print("[gate] greedy order[:256] sorted == s0: PASS")

    out = dict(arms={}, panel=args.panel, rng_seed=RNG_SEED, n=len(want))
    for arm, sub, m, core, src in arms:
        members = SUBSETS[sub]
        recs, sizes, cores = {}, [], []
        for i in want:
            s0 = np.asarray(s0_all[i], dtype=np.int64)
            U = build_union(XW, dropped, i, m, members)
            nU = int(U.size)
            # `build_union` indexes the DROPPED set; the core lives in the full
            # 1024-token space, so every rescue is mapped back through `drop[i]`
            if src == "none":
                rescue = np.array([], dtype=np.int64)
            elif src == "cos":
                rescue = dropped.drop[i][cos_topn(XW, dropped, i, nU)]
            elif src == "random":
                rescue = dropped.drop[i][random_n(dropped, i, nU, RNG_SEED)]
            else:
                rescue = dropped.drop[i][U]
            n_keep = BUDGET - nU
            if core == "C0":
                c = core_c0(order[i], n_keep)
            elif core == "C1":
                c = core_c1(X[i], fi, s0, nU)
            else:
                c = core_c2(X[i], fi, s0, nU)
            rset = set(np.asarray(rescue).tolist())
            c = set(np.asarray(sorted(c), dtype=np.int64).tolist())
            final = np.array(sorted(c | rset), dtype=np.int64)
            assert final.size == len(set(final.tolist())), f"{arm} {i}: duplicate"
            want_n = BUDGET if src != "none" else n_keep
            assert final.size == want_n, f"{arm} {i}: {final.size} != {want_n}"
            assert not (c & rset), f"{arm} {i}: rescue overlaps core"
            recs[keys[i]] = final.tolist()
            sizes.append(nU)
            cores.append(len(c))
        out["arms"][arm] = dict(subset=sub, m=m, core=core, rescue=src,
                                members=list(members),
                                n_rescue_mean=float(np.mean(sizes)),
                                n_core_mean=float(np.mean(cores)), sets=recs)
        print(f"[{arm:7s}] core={core} rescue={src:6s} |U| {np.mean(sizes):5.1f} "
              f"core {np.mean(cores):5.1f}")

    name = f"m7_sets_{args.panel}.json"
    dump_json(name, out)
    print(f"[saved] {name}")


if __name__ == "__main__":
    main()
