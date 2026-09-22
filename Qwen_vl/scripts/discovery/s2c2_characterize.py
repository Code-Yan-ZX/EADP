"""
S2-C2 step 4: what, if anything, distinguishes the high-value disagreements?

``--extract`` computes a cheap property vector for all 1024 visual tokens of each
held-out instance, from caches only (no LLM forward, no new model, no training):

  g2 / g2_pct        P1-G2 gradient teacher score and its rank percentile
  lin / lin_pct      LIN_L4 student score and its rank percentile
  n_l2, n_l4         hidden-state L2 norm at layer 2 and layer 4
  d_norm             ||h_L4 - h_L2||  (how much the representation moves)
  cos_l2_l4          cosine between the layer-2 and layer-4 representations
  sim_self           mean cosine to the other tokens of the *same* selected set
  sim_other          mean cosine to the tokens of the *other* selected set
  sim_sel_S / _T     mean cosine to S / to T
  nb_sim             mean cosine to the 8 spatial neighbours (local homogeneity)
  uniq               max cosine to any other token (high = a duplicated token)
  row, col, r_center spatial position on the 32x32 grid

``--compare`` then contrasts token groups, bootstrapping **at the instance level**
(a group mean is averaged within instance first, then across resampled
instances), because a token count of tens of thousands is not tens of thousands
of independent observations.

Everything here is descriptive. A property that does not separate the groups is
reported as a negative result, not fitted until it does.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT  # noqa: E402
from s2c2_common import (BUDGET, DS_ALL, FEATURES_META, FEAT_L2, FEAT_L4,  # noqa: E402
                         STUDENT_SCORES, TEACHER_SCORES, cls_name, load_case)

N_TOK = 1024
PROPS = ["g2", "g2_pct", "lin", "lin_pct", "n_l2", "n_l4", "d_norm", "cos_l2_l4",
         "sim_sel_S", "sim_sel_T", "nb_sim", "uniq", "row", "col", "r_center"]
GRID = 32


def pct_rank(x):
    r = np.argsort(np.argsort(x, kind="stable"))
    return r / (len(x) - 1)


def extract():
    cases = load_case()
    meta = json.load(open(os.path.join(OUT, FEATURES_META)))
    jmap = {p["key"]: j for j, p in enumerate(meta["plan"])}

    T = np.load(os.path.join(OUT, TEACHER_SCORES))
    S = np.load(os.path.join(OUT, STUDENT_SCORES))
    hL2 = np.load(os.path.join(OUT, FEAT_L2), mmap_mode="r")
    hL4 = np.load(os.path.join(OUT, FEAT_L4), mmap_mode="r")

    rows, cols = np.divmod(np.arange(N_TOK), GRID)
    r_center = np.sqrt((rows - (GRID - 1) / 2.0) ** 2 + (cols - (GRID - 1) / 2.0) ** 2)
    nb = [[] for _ in range(N_TOK)]
    for t in range(N_TOK):
        r, c = divmod(t, GRID)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < GRID and 0 <= cc < GRID:
                    nb[t].append(rr * GRID + cc)
    nb = [np.asarray(v) for v in nb]

    X = {p: np.zeros((len(cases), N_TOK), dtype=np.float32) for p in PROPS}
    for ci, c in enumerate(cases):
        j = jmap[c["key"]]
        a2 = np.asarray(hL2[j], dtype=np.float32)
        a4 = np.asarray(hL4[j], dtype=np.float32)
        n2 = np.linalg.norm(a2, axis=1)
        n4 = np.linalg.norm(a4, axis=1)
        u2 = a2 / np.maximum(n2[:, None], 1e-6)
        u4 = a4 / np.maximum(n4[:, None], 1e-6)
        sim = u4 @ u4.T                                    # (1024, 1024) layer 4

        Tset = np.zeros(N_TOK, dtype=bool); Tset[c["T"]] = True
        Sset = np.zeros(N_TOK, dtype=bool); Sset[c["S"]] = True

        def mean_sim_to(mask):
            """Mean cosine from every token to the members of `mask`, with a
            token's similarity to itself removed when it is in the mask."""
            tot = sim[:, mask].sum(axis=1)
            cnt = np.full(N_TOK, float(mask.sum()), dtype=np.float32)
            cnt[mask] -= 1.0                                # drop self-similarity
            out = tot / np.maximum(cnt, 1.0)
            return out

        sim_sel_S = mean_sim_to(Sset)
        sim_sel_T = mean_sim_to(Tset)

        nb_sim = np.array([sim[t, nb[t]].mean() for t in range(N_TOK)], dtype=np.float32)
        s_sorted = np.sort(sim, axis=1)
        uniq = s_sorted[:, -2]                              # largest non-self similarity

        g2 = np.asarray(T[f"{c['ds']}_{c['idx']}"], dtype=np.float32)
        lin = np.asarray(S[f"LIN_L4__{c['ds']}_{c['idx']}"], dtype=np.float32)

        X["g2"][ci] = g2; X["g2_pct"][ci] = pct_rank(g2)
        X["lin"][ci] = lin; X["lin_pct"][ci] = pct_rank(lin)
        X["n_l2"][ci] = n2; X["n_l4"][ci] = n4
        X["d_norm"][ci] = np.linalg.norm(a4 - a2, axis=1)
        X["cos_l2_l4"][ci] = (u2 * u4).sum(axis=1)
        X["sim_sel_S"][ci] = sim_sel_S; X["sim_sel_T"][ci] = sim_sel_T
        X["nb_sim"][ci] = nb_sim; X["uniq"][ci] = uniq
        X["row"][ci] = rows; X["col"][ci] = cols; X["r_center"][ci] = r_center
        if (ci + 1) % 30 == 0:
            print(f"  {ci+1}/{len(cases)}")

    path = os.path.join(OUT, "s2c2_token_props.npz")
    np.savez_compressed(path, keys=np.array([c["key"] for c in cases]), **X)
    print(f"[saved] {path}")


# ---------------------------------------------------------------------------
def group_masks(cases, rescue_plan=None):
    """Boolean (n_instances, 1024) masks for each group of interest."""
    n = len(cases)
    G = {}
    names = ["A_rescue_block", "A2_rescue_added", "B_teacher_only", "B2_teacher_only_far",
             "C_student_only", "C2_student_only_removed", "C3_student_only_kept",
             "D_shared", "F_all_added_any"]
    for k in names:
        G[k] = np.zeros((n, N_TOK), dtype=bool)
    plan_by = {}
    if rescue_plan:
        for r in rescue_plan["rescuable"]:
            plan_by[r["key"]] = r

    for ci, c in enumerate(cases):
        S_only, T_only = c["S_only"], c["T_only"]
        m = len(T_only)
        cls = cls_name(c["hit_teacher"], c["hit_student"])
        if cls != "student_wrong_teacher_correct":
            continue
        rec = plan_by.get(c["key"], {})
        kstar = rec.get("ref_k")
        # the level actually used for the rescue (per fallback rule below)
        G["A2_rescue_added"][ci, T_only[:min(kstar or 0, m)]] = True
        G["C2_student_only_removed"][ci, S_only[m - min(kstar or 0, m):]] = True
        blk = rec.get("block")
        if kstar and blk:
            lo = max(kstar - blk, 0)
            G["A_rescue_block"][ci, T_only[lo:kstar]] = True
        G["B_teacher_only"][ci, T_only] = True
        G["C_student_only"][ci, S_only] = True
        G["D_shared"][ci, c["C"]] = True
    # B2 / C3 / F are defined over *all* 150 instances
    for ci, c in enumerate(cases):
        S_only, T_only = c["S_only"], c["T_only"]
        m = len(T_only)
        if cls_name(c["hit_teacher"], c["hit_student"]) == "student_wrong_teacher_correct":
            rec = plan_by.get(c["key"], {})
            kstar = min(rec.get("ref_k") or 0, m)
            G["C3_student_only_kept"][ci, S_only[:m - kstar]] = True
            G["F_all_added_any"][ci, T_only[:kstar]] = True
        else:
            G["C3_student_only_kept"][ci, S_only] = True
        G["B2_teacher_only_far"][ci, T_only] = True
    return G


def boot_diff(a_by_inst, b_by_inst, n_boot=4000, seed=0):
    """Instance-level paired bootstrap of mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    n = len(a_by_inst)
    A = np.asarray(a_by_inst, dtype=float)
    B = np.asarray(b_by_inst, dtype=float)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = A[idx].mean(axis=1) - B[idx].mean(axis=1)
    return (float(np.mean(A) - np.mean(B)),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def per_instance_mean(X, mask):
    """(n, 1024) values + (n, 1024) mask -> per-instance group mean (nan if empty)."""
    out = np.full(len(X), np.nan)
    for i in range(len(X)):
        sel = mask[i]
        if sel.any():
            out[i] = X[i][sel].mean()
    return out


def compare(rescue_plan_path=None):
    z = np.load(os.path.join(OUT, "s2c2_token_props.npz"), allow_pickle=True)
    keys = list(z["keys"])
    cases = load_case()
    assert [c["key"] for c in cases] == keys
    plan = json.load(open(rescue_plan_path)) if rescue_plan_path else None
    if plan:
        for r in plan["rescuable"]:
            r["ref_k"] = r.get("ref_k") or (r["first_rescue_k"] if isinstance(
                r["first_rescue_k"], int) else None)
    G = group_masks(cases, plan)

    pairs = [("A_rescue_block", "B_teacher_only"),
             ("A2_rescue_added", "B_teacher_only"),
             ("B_teacher_only", "C_student_only"),
             ("C2_student_only_removed", "C3_student_only_kept"),
             ("B_teacher_only", "D_shared"),
             ("C_student_only", "D_shared")]

    out = {}
    for scope in DS_ALL + ["ALL"]:
        if scope == "ALL":
            idxs = list(range(len(cases)))
        else:
            idxs = [i for i, c in enumerate(cases) if c["ds"] == scope]
        sc = {}
        for a, b in pairs:
            ma, mb = G[a][idxs], G[b][idxs]
            if ma.sum() == 0 or mb.sum() == 0:
                continue
            rec = {}
            for p in PROPS:
                X = z[p][idxs]
                A = per_instance_mean(X, ma)
                B = per_instance_mean(X, mb)
                ok = ~(np.isnan(A) | np.isnan(B))
                if ok.sum() < 3:
                    continue
                d, lo, hi = boot_diff(A[ok], B[ok])
                sd = np.sqrt((np.nanvar(A[ok]) + np.nanvar(B[ok])) / 2) + 1e-12
                rec[p] = dict(delta=d, lo=lo, hi=hi, cohen_d=d / sd,
                              mean_a=float(np.nanmean(A[ok])),
                              mean_b=float(np.nanmean(B[ok])),
                              n_inst=int(ok.sum()),
                              significant=bool(lo > 0 or hi < 0))
            sc[f"{a}_vs_{b}"] = rec
        out[scope] = sc
    path = os.path.join(OUT, "s2c2_characterize.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[saved] {path}")

    for scope in DS_ALL + ["ALL"]:
        print(f"\n=== {scope} ===")
        for a, b in pairs:
            rec = out[scope].get(f"{a}_vs_{b}")
            if not rec:
                continue
            print(f"  {a} vs {b}")
            for p in PROPS:
                if p not in rec:
                    continue
                r = rec[p]
                star = " *" if r["significant"] else "  "
                print(f"     {p:12s} {r['mean_a']:9.4f} vs {r['mean_b']:9.4f} "
                      f"d={r['cohen_d']:+6.2f}{star}  "
                      f"[{r['lo']:+.4f},{r['hi']:+.4f}] n={r['n_inst']}")


def spatial_stats(rescue_plan_path=None):
    """Group-level spatial structure: is a group one blob, or scattered singles?

    Per instance and per group: the number of distinct 8x8 blocks it touches
    (a token grid of 32x32 has 16 such blocks), the mean grid distance between
    its members, and the mean distance to its own centroid. A group that is one
    contiguous region looks completely different from one that is 30 isolated
    singles -- a distinction no per-token mean can express.
    """
    cases = load_case()
    plan = json.load(open(rescue_plan_path)) if rescue_plan_path else None
    if plan:
        for r in plan["rescuable"]:
            r["ref_k"] = r.get("ref_k") or (r["first_rescue_k"] if isinstance(
                r["first_rescue_k"], int) else None)
    G = group_masks(cases, plan)
    rc, cc = np.divmod(np.arange(N_TOK), GRID)
    pos = np.stack([rc, cc], axis=1).astype(float)

    names = [k for k in G if G[k].any()]
    stats = {}
    for name in names:
        nblk, spread, cent, size = [], [], [], []
        for i in range(len(cases)):
            sel = G[name][i]
            if not sel.any():
                continue
            p = pos[sel]
            nblk.append(len({(int(r) // 8) * 4 + (int(c) // 8) for r, c in p}))
            size.append(len(p))
            if len(p) > 1:
                d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
                spread.append(d[np.triu_indices(len(p), 1)].mean())
            cent.append(np.linalg.norm(p - p.mean(axis=0), axis=1).mean())
        stats[name] = dict(n_inst=len(size), mean_size=float(np.mean(size)),
                           blocks_per_inst=float(np.mean(nblk)),
                           blocks_per_token=float(np.mean(np.array(nblk) / np.array(size))),
                           pairwise_dist=float(np.mean(spread)) if spread else None,
                           radius=float(np.mean(cent)))
    out = {}
    for scope in DS_ALL + ["ALL"]:
        idxs = (list(range(len(cases))) if scope == "ALL"
                else [i for i, c in enumerate(cases) if c["ds"] == scope])
        sc = {}
        for name in names:
            vals = {}
            for key, fn in (("blocks", lambda i: len({(int(r) // 8) * 4 + (int(c) // 8)
                                                      for r, c in pos[G[name][i]]})),
                            ("size", lambda i: int(G[name][i].sum())),
                            ("radius", lambda i: float(np.linalg.norm(
                                pos[G[name][i]] - pos[G[name][i]].mean(axis=0),
                                axis=1).mean()))):
                v = [fn(i) for i in idxs if G[name][i].any()]
                vals[key] = (float(np.mean(v)) if v else None, len(v))
            sc[name] = vals
        out[scope] = sc
    path = os.path.join(OUT, "s2c2_spatial.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[saved] {path}")
    for scope in DS_ALL + ["ALL"]:
        print(f"\n=== {scope} (spatial) ===")
        for name in names:
            v = out[scope][name]
            if v["size"][1] == 0:
                continue
            print(f"  {name:26s} n={v['size'][1]:3d} tokens/inst={v['size'][0]:6.1f} "
                  f"distinct 8x8 blocks={v['blocks'][0]:5.2f} "
                  f"blocks/token={v['blocks'][0]/v['size'][0]:.3f} "
                  f"radius={v['radius'][0]:5.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--spatial", action="store_true")
    ap.add_argument("--rescue-plan", default=None)
    args = ap.parse_args()
    if args.extract:
        extract()
    if args.compare:
        compare(args.rescue_plan)
    if args.spatial:
        spatial_stats(args.rescue_plan)


if __name__ == "__main__":
    main()
