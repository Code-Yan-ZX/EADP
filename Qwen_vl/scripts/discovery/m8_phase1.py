"""
M8 step 3 -- Phase 1, the depth-wall diagnostic.  No generation is run here.

The gate, frozen before any number in this stage was read (brief Phase 1):

    STRONG  r=8 mean teacher rank < 40 (ideally < 25), clearly better than
            cos_s0c and than the best M6 cheap scorer, same direction on
            several pools.
    STOP    mean rank still > 70-100, or head recall improves while depth does
            not.  A head-recall gain with no depth gain is not a rescue.

Every rule is confined to its pool: the pool is the candidate set a nomination
step would deliver, and the rule under test only *ranks inside it*.  That is the
question M6 left open -- "the union is enriched but no cheap score ranks inside
it" -- and the one ZO is being asked to answer.

Reported for every (pool x r): head recall, mean and median teacher rank, and
the same quantities for the matched random draw, the best single cheap proxy,
the M6 composition rules, ZO-P at four direction budgets, and the oracle.

Usage
    python scripts/discovery/m8_phase1.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUT_DIR                                       # noqa: E402
from m6_common import (AUX, D4_Q, Dropped, dump_json, load_bank,    # noqa: E402
                       load_eapd_order, pctile, paired_bootstrap)
from m8_common import (BOOT_SEED, HEADS, ORIENT, POOLS, POOL_DOC,   # noqa: E402
                       RANDOM_SEEDS, RS, ZO_DIRECTIONS, build_pool, evaluate,
                       pool_stats, rank_by_score, r_cheap_in_pool,
                       r_oracle_in_pool, r_random_in_pool)

TAG = "m8_phase1"
HOLDOUT = ("val", "test")     # every headline number is on the held-out 210
CHEAP = ("cos_s0c", "imp", "nn4_recon", "red_s0_top8", "loc_std", "vis_norm")


class Sub:
    """`Dropped`, restricted to a row subset and re-indexed positionally.

    `Dropped` is indexed by bank row; the analysis below indexes pools by
    position within the held-out panel.  Re-indexing here rather than passing
    two index spaces around is what keeps a positional lookup from silently
    reading the wrong instance.
    """

    def __init__(self, dropped: Dropped, rows):
        self.rows = list(rows)
        self.n = len(self.rows)
        self.drop = [dropped.drop[i] for i in self.rows]
        self.tr = [dropped.tr[i] for i in self.rows]
        self.head = {q: [dropped.head[q][i] for i in self.rows]
                     for q in dropped.head}

    def head_hits(self, j, local_idx, q):
        return len(set(np.asarray(local_idx).ravel().tolist()) & self.head[q][j])


def m6_score_in_pool(bank, dropped, i, pool, kind):
    """M6's own composition rules, restricted to the pool.

    The percentiles and the D4 thresholds are M6's, computed over the whole
    dropped set exactly as M6 computed them; only the *selection* is confined
    to the pool, which is what makes this a matched comparison rather than a
    different rule.
    """
    d = dropped.drop[i]
    P = np.stack([pctile(ORIENT[nm] * bank["X"][i][d, bank["fi"][nm]])
                  for nm in AUX])
    eadp_pct = pctile(bank["X"][i][d, bank["fi"]["imp"]])
    if kind == "D4":
        sc = ((P >= 1.0 - D4_Q).any(0)
              & (eadp_pct < 1.0 - D4_Q)).astype(float) * 2.0 + P.max(0)
    elif kind == "maxfusion":
        sc = P.max(0)
    else:
        raise KeyError(kind)
    return sc[np.asarray(pool)]


def main():
    bank = load_bank()
    dropped = Dropped(bank)
    order = load_eapd_order(bank)
    z = np.load(os.path.join(OUTPUT_DIR, "m8_zop.npz"), allow_pickle=False)
    assert [str(k) for k in z["key"]] == bank["key"], "ZO-P rows not aligned"
    ZOP = {m: z[f"S_m{m}"] for m in ZO_DIRECTIONS}
    hold = [i for i in range(bank["n"]) if bank["split"][i] in HOLDOUT]
    sub = Sub(dropped, hold)
    print(f"[phase1] {len(hold)} held-out instances; pools {POOLS}")

    pools = {nm: [build_pool(bank, dropped, i, nm) for i in hold] for nm in POOLS}
    out = dict(n_holdout=len(hold), pools={}, doc=POOL_DOC,
               random_seeds=list(RANDOM_SEEDS), boot_seed=BOOT_SEED)

    # ---------------------------------------------------------- pool audit --
    out["pool_audit"] = {}
    for nm in POOLS:
        st = pool_stats(pools[nm], sub, q=8)
        st["doc"] = POOL_DOC[nm]
        out["pool_audit"][nm] = st
        print(f"  {nm}: |pool|={st['mean_size']:.1f}  cov@8={st['cov8']:.4f} "
              f"yield@8={st['yield8']:.4f} (chance {st['chance']:.4f})  "
              f"meanTR={st['mean_teacher_rank']:.1f}")

    # ------------------------------------------------------- the rule grid --
    def picks_for(rule, pl, r):
        """One rule's pick on every held-out instance, for pool list `pl`."""
        if rule == "random":
            raise ValueError("random is handled by its own seed loop")
        if rule == "oracle":
            return [r_oracle_in_pool(sub, j, pl[j], r) for j in range(sub.n)]
        if rule in CHEAP:
            return [r_cheap_in_pool(bank, sub, j, pl[j], rule, r)
                    for j in range(sub.n)]
        if rule in ("D4", "maxfusion"):
            return [rank_by_score(pl[j],
                                  m6_score_in_pool(bank, sub, j, pl[j], rule))[:r]
                    for j in range(sub.n)]
        if rule.startswith("zop"):
            m = int(rule[3:])
            return [rank_by_score(pl[j],
                                  ZOP[m][hold[j]][sub.drop[j][np.asarray(pl[j])]])[:r]
                    for j in range(sub.n)]
        raise KeyError(rule)

    rules = (["cos_s0c", "imp", "nn4_recon", "red_s0_top8", "D4", "maxfusion",
              "oracle"] + [f"zop{m}" for m in ZO_DIRECTIONS])
    grid, recs = {}, {}
    for pname in POOLS:
        pl = pools[pname]
        for r in RS:
            cell = {}
            rnd = np.stack([evaluate([r_random_in_pool(pl[j], r, s, j)
                                      for j in range(sub.n)], sub, r)["recall"]
                            for s in RANDOM_SEEDS])
            cell["random"] = dict(recall_mean=float(rnd.mean()),
                                  recall_seed_sd=float(rnd.mean(axis=1).std()),
                                  mean_teacher_rank=None,
                                  median_teacher_rank=None)
            recs[f"{pname}|r{r}|random"] = rnd.mean(axis=0)
            for rule in rules:
                ev = evaluate(picks_for(rule, pl, r), sub, r)
                cell[rule] = dict(recall_mean=ev["recall_mean"],
                                  mean_teacher_rank=ev["mean_teacher_rank"],
                                  median_teacher_rank=ev["median_teacher_rank"],
                                  top8=ev["top8"], top16=ev["top16"],
                                  top32=ev["top32"])
                recs[f"{pname}|r{r}|{rule}"] = ev["recall"]
            grid[f"{pname}|r{r}"] = cell
            print(f"  {pname} r={r:2d}: " + "  ".join(
                f"{k}={cell[k]['recall_mean']:.3f}"
                + (f"/{cell[k]['mean_teacher_rank']:.0f}"
                   if cell[k]['mean_teacher_rank'] is not None else "")
                for k in ("random", "cos_s0c", "D4", "zop8", "oracle")))
    out["grid"] = grid

    # ------------------------------------------------- paired significances --
    sig = {}
    for pname in POOLS:
        for r in RS:
            base = recs[f"{pname}|r{r}|cos_s0c"]
            rnd = recs[f"{pname}|r{r}|random"]
            for k in [f"zop{m}" for m in ZO_DIRECTIONS] + ["D4", "maxfusion",
                                                           "oracle"]:
                sig[f"{pname}|r{r}|{k}-vs-cos"] = paired_bootstrap(
                    recs[f"{pname}|r{r}|{k}"], base, seed=BOOT_SEED)
            for m in ZO_DIRECTIONS:
                sig[f"{pname}|r{r}|zop{m}-vs-random"] = paired_bootstrap(
                    recs[f"{pname}|r{r}|zop{m}"], rnd, seed=BOOT_SEED)
    out["significance"] = sig

    # ----------------------------------------------- ZO-P as a global rule --
    # Would the estimator, applied to all 1024 tokens with the incumbent's own
    # budget, pick the teacher's head?  This is the *global ZOO-Prune* arm.
    glob = {}
    for m in ZO_DIRECTIONS:
        pick = [np.argsort(-ZOP[m][i], kind="stable")[:256] for i in hold]
        glob[f"zop{m}"] = {f"kept_head{q}": float(np.mean(
            [len(set(p.tolist()) & sub.head[q][j]) / q
             for j, p in enumerate(pick)])) for q in HEADS}
    glob["B2_s0"] = {f"kept_head{q}": float(np.mean(
        [len(set(bank["s0"][i].tolist()) & sub.head[q][j]) / q
         for j, i in enumerate(hold)])) for q in HEADS}
    glob["oracle_256"] = {f"kept_head{q}": float(np.mean(
        [len(sub.head[q][j] & set(
            np.argsort(sub.tr[j], kind="stable")[:256].tolist())) / q
         for j in range(sub.n)])) for q in HEADS}
    out["global_kept"] = glob
    print("\n[global] fraction of the teacher's head kept by a 256-token set "
          "chosen from all 1024:")
    for k, v in glob.items():
        print(f"  {k:10s} " + "  ".join(f"{kk}={vv:.4f}" for kk, vv in v.items()))

    dump_json(f"{TAG}.json", out)
    print(f"\n[saved] {TAG}.json")


if __name__ == "__main__":
    main()
