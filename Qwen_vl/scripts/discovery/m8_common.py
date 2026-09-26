"""
M8 -- Targeted Zeroth-Order Audit.

The question this stage exists to answer, and the only one:

    Can a zeroth-order sensitivity estimator pull the tokens that actually
    carry the answer -- the P1-G2 gradient teacher's head -- from the depth the
    cheap proxies leave them at (mean teacher rank ~131) to a depth where a
    rescue converts (mean teacher rank < 40)?

Five earlier rounds closed the *learned* and the *cheap-forward* forms of this
question (S2-C0, S2-C2, S2-C5, M3-v0, M3-v2, M6, M7).  Every one of them ranked
tokens with a quantity the incumbent's pruner already produces.  This stage is
the first to *measure the model* rather than read a feature off it: it borrows
ZOO-Prune's (CVPR 2026) zeroth-order estimator and asks whether spending a real
measurement on a small nominated pool beats reading 27 cheap columns.

Two estimators, deliberately distinguished
------------------------------------------
ZO-P  the ZOO-Prune estimator, reproduced as published: perturb the
      pre-projector vision features by +-h*u with u a shared unit Gaussian
      bank, run the MM-projector, score each token by the mean L2 norm of the
      central difference of its projected embedding.  Query-free, LLM-free,
      cheap.  This is the *global* estimator.
ZO-L  the same idea one level up: perturb the visual input embedding of a
      single token, run the LLM, score it by the change in the P1-G2 objective.
      This is the estimator the M8 hypothesis needs -- it is the forward
      analogue of the gradient teacher -- and it is the *targeted* one, since
      it costs one full forward per token and can only be afforded on a pool.

The distinction is the whole stage.  ZO-P is cheap because the projector is
cheap; ZO-L is the estimator that could in principle break the depth wall, and
its affordability is the hypothesis under test.

Deployability contract, inherited unchanged and binding: forward-only, pre-LLM
for ZO-P, no gradient, no backward, no teacher at inference, no generation
feedback.  The P1-G2 teacher is an offline oracle label only, used to *grade*
an estimator and never to select a token that reaches the model.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m6_common import (BUDGET, DS_ORDER, N_VIS, SHORT, Dropped, dump_json,  # noqa: E402
                       load_bank, load_eapd_order, paired_bootstrap)

# ------------------------------------------------------------------ proxies --
# M6's orientations, fitted there on the fit split.  Carried unchanged so a
# cheap baseline here is the same arm it was there.
ORIENT = {"cos_s0c": -1, "red_s0_top8": -1, "imp": -1,
          "nn4_recon": +1, "loc_std": +1, "vis_norm": -1}

# The M6 "strong proxy" set the brief names for the union pools P3/P4.
UNION_MEMBERS = ("cos_s0c", "red_s0_top8", "imp", "nn4_recon")

# ------------------------------------------------------------ frozen knobs --
# Fixed before any Phase-1 number was read.
ZO_DIRECTIONS = (1, 2, 4, 8)      # the probe budget the brief asks for
ZO_H = 0.01                       # ZOO-Prune's own step size
RS = (8, 16, 32)                  # rescue sizes
HEADS = (8, 16, 32)               # teacher-head depths, matched to r
RANDOM_SEEDS = tuple(range(20))
BOOT = 2000
BOOT_SEED = 20260927

POOLS = ("P0", "P1", "P2", "P3", "P4")
POOL_DOC = {
    "P0": "all 768 dropped tokens -- the global upper bound",
    "P1": "cos_s0c top-32 dropped",
    "P2": "cos_s0c top-64 dropped",
    "P3": "union of the 4 strong proxies' own top-8",
    "P4": "union of the 4 strong proxies' own top-16",
}


# ===========================================================================
# candidate pools -- small and enriched, never searched over
# ===========================================================================
def topk_local(bank, dropped, i, name, k) -> np.ndarray:
    """One proxy's top-k dropped tokens, as LOCAL indices into D_i."""
    sc = ORIENT[name] * bank["X"][i][dropped.drop[i], bank["fi"][name]]
    return np.argsort(-sc, kind="stable")[:k]


def build_pool(bank, dropped, i, name) -> np.ndarray:
    """The pool as LOCAL indices into D_i.  Sorted for reproducibility only."""
    if name == "P0":
        return np.arange(dropped.drop[i].size)
    if name == "P1":
        return np.sort(topk_local(bank, dropped, i, "cos_s0c", 32))
    if name == "P2":
        return np.sort(topk_local(bank, dropped, i, "cos_s0c", 64))
    if name in ("P3", "P4"):
        m = 8 if name == "P3" else 16
        u = set()
        for nm in UNION_MEMBERS:
            u |= set(topk_local(bank, dropped, i, nm, m).tolist())
        return np.array(sorted(u))
    raise KeyError(name)


def pools_all(bank, dropped, names=POOLS) -> dict:
    return {nm: [build_pool(bank, dropped, i, nm) for i in range(dropped.n)]
            for nm in names}


# ===========================================================================
# rankings *inside* a pool -- every rule is confined to the pool
# ===========================================================================
def rank_by_score(pool, score) -> np.ndarray:
    """Pool members ordered by descending score, ties by index."""
    return np.asarray(pool)[np.argsort(-np.asarray(score, dtype=np.float64),
                                       kind="stable")]


def r_random_in_pool(pool, r, seed, i) -> np.ndarray:
    rng = np.random.default_rng(seed * 1000003 + i)
    return np.asarray(pool)[rng.permutation(len(pool))[:r]]


def r_cheap_in_pool(bank, dropped, i, pool, name, r) -> np.ndarray:
    """A cheap proxy, restricted to the pool -- the 'read a column' control."""
    sc = ORIENT[name] * bank["X"][i][dropped.drop[i][np.asarray(pool)],
                                    bank["fi"][name]]
    return rank_by_score(pool, sc)[:r]


def r_oracle_in_pool(dropped, i, pool, r) -> np.ndarray:
    """NOT deployable.  The best r pool members by teacher rank -- the ceiling
    of ANY rule confined to this pool.  This is what a perfect auditor inside
    the pool would deliver, and it is the number the depth gate is read on."""
    pool = np.asarray(pool)
    if pool.size <= r:
        return pool
    return pool[np.argsort(dropped.tr[i][pool], kind="stable")[:r]]


# ===========================================================================
# metrics -- per instance, then averaged, exactly as M6/M7
# ===========================================================================
def score_pool(dropped: Dropped, i: int, pick) -> dict:
    pick = np.asarray(pick).ravel()
    tr = dropped.tr[i][pick]
    return dict(mean_tr=float(tr.mean()), med_tr=float(np.median(tr)),
                head={q: float(dropped.head_hits(i, pick, q) / q) for q in HEADS})


def evaluate(picks, dropped: Dropped, r: int) -> dict:
    """Per-instance recall against head_r, plus the depth statistics.

    `recall@r` is |pick ∩ teacher-top-r dropped| / r, per instance then averaged
    -- M3-v0's `overlap@r`, so the numbers are directly comparable to the
    stages that measured the depth wall.
    """
    n = len(picks)
    rec = np.zeros(n)
    mtr = np.zeros(n)
    dtr = np.zeros(n)
    hits = {q: np.zeros(n) for q in HEADS}
    for i, p in enumerate(picks):
        p = np.asarray(p).ravel()
        h = dropped.head_hits(i, p, r)
        rec[i] = h / r
        tr = dropped.tr[i][p]
        mtr[i] = tr.mean()
        dtr[i] = np.median(tr)
        for q in HEADS:
            hits[q][i] = dropped.head_hits(i, p, q) / q
    return dict(recall=rec, mean_tr=mtr, med_tr=dtr, hits=hits,
                mean_teacher_rank=float(mtr.mean()),
                median_teacher_rank=float(np.median(dtr)),
                recall_mean=float(rec.mean()),
                top8=float(hits[8].mean()), top16=float(hits[16].mean()),
                top32=float(hits[32].mean()))


def by_benchmark(arr: np.ndarray, bank: dict) -> dict:
    out = {}
    for ds in DS_ORDER:
        idx = [i for i in range(len(bank["ds"])) if bank["ds"][i] == ds]
        out[SHORT[ds]] = float(np.asarray(arr)[idx].mean())
    out["macro"] = float(np.mean(list(out.values())))
    return out


def pool_stats(pools, dropped: Dropped, q=8) -> dict:
    """Pool size, teacher enrichment, and the depth of what the pool holds."""
    sizes = np.array([len(p) for p in pools], dtype=float)
    cov = np.array([len(set(p.tolist()) & dropped.head[q][i]) / q
                    for i, p in enumerate(pools)])
    yield_ = np.array([len(set(p.tolist()) & dropped.head[q][i]) / max(len(p), 1)
                       for i, p in enumerate(pools)])
    tr = np.concatenate([dropped.tr[i][p] for i, p in enumerate(pools)])
    return dict(mean_size=float(sizes.mean()),
                p10=float(np.percentile(sizes, 10)),
                p90=float(np.percentile(sizes, 90)),
                cov8=float(cov.mean()), yield8=float(yield_.mean()),
                chance=float(sizes.mean() / N_VIS),
                mean_teacher_rank=float(tr.mean()),
                median_teacher_rank=float(np.median(tr)))
