"""
M7 -- Candidate-Union Hedging.

The formulation
---------------
    U        = union over proxies of { that proxy's top-m dropped tokens }
    S_final  = core(256 - |U|)  UNION  U            |S_final| = 256

M6 established that the proxies' *union* holds more teacher-head tokens than any
single proxy, and that no cheap rule can rank inside it.  M7 takes that finding
at face value and stops trying: if the union cannot be ranked, keep all of it
and pay for it by shrinking the incumbent core.

Nothing here trains anything, fits a classifier, or ranks inside the union.  The
only fitted quantity is each proxy's *sign*, inherited from M6 where it was
fixed on the fit split.

Deployability contract, inherited unchanged and binding: forward-only, pre-LLM,
no gradient, no backward, no teacher at inference, no generation feedback.  The
P1-G2 teacher is an offline oracle label only, and is never used to select a
token that reaches the model.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m6_common import (BUDGET, DS_ORDER, N_VIS, SHORT, Dropped, dump_json,
                       load_bank, load_eapd_order)                     # noqa: E402

# ------------------------------------------------------------------ proxies --
# Orientation signs are M6's, fitted there on the fit split.  They are
# re-verified here against the fit split before anything is reported.
ORIENT = {"cos_s0c": -1, "red_s0_top8": -1, "imp": -1,
          "nn4_recon": +1, "loc_std": +1, "vis_norm": -1}

# Declared proxy subsets, frozen before the audit ran.  A is the single best
# proxy (the thing the union has to beat); E is the "everything" control.
SUBSETS = {
    "A-cos": ("cos_s0c",),
    "B-cos+red": ("cos_s0c", "red_s0_top8"),
    "C-cos+red+imp": ("cos_s0c", "red_s0_top8", "imp"),
    "D-cos+red+imp+nn4": ("cos_s0c", "red_s0_top8", "imp", "nn4_recon"),
    "E-all6": ("cos_s0c", "red_s0_top8", "imp", "nn4_recon", "loc_std", "vis_norm"),
}
M_GRID = (2, 4, 8, 12, 16)
HEADS = (8, 16, 32)
RANDOM_SEEDS = tuple(range(10))


class DropOnly:
    """The dropped set of each instance, with no teacher attached.

    M7 selects nothing with the teacher, so the confirmation panel -- which has
    no teacher scores -- can be run through the identical nomination and set
    construction.  Only `coverage`-style diagnostics need `head`, and those are
    simply not computed on this panel.
    """

    def __init__(self, s0):
        self.n = len(s0)
        self.drop = [np.setdiff1d(np.arange(N_VIS), np.asarray(s, dtype=np.int64))
                     for s in s0]
        for i, d in enumerate(self.drop):
            assert d.size == N_VIS - BUDGET, f"instance {i}: |D| = {d.size}"


# ===========================================================================
# nomination and union -- no ranking anywhere inside U
# ===========================================================================
def nominate(bank, dropped, i, m, name) -> set:
    """One proxy's top-m dropped tokens, as LOCAL indices into D_i."""
    sc = ORIENT[name] * bank["X"][i][dropped.drop[i], bank["fi"][name]]
    return set(np.argsort(-sc, kind="stable")[:m].tolist())


def build_union(bank, dropped, i, m, members) -> np.ndarray:
    """Set union of the members' independent top-m nominations.  Sorted for
    reproducibility only -- the order carries no score and nothing reads it."""
    u = set()
    for nm in members:
        u |= nominate(bank, dropped, i, m, nm)
    return np.array(sorted(u))


def union_all(bank, dropped, m, members):
    return [build_union(bank, dropped, i, m, members) for i in range(dropped.n)]


def cos_topn(bank, dropped, i, n_take) -> np.ndarray:
    """The strongest single proxy nominating the SAME number of tokens as U.

    This is the matched-slot control the whole stage turns on: if keeping the
    whole union is worth anything, it must beat simply asking `cos_s0c` for
    |U| tokens.
    """
    sc = ORIENT["cos_s0c"] * bank["X"][i][dropped.drop[i], bank["fi"]["cos_s0c"]]
    return np.argsort(-sc, kind="stable")[:n_take]


def eadp_next_n(order, dropped, i, n_take) -> np.ndarray:
    """The incumbent's own continuation, to the same slot count."""
    loc = {int(t): j for j, t in enumerate(dropped.drop[i])}
    out = [loc[int(t)] for t in order[i][BUDGET:BUDGET + n_take]
           if int(t) in loc]
    return np.array(out[:n_take])


def random_n(dropped, i, n_take, seed) -> np.ndarray:
    rng = np.random.default_rng(seed * 1000003 + i)
    return rng.permutation(dropped.drop[i].size)[:n_take]


def oracle_from_union(bank, dropped, i, pool, n_take) -> np.ndarray:
    """NOT deployable.  The best `n_take` tokens the union pool contains, by
    teacher rank -- the ceiling of any rule confined to that pool."""
    if pool.size <= n_take:
        return pool
    return pool[np.argsort(dropped.tr[i][pool], kind="stable")[:n_take]]


# ===========================================================================
# metrics -- per instance, then averaged
# ===========================================================================
def coverage(picks, dropped: Dropped, q: int) -> np.ndarray:
    """Per-instance |pick ∩ teacher-top-q dropped| / q."""
    return np.array([len(set(np.asarray(p).ravel().tolist()) & dropped.head[q][i]) / q
                     for i, p in enumerate(picks)])


def coverage_per_slot(picks, dropped: Dropped, q: int) -> np.ndarray:
    """Per-instance |pick ∩ head_q| / |pick| -- yield per budget slot."""
    out = []
    for i, p in enumerate(picks):
        p = np.asarray(p).ravel()
        out.append(len(set(p.tolist()) & dropped.head[q][i]) / max(p.size, 1))
    return np.array(out)


def mean_teacher_rank(picks, dropped: Dropped) -> float:
    tr = np.concatenate([dropped.tr[i][np.asarray(p).ravel()]
                         for i, p in enumerate(picks)])
    return float(tr.mean())


def size_stats(picks) -> dict:
    s = np.array([np.asarray(p).ravel().size for p in picks], dtype=float)
    return dict(mean=float(s.mean()), median=float(np.median(s)),
                p10=float(np.percentile(s, 10)), p90=float(np.percentile(s, 90)),
                min=int(s.min()), max=int(s.max()))


def pairwise_overlap(bank, dropped, m, members) -> dict:
    """How much the members' own top-m lists overlap, and how much each adds."""
    out = {}
    jac, add = [], {}
    for i in range(dropped.n):
        sets = {nm: nominate(bank, dropped, i, m, nm) for nm in members}
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                A, B = sets[members[a]], sets[members[b]]
                jac.append(len(A & B) / max(len(A | B), 1))
        for nm in members:
            others = set().union(*[sets[o] for o in members if o != nm]) \
                if len(members) > 1 else set()
            add.setdefault(nm, []).append(len(sets[nm] - others) / m)
    out["mean_pairwise_jaccard"] = float(np.mean(jac)) if jac else 1.0
    out["unique_share"] = {nm: float(np.mean(v)) for nm, v in add.items()}
    return out


def by_benchmark(arr: np.ndarray, bank: dict) -> dict:
    out = {}
    for ds in DS_ORDER:
        idx = [i for i in range(len(bank["ds"])) if bank["ds"][i] == ds]
        out[SHORT[ds]] = float(arr[idx].mean())
    out["macro"] = float(np.mean(list(out.values())))
    return out
