"""
M6 -- Disagreement-Rescue Composition.

The question this stage exists to answer, and the only one:

    If a token B2 dropped ranks low in EADP importance but high under some
    *other* cheap forward-only proxy, does that cross-proxy disagreement
    enrich the set of tokens the gradient teacher would have kept?

Four earlier rounds closed the *learned* form of this question.  S2-C2 found no
cheap token property separates the value-carrying teacher-only tokens from
ordinary ones; M3-v0 and M3-v2 found the teacher's head is not recoverable by a
student at all, and that a rescue at mean teacher rank 74-132 is worth nothing
downstream.  This stage does not train anything.  It asks whether the
*disagreement* between proxies -- a relational signal no earlier stage tested --
carries what no single proxy and no student did.

Nothing here trains a model, fits a classifier or searches a hyper-parameter.
The only fitted quantity is the *sign* of each proxy's orientation, fixed on the
fit split and never on the split it is reported on.

The gate, frozen before the numbers were read (brief Phase 1):

    If the disagreement rule does not beat RANDOM, EADP-NEXT and the best
    single proxy on critical recall, the line stops here.

Deployability contract, inherited unchanged from M3/M4/M5 and binding:
forward-only, pre-LLM, no gradient, no backward, no teacher at inference, no
generation feedback.  The teacher is an offline oracle label only.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUT_DIR                                       # noqa: E402

# ------------------------------------------------------------------ geometry --
N_VIS = 1024
BUDGET = 256
GRID = 32
DS_ORDER = ("TextVQA_VAL", "DocVQA_VAL", "OCRBench")
SHORT = {"TextVQA_VAL": "TextVQA", "DocVQA_VAL": "DocVQA", "OCRBench": "OCRBench"}

M5_BANK = "m5_bank.npz"
EAPD_ORDER = "m6_eapd_order.npz"

# ------------------------------------------------------------ frozen knobs --
# Fixed before any number in this stage was read.
KS = (8, 16, 32, 64)              # rescue pool sizes
QS = (8, 16, 32, 64)              # teacher-head depths, matched to k
FIT_SPLIT = "fit"                 # the ONLY split any orientation is fitted on
HOLDOUT = ("val", "test")         # what every headline number is reported on
RANDOM_SEEDS = tuple(range(20))   # the random control's spread, not one draw
D4_Q = 0.10                       # D4's "top-q%" threshold
BOOT = 2000                       # paired bootstrap resamples
BOOT_SEED = 20260926

# --------------------------------------------------------------- proxies ----
# The five deployable proxies this stage declares, chosen to span distinct
# mechanisms rather than to span the feature list.  `imp` is the incumbent's own
# score and is carried separately as EADP's own view; the other five are the
# auxiliary set the disagreement rules read.
#
#   imp          EADP importance, post-smoothing ^beta -- the B2 score itself
#   cos_s0c      cosine to the retained-set centroid -- representational novelty
#   nn4_recon    reconstruction residual from the 4 nearest retained tokens --
#                genuine approximation error
#   red_s0_top8  mean of the 8 largest cosines to other retained tokens --
#                redundancy inside S0
#   loc_std      spread of the EADP local match across instruction tokens
#   vis_norm     ||v_i|| -- scale, carried as the magnitude control
PRIMARY = ("imp", "cos_s0c", "nn4_recon", "red_s0_top8", "loc_std", "vis_norm")
INCUMBENT = "imp"
AUX = ("cos_s0c", "nn4_recon", "red_s0_top8", "loc_std", "vis_norm")
CONTROL = "vis_norm"

# ===========================================================================
# bank
# ===========================================================================


def load_bank(path: str = M5_BANK) -> dict:
    """The M5 bank: 450 instances x 1024 tokens x 27 cheap forward-only columns.

    `X[i, j, c]` is column `c` of `FEATURES` for token `j` of instance `i`;
    `s0[i]` is B2's retained set (sorted, 256); `g2[i]` is the frozen P1-G2
    gradient teacher (bit-identical to `s2b_gradient_scores.npz`, verified).
    """
    z = np.load(os.path.join(OUTPUT_DIR, path), allow_pickle=False)
    out = {k: z[k] for k in ("key", "ds", "idx", "split", "s0", "g2", "X",
                             "feature_names")}
    out["feature_names"] = [str(s) for s in out["feature_names"]]
    out["key"] = [str(s) for s in out["key"]]
    out["ds"] = [str(s) for s in out["ds"]]
    out["split"] = [str(s) for s in out["split"]]
    out["fi"] = {n: i for i, n in enumerate(out["feature_names"])}
    out["n"] = len(out["key"])
    return out


def load_eapd_order(bank: dict) -> np.ndarray:
    """The EADP selector's full greedy order, (n, 1024), positions 0..255 == S0.

    Produced by `m6_eapd_order.py` on the real engine.  Because the facility-
    location greedy's step t depends only on picks 0..t-1, the first 256 picks of
    a 1024-step run are exactly the 256 picks of the 256-step run the incumbent
    performs -- so this is the incumbent's own continuation, not an
    approximation of it.  The producer asserts both directions of that.
    """
    path = os.path.join(OUTPUT_DIR, EAPD_ORDER)
    z = np.load(path, allow_pickle=False)
    order, keys = z["order"], [str(s) for s in z["key"]]
    assert keys == bank["key"], "EADP order is not row-aligned with the bank"
    return order


# ===========================================================================
# the dropped set, and the teacher's ranking inside it
# ===========================================================================
class Dropped:
    """Per-instance view of the 768 tokens B2 dropped, and the teacher on them.

    `tr[i]` is the teacher rank *within the dropped set* of each dropped token,
    0 = the teacher's best dropped token.  Every metric in this stage is
    computed per instance and then averaged; nothing is ever pooled into a
    global top-k across instances.
    """

    def __init__(self, bank: dict):
        n = bank["n"]
        self.n = n
        self.drop = []
        self.tr = []
        self.head = {q: [] for q in QS}
        for i in range(n):
            d = np.setdiff1d(np.arange(N_VIS), bank["s0"][i])
            assert d.size == N_VIS - BUDGET, f"instance {i}: |D| = {d.size}"
            t = bank["g2"][i][d]
            # stable: ties resolve by index, identically for every rule
            r = np.argsort(np.argsort(-t, kind="stable"), kind="stable")
            self.drop.append(d)
            self.tr.append(r)
            for q in QS:
                self.head[q].append(set(np.where(r < q)[0].tolist()))

    def head_hits(self, i: int, local_idx, q: int) -> int:
        return len(set(np.asarray(local_idx).ravel().tolist()) & self.head[q][i])


# ===========================================================================
# orientation -- the ONLY fitted quantity, and it is fitted on `fit` alone
# ===========================================================================
def fit_orientations(bank: dict, dropped: Dropped, proxies=PRIMARY,
                     q: int = 16, k: int = 16) -> dict:
    """Sign of each proxy, chosen on the fit split by recall@k against head_q.

    A single binary choice per proxy.  Reported for every proxy so the choice is
    auditable, and never re-fitted on val or test.
    """
    idx = [i for i in range(bank["n"]) if bank["split"][i] == FIT_SPLIT]
    out = {}
    for nm in proxies:
        c = bank["fi"][nm]
        best, best_sgn = -1.0, 1
        for sgn in (1, -1):
            hit = 0
            for i in idx:
                sc = sgn * bank["X"][i][dropped.drop[i], c]
                o = np.argsort(-sc, kind="stable")[:k]
                hit += dropped.head_hits(i, o, q)
            r = hit / (len(idx) * q)
            if r > best:
                best, best_sgn = r, sgn
        out[nm] = dict(sign=int(best_sgn), fit_recall=float(best))
    return out


def oriented(bank: dict, dropped: Dropped, i: int, nm: str, orient: dict):
    """The instance's dropped-token scores for `nm`, oriented so high = critical."""
    return orient[nm]["sign"] * bank["X"][i][dropped.drop[i], bank["fi"][nm]]


def pctile(v: np.ndarray) -> np.ndarray:
    """Percentile in [0, 1] within the instance's dropped set; 1 = highest."""
    if v.size < 2:
        return np.zeros_like(v, dtype=np.float64)
    return np.argsort(np.argsort(v, kind="stable"), kind="stable") / (v.size - 1.0)


# ===========================================================================
# the rules -- every one returns exactly k LOCAL indices into drop[i]
# ===========================================================================
def _topk(sc: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-np.asarray(sc, dtype=np.float64), kind="stable")[:k]


def r_random(dropped, i, k, seed):
    rng = np.random.default_rng(seed * 1000003 + i)
    return rng.permutation(dropped.drop[i].size)[:k]


def union_pool(bank, dropped, orient, i, k, members):
    """The union of what the members' own top-k lists offer, as local indices."""
    pool = set()
    for nm in members:
        pool |= set(_topk(oriented(bank, dropped, i, nm, orient), k).tolist())
    return np.array(sorted(pool))


def r_random_in_pool(bank, dropped, orient, i, k, members, seed):
    """Uniform draw from the union pool -- the control that separates 'the pool
    is enriched' from 'the disagreement score ranks the pool well'."""
    pool = union_pool(bank, dropped, orient, i, k, members)
    rng = np.random.default_rng(seed * 7919 + i)
    return pool[rng.permutation(pool.size)[:k]]


def eadp_next_local(eapd_order, dropped, i, k):
    """Greedy positions 256..256+k-1, mapped to local dropped-set indices."""
    loc = {int(t): j for j, t in enumerate(dropped.drop[i])}
    out = [loc[int(t)] for t in eapd_order[i][BUDGET:BUDGET + k] if int(t) in loc]
    assert len(out) == k, f"instance {i}: EADP-next short ({len(out)})"
    return np.array(out)


def r_single(bank, dropped, orient, i, nm, k):
    return _topk(oriented(bank, dropped, i, nm, orient), k)


def r_union(bank, dropped, orient, i, k, members, cap=None):
    """Round-robin over `members`' own top lists, deduped, filled to exactly k."""
    per = cap if cap is not None else max(1, int(np.ceil(k / len(members))))
    out, seen = [], set()
    for nm in members:
        for j in _topk(oriented(bank, dropped, i, nm, orient), per):
            j = int(j)
            if j not in seen:
                seen.add(j)
                out.append(j)
    # fill from the pooled oriented-percentile max if round-robin came up short
    if len(out) < k:
        P = np.stack([pctile(oriented(bank, dropped, i, nm, orient))
                      for nm in members])
        for j in np.argsort(-P.max(0), kind="stable"):
            if int(j) not in seen:
                seen.add(int(j))
                out.append(int(j))
            if len(out) == k:
                break
    return np.array(out[:k])


def r_maxfusion(bank, dropped, orient, i, k, members):
    P = np.stack([pctile(oriented(bank, dropped, i, nm, orient))
                  for nm in members])
    return _topk(P.max(0), k)


def r_minfusion(bank, dropped, orient, i, k, members):
    """Agreement: tokens every member rates highly."""
    P = np.stack([pctile(oriented(bank, dropped, i, nm, orient))
                  for nm in members])
    return _topk(P.min(0), k)


def r_disagree(bank, dropped, orient, i, k, kind, members=AUX, q=D4_Q):
    """The disagreement family.  `eadp_pct` is the RAW EADP percentile, so a
    high score means 'EADP rates it low, some auxiliary rates it high'."""
    eadp_pct = pctile(bank["X"][i][dropped.drop[i], bank["fi"][INCUMBENT]])
    P = np.stack([pctile(oriented(bank, dropped, i, nm, orient))
                  for nm in members])
    if kind == "D1":                       # max aux percentile - EADP percentile
        sc = P.max(0) - eadp_pct
    elif kind == "D2":                     # EADP rank - best aux rank (raw ranks)
        rd = dropped.drop[i].size
        aux_rank = np.stack([np.argsort(np.argsort(
            -oriented(bank, dropped, i, nm, orient), kind="stable"),
            kind="stable") for nm in members]).min(0) / (rd - 1.0)
        eadp_rank = np.argsort(np.argsort(
            -bank["X"][i][dropped.drop[i], bank["fi"][INCUMBENT]],
            kind="stable"), kind="stable") / (rd - 1.0)
        sc = eadp_rank - aux_rank
    elif kind == "D3":                     # rank range across ALL proxies
        # all members in the SAME orientation (EADP flipped, since raw EADP
        # importance points the other way), so the range measures how much the
        # proxies disagree about a token rather than an orientation artefact
        Pall = np.concatenate([P, (1.0 - eadp_pct)[None, :]], axis=0)
        sc = Pall.max(0) - Pall.min(0)
    elif kind == "D4":                     # binary: some aux in top-q%, EADP not
        hi = (P >= 1.0 - q).any(0)
        lo = eadp_pct < 1.0 - q
        sc = (hi & lo).astype(np.float64) * 2.0 + P.max(0)
    else:
        raise ValueError(kind)
    return _topk(sc, k)


def r_oracle_union(bank, dropped, orient, i, k, members):
    """NOT deployable.  The ceiling of ANY rule confined to the union of the
    members' own top-k lists.

    The pool is the union of what the members already offer; the pick is the k
    pool members with the best teacher rank.  That is the most a perfect
    combiner could extract without looking outside the proxies' own candidates,
    so it bounds the whole R3/R4/R5 family.  Comparing it to the best single
    proxy is the sharpest statement of whether combination has anything to add.
    """
    pool = set()
    for nm in members:
        pool |= set(_topk(oriented(bank, dropped, i, nm, orient), k).tolist())
    pool = np.array(sorted(pool))
    if pool.size <= k:
        return pool
    order = np.argsort(dropped.tr[i][pool], kind="stable")[:k]
    return pool[order]


# ===========================================================================
# metrics
# ===========================================================================
def per_instance(picks, dropped: Dropped, q: int) -> dict:
    """Per-instance arrays, never pooled across instances.

    `recall[i]` is instance i's own |pick ∩ teacher-top-q| / q, so every average
    below is a mean of per-instance recalls and never a pooled global top-k.
    """
    n = dropped.n
    rec = np.zeros(n)
    prec = np.zeros(n)
    trs = []
    for i in range(n):
        p = np.asarray(picks[i]).ravel()
        assert p.size > 0, f"instance {i}: empty pick"
        h = dropped.head_hits(i, p, q)
        rec[i] = h / q
        prec[i] = h / p.size
        trs.append(dropped.tr[i][p])
    return dict(recall=rec, precision=prec, teacher_rank=trs,
                picked=np.array([np.asarray(p).ravel().size for p in picks]))


def aggregate(pi: dict, idx=None) -> dict:
    """Summarise `per_instance` output over an instance subset."""
    if idx is None:
        idx = np.arange(pi["recall"].size)
    idx = np.asarray(idx)
    tr = np.concatenate([pi["teacher_rank"][i] for i in idx])
    rec = pi["recall"][idx]
    chance = float(pi["picked"][idx].mean()) / 768.0
    return dict(recall=float(rec.mean()),
                mean_teacher_rank=float(tr.mean()),
                median_teacher_rank=float(np.median(tr)),
                precision_in_head=float(pi["precision"][idx].mean()),
                enrichment=float(rec.mean() / chance),
                chance=chance, n=int(idx.size))


def by_benchmark(pi: dict, bank: dict) -> dict:
    out = {}
    for ds in DS_ORDER:
        idx = [i for i in range(len(bank["ds"])) if bank["ds"][i] == ds]
        out[SHORT[ds]] = float(pi["recall"][idx].mean())
    out["macro"] = float(np.mean(list(out.values())))
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, seed=BOOT_SEED, n=BOOT):
    """Paired bootstrap CI on mean(a - b) over instances."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n, d.size))
    means = d[idx].mean(axis=1)
    return dict(delta=float(d.mean()),
                lo=float(np.percentile(means, 2.5)),
                hi=float(np.percentile(means, 97.5)),
                p_two_sided=float(2 * min((means <= 0).mean(), (means >= 0).mean())),
                n_win=int((d > 0).sum()), n_loss=int((d < 0).sum()),
                n_tie=int((d == 0).sum()))


def dump_json(name: str, obj) -> str:
    """Atomic: a crash mid-write must not destroy a completed grid."""
    path = os.path.join(OUTPUT_DIR, name)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=_default)
    os.replace(tmp, path)
    return path


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not serialisable: {type(o)}")
