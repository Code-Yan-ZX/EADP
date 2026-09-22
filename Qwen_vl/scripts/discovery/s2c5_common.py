"""
S2-C5 shared definitions: the token-local vs set-dependent factorization question.

The stage's question, from S2-C4:

  S2-C4 closed the per-image *direction* route: a direction fitted on the image
  itself recovers 95 % of the teacher's Top-8, but no cheap forward descriptor of
  image j predicts which fit direction transfers to image j (NN picks land below
  the plain mean direction). So the adaptation variable is not recoverable as a
  *shared linear direction selected per image*.

  That leaves exactly one open reading of the S2-C3 wall. Either

    LOCAL   the teacher's head signal is a fixed nonlinear function of the
            token's own L4 hidden state, and the linear probe underfits it
            (-> an MLP per token closes the gap, no context needed), or

    GLOBAL  it additionally depends on the image as a whole through a
            permutation-invariant summary (-> an image-level code modulates the
            per-token function), or

    SET     it genuinely depends on which other tokens are present, i.e. on
            pairwise/set configuration (-> token-token interaction is required).

Nothing here trains a final method. Everything is a controlled read-out of
cached layer-4 hidden states plus the P1-G2 teacher ranking, holding the split,
the target and the loss family fixed at S2-C3's HEAD_RANK configuration.

--------------------------------------------------------------------------
Frozen configuration (reused from S2-C1/C2/C3/C4 -- never re-tuned here)
--------------------------------------------------------------------------
  model            Qwen3-VL-8B, vision tower + LLM frozen, never loaded
  features         cached L4 visual hidden states, 1024 x 4096, fp16 on disk
  tokens           1024 visual tokens per image (T = 256 selection budget)
  split            fit 240 / val 60 / held-out 150, image-level
  teacher          S2-B P1-G2 gradient maps (s2b_gradient_scores.npz)
  target           HEAD_RANK: positives = teacher Top-32, rank-graded weights
                   w ~ 32/(r+1) mean-normalised; loss = balanced BCE +
                   lambda * weighted all-pairs margin ranking, lambda = 0.5
  early stopping   validation Top-256 overlap (the S2-C1/C3 criterion)
  primary metric   held-out teacher Top-8 recall@256
  supplementary    Top-16/32 recall@256, exact Top-8/16/32 agreement,
                   Top-256 overlap

Top-256 overlap is *not* allowed to veto a head-recovery gain: S2-C3 measured
HEAD_RANK at overlap256 = 0.52 against BASE's 0.56 while gaining +2.1 pt of
head_recall@8, and S2-C2 established that the downstream value is paid in head
recall, not bulk overlap.

--------------------------------------------------------------------------
What must never happen
--------------------------------------------------------------------------
No arm may read the held-out image's teacher map, and no arm may read the query
(the S2-C1 QRY arm is the only query-conditioned scorer in the project, and it is
not part of this ladder). Selection among arms is by held-out head_recall@8 with
a paired bootstrap CI; the n=8 causal tier is not used for method selection
anywhere in this stage.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                        # noqa: E402
from s2c3_common import (AGREE_KS, BUDGET, DS_ALL, HEAD_KS, LAYER,  # noqa: E402,F401
                         N_VIS, RANK_K, TEACHER, aggregate, head_metrics,
                         load_plan, pos_and_weights, teacher_orders)

FEATS = f"s2c1_feats_L{LAYER}.npy"
QUERY = f"s2c1_query_L{LAYER}.npy"

# ------------------------------------------------------------- score banks ----
# Every frozen reference scorer this stage compares against, and where its cached
# score vectors live. Naming is the S2-C1/C3 arm name with its seed tag.
BANKS = {
    "LIN_L4": ("s2c1_scores.npz", "LIN_L4", (None,)),
    "LIN_L2": ("s2c1_scores.npz", "LIN_L2", (None,)),
    "QRY_L4": ("s2c1_scores.npz", "QRY_L4", (None,)),
    "BASE": ("s2c3_scores.npz", "BASE", (0, 1, 2)),
    "HEAD_BIN": ("s2c3_scores.npz", "HEAD_BIN", (0, 1, 2)),
    "HEAD_MULTI": ("s2c3_scores.npz", "HEAD_MULTI", (0, 1, 2)),
    "HEAD_RANK": ("s2c3_scores.npz", "HEAD_RANK", (0, 1, 2)),
}
FROZEN_REF = "HEAD_RANK"        # arm A of the ladder: the thing to beat
LOCAL_LIN = "LIN_L4"            # the S2-C1 token-local linear probe


def bank_tag(arm: str, seed) -> str:
    return arm if seed is None else f"{arm}_s{seed}"


def load_bank(arm: str) -> dict:
    """{seed or None: {key: score vector}} for a cached arm."""
    fname, prefix, seeds = BANKS[arm]
    Z = np.load(os.path.join(OUT, fname))
    out = {}
    for s in seeds:
        tag = bank_tag(arm, s)
        out[s] = {k.split("__", 1)[1]: Z[k].astype(np.float64)
                  for k in Z.files if k.startswith(tag + "__")}
    return out


def seed_mean_scores(per_seed: dict) -> dict:
    """Mean of the per-seed score vectors.

    Averaging the *score functions* (equivalently the directions, since every
    frozen arm is affine) is what S2-C4 called ``HEAD_RANK_seedavg``; the two
    agree to within the seed spread, which is reported separately.
    """
    seeds = list(per_seed)
    keys = list(per_seed[seeds[0]])
    return {k: np.mean([per_seed[s][k] for s in seeds], axis=0) for k in keys}


# ------------------------------------------------------------- rank helpers ----
def student_ranks(score: np.ndarray) -> np.ndarray:
    """1-based descending rank of every token under ``score`` (1 = best).

    Ties break by ascending token index, matching every ``argsort`` in the
    project (``kind='stable'``), so the ranks are the ones the selection uses.
    """
    o = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    r = np.empty(len(o), dtype=np.int64)
    r[o] = np.arange(1, len(o) + 1, dtype=np.int64)
    return r


def missed_rank_stats(ranks: np.ndarray) -> dict:
    """Rank distribution of a set of missed tokens (1-based student ranks)."""
    if len(ranks) == 0:
        return {"n": 0}
    r = np.asarray(ranks, dtype=np.float64)
    q = np.percentile(r, [5, 25, 50, 75, 90, 95])
    return {
        "n": int(len(r)),
        "mean": float(r.mean()),
        "median": float(q[2]),
        "p05": float(q[0]), "p25": float(q[1]), "p75": float(q[3]),
        "p90": float(q[4]), "p95": float(q[5]),
        "min": float(r.min()), "max": float(r.max()),
    }


def rank_buckets(ranks: np.ndarray, edges=(256, 320, 384, 512, 768, 1024)) -> dict:
    """Fraction of the given tokens with student rank <= each edge."""
    r = np.asarray(ranks, dtype=np.float64)
    if len(r) == 0:
        return {f"frac_le_{e}": float("nan") for e in edges} | {"n": 0}
    out = {f"frac_le_{e}": float((r <= e).mean()) for e in edges}
    out["n"] = int(len(r))
    return out


def excess_buckets(ranks: np.ndarray, edges=(32, 64, 128, 256, 512)) -> dict:
    """Fraction of missed tokens within ``edge`` places past the 256 frontier."""
    r = np.asarray(ranks, dtype=np.float64)
    if len(r) == 0:
        return {f"frac_within_{e}": float("nan") for e in edges} | {"n": 0}
    ex = r - BUDGET
    return {f"frac_within_{e}": float((ex <= e).mean()) for e in edges}


def recall_curve(score: np.ndarray, torder: np.ndarray,
                 ks=(8, 16, 32), cs=(256, 288, 320, 384, 448, 512, 640, 768, 896, 1024)) -> dict:
    """Teacher-Top-k recall if the candidate pool were enlarged to C tokens.

    This is the *oracle coverage* of a frontier-only contextual scorer: a scorer
    that is allowed to re-rank a pool of the student's top C tokens can never
    exceed recall@C, whatever it learns inside the pool.
    """
    sel = set(np.argsort(-score, kind="stable")[:max(cs)].tolist())
    out = {}
    for k in ks:
        head = torder[:k]
        for c in cs:
            pool = set(np.argsort(-score, kind="stable")[:c].tolist())
            out[f"recall{k}@{c}"] = float(np.isin(head, list(pool)).mean())
    return out


def image_ranks(score: np.ndarray, torder: np.ndarray, k: int):
    """(missed mask over the teacher's Top-k, student ranks of those tokens)."""
    head = torder[:k]
    ranks = student_ranks(score)
    hr = ranks[head]
    missed = hr > BUDGET
    return head, ranks, hr, missed


def summarise(per_image: list[dict]) -> dict:
    """Mean of every numeric field across images (distribution-free)."""
    if not per_image:
        return {}
    keys = [k for k in per_image[0] if isinstance(per_image[0][k], (int, float))
            and not isinstance(per_image[0][k], bool)]
    return {k: float(np.mean([r[k] for r in per_image])) for k in keys}


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def dump(name: str, obj: dict):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(jsonable(obj), f, indent=1)
    print(f"[saved] {name}")
    return path


# -------------------------------------------------------------- test rows ----
def test_rows():
    """The held-out 150: rows, keys, benchmark label, teacher order."""
    meta, plan, keys, rows_of = load_plan()
    G = np.load(os.path.join(OUT, TEACHER))
    orders = teacher_orders(G, keys)
    rows = rows_of["test"]
    return (rows, [keys[i] for i in rows], [meta["plan"][i]["ds"] for i in rows],
            orders, meta, plan, keys, rows_of)
