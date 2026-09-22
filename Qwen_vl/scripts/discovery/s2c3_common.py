"""
S2-C3 shared definitions: head-oriented supervision targets and head metrics.

The stage's question is whether the S2-C1 *objective* was wrong (the student was
trained on the bulk of the teacher's Top-256 instead of the head of its ranking)
or whether the *scorer form* is the wall (a token-local linear read-out of L4
hidden states cannot express the head even when asked to).

So everything that is not the target is held fixed at the S2-C1 configuration:
same feature cache (L4, 1024x4096 fp16), same split (fit 240 / val 60 / held-out
150), same scorer (LIN_L4, 4 097 parameters), same loss *form* (balanced BCE +
lambda * pairwise margin ranking over all pairs, no sampling), same optimizer and
schedule, same early-stopping criterion (validation Top-256 overlap). Only the
teacher reading changes.

--------------------------------------------------------------------------
The four targets (fixed; no sweep, no tuning)
--------------------------------------------------------------------------
Let r_i be token i's rank in the teacher's P1-G2 map (0 = best, 1023 = worst).

  BASE        positives = Top-256, uniform weights
              -> the exact S2-C1 configuration; the reference arm.

  HEAD_BIN    positives = Top-32, uniform weights
              -> same loss form, positive set moved to the head. Asks directly
                 "is the head of the teacher's ranking a learnable class?"

  HEAD_MULTI  positives = Top-256, tier-graded weights
              -> same positive set as BASE, but every token's supervision weight
                 is a decreasing function of teacher rank (5 fixed tiers).
                 Asks "does grading the wide target by rank band help?"

  HEAD_RANK   positives = Top-32, rank-graded weights (w ~ 1/(r+1))
              -> the head positive set with the strongest possible emphasis on
                 the very top of it. Asks "can the ordering of the head be
                 optimised directly?"

BASE differs from HEAD_BIN only in the positive set; BASE differs from
HEAD_MULTI only in the weighting; HEAD_RANK changes both. Every arm keeps
lambda = 0.5 and margin = 1.0, and every arm's per-token and per-pair weights are
normalised to mean 1 so that lambda means the same thing in all four.

--------------------------------------------------------------------------
Metrics
--------------------------------------------------------------------------
Top-256 overlap is kept but demoted to secondary. The primary metrics are the
ones S2-C2 identified as the currency the downstream value is actually paid in:

  head_recall@k   fraction of the teacher's Top-k that survives inside the
                  selected 256   (k in 8/16/32/64) -- "kept anywhere"
  head_agree@k    |student Top-k  ∩  teacher Top-k| / k  (k in 8/16/32)
                  -- "ordered at the top", not just retained

S2-C2 measured the LIN_L4 baseline at head_recall@8 = 0.736 and
head_agree@8 = 0.192, against a 0.008 chance rate.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT  # noqa: E402

DS_ALL = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]
BUDGET = 256
N_VIS = 1024
D_MODEL = 4096
LAYER = 4                      # the LIN_L4 baseline layer; S2-C1 showed L2/L4 tie

FEATS = f"s2c1_feats_L{LAYER}.npy"
FEATURES_META = "s2c1_features.json"
TEACHER = "s2b_gradient_scores.npz"

# ---------------------------------------------------------------- targets ----
ARMS = ["BASE", "HEAD_BIN", "HEAD_MULTI", "HEAD_RANK"]

# tier -> (upper rank bound, weight). Fixed constants, not searched.
MULTI_TIERS = ((8, 4.0), (32, 2.0), (128, 1.5), (256, 1.0), (N_VIS, 0.25))
RANK_K = 32                    # HEAD_RANK's positive set

SEEDS = (0, 1, 2)
SEED_TAG = f"_s{SEEDS[0]}"


def pos_and_weights(arm: str, order: np.ndarray):
    """Return (pos_idx, token_weight, pair_weight, pos_weight) for one image.

    ``order`` is the teacher's descending-rank token order. ``token_weight`` has
    length N_VIS and mean 1 (it is the per-token BCE weight); ``pair_weight`` has
    length len(pos_idx) and mean 1 (it weights each positive's pairs against the
    negatives). Keeping both at mean 1 is what makes a single lambda comparable
    across arms.
    """
    if arm == "BASE":
        pos = order[:BUDGET]
        return pos, np.ones(N_VIS, np.float32), np.ones(len(pos), np.float32), 3.0

    if arm == "HEAD_BIN":
        pos = order[:32]
        return pos, np.ones(N_VIS, np.float32), np.ones(len(pos), np.float32), 31.0

    if arm == "HEAD_MULTI":
        pos = order[:BUDGET]
        rank = np.empty(N_VIS, np.int32)
        rank[order] = np.arange(N_VIS, dtype=np.int32)
        w = np.empty(N_VIS, np.float32)
        lo = 0
        for hi, wt in MULTI_TIERS:
            w[(rank >= lo) & (rank < hi)] = wt
            lo = hi
        w = (w / w.mean()).astype(np.float32)           # BCE weight, mean 1 over tokens
        pw = w[pos].astype(np.float32)
        pw /= pw.mean()                                 # pair weight, mean 1 over positives
        return pos, w, pw, 3.0

    if arm == "HEAD_RANK":
        pos = order[:RANK_K]
        r = np.arange(RANK_K, dtype=np.float32)          # 0 .. 31 = teacher rank
        pw = (RANK_K / (r + 1.0)).astype(np.float32)
        pw /= pw.mean()
        return pos, np.ones(N_VIS, np.float32), pw, 31.0

    raise KeyError(arm)


def target_description(arm: str) -> dict:
    return {
        "BASE": dict(positives="teacher Top-256", weighting="uniform",
                     pos_weight=3.0, note="exact S2-C1 LIN_L4 configuration"),
        "HEAD_BIN": dict(positives="teacher Top-32", weighting="uniform",
                         pos_weight=31.0, note="head membership, same loss form"),
        "HEAD_MULTI": dict(positives="teacher Top-256",
                           weighting="tier-graded " +
                                     " ".join(f"(<{h}:{w})" for h, w in MULTI_TIERS),
                           pos_weight=3.0, note="wide target graded by rank band"),
        "HEAD_RANK": dict(positives=f"teacher Top-{RANK_K}",
                          weighting=f"rank-graded w={RANK_K}/(r+1), mean-normalised",
                          pos_weight=31.0, note="head ordering optimised directly"),
    }[arm]


# ---------------------------------------------------------------- metrics ----
HEAD_KS = (8, 16, 32, 64)
AGREE_KS = (8, 16, 32)


def head_metrics(score: np.ndarray, torder: np.ndarray, budget: int = BUDGET) -> dict:
    """Every metric this stage reports, from one score vector and one teacher order."""
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    sel = order[:budget]
    sel_set = np.zeros(len(order), dtype=bool)
    sel_set[sel] = True

    out = {"overlap256": float(sel_set[torder[:budget]].mean())}
    for k in HEAD_KS:
        kept = float(sel_set[torder[:k]].sum())
        out[f"head_recall{k}"] = kept / k
        out[f"head_miss{k}"] = float(k - kept)
    for k in AGREE_KS:
        out[f"head_agree{k}"] = len(set(sel[:k].tolist()) & set(torder[:k].tolist())) / k

    # separability diagnostics -- reported, never used to select (S2-C0/S2-C1)
    y = np.zeros(N_VIS, dtype=bool)
    y[torder[:budget]] = True
    r = np.argsort(np.argsort(np.asarray(score, dtype=np.float64), kind="stable"),
                   kind="stable").astype(np.float64) + 1
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    out["auroc"] = float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    hit = np.isin(order, np.nonzero(y)[0])
    prec = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    out["ap"] = float((prec * hit).sum() / n_pos)
    out["spearman"] = float(np.corrcoef(
        np.argsort(np.argsort(score)), np.argsort(np.argsort(-torder)))[0, 1])
    return out


def teacher_orders(G, keys):
    """Descending teacher rank order per key (missing keys -> an unusable order)."""
    out = {}
    for k in keys:
        if k in G.files:
            out[k] = np.argsort(-G[k].astype(np.float64), kind="stable")
        else:
            out[k] = np.arange(N_VIS)
    return out


def aggregate(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k != "overlap256" or True]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def load_plan():
    meta = json.load(open(os.path.join(OUT, FEATURES_META)))
    plan = {p["key"]: p for p in meta["plan"]}
    keys = [p["key"] for p in meta["plan"]]
    rows_of = {s: [i for i, p in enumerate(meta["plan"]) if p["split"] == s]
               for s in ("fit", "val", "test", "causal")}
    return meta, plan, keys, rows_of
