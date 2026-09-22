"""
S2-C4 shared definitions: per-image oracle directions, their geometry, and the
transfer machinery that asks whether a *shared* direction is the wall.

The stage's question, from S2-C3 §6:

  S2-C3 showed that a direction fitted **on the image itself** recovers 98.2 % of
  the teacher's Top-8, while the best direction fitted across 240 images reaches
  76.6 %. That gap is the object this stage attacks. Two readings of it:

    the direction is genuinely image-specific, and the per-image variation has
    predictable structure (-> an image-adaptive scorer is the next method);

    the direction is image-specific but the variation is unstructured, i.e. no
    cheap forward statistic of image j can pick a direction that works on image
    j (-> token-local scoring is out, and only set/context-dependent scoring is
    left).

Nothing here trains a scorer. Everything is a closed-form read-out of cached
layer-4 hidden states plus the P1-G2 teacher ranking.

--------------------------------------------------------------------------
Objects
--------------------------------------------------------------------------
For image i, with h_i the standardised layer-4 visual hidden states (1024 x 4096)
and `pos` the teacher's Top-k token indices:

    w_i = mean(h_i[pos]) - mean(h_i[~pos])          (mean-difference / centroid)
    w_i <- w_i / ||w_i||_2                          (unit direction)

Top-32 is the primary oracle (S2-C3 measured the Top-32 head as the scale at which
the teacher's own tokens become linearly self-consistent: A32 = 0.903 against
B32 = 0.691 +/- 0.127, where the Top-8 analogue was only 0.6 spread apart). Top-8
is carried as a supplementary.

A direction is only ever used to *score*: score_{j,t} = w_i . h_{j,t}. Because
every direction is unit-norm, scores are on a common scale and comparable across
directions without calibration (S2-B established calibration is a no-op for Top-K).

--------------------------------------------------------------------------
What must never happen
--------------------------------------------------------------------------
The held-out image's teacher map may not be used to *choose* a direction -- only
to evaluate it. Every deployable arm below therefore selects its direction from
fit-set directions using a descriptor computed from h_j alone. The two arms that
do read w_j (SELF, PCA_ORACLE) are labelled optimistic and are upper bounds, not
methods.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                        # noqa: E402
from s2c3_common import (BUDGET, LAYER, N_VIS, TEACHER,        # noqa: E402,F401
                         load_plan, teacher_orders)

FEATS = f"s2c1_feats_L{LAYER}.npy"
DIRS = "s2c4_dirs.npz"

K_HEAD = 32                    # primary oracle target: the teacher's Top-32
K_HEAD8 = 8                    # supplementary
HEAD_KS = (8, 16, 32)
AGREE_KS = (8, 16, 32)

# fixed retrieval neighbourhood sizes -- reported as a small curve, not swept
KS_RETRIEVAL = (1, 4, 8)

DESCRIPTORS = ("mean", "std", "mean+std", "meanstd+delta")


# ------------------------------------------------------------- directions ----
def direction(h: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Mean-difference direction of a token set, unit-normalised.

    ``h`` is (n_tokens, d) standardised; ``pos`` indexes the set. The complement
    mean is taken over all tokens not in ``pos`` -- the same estimator as
    ``s2c3_oracle.direction``, so numbers are comparable to S2-C3 §6.
    """
    mask = np.zeros(len(h), dtype=bool)
    mask[pos] = True
    w = h[mask].mean(0) - h[~mask].mean(0)
    n = float(np.linalg.norm(w))
    return (w / n if n > 0 else w).astype(np.float32)


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return (v / n if n > 0 else v).astype(np.float32)


def standardizer(H, fit_rows):
    """Fit-split per-dimension mean/std -- the exact S2-C1 preprocessing."""
    s1 = np.zeros(H.shape[-1], dtype=np.float64)
    s2 = np.zeros(H.shape[-1], dtype=np.float64)
    n = 0
    for j in fit_rows:
        x = H[j].astype(np.float32)
        s1 += x.sum(axis=0, dtype=np.float64)
        s2 += np.square(x, dtype=np.float64).sum(axis=0)
        n += x.shape[0]
    mu = (s1 / n).astype(np.float32)
    sd = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 1e-12)).astype(np.float32)
    return mu, sd


def descriptors(h: np.ndarray, h2: np.ndarray | None = None) -> dict:
    """Cheap image-level descriptors, all obtainable from the current forward pass.

    ``h`` is the standardised layer-4 block (n_tokens, d); ``h2`` the standardised
    layer-2 block when cached (used only for the supplementary delta variant).
    No teacher information, no query information.
    """
    mean = h.mean(0).astype(np.float32)
    std = h.std(0).astype(np.float32)
    out = {
        "mean": mean,
        "std": std,
        "mean+std": np.concatenate([mean, std]),
    }
    if h2 is not None:
        delta = (h.mean(0) - h2.mean(0)).astype(np.float32)
        out["meanstd+delta"] = np.concatenate([mean, std, delta])
    return out


# ---------------------------------------------------------------- metrics ----
def metrics_single(score: np.ndarray, torder: np.ndarray) -> dict:
    """head_recall@k / head_agree@k / overlap256 for one score vector."""
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    sel = np.zeros(len(order), dtype=bool)
    sel[order[:BUDGET]] = True
    out = {"overlap256": float(sel[torder[:BUDGET]].mean())}
    for k in HEAD_KS:
        out[f"head_recall{k}"] = float(sel[torder[:k]].mean())
        out[f"head_miss{k}"] = float(k - sel[torder[:k]].sum())
    for k in AGREE_KS:
        out[f"head_agree{k}"] = len(
            set(order[:k].tolist()) & set(torder[:k].tolist())) / k
    return out


def metrics_matrix(S: np.ndarray, torders: np.ndarray) -> dict:
    """Metrics for every (test image, fit direction) pair at once.

    ``S`` is (n_test, n_tokens, n_dirs) -- the score every direction assigns to
    every token of every test image. ``torders`` is (n_test, n_tokens) descending
    teacher rank. Returns metric name -> (n_test, n_dirs) array.
    """
    n_test, n_tok, n_dirs = S.shape
    order = np.argsort(-S, axis=1, kind="stable")[:, :BUDGET, :]   # top-256 per pair
    sel = np.zeros((n_test, n_tok, n_dirs), dtype=bool)
    np.put_along_axis(sel, order, True, axis=1)

    jj = np.arange(n_test)[:, None, None]
    d = np.arange(n_dirs)[None, :]
    out = {}
    for k in HEAD_KS:
        idx = torders[:, :k][:, :, None]                            # (n_test, k, 1)
        kept = sel[jj, idx, d[:, None, :]].sum(axis=1)              # (n_test, n_dirs)
        out[f"head_recall{k}"] = kept / float(k)
        out[f"head_miss{k}"] = float(k) - kept

    # agreement needs the teacher's Top-k *for that k* -- masking against the
    # teacher's Top-256 instead would silently score |student Top-k ∩ teacher
    # Top-256| / k, which is a different quantity (verified against a brute-force
    # recomputation in s2c4_verify.py)
    for k in AGREE_KS:
        tmask = np.zeros((n_test, n_tok), dtype=bool)
        for r in range(n_test):
            tmask[r, torders[r, :k]] = True
        topk = order[:, :k, :]                                      # (n_test, k, n_dirs)
        # tmask carries no direction axis: the image index is broadcast, not paired
        hit = tmask[np.arange(n_test)[:, None, None], topk]
        out[f"head_agree{k}"] = hit.sum(axis=1) / float(k)

    # overlap256 = fraction of the teacher's Top-256 inside the selected 256
    idx = torders[:, :BUDGET][:, :, None]
    out["overlap256"] = sel[jj, idx, d[:, None, :]].sum(axis=1) / float(BUDGET)
    return out


# ------------------------------------------------------- trained directions ----
TRAINED = {
    "S2C1_LIN_L4": ("s2c1_LIN_L4.pt", None),
    "HEAD_BIN": ("s2c3_HEAD_BIN_s{seed}.pt", (0, 1, 2)),
    "HEAD_RANK": ("s2c3_HEAD_RANK_s{seed}.pt", (0, 1, 2)),
}


def load_trained_direction(name: str, seed: int | None = None) -> np.ndarray:
    """The weight vector of a trained LINEAR scorer, as a unit direction.

    These live in the same standardised layer-4 space as the oracle directions, so
    they are directly comparable; the bias is irrelevant to Top-K (S2-B: Top-K is
    rank-only) and is dropped.
    """
    import torch
    pat, seeds = TRAINED[name]
    if seeds is not None:
        assert seed is not None, f"{name} needs a seed"
        pat = pat.format(seed=seed)
    ck = torch.load(os.path.join(OUT, pat), map_location="cpu", weights_only=False)
    w = ck["out.weight"].detach().cpu().numpy().reshape(-1).astype(np.float32)
    return unit(w)


def trained_direction_matrix(name: str) -> np.ndarray:
    _, seeds = TRAINED[name]
    if seeds is None:
        return load_trained_direction(name)[None, :]
    return np.stack([load_trained_direction(name, s) for s in seeds])


# ---------------------------------------------------------------- loading ----
def load_cache():
    """The S2-C4 direction cache built by ``s2c4_build.py``."""
    Z = np.load(os.path.join(OUT, DIRS), allow_pickle=True)
    return {k: Z[k] for k in Z.files}


def load_splits():
    meta, plan, keys, rows_of = load_plan()
    return meta, plan, keys, rows_of


def bootstrap_ci(deltas: np.ndarray, n_boot: int = 10000, seed: int = 0):
    """Paired bootstrap on the mean of per-instance differences (as S2-C2/C3)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boot = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return (float(d.mean()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


def paired(a: np.ndarray, b: np.ndarray, n_boot: int = 10000):
    """mean(a - b) with a paired bootstrap CI, in the caller's units."""
    return bootstrap_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float),
                        n_boot=n_boot)


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
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
