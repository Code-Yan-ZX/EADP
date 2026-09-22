"""
S2-C6 shared definitions: residual signal localization.

The stage's question, from S2-C5A:

  A nonlinear token-local function of the cached L4 snapshot lands at 0.8047
  held-out teacher Top-8 recall@256. Two explanations for why it stops there were
  never separated, and S2-C5A explicitly refused to call ~0.80 a ceiling of the
  family because of it:

    DATA         240 fit images is the limit -- the curve has not flattened.
    REPRESENT    the single L4 snapshot is the limit, and the missing signal is
                 carried by the layer trajectory or by the query.

S2-C6 separates them with three parts, all offline, all on the S2-C1 caches:

  A  teacher-data scaling -- the identical LOCAL-MLP config at n = 60/120/180/240
     nested benchmark-stratified fit subsets, to see whether the curve is still
     rising at the largest n this stage can train.
  B  representation localization -- L2, DELTA = h4 - h2, L2+L4, L4+DELTA, each
     parameter-matched to L4, plus an L4+L4 architecture control.
  C  query re-check under the current objective -- L4 plus a low-rank additive
     query term, with a shuffled-query control.

Nothing here trains a final method, sweeps an architecture or tunes a
hyperparameter. The pre-registration is ``docs/scoring_search_s2c6_prereg.md``
and it was written before the first training run.

--------------------------------------------------------------------------
Why the fit-split statistics follow the fit set
--------------------------------------------------------------------------
Part A changes the number of fit images. The per-dimension standardization
mean/std is recomputed on the n images the arm is allowed to see, never inherited
from the 240-image arm: handing the n=60 arm statistics estimated from 240 images
would leak the 180 images that arm is pretending not to have, which is exactly the
kind of leak this project audits for. The statistics are preprocessing that
follows the fit set, not a second degree of freedom.

--------------------------------------------------------------------------
What must never happen
--------------------------------------------------------------------------
No held-out quantity is computed inside an epoch loop. Checkpoints are selected
on the 60-image validation split's teacher Top-8 recall@256 (the S2-C5A H8 rule)
and the held-out 150 is measured once, after the checkpoint is frozen. Top-256
overlap is recorded and is never used to select anything.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                        # noqa: E402,F401
from s2c3_common import (BUDGET, HEAD_KS, LAYER, N_VIS, SEEDS,  # noqa: E402,F401
                         TEACHER, aggregate, head_metrics, load_plan,
                         teacher_orders)

DS_ORDER = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]

# ------------------------------------------------------------- frozen config --
# The S2-C5 / S2-C5A configuration, verbatim. Re-tuned nowhere in this stage.
TARGET_ARM = "HEAD_RANK"
LR = 1e-3
WD = 1e-4
BATCH = 8
MAX_EPOCHS = 80
PATIENCE = 20
BS_EVAL = 16
D_MODEL = 4096

# ------------------------------------------------------------- S2-C6 config ---
LAYER_OF = {"L2": 2, "L4": 4}
NS = (60, 120, 180, 240)
SUBSET_SEED = 20260923         # the one draw that builds the nested subsets
MARGIN = 0.01                  # the S2-C5 / S2-C5A constant
NBOOT = 10000
DEPLOY_TARGET = 0.82
DEPLOY_ASPIRATIONAL = 0.85
SELECTIONS = ("H8", "OV")
SEL_KEY = {"H8": "head_recall8", "OV": "overlap256"}
COL = "head_recall8"
PERIMAGE_COLS = ("head_recall8", "head_recall16", "head_recall32",
                 "head_agree8", "head_agree16", "head_agree32", "overlap256")

# ------------------------------------------------------------------ channels --
# A channel is what one input projection reads. ``("L4", None)`` is the snapshot
# itself; ``("L4", "L2")`` is the trajectory change h_L4 - h_L2. Both operands are
# read as fp16 (that is how they are cached) and combined *after* the fp32 cast,
# so the subtraction never happens in half precision.
CHANNELS = {
    "h2": ("L2", None),
    "h4": ("L4", None),
    "delta": ("L4", "L2"),
}


# ============================================================ nested subsets ===
def nested_fit_rows(rows_of, keys, n):
    """First ``n`` rows of one fixed benchmark-stratified permutation of the fit set.

    Nested by construction: the result for n1 < n2 is a subset of the result for
    n2, because both are prefixes of the same per-benchmark permutation. The draw
    is fixed by ``SUBSET_SEED`` and never re-drawn, so `n` is the only thing that
    changes across Part A's arms.
    """
    fit = list(rows_of["fit"])
    per_ds = n // len(DS_ORDER)
    assert n % len(DS_ORDER) == 0, f"n={n} is not benchmark-divisible"
    out = []
    for i, ds in enumerate(DS_ORDER):
        pool = [r for r in fit if keys[r].startswith(ds + "_")]
        assert per_ds <= len(pool), f"n={n}: {ds} has only {len(pool)} fit rows"
        rng = np.random.default_rng(SUBSET_SEED + 101 * (i + 1))
        perm = rng.permutation(len(pool))
        out.extend(pool[j] for j in perm[:per_ds])
    assert len(out) == n
    return sorted(out)


# ============================================================ feature access ==
def channel_stats(channel: str, rows) -> tuple:
    """Fit-split per-dimension mean/std of one channel, in float64 accumulators.

    Row-by-row, in the same order as ``s2c1_train.standardize_stats``, so that for
    a single-layer channel this returns *bit-identical* statistics to the ones
    S2-C1/C3/C5 trained on. That is what lets the L4 reference here be checked
    against the published S2-C5A number as an exact reproduction rather than an
    approximate one: a different accumulation order would perturb the sums at
    ~1e-8 and put a spurious gap between two configurations that are meant to be
    the same configuration.
    """
    hi, lo = CHANNELS[channel]
    Hh = np.load(os.path.join(OUT, f"s2c1_feats_L{LAYER_OF[hi]}.npy"),
                 mmap_mode="r")
    Hl = (np.load(os.path.join(OUT, f"s2c1_feats_L{LAYER_OF[lo]}.npy"),
                  mmap_mode="r") if lo is not None else None)
    s1 = np.zeros(D_MODEL, dtype=np.float64)
    s2 = np.zeros(D_MODEL, dtype=np.float64)
    n = 0
    for j in rows:
        x = Hh[j].astype(np.float32)
        if Hl is not None:
            x = x - Hl[j].astype(np.float32)
        s1 += x.sum(axis=0, dtype=np.float64)
        s2 += np.square(x, dtype=np.float64).sum(axis=0)
        n += x.shape[0]
    mu = s1 / n
    var = np.maximum(s2 / n - mu ** 2, 1e-12)
    return mu.astype(np.float32), np.sqrt(var).astype(np.float32)


def delta_noise_report(mu, sd, channel: str = "delta") -> dict:
    """Is the delta arm measuring signal, or the fp16 storage floor?

    ``s2c1_feats_*.npy`` is fp16, so a difference of two cached hidden states has
    an absolute quantisation floor of roughly ``2**-11 * E|h|`` per element. A
    dimension whose standard deviation sits at that floor would turn the DELTA arm
    into a quantisation-noise detector, so the comparison is measured and reported
    rather than assumed safe.
    """
    sd = np.asarray(sd, dtype=np.float64)
    q = np.percentile(sd, [0, 1, 5, 50, 95, 100])
    floor = float(2 ** -11 * 0.475)        # E|h| = 0.475, measured on this cache
    return {
        "channel": channel,
        "sd_min": float(q[0]), "sd_p1": float(q[1]), "sd_p5": float(q[2]),
        "sd_median": float(q[3]), "sd_p95": float(q[4]), "sd_max": float(q[5]),
        "fp16_abs_floor_est": floor,
        "min_sd_over_floor": float(q[0] / floor),
        "max_standardization_gain": float(1.0 / max(q[0], 1e-12)),
        "n_dims_below_1e-3": int((sd < 1e-3).sum()),
        "note": "sd is the fit-split per-dimension standard deviation of the "
                "channel; min_sd_over_floor >> 1 means the channel is not "
                "quantisation-limited",
    }


class ViewSource:
    """Random access to a set of standardized input channels over one row set.

    ``cache=True`` pulls the *raw fp16* rows into RAM once: the fit split is read
    on every epoch and re-reading 2 GB of memmap per epoch would be pure I/O
    waste. Normalisation and the delta subtraction happen on the device, per
    batch, exactly as ``s2c1_train.FeatureSource`` does for a single layer.
    """

    def __init__(self, channels, rows, stats, dev, cache=False,
                 q_layer: int | None = None):
        self.channels = tuple(channels)
        self.rows = np.asarray(rows)
        self.dev = dev
        self.cache = cache
        names = sorted({L for c in self.channels for L in CHANNELS[c] if L})
        self.mm, self.arr = {}, {}
        for name in names:
            m = np.load(os.path.join(OUT, f"s2c1_feats_L{LAYER_OF[name]}.npy"),
                        mmap_mode="r")
            if cache:
                self.arr[name] = np.ascontiguousarray(m[self.rows])
            else:
                self.mm[name] = m
        self.stats = {c: (np.asarray(stats[c][0], dtype=np.float32),
                          np.asarray(stats[c][1], dtype=np.float32))
                      for c in self.channels}
        self.q = self.q_mu = self.q_sd = None
        if q_layer is not None:
            Q = np.load(os.path.join(OUT, f"s2c1_query_L{q_layer}.npy"),
                        mmap_mode="r")
            self.q = (np.ascontiguousarray(Q[self.rows]) if cache else Q)
            # The S2-C1 convention: the query is standardized with the *hidden
            # state* statistics of its own layer (s2c1_train.FeatureSource.batch).
            key = {2: "h2", 4: "h4"}[q_layer]
            ref = stats[key]
            self.q_mu = np.asarray(ref[0], dtype=np.float32)
            self.q_sd = np.asarray(ref[1], dtype=np.float32)

    def _raw(self, name, pos):
        if self.cache:
            return self.arr[name][pos]
        return self.mm[name][self.rows[pos]]

    def channel(self, c, pos, alt_pos=None):
        """One standardized channel for ``pos``, or for ``alt_pos`` if given.

        ``alt_pos`` is the WRONG-TRAJECTORY control: the same channel read from a
        different image's rows, so an arm can be scored with one half of its input
        borrowed and the other half honest.
        """
        hi, lo = CHANNELS[c]
        src = pos if alt_pos is None else alt_pos
        x = torch.from_numpy(np.ascontiguousarray(
            self._raw(hi, src))).to(self.dev).float()
        if lo is not None:
            x = x - torch.from_numpy(np.ascontiguousarray(
                self._raw(lo, src))).to(self.dev).float()
        mu, sd = self.stats[c]
        return (x - torch.from_numpy(mu).to(self.dev)) \
            / torch.from_numpy(sd).to(self.dev)

    def query(self, pos, q_perm=None):
        idx = np.asarray(pos) if q_perm is None else np.asarray(q_perm)[pos]
        q = torch.from_numpy(np.ascontiguousarray(self.q[idx])).to(self.dev).float()
        return (q - torch.from_numpy(self.q_mu).to(self.dev)) \
            / torch.from_numpy(self.q_sd).to(self.dev)

    def batch(self, pos, need_q=False, q_perm=None, ch_perm=None, ch_which=1):
        """``pos``: positions into this split's row space.

        ``q_perm`` optionally replaces each position's query with another row's
        (the shuffled-query control), and ``ch_perm`` optionally replaces channel
        ``ch_which`` with another row's copy of it (the wrong-trajectory control).
        Both are indexed in this split's own row space, so
        ``q_perm = ch_perm = arange(n)`` is the honest pass.
        """
        pos = list(pos)
        if ch_perm is None:
            xs = [self.channel(c, pos) for c in self.channels]
        else:
            alt = list(np.asarray(ch_perm)[pos])
            xs = [self.channel(c, alt if i == ch_which else pos)
                  for i, c in enumerate(self.channels)]
        q = self.query(pos, q_perm) if need_q else None
        return xs, q


# ================================================================ statistics ==
def boot(d: np.ndarray, n_boot: int = NBOOT, seed: int = 0) -> dict:
    """Percentile bootstrap CI of a mean, over the held-out images."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "ci_excludes_zero": bool(lo > 0 or hi < 0), "n": int(len(d))}


def compare(A: np.ndarray, B: np.ndarray, margin: float = MARGIN) -> dict:
    """A - B with the pre-registered rule applied.

    ``A`` and ``B`` are (n_seeds, n_images) per-image held-out head_recall@8. The
    CI is taken on the seed-averaged difference -- the estimator actually
    reported -- and the per-seed row means are kept so a gain carried by one seed
    is visible rather than averaged away.

    RESOLVED GAIN requires mean >= margin, CI lower bound > 0 and all three
    seed-matched deltas > 0. BELOW-MARGIN is the case where the CI excludes zero
    but the other two conditions fail: a real but sub-margin effect, which is
    reported and never given a verdict.
    """
    D = np.asarray(A) - np.asarray(B)
    per_seed = [float(D[i].mean()) for i in range(D.shape[0])]
    out = boot(D.mean(0))
    out["per_seed"] = per_seed
    out["all_seeds_positive"] = bool(all(x > 0 for x in per_seed))
    out["seed_min"], out["seed_max"] = float(min(per_seed)), float(max(per_seed))
    out["resolved"] = bool(out["mean"] >= margin and out["lo"] > 0
                           and out["all_seeds_positive"])
    out["below_margin"] = bool(out["lo"] > 0 and not out["resolved"])
    return out


def derangement(n: int, seed: int) -> np.ndarray:
    """A permutation of range(n) with no fixed point (the S2-C5 construction)."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    for i in range(n):
        if p[i] == i:
            j = (i + 1) % n
            p[i], p[j] = p[j], p[i]
    assert not (p == np.arange(n)).any()
    return p


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


def dump(name: str, obj) -> str:
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(jsonable(obj), f, indent=1)
    print(f"[saved] {name}")
    return path
