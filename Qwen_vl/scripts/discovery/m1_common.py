"""
M1 shared definitions: is the L4 LOCAL-MLP gain data, or optimizer steps?

S2-C6 Part A reported the LOCAL-MLP curve 0.7644 / 0.7853 / 0.7922 / 0.8047 at
n = 60 / 120 / 180 / 240 fit images and formally returned DATA-PLATEAU while
saying the curve had not flattened. In that protocol one "epoch" is ceil(n/8)
optimizer updates, so `n` changes *both* the number of independent
teacher-labelled images *and* the number of updates executed before the
checkpoint is frozen. At the H8-selected checkpoint 60 -> 240 multiplies the
update count by 1.875x and the images seen by 2.0x while multiplying the image
pool by 4x.

M1 separates the two. It changes nothing else: same arm (L4 / LOCAL-MLP), same
target (HEAD_RANK), same loss, same optimizer, same selection rule, no query, no
trajectory, no context, no wider model.

--------------------------------------------------------------------------
The row space
--------------------------------------------------------------------------
The S2-C1 cache holds 465 rows: fit 240 / val 60 / test 150 / causal 15. M1 needs
960 fit rows, i.e. 240 more per benchmark, so it generates 720 new instances and
concatenates them *after* the existing 465:

    combined rows  0 .. 464   -> s2c1_feats_L4.npy      (unchanged, never copied)
    combined rows  465 .. 1184 -> m1_feats_L4_extra.npy  (one L4 forward each)

Nothing is ever written to the existing cache, so the n=240 rung of every ladder
reads the *same bytes* S2-C6 read, which is what makes the reproduction gate in
``m1_train.py`` an equality check rather than an approximation.

--------------------------------------------------------------------------
Why the extension draw excludes everything the existing plan touches
--------------------------------------------------------------------------
val and test must not move: they are the frozen 60 and the frozen 150, and the
held-out 150 is the primary metric's sample. So the extension pool for benchmark
i is the complement of ``sample_indices(n_total_i, 150)`` (the whole frozen bank,
fit rows included) and of that benchmark's S2-A causal cases. A row that is
already in the plan, at any split, is never re-drawn.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                              # noqa: E402,F401
from common import sample_indices                                     # noqa: E402
from s2c3_common import (BUDGET, HEAD_KS, LAYER, N_VIS, SEEDS,        # noqa: E402,F401
                         TEACHER, aggregate, head_metrics, teacher_orders)

DS_ORDER = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]

# --------------------------------------------------------------- M1 config ----
EXTRASET_SEED = 20260924        # the one draw that builds the extension pool
N_EXTRA_PER_DS = 240            # 80 feed n=480, all 240 feed n=960
N_BASE_PER_DS = 80              # the frozen fit rows, per benchmark

M1_NS = (240, 480, 960)         # the pre-registered primary ladder
C6_ALIGN_NS = (60, 120, 180)    # the step-controlled counterpart of S2-C6 A
ALL_NS = C6_ALIGN_NS + M1_NS

C6_NS = (60, 120, 180, 240)     # S2-C6 Part A, for the reproduction gate
SUBSET_SEED = 20260923          # S2-C6's nested-subset draw, reused verbatim

# primary protocol: fixed update budget for every n
MAX_STEPS = 900
VAL_EVERY = 30
N_VAL_POINTS = MAX_STEPS // VAL_EVERY          # 30
BATCH = 8
LR = 1e-3
WD = 1e-4
BS_EVAL = 16
MARGIN = 0.01
NBOOT = 10000

# secondary protocol: S2-C6's own budget, reproduced verbatim
MAX_EPOCHS = 80
PATIENCE = 20

TARGET_ARM = "HEAD_RANK"
ARM = "L4"
COL = "head_recall8"
PERIMAGE_COLS = ("head_recall8", "head_recall16", "head_recall32",
                 "head_agree8", "head_agree16", "head_agree32", "overlap256")

EXTRA_FEATS = "m1_feats_L4_extra.npy"
EXTRA_TEACHER = "m1_gradient_scores_extra.npz"
EXTRA_TEACHER_META = "m1_gradient_scores_extra_meta.json"
M1_PLAN = "m1_plan.json"

FEATS_ORIG = f"s2c1_feats_L{LAYER}.npy"
FEATURES_META = "s2c1_features.json"
N_ORIG = 465


# ============================================================ extension pool ==
def causal_indices(cases) -> dict:
    """Benchmark -> set of S2-A causal indices (from s2a_gradient_viability)."""
    out = {ds: set() for ds in DS_ORDER}
    for k, v in cases.items():
        out[v["ds"]].add(int(v["idx"]))
    return out


def build_extension(datasets, n_per_ds: int = N_EXTRA_PER_DS) -> list:
    """The 720 extension instances, benchmark-major, in fixed draw order.

    ``datasets`` maps benchmark -> a dataset object with ``.data``. The draw is a
    single fixed permutation of each benchmark's free rows under
    ``EXTRASET_SEED + 101*(i+1)``, so it is reproducible and never re-drawn.
    """
    cases = json.load(open(os.path.join(OUT, "s2a_gradient_viability.json")))["cases"]
    causal = causal_indices(cases)
    out = []
    for i, ds in enumerate(DS_ORDER):
        n_total = len(datasets[ds].data)
        used = set(sample_indices(n_total, 150)) | causal[ds]
        free = sorted(set(range(n_total)) - used)
        assert len(free) >= n_per_ds, f"{ds}: only {len(free)} free rows"
        rng = np.random.default_rng(EXTRASET_SEED + 101 * (i + 1))
        perm = rng.permutation(len(free))
        for r, j in enumerate(perm[:n_per_ds]):
            out.append(dict(key=f"{ds}_{free[j]}", ds=ds, idx=int(free[j]),
                            extra_rank=r, causal=False, source="extra"))
    assert len(out) == n_per_ds * len(DS_ORDER)
    return out


# ================================================================ M1 plan =====
def load_m1_plan():
    """Combined plan: the 465 S2-C1 rows (unchanged) then the 720 extension rows.

    Returns ``(meta, plan, keys, rows_of)`` where ``rows_of`` has the S2-C6 split
    names for the original rows and an extra ``"extra"`` group for the extension,
    and ``rows_of["extra"][ds]`` is the extension row list for that benchmark in
    draw order (``extra_rank`` ascending) -- the order the ladders slice.
    """
    meta = json.load(open(os.path.join(OUT, FEATURES_META)))
    plan = [dict(p, source="orig", extra_rank=None) for p in meta["plan"]]
    ext = json.load(open(os.path.join(OUT, M1_PLAN)))
    for p in ext["instances"]:
        plan.append(dict(key=p["key"], ds=p["ds"], idx=int(p["idx"]),
                         split="extra", causal=False, source="extra",
                         extra_rank=int(p["extra_rank"])))
    keys = [p["key"] for p in plan]
    assert len(set(keys)) == len(keys), "duplicate key in the combined plan"
    rows_of = {s: [i for i, p in enumerate(plan) if p["split"] == s]
               for s in ("fit", "val", "test", "causal")}
    rows_of["extra"] = {ds: [i for i, p in enumerate(plan)
                             if p["source"] == "extra" and p["ds"] == ds]
                        for ds in DS_ORDER}
    for ds in DS_ORDER:
        r = rows_of["extra"][ds]
        assert len(r) == N_EXTRA_PER_DS, f"{ds}: {len(r)} extension rows"
        ranks = [plan[i]["extra_rank"] for i in r]
        assert ranks == sorted(ranks), f"{ds}: extension rows out of draw order"
    assert len(rows_of["fit"]) == 240 and len(rows_of["val"]) == 60 \
        and len(rows_of["test"]) == 150
    return meta, plan, keys, rows_of


# ============================================================ nested subsets ==
def c6_fit_rows(rows_of, keys, n):
    """S2-C6 Part A's nested per-benchmark subset, reproduced verbatim.

    Bit-for-bit the same construction as ``s2c6_common.nested_fit_rows``: the
    first ``n // 3`` rows of one fixed permutation of each benchmark's fit rows.
    Kept here rather than imported so that the reproduction gate can compare two
    independent implementations of the same published definition.
    """
    assert n % len(DS_ORDER) == 0, f"n={n} is not benchmark-divisible"
    per_ds = n // len(DS_ORDER)
    fit = list(rows_of["fit"])
    out = []
    for i, ds in enumerate(DS_ORDER):
        pool = [r for r in fit if keys[r].startswith(ds + "_")]
        assert per_ds <= len(pool), f"n={n}: {ds} has only {len(pool)} fit rows"
        rng = np.random.default_rng(SUBSET_SEED + 101 * (i + 1))
        perm = rng.permutation(len(pool))
        out.extend(pool[j] for j in perm[:per_ds])
    assert len(out) == n
    return sorted(out)


def ladder_fit_rows(rows_of, keys, n):
    """The M1 fit ladder: S(240) subset of S(480) subset of S(960).

    n <= 240   the S2-C6 nested subset (so n=240 is the published fit set).
    n >  240   the frozen 240 fit rows plus the first ``n//3 - 80`` extension
               rows of each benchmark, in draw order.
    """
    if n <= max(C6_NS):
        return c6_fit_rows(rows_of, keys, n)
    assert n % len(DS_ORDER) == 0, f"n={n} is not benchmark-divisible"
    k = n // len(DS_ORDER) - N_BASE_PER_DS
    assert 0 <= k <= N_EXTRA_PER_DS, f"n={n}: {k} extension rows per benchmark"
    out = list(rows_of["fit"])
    for ds in DS_ORDER:
        out.extend(rows_of["extra"][ds][:k])
    assert len(out) == n
    return sorted(out)


# ============================================================ feature access ==
class M1Store:
    """Random access to the combined fp16 L4 row space (465 original + 720 extra).

    Rows below ``N_ORIG`` are served straight out of the untouched S2-C1 memmap,
    so an arm at n <= 240 reads exactly the bytes S2-C6 read.
    """

    def __init__(self):
        self.orig = np.load(os.path.join(OUT, FEATS_ORIG), mmap_mode="r")
        self.extra = np.load(os.path.join(OUT, EXTRA_FEATS), mmap_mode="r")
        assert self.orig.shape[0] == N_ORIG, self.orig.shape

    def n_rows(self):
        return N_ORIG + self.extra.shape[0]

    def rows(self, idx) -> np.ndarray:
        """Fancy-index the combined row space, preserving the caller's order."""
        idx = np.asarray(idx)
        out = np.empty((len(idx), N_VIS, self.orig.shape[2]), dtype=np.float16)
        lo = idx < N_ORIG
        if lo.any():
            out[lo] = self.orig[idx[lo]]
        if (~lo).any():
            out[~lo] = self.extra[idx[~lo] - N_ORIG]
        return out


def channel_stats_l4(rows, store: M1Store = None) -> tuple:
    """Fit-split per-dimension mean/std of h4, in float64 accumulators.

    Row by row, in the caller's order, with the same accumulator updates as
    ``s2c6_common.channel_stats('h4', rows)``. For a row list drawn entirely from
    the original 465 this returns *bit-identical* statistics to S2-C6's -- which
    is a precondition for the n=240 reproduction gate, not a nicety: a different
    accumulation order perturbs the sums at ~1e-8 and would put a spurious gap
    between two configurations that are meant to be the same configuration.
    """
    store = store or M1Store()
    Hh = store.orig
    s1 = np.zeros(Hh.shape[2], dtype=np.float64)
    s2 = np.zeros(Hh.shape[2], dtype=np.float64)
    n = 0
    for j in rows:
        x = (Hh[j] if j < N_ORIG else store.extra[j - N_ORIG]).astype(np.float32)
        s1 += x.sum(axis=0, dtype=np.float64)
        s2 += np.square(x, dtype=np.float64).sum(axis=0)
        n += x.shape[0]
    mu = s1 / n
    var = np.maximum(s2 / n - mu ** 2, 1e-12)
    return mu.astype(np.float32), np.sqrt(var).astype(np.float32)


def cached_stats_l4(n, fit_rows, tag="m1"):
    """``channel_stats_l4`` memoised per n.

    The two protocols train at the same n values on the same ladders, so without
    a cache the same float64 pass over up to 8 GB of fit rows runs twice for no
    reason. Caching also *guarantees* the two protocols at one n share identical
    preprocessing, rather than relying on the accumulation being re-run
    identically.
    """
    path = os.path.join(OUT, f"{tag}_stats_n{n}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["mu"].astype(np.float32), z["sd"].astype(np.float32)
    mu, sd = channel_stats_l4(fit_rows)
    np.savez(path, mu=mu, sd=sd)
    return mu, sd


class FeatSource:
    """Standardized h4 for a set of combined rows -- the L4 arm's only input.

    ``cache=True`` pulls the raw fp16 rows into RAM once (the fit split is read
    on every step). Normalisation happens on the device, per batch, exactly as
    ``s2c6_common.ViewSource`` does it.
    """

    CHANNEL = "h4"

    def __init__(self, rows, stats, dev, cache=False, store: M1Store = None):
        self.store = store or M1Store()
        self.rows = np.asarray(rows)
        self.dev = dev
        self.cache = cache
        self.arr = (np.ascontiguousarray(self.store.rows(self.rows))
                    if cache else None)
        self.mu = np.asarray(stats[0], dtype=np.float32)
        self.sd = np.asarray(stats[1], dtype=np.float32)

    def _raw(self, pos):
        if self.cache:
            return self.arr[pos]
        return self.store.rows(self.rows[np.asarray(pos)])

    def channel(self, pos):
        x = torch.from_numpy(np.ascontiguousarray(self._raw(pos))).to(
            self.dev).float()
        return (x - torch.from_numpy(self.mu).to(self.dev)) \
            / torch.from_numpy(self.sd).to(self.dev)

    def batch(self, pos):
        return [self.channel(pos)]


# ================================================================ statistics ==
def boot(d, n_boot: int = NBOOT, seed: int = 0) -> dict:
    """Percentile bootstrap CI of a mean, over the held-out images."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "ci_excludes_zero": bool(lo > 0 or hi < 0), "n": int(len(d))}


def compare(A, B, margin: float = MARGIN, n_boot: int = NBOOT) -> dict:
    """A - B with M1's segment labels (prereg 5.2).

    ``A`` and ``B`` are (n_seeds, n_images) per-image held-out head_recall@8.
    The CI is on the seed-averaged difference -- the estimator actually reported
    -- and the per-seed row means are kept so a gain carried by one seed is
    visible rather than averaged away.
    """
    D = np.asarray(A) - np.asarray(B)
    per_seed = [float(D[i].mean()) for i in range(D.shape[0])]
    out = boot(D.mean(0), n_boot=n_boot)
    out["per_seed"] = per_seed
    out["all_seeds_positive"] = bool(all(x > 0 for x in per_seed))
    out["seed_min"], out["seed_max"] = float(min(per_seed)), float(max(per_seed))
    out["resolved"] = bool(out["mean"] >= margin and out["lo"] > 0
                           and out["all_seeds_positive"])
    out["below_margin"] = bool(out["lo"] > 0 and not out["resolved"])
    # EQUIVALENT-NULL is the bounded-interval reading: the data exclude a gain of
    # at least MARGIN. It is *not* the negation of RESOLVED and is never promoted
    # into a verdict by itself (prereg 5.3).
    out["equivalent_null"] = bool(out["hi"] < margin)
    out["ambiguous"] = bool(not out["resolved"] and not out["equivalent_null"]
                            and not out["below_margin"])
    out["label"] = ("RESOLVED" if out["resolved"] else
                    "BELOW-MARGIN" if out["below_margin"] else
                    "EQUIVALENT-NULL" if out["equivalent_null"] else "AMBIGUOUS")
    return out


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
