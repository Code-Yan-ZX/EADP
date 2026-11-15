"""
M14 -- Facility-lossless acceleration.

Research question (frozen before any number was read):

    Can we obtain facility-level selection quality without facility-level
    computational cost?

This milestone does NOT touch EADP Stage 1.  The importance map (post-smoothing,
post-polarization, alpha=0.5 / beta=2.0 / T=100 / q=0.2) is treated as a frozen
input.  Every solver in this milestone receives the *identical* importance map
and the *identical* visual-visual similarity matrix per sample and is judged on
how faithfully and how quickly it reproduces the official greedy facility
selection.

Fixed reference (PART 0)
------------------------
selector   : model.pruner._greed_select_impl  (official EADP facility greedy)
sim kernel : rebound  = 0.5 * (cos + 1)   (official _sim_visual_impl)
budget     : K = 256 (sweeps 64/128/256/512 for latency only)
samples    : 150 evenly spaced per benchmark (TextVQA_VAL / DocVQA_VAL /
             OCRBench), offset 0 -- the same bank diag_selectors.py used for
             the +19.84-point facility-vs-topk measurement.
hyper-params: alpha=0.5, beta=2.0, entropy_keep_ratio=0.2, M_temp=0.01,
             smoothing kernel 3 sigma 1.0  -- all unchanged.

Success criteria (frozen, from the brief):
    A  accuracy delta vs exact facility  best <= 0.2 pt, loose bound 0.5 pt
    B  selected-set overlap              > 90 %
    C  objective gap                     < 1 %
    D  selector latency                  >= 2x speedup to continue;
                                           ideal 74 -> <20 ms, stretch <10 ms

All offline solver work runs on a captured cache (m14_capture.py): per sample,
the post-polarization importance map, the rebound similarity matrix and the
post-merger visual features.  No model is needed after capture.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # Qwen_vl/scripts/discovery
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # Qwen_vl

import common  # noqa: E402  (discovery common: paths, env bootstrap)
from common import OUTPUT_DIR as DISCOVERY_OUT  # noqa: E402

M14_OUT = os.path.join(os.path.dirname(DISCOVERY_OUT), "m14")
os.makedirs(M14_OUT, exist_ok=True)

DS_ORDER = ("TextVQA_VAL", "DocVQA_VAL", "OCRBench")
SHORT = {"TextVQA_VAL": "TextVQA", "DocVQA_VAL": "DocVQA", "OCRBench": "OCRBench"}

# ---- frozen experiment knobs (fixed before any number was read) -------------
N_VIS = 1024
BUDGET = 256
K_SWEEP = (64, 128, 256, 512)
SIM_MODE = "rebound"

# PART 2 grid: shortlist sizes for screen->verify
SHORTLIST_GRID = (16, 32, 64, 128)
# PART 3 grid: sparse coverage-graph fan-out
SPARSE_R_GRID = (8, 16, 32, 64)
# PART 4 grid: demand-prototype counts
CORESET_M_GRID = (64, 128, 256)

# timing protocol (matches M13: warm-up then repeated synchronised blocks)
N_WARMUP = 10
N_REPEAT = 30

CAPTURE_NPY = "m14_capture.npz"     # keys: importance, ds, idx, key (+ meta)
CAPTURE_SIM = "m14_sim.f32.npy"     # (450, 1024, 1024) float32 memmap
CAPTURE_VIS = "m14_vis.u16.npy"     # (450, 1024, 4096) uint16==bf16 memmap
REFERENCE_JSON = "m14_reference.json"


# ---------------------------------------------------------------- objective --
def facility_objective(sim: torch.Tensor, importance: torch.Tensor,
                       sel: torch.Tensor) -> torch.Tensor:
    """Exact weighted facility objective F(S) = sum_j w_j max_{u in S} S_uj.

    sim: (N, N) float32, importance: (N,) float32, sel: (K,) long indices.
    Returns a scalar tensor.
    """
    cov = sim[sel].max(dim=0).values          # (N,)
    return (cov * importance).sum()


def set_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """|A & B| / |A| -- fraction of the reference set recovered."""
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / max(1, len(sa))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / max(1, len(sa | sb))


# ------------------------------------------------------------------ timing ---
def time_cuda(fn, n_warmup: int = N_WARMUP, n_repeat: int = N_REPEAT):
    """CUDA-event timing of fn(); returns dict in ms. Same protocol as M13."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    for i in range(n_repeat):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ts = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return {"mean": float(ts.mean()), "median": float(np.median(ts)),
            "std": float(ts.std()), "min": float(ts.min()),
            "max": float(ts.max()), "n": n_repeat}


def time_wall(fn, n_warmup: int = N_WARMUP, n_repeat: int = N_REPEAT):
    """Synchronised wall-clock timing (captures CPU-side launch overhead too)."""
    import time as _time
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_repeat):
        t0 = _time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((_time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    return {"mean": float(ts.mean()), "median": float(np.median(ts)),
            "std": float(ts.std()), "min": float(ts.min()),
            "max": float(ts.max()), "n": n_repeat}


# ------------------------------------------------------------------- cache ---
def load_capture():
    """Load the frozen per-sample cache: importance maps, sim matrices, feats."""
    imp_npz = np.load(os.path.join(M14_OUT, CAPTURE_NPY), allow_pickle=True)
    sim = np.load(os.path.join(M14_OUT, CAPTURE_SIM), mmap_mode="r")
    vis = np.load(os.path.join(M14_OUT, CAPTURE_VIS), mmap_mode="r")
    return imp_npz, sim, vis


def sample_entry(imp_npz, sim, vis, i: int):
    """Per-sample tensors on GPU: (importance (N,), sim (N,N) f32, vis (N,D))."""
    imp = torch.from_numpy(imp_npz["importance"][i].astype(np.float32)).cuda()
    s = torch.from_numpy(np.ascontiguousarray(sim[i])).cuda().float()
    v_u16 = torch.from_numpy(np.ascontiguousarray(vis[i]))
    v = v_u16.view(torch.bfloat16).cuda().float()
    return imp, s, v
