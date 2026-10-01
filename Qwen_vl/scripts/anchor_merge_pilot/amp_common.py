"""Anchor-Merge Pilot — shared plumbing (protocol: docs/anchor_merge_pilot_protocol.md).

Fixed-anchor content aggregation on top of the official EADP facility selector
(b1, K=256) in the native E0 engine.  No training, forward-only, pre-LLM.

Merge math (frozen):
  S        = official facility-location on RAW main features      |S| = K
  gid(i)   = argmax_j cos(x_i, a_j) over RAW main features, ties -> lowest j
  G_j      = {a_j} + assigned dropped tokens
  uniform  : m_j = mean over G_j (anchor included)
  sim      : w = softmax(cos(x_i, a_j)/tau) inside G_j (anchor logit = 1/tau)
  y_j      = a_j + lam * (m_j - a_j)
  pooling  : FP32 accumulation via DETERMINISTIC sort+cumsum group sums
             (no atomics; M4 measured a 1.1-1.3 macro run-to-run drift from
             index_add_'s atomicAdd before switching to this idiom)
  streams  : main V and each DeepStack stream use the SAME assignment and
             weights; each stream aggregates its own values
  positions: compressed tokens keep the anchor's 3-D position (declared
             approximation; the engine's prefill handles this via keep_idx)
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time
from typing import Optional

import torch
import torch.nn.functional as F

# -- path / env bootstrap (same convention as scripts/discovery/common.py) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
QWEN_ROOT = os.path.dirname(os.path.dirname(_HERE))
DISC_DIR = os.path.join(QWEN_ROOT, "scripts", "discovery")
E0_DIR = os.path.join(QWEN_ROOT, "scripts", "e0")
for _p in (os.path.join(QWEN_ROOT, "VLMEvalKit"), QWEN_ROOT, DISC_DIR, E0_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common  # noqa: E402  (sets LMUData / HF_HOME / offline env defaults)

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "anchor_merge_pilot")
os.makedirs(OUT_DIR, exist_ok=True)

DS_LIST = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]
K = 256

# frozen arm registry (protocol §3 / §6; round-2 axes appended per
# docs/anchor_merge_pilot_round2_protocol.md)
DEV_ARMS = ["BASE", "U025", "U050", "U100", "S025"]
CANDIDATES = ["U025", "U050", "U100", "S025"]
CONFIRM_CONTROLS = ["NORM", "SHUF"]
R2_SCOPES = ["MAIN025", "DS025"]           # BOTH == U025 (round 1)
R2_B2 = ["B2BASE", "B2U025"]
R2_K128 = ["K128BASE", "K128U025", "B2K128BASE", "B2K128U025"]

SHUF_SEED = 20261001          # protocol §6 (frozen)
BOOT_SEED = 20261001
N_BOOT = 5000


def arm_cfg(name: str, winner_lam: float = 0.0,
            winner_kind: str = "uniform") -> dict:
    if name == "BASE":
        return dict(kind="base", lam=0.0, selector="b1", K=256)
    if name == "U025":
        return dict(kind="uniform", lam=0.25, selector="b1", K=256)
    if name == "U050":
        return dict(kind="uniform", lam=0.50, selector="b1", K=256)
    if name == "U100":
        return dict(kind="uniform", lam=1.00, selector="b1", K=256)
    if name == "S025":
        return dict(kind="sim", lam=0.25, tau=0.1, selector="b1", K=256)
    if name == "NORM":       # winner's y, rescaled to its own norm
        return dict(kind="norm", lam=winner_lam, winner_kind=winner_kind,
                    selector="b1", K=256)
    if name == "SHUF":       # winner's delta, permuted + rescaled
        return dict(kind="shuf", lam=winner_lam, seed=SHUF_SEED,
                    winner_kind=winner_kind, selector="b1", K=256)
    # ---------------- round 2 (docs/anchor_merge_pilot_round2_protocol.md) --
    if name == "MAIN025":    # aggregate ONLY the main feature stream
        return dict(kind="uniform", lam=0.25, scope="main",
                    selector="b1", K=256)
    if name == "DS025":      # aggregate ONLY the three DeepStack streams
        return dict(kind="uniform", lam=0.25, scope="ds",
                    selector="b1", K=256)
    if name == "B2BASE":     # block8 selection, identity gather
        return dict(kind="base", lam=0.0, selector="b2", K=256)
    if name == "B2U025":     # block8 anchors + uniform merge
        return dict(kind="uniform", lam=0.25, selector="b2", K=256)
    if name == "K128BASE":   # official facility @128, identity gather
        return dict(kind="base", lam=0.0, selector="b1", K=128)
    if name == "K128U025":
        return dict(kind="uniform", lam=0.25, selector="b1", K=128)
    if name == "B2K128BASE":
        return dict(kind="base", lam=0.0, selector="b2", K=128)
    if name == "B2K128U025":
        return dict(kind="uniform", lam=0.25, selector="b2", K=128)
    raise KeyError(name)


# ---------------------------------------------------------------------------
# deterministic group sum (verbatim idiom from m4_common._group_sum)
# ---------------------------------------------------------------------------
def group_sum(v: torch.Tensor, gid: torch.Tensor, r: int) -> torch.Tensor:
    """(r, D) group sums of a (n, D) tensor — deterministic, no atomics."""
    if gid.numel() == 0:
        return torch.zeros(r, v.shape[1], dtype=v.dtype, device=v.device)
    order = torch.argsort(gid, stable=True)
    gs = gid[order]
    cs = torch.cumsum(v[order], dim=0)
    ar = torch.arange(r, device=gid.device)
    start = torch.searchsorted(gs, ar, right=False)
    end = torch.searchsorted(gs, ar, right=True)
    hi = cs[(end.clamp_min(1) - 1)]
    lo_idx = (start - 1).clamp_min(0)
    lo = torch.where((start > 0).unsqueeze(1), cs[lo_idx],
                     torch.zeros_like(cs[lo_idx]))
    out = torch.zeros(r, v.shape[1], dtype=v.dtype, device=v.device)
    nonempty = end > start
    out[nonempty] = (hi - lo)[nonempty]
    return out


# ---------------------------------------------------------------------------
# selection / assignment (all on RAW main features, FP32)
# ---------------------------------------------------------------------------
def official_facility_keep(ctx: dict, K_: int) -> torch.Tensor:
    """Official EADP scoring + official facility-location (selector 'b1').

    Returns global visual indices, ascending.  Requires ctx with
    prep/V/engine/text_mean/text_seq (same keys engine.generate builds).
    """
    from model.e0_selectors import _eadp_parts
    return _eadp_parts(K_, ctx, "facility")


def compute_assignment(V: torch.Tensor, keep: torch.Tensor):
    """cos-argmax assignment of dropped tokens to anchors (main space, FP32).

    Returns (dropped_idx [Nd], gid [Nd], sim [Nd, K] or None).
    gid values are anchor RANKS into `keep` (0..K-1).
    """
    N = V.shape[0]
    dev = V.device
    dropped = torch.ones(N, dtype=torch.bool, device=dev)
    dropped[keep.to(dev)] = False
    dropped_idx = torch.nonzero(dropped, as_tuple=True)[0]
    if dropped_idx.numel() == 0:
        return dropped_idx, torch.empty(0, dtype=torch.long, device=dev), None
    a = F.normalize(V[keep].float(), dim=1)
    x = F.normalize(V[dropped_idx].float(), dim=1)
    sim = x @ a.t()                       # [Nd, K] fp32
    gid = sim.argmax(dim=1)               # first-max -> lowest anchor rank
    return dropped_idx, gid, sim


# ---------------------------------------------------------------------------
# per-stream merge
# ---------------------------------------------------------------------------
def merge_stream(feat: torch.Tensor, keep: torch.Tensor, dropped_idx,
                 gid, kind: str, lam: float, tau: float = 0.1,
                 sim: Optional[torch.Tensor] = None,
                 perm: Optional[torch.Tensor] = None,
                 winner_kind: Optional[str] = None) -> torch.Tensor:
    """Merge ONE feature stream.  Returns y [K, D] in keep order, model dtype.

    feat: [N, D] full-resolution stream (main V or one DeepStack stream).
    sim : member-to-anchor cosine matrix (main space) — REQUIRED for kind='sim'
          but the SAME matrix is reused for every stream (frozen rule: the
          weights come from the main feature space only).
    perm: anchor permutation for kind='shuf' (same across streams).
    winner_kind: for the norm/shuf controls — the winner arm's kind so the
          controls derive their Δ from the winner's actual y.
    """
    K_ = int(keep.numel())
    dt = feat.dtype
    a = feat[keep].float()                                   # [K, D]
    if dropped_idx.numel() == 0 or kind == "base":
        return a.to(dt)
    mem = feat[dropped_idx].float()                          # [Nd, D]
    counts = torch.bincount(gid, minlength=K_).float()       # deterministic

    if kind == "uniform":
        s = group_sum(mem, gid, K_) + a                      # anchor in G_j
        m = s / (counts.unsqueeze(1) + 1.0)
        y = a + lam * (m - a)

    elif kind == "sim":
        assert sim is not None, "kind='sim' needs the frozen main-space cos"
        logits = sim.gather(1, gid.unsqueeze(1)).squeeze(1) / tau   # [Nd]
        # group max incl the anchor logit (1/tau); max reduction is
        # order-independent, hence deterministic
        gmax = torch.full((K_,), 1.0 / tau, device=feat.device,
                          dtype=torch.float32)
        gmax.scatter_reduce_(0, gid, logits, reduce="amax", include_self=True)
        u = torch.exp(logits - gmax[gid])                    # [Nd]
        u_a = torch.exp(1.0 / tau - gmax)                    # anchor weight [K]
        z = group_sum(u.unsqueeze(1), gid, K_).squeeze(1) + u_a
        m = (group_sum((u.unsqueeze(1) * mem), gid, K_) + u_a.unsqueeze(1) * a) \
            / z.unsqueeze(1)
        y = a + lam * (m - a)

    elif kind == "lam0":                                     # gate-only path
        m = a                                                # any finite value; y = a exactly
        y = a + 0.0 * (m - a)

    elif kind in ("norm", "shuf"):
        assert winner_kind is not None, "controls need the winner's kind"
        # the winner's actual y for THIS stream (same assignment/weights rule)
        y_full = merge_stream(feat, keep, dropped_idx, gid, winner_kind,
                              lam, tau, sim, None).float()
        if kind == "norm":
            # y_j = a_j * (||y_j|| / ||a_j||); ||a_j|| == 0 keeps y_j
            na = a.norm(dim=1)
            ny = y_full.norm(dim=1)
            scale = ny / na.clamp_min(1e-12)
            y = torch.where((na > 0).unsqueeze(1), a * scale.unsqueeze(1),
                            y_full)
        else:
            delta = y_full - a                               # [K, D]
            dperm = delta[perm]                              # permuted delta
            nd = delta.norm(dim=1)      # target norm (0 -> delta' = 0)
            np_ = dperm.norm(dim=1)     # source norm (0 -> source IS zero)
            rs = nd / np_.clamp_min(1e-12)  # np_==0 -> source vector is 0
            y = a + dperm * rs.unsqueeze(1)

    else:
        raise KeyError(kind)
    return y.to(dt)


# ---------------------------------------------------------------------------
# bank (frozen S + assignment, protocol §3.3; round-2 banks carry the
# selector and budget in the filename)
# ---------------------------------------------------------------------------
def bank_path(split: str, ds: str, selector: str = "b1",
              K: int = 256, legacy_ok: bool = True) -> str:
    if (selector, K) == ("b1", 256) and legacy_ok:
        legacy = os.path.join(OUT_DIR, f"bank_{split}_{ds}.json.gz")
        if os.path.exists(legacy):
            return legacy
    return os.path.join(OUT_DIR,
                        f"bank_{split}_{selector}_K{K}_{ds}.json.gz")


def load_bank(split: str, ds: str, selector: str = "b1", K: int = 256) -> dict:
    p = bank_path(split, ds, selector, K)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    with gzip.open(p, "rt") as f:
        return json.load(f)


def bank_sha256(split: str, ds: str, selector: str = "b1",
                K: int = 256) -> str:
    with open(bank_path(split, ds, selector, K), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# one-sample runner (mirrors engine.generate's stage timings)
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_one(eng, msg, ds, bank_rec, cfg: dict, max_new_tokens: int = 2048,
            ignore_eos: bool = False, timings: Optional[dict] = None,
            diag: Optional[dict] = None):
    """One generation for one arm.  Returns engine.generate-shaped dict."""
    timings = timings if timings is not None else {}
    K_ = int(cfg.get("K", 256))
    wall0 = time.perf_counter()
    prep = eng.prepare(msg, ds)
    timings["image_preprocess_ms"] = (time.perf_counter() - wall0) * 1e3

    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    V, DS = eng.encode(prep)
    ev[1].record()
    torch.cuda.synchronize()
    timings["vision_ms"] = ev[0].elapsed_time(ev[1])

    n_vis = prep["n_vis"]
    keep = torch.as_tensor(bank_rec["keep"], dtype=torch.long, device=V.device)
    if n_vis <= K_:                       # degenerate: keep everything
        keep = torch.arange(n_vis, device=V.device)

    kind, lam = cfg["kind"], cfg.get("lam", 0.0)
    scope = cfg.get("scope", "both")      # 'both' | 'main' | 'ds' (round 2)
    V_sel = DS_sel = None
    perm = None
    if kind not in ("base",):
        t0 = time.perf_counter()
        dropped_idx, gid, sim = compute_assignment(V, keep)
        if kind == "shuf":
            g = torch.Generator(device="cpu")
            g.manual_seed(int(cfg["seed"]))
            perm = torch.randperm(keep.numel(), generator=g).to(V.device)
        wk = cfg.get("winner_kind")
        if scope in ("both", "main"):
            y = merge_stream(V, keep, dropped_idx, gid, kind, lam,
                             cfg.get("tau", 0.1), sim, perm, winner_kind=wk)
            V_sel = y
        if scope in ("both", "ds"):
            ys = [merge_stream(ds_, keep, dropped_idx, gid, kind, lam,
                               cfg.get("tau", 0.1), sim, perm,
                               winner_kind=wk) for ds_ in DS]
            DS_sel = ys
        torch.cuda.synchronize()
        timings["merge_ms"] = (time.perf_counter() - t0) * 1e3

    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    st = eng.prefill(prep, V, DS, keep, V_sel=V_sel, DS_sel=DS_sel)
    ev[1].record()
    torch.cuda.synchronize()
    timings["llm_prefill_ms"] = ev[0].elapsed_time(ev[1])
    timings["ttft_ms"] = (time.perf_counter() - wall0) * 1e3

    wall1 = time.perf_counter()
    gen_ids, text = eng.decode(st, max_new_tokens, ignore_eos=ignore_eos)
    timings["decode_wall_ms"] = (time.perf_counter() - wall1) * 1e3
    torch.cuda.synchronize()

    meta = eng.invariants(prep, st, V, DS, n_decode=len(gen_ids))
    if diag is not None:
        diag["merge_ms"] = timings.get("merge_ms")
    return dict(text=text, gen_ids=gen_ids, keep_idx=st.keep_idx, meta=meta,
                timings=timings, state=st)
