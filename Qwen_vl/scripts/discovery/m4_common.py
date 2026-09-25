"""
M4-v0 -- Residual Evidence Compression (REC).

The formulation
---------------
    BASE SELECT (B2)  ->  EVICT r  ->  COMPRESS the rejected evidence into r capsules

MissGuard asked *which* dropped token is the critical miss.  Four independent
stages (S2-C2, S3-B, M3-v0, M3-v2) closed that question: the teacher's head is
not recoverable from pre-LLM features, and the model class is not the limit.
REC changes the question.  It stops trying to name the missed token and asks
whether the *evidence* the base selector rejected can be carried forward in
compressed form:

    S0 = B2's 256 selected tokens
    E  = r tokens evicted from S0 by a fixed rule      (r = 8 / 16 / 32)
    A  = S0 \\ E                                        (|A| = 256 - r)
    D  = all 1024 tokens \\ A                           (the 768 B2 dropped,
                                                         plus the r evicted)
    C  = {c_1 .. c_r}   one capsule per spatial region, pooling D
    S_REC = A u C                                       (|S_REC| = 256)

Every capsule is a convex combination of the post-merger vision features the
tower already produced, so it stays in the space the LLM was trained to read.
No residual vector (v_i - v_anchor) is ever fed forward: the residual decides
only *how much* a dropped token contributes to its capsule, never *what* is
added.  That keeps the arm inside the feature distribution the decoder expects.

The residual weight
-------------------
    u_i = 1 - max_{a in A} cos(v_i, v_a)

A dropped token the retained set already explains has u ~ 0 and contributes
little; a token no retained anchor resembles has u large and dominates its
capsule.  This is the one cheap, forward-only, teacher-free definition of
"unexplainedness" the brief allows.  Nothing here uses a gradient, a decoder
layer, a backward pass or the LLM at all.

Everything is computed on the device the tensors already live on.  Only the
final index set crosses to the host.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUT_DIR                                       # noqa: E402
from instrumented import SELECTORS, TimedEADPPruner                 # noqa: E402
from m3_common import evict_t                                       # noqa: E402

# ------------------------------------------------------------------ geometry --
N_VIS = 1024
GRID = 32
BUDGET = 256
BASE_SELECTOR = "block8"          # == B2, the current system frontier
D_VIS = 4096                      # Qwen3-VL-8B post-merger width

# ------------------------------------------------------------- frozen knobs --
# The brief fixes these before the grid runs and forbids searching them.
EVICT_RULE = "maxred"             # M3-v0's best-and-cheapest eviction
TAU = 0.05                        # residual softmax temperature
SHUF_OFFSET = 7                   # content-free control: rotate by 7 instances
R_GRID = (8, 16, 32)

# ------------------------------------------------------------------ weights --
WEIGHTS = ("residual", "mean", "imp", "shuf", "anchor_mean")
ASSIGN = ("spatial", "fps")


# ===========================================================================
# spatial partition
# ===========================================================================
def region_factor(r: int) -> tuple:
    """The most-square factorisation rows*cols == r (rows <= cols).

    r=8 -> (2,4), r=16 -> (4,4), r=32 -> (4,8).  Deterministic, rule-driven and
    uniform over the 32x32 grid; there is no clustering search and no seed.
    """
    best = None
    for rows in range(1, int(math.isqrt(r)) + 1):
        if r % rows:
            continue
        cols = r // rows
        if best is None or cols * best[0] < best[1] * rows:
            best = (rows, cols)
    rows, cols = best
    assert rows * cols == r
    return rows, cols


_REGION_CACHE: dict = {}


def partition_ids(r: int, grid: int = GRID, device=None) -> torch.Tensor:
    """(N,) region id in [0, r) for every token of the 32x32 grid."""
    key = (r, grid)
    if key not in _REGION_CACHE:
        rows, cols = region_factor(r)
        yy = torch.arange(grid).repeat_interleave(grid)          # (N,)
        xx = torch.arange(grid).repeat(grid)
        rr = (yy * rows // grid).clamp(max=rows - 1)
        cc = (xx * cols // grid).clamp(max=cols - 1)
        _REGION_CACHE[key] = (rr * cols + cc).to(torch.long)
    out = _REGION_CACHE[key]
    return out if device is None else out.to(device)


def region_map(r: int, grid: int = GRID) -> np.ndarray:
    """(grid, grid) int map of region ids, for the figures."""
    return partition_ids(r, grid).reshape(grid, grid).numpy()


# ===========================================================================
# capsule construction -- the whole method, on the device
# ===========================================================================
def residual_u(vis: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """u_i = 1 - max_{a in A} cos(v_i, v_a), for every token i.

    Computed straight from the vision features rather than from the selector's
    similarity matrix, so the definition holds whatever `sim_mode` the pruner
    runs under.  One (1024 x D) @ (D x |A|) matmul: ~2 GFLOP, ~0.1 ms.
    """
    v = vis.float()
    vn = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cos = vn @ vn[A].t()                                    # (N, |A|)
    return 1.0 - cos.max(dim=1).values


def _fps_assign(vis: torch.Tensor, A: torch.Tensor, dropped: torch.Tensor,
                r: int, imp: torch.Tensor) -> torch.Tensor:
    """Feature-space assignment: r farthest-point seeds over A, nearest seed wins.

    The feature-space analogue of the spatial partition, and the brief's
    ablation B.  Seeds are chosen by farthest-point sampling over the retained
    anchors (deterministic: the seed is the highest-importance anchor, and every
    later seed is the anchor farthest from those already chosen), then every
    dropped token joins the seed it is closest to.  No clustering search, no
    iteration, no seed parameter.
    """
    va = vis.float()[A]
    van = va / va.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    aa = van @ van.t()                                       # (|A|, |A|)
    r = min(r, A.numel())
    first = int(torch.argmax(imp[A]))
    seeds = [first]
    best = aa[first].clone()          # max similarity to the chosen set
    for _ in range(1, r):
        # FARTHEST point, so argmin of the running max-similarity.  An already
        # chosen seed has best == 1, the largest value possible, so argmin can
        # never reselect one.
        nxt = int(torch.argmin(best))
        seeds.append(nxt)
        best = torch.maximum(best, aa[nxt])
    sd = torch.tensor(seeds, dtype=torch.long, device=vis.device)
    vd = vis.float()[dropped]
    vdn = vd / vd.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.argmax(vdn @ van[sd].t(), dim=1)            # (|D|,) in [0, r)


def build_capsules(vis: torch.Tensor, imp: torch.Tensor, u: torch.Tensor,
                   A: torch.Tensor, dropped: torch.Tensor, r: int,
                   weights: str = "residual", assign: str = "spatial",
                   tau: float = TAU, tau_imp: float = None,
                   norm_restore: bool = False) -> tuple:
    """Return (capsules (r, D), stats dict).  Pure function, no state.

    `weights`
        residual     softmax(u_i / tau) within the group   -- the method
        mean         uniform within the group              -- plain mean merge
        imp          softmax(imp_i / tau_imp) within group -- importance merge
        shuf         `u` is another instance's residual map (injected by the
                     harness), so the weights carry no information about THIS
                     image.  The content-free control.
        anchor_mean  no dropped token at all: the capsule is the mean of the
                     RETAINED tokens in the group.  Separates "the capsule
                     carries rejected evidence" from "a region summary of the
                     retained set is itself useful".
    `assign`
        spatial      group = the token's own grid cell (the 32x32 partition)
        fps          group = the nearest of r farthest-point anchors
    """
    assert weights in WEIGHTS, weights
    assert assign in ASSIGN, assign
    dev = vis.device
    n, d = vis.shape
    v = vis.float()

    if weights == "anchor_mean":
        # The group's RETAINED tokens; every cell of a 32x32/r partition holds
        # 240/r >= 7 of them, so a fallback is a formality, not a path.
        rid = partition_ids(r, device=dev)
        w1 = torch.ones(A.numel(), device=dev)
        caps, counts, empty = _pool(v[A], w1, rid[A], r, d,
                                    fallback=v[A].mean(0), want_counts=True)
        return caps, dict(n_dropped=int(dropped.numel()), n_anchor=int(A.numel()),
                          n_empty=int(empty), counts=counts.detach().cpu().tolist(),
                          ess=float(A.numel()) / r,
                          cap_norm=float(caps.norm(dim=1).median()))

    if assign == "fps":
        gid = _fps_assign(v, A, dropped, r, imp)             # (|D|,)
    else:
        gid = partition_ids(r, device=dev)[dropped]          # (|D|,)

    if weights == "mean":
        w = torch.ones(dropped.numel(), device=dev)
    elif weights == "imp":
        w = _group_softmax(imp[dropped], gid, r, tau_imp)
    else:
        # "residual" and "shuf" differ only in WHERE u came from; the shuf arm
        # injects another instance's map through `rec["shuf_u"]`, so the code
        # path is deliberately identical.
        w = _group_softmax(u[dropped], gid, r, tau)

    caps, counts, empty = _pool(v[dropped], w, gid, r, d,
                                fallback=v[A].mean(0), want_counts=True)

    if norm_restore:
        # A convex combination of k diverse tokens has a smaller norm than a
        # token (~0.57x at r=16 measured offline).  Rescaling each capsule to
        # the mean norm of the tokens it pooled removes that one distribution
        # shift while leaving its direction untouched.
        #
        # Empty cells are excluded: they hold the fallback vector, whose norm is
        # not the norm of anything they pooled, and `counts == 0` there would
        # divide them to zero -- a capsule silently deleted from the sequence.
        num = _group_sum(v[dropped].norm(dim=1).unsqueeze(1), gid, r).squeeze(1)
        tgt = torch.where(counts > 0, num / counts.clamp_min(1),
                          caps.norm(dim=1))
        cur = caps.norm(dim=1).clamp_min(1e-8)
        caps = caps * (tgt / cur).unsqueeze(1)

    stats = dict(n_dropped=int(dropped.numel()), n_anchor=int(A.numel()),
                 n_empty=int(empty), counts=counts.detach().cpu().tolist(),
                 ess=float(_ess(w, gid, r).mean()),
                 cap_norm=float(caps.norm(dim=1).median()))
    return caps, stats


def _group_sum(v: torch.Tensor, gid: torch.Tensor, r: int) -> torch.Tensor:
    """(r, D) group sums of a (n, D) tensor -- DETERMINISTIC.

    `Tensor.index_add_` reduces with atomicAdd on CUDA, so its summation order
    varies from launch to launch. The difference is only ~2e-7 in fp32, but a
    capsule is cast to bf16 (relative eps ~8e-3) before it enters the decoder,
    so that difference occasionally flips a rounding and with it a greedy
    token: the first M4 grid gave the SAME arm 58.06 and 59.18 macro in two
    fresh processes. Sorting by group and differencing a prefix sum has no
    atomics, so the arm becomes reproducible, which is a property a deployable
    method needs and a measurement needs more.
    """
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


def _group_softmax(x: torch.Tensor, gid: torch.Tensor, r: int,
                   tau: float) -> torch.Tensor:
    """softmax(x / tau) inside each group; stable, no Python loop over groups."""
    if tau is None:
        tau = TAU
    m = torch.full((r,), float("-inf"), device=x.device)
    m = m.index_reduce(0, gid, x, "amax", include_self=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    e = torch.exp((x - m[gid]) / tau)
    z = _group_sum(e.unsqueeze(1), gid, r).squeeze(1)
    return e / z[gid].clamp_min(1e-12)


def _ess(w: torch.Tensor, gid: torch.Tensor, r: int) -> torch.Tensor:
    """Effective sample size (1 / sum w^2) per NON-EMPTY group.

    Empty groups are dropped rather than returned as a huge reciprocal: a
    partition cell with nothing in it has no effective sample size at all, and
    averaging a sentinel into the diagnostic would report a number that no
    capsule ever had.
    """
    tot = _group_sum(w.unsqueeze(1), gid, r).squeeze(1)
    p = w / tot[gid].clamp_min(1e-12)
    ss = _group_sum((p * p).unsqueeze(1), gid, r).squeeze(1)
    return ss[tot > 0].clamp_min(1e-12).reciprocal()


def _pool(v: torch.Tensor, w: torch.Tensor, gid: torch.Tensor, r: int, d: int,
          fallback: torch.Tensor, want_counts: bool = False):
    """Group-weighted mean of `v` under `gid`, with a fallback for empty groups.

    Every reduction here goes through `_group_sum`, never `index_add_`, so the
    capsule a cell produces is a function of its inputs and nothing else.
    """
    tot = _group_sum(w.unsqueeze(1), gid, r).squeeze(1)
    wn = w / tot[gid].clamp_min(1e-12)
    num = _group_sum(v * wn.unsqueeze(1), gid, r)
    den = _group_sum(wn.unsqueeze(1), gid, r).squeeze(1)
    empty = (den <= 0).nonzero(as_tuple=True)[0]
    if empty.numel():
        num[empty] = fallback
    if want_counts:
        counts = _group_sum(torch.ones(gid.numel(), 1, dtype=v.dtype,
                                       device=v.device), gid, r).squeeze(1)
        return num, counts, empty.numel()
    return num


# ===========================================================================
# the pruner
# ===========================================================================
class RECPruner(TimedEADPPruner):
    """B2's selector, then a budget-neutral compression of what it rejected.

    `rec` describes the arm:
        mode          "capsule" | "evict" | "none"
        r             number of capsules (== number of evictions)
        weights       see `build_capsules`
        assign        "spatial" | "fps"
        tau, tau_imp  softmax temperatures
        norm_restore  rescale each capsule to the pooled tokens' mean norm
        shuf_u        (N,) residual map of ANOTHER instance (weights="shuf")
        rule          eviction rule, default "maxred"

    `mode="none"` or `r=0` reproduces the incumbent's pruner byte-for-byte:
    same score, same similarity, same selector, same sorted index set.  That is
    the r=0 identity gate, and it is the only arm that may claim B2's numbers.
    """

    def __init__(self, *a, rec: dict = None, **kw):
        super().__init__(*a, **kw)
        self.rec = dict(rec or {})
        self.last_miss = {}
        self.last_miss_events = None

    # ------------------------------------------------------------------ core
    @torch.no_grad()
    def forward(self, image_features, text_embeds_llm, text_embeds_seq_llm,
                grid_thw):
        rec = self.rec
        mode = rec.get("mode", "none")
        r = int(rec.get("r", 0))
        if mode == "none" or r <= 0:
            self.last_miss = dict(mode=mode, r=r, requested_r=r, identity=True,
                                  missguard_ms=0.0)
            return super().forward(image_features, text_embeds_llm,
                                   text_embeds_seq_llm, grid_thw)

        self.keep_gpu = True
        self.last_gpu = {}
        super().forward(image_features, text_embeds_llm,
                        text_embeds_seq_llm, grid_thw)
        self.keep_gpu = False
        # The window opens AFTER the base pass returns: it measures only what
        # B2 does not already do, so `rec_ms` can never nest the selector or the
        # EADP scoring window (the M2 amendment's double-count, in mirror image).
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()

        g = self.last_gpu
        self.last_gpu = {}
        n = image_features.shape[0]
        s0 = torch.sort(g["select_idx"][0].to(torch.long)).values
        imp = g["importance"].reshape(-1).float()
        vis = g["image_features"].float()
        sim = g["sim_matrix"].reshape(n, n).float()

        # ---- evict r tokens of S0 by a rule fixed before the grid ran -------
        ev = evict_t(s0, imp, sim, r, rule=rec.get("rule", EVICT_RULE))
        mask = torch.ones(n, dtype=torch.bool, device=imp.device)
        mask[ev] = False
        A = s0[mask[s0]]                                     # (256-r,)
        keep = torch.zeros(n, dtype=torch.bool, device=imp.device)
        keep[A] = True
        dropped = (~keep).nonzero(as_tuple=True)[0]          # (768+r,)

        if mode == "evict":
            # Diagnostic only: the eviction with nothing put back.  It is NOT a
            # candidate -- it breaks the 256-token budget on purpose, so that
            # "what the capsules add" can be separated from "what the eviction
            # costs".  Never reported as a method.
            assert A.numel() == BUDGET - r
            ev1.record()
            self.last_miss_events = (ev0, ev1)
            self.last_miss = dict(mode=mode, r=r, requested_r=r, identity=False,
                                  n_capsules=0, n_evicted=int(ev.numel()),
                                  n_tokens=int(A.numel()),
                                  s0_idx=s0.detach().cpu().tolist(),
                                  a_idx=A.detach().cpu().tolist(),
                                  evict_idx=ev.detach().cpu().tolist())
            return vis[A].to(image_features.dtype), [int(A.numel())]

        # ---- residual unexplainedness and the r capsules -------------------
        u = residual_u(vis, A)
        if rec.get("weights") == "shuf":
            assert rec.get("shuf_u") is not None, \
                "weights='shuf' needs the injected residual map"
            u = torch.as_tensor(rec["shuf_u"], device=u.device).float()
            assert u.numel() == n
        caps, cstats = build_capsules(
            vis, imp, u, A, dropped, r,
            weights=rec.get("weights", "residual"),
            assign=rec.get("assign", "spatial"),
            tau=float(rec.get("tau", TAU)),
            tau_imp=rec.get("tau_imp"),
            norm_restore=bool(rec.get("norm_restore", False)))
        assert caps.shape == (r, vis.shape[1]), caps.shape

        # The pooling runs in fp32 (a 64-term weighted sum in bf16 would lose
        # most of its mantissa) and the result is cast back to whatever the
        # vision tower emitted, so the decoder sees exactly one dtype -- the
        # base pruner's own output dtype, not a new one.
        out = torch.cat([vis[A], caps], dim=0).to(image_features.dtype)
        ev1.record()
        assert out.shape[0] == BUDGET, out.shape
        self.last_miss_events = (ev0, ev1)
        self.last_miss = dict(
            mode=mode, r=r, requested_r=r, identity=False,
            weights=rec.get("weights", "residual"), assign=rec.get("assign", "spatial"),
            tau=float(rec.get("tau", TAU)), norm_restore=bool(rec.get("norm_restore", False)),
            n_capsules=int(r), n_evicted=int(ev.numel()), n_tokens=int(out.shape[0]),
            s0_idx=s0.detach().cpu().tolist(),
            evict_idx=ev.detach().cpu().tolist(),
            a_idx=A.detach().cpu().tolist(),
            cap_stats=cstats,
            # Which instance the SHUF weights came from, recorded so the gate
            # can prove the control was never fed its own image.
            shuf_src=rec.get("shuf_src"),
            # per-region pooled count, so a degenerate partition is visible in
            # the record rather than inferred from a score
            u_p50=float(u.median()), u_p90=float(u.quantile(0.9)))
        return out, [int(out.shape[0])]

    # ------------------------------------------------------------ readout --
    def read_miss_ms(self) -> float:
        """Consume the pending event pair (one sync).  Named as in M3 so the M2
        accuracy and paired-perf harnesses read REC's window unchanged."""
        if self.last_miss_events is None:
            return float(self.last_miss.get("missguard_ms") or 0.0)
        a, b = self.last_miss_events
        self.last_miss_events = None
        torch.cuda.synchronize()
        ms = float(a.elapsed_time(b))
        self.last_miss["missguard_ms"] = ms
        return ms


def install_rec(eng, model, rec: dict, selector: str = BASE_SELECTOR):
    """Install the compress-the-rejected pruner on BOTH hooks of an engine.

    `model.pruner` must be rebound as well as `eng.pruner`: the engine's prellm
    branch uses the latter, but the vlmeval wrapper's own path uses the former,
    so a half-install would let a wrapper-side call silently evaluate plain B2
    while the log says REC.
    """
    old = eng.pruner
    pr = RECPruner(
        visual_token_num=old.visual_token_num, alpha=old.alpha, beta=old.beta,
        visual_dim=old.visual_dim, spatial_merge_size=old.spatial_merge_size,
        selector=selector, capture=False, rec=rec,
    ).to(next(model.model.parameters()).device)
    pr.eval()
    pr.sim_mode = "rebound"
    eng.pruner = pr
    model.pruner = pr
    assert eng.pruner is model.pruner, "dual-hook install failed"
    return eng
