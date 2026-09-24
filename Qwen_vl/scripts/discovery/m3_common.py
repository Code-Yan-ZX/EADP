"""
M3-v0 -- MissGuard: critical-miss correction on top of a frozen base selector.

The formulation
---------------
    BASE SELECT  ->  AUDIT MISSES  ->  BUDGET-NEUTRAL CORRECT

The base selector is B2 verbatim (official EADP importance + `block8` coverage
greedy at T=256).  MissGuard does not re-rank the 1024 visual tokens and does
not re-design the base selector.  It asks a narrower question:

    among the 768 tokens B2 DROPPED, are there a few the teacher would have
    kept, and can a forward-only student find them?

Whatever it finds (r tokens) is paid for by evicting r tokens from the retained
set, so |S_final| = 256 always.  The visual-token budget is invariant.

What is deliberately NOT here (v0 scope)
----------------------------------------
* no L2/L4 hidden state, no decoder-layer activation of any kind: the student
  reads the vision tower's own post-merger output and the EADP scoring
  intermediates, both of which exist before any LLM layer runs;
* no backward pass at inference, no cached teacher score at inference;
* no generation feedback, no adaptive r, no dynamic depth;
* no base-selector re-design and no architecture search: the student is one
  small MLP with a single linear view of the vision feature.

Everything the audit needs is computed on the device the tensors already live
on; only the r chosen indices and a handful of scalars cross to the host.

The offline teacher
-------------------
Training labels come from the frozen P1-G2 gradient teacher
(`s2b_gradient_scores.npz`, one 1024-vector per instance).  The teacher writes
labels for the fit/val rows only.  The held-out 150 never contributes a label,
and nothing at inference touches it -- except the deliberately-labelled ORACLE
arms, which exist to measure the formulation's ceiling.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUT_DIR                                       # noqa: E402
from instrumented import SELECTORS, TimedEADPPruner                 # noqa: E402

# ------------------------------------------------------------------ geometry --
N_VIS = 1024
GRID = 32
BUDGET = 256
BASE_SELECTOR = "block8"          # == B2, the current system frontier
TEACHER_NPZ = "s2b_gradient_scores.npz"
D_VIS = 4096                      # Qwen3-VL-8B post-merger width

# --------------------------------------------------------------- miss config --
# The two natural positive definitions of the brief's §2 (pilot scope).
POS_K = {"loose": 256, "strict": 64}

EVICT_RULES = ("lowimp", "maxred", "combo")

# Hand-built audit features.  Every one is a function of quantities the
# incumbent's own pre-LLM pruner already produces.
FEATURES = (
    "imp",          # EADP importance (post-smoothing, ^beta) -- the B2 score
    "fused",        # alpha*global + (1-alpha)*local, pre min-max normalisation
    "glob",         # global instruction similarity
    "loc",          # entropy-filtered local similarity
    "imp_pre",      # post-smoothing, pre ^beta
    "red_s0",       # max cosine to a RETAINED token (redundancy with S0)
    "red_all",      # max cosine to any other token
    "nn_drop",      # max cosine to a DROPPED token
    "nb_mean",      # 3x3 neighbourhood mean of imp (self excluded)
    "nb_max",       # 3x3 neighbourhood max of imp (self excluded)
    "nb_std",       # 3x3 neighbourhood std of imp  -- edge/peak structure
    "nb_mean_f",    # 3x3 neighbourhood mean of fused
    "x", "y",       # grid coordinates, normalised to [-1, 1]
    "cos_s0c",      # cosine to the S0 centroid
    "cos_top64c",   # cosine to the centroid of the top-64 by imp
    "vis_norm",     # ||vis[i]||
    "rank_imp",     # rank of imp over all 1024, normalised to [0, 1]
    # --- v1 additions (see docs/m3_missguard_v0.md §5.1) --------------------
    "loc_max",      # best single instruction-token match, pre entropy filter
    "loc_top5",     # mean of the 5 best instruction-token matches
    "loc_std",      # spread of the match across instruction tokens
    "nb_cos",       # mean cosine to the 3x3 neighbourhood in VISION space
)


# ===========================================================================
# student
# ===========================================================================
class MissStudent(nn.Module):
    """Small MLP over the audit features AND the vision feature.

    score_i = f( [ GELU(W_v v_i + b_v) ; GELU(W_h x_i + b_h) ] )

    The vision view is one linear layer of the post-merger feature the vision
    tower already produced.  That is the pre-LLM analogue of the token-local
    family S2-C5/S2-C6 explored on layer-4 hidden states, and it is the only
    part of the input that is not a hand-built scalar -- no attention, no
    context, no decoder layer.
    """

    def __init__(self, d_hand: int, d_vis: int = D_VIS, d_v: int = 64,
                 d_h: int = 64):
        super().__init__()
        self.proj_v = nn.Linear(d_vis, d_v)
        self.proj_h = nn.Linear(d_hand, d_h)
        self.fc1 = nn.Linear(d_v + d_h, d_h)
        self.fc2 = nn.Linear(d_h, d_h // 2)
        self.fc3 = nn.Linear(d_h // 2, 1)
        self.d_hand, self.d_vis, self.d_v, self.d_h = d_hand, d_vis, d_v, d_h

    def forward(self, xh, v):
        a = F.gelu(self.proj_v(v))
        b = F.gelu(self.proj_h(xh))
        z = torch.cat([a, b], dim=-1)
        z = F.gelu(self.fc1(z))
        z = F.gelu(self.fc2(z))
        return self.fc3(z).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Standardiser:
    """Per-dimension standardisation fitted on the fit rows only."""

    def __init__(self, mu, sd):
        self.mu, self.sd = mu, sd

    @classmethod
    def fit(cls, X: np.ndarray):
        mu = X.mean(0).astype(np.float32)
        sd = X.std(0).astype(np.float32)
        sd[sd < 1e-6] = 1.0
        return cls(mu, sd)

    def to_torch(self, device, dtype=torch.float32):
        return (torch.from_numpy(self.mu).to(device=device, dtype=dtype),
                torch.from_numpy(self.sd).to(device=device, dtype=dtype))


# ===========================================================================
# hand-built features -- ONE implementation, torch, on the device
# ===========================================================================
def handcrafted_t(g: dict, vis: torch.Tensor, s0: torch.Tensor,
                  grid: int = GRID) -> torch.Tensor:
    """(N, F) audit features for every visual token, computed where they live.

    `g` is `TimedEADPPruner.last_gpu` (device tensors, see `keep_gpu`); `vis`
    is the (N, D) post-merger vision feature.  Only the returned (N, F) tensor
    ever needs to exist, and it is small enough to stay on the device.
    """
    dev = vis.device
    imp = g["importance"].reshape(-1).float()
    fused = g["fused"].reshape(-1).float()
    glob = g["global_sim"].reshape(-1).float()
    loc = g["local_sim"].reshape(-1).float()
    imp_pre = g["importance_post_smooth"].reshape(-1).float()
    n = imp.shape[0]
    # (1, N, N) -> (N, N): the batch axis is a single image and must not
    # survive, or every row/column index below would address the wrong axis.
    sim = g["sim_matrix"].reshape(n, n).float()

    keep = torch.zeros(n, dtype=torch.bool, device=dev)
    keep[s0] = True
    neg = torch.tensor(float("-inf"), device=dev)

    # red_s0: max cosine to a retained token, self excluded.  The column
    # position of a kept row comes from the mask itself, so an UNSORTED s0
    # (the selector's raw output) is handled correctly.
    sim_k = sim.masked_fill(~keep.unsqueeze(0), neg).clone()
    rows_self = keep.nonzero(as_tuple=True)[0]
    col = keep.cumsum(0)[rows_self] - 1
    sim_k[rows_self, col] = neg
    red_s0 = sim_k.max(dim=1).values

    sim_ns = sim.clone()
    sim_ns.fill_diagonal_(neg)
    red_all = sim_ns.max(dim=1).values
    nn_drop = sim_ns.masked_fill(keep.unsqueeze(0), neg).max(dim=1).values

    # 3x3 neighbourhood statistics of a 32x32 map, self excluded
    def nb(field):
        m = field.reshape(grid, grid)
        p = F.pad(m[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
        acc = torch.zeros_like(m)
        mx = torch.full_like(m, float("-inf"))
        sq = torch.zeros_like(m)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                if dy == 1 and dx == 1:
                    continue
                w = p[dy:dy + grid, dx:dx + grid]
                acc = acc + w
                mx = torch.maximum(mx, w)
                sq = sq + w * w
        mean = acc / 8.0
        std = torch.sqrt(torch.clamp(sq / 8.0 - mean * mean, min=0.0))
        return mean.reshape(-1), mx.reshape(-1), std.reshape(-1)

    nb_mean, nb_max, nb_std = nb(imp)
    nb_mean_f, _, _ = nb(fused)

    yy = torch.div(torch.arange(n, device=dev), grid, rounding_mode="floor")
    xx = torch.arange(n, device=dev) % grid
    x = (xx.float() / (grid - 1.0)) * 2 - 1
    y = (yy.float() / (grid - 1.0)) * 2 - 1

    v = vis.float()
    vn = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    s0c = vn[s0].mean(0)
    s0c = s0c / s0c.norm().clamp_min(1e-8)
    t64 = torch.argsort(-imp)[:64]
    t64c = vn[t64].mean(0)
    t64c = t64c / t64c.norm().clamp_min(1e-8)
    rank_imp = torch.argsort(torch.argsort(imp)).float() / (n - 1.0)

    # --- v1: the instruction match, unaggregated ---------------------------
    lsa = g.get("local_sim_all")
    if lsa is None:
        loc_max = loc_top5 = loc_std = loc
    else:
        L = lsa.reshape(n, -1)                       # (N, num_instr_tokens)
        k = min(5, L.shape[1])
        loc_max = L.max(dim=1).values
        loc_top5 = L.topk(k, dim=1).values.mean(dim=1)
        loc_std = L.std(dim=1)

    # --- v1: feature-space local homogeneity -------------------------------
    # A token whose 3x3 neighbours point the same way as itself is redundant
    # whatever its importance score says; one at a feature-space edge is not.
    vg = vn.t().reshape(vn.shape[1], grid, grid)
    pad = F.pad(vg, (1, 1, 1, 1), mode="replicate")
    acc = torch.zeros(grid, grid, device=dev, dtype=vg.dtype)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            acc = acc + (vg * pad[:, dy:dy + grid, dx:dx + grid]).sum(0)
    nb_cos = (acc / 8.0).reshape(-1)

    X = torch.stack([imp, fused, glob, loc, imp_pre, red_s0, red_all, nn_drop,
                     nb_mean, nb_max, nb_std, nb_mean_f, x, y,
                     vn @ s0c, vn @ t64c, v.norm(dim=-1), rank_imp,
                     loc_max, loc_top5, loc_std, nb_cos], dim=1)
    assert X.shape[1] == len(FEATURES), (X.shape, len(FEATURES))
    return X.contiguous()


# ===========================================================================
# eviction
# ===========================================================================
def _z(v: torch.Tensor) -> torch.Tensor:
    s = v.std()
    return (v - v.mean()) / (s if float(s) > 1e-9 else 1.0)


def evict_t(s0: torch.Tensor, imp: torch.Tensor, sim: torch.Tensor, r: int,
            rule: str = "combo", lam: float = 1.0,
            tval: torch.Tensor = None) -> torch.Tensor:
    """Choose r tokens of S0 to drop, on the device.  No sequential greedy.

    lowimp   lowest EADP importance -- the tokens the base score liked least.
    maxred   highest redundancy: largest cosine to the *rest* of S0, i.e. the
             token whose coverage the other retained tokens already provide.
    combo    z(imp) - lam * z(red), ascending -- low importance AND redundant.
    teacher  lowest teacher score.  NOT deployable: it exists only to build the
             double oracle (teacher rescue AND teacher eviction), which
             separates "what the miss predictor can add" from "what the
             eviction rule costs".  Never used by a learned arm.
    """
    if r <= 0:
        return torch.zeros(0, dtype=torch.long, device=imp.device)
    if rule == "lowimp":
        cost = _z(imp[s0])
    elif rule == "teacher":
        assert tval is not None, "teacher eviction needs the teacher score"
        cost = _z(tval[s0])
    else:
        sub = sim[s0][:, s0].clone()
        sub.fill_diagonal_(float("-inf"))
        red = sub.max(dim=1).values
        if rule == "maxred":
            cost = -_z(red)
        elif rule == "combo":
            cost = _z(imp[s0]) - lam * _z(red)
        else:
            raise KeyError(rule)
    return s0[torch.argsort(cost)[:r]]


# ===========================================================================
# the pruner
# ===========================================================================
class MissGuardPruner(TimedEADPPruner):
    """B2's selector, then a budget-neutral correction of its own misses.

    `miss` describes the correction:
        source    "learned" | "teacher" | "random" | "none"
        r         number of swaps (0 == B2 identity)
        rule      eviction rule (see `evict_t`)
        lam       combo weight
        student/mu_hand/sd_hand/mu_vis/sd_vis  the frozen student
        teacher   (N,) teacher score for THIS instance (source="teacher")
        key       instance key, used to seed the random control reproducibly

    With `source="none"` or `r=0` the forward pass is byte-identical to the
    incumbent's own pruner: same score, same similarity, same selector, same
    sorted index set.  That is the r=0 identity gate.
    """

    def __init__(self, *a, miss: dict = None, **kw):
        super().__init__(*a, **kw)
        self.miss = dict(miss or {})
        self.last_miss = {}
        self.last_miss_events = None

    # ------------------------------------------------------------------ core
    @torch.no_grad()
    def forward(self, image_features, text_embeds_llm, text_embeds_seq_llm,
                grid_thw):
        miss = self.miss
        src = miss.get("source", "none")
        r = int(miss.get("r", 0))
        if src == "none" or r <= 0:
            # Identity path.  Record the REQUEST, not just the absence, so a
            # harness that silently dropped the miss dict cannot look like a
            # deliberate r=0 gate that passed.
            self.last_miss = dict(source=src, r=r, requested_r=r, identity=True,
                                  missguard_ms=0.0)
            return super().forward(image_features, text_embeds_llm,
                                   text_embeds_seq_llm, grid_thw)

        self.keep_gpu = True
        self.last_gpu = {}
        super().forward(image_features, text_embeds_llm,
                        text_embeds_seq_llm, grid_thw)
        self.keep_gpu = False
        # The window opens AFTER the base pass returns, so it measures only what
        # B2 does not do.  Bracketing the base call too would make `miss_ms`
        # contain `selector_ms` and `eadp_scoring_ms`, and the stage sum would
        # then count the base selection twice -- the M2 amendment's double-count
        # artefact, in mirror image (there a window was counted twice; here it
        # would be nested inside another).
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        g = self.last_gpu
        self.last_gpu = {}

        n = image_features.shape[0]
        s0 = torch.sort(g["select_idx"][0].to(torch.long)).values
        sim = g["sim_matrix"].reshape(n, n).float()
        imp = g["importance"].reshape(-1).float()
        # Two views of the same tensor, each matching the bank exactly:
        #   vis_raw -- what `m3_features.py` fed `handcrafted_t`;
        #   vis_q   -- the fp16 form the bank STORED, which is what the vision
        #              standardiser was fitted on.
        # Using one where the other belongs is a silent ~1e-3 train/serve skew.
        vis_raw = g["image_features"].float()
        vis_q = vis_raw.to(torch.float16).float()
        keep = torch.zeros(n, dtype=torch.bool, device=imp.device)
        keep[s0] = True
        dropped = (~keep).nonzero(as_tuple=True)[0]

        if src == "teacher":
            tsc = torch.as_tensor(miss["teacher"], device=imp.device).float()
            rescue = dropped[torch.argsort(-tsc[dropped])[:r]]
        elif src == "random":
            gen = torch.Generator(device="cpu").manual_seed(
                int(miss.get("rng_seed", 0)) + _stable_hash(miss.get("key", "")))
            rescue = dropped[torch.randperm(dropped.numel(), generator=gen)[:r]
                             .to(dropped.device)]
        else:
            Xh = handcrafted_t(g, vis_raw, s0)
            if miss.get("feature_idx") is not None:
                Xh = Xh[:, miss["feature_idx"]]
            mu_h, sd_h = miss["mu_hand"], miss["sd_hand"]
            mu_v, sd_v = miss["mu_vis"], miss["sd_vis"]
            zh = (Xh[dropped] - mu_h) / sd_h
            zv = (vis_q[dropped] - mu_v) / sd_v
            if miss.get("per_instance_z"):
                zh = (zh - zh.mean(0, keepdim=True)) / zh.std(0, keepdim=True).clamp_min(1e-6)
            sc = miss["student"](zh, zv).float()
            rescue = dropped[torch.argsort(-sc)[:r]]

        ev = evict_t(s0, imp, sim, r, rule=miss.get("rule", "combo"),
                     lam=float(miss.get("lam", 1.0)),
                     tval=(torch.as_tensor(miss["teacher"], device=imp.device).float()
                           if miss.get("teacher") is not None else None))
        mask = torch.ones(n, dtype=torch.bool, device=imp.device)
        mask[ev] = False
        s_final = torch.cat([s0[mask[s0]], rescue])
        ev1.record()

        assert s_final.numel() == BUDGET == s0.numel(), (s_final.numel(), s0.numel())
        assert torch.unique(s_final).numel() == BUDGET, "duplicate index in S_final"
        self.last_miss_events = (ev0, ev1)
        self.last_miss = dict(
            source=src, r=r, requested_r=r, rule=miss.get("rule"),
            identity=False,
            n_rescued=int(rescue.numel()), n_evicted=int(ev.numel()),
            rescue_idx=rescue.detach().cpu().tolist(),
            evict_idx=ev.detach().cpu().tolist(),
            s0_idx=s0.detach().cpu().tolist(),
            # audit trail: how much of the teacher's own top-256 the rescue hit
            rescue_in_teacher256=None if miss.get("teacher") is None else int(
                torch.isin(rescue, torch.argsort(
                    -torch.as_tensor(miss["teacher"], device=imp.device)
                )[:256]).sum().item()),
        )
        idx = torch.sort(s_final).values
        return image_features[idx], [int(idx.numel())]

    # ------------------------------------------------------------ readout --
    def read_miss_ms(self) -> float:
        """Consume the pending event pair (one sync).  Called by the harness
        AFTER the prefill, so the miss window never forces an extra sync
        inside the measured region."""
        if self.last_miss_events is None:
            return float(self.last_miss.get("missguard_ms") or 0.0)
        a, b = self.last_miss_events
        self.last_miss_events = None
        torch.cuda.synchronize()
        ms = float(a.elapsed_time(b))
        self.last_miss["missguard_ms"] = ms
        return ms


def feature_index(names) -> torch.Tensor:
    """Column indices of a checkpoint's feature list inside today's FEATURES.

    A checkpoint records the features it was trained on.  `handcrafted_t` always
    emits the full current list, so a student trained before a feature was added
    simply selects its own columns and keeps working -- the prefix order of
    FEATURES is what makes that safe, and this function is where it is checked.
    """
    idx = []
    for nm in names:
        if nm not in FEATURES:
            raise KeyError(f"checkpoint wants feature {nm!r}, not in FEATURES")
        idx.append(FEATURES.index(nm))
    return torch.tensor(idx, dtype=torch.long)


def install_missguard(eng, model, miss: dict, selector: str = BASE_SELECTOR):
    """Install the audit-and-correct pruner on BOTH hooks of an engine.

    `model.pruner` must be rebound as well as `eng.pruner`: the engine's prellm
    branch uses the latter, but the vlmeval wrapper's own path uses the former,
    so a half-install would let any wrapper-side call silently evaluate plain B2
    while the log says MissGuard.
    """
    old = eng.pruner
    mg = MissGuardPruner(
        visual_token_num=old.visual_token_num, alpha=old.alpha, beta=old.beta,
        visual_dim=old.visual_dim, spatial_merge_size=old.spatial_merge_size,
        selector=selector, capture=False, miss=miss,
    ).to(next(model.model.parameters()).device)
    mg.eval()
    mg.sim_mode = "rebound"
    eng.pruner = mg
    model.pruner = mg
    assert eng.pruner is model.pruner, "dual-hook install failed"
    return eng


def _stable_hash(s: str) -> int:
    """Deterministic across processes (unlike hash())."""
    h = 2166136261
    for ch in s.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


# ===========================================================================
# helpers shared by the offline stages
# ===========================================================================
def teacher_topk(g2, s0, k: int) -> np.ndarray:
    """Teacher Top-k minus the base set -- the miss set M of the brief."""
    g2 = np.asarray(g2, dtype=np.float32)
    top = np.argsort(-g2)[:k]
    keep = np.zeros(g2.shape[0], dtype=bool)
    keep[np.asarray(s0)] = True
    return top[~keep[top]]


def load_teacher() -> dict:
    z = np.load(os.path.join(OUTPUT_DIR, TEACHER_NPZ))
    return {k: np.asarray(z[k], dtype=np.float32) for k in z.files}
