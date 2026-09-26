"""
M5 -- Safe Removal (SafeTrim).

The formulation
---------------
    BASE SELECT (B2)  ->  S0, |S0| = 256
    PREDICT RISK      ->  r_i = P(deleting i hurts the answer), for i in S0
    TRIM              ->  S* = S0 \\ {the k lowest-risk tokens}   (|S*| = 256 - k)

Four rounds of this project (S2-C2, S3-B, M3-v0, M3-v2) closed the *importance*
question -- "which dropped token is the critical miss" -- because the teacher's
head is not recoverable from pre-LLM features.  Safe Removal changes the
question rather than the model class.  It never looks at the 768 tokens B2
dropped and never asks which token is most valuable.  It asks the opposite and
strictly easier question about the 256 tokens B2 already kept:

    which of these can I remove without changing the answer?

The lead this line is built on is in M4's own grid.  `EVICT-r8` -- delete the 8
most redundant retained anchors, put nothing back, end at 248 tokens -- was the
largest positive effect M4 measured (+3.77 macro, CI [-0.2, +8.0], 8 fixed /
3 broken, 0.63 ms), and it was not the only time: M3-v0's `RND-maxred-r8` =
61.82 and `MG-maxred-r8` = 64.98 both sit above B2.  A prescribed budget of 256
is not obviously the optimal retained size, and "safe to remove" may be a far
more learnable predicate than "critical to keep".

Deployability contract (unchanged from M3/M4, and binding)
----------------------------------------------------------
Forward-only, pre-LLM.  Every feature is a function of quantities the
incumbent's own pruner already produces -- the EADP score and its components,
the selector's similarity matrix, the post-merger vision feature the tower
already emitted -- plus the retained set S0 itself.  No gradient, no decoder
layer, no backward pass, no teacher at inference, no generation feedback.
Nothing here reads an accuracy number to set a knob.

Teacher (offline only, never at inference)
------------------------------------------
The label is the one the brief demands, and it is NOT the gradient ranking:

    d_i = L(y | S0 \\ {i}) - L(y | S0)

teacher-forced gold-answer NLL under PRE-LLM delivery, the same harness S3-A
used.  d_i > 0 means deleting i hurts; d_i <= 0 means it is free or helpful.
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
D_VIS = 4096                      # Qwen3-VL-8B post-merger width

BANK = "m3_bank.npz"              # the frozen 450 (fit 240 / val 60 / test 150)
TEACHER_NPZ = "s2b_gradient_scores.npz"   # the critical-head teacher, for the
                                          # formulation comparison only

# ------------------------------------------------------------ frozen knobs --
# The brief fixes these before any accuracy number is read.
TRIM_GRID = (4, 8, 12, 16)        # k, the number of tokens deleted
PRIMARY_K = 8                     # M4's EVICT-r8 hypothesis
RANDOM_SEEDS = (0, 1, 2, 3, 4)
N_FIT_SEEDS = 3                   # probe seeds; the spread is reported

# ===========================================================================
# the feature set -- every column a deployable pre-LLM quantity
# ===========================================================================
#   EADP score and its components
#       imp          the B2 score itself (post-smoothing, ^beta)
#       imp_pre      post-smoothing, pre ^beta
#       fused        alpha*global + (1-alpha)*local, pre min-max
#       glob         global instruction similarity
#       loc          entropy-filtered local similarity
#       loc_max      best single instruction-token match, pre entropy filter
#       loc_top5     mean of the 5 best instruction-token matches
#       loc_std      spread of the match across instruction tokens
#   redundancy inside the retained set -- the safe-removal axis
#       red_s0       max cosine to ANOTHER retained token (self excluded)
#       red_s0_top8  mean of the 8 largest cosines to other retained tokens
#       resid_u      1 - red_s0: how poorly the rest of S0 reconstructs it
#       nn4_recon    ||v_i - proj_span(4 nearest retained)|| / ||v_i||
#       red_all      max cosine to any other token
#       nn_drop      max cosine to a DROPPED token
#       nn_drop_min  1 - nn_drop: is there a dropped token that already covers it
#   local structure
#       nb_mean/nb_max/nb_std   3x3 neighbourhood statistics of imp
#       nb_mean_f               3x3 neighbourhood mean of fused
#       nb_red                  3x3 neighbourhood mean of red_s0
#       nb_cos                  mean cosine to the 3x3 neighbourhood in vision space
#   geometry and scale
#       x, y         grid coordinates in [-1, 1]
#       vis_norm     ||vis[i]||
#       rank_imp     rank of imp over all 1024, in [0, 1]
#       cos_s0c      cosine to the S0 centroid
#       cos_top64c   cosine to the centroid of the top-64 by imp
FEATURES = (
    "imp", "imp_pre", "fused", "glob", "loc", "loc_max", "loc_top5", "loc_std",
    "red_s0", "red_s0_top8", "resid_u", "nn4_recon", "red_all", "nn_drop",
    "nn_drop_min",
    "nb_mean", "nb_max", "nb_std", "nb_mean_f", "nb_red", "nb_cos",
    "x", "y", "vis_norm", "rank_imp", "cos_s0c", "cos_top64c",
)
N_FEAT = len(FEATURES)
FEAT_INDEX = {n: i for i, n in enumerate(FEATURES)}
# The three redundancy views the trimming baselines read directly.  Named here
# so the baseline arms and the probe cannot drift onto different columns.
RED_COL = FEAT_INDEX["red_s0"]
IMP_COL = FEAT_INDEX["imp"]
RECON_COL = FEAT_INDEX["nn4_recon"]


# ===========================================================================
# bank
# ===========================================================================
M5_BANK = "m5_bank.npz"
M5_VIS = "m5_vis.npy"


def load_bank(path: str = None) -> dict:
    """The M3 bank -- the frozen 450 with their B2 sets and the gradient teacher.

    Kept as its own loader because it is the reference the M5 bank is gated
    against (G-S0): its `s0` is what M3 and M4 delivered, so agreeing with it is
    what makes an M5 number comparable to an M3/M4 number.
    """
    z = np.load(os.path.join(OUTPUT_DIR, path or BANK), allow_pickle=False)
    out = {k: z[k] for k in ("key", "ds", "idx", "split", "s0", "g2", "X", "vis")}
    out["_index"] = {str(k): i for i, k in enumerate(out["key"])}
    return out


def load_m5_bank() -> dict:
    """The M5 bank: bit-exact bf16 vision features + the M5 feature matrix.

    `vis` is a read-only memory map of the (450, 1024, 4096) uint16 array; a
    row is turned into bf16 by `vis_row()`.  The m3 bank's fp16 column moved L
    by up to 2.8e-2 nats through a bf16 rounding flip (`m5_cost.json`), which is
    the same order as the effects this line measures, so M5 uses this one.
    """
    z = np.load(os.path.join(OUTPUT_DIR, M5_BANK), allow_pickle=False)
    out = {k: z[k] for k in ("key", "ds", "idx", "split", "s0", "g2", "X",
                             "feature_names")}
    out["vis"] = np.load(os.path.join(OUTPUT_DIR, M5_VIS), mmap_mode="r")
    out["_index"] = {str(k): i for i, k in enumerate(out["key"])}
    return out


def vis_row(bank: dict, i: int, device=None) -> torch.Tensor:
    """Row `i` of the bank's vision cache as a bf16 tensor on `device`."""
    t = torch.from_numpy(np.array(bank["vis"][i])).view(torch.bfloat16)
    return t if device is None else t.to(device)


def bank_items(model, bank: dict, splits=("fit", "val")):
    """Bank rows in plan order, with the VLMEvalKit message built for each.

    The message is rebuilt here rather than stored, exactly as `m3_features.py`
    did, so the teacher sees the same prompt the accuracy run will.
    """
    import common
    want = [i for i, s in enumerate(bank["split"]) if s in splits]
    cache, items = {}, []
    for i in want:
        ds, idx = str(bank["ds"][i]), int(bank["idx"][i])
        if ds not in cache:
            cache[ds] = common.build_dataset(ds)
        dataset = cache[ds]
        model.set_dump_image(dataset.dump_image)
        row = dataset.data.iloc[idx]
        items.append(dict(key=str(bank["key"][i]), ds=ds, idx=idx,
                          split=str(bank["split"][i]), bank_row=i, row=row,
                          msg=common.build_message(model, dataset, ds, row)))
    return items


# ===========================================================================
# features -- THE function, called by the offline builder and by the live
# pruner, so there is no train/serve skew by construction
# ===========================================================================
def _neighbourhood(field: torch.Tensor, grid: int = GRID):
    """3x3 neighbourhood mean/max/std of a (N,) map, self excluded."""
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


def safe_features(g: dict, vis: torch.Tensor, s0: torch.Tensor,
                  grid: int = GRID) -> torch.Tensor:
    """(N, N_FEAT) deployable features for every visual token.

    `g` is `TimedEADPPruner.last_gpu` (device tensors, see `keep_gpu`); `vis`
    is the (N, D) post-merger feature the vision tower already produced; `s0`
    is the retained index set (any order).  Nothing here touches the LLM.
    """
    dev = vis.device
    imp = g["importance"].reshape(-1).float()
    fused = g["fused"].reshape(-1).float()
    glob = g["global_sim"].reshape(-1).float()
    loc = g["local_sim"].reshape(-1).float()
    imp_pre = g["importance_post_smooth"].reshape(-1).float()
    n = imp.shape[0]
    sim = g["sim_matrix"].reshape(n, n).float()
    neg = torch.tensor(float("-inf"), device=dev)

    keep = torch.zeros(n, dtype=torch.bool, device=dev)
    keep[s0.to(torch.long)] = True

    # ---- redundancy inside the retained set, self excluded -----------------
    # The column position of a kept row comes from the mask itself, so an
    # UNSORTED s0 is handled correctly.
    sim_k = sim.masked_fill(~keep.unsqueeze(0), neg).clone()
    rows_self = keep.nonzero(as_tuple=True)[0]
    col = keep.cumsum(0)[rows_self] - 1
    sim_k[rows_self, col] = neg
    red_s0 = sim_k.max(dim=1).values
    k8 = min(8, int(keep.sum()) - 1)
    red_s0_top8 = sim_k.topk(max(k8, 1), dim=1).values.mean(dim=1)

    sim_ns = sim.clone()
    sim_ns.fill_diagonal_(neg)
    red_all = sim_ns.max(dim=1).values
    nn_drop = sim_ns.masked_fill(keep.unsqueeze(0), neg).max(dim=1).values
    nn_drop = torch.where(torch.isfinite(nn_drop), nn_drop,
                          torch.zeros_like(nn_drop))

    # ---- reconstruction error against the retained neighbours --------------
    v = vis.float()
    vn = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    a_idx = keep.nonzero(as_tuple=True)[0]
    cos_to_a = vn @ vn[a_idx].t()                        # (N, |A|)
    kk = min(4, a_idx.numel())
    nn_pos = torch.topk(cos_to_a, kk, dim=1).indices     # (N, kk)
    A = vn[a_idx][nn_pos]                                # (N, kk, D)
    gram = A @ A.transpose(1, 2)                         # (N, kk, kk)
    rhs = (A @ vn.unsqueeze(-1)).squeeze(-1)             # (N, kk)
    gram = gram + 1e-4 * torch.eye(kk, device=dev, dtype=gram.dtype)
    coef = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)
    proj = (coef.unsqueeze(-1) * A).sum(dim=1)           # (N, D)
    nn4_recon = (vn - proj).norm(dim=-1) / vn.norm(dim=-1).clamp_min(1e-8)

    # ---- local structure ---------------------------------------------------
    nb_mean, nb_max, nb_std = _neighbourhood(imp, grid)
    nb_mean_f, _, _ = _neighbourhood(fused, grid)
    nb_red, _, _ = _neighbourhood(red_s0, grid)
    vg = vn.t().reshape(vn.shape[1], grid, grid)
    pad = F.pad(vg, (1, 1, 1, 1), mode="replicate")
    acc = torch.zeros(grid, grid, device=dev, dtype=vg.dtype)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            acc = acc + (pad[:, dy:dy + grid, dx:dx + grid] * vg).sum(0)
    nb_cos = (acc / 8.0).reshape(-1)

    # ---- geometry and scale ------------------------------------------------
    yy = torch.div(torch.arange(n, device=dev), grid, rounding_mode="floor")
    xx = torch.arange(n, device=dev) % grid
    x = (xx.float() / (grid - 1.0)) * 2 - 1
    y = (yy.float() / (grid - 1.0)) * 2 - 1
    vis_norm = v.norm(dim=-1)
    rank_imp = torch.argsort(torch.argsort(imp)).float() / (n - 1.0)

    s0c = vn[a_idx].mean(0)
    s0c = s0c / s0c.norm().clamp_min(1e-8)
    t64 = torch.argsort(-imp)[:64]
    t64c = vn[t64].mean(0)
    t64c = t64c / t64c.norm().clamp_min(1e-8)

    # ---- the instruction match, unaggregated -------------------------------
    lsa = g.get("local_sim_all")
    if lsa is None:
        loc_max = loc_top5 = loc_std = loc
    else:
        L = lsa.reshape(n, -1)
        k5 = min(5, L.shape[1])
        loc_max = L.max(dim=1).values
        loc_top5 = L.topk(k5, dim=1).values.mean(dim=1)
        loc_std = L.std(dim=1)

    cols = [imp, imp_pre, fused, glob, loc, loc_max, loc_top5, loc_std,
            red_s0, red_s0_top8, 1.0 - red_s0, nn4_recon, red_all, nn_drop,
            1.0 - nn_drop,
            nb_mean, nb_max, nb_std, nb_mean_f, nb_red, nb_cos,
            x, y, vis_norm, rank_imp,
            (vn @ s0c), (vn @ t64c)]
    X = torch.stack(cols, dim=1)
    assert X.shape == (n, N_FEAT), (X.shape, N_FEAT)
    assert torch.isfinite(X).all(), "non-finite feature"
    return X


# ===========================================================================
# the probe
# ===========================================================================
class SafeRisk(nn.Module):
    """risk_i = P(deleting i changes the answer for the worse).

    Two families, both deliberately small:
        `hand`  a logistic regression (or 1-hidden-layer MLP) on the scalar
                features only.
        `vis`   the same head, with one extra linear view of the post-merger
                vision feature -- the token-local family S2-C5/M3 used, and the
                only part of the input that is not a hand-built scalar.
    The brief asks for exactly this and forbids an architecture search.
    """

    def __init__(self, kind: str = "lr", d_hand: int = N_FEAT,
                 d_vis: int = D_VIS, d_v: int = 64, d_h: int = 64,
                 hidden: int = 0):
        super().__init__()
        self.kind, self.hidden = kind, hidden
        if kind == "lr":
            self.head = nn.Linear(d_hand, 1)
        elif kind == "mlp":
            layers, d = [], d_hand
            for _ in range(max(hidden, 1)):
                layers += [nn.Linear(d, d_h), nn.GELU()]
                d = d_h
            layers += [nn.Linear(d, 1)]
            self.head = nn.Sequential(*layers)
        elif kind == "vis":
            self.proj_v = nn.Linear(d_vis, d_v)
            layers, d = [], d_hand + d_v
            for _ in range(max(hidden, 1)):
                layers += [nn.Linear(d, d_h), nn.GELU()]
                d = d_h
            layers += [nn.Linear(d, 1)]
            self.head = nn.Sequential(*layers)
        else:
            raise KeyError(kind)
        self.d_hand, self.d_vis = d_hand, d_vis

    def forward(self, X: torch.Tensor, V: torch.Tensor = None) -> torch.Tensor:
        if self.kind == "vis":
            assert V is not None, "the vis family needs the vision feature"
            x = torch.cat([X, F.gelu(self.proj_v(V))], dim=-1)
        else:
            x = X
        return self.head(x).squeeze(-1)

    @torch.no_grad()
    def risk(self, X: torch.Tensor, V: torch.Tensor = None) -> torch.Tensor:
        return torch.sigmoid(self.forward(X, V))


# ===========================================================================
# the pruner -- B2, then trim the k lowest-risk retained tokens
# ===========================================================================
class SafeTrimPruner(TimedEADPPruner):
    """B2's selector, then deletion of the k tokens the probe calls safest.

    `trim` describes the arm:
        mode      "probe" | "rule" | "none"
        k         number of tokens deleted
        rule      for mode="rule": "maxred" | "lowimp" | "recon" | "random"
        probe     a SafeRisk instance (mode="probe")
        mu, sd    the feature standardisation fitted on the fit split
        seed      for mode="rule" rule="random"

    `mode="none"` or `k=0` reproduces the incumbent's pruner byte-for-byte and
    is the identity gate: same score, same similarity, same selector, same
    sorted index set.  It is the only arm that may claim B2's numbers.
    """

    def __init__(self, *a, trim: dict = None, **kw):
        super().__init__(*a, **kw)
        self.trim = dict(trim or {})
        self.last_miss = {}
        self.last_miss_events = None

    @torch.no_grad()
    def forward(self, image_features, text_embeds_llm, text_embeds_seq_llm,
                grid_thw):
        trim = self.trim
        mode = trim.get("mode", "none")
        k = int(trim.get("k", 0))
        if mode == "none" or k <= 0:
            self.last_miss = dict(mode=mode, k=k, identity=True, trim_ms=0.0)
            return super().forward(image_features, text_embeds_llm,
                                   text_embeds_seq_llm, grid_thw)

        self.keep_gpu = True
        self.last_gpu = {}
        super().forward(image_features, text_embeds_llm,
                        text_embeds_seq_llm, grid_thw)
        self.keep_gpu = False
        # The window opens AFTER the base pass returns, so `trim_ms` measures
        # only what B2 does not already do (the M2 amendment's double-count,
        # in mirror image -- same discipline as REC's `rec_ms`).
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()

        g = self.last_gpu
        self.last_gpu = {}
        n = image_features.shape[0]
        s0 = torch.sort(g["select_idx"][0].to(torch.long)).values
        vis = g["image_features"].float()

        if mode == "rule":
            drop = self._rule_drop(g, s0, vis, k, trim.get("rule", "maxred"),
                                   trim.get("seed", 0))
            risk = risk_proof = None
        elif mode == "probe":
            X = safe_features(g, vis, s0)
            mu = torch.as_tensor(trim["mu"], device=X.device, dtype=X.dtype)
            sd = torch.as_tensor(trim["sd"], device=X.device, dtype=X.dtype)
            Xn = (X - mu) / sd
            Xs = Xn[s0]
            Vs = vis[s0] if trim.get("kind") == "vis" else None
            r = trim["probe"].risk(Xs, Vs)
            # Ties broken by index, so the arm is a pure function of its inputs
            # (the M4 non-determinism lesson, applied to an argmin).
            order = torch.argsort(r, stable=True)
            drop = s0[order[:k]]
            risk = r
            # A complete proof of the argmin that costs k+1 floats instead of
            # 256: if the largest risk among the dropped tokens is <= the
            # smallest risk among the kept ones, the dropped set IS the k
            # lowest-risk tokens.  Gate `probe_drop_is_argmin_risk` reads these
            # two numbers rather than a stored risk vector, so the gate cannot
            # be vacuous.
            rs = r[order]
            risk_proof = dict(max_dropped=float(rs[k - 1]),
                              min_kept=float(rs[k]) if r.numel() > k else None)
        else:
            raise KeyError(mode)

        keep_mask = torch.ones(n, dtype=torch.bool, device=vis.device)
        keep_mask[drop] = False
        A = s0[keep_mask[s0]]
        assert A.numel() == BUDGET - k, (A.numel(), BUDGET - k)

        ev1.record()
        self.last_miss_events = (ev0, ev1)
        self.last_miss = dict(
            mode=mode, k=k, requested_k=k, identity=False,
            rule=trim.get("rule"), kind=trim.get("kind"),
            n_dropped=int(drop.numel()), n_tokens=int(A.numel()),
            s0_idx=s0.detach().cpu().tolist(),
            drop_idx=drop.detach().cpu().tolist(),
            a_idx=A.detach().cpu().tolist(),
            risk_proof=(risk_proof if risk is not None else None))
        return vis[A].to(image_features.dtype), [int(A.numel())]

    # ---------------------------------------------------------------- rules --
    @staticmethod
    def _rule_drop(g, s0, vis, k, rule, seed):
        """The training-free trimming rules, on the device, no search."""
        n = vis.shape[0]
        if rule == "random":
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            perm = torch.randperm(s0.numel(), generator=gen)
            return s0[perm[:k].to(s0.device)]
        if rule == "lowimp":
            imp = g["importance"].reshape(-1).float()
            return s0[torch.argsort(imp[s0], stable=True)[:k]]
        X = safe_features(g, vis, s0)
        if rule == "maxred":
            col = X[:, RED_COL]
            return s0[torch.argsort(-col, stable=True)[:k]]
        if rule == "recon":
            col = X[:, RECON_COL]
            return s0[torch.argsort(col, stable=True)[:k]]
        raise KeyError(rule)

    # ------------------------------------------------------------ readout --
    def read_miss_ms(self) -> float:
        """Consume the pending event pair (one sync).  Named as in M3/M4 so the
        M2 accuracy and paired-perf harnesses read this window unchanged."""
        if self.last_miss_events is None:
            return float(self.last_miss.get("trim_ms") or 0.0)
        a, b = self.last_miss_events
        self.last_miss_events = None
        torch.cuda.synchronize()
        ms = float(a.elapsed_time(b))
        self.last_miss["trim_ms"] = ms
        return ms


def install_trim(eng, model, trim: dict, selector: str = BASE_SELECTOR):
    """Install the trimming pruner on BOTH hooks of an engine.

    `model.pruner` must be rebound as well as `eng.pruner`: the engine's prellm
    branch uses the latter, but the vlmeval wrapper's own path uses the former,
    so a half-install would let a wrapper-side call silently evaluate plain B2
    while the log says SafeTrim.
    """
    old = eng.pruner
    dev = next(model.model.parameters()).device
    pr = SafeTrimPruner(
        visual_token_num=old.visual_token_num, alpha=old.alpha, beta=old.beta,
        visual_dim=old.visual_dim, spatial_merge_size=old.spatial_merge_size,
        selector=selector, capture=False, trim=trim,
    ).to(dev)
    # The probe is fitted on CPU in `m5_probe.py` and scored against tensors
    # that live on the device, so it travels with the pruner rather than being
    # moved by hand at every call site.
    if pr.trim.get("probe") is not None:
        pr.trim["probe"] = pr.trim["probe"].to(dev).eval()
    pr.eval()
    pr.sim_mode = "rebound"
    eng.pruner = pr
    model.pruner = pr
    assert eng.pruner is model.pruner, "dual-hook install failed"
    return eng
