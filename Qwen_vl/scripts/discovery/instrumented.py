"""
Instrumented EADP pruner + pluggable final-selection operators.

Design rule: the **importance scoring half is never touched**. Every stage up to
and including score polarization calls the official helpers in ``model.pruner``
verbatim, so any difference we measure is attributable to the final selection
operator alone. Only ``_greed_select_impl`` is swapped.

The official implementation in ``model/pruner.py`` is not modified; we
re-assemble its stages here with CUDA event brackets in between.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from model.pruner import (  # noqa: E402  (sys.path set by common.py)
    _entropy_filter_impl,
    _greed_select_impl,
    _local_aggregation_impl,
    _sim_visual_impl,
    _spatial_smoothing_impl,
    VisualTokenPruner,
)

from common import CudaTimer  # noqa: E402

# ---------------------------------------------------------------------------
# Paper-style stage names used in the timing dict (cf. Table 14 of the paper).
# ---------------------------------------------------------------------------
STAGE_PREP = "prep_norm"
STAGE_GLOBAL = "global_guidance"
STAGE_DENSE = "dense_guidance"
STAGE_FUSION = "score_fusion"
STAGE_SMOOTH = "smoothing"
STAGE_POLAR = "polarization"
STAGE_SELECT = "facility_location"

PAPER_STAGE_ORDER = [
    STAGE_GLOBAL,
    STAGE_DENSE,
    STAGE_FUSION,
    STAGE_SMOOTH,
    STAGE_POLAR,
    STAGE_SELECT,
]
ALL_STAGE_ORDER = [STAGE_PREP] + PAPER_STAGE_ORDER


# ===========================================================================
# Final-selection operators
# ===========================================================================
def _tie_break_topk(importance: torch.Tensor, k: int) -> torch.Tensor:
    """Deterministic top-k (largest=True, stable ordering by index)."""
    return torch.topk(importance, k=k, dim=1, largest=True, sorted=True).indices


def select_topk(importance: torch.Tensor, sim_matrix: torch.Tensor, token_num: int):
    """(B) Same EADP importance score, final selection = plain Top-K."""
    B, N = importance.shape
    T = min(token_num, N)
    idx = _tie_break_topk(importance, T)
    return idx, {}


def select_facility(importance: torch.Tensor, sim_matrix: torch.Tensor, token_num: int):
    """(A) Official EADP facility-location greedy selection."""
    idx, quotas = _greed_select_impl(importance, sim_matrix, token_num)
    return idx, {"quotas": quotas}


def select_facility_fast(importance: torch.Tensor, sim_matrix: torch.Tensor, token_num: int):
    """
    Vectorised re-formulation of the SAME greedy objective.

    The official loop materialises three (B, N, N) temporaries per step
    (``torch.maximum``, the multiply, the sum-reduction). We compute the
    identical marginal gains with a single BLAS mat-vec plus in-place ops, which
    yields a bit-comparable selection with far less memory traffic.
    """
    B, N = importance.shape
    device = importance.device
    T = min(token_num, N)
    idx = torch.zeros(B, T, dtype=torch.long, device=device)
    cur_max = torch.zeros(B, N, device=device)
    selected = torch.zeros(B, N, dtype=torch.bool, device=device)
    for t in range(T):
        gain = torch.clamp(sim_matrix - cur_max.unsqueeze(1), min=0.0)
        gain = torch.matmul(gain, importance.unsqueeze(-1)).squeeze(-1)  # (B, N)
        gain = gain.masked_fill(selected, -1e9)
        best = torch.argmax(gain, dim=1)
        idx[:, t] = best
        rows = torch.arange(B, device=device)
        cur_max = torch.maximum(cur_max, sim_matrix[rows, best, :])
        selected[rows, best] = True
    return idx, {}


def select_lazy_greedy(importance: torch.Tensor, sim_matrix: torch.Tensor, token_num: int):
    """
    CELF / lazy greedy for the SAME facility-location objective.

    Marginal gains are non-increasing under submodularity, so a max-heap of
    stale gains plus lazy re-evaluation returns the *identical* greedy set while
    evaluating only O(1) candidate gains per step instead of all N.
    """
    B, N = importance.shape
    device = importance.device
    T = min(token_num, N)
    idx = torch.zeros(B, T, dtype=torch.long, device=device)
    for b in range(B):
        s = importance[b]
        S = sim_matrix[b]
        cur_max = torch.zeros(N, device=device)
        selected = torch.zeros(N, dtype=torch.bool, device=device)
        gains = torch.matmul(S, s)  # cur_max = 0 -> sum_j s_j * S[u, j]
        heap = [(-float(gains[i]), i) for i in range(N)]
        import heapq

        heapq.heapify(heap)
        for t in range(T):
            while True:
                neg_g, u = heapq.heappop(heap)
                if bool(selected[u]):
                    continue
                true_g = float((s * torch.clamp(S[u] - cur_max, min=0.0)).sum())
                if not heap or -heap[0][0] <= true_g + 1e-9:
                    break
                heapq.heappush(heap, (-true_g, u))
            idx[b, t] = u
            selected[u] = True
            cur_max = torch.maximum(cur_max, S[u])
    return idx, {}


def select_stochastic_greedy(
    importance: torch.Tensor,
    sim_matrix: torch.Tensor,
    token_num: int,
    eps: float = 0.1,
    generator: torch.Generator = None,
):
    """
    Stochastic greedy (Mirzasoleiman et al. 2015) on the same objective.

    Each round draws a random candidate subset of size
    ``n/T * log(1/eps)``, giving a (1 - 1/e - eps) guarantee at a fraction of
    the per-round cost of exact greedy.
    """
    B, N = importance.shape
    device = importance.device
    T = min(token_num, N)
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(1234)
    sample_size = max(1, int(math.ceil(N / max(T, 1) * math.log(1.0 / max(eps, 1e-6)))))
    sample_size = min(sample_size, N)
    idx = torch.zeros(B, T, dtype=torch.long, device=device)
    for b in range(B):
        s = importance[b]
        S = sim_matrix[b]
        cur_max = torch.zeros(N, device=device)
        selected = torch.zeros(N, dtype=torch.bool, device=device)
        for t in range(T):
            perm = torch.randperm(N, device=device, generator=generator)
            cand = perm[:sample_size]
            gain = torch.clamp(S[cand] - cur_max, min=0.0) @ s
            gain = gain.masked_fill(selected[cand], -1e9)
            u = cand[int(torch.argmax(gain))]
            idx[b, t] = u
            selected[u] = True
            cur_max = torch.maximum(cur_max, S[u])
    return idx, {}


def select_block_greedy(
    importance: torch.Tensor,
    sim_matrix: torch.Tensor,
    token_num: int,
    block: int = 32,
):
    """
    Block-parallel greedy on the SAME facility-location objective.

    Exact greedy needs T *sequential* iterations, each launching ~6 kernels over
    an (N, N) tensor. Here each super-step selects `block` tokens at once from a
    single marginal-gain evaluation and updates the coverage state once, so the
    number of sequential steps drops from T to T/block.

    ``block == 1`` reproduces the official greedy selection exactly.
    ``block == T`` is a single coverage-weighted shot.
    """
    B, N = importance.shape
    device = importance.device
    T = min(token_num, N)
    idx = torch.zeros(B, T, dtype=torch.long, device=device)
    cur_max = torch.zeros(B, N, device=device)
    selected = torch.zeros(B, N, dtype=torch.bool, device=device)
    rows = torch.arange(B, device=device)
    filled = 0
    while filled < T:
        take = min(block, T - filled)
        gain = torch.clamp(sim_matrix - cur_max.unsqueeze(1), min=0.0)
        gain = torch.matmul(gain, importance.unsqueeze(-1)).squeeze(-1)
        gain = gain.masked_fill(selected, -1e9)
        top = torch.topk(gain, k=take, dim=1).indices            # (B, take)
        idx[:, filled:filled + take] = top
        block_sim = sim_matrix[rows.unsqueeze(1), top]           # (B, take, N)
        cur_max = torch.maximum(cur_max, block_sim.max(dim=1).values)
        selected[rows.unsqueeze(1), top] = True
        filled += take
    return idx, {}


def select_grid(
    importance: torch.Tensor,
    sim_matrix: torch.Tensor,
    token_num: int,
    grid: int = 32,
):
    """
    One-shot stratified selection: partition the token grid into ~T cells and
    keep the most important token in each.

    This buys *coverage* directly -- every region of the image is represented by
    construction -- with a handful of kernels and **no sequential loop at all**,
    unlike the greedy coverage objective. Fully vectorised: no Python iteration
    over cells.

    Motivation: `farthest` (pure diversity, importance used only for the seed)
    loses only 3.6 points versus facility location, which suggests most of the
    objective's value is the coverage structure rather than the importance
    weighting inside it. If a stratified snapshot captures that structure, the
    whole O(T) sequential loop may be unnecessary.
    """
    B, N = importance.shape
    device = importance.device
    T = min(token_num, N)
    side = int(round(grid))
    # cell grid dims: rows x cols >= T, as square as possible
    rows = int(math.ceil(math.sqrt(T)))
    cols = int(math.ceil(T / rows))

    out = torch.zeros(B, T, dtype=torch.long, device=device)
    for b in range(B):
        imp = importance[b]
        idx = torch.arange(N, device=device)
        r = (idx // side).clamp(max=side - 1)
        c = (idx % side).clamp(max=side - 1)
        cell = (r * cols // side).clamp(max=rows - 1) * cols + (c * cols // side).clamp(max=cols - 1)
        n_cells = int(cell.max().item()) + 1
        neg = torch.full((n_cells,), float("-inf"), device=device)
        cell_max = neg.scatter_reduce(0, cell, imp, reduce="amax", include_self=True)
        cand = (imp == cell_max[cell]).nonzero(as_tuple=True)[0]
        first = torch.full((n_cells,), N, dtype=torch.long, device=device)
        first = first.scatter_reduce(0, cell[cand], cand, reduce="amin", include_self=True)
        chosen = first[first < N]
        if chosen.numel() >= T:
            # more cells than budget: keep the most important of them
            order = torch.argsort(imp[chosen], descending=True)[:T]
            chosen = chosen[order]
        else:
            # fewer cells than budget: fill the remainder by global importance
            rest = torch.ones(N, dtype=torch.bool, device=device)
            rest[chosen] = False
            need = T - chosen.numel()
            extra = torch.argsort(imp.masked_fill(~rest, float("-inf")), descending=True)[:need]
            chosen = torch.cat([chosen, extra])
        out[b] = chosen[:T]
    return out, {}


def select_topk_nms(
    importance: torch.Tensor,
    sim_matrix: torch.Tensor,
    token_num: int,
    radius: int = 4,
    grid: int = 32,
):
    """
    (C1) Cheap structured selector: farthest-from-selected greedy with a hard
    spatial exclusion radius. O(T * r^2) instead of O(T * N^2).
    """
    B, N = importance.shape
    device = importance.device
    T = min(token_num, N)
    idx = torch.zeros(B, T, dtype=torch.long, device=device)
    rows = torch.arange(grid, device=device)
    cols = torch.arange(grid, device=device)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    rr = rr.reshape(-1)[:N]
    cc = cc.reshape(-1)[:N]
    for b in range(B):
        imp = importance[b].clone()
        for t in range(T):
            u = int(torch.argmax(imp))
            idx[b, t] = u
            d2 = (rr - rr[u]) ** 2 + (cc - cc[u]) ** 2
            imp = imp.masked_fill(d2 <= radius * radius, -1e9)
    return idx, {}


def select_farthest(
    importance: torch.Tensor,
    sim_matrix: torch.Tensor,
    token_num: int,
    first: str = "top1",
):
    """
    (C2) Diversity-only greedy (farthest-point sampling in feature-similarity
    space): ignore importance after the seed. Cost O(T * N) with a precomputed
    similarity matrix.
    """
    B, N = importance.shape
    device = importance.device
    T = min(token_num, N)
    idx = torch.zeros(B, T, dtype=torch.long, device=device)
    for b in range(B):
        S = sim_matrix[b]
        cur_max = torch.full((N,), -1.0, device=device)
        for t in range(T):
            if t == 0:
                u = int(torch.argmax(importance[b]))
            else:
                u = int(torch.argmin(cur_max))
            idx[b, t] = u
            cur_max = torch.maximum(cur_max, S[u])
    return idx, {}


SELECTORS = {
    "facility": select_facility,              # (A) official
    "facility_fast": select_facility_fast,    # (A') vectorised, same objective
    "lazy_greedy": select_lazy_greedy,        # (A'') CELF, same objective
    "block_greedy": select_block_greedy,      # (C4) same objective, T/block steps
    "grid": select_grid,                      # (C5) one-shot stratified coverage
    "block8": lambda i, s, t: select_block_greedy(i, s, t, block=8),
    "block32": lambda i, s, t: select_block_greedy(i, s, t, block=32),
    "stochastic": select_stochastic_greedy,   # (C3)
    "topk": select_topk,                      # (B)
    "topk_nms": select_topk_nms,              # (C1)
    "farthest": select_farthest,              # (C2)
}


# ===========================================================================
# Instrumented pruner
# ===========================================================================
class TimedEADPPruner(VisualTokenPruner):
    """EADP pruner with per-stage CUDA timing, map capture and swappable selector."""

    def __init__(
        self,
        *args,
        selector: str = "facility",
        capture: bool = False,
        selector_kwargs: dict = None,
        sim_mode: str = "rebound",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if selector not in SELECTORS:
            raise KeyError(f"unknown selector {selector!r}; have {sorted(SELECTORS)}")
        self.selector_name = selector
        self.selector_kwargs = dict(selector_kwargs or {})
        self.sim_mode = sim_mode
        self.capture = capture
        # `capture` keeps CPU copies of the score-map intermediates for offline
        # analysis and costs ~0.5 ms.  `keep_gpu` keeps the SAME tensors on the
        # device they were produced on, so a consumer that only needs
        # reductions (M3's audit features) never pays a host round-trip.  Both
        # are off by default and neither changes a single returned value.
        self.keep_gpu = False
        self.last_gpu = {}
        self.last_timing = {}
        self.last_capture = {}

    def _similarity(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Visual-visual similarity for the selection stage only (scoring untouched).

        ``rebound`` is the official 0.5*(cos+1) projection. It is non-negative,
        but it maps orthogonal pairs to 0.5 and anti-correlated pairs to 0, which
        compresses the usable dynamic range of the coverage gain when LLM-space
        cosines already sit in a narrow high-similarity band.
        """
        if self.sim_mode == "rebound":
            return _sim_visual_impl(feats)
        v = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos = torch.matmul(v.float(), v.float().transpose(1, 2)).clamp(-1.0, 1.0)
        if self.sim_mode == "clamp":
            return cos.clamp(min=0.0)
        if self.sim_mode == "exp":
            return torch.exp(cos / 0.1)
        if self.sim_mode == "softmax":
            return torch.softmax(cos / 0.05, dim=-1)
        raise KeyError(f"unknown sim_mode {self.sim_mode!r}")

    # -- scoring: delegated verbatim to the official implementation ---------
    def _score(self, image_features, text_embeds, text_embeds_seq, grid_h, grid_w, timer):
        ie = image_features.float()
        te = text_embeds.float()
        ts = text_embeds_seq.float()

        timer.start(STAGE_PREP)
        ie = ie / ie.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        te = te / te.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        ts = ts / ts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        timer.stop(STAGE_PREP)

        timer.start(STAGE_GLOBAL)
        global_sim = -torch.einsum("bnc,mc->bnm", ie, te)
        timer.stop(STAGE_GLOBAL)

        timer.start(STAGE_DENSE)
        local_sim_all = -torch.einsum("bnc,mlc->bnml", ie, ts)
        filtered_local_sim, top_entropy_vals = _entropy_filter_impl(
            local_sim_all, T=100.0, entropy_keep_ratio=0.2
        )
        local_sim = _local_aggregation_impl(
            filtered_local_sim, top_entropy_vals, M_temp=0.01, strategy="negative_entropy"
        )
        timer.stop(STAGE_DENSE)

        timer.start(STAGE_FUSION)
        text_sim_all = self.alpha * global_sim + (1 - self.alpha) * local_sim
        text_sim = text_sim_all.mean(dim=-1)
        importance = (text_sim - text_sim.min(dim=-1, keepdim=True).values + 1e-6) / (
            text_sim.max(dim=-1, keepdim=True).values
            - text_sim.min(dim=-1, keepdim=True).values
            + 1e-6
        )
        timer.stop(STAGE_FUSION)

        timer.start(STAGE_SMOOTH)
        importance = _spatial_smoothing_impl(importance, grid_h, grid_w, kernel_size=3, sigma=1.0)
        timer.stop(STAGE_SMOOTH)
        pre_polar = importance

        timer.start(STAGE_POLAR)
        importance = importance ** self.beta
        timer.stop(STAGE_POLAR)

        if self.keep_gpu:
            self.last_gpu.update(
                global_sim=global_sim, local_sim=local_sim, fused=text_sim,
                importance=importance, importance_post_smooth=pre_polar,
                # The RAW per-instruction-token similarity, before the entropy
                # filter and the weighted mean that collapse it into
                # `local_sim`.  M3's audit reads it to ask a question EADP's
                # aggregate cannot: does this visual token match SOME single
                # instruction token sharply, rather than many of them mildly?
                local_sim_all=local_sim_all,
            )

        if self.capture:
            # text-side entropy, exactly as used by the entropy filter
            sim_probs = torch.softmax(local_sim_all * 100.0, dim=1)
            eps = 1e-12
            text_entropy = -torch.sum(sim_probs * torch.log(sim_probs + eps), dim=1)
            self.last_capture.update(
                text_entropy=text_entropy.detach().float().cpu(),
                global_sim=global_sim.detach().float().cpu(),
                local_sim=local_sim.detach().float().cpu(),
                fused=text_sim.detach().float().cpu(),
                importance_pre_smooth=text_sim.detach().float().cpu(),
                importance_post_smooth=pre_polar.detach().float().cpu(),
                importance=importance.detach().float().cpu(),
                top_entropy_vals=top_entropy_vals.detach().float().cpu(),
            )
        return importance

    @torch.no_grad()
    def forward(self, image_features, text_embeds_llm, text_embeds_seq_llm, grid_thw):
        device = image_features.device
        sms = self.spatial_merge_size
        split_sizes = (grid_thw.prod(-1) // (sms ** 2)).tolist()
        image_features_list = torch.split(image_features, split_sizes, dim=0)

        pruned_features_list = []
        pruned_split_sizes = []
        timer = CudaTimer()
        self.last_capture = {}

        for img_idx, img_feats in enumerate(image_features_list):
            N_i = img_feats.shape[0]
            t_i, h_i, w_i = grid_thw[img_idx].tolist()
            grid_h = int(h_i) // sms
            grid_w = int(w_i) // sms

            token_num = min(self.visual_token_num, N_i)
            if token_num >= N_i:
                pruned_features_list.append(img_feats)
                pruned_split_sizes.append(N_i)
                continue

            img_feats_batch = img_feats.unsqueeze(0)
            text_emb = text_embeds_llm[img_idx : img_idx + 1]
            text_emb_seq = text_embeds_seq_llm[img_idx : img_idx + 1]

            importance = self._score(
                img_feats_batch, text_emb, text_emb_seq, grid_h, grid_w, timer
            )

            timer.start(STAGE_SELECT)
            sim_matrix = self._similarity(img_feats_batch)
            select_idx, extra = SELECTORS[self.selector_name](
                importance, sim_matrix, token_num, **self.selector_kwargs
            )
            timer.stop(STAGE_SELECT)

            if self.capture:
                self.last_capture.update(
                    sim_matrix=sim_matrix.detach().float().cpu(),
                    select_idx=select_idx.detach().cpu(),
                    selector=self.selector_name,
                )
            if self.keep_gpu:
                self.last_gpu.update(sim_matrix=sim_matrix.detach(),
                                     select_idx=select_idx.detach(),
                                     image_features=img_feats.detach())

            select_idx_sorted = select_idx[0].sort().values
            pruned_feats = img_feats[select_idx_sorted]
            pruned_features_list.append(pruned_feats)
            pruned_split_sizes.append(pruned_feats.shape[0])

        self.last_timing = timer.finish()
        pruned_features = torch.cat(pruned_features_list, dim=0)
        return pruned_features, pruned_split_sizes


def attach_pruner(model, selector: str = "facility", capture: bool = False, **selector_kwargs):
    """Swap the model's pruner for an instrumented one with the same geometry."""
    old = model.pruner
    new = TimedEADPPruner(
        visual_token_num=old.visual_token_num,
        alpha=old.alpha,
        beta=old.beta,
        visual_dim=old.visual_dim,
        spatial_merge_size=old.spatial_merge_size,
        selector=selector,
        capture=capture,
        selector_kwargs=selector_kwargs,
    ).to(next(model.model.parameters()).device)
    new.eval()
    model.pruner = new
    return new
