"""
M14 solvers -- accelerate the facility-location computation, never replace it.

Every solver here receives the identical (importance, sim) pair the official
greedy receives and returns (select_idx (1, K), info dict).  All keep the
facility objective's three ingredients: relevance weighting w_j, coverage
max_{u in S} S_uj with diminishing returns, greedy argmax.

Notation:  gain(u | S) = sum_j w_j [S_uj - c_j]_+ ,  c_j = max_{v in S} S_vj.

  screen_verify   greedy with a CELF shortlist + upper-bound verification loop.
                  Data-dependent widening; exact by construction (syncs once
                  per verification round).
  screen_fixed    fixed R widening rounds, no data-dependent branching, no
                  sync (CUDA-graph friendly).  Winner recall < 1 in principle;
                  the offline eval measures the violation rate.
  sparse_graph    greedy on the sparse top-r kNN coverage graph.
  demand_coreset  greedy on M weighted demand prototypes (k-means on the
                  demand coverage profile); candidate pool untouched.
  sparse_screen   PART 5 combination: sparse graph + shortlist.

Sparse-graph note: the top-r neighbours of a demand j (largest S_uj over u)
are the same set whether computed per demand (column top-r) or per candidate
(row top-r).  We use the row form, vals_c/lists_c (N, r), because then every
candidate's sparse gain is a pure gather + (N, r) reduction -- no scatter, no
atomics.  Coverage of demand j by any candidate outside its top-r set is
bounded by the r-th largest S_uj, which is the truncation error we measure.
"""
from __future__ import annotations

import torch


NEG = -1e9


def _dense_gain(sim_rows: torch.Tensor, cur_max: torch.Tensor,
                importance: torch.Tensor) -> torch.Tensor:
    """sim_rows: (L, N); cur_max: (N,); importance: (N,) -> (L,) exact gains."""
    return torch.clamp(sim_rows - cur_max.unsqueeze(0), min=0.0) @ importance


# ===========================================================================
# PART 2 -- screen -> verify
# ===========================================================================
def select_screen_verify(importance, sim, token_num, shortlist=32):
    """
    Facility greedy with a CELF-style shortlist and a verification loop.

    Per step:
      1. shortlist = top-L unselected by STALE gain (a valid upper bound: it
         is the exact gain from the last evaluation, stale by submodularity).
      2. exact gains for the shortlist only (gather L rows of sim).
      3. accept the shortlist argmax iff its exact gain >= the largest stale
         gain among unselected candidates outside the shortlist; otherwise
         double the shortlist and re-verify.

    Returns the identical selection to the official greedy.
    """
    B, N = importance.shape
    assert B == 1, "m14 solvers operate per image (B=1)"
    w = importance[0]
    S = sim[0]
    T = min(token_num, N)
    device = S.device

    idx = torch.empty(1, T, dtype=torch.long, device=device)
    cur_max = torch.zeros(N, device=device)
    stale = S @ w                       # exact gains at c = 0
    selected = torch.zeros(N, dtype=torch.bool, device=device)

    rounds = torch.zeros(T, dtype=torch.long)
    first_hit = torch.zeros(T, dtype=torch.bool)
    L = min(shortlist, N)

    for t in range(T):
        n_round = 0
        width = L
        while True:
            n_round += 1
            key = stale.masked_fill(selected, NEG)
            top = torch.topk(key, min(width, N)).indices      # (L',)
            gain = _dense_gain(S[top], cur_max, w)            # (L',)
            gain = gain.masked_fill(selected[top], NEG)       # topk padding guard
            g_star, pos = gain.max(dim=0)
            cand = top[pos]
            in_short = torch.zeros(N, dtype=torch.bool, device=device)
            in_short[top] = True
            rest_max = key.masked_fill(in_short, NEG).max()
            if bool(g_star >= rest_max):
                if n_round == 1:
                    first_hit[t] = True
                rounds[t] = n_round
                break
            width *= 2
            if width > N:
                # degenerate: fall back to one full exact pass
                gain_all = _dense_gain(S, cur_max, w).masked_fill(selected, NEG)
                cand = gain_all.argmax()
                stale.copy_(gain_all)
                rounds[t] = n_round
                break

        idx[0, t] = cand
        cur_max = torch.maximum(cur_max, S[cand])
        selected[cand] = True
        stale[top] = gain                # refresh evaluated candidates (CELF)
        stale[cand] = NEG                # selected; never reconsidered

    info = {"shortlist_rounds": rounds, "first_shortlist_hit": first_hit}
    return idx, info


def select_screen_fixed(importance, sim, token_num, shortlist=32, rounds=2):
    """
    Fixed-structure screen: top-L stale screen, exact gains on the shortlist,
    widen by another L stale candidates (rounds - 1) times, take the exact
    argmax over the union.  No data-dependent branching, no host sync.

    Exactness condition per step: the winner's exact gain must be >= every
    un-evaluated candidate's stale upper bound.  The offline eval measures how
    often this holds (verify_violation) instead of enforcing it with a sync.
    """
    B, N = importance.shape
    w = importance[0]
    S = sim[0]
    T = min(token_num, N)
    device = S.device

    idx = torch.empty(1, T, dtype=torch.long, device=device)
    cur_max = torch.zeros(N, device=device)
    stale = S @ w
    selected = torch.zeros(N, dtype=torch.bool, device=device)

    violation = torch.zeros(T, dtype=torch.bool)
    L = min(shortlist, N)

    for t in range(T):
        key = stale.masked_fill(selected, NEG)
        width = min(shortlist * rounds, N)
        top = torch.topk(key, width).indices
        gain = _dense_gain(S[top], cur_max, w)
        gain = gain.masked_fill(selected[top], NEG)
        pos = gain.argmax()
        cand = top[pos]
        idx[0, t] = cand
        # exactness check (recorded, not enforced -- no sync branch on it)
        g_star = gain[pos]
        in_short = torch.zeros(N, dtype=torch.bool, device=device)
        in_short[top] = True
        rest_max = key.masked_fill(in_short, NEG).max()
        if bool(g_star < rest_max):
            violation[t] = True
        cur_max = torch.maximum(cur_max, S[cand])
        selected[cand] = True
        stale[top] = gain
        stale[cand] = NEG

    return idx, {"verify_violation": violation}


# ===========================================================================
# PART 3 -- sparse coverage graph (per-demand top-r kNN)
# ===========================================================================
def build_sparse_graph(sim: torch.Tensor, r: int):
    """Per-demand top-r covering candidates.  sim: (N, N).

    Returns (lists_d (N, r) long, vals_d (N, r) float): lists_d[j] = the r
    candidates that cover demand j best, vals_d[j] = the matching S_uj.

    Truncation direction matters: coverage is truncated per DEMAND (column
    top-r), so every demand keeps its best r coverers and coverage by any
    candidate outside the list is bounded by vals_d[j].min().  A per-candidate
    (row) truncation would instead silence a candidate's coverage of any
    demand outside its own best-r row, which is not the facility objective.
    """
    vals, lists = torch.topk(sim, r, dim=0)      # (r, N): top-r over candidates
    return lists.t().contiguous(), vals.t().contiguous()


def sparse_objective(lists_d, vals_d, importance, sel):
    """F_r(S): coverage restricted to the top-r graph, for diagnostics."""
    w = importance[0] if importance.dim() == 2 else importance
    member = torch.isin(lists_d, sel)            # (N, r)
    cov = torch.where(member, vals_d, torch.zeros_like(vals_d)).max(dim=1).values
    return (cov * w).sum()


def select_sparse_graph(importance, sim, token_num, r=32):
    """
    Greedy on the sparse top-r coverage graph.  Objective:
        F_r(S) = sum_j w_j max_{u in S, u in lists_d[j]} S_uj
    Relevance weighting, diminishing returns and greedy argmax unchanged.
    """
    B, N = importance.shape
    w = importance[0]
    S = sim[0]
    T = min(token_num, N)
    device = S.device

    lists_d, vals_d = build_sparse_graph(S, r)               # (N, r)
    flat_lists = lists_d.reshape(-1)
    wd = w.unsqueeze(1).expand(N, r).reshape(-1)             # demand weight/pair

    idx = torch.empty(1, T, dtype=torch.long, device=device)
    cur_max = torch.zeros(N, device=device)                  # per demand
    selected = torch.zeros(N, dtype=torch.bool, device=device)
    gain = torch.empty(N, device=device)

    for t in range(T):
        c_pair = cur_max.unsqueeze(1).expand(N, r)           # view, no copy
        g = torch.clamp(vals_d - c_pair, min=0.0) * wd       # (N, r)
        gain.zero_()
        gain.index_add_(0, flat_lists, g.reshape(-1))
        gain = gain.masked_fill(selected, NEG)
        cand = gain.argmax()
        idx[0, t] = cand
        # coverage update: only demands whose list contains cand can improve
        member = (lists_d == cand).any(dim=1)
        row = S[cand]
        cur_max = torch.where(member & (row > cur_max), row, cur_max)
        selected[cand] = True

    return idx, {}


# ===========================================================================
# PART 4 -- demand-side coreset (weighted demand prototypes)
# ===========================================================================
def _kmeans(points: torch.Tensor, M: int, iters: int = 15):
    """Deterministic farthest-first init + Lloyd iterations.  points: (N, d)."""
    N, d = points.shape
    M = min(M, N)
    centers = torch.empty(M, d, dtype=points.dtype, device=points.device)
    first = int(torch.argmax(points.norm(dim=1)))
    centers[0] = points[first]
    d2 = ((points - centers[0]) ** 2).sum(dim=1)
    for m in range(1, M):
        centers[m] = points[int(d2.argmax())]
        d2 = torch.minimum(d2, ((points - centers[m]) ** 2).sum(dim=1))
    assign = torch.zeros(N, dtype=torch.long, device=points.device)
    for _ in range(iters):
        d = torch.cdist(points, centers)                     # (N, M)
        assign = d.argmin(dim=1)
        # one-hot mean update, no python loop over clusters
        onehot = torch.zeros(N, M, device=points.device)
        onehot[torch.arange(N, device=points.device), assign] = 1.0
        counts = onehot.sum(dim=0).clamp(min=1.0).unsqueeze(1)
        centers = (onehot.t() @ points) / counts
    return assign


def select_demand_coreset(importance, sim, token_num, M=128, iters=15):
    """
    Facility greedy on M weighted demand prototypes.  The candidate pool is
    untouched -- every original token stays selectable.  Demands are clustered
    by their coverage profile (their column of S -- exactly the quantity the
    greedy reads); each prototype's coverage by candidate u is the mean of
    S_uj over its cluster, and its weight is the cluster's demand mass.
    """
    B, N = importance.shape
    w = importance[0]
    S = sim[0]
    T = min(token_num, N)
    device = S.device

    assign = _kmeans(S.t().contiguous(), M, iters=iters)         # (N,)
    M_eff = int(assign.max().item()) + 1
    P = torch.zeros(M_eff, N, device=device)
    P.index_add_(0, assign, S.t())                               # (M, N) sums
    counts = torch.bincount(assign, minlength=M_eff).clamp(min=1).float()
    P = P / counts.unsqueeze(1)
    W = counts / counts.sum() * w.sum()      # total demand mass preserved

    idx = torch.empty(1, T, dtype=torch.long, device=device)
    cur_max = torch.zeros(M_eff, device=device)
    selected = torch.zeros(N, dtype=torch.bool, device=device)

    for t in range(T):
        gain = torch.clamp(P - cur_max.unsqueeze(1), min=0.0) @ W    # (N,)
        gain = gain.masked_fill(selected, NEG)
        cand = gain.argmax()
        idx[0, t] = cand
        cur_max = torch.maximum(cur_max, P[cand])
        selected[cand] = True

    return idx, {}


# ===========================================================================
# PART 5 -- combination: sparse graph + shortlist
# ===========================================================================
def select_sparse_screen(importance, sim, token_num, r=32, shortlist=32,
                         rounds=2):
    """
    Sparse-graph greedy with a stale-gain shortlist: per step only
    shortlist*rounds candidates get their sparse exact gain recomputed
    ((L, r) work); the rest keep stale sparse upper bounds.  No host sync.
    """
    B, N = importance.shape
    w = importance[0]
    S = sim[0]
    T = min(token_num, N)
    device = S.device

    lists_c, vals_c = build_sparse_graph(S, r)           # (N, r)
    wc = w[lists_c]

    idx = torch.empty(1, T, dtype=torch.long, device=device)
    cur_max = torch.zeros(N, device=device)
    selected = torch.zeros(N, dtype=torch.bool, device=device)

    # stale sparse gains at c = 0
    stale = (torch.clamp(vals_c, min=0.0) * wc).sum(dim=1)

    for t in range(T):
        key = stale.masked_fill(selected, NEG)
        width = min(shortlist * rounds, N)
        top = torch.topk(key, width).indices
        sub_l = lists_c[top]                              # (L, r)
        sub_v = torch.gather(vals_c, 0,
                             top.unsqueeze(1).expand(-1, r))
        cc = cur_max[sub_l]                               # (L, r)
        g = torch.clamp(sub_v - cc, min=0.0) * wc[top]    # (L, r)
        gain = g.sum(dim=1)                               # (L,)
        gain = gain.masked_fill(selected[top], NEG)
        pos = gain.argmax()
        cand = top[pos]
        idx[0, t] = cand
        lj = lists_c[cand]
        cur_max.scatter_reduce_(0, lj, vals_c[cand], reduce="amax",
                                include_self=True)
        selected[cand] = True
        stale[top] = gain
        stale[cand] = NEG

    return idx, {}


SOLVERS = {
    "screen_verify": select_screen_verify,
    "screen_fixed": select_screen_fixed,
    "sparse_graph": select_sparse_graph,
    "demand_coreset": select_demand_coreset,
    "sparse_screen": select_sparse_screen,
}
