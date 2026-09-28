"""
SAGE -- Set-Conditioned Answer-Gain Exchange. Shared module.

The deployability contract, inherited from M3-M9 and binding: forward-only,
pre-LLM, no gradient, no backward, no teacher at inference, no generation
feedback. Gold answers label OFFLINE fit/val/hindsight only; at deployment the
critic sees the vision tower's post-merger features and the instruction
embedding the incumbent's pruner already computes, and nothing else.

Every pre-registered constant lives here (refine-logs/EXPERIMENT_PLAN.md,
decisions 1-13) so a knob cannot drift between the labeling, training and
deployment scripts.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUT_DIR                                        # noqa: E402
from m5_common import BUDGET, D_VIS, GRID, N_VIS                     # noqa: E402

# ---------------------------------------------------------------- frozen knobs
SAGE_SEED = 20260928            # the one draw seed for edge sampling / plans
M_SEEDS = 16                    # removal seeds inside S0 (decision 2)
N_INSERT = 2 * M_SEEDS          # insertion seeds from unretained (decision 2)
G_GRID = (1, 2, 4, 8)           # exchange sizes on validation (decision 1)
EDGE_CAP = 512                  # |E(x)| cap; m*n_insert = 512 is the max
N_SAMPLE_FIT = 8                # labeled edges per fit/val image (decision 5)
N_SAMPLE_CONF = 4               # sampled edges per confirmation image (d. 12)
WIDTHS = (256, 1024)            # critic capacity grid (decision 7)
TAU_GRID = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
TTFT_MULT = 1.10                # decision 9: median TTFT <= 1.10 x B2 median
TRAIN_EPOCHS = 100
TRAIN_BATCH = 256
TRAIN_LR = 1e-3
TRAIN_WD = 1e-4
DS_ORDER = ("TextVQA_VAL", "DocVQA_VAL", "OCRBench")
N_CONF_PER_DS = 240             # confirmation rows per benchmark (decision 11)

D_PHI = 4 * D_VIS + 4           # 3 group means + question mean + 4 cosines
D_UNARY = 8                     # mean/max of imp and cos over each group

# ---------------------------------------------------------------- file names --
F_PLAN = "sage_plan.json"
F_QBANK = "sage_qbank.npz"          # (450, 4096) mean instruction embeddings
F_LABELS = "sage_labels_{split}.json"
F_PHI = "sage_phi_{split}_g{g}.npz"
F_CRITIC = "sage_critic_g{g}_w{w}.pt"
F_CRITIC_SUMMARY = "sage_critic_summary.json"
F_CALIB = "sage_calib.json"
F_VAL_DEPLOY = "sage_valdeploy_g{g}_w{w}.json"
F_CONF = "sage_conf_{arm}.json"


# ===========================================================================
# the edge support E(x) -- pure numpy, deterministic, identical offline and live
# ===========================================================================
def _pctile_in(v: np.ndarray) -> np.ndarray:
    """Percentile in [0, 1] within the given set; ties keep index order."""
    if v.size < 2:
        return np.zeros_like(v, dtype=np.float64)
    return np.argsort(np.argsort(v, kind="stable"), kind="stable") / (v.size - 1.0)


def _grid_pos(idx: np.ndarray) -> tuple:
    idx = np.asarray(idx, dtype=np.int64)
    return idx // GRID, idx % GRID


def _group(seed: int, eligible: np.ndarray, g: int) -> np.ndarray:
    """Seed + its (g-1) nearest eligible grid neighbours, Chebyshev distance,
    distance ties by token index (the source's fixed rule). Returns g indices."""
    if g == 1:
        return np.array([seed], dtype=np.int64)
    eligible = eligible[eligible != seed]          # the seed is not its own neighbour
    ey, ex = _grid_pos(eligible)
    sy, sx = seed // GRID, seed % GRID
    d = np.maximum(np.abs(ey - sy), np.abs(ex - sx))
    order = np.lexsort((eligible, d))          # ties -> token index
    pick = np.concatenate(([seed], eligible[order[:g - 1]]))
    assert np.unique(pick).size == g
    return np.sort(pick)


def build_edge_support(imp: np.ndarray, cos_s0c: np.ndarray, s0: np.ndarray,
                       m: int = M_SEEDS, n_insert: int = N_INSERT,
                       g: int = 1) -> list:
    """E(x) for one image, per decisions 2-4. Deterministic in (imp, cos, s0).

    `imp` high = keep (the B2 score); `cos_s0c` high = keep. Removal seeds are
    the m lowest-imp tokens of S0; insertion seeds the n_insert highest unretained
    tokens by the mean of within-set percentiles of imp and cos_s0c.
    """
    s0 = np.sort(np.asarray(s0, dtype=np.int64))
    assert s0.size == BUDGET
    in_s0 = np.zeros(N_VIS, dtype=bool)
    in_s0[s0] = True

    # removal seeds: lowest imp inside S0, ties by token index
    r_order = np.lexsort((s0, imp[s0]))[:m]
    r_seeds = s0[r_order]
    # insertion seeds: fusion of percentiles over the unretained set
    u = np.where(~in_s0)[0]
    sc = (_pctile_in(imp[u]) + _pctile_in(cos_s0c[u])) / 2.0
    i_order = np.lexsort((u, -sc))[:n_insert]
    i_seeds = u[i_order]

    r_elig = s0
    i_elig = u
    edges, seen = [], set()
    for rs in r_seeds:
        Gm = _group(int(rs), r_elig, g)
        for isd in i_seeds:
            Gp = _group(int(isd), i_elig, g)
            k = (Gm.tobytes(), Gp.tobytes())
            if k in seen:
                continue
            seen.add(k)
            edges.append(dict(minus=Gm.tolist(), plus=Gp.tolist()))
            if len(edges) >= EDGE_CAP:
                return edges
    return edges


def validate_edges(edges: list, s0: np.ndarray, g: int) -> None:
    """G-SET gate: every edge is a feasible equal-budget exchange (eq. 1)."""
    s0 = np.sort(np.asarray(s0, dtype=np.int64))
    s0set = set(s0.tolist())
    for e in edges:
        Gm, Gp = np.asarray(e["minus"]), np.asarray(e["plus"])
        assert Gm.size == Gp.size == g, f"bad group size {Gm.size}/{Gp.size}"
        assert np.unique(Gm).size == g and np.unique(Gp).size == g, "dup in group"
        assert set(Gm.tolist()) <= s0set, "G- not inside S0"
        assert not (set(Gp.tolist()) & s0set), "G+ not outside S0"
        assert not (set(Gm.tolist()) & set(Gp.tolist())), "G- overlaps G+"
        Se = (s0set - set(Gm.tolist())) | set(Gp.tolist())
        assert len(Se) == BUDGET, "|Se| != K"


def sample_edges(edges: list, n: int, seed: int, image_rank: int) -> list:
    """Uniform sample of n edges, deterministic per (seed, image)."""
    if len(edges) <= n:
        return list(edges)
    rng = np.random.default_rng(seed * 1000003 + image_rank)
    pick = rng.permutation(len(edges))[:n]
    return [edges[j] for j in sorted(pick.tolist())]


# ===========================================================================
# critic features -- the SAME function offline (m5_vis rows) and live (the
# pruner's own post-merger features), so there is no train/serve skew
# ===========================================================================
def cos_to_s0c(vis: torch.Tensor, s0: torch.Tensor) -> torch.Tensor:
    """Cosine of every token to the S0 centroid -- the bank's `cos_s0c` column,
    recomputed from raw features (m5_common.safe_features convention)."""
    v = vis.float()
    vn = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    c = vn[s0.to(torch.long)].mean(0)
    c = c / c.norm().clamp_min(1e-8)
    return vn @ c


@torch.no_grad()
def phi_batch(vis: torch.Tensor, q: torch.Tensor, s0: torch.Tensor,
              edges: list) -> torch.Tensor:
    """(E, D_PHI) critic features for one image's edge list (decision 6).

    `vis` (N_VIS, D_VIS) raw post-merger features on any device; `q` (D_VI,)
    mean instruction embedding; `s0` (256,) sorted retained set. Group means are
    means of RAW projected vectors; the four cosine summaries are taken against
    the per-edge survivor centroid on L2-normalised vectors.
    """
    E, g = len(edges), len(edges[0]["minus"])
    dev = vis.device
    vis = vis.float()
    Gm = torch.tensor([e["minus"] for e in edges], dtype=torch.long, device=dev)
    Gp = torch.tensor([e["plus"] for e in edges], dtype=torch.long, device=dev)
    s0 = s0.to(dev)

    sum_s0 = vis[s0].sum(0)                                  # (D,)
    sum_Gm = vis[Gm].sum(1)                                  # (E, D)
    sum_Gp = vis[Gp].sum(1)
    surv_mean = (sum_s0.unsqueeze(0) - sum_Gm) / (BUDGET - g)
    minus_mean = sum_Gm / g
    plus_mean = sum_Gp / g

    vn = vis / vis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    c = surv_mean / surv_mean.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cos_m = torch.einsum("egd,ed->eg", vn[Gm], c)
    cos_p = torch.einsum("egd,ed->eg", vn[Gp], c)

    phi = torch.cat([surv_mean, minus_mean, plus_mean,
                     q.float().unsqueeze(0).expand(E, -1),
                     cos_m.mean(1, keepdim=True), cos_m.max(1, keepdim=True).values,
                     cos_p.mean(1, keepdim=True), cos_p.max(1, keepdim=True).values],
                    dim=1)
    assert phi.shape == (E, D_PHI)
    assert torch.isfinite(phi).all()
    return phi


@torch.no_grad()
def unary_batch(imp: torch.Tensor, cos: torch.Tensor, edges: list) -> torch.Tensor:
    """(E, D_UNARY) the control's features: mean/max of imp and cos over each
    group -- no survivor conditioning, no question embedding."""
    dev = imp.device
    Gm = torch.tensor([e["minus"] for e in edges], dtype=torch.long, device=dev)
    Gp = torch.tensor([e["plus"] for e in edges], dtype=torch.long, device=dev)
    imp, cos = imp.float(), cos.float()
    return torch.cat([imp[Gm].mean(1, keepdim=True), imp[Gm].max(1, keepdim=True).values,
                      imp[Gp].mean(1, keepdim=True), imp[Gp].max(1, keepdim=True).values,
                      cos[Gm].mean(1, keepdim=True), cos[Gm].max(1, keepdim=True).values,
                      cos[Gp].mean(1, keepdim=True), cos[Gp].max(1, keepdim=True).values],
                     dim=1)


class SageCritic(torch.nn.Module):
    """2-hidden-layer GELU MLP (decision 7); the same class serves the UNARY
    control on its 8 unary features."""

    def __init__(self, d_in: int, d_hid: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_hid), torch.nn.GELU(),
            torch.nn.Linear(d_hid, d_hid), torch.nn.GELU(),
            torch.nn.Linear(d_hid, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ===========================================================================
# dump / TTFT helpers (conventions inherited from m6_common / m2_perf_paired)
# ===========================================================================
def dump_json(name: str, obj) -> str:
    import json

    path = os.path.join(OUTPUT_DIR, name)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    print(f"[saved] {name}")
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def load_json(name: str):
    import json

    with open(os.path.join(OUTPUT_DIR, name)) as f:
        return json.load(f)


class TTFT:
    """Wall TTFT, m2_perf_paired's convention: sync -> [prepare + prefill] ->
    sync; the first token sits in the prefill's final logits."""

    def __init__(self):
        self.ms = None

    def __enter__(self):
        import time

        torch.cuda.synchronize()
        self._t = time.perf_counter()
        return self

    def __exit__(self, *a):
        import time

        torch.cuda.synchronize()
        self.ms = (time.perf_counter() - self._t) * 1e3
