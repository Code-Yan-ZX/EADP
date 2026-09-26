"""
M6 step 4 -- what the proxies actually cost.

The verdict is REFUTED, so no method was built and no TTFT was measured.  What
is still worth pinning down is the marginal cost of the audit itself, because it
decides whether a *future* round could afford these signals: the incumbent's
pruner already produces the EADP score, its components, the visual similarity
matrix and the tower features, so most of the 27 columns are reductions over
tensors that already exist and cost nothing.

Only `nn4_recon` (a (1024, |S0|) matmul plus 1024 4x4 solves) and `cos_s0c`
(a centroid and a dot product) do work the pruner does not already do.  This
script times exactly those two, on the bank's own cached vision features, with
the pruner's own similarity convention.

Usage
    python scripts/discovery/m6_cost.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUT_DIR                                       # noqa: E402
from m6_common import BUDGET, N_VIS, dump_json, load_bank           # noqa: E402

VIS = "m5_vis.npy"
N_WARM = 10
N_ITER = 50


def nn4_recon(v: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Verbatim from `m5_common.safe_features`."""
    vn = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    a_idx = keep.nonzero(as_tuple=True)[0]
    cos_to_a = vn @ vn[a_idx].t()
    kk = min(4, a_idx.numel())
    nn_pos = torch.topk(cos_to_a, kk, dim=1).indices
    A = vn[a_idx][nn_pos]
    gram = A @ A.transpose(1, 2)
    rhs = (A @ vn.unsqueeze(-1)).squeeze(-1)
    gram = gram + 1e-4 * torch.eye(kk, device=v.device, dtype=gram.dtype)
    coef = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)
    proj = (coef.unsqueeze(-1) * A).sum(dim=1)
    return (vn - proj).norm(dim=-1) / vn.norm(dim=-1).clamp_min(1e-8)


def cos_s0c(v: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    vn = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    c = vn[keep.nonzero(as_tuple=True)[0]].mean(0)
    return vn @ (c / c.norm().clamp_min(1e-8))


def main():
    bank = load_bank()
    vis = np.load(os.path.join(OUTPUT_DIR, VIS), mmap_mode="r")
    dev = "cuda"
    rows = [0, 1, 2, 3]
    # `safe_features` is per-instance (one (1024, D) image at a time), so the
    # benchmark is too -- a batch would not measure the deployed shape.
    vv, kk = [], []
    for r in rows:
        vv.append(torch.from_numpy(np.array(vis[r])).view(torch.bfloat16)
                  .float().to(dev))
        m = torch.zeros(N_VIS, dtype=torch.bool, device=dev)
        m[torch.from_numpy(bank["s0"][r].astype(np.int64)).to(dev)] = True
        kk.append(m)

    out = {}
    for name, fn in (("nn4_recon", nn4_recon), ("cos_s0c", cos_s0c)):
        def one():
            for v, k in zip(vv, kk):
                fn(v, k)
        for _ in range(N_WARM):
            one()
        torch.cuda.synchronize()
        ts = []
        for _ in range(N_ITER):
            t0 = time.perf_counter()
            one()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000 / len(rows))
        out[name] = dict(median_ms=float(np.median(ts)),
                         p90_ms=float(np.percentile(ts, 90)))
        print(f"  {name:12s} median {out[name]['median_ms']:.3f} ms  "
              f"p90 {out[name]['p90_ms']:.3f} ms   (per instance)")

    free = [c for c in bank["feature_names"]
            if c not in ("nn4_recon", "cos_s0c")]
    dump_json("m6_cost.json", dict(
        n_warm=N_WARM, n_iter=N_ITER, batch=len(rows), device=dev,
        timed=out,
        total_marginal_ms=float(sum(v["median_ms"] for v in out.values())),
        free_columns=free,
        note=("only nn4_recon and cos_s0c do work the incumbent's pruner does not "
              "already do; the other 25 columns are reductions over tensors it "
              "already produces (score, components, sim_matrix, tower features)")))
    print(f"[saved] m6_cost.json  total marginal "
          f"{sum(v['median_ms'] for v in out.values()):.3f} ms per instance")


if __name__ == "__main__":
    main()
