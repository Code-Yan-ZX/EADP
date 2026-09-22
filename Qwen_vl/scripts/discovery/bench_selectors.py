#!/usr/bin/env python3
"""
Micro-benchmark and equivalence check for the selection operators.

Uses synthetic importance / similarity tensors with the real geometry
(N = 1024 merged tokens), so it isolates the selection stage from the model.

  1. equivalence: do the fast variants return the same SET as the official
     greedy?  (they should -- same objective, same greedy rule)
  2. scaling: how does each operator's latency grow with the token budget T
     and with the candidate pool N?
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import time_callable  # noqa: E402
from instrumented import SELECTORS  # noqa: E402


def make_inputs(N, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    feats = torch.randn(N, 512, device=device, generator=g)
    v = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = 0.5 * (torch.matmul(v, v.t()).clamp(-1, 1) + 1.0)
    importance = torch.rand(1, N, device=device, generator=g)
    # make importance peaked, like a real relevance map
    importance = importance ** 3
    return importance, sim.unsqueeze(0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"equivalence": {}, "scaling": {}}

    print("=== equivalence vs official greedy (N=1024, T=256) ===")
    imp, sim = make_inputs(1024)
    ref, _ = SELECTORS["facility"](imp, sim, 256)
    ref_set = set(ref[0].tolist())
    for name in ("facility_fast", "lazy_greedy", "stochastic", "topk", "topk_nms", "farthest"):
        try:
            got, _ = SELECTORS[name](imp, sim, 256)
            gs = set(got[0].tolist())
            jac = len(gs & ref_set) / len(gs | ref_set)
            exact = gs == ref_set
            print(f"  {name:16s} jaccard_vs_official={jac:.3f}  identical={exact}")
            out["equivalence"][name] = {"jaccard": jac, "identical": bool(exact)}
        except Exception as e:
            print(f"  {name:16s} FAILED {e}")
            out["equivalence"][name] = {"error": str(e)}

    print("\n=== latency vs token budget T (N=1024) ===")
    print(f"  {'T':>5} " + " ".join(f"{n:>15}" for n in
          ("facility", "facility_fast", "lazy_greedy", "stochastic", "topk")))
    for T in (64, 128, 256, 512, 1024):
        row = {}
        for name in ("facility", "facility_fast", "lazy_greedy", "stochastic", "topk"):
            try:
                m, s = time_callable(lambda n=name: SELECTORS[n](imp, sim, T), 3, 1)
                row[name] = {"mean": m, "std": s}
            except Exception as e:
                row[name] = {"error": str(e)}
        out["scaling"][f"T{T}"] = row
        cells = []
        for n in ("facility", "facility_fast", "lazy_greedy", "stochastic", "topk"):
            v = row[n].get("mean")
            cells.append(f"{v:11.2f}ms" if v is not None else f"{'ERR':>13}")
        print(f"  {T:>5} " + " ".join(f"{c:>15}" for c in cells))

    print("\n=== latency vs candidate pool N (T=256) ===")
    for N in (512, 1024, 2048):
        imp_n, sim_n = make_inputs(N)
        cells = []
        for name in ("facility", "facility_fast", "lazy_greedy", "stochastic"):
            m, _ = time_callable(lambda n=name, i=imp_n, s=sim_n: SELECTORS[n](i, s, 256), 3, 1)
            cells.append(f"{name}={m:8.2f}ms")
            out["scaling"].setdefault(f"N{N}", {})[name] = {"mean": m}
        print(f"  N={N:>5}  " + "  ".join(cells))

    path = os.path.join(common.ensure_out_dir(), "bench_selectors.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
