#!/usr/bin/env python3
"""
M14 PART 1 -- facility computational anatomy.

Question: at K=256 on N=1024, where does the official greedy's time actually
go?  Measured, not guessed, on the real captured importance/sim maps.

Decomposition, CUDA-event brackets around every per-step op of the official
greedy (same ops, same order as model.pruner._greed_select_impl):

    covmax   torch.maximum(sim, cur_max)            (N, N) write
    diff     new_coverage - cur_max                 (N, N) read + write
    mul      gains * importance                     (N, N) fused
    sum      weighted.sum(dim=2)                    (N,)   reduction
    mask     total_gain[selected_mask] = -1e9       bool index put
    argmax   torch.argmax + write into select_idx
    update   sim[best] row -> maximum into cur_max, set selected flag
    loop     residual (launch gaps, scheduling)

Also: end-to-end selector latency (CUDA-event and synchronised wall-clock,
10 warm-up + 30 reps, median over 450 samples) for facility / facility_fast /
lazy_greedy at K = 64 / 128 / 256 / 512, plus a torch.profiler kernel census
of one official call.

Outputs: outputs/m14/m14_anatomy.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "discovery"))

import m14_common as C  # noqa: E402
from m14_common import time_cuda, time_wall  # noqa: E402
from model.pruner import _greed_select_impl, _sim_visual_impl  # noqa: E402
from instrumented import SELECTORS  # noqa: E402

COMPONENTS = ("covmax", "diff", "mul", "sum", "mask", "argmax", "update", "loop")


def greedy_bracketed(importance, sim, K):
    """Official greedy with per-op CUDA events; returns (select_idx, per-op ms)."""
    B, N = importance.shape
    T = min(K, N)
    device = importance.device

    ev = [[torch.cuda.Event(enable_timing=True) for _ in range(8)]
          for _ in range(T)]
    select_idx = torch.zeros(B, T, dtype=torch.long, device=device)
    cur_max = torch.zeros(B, N, device=device)
    selected_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    bi = torch.arange(B, device=device)

    for t in range(T):
        cur_max_exp = cur_max.unsqueeze(1)
        ev[t][0].record()
        new_cov = torch.maximum(sim, cur_max_exp)
        ev[t][1].record()
        gains = new_cov - cur_max_exp
        ev[t][2].record()
        weighted = gains * importance.unsqueeze(1)
        ev[t][3].record()
        total_gain = weighted.sum(dim=2)
        ev[t][4].record()
        total_gain[selected_mask] = -1e9
        ev[t][5].record()
        best = torch.argmax(total_gain, dim=1)
        select_idx[:, t] = best
        ev[t][6].record()
        chosen = sim[bi, best, :]
        cur_max = torch.maximum(cur_max, chosen)
        selected_mask[bi, best] = True
        ev[t][7].record()
    torch.cuda.synchronize()

    acc = {k: 0.0 for k in COMPONENTS}
    for t in range(T):
        acc["covmax"] += ev[t][0].elapsed_time(ev[t][1])
        acc["diff"] += ev[t][1].elapsed_time(ev[t][2])
        acc["mul"] += ev[t][2].elapsed_time(ev[t][3])
        acc["sum"] += ev[t][3].elapsed_time(ev[t][4])
        acc["mask"] += ev[t][4].elapsed_time(ev[t][5])
        acc["argmax"] += ev[t][5].elapsed_time(ev[t][6])
        acc["update"] += ev[t][6].elapsed_time(ev[t][7])
        if t + 1 < T:
            acc["loop"] += ev[t][7].elapsed_time(ev[t + 1][0])
    total = ev[0][0].elapsed_time(ev[T - 1][7])
    return select_idx, {k: v for k, v in acc.items()}, total


def kernel_census(imp, sim, K):
    try:
        from torch.profiler import profile, ProfilerActivity
    except Exception as e:
        return {"error": str(e)}
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _greed_select_impl(imp, sim, K)
        torch.cuda.synchronize()
    rows = []
    for e in prof.key_averages():
        dt = getattr(e, "self_device_time_total", 0) or 0
        if dt > 0:
            rows.append({"name": e.key[:80], "count": int(e.count),
                         "device_us": float(dt)})
    rows.sort(key=lambda r: -r["device_us"])
    return {"total_device_us": float(sum(r["device_us"] for r in rows)),
            "top": rows[:20]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    imp_npz, sim_mm, vis_mm = C.load_capture()
    n = len(imp_npz["ds"])
    if args.limit:
        n = min(n, args.limit)
    print(f"[anatomy] {n} samples")

    # 1. sim construction (rebound kernel), bf16 features in
    _, _, vis0 = C.sample_entry(imp_npz, sim_mm, vis_mm, 0)
    sim_build = time_cuda(lambda: _sim_visual_impl(vis0.unsqueeze(0)))

    # 2. per-op breakdown at K=256 (mean over samples), plus K sweep of totals
    per_op = {}
    totals = {}
    for K in C.K_SWEEP:
        acc = {k: 0.0 for k in COMPONENTS}
        tot = 0.0
        for i in range(n):
            imp, sim, _ = C.sample_entry(imp_npz, sim_mm, vis_mm, i)
            _, comps, total = greedy_bracketed(imp.unsqueeze(0), sim.unsqueeze(0), K)
            for k, v in comps.items():
                acc[k] += v
            tot += total
        per_op[f"K{K}"] = {k: v / n for k, v in acc.items()}
        totals[f"K{K}"] = tot / n
        if K == 256:
            print("[per-op ms @K256, total over 256 steps]")
            for k, v in per_op["K256"].items():
                print(f"  {k:8s} {v:9.3f}  ({100 * v / max(tot / n, 1e-9):5.1f}%)")

    # 3. end-to-end latency vs K for the three official-objective solvers
    latency = {}
    for K in C.K_SWEEP:
        row = {}
        for name in ("facility", "facility_fast", "lazy_greedy"):
            cuda_ms, wall_ms = [], []
            for i in range(n):
                imp, sim, _ = C.sample_entry(imp_npz, sim_mm, vis_mm, i)
                imp, sim = imp.unsqueeze(0), sim.unsqueeze(0)
                fn = (lambda nm=name, k=K, m=imp, s=sim: SELECTORS[nm](m, s, k))
                cuda_ms.append(time_cuda(fn, n_warmup=5, n_repeat=15)["median"])
                wall_ms.append(time_wall(fn, n_warmup=5, n_repeat=15)["median"])
            row[name] = {"cuda_ms_median": float(np.median(cuda_ms)),
                         "wall_ms_median": float(np.median(wall_ms))}
            print(f"K={K:4d} {name:14s} cuda={row[name]['cuda_ms_median']:8.2f}ms "
                  f"wall={row[name]['wall_ms_median']:8.2f}ms")
        latency[f"K{K}"] = row

    # 4. kernel census of one official call at K=256
    imp0, sim0, _ = C.sample_entry(imp_npz, sim_mm, vis_mm, 0)
    census = kernel_census(imp0.unsqueeze(0), sim0.unsqueeze(0), 256)

    out = {"n_samples": n,
           "sim_build_ms": sim_build,
           "per_op_ms_sum_over_steps": per_op,
           "greedy_total_ms_eventbrackets": totals,
           "selector_latency": latency,
           "kernel_census": census}
    path = os.path.join(C.M14_OUT, "m14_anatomy.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
