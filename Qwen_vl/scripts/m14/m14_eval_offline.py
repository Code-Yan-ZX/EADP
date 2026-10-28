#!/usr/bin/env python3
"""
M14 offline evaluation -- every solver against the exact facility reference,
on the captured (importance, sim) pairs.  No model, no generation.

Per (sample, config):
  reference   official facility greedy: selection, per-step choices, objective
  metrics     selected-set overlap |&|/|ref|, exact (dense) objective gap,
              per-step choice agreement until first divergence
  diagnostics screen variants: is the true exact winner inside the evaluated
              shortlist (winner recall), verify violations, stale-rank of the
              exact winner by greedy tertile (adaptive-shrinking evidence)
  timing      selection latency, CUDA events + sync wall clock, warm-up 5 /
              reps 15 (grid); graph-build / k-means precompute timed apart

Outputs: outputs/m14/m14_offline.json + m14_offline_sets.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "discovery"))

import m14_common as C  # noqa: E402
from m14_common import (facility_objective, set_overlap, time_cuda,  # noqa: E402
                        time_wall)
from model.pruner import _greed_select_impl  # noqa: E402
import m14_solvers as SV  # noqa: E402
from m14_solvers import build_sparse_graph, _dense_gain  # noqa: E402


# ---------------------------------------------------------------------------
# config grid (frozen before any number was read)
# ---------------------------------------------------------------------------
def make_configs():
    cfgs = []
    for L in C.SHORTLIST_GRID:
        cfgs.append(("screen_verify", dict(shortlist=L)))
    for L in C.SHORTLIST_GRID:
        for R in (1, 2):
            cfgs.append(("screen_fixed", dict(shortlist=L, rounds=R)))
    for r in C.SPARSE_R_GRID:
        cfgs.append(("sparse_graph", dict(r=r)))
    for M in C.CORESET_M_GRID:
        cfgs.append(("demand_coreset", dict(M=M)))
    for r in (16, 32, 64):
        cfgs.append(("sparse_screen", dict(r=r, shortlist=32, rounds=2)))
    cfgs.append(("sparse_screen", dict(r=32, shortlist=64, rounds=2)))
    return cfgs


# ---------------------------------------------------------------------------
# one reference run + one diagnostics/timing context per sample
# ---------------------------------------------------------------------------
def run_config(name, kwargs, imp, sim, K):
    fn = SV.SOLVERS[name]
    t0 = time.perf_counter()
    idx, info = fn(imp, sim, K, **kwargs)
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - t0) * 1e3
    return idx, info, setup_ms


def diagnostics(name, kwargs, imp, sim, K, sel_solver, sel_ref):
    """True-winner-in-shortlist recall + choice agreement (screen variants).

    Replays the exact greedy and, at every step, checks whether the exact
    winner would be inside the shortlist the solver's screen would form.
    Implemented for screen_fixed / sparse_screen / screen_verify (the last is
    exact by construction; its recall is verified, not assumed).
    """
    w = imp[0]
    S = sim[0]
    T = min(K, S.shape[0])
    N = S.shape[0]

    cur_max = torch.zeros(N, device=S.device)
    selected = torch.zeros(N, dtype=torch.bool, device=S.device)
    stale = S @ w
    hit = np.zeros(T, dtype=bool)
    agree = np.zeros(T, dtype=bool)
    stale_rank = np.full(T, np.nan, dtype=np.float32)

    ref_order = sel_ref[0].tolist()
    solver_order = sel_solver[0].tolist()
    L = kwargs.get("shortlist", 0)
    if name == "screen_verify" and not L:
        L = 32

    diverged = False
    for t in range(T):
        key = stale.masked_fill(selected, SV.NEG)
        # exact winner at this step (dense full pass, offline only)
        gain_all = _dense_gain(S, cur_max, w).masked_fill(selected, SV.NEG)
        winner = int(gain_all.argmax())
        order = torch.argsort(key, descending=True)
        rank = int((order == winner).nonzero()[0])
        stale_rank[t] = rank
        if L:
            hit[t] = rank < min(L * kwargs.get("rounds", 1) + 0, N)
        agree[t] = (not diverged) and (solver_order[t] == ref_order[t])
        if solver_order[t] != ref_order[t]:
            diverged = True
        # advance the EXACT greedy state (upper-bound replay)
        cur_max = torch.maximum(cur_max, S[winner])
        selected[winner] = True
        # refresh stale exactly like CELF would after evaluating the winner
        g_all = torch.clamp(S - cur_max.unsqueeze(0), min=0.0) @ w
        stale = g_all.masked_fill(selected, SV.NEG)

    return {"winner_in_shortlist": hit, "choice_agreement": agree,
            "exact_winner_stale_rank": stale_rank}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timing", type=int, default=1)
    ap.add_argument("--out", default="m14_offline.json")
    args = ap.parse_args()

    imp_npz, sim_mm, _vis = C.load_capture()
    n = len(imp_npz["ds"])
    if args.limit:
        n = min(n, args.limit)
    cfgs = make_configs()
    print(f"[offline] {n} samples x {len(cfgs)} configs, K={C.BUDGET}")

    K = C.BUDGET
    all_sets = {}
    summary = {}

    # per-sample reference (computed once)
    ref_sets, ref_objs, ref_order = [], [], []
    for i in range(n):
        imp, sim, _ = C.sample_entry(imp_npz, sim_mm, _vis, i)
        sel, _ = _greed_select_impl(imp.unsqueeze(0), sim.unsqueeze(0), K)
        ref_sets.append(sel[0].cpu().numpy())
        ref_objs.append(float(facility_objective(sim[0], imp[0], sel[0])))
        ref_order.append(sel[0].cpu().numpy())
    ref_sets_a = np.stack(ref_sets)
    print(f"[reference] done. mean objective = {np.mean(ref_objs):.4f}")

    for name, kwargs in cfgs:
        key = f"{name}|" + "|".join(f"{k}={v}" for k, v in kwargs.items())
        overlaps, gaps, agree_until = [], [], []
        diag_hit = []
        viol_rate = []
        per_sample = []
        for i in range(n):
            imp, sim, _ = C.sample_entry(imp_npz, sim_mm, _vis, i)
            idx, info, _ = run_config(name, kwargs, imp.unsqueeze(0),
                                      sim.unsqueeze(0), K)
            sel = idx[0].cpu().numpy()
            all_sets[f"{key}::{i}"] = sel.astype(np.int32)
            ov = set_overlap(sel, ref_sets_a[i])
            F = float(facility_objective(sim[0], imp[0], idx[0]))
            gaps.append((ref_objs[i] - F) / max(abs(ref_objs[i]), 1e-9))
            overlaps.append(ov)
            per_sample.append({"overlap": ov, "obj_gap": gaps[-1]})
            # choice agreement (first divergence length)
            agree = next((t for t in range(K) if sel[t] != ref_sets_a[i][t]), K)
            agree_until.append(agree)
            if name in ("screen_verify", "screen_fixed", "sparse_screen"):
                d = diagnostics(name, kwargs, imp.unsqueeze(0), sim.unsqueeze(0), K, idx, ref_order[i][None, :])
                diag_hit.append(float(np.mean(d["winner_in_shortlist"])))
                if "verify_violation" in info:
                    viol_rate.append(float(info["verify_violation"].float().mean()))
        rec = {
            "overlap_mean": float(np.mean(overlaps)),
            "overlap_p05": float(np.percentile(overlaps, 5)),
            "obj_gap_mean": float(np.mean(gaps)),
            "obj_gap_p95_abs": float(np.percentile(np.abs(gaps), 95)),
            "agree_steps_median": float(np.median(agree_until)),
            "agree_full_frac": float(np.mean([a == K for a in agree_until])),
            "winner_recall": float(np.mean(diag_hit)) if diag_hit else None,
            "verify_violation": float(np.mean(viol_rate)) if viol_rate else None,
        }
        summary[key] = rec
        print(f"{key:52s} ov={rec['overlap_mean']:.4f} "
              f"gap={rec['obj_gap_mean'] * 100:+.3f}% "
              f"agree={rec['agree_steps_median']:6.1f} "
              f"recall={rec['winner_recall']} viol={rec['verify_violation']}")

    # ---------------- timing pass ------------------------------------------
    timing = {}
    if args.timing:
        for name, kwargs in cfgs:
            key = f"{name}|" + "|".join(f"{k}={v}" for k, v in kwargs.items())
            cuda_ms, wall_ms, pre_ms = [], [], []
            for i in range(n):
                imp, sim, _ = C.sample_entry(imp_npz, sim_mm, _vis, i)
                imp, sim = imp.unsqueeze(0), sim.unsqueeze(0)
                fn = (lambda nm=name, kw=kwargs, m=imp, s=sim:
                      SV.SOLVERS[nm](m, s, K, **kw))
                cuda_ms.append(time_cuda(fn, n_warmup=5, n_repeat=15)["median"])
                wall_ms.append(time_wall(fn, n_warmup=5, n_repeat=15)["median"])
                if name in ("sparse_graph", "sparse_screen"):
                    pre_ms.append(time_cuda(
                        lambda s=sim: build_sparse_graph(s[0], kwargs.get("r", 32)),
                        n_warmup=5, n_repeat=15)["median"])
                elif name == "demand_coreset":
                    pre_ms.append(float("nan"))
            timing[key] = {"cuda_ms_median": float(np.median(cuda_ms)),
                           "wall_ms_median": float(np.median(wall_ms)),
                           "precompute_ms": float(np.nanmean(pre_ms))
                           if pre_ms else 0.0}
            print(f"[timing] {key:52s} cuda={timing[key]['cuda_ms_median']:8.2f}ms "
                  f"wall={timing[key]['wall_ms_median']:8.2f}ms")

    out = {"config": {"K": K, "N": C.N_VIS, "n_samples": n,
                      "sim_mode": C.SIM_MODE},
           "reference_objective_mean": float(np.mean(ref_objs)),
           "summary": summary, "timing": timing}
    with open(os.path.join(C.M14_OUT, args.out), "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(os.path.join(C.M14_OUT, "m14_offline_sets.npz"),
                        ref=ref_sets_a, **{k: v for k, v in all_sets.items()})
    print(f"[saved] {os.path.join(C.M14_OUT, args.out)}")


if __name__ == "__main__":
    main()
