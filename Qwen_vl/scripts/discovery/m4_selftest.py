"""
M4-v0 step 0 -- pure-function self-test for the REC capsule math.

No model, no GPU, no data: synthetic tensors with known answers.  This runs in
seconds and is the only place the pooling arithmetic is checked against a
closed form, so that an accuracy result cannot be explained away as a shape bug.

Usage
    python scripts/discovery/m4_selftest.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m4_common import (BUDGET, GRID, N_VIS, R_GRID, _fps_assign,       # noqa: E402
                       build_capsules, evict_t, partition_ids, region_factor,
                       residual_u)

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main():
    torch.manual_seed(0)
    dev = "cpu"
    D = 64

    print("partition")
    want = {8: (2, 4), 16: (4, 4), 32: (4, 8)}
    for r in R_GRID:
        check(f"region_factor({r})", region_factor(r) == want[r],
              f"{region_factor(r)}")
        rid = partition_ids(r, device=dev)
        rows, cols = region_factor(r)
        sizes = torch.bincount(rid, minlength=r)
        check(f"r={r} covers all {N_VIS} tokens",
              rid.numel() == N_VIS and int(rid.min()) == 0
              and int(rid.max()) == r - 1)
        check(f"r={r} cells are {N_VIS // r}-token rectangles",
              bool((sizes == N_VIS // r).all()), f"{sizes.tolist()}")
        # cell (a, b) must be a contiguous rectangle of the grid
        m = rid.reshape(GRID, GRID)
        ok = all(len(torch.unique(m[a * (GRID // rows):(a + 1) * (GRID // rows),
                                    b * (GRID // cols):(b + 1) * (GRID // cols)])) == 1
                 for a in range(rows) for b in range(cols))
        check(f"r={r} cells are spatial rectangles", ok)

    print("\nresidual_u")
    vis = torch.randn(N_VIS, D)
    A = torch.arange(0, 240)
    # a token copied from an anchor is perfectly explained
    vis[500] = vis[3]
    u = residual_u(vis, A)
    check("copied anchor -> u == 0", float(u[500]) < 1e-5, f"{float(u[500]):.2e}")
    check("u(anchor) == 0", float(u[A].abs().max()) < 1e-5)
    # u = 2 * (1 - max_a sim) under the selector's rebound similarity
    vn = vis / vis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sim = 0.5 * ((vn @ vn.t()).clamp(-1, 1) + 1)
    check("u == 2*(1 - max_a sim)",
          bool((u - 2 * (1 - sim[:, A].max(1).values)).abs().max() < 1e-5))
    check("u in [0, 2]", float(u.min()) >= -1e-6 and float(u.max()) <= 2 + 1e-6)

    print("\ncapsules: uniform weights == arithmetic mean of the pooled tokens")
    r = 16
    rid = partition_ids(r, device=dev)
    # A realistic anchor set: 240 of 1024 tokens spread over the grid.  The
    # contiguous prefix used above leaves whole cells with no anchor and no
    # dropped token, which the real data never does (0 empty cells over 150
    # instances, m4_offline.json) -- that path is tested on its own below.
    A = torch.randperm(N_VIS)[:240]
    u = residual_u(vis, A)
    keep = torch.zeros(N_VIS, dtype=torch.bool)
    keep[A] = True
    dropped = (~keep).nonzero(as_tuple=True)[0]
    imp = torch.rand(N_VIS)
    caps, st = build_capsules(vis, imp, u, A, dropped, r, "mean")
    ok = True
    for g in range(r):
        idx = dropped[rid[dropped] == g]
        if idx.numel():
            ok &= bool((caps[g] - vis[idx].mean(0)).abs().max() < 1e-5)
    check("MEAN-r16 == per-cell arithmetic mean", ok)
    check("capsule count and width", caps.shape == (r, D), f"{tuple(caps.shape)}")
    check("no empty cell on a spread anchor set", st["n_empty"] == 0,
          f"{st['n_empty']}")

    print("\ncapsules: residual weights, temperature limits")
    hot, st_h = build_capsules(vis, imp, u, A, dropped, r, "residual", "spatial", 1e-4)
    ok = True
    for g in range(r):
        idx = dropped[rid[dropped] == g]
        if idx.numel():
            # tau -> 0 is one-hot at the argmax UP TO the float32 resolution of
            # (u - u_max)/tau, so the exact claim is: the capsule is the mean of
            # the tokens whose u is within 1e-3 of the cell maximum.
            near = idx[(u[idx].max() - u[idx]) < 1e-3]
            ok &= bool((hot[g] - vis[near].mean(0)).abs().max() < 1e-3)
    check("tau -> 0 collapses onto the cell's most-unexplained token", ok)
    check("tau -> 0 gives ESS ~ 1", st_h["ess"] < 1.5, f"{st_h['ess']:.2f}")
    flat, stf = build_capsules(vis, imp, u, A, dropped, r, "residual", "spatial", 1e4)
    check("tau -> inf recovers the plain mean",
          bool((flat - caps).abs().max() < 1e-3))
    check("ESS is monotone in tau",
          st["ess"] >= stf["ess"] > st_h["ess"],
          f"{st['ess']:.1f} >= {stf['ess']:.1f} > {st_h['ess']:.1f}")

    print("\ncapsules: the empty-cell fallback path")
    # A contiguous-prefix anchor set (the pathological case the real data never
    # produces) leaves whole cells with nothing to pool.  Those capsules must
    # fall back to the anchor mean rather than becoming zeros, and norm-restore
    # must leave them alone instead of dividing them by a zero count.
    Ap = torch.arange(0, 240)
    up = residual_u(vis, Ap)
    keepp = torch.zeros(N_VIS, dtype=torch.bool)
    keepp[Ap] = True
    dropp = (~keepp).nonzero(as_tuple=True)[0]
    _, stp = build_capsules(vis, imp, up, Ap, dropp, r, "mean")
    check("a prefix anchor set does leave cells empty (the path is reachable)",
          stp["n_empty"] > 0, f"{stp['n_empty']}")
    fallback = vis[Ap].mean(0)
    cp, _ = build_capsules(vis, imp, up, Ap, dropp, r, "mean")
    empty_cells = [g for g in range(r) if (rid[dropp] == g).sum() == 0]
    check("empty cells take the anchor-mean fallback",
          all(bool((cp[g] - fallback).abs().max() < 1e-5) for g in empty_cells))
    np_, _ = build_capsules(vis, imp, up, Ap, dropp, r, "residual", "spatial",
                            0.05, norm_restore=True)
    check("norm-restore never zeroes an empty cell",
          all(float(np_[g].norm()) > 1e-6 for g in empty_cells),
          f"{[round(float(np_[g].norm()), 4) for g in empty_cells]}")

    print("\ncapsules: norm restore")
    nr, _ = build_capsules(vis, imp, u, A, dropped, r, "residual", "spatial",
                           0.05, norm_restore=True)
    raw, _ = build_capsules(vis, imp, u, A, dropped, r, "residual", "spatial", 0.05)
    ok = True
    for g in range(r):
        idx = dropped[rid[dropped] == g]
        if not idx.numel():
            continue
        tgt = vis[idx].norm(dim=1).mean()
        ok &= bool(abs(float(nr[g].norm()) - float(tgt)) < 1e-4)
        # direction must be untouched
        cos = torch.nn.functional.cosine_similarity(nr[g], raw[g], dim=0)
        ok &= bool(float(cos) > 1 - 1e-6)
    check("norm-restore hits the pooled mean norm, direction unchanged", ok)
    check("norm-restore only ever grows a capsule",
          bool(nr.norm(dim=1).min() >= raw.norm(dim=1).min() - 1e-6))
    check("a convex combination really does shrink (why the arm exists)",
          bool(raw.norm(dim=1).median() < vis.norm(dim=1).median()))

    print("\ncapsules: anchor_mean carries no dropped token")
    am, _ = build_capsules(vis, imp, u, A, dropped, r, "anchor_mean")
    ok = True
    for g in range(r):
        idx = A[rid[A] == g]
        ok &= bool((am[g] - vis[idx].mean(0)).abs().max() < 1e-5)
    check("ANCH-r16 == per-cell mean of RETAINED tokens", ok)

    print("\nassignment: farthest-point seeds")
    gid = _fps_assign(vis, A, dropped, r, imp)
    check("fps returns r groups", int(gid.max()) == r - 1 and int(gid.min()) == 0,
          f"[{int(gid.min())},{int(gid.max())}]")
    gid2 = _fps_assign(vis, A, dropped, r, imp)
    check("fps is deterministic", bool((gid == gid2).all()))
    check("fps assigns every dropped token",
          gid.numel() == dropped.numel())
    caps_f, _ = build_capsules(vis, imp, u, A, dropped, r, "residual", "fps", 0.05)
    check("fps capsules have the same shape", caps_f.shape == (r, D))
    check("fps differs from the spatial partition",
          not bool((gid == rid[dropped]).all()))

    print("\ndeterminism and reduction accuracy")
    # The first M4 grid was NOT reproducible: `index_add_` reduces with
    # atomicAdd on CUDA, and a ~2e-7 difference in a capsule is enough to flip
    # a bf16 rounding and an occasional greedy token (the same arm scored 58.06
    # and 59.18 macro in two fresh processes).  The pooling now sorts by group
    # and differences a prefix sum, and these two checks are what keep it that
    # way: bitwise reproducibility, and agreement with a float64 reference.
    reps = [build_capsules(vis, imp, u, A, dropped, r, "residual", "spatial", 0.05)[0]
            for _ in range(6)]
    check("capsules are bitwise reproducible",
          all(bool(torch.equal(reps[0], x)) for x in reps[1:]))
    ref = torch.zeros(r, D, dtype=torch.float64)
    gid = partition_ids(r)[dropped]
    for g in range(r):
        idx = dropped[gid == g]
        if idx.numel():
            wg = torch.softmax(u[idx].double() / 0.05, dim=0)
            ref[g] = (wg.unsqueeze(1) * vis[idx].double()).sum(0)
    err = float((reps[0].double() - ref).abs().max())
    check("segment sum matches a float64 reference", err < 1e-4, f"max err {err:.2e}")
    check("the anchor_mean branch is reproducible too",
          bool(torch.equal(
              build_capsules(vis, imp, u, A, dropped, r, "anchor_mean")[0],
              build_capsules(vis, imp, u, A, dropped, r, "anchor_mean")[0])))

    print("\neviction: maxred")
    s0 = torch.arange(256)
    vn2 = vis / vis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sim2 = 0.5 * ((vn2 @ vn2.t()).clamp(-1, 1) + 1)
    vis[900] = vis[7]                        # make one S0 member perfectly redundant
    s0 = torch.cat([s0, torch.tensor([900])])[:256]
    s0 = torch.arange(256)
    vn2 = vis / vis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sim2 = 0.5 * ((vn2 @ vn2.t()).clamp(-1, 1) + 1)
    ev = evict_t(s0, imp, sim2, 8, rule="maxred")
    sub = sim2[s0][:, s0].clone()
    sub.fill_diagonal_(float("-inf"))
    red = sub.max(1).values
    want_ev = s0[torch.argsort(-red)[:8]]
    check("maxred evicts the 8 most redundant S0 tokens",
          bool((torch.sort(ev).values == torch.sort(want_ev).values).all()))
    A2 = s0[~torch.isin(s0, ev)]
    check("A has 256 - r tokens", A2.numel() == BUDGET - 8, f"{A2.numel()}")

    print()
    if FAILS:
        print(f"FAILED {len(FAILS)}: {FAILS}")
        return 1
    print("all REC self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
