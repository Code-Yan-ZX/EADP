"""
M4-v0 step 1 -- offline diagnostics and the residual maps.

Nothing here generates a token.  It runs the exact live capsule code
(`m4_common.evict_t` / `residual_u` / `build_capsules`) over the frozen held-out
150 and answers the questions that must be settled BEFORE the accuracy grid
runs, because they fix knobs the brief forbids searching later:

  1. What does the residual map look like?  Is u_i = 1 - max_{a in A} cos(v_i,
     v_a) a live quantity with usable spread, or is it a constant?
  2. How much does a capsule shrink relative to the tokens it pools?  A convex
     combination of k diverse vectors has a smaller norm than a token, and that
     is a distribution shift the decoder sees.  This decides whether the
     norm-restore arm is worth its place in the grid.
  3. What temperature makes the residual softmax neither uniform nor one-hot?
     Fixed by an effective-sample-size rule, not by accuracy.
  4. What tau_imp gives the importance merge the SAME effective sample size as
     the residual merge?  Without that, IMP-r16 is not a controlled comparison.
  5. Does any partition cell ever come up empty?

It also writes `m4_u_maps.npz`: the per-instance residual map under each r,
which is what the content-free SHUF arm injects (another instance's map, the
M2 `C1-SHUF` idiom).

The bank stores fp16 `vis` and was built by `m3_features.py`; the maps it yields
therefore differ from the live fp32 maps at the ~1e-3 level.  That is fine for a
content-free control and for a figure, and it is why the SHUF arm is the ONLY
arm fed from this file -- no candidate arm ever reads it.

Usage
    python scripts/discovery/m4_offline.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUT_DIR                                       # noqa: E402
from m4_common import (EVICT_RULE, GRID, N_VIS, R_GRID, TAU,        # noqa: E402
                       build_capsules, evict_t, partition_ids, region_factor,
                       region_map, residual_u)

BANK = "m3_bank_v1.npz"
TAG = "m4_offline"
TAU_GRID = (0.02, 0.05, 0.1, 0.2)
DEV = "cuda"


def load_bank():
    """One decompress of the 3.7 GB `vis` member, kept in RAM.

    An `NpzFile` re-decompresses the whole member on every `z["vis"]` access, so
    indexing it inside the per-instance loop would cost ~12 s per instance
    instead of ~12 s once.
    """
    z = np.load(os.path.join(OUTPUT_DIR, BANK), allow_pickle=False)
    names = [str(x) for x in z["feature_names"]]
    out = dict(key=np.array([str(k) for k in z["key"]]),
               ds=np.array([str(k) for k in z["ds"]]),
               split=np.array([str(k) for k in z["split"]]),
               s0=z["s0"], X=z["X"], vis=z["vis"], IMP=names.index("imp"))
    out["test"] = np.where(out["split"] == "test")[0]
    return out


def instance_state(z, i, r):
    """Live-code A and u for bank row i at capsule count r."""
    vis = torch.from_numpy(z["vis"][i].astype(np.float32)).to(DEV)
    imp = torch.from_numpy(z["X"][i, :, z["IMP"]].astype(np.float32)).to(DEV)
    s0 = torch.from_numpy(np.sort(z["s0"][i]).astype(np.int64)).to(DEV)
    vn = vis / vis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sim = 0.5 * ((vn @ vn.t()).clamp(-1.0, 1.0) + 1.0)
    ev = evict_t(s0, imp, sim, r, rule=EVICT_RULE)
    keep = torch.zeros(N_VIS, dtype=torch.bool, device=DEV)
    A = s0[~torch.isin(s0, ev)]
    keep[A] = True
    dropped = (~keep).nonzero(as_tuple=True)[0]
    u = residual_u(vis, A)
    return vis, imp, s0, A, dropped, u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    z = load_bank()
    rows = z["test"][: args.limit or None]
    print(f"[bank] {len(z['key'])} rows, {len(rows)} held-out")

    for r in R_GRID:
        rows_f, cols_f = region_factor(r)
        print(f"  r={r:2d} partition {rows_f}x{cols_f}  "
              f"cell {GRID // rows_f}x{GRID // cols_f} = "
              f"{N_VIS // r} tokens")

    rec = {r: dict(key=[], u=[], a_per_cell=[], n_empty=[], cap_norm=[],
                   pool_norm=[], ess={t: [] for t in TAU_GRID}) for r in R_GRID}
    imp_ess16, u_ess16, u_pct = [], [], []
    t0 = time.time()
    for j, i in enumerate(rows):
        key = z["key"][i]
        for r in R_GRID:
            vis, imp, s0, A, drop, u = instance_state(z, i, r)
            rid = partition_ids(r, device=DEV)
            if r == 16:
                u_pct.append(np.percentile(u[drop].cpu().numpy(), [10, 50, 90]))
                u_ess16.append([build_capsules(vis, imp, u, A, drop, r,
                                               "residual", "spatial", t)[1]["ess"]
                                for t in TAU_GRID])
                imp_ess16.append([build_capsules(vis, imp, u, A, drop, r,
                                                 "imp", "spatial", TAU, t)[1]["ess"]
                                  for t in TAU_GRID])
            for t in TAU_GRID:
                caps, st = build_capsules(vis, imp, u, A, drop, r, "residual",
                                          "spatial", t)
                rec[r]["ess"][t].append(st["ess"])
                if t == TAU:
                    rec[r]["cap_norm"].append(st["cap_norm"])
                    rec[r]["pool_norm"].append(
                        float(vis[drop].norm(dim=1).mean()))
                    rec[r]["n_empty"].append(st["n_empty"])
            rec[r]["key"].append(key)
            rec[r]["u"].append(u.half().cpu().numpy())
            cnt = torch.zeros(r, device=DEV).index_add_(
                0, rid[A], torch.ones(A.numel(), device=DEV))
            rec[r]["a_per_cell"].append(cnt.cpu().numpy())
        if (j + 1) % 25 == 0:
            print(f"  {j+1}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------- report --
    out = dict(bank=BANK, n=len(rows), evict_rule=EVICT_RULE, tau=TAU,
               tau_grid=list(TAU_GRID), r_grid=list(R_GRID))
    print("\n--- u (unexplainedness), dropped tokens, held-out 150 ---")
    P = np.array(u_pct)
    print(f"  p10 {P[:,0].mean():.4f}   p50 {P[:,1].mean():.4f}   "
          f"p90 {P[:,2].mean():.4f}")
    out["u_dropped_pct"] = dict(p10=float(P[:, 0].mean()),
                                p50=float(P[:, 1].mean()),
                                p90=float(P[:, 2].mean()))

    print("\n--- effective sample size per capsule (mean over 150) ---")
    print("   r  " + "".join(f"   tau={t:<5}" for t in TAU_GRID))
    out["ess"] = {}
    for r in R_GRID:
        e = [np.mean(rec[r]["ess"][t]) for t in TAU_GRID]
        out["ess"][r] = {t: float(v) for t, v in zip(TAU_GRID, e)}
        print(f"  {r:2d}  " + "".join(f"  {v:9.2f}" for v in e) +
              f"    (pool {N_VIS // r} tokens/cell)")
    eu = np.mean(u_ess16, axis=0)
    ei = np.mean(imp_ess16, axis=0)
    print("\n  importance merge, same grid:")
    print("      " + "".join(f"  {v:9.2f}" for v in ei))
    # The controlled comparison: the tau_imp whose ESS matches the residual
    # merge's ESS at the primary temperature.  Interpolated in log tau.
    target = float(eu[list(TAU_GRID).index(TAU)])
    tau_imp = float(np.exp(np.interp(target, ei, np.log(TAU_GRID))))
    out["tau_imp_matched"] = dict(target_ess=target, tau_imp=tau_imp,
                                  ess_at=float(np.interp(target, ei, ei)))
    print(f"\n  -> matching ESS {target:.2f}: tau_imp = {tau_imp:.4f} "
          f"(gives {np.interp(target, ei, ei):.2f})")
    print(f"     u-merge ESS at tau={TAU}: {target:.2f}; "
          f"u-merge ESS by tau: {dict(zip(TAU_GRID, np.round(eu, 2)))}")

    print("\n--- capsule norm / pooled-token norm (tau=%.2f) ---" % TAU)
    out["norm_ratio"] = {}
    for r in R_GRID:
        ratio = np.array(rec[r]["cap_norm"]) / np.array(rec[r]["pool_norm"])
        out["norm_ratio"][r] = dict(median=float(np.median(ratio)),
                                    p10=float(np.percentile(ratio, 10)),
                                    p90=float(np.percentile(ratio, 90)))
        print(f"  r={r:2d}  median {np.median(ratio):.3f}  "
              f"[{np.percentile(ratio,10):.3f}, {np.percentile(ratio,90):.3f}]")
    worst = max(v["median"] for v in out["norm_ratio"].values())
    out["norm_restore_arm"] = bool(worst < 0.85)
    print(f"  -> worst median ratio {worst:.3f}: norm-restore arm "
          f"{'KEPT (shrink is material)' if worst < 0.85 else 'DROPPED (shrink is small)'}")

    print("\n--- partition occupancy ---")
    out["occupancy"] = {}
    for r in R_GRID:
        a = np.stack(rec[r]["a_per_cell"])
        empt = int(np.sum(rec[r]["n_empty"]))
        out["occupancy"][r] = dict(anchor_per_cell_min=int(a.min()),
                                   anchor_per_cell_med=float(np.median(a)),
                                   empty_cells_total=empt)
        print(f"  r={r:2d}  anchors/cell min {int(a.min())} median "
              f"{np.median(a):.1f};  empty cells over 150 instances: {empt}")

    # ---- the residual maps the SHUF arm injects ---------------------------
    maps = {f"u_r{r}": np.stack(rec[r]["u"]).astype(np.float16) for r in R_GRID}
    np.savez(os.path.join(OUTPUT_DIR, f"{args.tag}_umaps.npz"),
             key=np.array(rec[16]["key"]), **maps)
    out["umaps"] = f"{args.tag}_umaps.npz"
    out["region_maps"] = {r: region_map(r).tolist() for r in R_GRID}
    with open(os.path.join(OUTPUT_DIR, f"{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[done] -> {args.tag}.json  {args.tag}_umaps.npz  "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
