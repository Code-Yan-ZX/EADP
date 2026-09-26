"""
M8 step 8 -- the LLM-level estimator, put through the same Phase-1 gate.

`m8_zol.py` measures whether ZO-L is *resolvable*.  This measures what the
resolution buys: the same depth-wall table §4 runs for ZO-P, run for the
estimator the M8 hypothesis actually needs.

The estimator, on the P1 pool of 32 dropped tokens:

    score(i) = mean_j | J(v + h*u_{i,j}) - J(v - h*u_{i,j}) | / 2h
               u_{i,j} supported on token i only, drawn independently

with `m` directions per token, central differences, at the 1024-token context
the teacher itself uses.  Rows per instance = 2*m*32.

Reported against the identical controls §4 uses: random-in-pool, `cos_s0c`,
and the in-pool oracle.  The gate is the brief's: mean teacher rank < 40 at
r = 8.

Usage
    python scripts/discovery/m8_zol_phase1.py --limit 60 --dirs 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                        # noqa: E402
from common import OUTPUT_DIR, eadp_model_name                       # noqa: E402
from m5_common import BUDGET, N_VIS, bank_items, load_m5_bank, vis_row  # noqa: E402
from m6_common import Dropped, paired_bootstrap                      # noqa: E402
from m8_common import (BOOT_SEED, ORIENT, RS, dump_json, evaluate,   # noqa: E402
                       rank_by_score, r_cheap_in_pool, r_oracle_in_pool,
                       r_random_in_pool, topk_local)
from m8_zol import H_GRID, ZOLHarness                                # noqa: E402
from scipy.stats import spearmanr                                    # noqa: E402

TAG = "m8_zol_phase1"
POOL = 32
H_REL = 0.1          # scale-relative step: the largest the response stays sane at
RANDOM_SEEDS = tuple(range(20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--dirs", type=int, default=2)
    ap.add_argument("--h", type=float, default=H_REL)
    args = ap.parse_args()

    bank = load_m5_bank()
    bank["fi"] = {str(nm): j for j, nm in enumerate(bank["feature_names"])}
    bank["n"] = len(bank["key"])
    dropped = Dropped(bank)
    hold = [i for i in range(bank["n"]) if bank["split"][i] in ("val", "test")]
    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0), max_new_tokens=8)
    model.model.eval()
    torch.set_grad_enabled(False)
    H = ZOLHarness(model)

    items = [it for it in bank_items(model, bank, splits=("val", "test"))
             if it["bank_row"] in hold][:args.limit]
    print(f"[ZO-L phase1] {len(items)} instances, pool {POOL}, m={args.dirs} "
          f"directions, h_rel={args.h}; rows/instance = {2*args.dirs*POOL}")

    rows, t0 = [], time.time()
    for j, it in enumerate(items):
        i = it["bank_row"]
        prep = H.prep(it["msg"], it["ds"], vis_row(bank, i, H.dev))
        vis = prep["vis"]
        vn = vis.float().norm(dim=-1)
        sc = (vn / np.sqrt(vis.shape[1])).to(H.dev, vis.dtype)
        pool = topk_local(bank, dropped, i, "cos_s0c", POOL)
        gidx = torch.as_tensor(dropped.drop[i][pool], device=H.dev)

        base = H.logits(prep, vis.unsqueeze(0))[0][0]
        k0 = int(torch.argmax(base))
        ulp = abs(float(base[k0])) * 2.0 ** -8

        ar = torch.arange(POOL, device=H.dev)
        dJ = np.zeros(POOL)
        dL = np.zeros(POOL)
        for m in range(args.dirs):
            U = torch.randn(POOL, vis.shape[1],
                            generator=torch.Generator().manual_seed(m),
                            dtype=torch.float32)
            U = (U / U.norm(dim=-1, keepdim=True)).to(H.dev, vis.dtype)
            d = (args.h * sc[gidx].unsqueeze(1) * U).to(vis.dtype)
            vp = vis.unsqueeze(0).expand(POOL, -1, -1).clone()
            vm = vp.clone()
            vp[ar, gidx] = vis[gidx] + d
            vm[ar, gidx] = vis[gidx] - d
            vb = torch.stack([vp, vm], 1).reshape(2 * POOL, N_VIS, -1)
            del vp, vm
            lg, _ = H.logits(prep, vb)
            dJ += ((lg[:POOL, k0] - lg[POOL:, k0]) / (2 * args.h)).cpu().numpy()
            dL += ((lg[:POOL] - lg[POOL:]).norm(dim=-1) / (2 * args.h)).cpu().numpy()
            del vb, lg
        dJ /= args.dirs
        dL /= args.dirs

        g2p = np.abs(bank["g2"][i][dropped.drop[i][pool]])
        rows.append(dict(key=it["key"], bank_row=i, pool=pool.tolist(),
                         dJ=dJ.tolist(), dL=dL.tolist(), ulp=ulp,
                         n_distinct_dJ=int(np.unique(np.round(np.abs(dJ), 6)).size),
                         rhoT_J=float(spearmanr(np.abs(dJ), g2p).statistic),
                         rhoT_L=float(spearmanr(dL, g2p).statistic),
                         frac_zero_dJ=float(np.mean(np.abs(dJ) < 1e-9))))
        del prep, vis
        torch.cuda.empty_cache()
        if (j + 1) % 5 == 0 or j == len(items) - 1:
            print(f"  {j+1}/{len(items)} {time.time()-t0:.0f}s  "
                  f"rhoT(|dJ|)={np.mean([r['rhoT_J'] for r in rows]):+.3f}  "
                  f"frac dJ==0 {np.mean([r['frac_zero_dJ'] for r in rows]):.2f}  "
                  f"distinct |dJ| {np.mean([r['n_distinct_dJ'] for r in rows]):.1f}",
                  flush=True)

    sub = Dropped(bank)
    sub.n = len(rows)
    sub.drop = [dropped.drop[r["bank_row"]] for r in rows]
    sub.tr = [dropped.tr[r["bank_row"]] for r in rows]
    sub.head = {q: [dropped.head[q][r["bank_row"]] for r in rows]
                for q in dropped.head}

    out = dict(config=vars(args), n=len(rows), boot_seed=BOOT_SEED,
               mean_rhoT_J=float(np.mean([r["rhoT_J"] for r in rows])),
               mean_rhoT_L=float(np.mean([r["rhoT_L"] for r in rows])),
               mean_frac_zero_dJ=float(np.mean([r["frac_zero_dJ"] for r in rows])),
               mean_distinct_dJ=float(np.mean([r["n_distinct_dJ"] for r in rows])),
               mean_ulp=float(np.mean([r["ulp"] for r in rows])), grid={})
    for r in RS:
        cell = {}
        rnd = np.stack([evaluate([r_random_in_pool(np.asarray(rw["pool"]), r, s, k)
                                  for k, rw in enumerate(rows)], sub, r)["recall"]
                        for s in RANDOM_SEEDS])
        cell["random"] = dict(recall_mean=float(rnd.mean()))
        for tag, key, sign in (("zol_absdJ", "dJ", -1), ("zol_dJ", "dJ", +1),
                               ("zol_dL", "dL", -1)):
            picks = [rank_by_score(np.asarray(rw["pool"]),
                                   sign * np.asarray(rw[key]))[:r] for rw in rows]
            ev = evaluate(picks, sub, r)
            cell[tag] = dict(recall_mean=ev["recall_mean"],
                             mean_teacher_rank=ev["mean_teacher_rank"],
                             median_teacher_rank=ev["median_teacher_rank"],
                             top8=ev["top8"], top16=ev["top16"], top32=ev["top32"])
        for tag in ("cos_s0c",):
            # index the BANK by bank row and the panel by position: `sub.drop[k]`
            # is `dropped.drop[bank_row]`, and `r_cheap_in_pool` reads both.
            picks = [r_cheap_in_pool(bank, dropped, rw["bank_row"],
                                     np.asarray(rw["pool"]), tag, r)
                     for rw in rows]
            ev = evaluate(picks, sub, r)
            cell[tag] = dict(recall_mean=ev["recall_mean"],
                             mean_teacher_rank=ev["mean_teacher_rank"],
                             median_teacher_rank=ev["median_teacher_rank"])
        picks = [r_oracle_in_pool(sub, k, np.asarray(rw["pool"]), r)
                 for k, rw in enumerate(rows)]
        ev = evaluate(picks, sub, r)
        cell["oracle"] = dict(recall_mean=ev["recall_mean"],
                              mean_teacher_rank=ev["mean_teacher_rank"],
                              median_teacher_rank=ev["median_teacher_rank"])
        out["grid"][f"r{r}"] = cell
        print(f"  r={r}: " + "  ".join(
            f"{k}={cell[k]['recall_mean']:.3f}"
            + (f"/{cell[k]['mean_teacher_rank']:.0f}"
               if cell[k].get("mean_teacher_rank") is not None else "")
            for k in ("random", "cos_s0c", "zol_absdJ", "oracle")))

    out["rows"] = rows
    dump_json(f"{TAG}.json", out)
    print(f"\n[saved] {TAG}.json  ({time.time()-t0:.0f}s)")
    print(f"  mean rho(ZO-L |dJ|, |teacher|) = {out['mean_rhoT_J']:+.4f}")
    print(f"  mean rho(ZO-L dL,   |teacher|) = {out['mean_rhoT_L']:+.4f}")


if __name__ == "__main__":
    main()
