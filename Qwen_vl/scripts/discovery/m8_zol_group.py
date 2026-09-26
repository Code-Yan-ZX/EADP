"""
M8 step 7 -- the last escape hatch for a cheap LLM-level estimator.

Per-token sensitivity needs one forward per token, and §5 shows a single
token's response is below the representation's quantum.  There is one design
that would evade both: ZOO-Prune's own trick, applied one level up.  Perturb
*every* pool token at once with a shared direction, and use a structured
(orthogonal / Hadamard) sign pattern across K forwards to decode the per-token
values by inversion:

    forward k perturbs token i by  h * H[k,i] * u      (u shared, unit)
    dJ_k / h  =  sum_i H[k,i] * (g_i . u)
    =>  (g_i . u)  =  (1/K) sum_k H[k,i] * (dJ_k / h)      for K >= P

This is a real estimator and it costs K forwards instead of 2P.  It works iff
the aggregate response `sum_i H[k,i] (g_i . u)` rises above the output's
quantum -- and the aggregate grows only as sqrt(P) while the quantum is fixed.

This script measures that, rather than arguing it.  One batch per instance:

    identity                        the reference
    zero the whole pool             a deterministic, additive positive control
    perturb the pool, 8 shared dirs the aggregate the Hadamard design would use
    perturb all 768 dropped, 8 dirs the same at the largest pool available

Usage
    python scripts/discovery/m8_zol_group.py --limit 8
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
from m6_common import Dropped                                        # noqa: E402
from m8_common import dump_json, topk_local                          # noqa: E402

TAG = "m8_zol_group"
H_REL = 0.1
N_DIRS = 8
POOL = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()

    bank = load_m5_bank()
    bank["fi"] = {str(nm): j for j, nm in enumerate(bank["feature_names"])}
    bank["n"] = len(bank["key"])
    dropped = Dropped(bank)
    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0), max_new_tokens=8)
    model.model.eval()
    torch.set_grad_enabled(False)
    dev = next(model.model.parameters()).device
    emb = model.model.get_input_embeddings()
    image_token_id = model.model.config.image_token_id

    items = bank_items(model, bank, splits=("test",))[:args.limit]
    rows = []
    t0 = time.time()
    for j, it in enumerate(items):
        i = it["bank_row"]
        inputs = model._processor_inputs(
            model._build_messages(it["msg"], dataset=it["ds"]))
        ids = inputs["input_ids"]
        pos = (ids[0] == image_token_id).nonzero(as_tuple=True)[0]
        s, e = int(pos[0]), int(pos[-1]) + 1
        with torch.no_grad():
            E = emb(ids)
        prefix, suffix = E[0, :s], E[0, e:]
        vis = vis_row(bank, i, dev)
        vn = vis.float().norm(dim=-1)
        sc = (vn / np.sqrt(vis.shape[1])).to(dev, vis.dtype)

        pool = torch.as_tensor(dropped.drop[i][
            topk_local(bank, dropped, i, "cos_s0c", POOL)], device=dev)
        allD = torch.as_tensor(dropped.drop[i], device=dev)

        blocks = [vis.clone()]
        v = vis.clone()
        v[pool] = 0.0
        blocks.append(v)
        g = torch.Generator().manual_seed(1234)
        for k in range(N_DIRS):
            U = torch.randn(N_VIS, vis.shape[1], generator=g, dtype=torch.float32)
            U = (U / U.norm(dim=-1, keepdim=True)).to(dev, vis.dtype)
            v = vis.clone()
            v[pool] = vis[pool] + (H_REL * sc[pool].unsqueeze(1) * U[pool])
            blocks.append(v)
        for k in range(N_DIRS):
            U = torch.randn(N_VIS, vis.shape[1], generator=g, dtype=torch.float32)
            U = (U / U.norm(dim=-1, keepdim=True)).to(dev, vis.dtype)
            v = vis.clone()
            v[allD] = vis[allD] + (H_REL * sc[allD].unsqueeze(1) * U[allD])
            blocks.append(v)

        vb = torch.stack(blocks, 0)
        B = vb.shape[0]
        seq = torch.cat([prefix.unsqueeze(0).expand(B, -1, -1), vb,
                         suffix.unsqueeze(0).expand(B, -1, -1)], dim=1).contiguous()
        L = seq.shape[1]
        am = torch.ones(B, L, dtype=torch.long, device=dev)
        p1 = torch.arange(L, device=dev).view(1, 1, -1).expand(3, B, -1).contiguous()
        with torch.no_grad():
            o = model.model(inputs_embeds=seq, attention_mask=am, position_ids=p1,
                            use_cache=False, return_dict=True)
        lg = o.logits[:, -1].float()
        k0 = int(torch.argmax(lg[0]))
        J0 = float(lg[0, k0])
        ulp = 2.0 ** (int(np.floor(np.log2(abs(J0)))) - 7)
        dJ = (lg[:, k0] - lg[0, k0]).cpu().numpy()
        rows.append(dict(key=it["key"], J0=J0, ulp=ulp,
                         zero_pool=float(dJ[1]),
                         pool_shared=list(dJ[2:2 + N_DIRS].astype(float)),
                         all768_shared=list(dJ[2 + N_DIRS:].astype(float)),
                         n_pool=int(pool.numel()), n_all=int(allD.numel())))
        del o, seq, am, p1, vb, blocks
        torch.cuda.empty_cache()
        print(f"  {j+1}/{len(items)} {time.time()-t0:.0f}s  ulp={ulp:.3f}  "
              f"zero-pool {dJ[1]:+.3f}  pool-dirs {np.round(dJ[2:2+N_DIRS],3)}  "
              f"768-dirs {np.round(dJ[2+N_DIRS:],3)}", flush=True)

    z = np.array([r["zero_pool"] for r in rows])
    ps = np.array([r["pool_shared"] for r in rows])
    a7 = np.array([r["all768_shared"] for r in rows])
    us = np.array([r["ulp"] for r in rows])
    agg = dict(
        n=len(rows), h_rel=H_REL, n_dirs=N_DIRS, pool=POOL,
        ulp_mean=float(us.mean()),
        zero_pool=dict(mean=float(z.mean()), sd=float(z.std()),
                       in_quanta=float(np.mean(z / us))),
        pool_shared=dict(mean=float(ps.mean()), mean_abs=float(np.abs(ps).mean()),
                         sd_over_dirs=float(ps.std(axis=1).mean()),
                         in_quanta=float(np.mean(np.abs(ps) / us[:, None])),
                         frac_below_half_quantum=float(np.mean(
                             np.abs(ps) < 0.5 * us[:, None]))),
        all768_shared=dict(mean=float(a7.mean()),
                           mean_abs=float(np.abs(a7).mean()),
                           sd_over_dirs=float(a7.std(axis=1).mean()),
                           in_quanta=float(np.mean(np.abs(a7) / us[:, None])),
                           frac_below_half_quantum=float(np.mean(
                               np.abs(a7) < 0.5 * us[:, None]))),
        rows=rows)
    dump_json(f"{TAG}.json", agg)
    print(f"\n[mean over {len(rows)} instances]  quantum = {us.mean():.3f}")
    print(f"  zero the whole {POOL}-token pool : {z.mean():+.3f}  "
          f"({np.mean(z/us):.2f} quanta)  sd {z.std():.3f}")
    print(f"  perturb pool, shared dirs      : |dJ| "
          f"{np.mean(np.abs(ps)):.3f}  ({np.mean(np.abs(ps)/us[:,None]):.2f} quanta)  "
          f"sd across dirs {ps.std(axis=1).mean():.3f}  "
          f"below half a quantum on "
          f"{np.mean(np.abs(ps) < 0.5*us[:,None])*100:.0f}% of draws")
    print(f"  perturb all 768 dropped, dirs  : |dJ| "
          f"{np.mean(np.abs(a7)):.3f}  ({np.mean(np.abs(a7)/us[:,None]):.2f} quanta)  "
          f"sd across dirs {a7.std(axis=1).mean():.3f}")


if __name__ == "__main__":
    main()
