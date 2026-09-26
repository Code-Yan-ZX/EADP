"""
M8 step 2 -- the LLM-level zeroth-order estimator, and the resolution study
that decides whether it exists at all.

The estimator the M8 hypothesis needs
-------------------------------------
ZO-P (step 1) is cheap because the projector is cheap, and that is exactly why
it cannot see what the gradient teacher sees: it is a query-free, token-local
function of the pre-merger feature.  The estimator that could in principle
break the depth wall is the same zeroth-order idea one level up:

    J(v)      = max logit at the last prompt position          (P1-G2's own J)
    s(i)      = || [J(v + h*u_i) - J(v - h*u(i))] / 2h ||      u supported on i
    ZO-L(i)   = mean_j | dJ_{i,j} |

which is the central-difference approximation of the very gradient the teacher
computes by backward.  It is *targeted* by construction: it needs one forward
per token, so it can only ever be afforded on a nominated pool -- which is the
M8 hypothesis stated as an estimator.

What this script measures, and why in this order
------------------------------------------------
Before asking whether ZO-L ranks the teacher's head, it asks whether ZO-L is
*measurable*.  The deployed model is bf16, and a single visual token carries
1/1024 of the last position's attention.  Part A calibrates the per-token
effect size against the representation's own quantum:

    A1  the quantisation step of the response, measured as the logits' bf16 ulp
    A2  dJ as a function of how many tokens are zeroed, k = 1 .. 256, which
        gives the per-token effect in the linear regime
    A3  the resulting signal-to-quantum ratio for a ONE-token perturbation --
        if it is below 1 the estimator has no resolution and no ranking of its
        output can mean anything

Part B then runs the estimator on a real pool and reports its split-half
reliability and its correlation with the gradient teacher, so that the verdict
rests on a measurement rather than on the calibration argument alone.

Usage
    python scripts/discovery/m8_zol.py --limit 24
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
from m8_common import ORIENT, dump_json, topk_local                  # noqa: E402
from m6_common import Dropped                                      # noqa: E402

TAG = "m8_zol"
KS = (1, 2, 4, 8, 16, 32, 64, 256)     # tokens zeroed, for the effect-size curve
POOL = 32                              # tokens audited per instance
H_GRID = (0.01, 0.1, 0.5)              # official step, and two larger ones
SEEDS = (0, 1)                         # split-half
CHUNK = 32


class ZOLHarness:
    """PRE-LLM delivery, the teacher's own objective, no gold answer anywhere."""

    def __init__(self, model):
        self.vlm = model
        self.model = model.model
        self.dev = next(model.model.parameters()).device
        self.emb = self.model.get_input_embeddings()
        self.image_token_id = self.model.config.image_token_id

    def prep(self, message, ds, vis):
        inputs = self.vlm._processor_inputs(
            self.vlm._build_messages(message, dataset=ds))
        ids = inputs["input_ids"]
        pos = (ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
        s, e = int(pos[0]), int(pos[-1]) + 1
        assert e - s == N_VIS, f"{e - s} visual tokens"
        with torch.no_grad():
            emb = self.emb(ids)
        return dict(prefix=emb[0, :s].clone(), suffix=emb[0, e:].clone(),
                    vis=vis.to(self.dev))

    @torch.no_grad()
    def logits(self, prep, blocks):
        """blocks: (B, N_VIS, D) visual blocks -> (B, V) last-position logits."""
        outs = []
        for c0 in range(0, blocks.shape[0], CHUNK):
            b = blocks[c0:c0 + CHUNK]
            B = b.shape[0]
            seq = torch.cat([prep["prefix"].unsqueeze(0).expand(B, -1, -1), b,
                             prep["suffix"].unsqueeze(0).expand(B, -1, -1)],
                            dim=1).contiguous()
            L = seq.shape[1]
            am = torch.ones(B, L, dtype=torch.long, device=self.dev)
            p1 = torch.arange(L, device=self.dev).view(1, 1, -1) \
                .expand(3, B, -1).contiguous()
            o = self.model(inputs_embeds=seq, attention_mask=am, position_ids=p1,
                           use_cache=False, return_dict=True,
                           output_hidden_states=True)
            outs.append((o.logits[:, -1].float(),
                         o.hidden_states[-1][:, -1].float()))
            del o, seq, am, p1
            torch.cuda.empty_cache()
        return (torch.cat([x[0] for x in outs]),
                torch.cat([x[1] for x in outs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    args = ap.parse_args()

    bank = load_m5_bank()                 # carries the bf16 `vis` cache
    bank["fi"] = {str(nm): j for j, nm in enumerate(bank["feature_names"])}
    bank["n"] = len(bank["key"])
    dropped = Dropped(bank)
    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0), max_new_tokens=8)
    model.model.eval()
    torch.set_grad_enabled(False)
    H = ZOLHarness(model)

    items = [it for it in bank_items(model, bank, splits=tuple(args.splits))]
    if args.limit:
        items = items[:args.limit]
    print(f"[ZO-L] {len(items)} instances from {args.splits}; pool={POOL} "
          f"tokens; h in {H_GRID}")

    from scipy.stats import spearmanr
    rows, calib = [], []
    t0 = time.time()
    for j, it in enumerate(items):
        i = it["bank_row"]
        prep = H.prep(it["msg"], it["ds"], vis_row(bank, i, H.dev))
        vis = prep["vis"]
        vn = vis.float().norm(dim=-1)
        base, base_h = H.logits(prep, vis.unsqueeze(0))
        base = base[0]
        J0 = float(base.max())
        k0 = int(torch.argmax(base))
        # bf16 quantum of the response: 7 stored mantissa bits, so the step
        # is 2^(e-7) for 2^e <= |J| < 2^(e+1) -- NOT |J| * 2^-8, which is
        # 1.5x too small in this exponent band.
        ulp = 2.0 ** (int(np.floor(np.log2(abs(J0)))) - 7)

        # ---- A2: effect size vs how many tokens are zeroed ----------------
        order = np.argsort(-vn.cpu().numpy())      # deterministic, scale-ordered
        blocks = [vis.clone()]
        for k in KS:
            v = vis.clone()
            v[torch.as_tensor(order[:k], device=H.dev)] = 0.0
            blocks.append(v)
        lg, hs = H.logits(prep, torch.stack(blocks, 0))
        dJ = (lg[:, k0] - lg[0, k0]).cpu().numpy()
        dn = (lg - lg[0]).norm(dim=-1).cpu().numpy()
        dh = (hs - hs[0]).norm(dim=-1).cpu().numpy()
        calib.append(dict(key=it["key"], J0=J0, ulp=ulp, k=list(KS),
                          dJ=dJ[1:].tolist(), dlogits=dn[1:].tolist(),
                          dh_last=dh[1:].tolist(),
                          h_last_norm=float(base_h[0].norm())))

        # ---- B: the estimator on a real pool ------------------------------
        pool = topk_local(bank, dropped, i, "cos_s0c", POOL)
        gidx = dropped.drop[i][pool]
        g2p = bank["g2"][i][gidx]
        res = {}
        for h in H_GRID:
            sc = (vn / np.sqrt(vis.shape[1])).to(H.dev, vis.dtype)
            got = []
            for seed in SEEDS:
                U = torch.randn(POOL, vis.shape[1],
                                generator=torch.Generator().manual_seed(seed),
                                dtype=torch.float32)
                U = (U / U.norm(dim=-1, keepdim=True)).to(H.dev, vis.dtype)
                d = (h * sc[gidx].unsqueeze(1) * U).to(vis.dtype)
                ar = torch.arange(POOL, device=H.dev)
                vp = vis.unsqueeze(0).expand(POOL, -1, -1).clone()
                vm = vp.clone()
                vp[ar, gidx] = vis[gidx] + d
                vm[ar, gidx] = vis[gidx] - d
                vb = torch.stack([vp, vm], 1).reshape(2 * POOL, N_VIS, -1)
                del vp, vm
                lg2, hs2 = H.logits(prep, vb)
                got.append((((lg2[:POOL, k0] - lg2[POOL:, k0]) / (2 * h))
                            .cpu().numpy(),
                            ((lg2[:POOL] - lg2[POOL:]).norm(dim=-1) / (2 * h))
                            .cpu().numpy(),
                            ((hs2[:POOL] - hs2[POOL:]).norm(dim=-1) / (2 * h))
                            .cpu().numpy()))
            a, b = got
            res[str(h)] = dict(
                dJ_a=a[0].tolist(), dJ_b=b[0].tolist(),
                dL_a=a[1].tolist(), dL_b=b[1].tolist(),
                dH_a=a[2].tolist(), dH_b=b[2].tolist(),
                rho_H=float(spearmanr(a[2], b[2]).statistic),
                rhoT_H=float(spearmanr(a[2], np.abs(g2p)).statistic),
                spread_H=float(np.std(a[2]) / max(np.mean(a[2]), 1e-12)),
                rho_J=float(spearmanr(np.abs(a[0]), np.abs(b[0])).statistic),
                rho_L=float(spearmanr(a[1], b[1]).statistic),
                rhoT_J=float(spearmanr(np.abs(a[0]), np.abs(g2p)).statistic),
                rhoT_L=float(spearmanr(a[1], np.abs(g2p)).statistic),
                spread_J=float(np.std(np.abs(a[0])) / max(np.mean(np.abs(a[0])), 1e-12)),
                spread_L=float(np.std(a[1]) / max(np.mean(a[1]), 1e-12)))
        rows.append(dict(key=it["key"], ds=it["ds"], J0=J0, k0=k0, ulp=ulp,
                         pool_size=int(pool.size), res=res,
                         g2p=np.abs(g2p).tolist()))
        if (j + 1) % 4 == 0 or j == len(items) - 1:
            r = rows[-1]["res"]["0.1"]
            print(f"  {j+1}/{len(items)} {time.time()-t0:.0f}s  "
                  f"dJ(k=1)={calib[-1]['dJ'][0]:+.3f} ulp={ulp:.3f}  "
                  f"ZO-L h=0.1: rhoJ={r['rho_J']:.2f} rhoT_J={r['rhoT_J']:+.2f} "
                  f"rhoT_L={r['rhoT_L']:+.2f}", flush=True)

    dump_json(f"{TAG}.json", dict(config=vars(args), n=len(items), ks=list(KS),
                                  pool=POOL, h_grid=list(H_GRID),
                                  seeds=list(SEEDS), rows=rows, calib=calib,
                                  wall_seconds=float(time.time() - t0)))
    print(f"\n[saved] {TAG}.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
