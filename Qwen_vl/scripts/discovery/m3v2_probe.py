"""
M3-v2 -- mechanism probe: is the cross-token structure USED, or merely present?

The ablation table says A ~= B ~= C: query conditioning and retained-set
context buy nothing at the level of the head metric.  That has two very
different readings, and the head metric alone cannot tell them apart:

    (i)  the structure engages but carries no usable information, or
    (ii) the structure never engages -- the trained head simply ignores it.

This probe separates them at inference.  For each trained checkpoint it scores
every val instance with the full model and again with ONE input block replaced
by zeros, and reports how much the deployed top-16 actually moves.  If zeroing
`q` (or `c`) leaves the rescue set essentially unchanged, the block is inert in
the deployed path and reading (ii) holds; if it moves a lot while the metric
stays flat, the block is engaged and reading (i) holds.

It also reports the cross-attention's own entropy against the uniform value, so
a collapsed (uniform) attention map is visible directly rather than inferred.

Usage
    python scripts/discovery/m3v2_probe.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUT_DIR                                       # noqa: E402
from m3v2_common import (BankV2, build_auditor, fit_stats,           # noqa: E402
                         instance_head_metrics, standardise)

CFGS = [f"{o}{a}" for o in ("h1", "h2", "h3") for a in ("A", "B", "C")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--out", default="m3v2_probe")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bank = BankV2()
    stats = fit_stats(bank, bank.rows("fit"))
    val = bank.rows("val")

    # per-instance inputs, standardised exactly as the pruner does
    inputs = []
    for i in val:
        vis = torch.from_numpy(bank.vis[i:i + 1]).to(device)
        txt = torch.from_numpy(bank.txt[i:i + 1]).to(device)
        X = torch.from_numpy(bank.X[i:i + 1]).to(device)
        s0 = torch.from_numpy(bank.s0[i:i + 1]).to(device)
        mask = torch.arange(txt.shape[1], device=device)[None, :] < int(bank.txt_len[i])
        st = {k: torch.from_numpy(np.asarray(stats[k], np.float32)).to(device)
              for k in stats}
        with torch.no_grad():
            inputs.append((standardise(vis, st["mu_vis"], st["sd_vis"]),
                           standardise(txt, st["mu_txt"], st["sd_txt"]),
                           standardise(X, st["mu_hand"], st["sd_hand"]), s0, mask))

    report = {}
    for cfg in CFGS:
        for seed in args.seeds:
            tag = f"m3v2_auditor_{args.grid}{cfg}_s{seed}"
            p = os.path.join(OUTPUT_DIR, f"{tag}.pt")
            if not os.path.exists(p):
                continue
            ck = torch.load(p, map_location="cpu", weights_only=False)
            model = build_auditor(ck, device).eval()
            churn, ent = {k: [] for k in model.block_order}, {"xt": [], "ctx": []}
            per_full = []
            for (i, (v, t, x, s0, mask)) in zip(val, inputs):
                with torch.no_grad():
                    parts = model.blocks(v, t, x, s0, mask)
                    full = model.head_from(parts)
                    d = torch.from_numpy(bank.drop[i]).to(device)
                    sf = full[0][d].float().cpu().numpy()
                    top = set(np.argsort(-sf)[:16].tolist())
                    for blk in model.block_order:
                        s2 = model.head_from(parts, zero=blk)[0][d].float().cpu().numpy()
                        churn[blk].append(
                            1.0 - len(top & set(np.argsort(-s2)[:16].tolist())) / 16.0)
                    # attention entropies, against their uniform values
                    if model.use_query:
                        Th = model.proj_t(t)
                        q = model.xt.ln_q(parts["V"])
                        k = model.xt.ln_kv(Th)
                        B_, Lq, dd = q.shape
                        h, dh = model.xt.h, model.xt.dh
                        qq = model.xt.q(q).view(B_, Lq, h, dh).transpose(1, 2)
                        kk = model.xt.k(k).view(B_, Th.shape[1], h, dh).transpose(1, 2)
                        a = (qq @ kk.transpose(-1, -2)) / (dh ** 0.5)
                        a = a.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(-1)
                        # (h, n_dropped, Lk): entropy per head per candidate, then
                        # averaged -- normalised by log(Lk), so 1.0 is uniform.
                        pr = a[0][:, d, :]
                        hh = -(pr * pr.clamp_min(1e-12).log()).sum(-1)
                        ent["xt"].append(float(hh.mean()) / np.log(Th.shape[1]))
                    if model.use_retained:
                        s0v = parts["V"].gather(
                            1, s0[:, :, None].expand(-1, -1, parts["V"].shape[-1]))
                        slots = model.slot_attn(
                            model.slots.unsqueeze(0).expand(1, -1, -1), s0v)
                        q = model.ctx_attn.ln_q(parts["V"])
                        k = model.ctx_attn.ln_kv(slots)
                        h, dh = model.ctx_attn.h, model.ctx_attn.dh
                        qq = model.ctx_attn.q(q).view(1, -1, h, dh).transpose(1, 2)
                        kk = model.ctx_attn.k(k).view(1, slots.shape[1], h, dh).transpose(1, 2)
                        a = ((qq @ kk.transpose(-1, -2)) / (dh ** 0.5)).softmax(-1)
                        pr = a[0][:, d, :]
                        hh = -(pr * pr.clamp_min(1e-12).log()).sum(-1)
                        ent["ctx"].append(float(hh.mean()) / np.log(slots.shape[1]))
                per_full.append(instance_head_metrics(bank.g2[i], bank.drop[i], sf))
            report[tag] = dict(
                blocks=list(model.block_order),
                churn_top16={k: float(np.mean(v)) for k, v in churn.items()},
                attn_entropy_frac={k: float(np.mean(v)) for k, v in ent.items() if v},
                val_top16_recall_at_16=float(np.mean(
                    [m["top16_recall@16"] for m in per_full])),
                note="churn_top16 = fraction of the deployed top-16 that changes "
                     "when the block is zeroed; attn_entropy_frac = attention "
                     "entropy / log(n_keys), 1.0 = uniform")
            print(f"  {tag:22s} val {report[tag]['val_top16_recall_at_16']:.4f}  "
                  f"churn {report[tag]['churn_top16']}  "
                  f"entropy {report[tag]['attn_entropy_frac']}", flush=True)
            del model
            torch.cuda.empty_cache()

    with open(os.path.join(OUTPUT_DIR, f"{args.out}.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"[saved] {args.out}.json")


if __name__ == "__main__":
    main()
