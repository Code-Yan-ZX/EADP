"""FastV port (ECCV 2024) -- official repo pkunlp-icler/FastV @ d1659729.

Ported from ``src/FastV/llava-hf/transformers/src/transformers/models/llama/
modeling_llama.py::fastv_forward`` (their `fastv_k`-entry / `last-layer
attention` semantics), adapted to the E0 native engine per prereg §4:
  * prune AFTER LLM layer 2 (prereg table), scores = that layer's own
    attention (eager capture), head-mean, LAST-token query row;
  * keep the top round(N_vis*(1-R)) visual tokens; amendment A1: R := 1-K/N
    (K=256 is the official README's R=75% setting);
  * kept visual tokens keep their original 3-D coordinates; the causal mask
    and every layer's KV cache shrink with the same index (prereg §3.2.5);
    decode rope_deltas unchanged (full-sequence value).
"""

from __future__ import annotations

import torch

PRUNE_AFTER = 2


def build_prune_layers(K: int, n_vis: int = 1024, prune_after: int = PRUNE_AFTER):
    R = 1.0 - K / float(n_vis)

    def fn(ctx):
        attn = ctx["attn_weights"]              # [1, H, L, L] (eager, fp32 softmax)
        avg = attn[0].float().mean(0)           # [L, L] head-mean
        row = avg[-1]                           # last (prompt) token's query row
        vis = ctx["vis_mask"]
        vis_idx = torch.nonzero(vis, as_tuple=True)[0]
        n_keep = min(int(round(vis.sum().item() * (1.0 - R))),
                     int(vis.sum().item()))
        top = row[vis_idx].topk(n_keep).indices
        keep_vis = vis_idx[top.sort().values]   # raster order
        text_idx = torch.nonzero(~vis, as_tuple=True)[0]
        keep_local = torch.sort(torch.cat([text_idx, keep_vis])).values
        return keep_local, dict(n_vis_before=int(vis.sum()), n_vis_keep=n_keep,
                                R=R)

    return {prune_after: ("after", fn)}


def selector(K, ctx):
    """Registered as 'fastv': keep everything pre-LLM; prune inside."""
    return dict(keep_idx=None, prune_layers=build_prune_layers(K, ctx["prep"]["n_vis"]))
