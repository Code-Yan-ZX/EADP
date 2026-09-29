"""PyramidDrop (PDrop) port (CVPR 2025) -- official repo Cooperx521/PyramidDrop
@ 6444f304.

Ported from ``llava/model/modeling_llama_pdrop.py::pdrop_rank_drop``: at the
entry of each rank layer, recompute the layer's q/k on the current (normed)
hidden state, score every token by the head-mean softmaxed attention of the
LAST prompt token, and keep the top ``int(N_vis * ratio)`` visual tokens.
Adapted per prereg §4 and amendment A1: official layer list [8, 16, 24] is
kept; the geometric per-stage retention is r = (K/N)^(1/3) so the final stage
lands exactly on the budget K (official default r=0.5 corresponds to K=128
here). Kept visual tokens keep their 3-D coordinates; mask and every layer's
KV cache shrink with the same index (prereg §3.2.5).
"""

from __future__ import annotations

import torch

from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

LAYER_LIST = [8, 16, 24]


def build_prune_layers(K: int, n_vis: int = 1024, layer_list=None):
    layer_list = layer_list or LAYER_LIST
    r = (K / float(n_vis)) ** (1.0 / len(layer_list))
    ratios = [r ** (i + 1) for i in range(len(layer_list))]

    def make_fn(stage: int):
        def fn(ctx):
            h = ctx["hidden"]
            layer = ctx["layer"]
            attn = layer.self_attn
            hs = layer.input_layernorm(h)
            bsz, L, _ = hs.shape
            hidden_shape = (bsz, L, -1, attn.head_dim)
            q = attn.q_norm(attn.q_proj(hs).view(hidden_shape)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(hs).view(hidden_shape)).transpose(1, 2)
            cos, sin = ctx["cos_sin"]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            # head-mean softmaxed attention of the last prompt token
            scores = torch.matmul(q[:, :, -1:, :].float(),
                                  k.transpose(2, 3).float()) * attn.scaling
            attn_w = torch.softmax(scores, dim=-1)          # [1, H, 1, L]
            attn_avg = attn_w[0, :, 0, :].mean(0)           # [L]
            vis = ctx["vis_mask"]
            vis_idx = torch.nonzero(vis, as_tuple=True)[0]
            keep_len = min(int(n_vis * ratios[stage]), int(vis.sum()))
            top = attn_avg[vis_idx].topk(keep_len).indices
            keep_vis = vis_idx[top.sort().values]
            text_idx = torch.nonzero(~vis, as_tuple=True)[0]
            keep_local = torch.sort(torch.cat([text_idx, keep_vis])).values
            return keep_local, dict(stage=stage, ratio=ratios[stage],
                                    n_vis_before=int(vis.sum()),
                                    n_vis_keep=keep_len)
        return fn

    return {l: ("before", make_fn(i)) for i, l in enumerate(layer_list)}


def selector(K, ctx):
    return dict(keep_idx=None,
                prune_layers=build_prune_layers(K, ctx["prep"]["n_vis"]))
