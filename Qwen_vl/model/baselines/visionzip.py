"""VisionZip port (CVPR 2025) -- official repo dvlab-research/VisionZip
@ 8f86b55c, Qwen2.5-VL adaptation ``Qwen2_5_VL/qwen2_5vl_visionzip.py``
(select_pixel branch, lines ~1914-1966, and the ViT last-block logits/key
capture, lines ~536-616).

Mechanism (official): the LAST vision block's attention weights give a
received-attention score (head-mean, query-sum) for the dominant tokens
(top 65%); among the rest, 5% evenly-spaced anchors are selected and every
other token is aggregated into its nearest anchor by last-block key
similarity (normalised keys, cosine assignment); the anchor's LLM-side
feature becomes ``target + mean(assignees)``.

Qwen3-VL adaptation (prereg §4 + amendment A1): no window attention (all
blocks full attention, no window reordering); no CLS -- the official
Qwen2.5-VL last-block received-attention is used as-is; budget alignment
keeps the official 65:5 dominant:contextual ratio and scales to K
(dominant = round(13K/14), contextual = K - dominant).  Merged tokens keep
the anchor's 3-D coordinate (anchors ARE kept tokens); DeepStack streams are
averaged with the same assignment (prereg VisionZip row).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DOMINANT_RATIO = 0.65
CONTEXTUAL_RATIO = 0.05


def visual_forward_with_visionzip(visual, hidden_states, grid_thw):
    """Stock Qwen3 vision walk capturing the LAST block's attention weights
    and post-RoPE keys (mirrors model_fixed_res.qwen3_visual_forward_with_
    importance; same math as the stock visual pass)."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        apply_rotary_pos_emb_vision)

    hidden_states = visual.patch_embed(hidden_states)
    hidden_states = hidden_states + visual.fast_pos_embed_interpolate(grid_thw)
    rotary_pos_emb = visual.rot_pos_emb(grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    merge_unit = int(getattr(visual, "spatial_merge_unit",
                             visual.spatial_merge_size ** 2))
    deepstack_feature_lists = []
    last_attn = None
    last_key = None
    n_blocks = len(visual.blocks)

    for layer_num, blk in enumerate(visual.blocks):
        is_last = layer_num == n_blocks - 1
        hs_normed = blk.norm1(hidden_states)
        seq_length = hs_normed.shape[0]
        q, k, v = (blk.attn.qkv(hs_normed)
                   .reshape(seq_length, 3, blk.attn.num_heads, -1)
                   .permute(1, 0, 2, 3).unbind(0))
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        attn_outs, w_parts, k_parts = [], [], []
        for qi, ki, vi, ln in zip(torch.split(q, lengths, dim=2),
                                  torch.split(k, lengths, dim=2),
                                  torch.split(v, lengths, dim=2), lengths):
            w = torch.matmul(qi, ki.transpose(2, 3)) * blk.attn.scaling
            w = torch.softmax(w, dim=-1, dtype=torch.float32).to(qi.dtype)
            attn_outs.append(torch.matmul(w, vi).transpose(1, 2).contiguous())
            if is_last:
                w_parts.append(w[0])
                k_parts.append(ki[0])
        attn_output = torch.cat(attn_outs, dim=1).reshape(seq_length, -1)
        attn_output = blk.attn.proj(attn_output)
        if is_last:
            last_attn = torch.cat(w_parts, dim=1)     # [H, Np, Np]
            last_key = torch.cat(k_parts, dim=1)      # [H, Np, hd]

        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + blk.mlp(blk.norm2(hidden_states))
        if layer_num in visual.deepstack_visual_indexes:
            deepstack_feature_lists.append(
                visual.deepstack_merger_list[
                    visual.deepstack_visual_indexes.index(layer_num)](
                        hidden_states))

    V = visual.merger(hidden_states)

    with torch.no_grad():
        attn_mean = last_attn.float().mean(0).sum(0)          # [Np]
        attn_mean = attn_mean.view(attn_mean.shape[0] // merge_unit,
                                   -1).mean(dim=-1)           # [Nm]
        attn_key = last_key.float().mean(0)                   # [Np, hd]
        attn_key = attn_key.view(attn_key.shape[0] // merge_unit, merge_unit,
                                 -1).mean(dim=1)              # [Nm, hd]
    return V, deepstack_feature_lists, attn_mean, attn_key


def select_visionzip(V, DS, attn_mean, attn_key, K):
    """Official select_pixel branch; returns (keep_idx, V_sel, DS_sel)."""
    N = attn_mean.shape[0]
    K = min(K, N)
    dominant_num = int(round(K * DOMINANT_RATIO / (DOMINANT_RATIO + CONTEXTUAL_RATIO)))
    contextual_num = max(K - dominant_num, 1)

    topk_indices = torch.topk(attn_mean, dominant_num).indices
    dominant_mask = torch.zeros(N, dtype=torch.bool, device=V.device)
    dominant_mask[topk_indices] = True
    contextual_idx = torch.nonzero(~dominant_mask, as_tuple=True)[0]  # raster

    # evenly spaced anchors among the contextual candidates (official)
    step = max(1, contextual_idx.shape[0] // contextual_num)
    target_local = torch.arange(0, contextual_idx.shape[0], step,
                                device=V.device)[:contextual_num]
    anchor_idx = contextual_idx[target_local]
    assignee_local = contextual_idx[~torch.isin(
        torch.arange(contextual_idx.shape[0], device=V.device), target_local)]

    metric = attn_key / attn_key.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    target_metric = metric[anchor_idx].unsqueeze(0)            # [1, nc, hd]
    assignee_metric = metric[assignee_local].unsqueeze(0)      # [1, na, hd]
    sim = torch.bmm(assignee_metric, target_metric.transpose(1, 2))  # [1, na, nc]
    assign = sim.argmax(dim=2)[0]                              # [na]

    counts = torch.zeros(contextual_num, device=V.device)
    counts.index_add_(0, assign, torch.ones_like(assign, dtype=counts.dtype))
    counts = counts.clamp(min=1)

    keep_idx = torch.sort(torch.cat([topk_indices, anchor_idx])).values
    V_sel = V[keep_idx].clone()
    # anchors get target + mean(assignees); dominant keep their own features
    anchor_rows = (V[anchor_idx]
                   + (torch.zeros(contextual_num, V.shape[1], device=V.device,
                                  dtype=V.dtype)
                      .index_add_(0, assign, V[assignee_local]) / counts[:, None]))
    is_anchor = torch.isin(keep_idx, anchor_idx)
    V_sel[is_anchor] = anchor_rows.to(V.dtype)

    DS_sel = None
    if DS is not None:
        DS_sel = []
        for ds in DS:
            agg = (torch.zeros(contextual_num, ds.shape[1], device=V.device,
                               dtype=ds.dtype)
                   .index_add_(0, assign, ds[assignee_local]) / counts[:, None])
            ds_sel = ds[keep_idx].clone()
            ds_sel[is_anchor] = agg.to(ds.dtype)
            DS_sel.append(ds_sel)
    return keep_idx, V_sel, DS_sel


def selector(K, ctx):
    attn_mean, attn_key = ctx["vz"]
    V, DS = ctx["V"], ctx["DS"]
    keep_idx, V_sel, DS_sel = select_visionzip(V, DS, attn_mean, attn_key, K)
    return dict(keep_idx=keep_idx, V_sel=V_sel, DS_sel=DS_sel)
