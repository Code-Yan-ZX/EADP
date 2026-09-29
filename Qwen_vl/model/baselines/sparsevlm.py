"""SparseVLM port (ICML 2025) -- official repo Gumpest/SparseVLMs @ a9e71427.

Ported from ``llava/model/language_model/modelling_sparse_llama.py`` (inference
path) + ``score.py::attn_postprocess_topk`` + ``utils.py::cluster_and_merge``:
at pruning layers [2, 6, 15] keep the top text-guided visual tokens (head-mean
attention of the last prompt token, official per-stage schedule), then recycle
the top-30% of the removed tokens by score into DPC clusters (n/10+1 clusters,
official density-peak clustering verbatim; official merge = uniform mean of
cluster members) appended after the visual section.

Adaptations (frozen in amendment A1/A4 before any number):
  * per-stage retention = official 192-schedule [300, 200, 110] scaled by
    K/110 so the final stage lands on the budget K;
  * positions: kept tokens keep their 3-D coordinates (prereg §3.2.5); each
    recycled token takes its cluster CENTRE's 3-D coordinate;
  * DeepStack rows / cache K/V of a recycled token = the same uniform mean of
    its cluster members' rows (engine-side, native_qwen3._append_merged).
"""

from __future__ import annotations

import torch

PRUNE_LAYERS = [2, 6, 15]
# official sparse_token_list_192 (576-token LLaVA) normalised by its final
# stage count 110, then scaled to the budget K
STAGE_SHAPE = [300 / 110.0, 200 / 110.0, 110 / 110.0]


def _dpc_clusters(x: torch.Tensor, cluster_num: int):
    """Official cluster_and_merge clustering half (verbatim math):
    density peaks on scaled L2 distance; returns (idx_cluster [B, N],
    index_down [B, nc])."""
    B, N, C = x.shape
    x1 = x.unsqueeze(2)
    x2 = x.unsqueeze(1)
    distance = (x1 - x2).norm(dim=-1, p=2)
    dist_matrix = distance / (C ** 0.5)
    dist_nearest, _ = torch.topk(dist_matrix, k=cluster_num, dim=-1,
                                 largest=False)
    density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
    density = density + torch.rand(density.shape, device=density.device,
                                   dtype=density.dtype) * 1e-6
    mask = (density[:, None, :] > density[:, :, None]).type(x.dtype)
    dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
    dist, _ = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
    score = dist * density
    _, index_down = torch.topk(score, k=cluster_num, dim=-1)      # [B, nc]
    dist_center = dist_matrix[torch.arange(B, device=x.device)[:, None],
                              index_down]                        # [B, N, nc]
    idx_cluster = dist_center.argmin(dim=1)                       # [B, N]
    idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
    idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(
        B, cluster_num)
    idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = \
        idx_tmp.reshape(-1)
    return idx_cluster, index_down


def build_prune_layers(K: int, n_vis: int = 1024):
    def make_fn(stage: int):
        def fn(ctx):
            attn = ctx["attn_weights"]
            avg = attn[0].float().mean(0)
            row = avg[-1]
            vis = ctx["vis_mask"]
            vis_idx = torch.nonzero(vis, as_tuple=True)[0]
            keep_len = min(int(round(K * STAGE_SHAPE[stage])), int(vis.sum()))
            scores = row[vis_idx]
            top = scores.topk(keep_len).indices
            keep_vis = vis_idx[top.sort().values]
            text_idx = torch.nonzero(~vis, as_tuple=True)[0]

            # ---- recycling (official) ----------------------------------
            removed_sel = torch.ones(vis_idx.numel(), dtype=torch.bool,
                                     device=vis.device)
            removed_sel[top] = False
            removed_idx = vis_idx[removed_sel]
            removed_scores = scores[removed_sel]
            append = None
            if removed_idx.numel() > 0:
                n_merge = int(removed_idx.numel() * 0.3) + 1
                sel = removed_scores.topk(n_merge).indices.sort().values
                merge_idx = removed_idx[sel]                    # [m] local
                merge_h = ctx["hidden"][0, merge_idx, :].unsqueeze(0)
                cluster_num = int(merge_h.shape[1] / 10) + 1
                idx_cluster, index_down = _dpc_clusters(
                    merge_h, min(cluster_num, merge_h.shape[1]))
                nc = int(index_down.shape[1])
                assign = idx_cluster[0]                         # [m]
                W = (assign[None, :] ==
                     torch.arange(nc, device=assign.device)[:, None])
                W = (W.to(merge_h.dtype) / W.sum(dim=1, keepdim=True).clamp(min=1))
                centers_local = merge_idx[index_down[0]]        # [nc]
                append = dict(members=merge_idx, centers=centers_local, W=W.cpu())

            keep_local = torch.sort(torch.cat([text_idx, keep_vis])).values
            info = dict(stage=stage, n_vis_before=int(vis.sum()),
                        n_vis_keep=keep_len,
                        n_recycled=0 if append is None else int(append["W"].shape[0]))
            return keep_local, info, append
        return fn

    return {l: ("after", make_fn(i)) for i, l in enumerate(PRUNE_LAYERS)}


def selector(K, ctx):
    return dict(keep_idx=None,
                prune_layers=build_prune_layers(K, ctx["prep"]["n_vis"]))
