"""Stage-1 grounding-guidance variants of the official EADP scorer.

Everything OUTSIDE the text-token weighting is verbatim official EADP
(model/pruner.py via instrumented.TimedEADPPruner as instantiated by
e0_selectors._eadp_parts: alpha=0.5, beta=2.0, entropy T=100,
keep_ratio=0.2, M_temp=0.01, smoothing k=3 sigma=1.0, official facility
selector).  Only the dense-aggregation text weighting changes.

Motivation (protocol: scripts/stage1_grounding/PROTOCOL_NOTES.md):
the official aggregation weight softmax(-H / 0.01) is numerically
near-one-hot (entropy is in nats, /0.01 spans hundreds of logits), so a
literal additive fusion of a grounding term into those logits is a no-op.
The approved design therefore splits the official mechanism into its two
live parts and calibrates the second:

  mode="base" : official entropy filter + official negative_entropy
                aggregation (identity check path only);
  mode="only" : (ARM A) no entropy at all -- grounding weight over ALL
                instruction tokens, softmax(lam * z(g));
  mode="cal"  : (ARM B/C) official entropy 20% filter kept verbatim;
                within the kept set the near-one-hot entropy weight is
                REPLACED by grounding weight softmax(lam * z(g_kept)).

Grounding statistics are computed on the SAME text-token x visual-token
cosine matrix the official scorer already builds (local_sim_all, stored
negated; actual cosine = -local_sim_all).  No extra forward, no
parameters, no training.
"""

from __future__ import annotations

import torch

STAT_KEYS = ("g1", "g2", "g3", "peak")
MODES = ("base", "only", "cal")

# population constants (frozen round 1; deliberately NOT swept)
K_FRAC, K_MIN = 0.05, 8          # G1/G3 peak set: top-5%, >= 8 tokens
Q_FRAC, Q_MIN = 0.10, 8          # G2 background set: bottom-10%, >= 8 tokens
PEAK_C = 1.0                     # multi-peak threshold: mean + 1 std
ENTROPY_T = 100.0                # verbatim official
ENTROPY_KEEP = 0.2               # verbatim official

# last per-image diagnostic (tokens are attached by the caller who owns the
# tokenizer; the scorer only stores tensors).  Overwritten every call.
LAST_DIAG: dict | None = None


def grounding_stats(s: torch.Tensor) -> dict[str, torch.Tensor]:
    """Cheap per-text-token grounding statistics.

    s: (N, L) ACTUAL text-token x visual-token cosine (NOT the official
    negated storage).  All reductions over dim 0 (visual tokens).

      g1   top-k margin      : mean(top 5%) - mean(all)
      g2   peak-to-background: mean(top 5%) - mean(bottom 10%)
      g3   standardized peak : (mean(top 5%) - mean(all)) / std(all)
      peak multi-peak share  : frac(visual tokens > mean + 1 std)
    """
    N = s.shape[0]
    k = min(N, max(K_MIN, int(round(K_FRAC * N))))
    q = min(N, max(Q_MIN, int(round(Q_FRAC * N))))
    top = s.topk(k, dim=0).values.mean(0)                       # (L,)
    bot = s.topk(q, dim=0, largest=False).values.mean(0)        # (L,)
    mu = s.mean(0)
    sd = s.std(0).clamp_min(1e-6)
    return dict(
        g1=top - mu,
        g2=top - bot,
        g3=(top - mu) / sd,
        peak=(s > (mu + PEAK_C * sd).unsqueeze(0)).float().mean(0),
    )


def _z(g: torch.Tensor) -> torch.Tensor:
    return (g - g.mean()) / g.std().clamp_min(1e-6)


def stage1_local_sim(local_sim_all: torch.Tensor, mode: str, stat: str,
                     lam: float) -> torch.Tensor:
    """Replace ONLY the text-weighting inside the official dense stage.

    local_sim_all: (B, N, M, L) official negated cosine (B = M = 1 here).
    Returns local_sim (B, N, M) in the SAME negated convention.
    """
    B, N, M, L = local_sim_all.shape
    flat = local_sim_all[0, :, 0, :]                 # (N, L) negated cosine

    if mode == "base":
        from model.pruner import _entropy_filter_impl, _local_aggregation_impl
        filtered, H = _entropy_filter_impl(local_sim_all, T=ENTROPY_T,
                                           entropy_keep_ratio=ENTROPY_KEEP)
        local = _local_aggregation_impl(filtered, H, M_temp=0.01,
                                        strategy="negative_entropy")
        return local                                  # (1, N, 1)

    s = -flat                                         # actual cosine (N, L)
    g = grounding_stats(s)[stat]
    diag = dict(H=None, g=g.detach().float().cpu().tolist(),
                kept=None, stat=stat, mode=mode, lam=lam)

    if mode == "only":
        w = torch.softmax(lam * _z(g), dim=0)         # (L,)
    elif mode == "cal":
        # entropy verbatim (softmax over visual tokens, dim 1 of (B,N,M,L))
        probs = torch.softmax(local_sim_all * ENTROPY_T, dim=1)
        H = -(probs * torch.log(probs + 1e-12)).sum(dim=1)[0, 0]   # (L,)
        num_keep = max(1, int(L * ENTROPY_KEEP))
        kept = torch.topk(H, k=num_keep, largest=False).indices
        gk = g[kept]
        w_full = torch.zeros(L, device=g.device, dtype=g.dtype)
        w_full[kept] = torch.softmax(lam * _z(gk), dim=0)
        w = w_full
        diag["H"] = H.detach().float().cpu().tolist()
        diag["kept"] = kept.detach().cpu().tolist()
    else:
        raise KeyError(mode)

    local = (flat * w.unsqueeze(0)).sum(-1)           # (N,) weighted mean
    global LAST_DIAG
    LAST_DIAG = diag
    return local.view(1, N, 1)                        # (B, N, M), B = M = 1


def stage1_importance(image_features: torch.Tensor, text_embeds: torch.Tensor,
                      text_embeds_seq: torch.Tensor, grid_h: int, grid_w: int,
                      mode: str, stat: str, lam: float,
                      alpha: float = 0.5, beta: float = 2.0) -> torch.Tensor:
    """Official scoring pipeline with the text-weighting stage swapped.

    Every line outside stage1_local_sim mirrors instrumented.TimedEADPPruner
    ._score verbatim (same order of ops, FP32)."""
    from model.pruner import _spatial_smoothing_impl

    ie = image_features.float()
    te = text_embeds.float()
    ts = text_embeds_seq.float()
    ie = ie / ie.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    te = te / te.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ts = ts / ts.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    global_sim = -torch.einsum("bnc,mc->bnm", ie, te)
    local_sim_all = -torch.einsum("bnc,mlc->bnml", ie, ts)

    local_sim = stage1_local_sim(local_sim_all, mode, stat, lam)

    text_sim_all = alpha * global_sim + (1 - alpha) * local_sim
    text_sim = text_sim_all.mean(dim=-1)
    importance = (text_sim - text_sim.min(dim=-1, keepdim=True).values + 1e-6) / (
        text_sim.max(dim=-1, keepdim=True).values
        - text_sim.min(dim=-1, keepdim=True).values + 1e-6)
    importance = _spatial_smoothing_impl(importance, grid_h, grid_w,
                                         kernel_size=3, sigma=1.0)
    importance = importance ** beta
    return importance


def make_selector(mode: str, stat: str, lam: float):
    """Selector factory mirroring e0_selectors._eadp_parts, with the
    variant scorer in place of the official one; selection itself is the
    verbatim official facility-location."""
    from common import CudaTimer                      # scripts/discovery
    import instrumented

    def _sel(K, ctx):
        eng = ctx["engine"]
        prep = ctx["prep"]
        V, gthw = ctx["V"], prep["gthw"]
        sms = eng.inner.visual.spatial_merge_size

        split_sizes = (gthw.prod(-1) // (sms ** 2)).tolist()
        keep, offset = [], 0
        for i, n in enumerate(split_sizes):
            token_num = min(K, n)
            if token_num >= n:
                keep.append(torch.arange(offset, offset + n, device=V.device))
                offset += n
                continue
            feats = V[offset:offset + n].unsqueeze(0)
            te = ctx["text_mean"][i:i + 1]
            ts = ctx["text_seq"][i:i + 1]
            gh = int(gthw[i, 1]) // sms
            gw = int(gthw[i, 2]) // sms
            importance = stage1_importance(feats, te, ts, gh, gw,
                                           mode, stat, lam)
            from model.pruner import _sim_visual_impl
            sim_matrix = _sim_visual_impl(feats)
            sel, _ = instrumented.SELECTORS["facility"](
                importance, sim_matrix, token_num)
            idx = sel[0].sort().values
            keep.append(offset + idx.to(V.device))
            offset += n
        return torch.cat(keep)

    return _sel
