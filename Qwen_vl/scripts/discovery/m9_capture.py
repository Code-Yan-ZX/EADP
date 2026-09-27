"""
M9 -- Deferred Boundary Adjudication: early-layer text->visual capture.

Phase 0/1 machinery. The model forward is NEVER modified: read-only forward
hooks record the q/k/v projections and the layer's position_embeddings, and
after each captured layer the scores are recomputed by hand for a small slice
only -- the text query rows x the boundary key rows. Nothing materialises an
S x S attention matrix and no layer is switched to eager attention.

Architecture facts this relies on (Phase-0 audit, transformers 4.57.3,
Qwen/Qwen3-VL-8B-Instruct, config snapshot 0c351dd0):
    text layers            36 (0-based 0..35)
    Q heads / KV heads     32 / 8 (GQA group 4), head_dim 128, hidden 4096
    q/k norm               per-head RMSNorm BEFORE rope
    rope                   mrope (interleaved, sections [24,20,20]); the engine
                           feeds 1-D positions expanded to 3 identical rows, so
                           the effective rotation is the standard one -- and we
                           reuse the layer's own cos/sin and its own
                           apply_rotary_pos_emb, so the interleaving never
                           matters here
    attention              SDPA (no weights returned) -> why we recompute the
                           slice by hand from the captured q/k
    o_proj                 weight (4096, 32*128); head h occupies columns
                           [h*128, (h+1)*128); value of Q-head h comes from
                           KV-head h//4
    prompt layout          <|im_start|>user\n<|vision_start|> + 1024 visual +
                           <|vision_end|> + question + <|im_end|>\n
                           <|im_start|>assistant\n
                           -> the question sits AFTER the visual span and
                           attends to it causally at every layer
"""
from __future__ import annotations

import torch

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb,
)


class EarlyLayerCapture:
    """Capture text->boundary attention and value contribution on layers
    [lo, hi). Read-only: the layer outputs are untouched.

    Stored per captured layer, each (n_query, n_boundary) fp32:
        att_mean   mean over the 32 Q heads of the post-softmax attention prob
        avn        mean over heads of prob * ||v||  (attention x value norm)
        cmc_mn     mean over heads of || W_O_h (a_h * v_h) ||
        cmc_nm     || sum_h W_O_h (a_h * v_h) ||   (norm of the total update)
    """

    def __init__(self, text_model, layer_lo: int, layer_hi: int,
                 query_rows: torch.Tensor, boundary_rows: torch.Tensor,
                 retain_layer: int = None):
        self.text = text_model
        self.lo, self.hi = layer_lo, layer_hi
        self.query_rows = query_rows          # 1-D LongTensor, sequence rows
        self.boundary_rows = boundary_rows    # 1-D LongTensor, sequence rows
        self.retain_layer = retain_layer      # keep raw q/k/v of this layer
        self.retained = None
        self.layers = {}
        self._handles = []
        self._pending = {}

    # ------------------------------------------------------------------ run --
    def __enter__(self):
        for li in range(self.lo, self.hi):
            layer = self.text.layers[li]
            self._pending[li] = {}
            self._handles.append(
                layer.register_forward_pre_hook(
                    self._mk_pre(li), with_kwargs=True))
            for name in ("q_proj", "k_proj", "v_proj"):
                mod = getattr(layer.self_attn, name)
                self._handles.append(
                    mod.register_forward_hook(self._mk_proj(li, name)))
            self._handles.append(
                layer.register_forward_hook(self._mk_post(li)))
        return self

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._pending.clear()

    # ---------------------------------------------------------------- hooks --
    def _mk_pre(self, li):
        def pre(module, args, kwargs):
            self._pending[li]["pos_emb"] = kwargs.get("position_embeddings")
        return pre

    def _mk_proj(self, li, name):
        def hook(module, inputs, output):
            self._pending[li][name] = output
        return hook

    def _mk_post(self, li):
        def post(module, args, output):
            p = self._pending.pop(li)
            if self.retain_layer is not None and li == self.retain_layer:
                self.retained = dict(layer=module, pend=dict(p),
                                     pos_emb=p["pos_emb"],
                                     layer_in=args[0])
            self.layers[li] = self._scores(layer=module, pend=p)
            p.clear()
        return post

    # --------------------------------------------------------------- scores --
    @torch.no_grad()
    def _scores(self, layer, pend):
        attn = layer.self_attn
        q, k, v = pend["q_proj"], pend["k_proj"], pend["v_proj"]
        cos, sin = pend["pos_emb"]
        dev = q.device
        qrows = self.query_rows.to(dev)
        brows = self.boundary_rows.to(dev)

        S, H, KV, D = q.shape[1], 32, 8, 128
        grp = H // KV
        qn = attn.q_norm(q.view(1, S, H, D)).transpose(1, 2)   # (1,H,S,D)
        kn = attn.k_norm(k.view(1, S, KV, D)).transpose(1, 2)  # (1,KV,S,D)
        # rope on the sliced rows only (rotation is per-position, so slicing
        # rows and slicing cos/sin rows commute); the k result of the first
        # call is discarded -- the function wants a k, this one is a throwaway
        qs = qn[:, :, qrows, :]                                # (1,H,Q,D)
        qr, _ = apply_rotary_pos_emb(qs, kn[:, :, :qs.shape[2], :],
                                     cos[:, qrows, :], sin[:, qrows, :])
        kr, _ = apply_rotary_pos_emb(kn, kn[:, :, :1, :], cos, sin)
        # logits for the text query rows against ALL keys (softmax needs the
        # full row); GQA: Q head h reads KV head h//4
        qh = qr[0].float()                                     # (H,Q,D)
        khe = kr[0].float().repeat_interleave(grp, dim=0)      # (H,S,D)
        logits = torch.einsum("hqd,hsd->hqs", qh, khe) * attn.scaling
        causal = (torch.arange(S, device=dev)[None, :]
                  > qrows[:, None])                            # (Q,S)
        logits = logits.masked_fill(causal[None], float("-inf"))
        probs = torch.softmax(logits, dim=-1)                  # (H,Q,S)
        pb = probs[:, :, brows]                                # (H,Q,B)

        vh = v.view(1, S, KV, D).transpose(1, 2)[0].float()    # (KV,S,D)
        vb = vh[:, brows, :]                                   # (KV,B,D)
        w = attn.o_proj.weight.to(torch.float32)
        wh = w.view(w.shape[0], H, D).permute(1, 2, 0)         # (H,D,4096)

        att = pb.mean(0)                                       # (Q,B)
        avn = torch.zeros_like(att)
        cmc_mn = torch.zeros_like(att)
        cont_sum = torch.zeros(qh.shape[1], brows.numel(), w.shape[0],
                               device=dev)
        for h in range(H):
            a = pb[h]                                          # (Q,B)
            vv = vb[h // grp]                                  # (B,D)
            avn += (a * vv.norm(dim=-1)[None, :])
            c = a[:, :, None] * vv[None, :, :]                 # (Q,B,D)
            proj = c @ wh[h]                                   # (Q,B,4096)
            cmc_mn += proj.norm(dim=-1)
            cont_sum += proj
        avn /= H
        cmc_mn /= H
        out = dict(att_mean=att, avn=avn, cmc_mn=cmc_mn,
                   cmc_nm=cont_sum.norm(dim=-1))
        return {k: t.to(torch.float32).cpu() for k, t in out.items()}
