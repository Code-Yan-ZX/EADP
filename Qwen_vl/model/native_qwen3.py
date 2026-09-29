"""E0 native Qwen3-VL repair and unified injection (prereg §3.2).

The defect this repairs: every historical pruned path handed the LLM
``inputs_embeds`` only, so DeepStack's three injections were dropped and 3D
mRoPE degenerated to a 1-D running index (docs/scoring_search_m2_system.md
§1.4).  E0 restores both, once, here, for every arm.

Design (frozen by the prereg):
  * the stock decoder implementation is never copied.  Prefill and decode go
    through ``Qwen3VLTextModel.forward`` (the language model of the installed
    transformers 4.57.6), exactly the module the stock ``generate`` drives;
  * every arm only supplies a *selector*: ``(ctx) -> keep_idx`` -- raster-order
    ascending, duplicate-free visual indices to keep;
  * ALL feature streams shrink with the SAME keep_idx: the merged visual
    features, the three DeepStack streams, and the 3-D positions taken as a
    subset of the FULL-sequence ``get_rope_index`` output;
  * prefill passes ``position_ids`` explicitly; decode recomputes its position
    from ``cache_position + rope_deltas`` with ``rope_deltas =
    pos3d_full.max() + 1 - len(keep_seq)`` -- identical to the stock decode
    branch;
  * flags ``deepstack={on,off}`` and ``pos={mrope3d,1d}`` exist only for the
    N-ablations (A1/A2); defaults are on/mrope3d.  ``pos='1d'`` passes
    ``position_ids=None``, which reproduces the legacy 1-D running index in
    both prefill and decode (verified against the legacy branch of
    ``get_rope_index`` and the cached-delta decode path).

In-LLM pruning (FastV / PDrop / SparseVLM / PACE-DDAE) rides on
``run_llm_forward``: a manual walk over the *stock* decoder layers that can
compact -- at one designated layer -- the hidden states, the 3-D positions and
their cos/sin, the causal mask and every layer's KV cache with one shared
index vector.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

N_VIS_EXPECTED = 1024  # 1024x1024 -> 64x64 patches -> 32x32 merged tokens


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def unwrap_visual_output(output):
    """``model.visual`` returns (features, deepstack_list) in 4.57.6."""
    if isinstance(output, (tuple, list)):
        if len(output) == 2:
            return output[0], output[1]
        return output[0], None
    return output, None


@dataclass
class PrefillState:
    """Everything decode needs, plus the gate-audit trail."""
    cache: object
    logits: torch.Tensor            # last-position logits, [1, vocab]
    n_vis: int
    keep_idx: torch.Tensor          # [K] ascending
    keep_seq: torch.Tensor          # [Lk] ascending positions into the full seq
    seq_full: int
    lk: int
    rope_delta: int
    pos_mode: str                   # 'mrope3d' | '1d'
    deepstack: bool
    prefill_max_pos: int
    layer_calls_prefill: int
    positions_prefill: Optional[torch.Tensor]  # [3, 1, Lk] when mrope3d
    ds_lengths: Optional[list] = None
    decode_positions: List[int] = field(default_factory=list)
    decode_logits: List[torch.Tensor] = field(default_factory=list)


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
class NativeEngine:
    def __init__(self, vlm_wrapper):
        """``vlm_wrapper`` is the VLMEvalKit ``Qwen3VLChatFixedRes`` instance."""
        self.vlm = vlm_wrapper
        self.model = vlm_wrapper.model                  # ForConditionalGeneration
        self.inner = self.model.model                   # Qwen3VLModel
        self.text = self.inner.language_model           # Qwen3VLTextModel
        self.tok = vlm_wrapper.processor.tokenizer
        self.image_token_id = self.model.config.image_token_id
        self.eos_ids = set()
        for t in ([self.model.config.text_config.eos_token_id]
                  if not isinstance(self.model.config.text_config.eos_token_id, list)
                  else self.model.config.text_config.eos_token_id):
            if t is not None:
                self.eos_ids.add(int(t))
        self.eos_ids.add(int(self.tok.eos_token_id))
        self._layer_calls = 0
        self._hooks = [lyr.register_forward_hook(self._count_layer)
                       for lyr in self.text.layers]

    def _count_layer(self, module, inputs, output):
        self._layer_calls += 1

    # ------------------------------------------------------------- prepare --
    def prepare(self, message, dataset_name=None):
        """Processor inputs -> full-sequence bookkeeping (m2_gdep.prepare's
        native counterpart)."""
        inputs = self.vlm._processor_inputs(
            self.vlm._build_messages(message, dataset=dataset_name))
        ids = inputs["input_ids"]
        has_image = inputs.get("pixel_values", None) is not None
        prep = dict(inputs=inputs, ids=ids, has_image=has_image,
                    message=message, ds=dataset_name)
        if has_image:
            pos = (ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
            prep["img_start"] = int(pos[0])
            prep["img_end"] = int(pos[-1]) + 1
            prep["n_vis"] = int(prep["img_end"] - prep["img_start"])
            prep["gthw"] = inputs["image_grid_thw"]
            prep["pv"] = inputs["pixel_values"].type(self.inner.visual.dtype)
        else:
            prep.update(img_start=None, img_end=None, n_vis=0, gthw=None, pv=None)
        return prep

    # -------------------------------------------------------------- vision --
    @torch.no_grad()
    def encode(self, prep):
        """One vision pass; returns (V [N, D], DS list of 3 [N, D])."""
        out = self.inner.visual(prep["pv"], grid_thw=prep["gthw"])
        V, DS = unwrap_visual_output(out)
        return V, DS

    @torch.no_grad()
    def encode_with_importance(self, prep):
        """HiPrune's selection signal: per-layer merged-token vision attention.

        Same math as the stock visual pass (model_fixed_res.py's manual walk),
        so V/DS produced here may be swapped in for the stock output.  Only the
        HiPrune arm uses it.
        """
        from vlmeval.vlm.qwen3_vl.model_fixed_res import (
            qwen3_visual_forward_with_importance)
        V, DS, attn_list = qwen3_visual_forward_with_importance(
            self.inner.visual, prep["pv"], prep["gthw"])
        return V, DS, attn_list

    # ------------------------------------------------------ text embeds ----
    @torch.no_grad()
    def instruction_embeds(self, message, dataset_name=None):
        """Legacy selector convention: the instruction TEXT ALONE, tokenised
        with add_special_tokens=True (the wrapper's _tokenize_instruction,
        replicated here so the base FixedRes wrapper suffices)."""
        device = next(self.model.parameters()).device
        content = self.vlm._prepare_content(message, dataset=dataset_name)
        parts = [c["text"] for c in content if c.get("type") == "text"]
        instr = " ".join(parts).strip() if parts else "Describe this image."
        enc = self.tok(instr, return_tensors="pt", add_special_tokens=True,
                       padding=True, truncation=True,
                       max_length=self.tok.model_max_length)
        seq = self.model.get_input_embeddings()(enc.input_ids.to(device))
        return seq.mean(dim=1), seq          # mean [1, D], seq [1, M, D]

    # ------------------------------------------------------ rope / positions --
    def positions_full(self, prep):
        ids = prep["ids"]
        am = torch.ones_like(ids)
        pos3d, deltas = self.inner.get_rope_index(
            ids, prep["gthw"], None, attention_mask=am)
        return pos3d, deltas                 # [3,1,L], [1,1]

    @staticmethod
    def keep_sequence(prep, keep_idx: Optional[torch.Tensor]):
        """Text positions ∪ (img_start + keep_idx), ascending."""
        ids = prep["ids"][0]
        L = int(ids.shape[0])
        text_pos = torch.tensor([i for i in range(L)
                                 if not (prep["img_start"] <= i < prep["img_end"])],
                                device=ids.device, dtype=torch.long)
        if keep_idx is None:   # keep the FULL image span
            n_vis = prep["n_vis"]
            if n_vis == 0:
                return text_pos
            vis_pos = prep["img_start"] + torch.arange(n_vis, device=ids.device)
            return torch.sort(torch.cat([text_pos, vis_pos])).values
        if prep["n_vis"] == 0:
            return text_pos
        vis_pos = prep["img_start"] + keep_idx.to(ids.device)
        return torch.sort(torch.cat([text_pos, vis_pos])).values

    # ------------------------------------------------------------ selector --
    def select(self, name: str, K: int, ctx: dict) -> torch.Tensor:
        from model.e0_selectors import run_selector
        return run_selector(name, K, ctx)

    # ------------------------------------------------------------- prefill --
    @torch.no_grad()
    def prefill(self, prep, V, DS, keep_idx: Optional[torch.Tensor],
                deepstack: bool = True, pos: str = "mrope3d",
                llm_prune: Optional[dict] = None,
                full_lm_head: bool = False,
                V_sel: Optional[torch.Tensor] = None,
                DS_sel: Optional[list] = None) -> PrefillState:
        """Build the compacted prefill and run it through the stock text model.

        ``llm_prune`` (in-LLM arms only): {'layer': L, 'scores': fn(attn)}
        handled inside ``run_llm_forward``; ``prefill`` takes the normal path.
        """
        ids = prep["ids"]
        device = ids.device
        emb_layer = self.inner.get_input_embeddings()
        self._layer_calls = 0

        if prep["n_vis"] == 0:
            keep_seq = torch.arange(ids.shape[1], device=device)
            emb = emb_layer(ids)
            pos3d = None if pos == "1d" else self.positions_full(prep)[0]
            return self._forward_prefill(emb, keep_seq, None, None, pos3d,
                                         0, deepstack=False, pos=pos,
                                         seq_full=int(ids.shape[1]),
                                         full_lm_head=full_lm_head)

        keep_idx = keep_idx.to(device).long() if keep_idx is not None \
            else torch.arange(prep["n_vis"], device=device)
        keep_idx = torch.sort(keep_idx).values
        assert keep_idx.numel() == len(torch.unique(keep_idx)), "duplicate keep_idx"
        assert int(keep_idx[0]) >= 0 and int(keep_idx[-1]) < prep["n_vis"], "keep_idx out of range"

        keep_seq = self.keep_sequence(prep, keep_idx)
        V_k = V[keep_idx] if V_sel is None else V_sel
        DS_k = ([ds[keep_idx] for ds in DS] if deepstack else None) \
            if DS_sel is None else (DS_sel if deepstack else None)

        # full-embedding gather, then overwrite the visual slots in raster order
        emb = emb_layer(ids)[0]                       # [L, D]
        emb = emb[keep_seq]                           # [Lk, D]
        vis_slot = (keep_seq >= prep["img_start"]) & (keep_seq < prep["img_end"])
        emb = emb.clone()
        emb[vis_slot] = V_k.to(emb.dtype)

        pos3d_full, _ = self.positions_full(prep)
        pos3d_keep = pos3d_full[:, :, keep_seq]       # [3, 1, Lk]
        rope_delta = int(pos3d_full.max().item()) + 1 - int(keep_seq.shape[0])

        vmask = None
        if deepstack:
            vmask = vis_slot.unsqueeze(0)             # [1, Lk] bool

        st = self._forward_prefill(emb.unsqueeze(0), keep_seq, vmask, DS_k,
                                   pos3d_keep if pos == "mrope3d" else None,
                                   rope_delta, deepstack=deepstack, pos=pos,
                                   seq_full=int(ids.shape[1]),
                                   keep_idx=keep_idx,
                                   full_lm_head=full_lm_head)
        return st

    def _forward_prefill(self, emb, keep_seq, vmask, DS_k, pos3d, rope_delta,
                         deepstack, pos, seq_full, keep_idx=None,
                         full_lm_head=False) -> PrefillState:
        Lk = int(emb.shape[1])
        am = torch.ones(1, Lk, dtype=torch.long, device=emb.device)
        kw = dict(visual_pos_masks=vmask,
                  deepstack_visual_embeds=DS_k if deepstack else None)
        out = self.text(inputs_embeds=emb, attention_mask=am,
                        position_ids=pos3d, use_cache=True, **kw)
        if full_lm_head:
            # stock forward runs lm_head over EVERY position; the GEMM shape
            # changes bf16 accumulation order, so the identity gate compares
            # like-for-like (last-position-only lm_head is ~1 ulp different)
            logits = self.model.lm_head(out.last_hidden_state)[:, -1]
        else:
            logits = self.model.lm_head(out.last_hidden_state[:, -1:])
        max_pos = (int(pos3d.max().item()) if pos3d is not None else Lk - 1)
        return PrefillState(
            cache=out.past_key_values, logits=logits, n_vis=int(vmask.sum())
            if vmask is not None else 0,
            keep_idx=keep_idx if keep_idx is not None else torch.empty(0, dtype=torch.long),
            keep_seq=keep_seq, seq_full=seq_full, lk=Lk, rope_delta=rope_delta,
            pos_mode=pos, deepstack=deepstack, prefill_max_pos=max_pos,
            layer_calls_prefill=self._layer_calls,
            positions_prefill=pos3d,
            ds_lengths=[int(d.shape[0]) for d in DS_k] if DS_k is not None else None)

    # -------------------------------------------------------------- decode --
    @torch.no_grad()
    def decode(self, st: PrefillState, max_new_tokens: int,
               ignore_eos: bool = False, timings: Optional[dict] = None):
        """Greedy loop through the stock text model.  With ``pos='mrope3d'``
        the step-n position is ``prefill_max_pos + 1 + n``; with ``pos='1d'``
        ``position_ids=None`` lets the stock fallback continue the 1-D index."""
        gen_ids = []
        logits = st.logits
        for step in range(max_new_tokens):
            tok = int(logits.argmax(dim=-1).item())
            if not ignore_eos and tok in self.eos_ids:
                break
            gen_ids.append(tok)
            if len(gen_ids) >= max_new_tokens:
                break
            cp = st.lk + step
            if st.pos_mode == "mrope3d":
                p = cp + st.rope_delta
                position_ids = torch.full((3, 1, 1), p, dtype=torch.long,
                                          device=logits.device)
            else:
                position_ids = None
            am = torch.ones(1, cp + 1, dtype=torch.long, device=logits.device)
            out = self.text(input_ids=torch.tensor([[tok]], device=logits.device),
                            attention_mask=am, position_ids=position_ids,
                            past_key_values=st.cache,
                            cache_position=torch.tensor([cp], device=logits.device))
            step_logits = self.model.lm_head(out.last_hidden_state[:, -1:])
            st.decode_positions.append(cp + st.rope_delta
                                       if st.pos_mode == "mrope3d" else cp)
            st.decode_logits.append(step_logits)
            logits = step_logits
        text = self.tok.decode(gen_ids, skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)
        return gen_ids, text

    # ------------------------------------------------------------ generate --
    @torch.no_grad()
    def generate(self, message, dataset_name=None, K: int = 256,
                 selector: str = "identity", deepstack: bool = True,
                 pos: str = "mrope3d", max_new_tokens: int = 2048,
                 ignore_eos: bool = False, timings: Optional[dict] = None,
                 record_ctx: bool = False):
        """One full arm run.  Returns dict(text, keep_idx, meta, timings)."""
        timings = timings if timings is not None else {}
        wall0 = time.perf_counter()
        prep = self.prepare(message, dataset_name)
        timings["image_preprocess_ms"] = (time.perf_counter() - wall0) * 1e3

        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ev[0].record()
        vz_extra = None
        if selector == "hiprune":
            # HiPrune selects on per-layer vision attention; this walk returns
            # the same merged features / DeepStack streams as the stock pass.
            V, DS, attn_list = self.encode_with_importance(prep)
        elif selector == "visionzip":
            from model.baselines.visionzip import visual_forward_with_visionzip
            V, DS, attn_mean, attn_key = visual_forward_with_visionzip(
                self.inner.visual, prep["pv"], prep["gthw"])
            vz_extra = (attn_mean, attn_key)
        else:
            V, DS = self.encode(prep)
            attn_list = None
        ev[1].record()
        torch.cuda.synchronize()
        timings["vision_ms"] = ev[0].elapsed_time(ev[1])

        keep_idx = None
        prune_layers = None
        V_sel = DS_sel = None
        if prep["n_vis"] > 0 and not (selector == "identity" and K >= prep["n_vis"]):
            text_mean, text_seq = self.instruction_embeds(
                message, dataset_name)
            ctx = dict(prep=prep, V=V, DS=DS, K=K, engine=self,
                       text_mean=text_mean, text_seq=text_seq,
                       attn_list=attn_list, vz=vz_extra,
                       seed=timings.get("selector_seed"))
            if record_ctx:
                timings["ctx"] = ctx
            t_sel0 = time.perf_counter()
            sel_out = self.select(selector, K, ctx)
            torch.cuda.synchronize()
            timings["selector_ms"] = (time.perf_counter() - t_sel0) * 1e3
            if isinstance(sel_out, dict):
                keep_idx = sel_out.get("keep_idx")
                V_sel = sel_out.get("V_sel")
                DS_sel = sel_out.get("DS_sel")
                prune_layers = sel_out.get("prune_layers")
            else:
                keep_idx = sel_out

        if prune_layers:
            evp = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            evp[0].record()
            st = self.run_llm_forward(prep, V, DS, keep_idx, deepstack=deepstack,
                                      pos=pos, prune_layers=prune_layers)
            evp[1].record()
            torch.cuda.synchronize()
            timings["llm_prefill_ms"] = evp[0].elapsed_time(evp[1])
        else:
            evp = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            evp[0].record()
            st = self.prefill(prep, V, DS, keep_idx, deepstack=deepstack,
                              pos=pos, V_sel=V_sel, DS_sel=DS_sel)
            evp[1].record()
            torch.cuda.synchronize()
            timings["llm_prefill_ms"] = evp[0].elapsed_time(evp[1])
        timings["ttft_ms"] = (time.perf_counter() - wall0) * 1e3
        wall1 = time.perf_counter()
        gen_ids, text = self.decode(st, max_new_tokens, ignore_eos=ignore_eos)
        timings["decode_wall_ms"] = (time.perf_counter() - wall1) * 1e3
        torch.cuda.synchronize()
        meta = self.invariants(prep, st, V, DS, n_decode=len(gen_ids))
        return dict(text=text, gen_ids=gen_ids, keep_idx=st.keep_idx, meta=meta,
                    timings=timings, state=st)

    # ----------------------------------------------------------- invariants --
    def invariants(self, prep, st: PrefillState, V, DS, n_decode: int) -> dict:
        """N4-style invariant snapshot for one run."""
        m = dict(n_vis_full=prep["n_vis"], n_vis_kept=int(st.n_vis),
                 seq_full=st.seq_full, lk=st.lk, pos_mode=st.pos_mode,
                 deepstack=st.deepstack, decode_steps=n_decode)
        if st.keep_idx.numel() > 0:
            ki = st.keep_idx
            m["no_dup"] = bool(len(torch.unique(ki)) == ki.numel())
            m["in_range"] = bool(int(ki.min()) >= 0 and int(ki.max()) < prep["n_vis"])
            m["ascending"] = bool(bool((ki[1:] >= ki[:-1]).all()) if ki.numel() > 1 else True)
            m["ds_lengths"] = (st.ds_lengths if st.ds_lengths is not None
                               else [int(d[ki].shape[0]) for d in DS]
                               if st.deepstack else None)
        m["layer_calls"] = self._layer_calls
        m["layer_calls_expected"] = self.text.config.num_hidden_layers * (1 + n_decode)
        m["layer_calls_ok"] = (m["layer_calls"] == m["layer_calls_expected"])
        cache_len = int(st.cache.get_seq_length())
        m["cache_len"] = cache_len
        m["cache_ok"] = cache_len == st.lk + n_decode
        if st.decode_positions:
            m["decode_positions_head"] = st.decode_positions[:4]
        return m

    # ------------------------------------------------- in-LLM pruning path --
    @torch.no_grad()
    def run_llm_forward(self, prep, V, DS, keep_idx, deepstack: bool = True,
                        pos: str = "mrope3d",
                        prune_layers: Optional[dict] = None,
                        full_lm_head: bool = False):
        """Prefill walking the STOCK decoder layers, with compaction after
        (or before) designated layers.

        ``prune_layers``: {layer_idx: (mode, fn)}; mode 'after' runs the layer
        with eager attention (captured), then fn compacts its output; mode
        'before' hands fn the layer's q/k projections and the current hidden
        state and compacts the layer's input.  ``fn(ctx) -> keep_local`` gets
        ctx = dict(hidden, attn_weights, vis_mask, positions, layer_idx,
        engine, stage) and returns indices INTO THE CURRENT sequence to keep
        (text always keepable).  At every compaction the hidden states, the 3-D
        positions and their cos/sin, the causal mask, ``visual_pos_masks`` and
        every layer's KV cache shrink with the SAME index (prereg §3.2.5);
        DeepStack streams shrink with it when layers 0-2 are affected.
        ``rope_deltas`` is decided by the FULL sequence and is unaffected.
        """
        ids = prep["ids"]
        device = ids.device
        self._layer_calls = 0
        prune_layers = prune_layers or {}

        keep_idx = keep_idx.to(device).long() if keep_idx is not None \
            else torch.arange(prep["n_vis"], device=device)
        keep_idx = torch.sort(keep_idx).values
        keep_seq = self.keep_sequence(prep, keep_idx)

        emb_layer = self.inner.get_input_embeddings()
        emb = emb_layer(ids)[0][keep_seq].clone()
        vis_mask = (keep_seq >= prep["img_start"]) & (keep_seq < prep["img_end"])
        emb[vis_mask] = V[keep_idx].to(emb.dtype)
        DS_k = [ds[keep_idx] for ds in DS] if deepstack else None

        pos3d_full, _ = self.positions_full(prep)
        positions = pos3d_full[:, :, keep_seq]      # [3, 1, Lk]
        rope_delta = int(pos3d_full.max().item()) + 1 - int(keep_seq.shape[0])

        h = emb.unsqueeze(0)
        cache = DynamicCache(config=self.text.config)
        cos_sin = self.text.rotary_emb(h, positions)

        def rebuild_mask(h_now):
            Lc = int(h_now.shape[1])
            cp = torch.arange(Lc, device=device)
            am = torch.ones(1, Lc, dtype=torch.long, device=device)
            m = create_causal_mask(
                self.text.config, input_embeds=h_now, attention_mask=am,
                cache_position=cp, past_key_values=cache,
                position_ids=positions[0])
            return m, cp

        mask, cp = rebuild_mask(h)
        prune_info = []

        for j, layer in enumerate(self.text.layers):
            spec = prune_layers.get(j)

            if spec is not None and spec[0] == "before":
                keep_local, info = spec[1](dict(
                    hidden=h, attn_weights=None, vis_mask=vis_mask,
                    positions=positions, layer_idx=j, engine=self,
                    layer=layer, cos_sin=cos_sin))
                h, positions, cos_sin, vis_mask, DS_k, keep_idx, keep_seq = \
                    self._compact(h, positions, cos_sin, vis_mask, DS_k,
                                  keep_idx, keep_seq, keep_local)
                mask, cp = rebuild_mask(h)
                self._compact_cache(cache, keep_local)
                prune_info.append(dict(layer=j, mode="before", info=info,
                                       n_kept=int(h.shape[1])))

            if spec is not None and spec[0] == "after":
                attn_w = self._forward_layer_eager(layer, h, positions, mask,
                                                   cp, cache, cos_sin)
                self._layer_calls += 1
                out = spec[1](dict(
                    hidden=h, attn_weights=attn_w, vis_mask=vis_mask,
                    positions=positions, layer_idx=j, engine=self,
                    layer=layer, cos_sin=cos_sin))
                keep_local, info = out[0], out[1]
                append = out[2] if len(out) > 2 else None
                h, positions, cos_sin, vis_mask, DS_k, keep_idx, keep_seq = \
                    self._compact(h, positions, cos_sin, vis_mask, DS_k,
                                  keep_idx, keep_seq, keep_local)
                mask, cp = rebuild_mask(h)
                self._compact_cache(cache, keep_local)
                if append is not None:
                    h, positions, cos_sin, vis_mask, DS_k = self._append_merged(
                        h, positions, cos_sin, vis_mask, DS_k, cache, append)
                    mask, cp = rebuild_mask(h)
                prune_info.append(dict(layer=j, mode="after", info=info,
                                       n_kept=int(h.shape[1])))
                # DeepStack injects right after the layer output; the addition
                # commutes with the row selection above, so injecting on the
                # compacted hidden is bit-identical
                if deepstack and DS_k is not None and j < len(DS_k):
                    h = self.text._deepstack_process(h, vis_mask.unsqueeze(0),
                                                     DS_k[j])
            else:
                layer_outputs = layer(
                    h,
                    attention_mask=mask,
                    position_ids=positions[0],
                    past_key_values=cache,
                    cache_position=cp,
                    position_embeddings=cos_sin,
                )
                h = layer_outputs
                self._layer_calls += 1
                if deepstack and DS_k is not None and j < len(DS_k):
                    h = self.text._deepstack_process(h, vis_mask.unsqueeze(0),
                                                     DS_k[j])

        h = self.text.norm(h)
        if full_lm_head:
            logits = self.model.lm_head(h)[:, -1]
        else:
            logits = self.model.lm_head(h[:, -1:])
        st = PrefillState(
            cache=cache, logits=logits, n_vis=int(vis_mask.sum()),
            keep_idx=keep_idx, keep_seq=keep_seq, seq_full=int(ids.shape[1]),
            lk=int(h.shape[1]), rope_delta=rope_delta, pos_mode=pos,
            deepstack=deepstack, prefill_max_pos=int(positions.max().item()),
            layer_calls_prefill=self._layer_calls,
            positions_prefill=positions,
            ds_lengths=[int(d.shape[0]) for d in DS_k] if DS_k is not None else None)
        st.prune_info = prune_info
        return st

    @staticmethod
    def _compact(h, positions, cos_sin, vis_mask, DS_k, keep_idx, keep_seq,
                 keep_local):
        dev = h.device
        idx = keep_local.to(dev)
        cos, sin = cos_sin
        h = h[:, idx, :]
        positions = positions[:, :, idx]
        cos_sin = (cos[:, idx, :], sin[:, idx, :])
        vis_mask = vis_mask[idx]
        # DS rows are aligned with the OLD visual slots in raster order; the
        # survivors keep their relative order, so a boolean over the new
        # visual slots selects the DS rows.
        if DS_k is not None:
            DS_k = [ds[vis_mask] for ds in DS_k] if bool(vis_mask.any()) \
                else [ds[:0] for ds in DS_k]
        keep_seq = keep_seq[idx]
        return h, positions, cos_sin, vis_mask, DS_k, keep_idx, keep_seq

    @staticmethod
    def _compact_cache(cache, keep_local):
        for lyr in cache.layers:
            if lyr.keys is not None:
                lyr.keys = lyr.keys[:, :, keep_local, :]
                lyr.values = lyr.values[:, :, keep_local, :]

    @staticmethod
    def _append_merged(h, positions, cos_sin, vis_mask, DS_k, cache, append):
        """SparseVLM token recycling: insert merged tokens right after the
        visual block.  Each merged token takes its cluster CENTRE's 3-D
        coordinate / cos-sin; its hidden state, DeepStack rows and every
        cache layer's K/V are the official similarity-weighted average of the
        cluster members' rows (one shared weight matrix W)."""
        members = append["members"].to(h.device)          # [m] local indices
        centers = append["centers"].to(h.device)          # [nc] local indices
        W = append["W"].to(h.device, h.dtype)             # [nc, m], rows sum 1
        n_clusters = int(W.shape[0])

        extra_h = (W @ h[0, members, :]).unsqueeze(0)     # [1, nc, C]

        cos, sin = cos_sin
        extra_pos = positions[:, :, centers]              # [3, 1, nc]
        extra_cos = cos[:, centers, :]
        extra_sin = sin[:, centers, :]

        # insertion point: right after the LAST visual token of the block
        insert_at = int(torch.nonzero(vis_mask, as_tuple=True)[0].max().item()) + 1
        h = torch.cat([h[:, :insert_at], extra_h, h[:, insert_at:]], dim=1)
        positions = torch.cat([positions[:, :, :insert_at], extra_pos,
                               positions[:, :, insert_at:]], dim=2)
        cos_sin = (torch.cat([cos[:, :insert_at], extra_cos,
                              cos[:, insert_at:]], dim=1),
                   torch.cat([sin[:, :insert_at], extra_sin,
                              sin[:, insert_at:]], dim=1))
        ones = torch.ones(n_clusters, dtype=torch.bool, device=h.device)
        vis_mask = torch.cat([vis_mask[:insert_at], ones, vis_mask[insert_at:]])

        if DS_k is not None and any(ds is not None for ds in DS_k):
            # rows of DS_k are aligned with the visual entries in order; the
            # rank of a visual position = number of visual slots before it
            vis_idx = torch.nonzero(vis_mask, as_tuple=True)[0]
            rank = torch.searchsorted(vis_idx, members)
            DS_k = [torch.cat([ds, (W @ ds[rank, :]).to(ds.dtype)], dim=0)
                    if ds is not None else None for ds in DS_k]

        # cache rows: the same weight matrix over the members' K/V per layer
        for lyr in cache.layers:
            if lyr.keys is not None:
                mk = lyr.keys[0, :, members, :]               # [H, m, hd]
                mv = lyr.values[0, :, members, :]
                ek = (W.to(mk.dtype) @ mk).unsqueeze(0)       # [1, H, nc, hd]
                ev = (W.to(mv.dtype) @ mv).unsqueeze(0)
                lyr.keys = torch.cat([lyr.keys[:, :, :insert_at], ek,
                                      lyr.keys[:, :, insert_at:]], dim=2)
                lyr.values = torch.cat([lyr.values[:, :, :insert_at], ev,
                                        lyr.values[:, :, insert_at:]], dim=2)
        return h, positions, cos_sin, vis_mask, DS_k

    def _forward_layer_eager(self, layer, h, positions, mask, cp, cache,
                             cos_sin):
        """One stock decoder layer run with eager attention; returns the
        layer's attention weights [B, H, L, L].  The SDPA fast path passes
        ``mask=None`` and eager would then skip causal masking, so an explicit
        additive causal mask is built here for this layer only."""
        L = int(h.shape[1])
        causal = torch.triu(torch.full((1, 1, L, L),
                                       torch.finfo(h.dtype).min,
                                       device=h.device, dtype=h.dtype),
                            diagonal=1)
        attn = layer.self_attn
        store = {}
        old_impl = self.text.config._attn_implementation
        self.text.config._attn_implementation = "eager"
        hook = attn.register_forward_hook(lambda m, i, o: store.update(w=o[1]))
        try:
            layer(h, attention_mask=causal, position_ids=positions[0],
                  past_key_values=cache, cache_position=cp,
                  position_embeddings=cos_sin)
        finally:
            hook.remove()
            self.text.config._attn_implementation = old_impl
        return store.get("w")

