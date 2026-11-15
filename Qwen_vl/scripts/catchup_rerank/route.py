"""Two-stage CatchUp mechanism adaptation for native Qwen3-VL (4.57.6).

This is NOT the authors' four-stage LLaVA experiment.  The default decisions
are before decoder layers 2 and 7 (zero based), retaining K visual tokens in
each interval.  Deferred states remain independently addressable.  A dense
reference is an offline key diagnostic, never an execution-state replacement.
All arithmetic in the decoder uses the installed, unmodified stock layers.

Public API: prepare_route, prepare_dense_reference, run_route, identity_gate,
no_reentry_gate.  Preparation is reusable: every execution clones its cache.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb, repeat_kv,
)


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _lengths(cache, n_layers):
    return [int(cache.get_seq_length(i)) for i in range(n_layers)]


def _top(scores, k):
    return scores.topk(min(int(k), int(scores.numel()))).indices.sort().values


def _indices(n, device, values):
    values = torch.as_tensor(values, device=device, dtype=torch.long)
    if values.ndim != 1 or bool(((values < 0) | (values >= n)).any()):
        raise ValueError("visual indices must be a one-dimensional in-range array")
    if values.unique().numel() != values.numel():
        raise ValueError("duplicate visual indices")
    return values.sort().values


def _coords(positions, start, end):
    xy = positions[1:, 0, start:end].T.float()
    # Matches the official square-grid coordinate normalization.  Native
    # coordinates also make the rectangular case explicit rather than 1-D.
    span = (xy.max(0).values - xy.min(0).values).clamp_min(1)
    return (xy - xy.min(0).values) / span


def ltc_states(reference, current, targets, anchors, xy, topm=8,
               spatial_lambda=0.15, tau=0.07, gamma=0.8):
    """Return corrected target rows, using official no-norm LTC arithmetic.

    Inputs are immutable; targets/anchors and returned rows are raster-sorted.
    One reference depth is sufficient for this
    two-decision experiment.  Empty anchors give unchanged target states;
    callers must report that no correction was possible.
    """
    targets = _indices(len(current), current.device, targets)
    anchors = _indices(len(current), current.device, anchors)
    if reference.shape != current.shape or xy.shape != (len(current), 2):
        raise ValueError("LTC reference/current/coordinate shapes differ")
    if int(topm) < 1 or float(tau) <= 0:
        raise ValueError("LTC requires positive topm and tau")
    if not targets.numel() or not anchors.numel():
        return current[targets].clone()
    if bool(torch.isin(targets, anchors).any()):
        raise ValueError("LTC targets must be deferred; anchors must be active")
    p = torch.nn.functional.normalize(reference[targets].float(), dim=-1, eps=1e-6)
    q = torch.nn.functional.normalize(reference[anchors].float(), dim=-1, eps=1e-6)
    sim = p @ q.T
    dist = torch.linalg.vector_norm(xy[targets, None] - xy[None, anchors], dim=-1)
    score = sim - float(spatial_lambda) * dist
    top_score, local = score.topk(min(int(topm), int(anchors.numel())), dim=-1)
    chosen = anchors[local]
    weights = (top_score / float(tau)).softmax(-1).to(current.dtype)
    transitions = current[chosen] - reference[chosen]
    delta = (weights.unsqueeze(-1) * transitions).sum(1)
    return current[targets] + float(gamma) * delta


@torch.no_grad()
def attention_scores(eng, layer_idx, hidden, positions, img_start, img_end,
                     key_hidden=None):
    """Native last-query per-head softmax, averaged AFTER normalization.

    All nonvisual keys are included in every denominator.  ``key_hidden``
    may change visual keys, but the query always comes from ``hidden``.
    Qwen3 q_norm/k_norm, GQA expansion, and native mRoPE are mandatory.
    Returns [N] visual attention scores.
    """
    layer = eng.text.layers[int(layer_idx)]
    attn = layer.self_attn
    if hidden.shape[0] != 1:
        raise ValueError("the routing diagnostic supports batch_size=1")
    if key_hidden is None:
        key_hidden = hidden
    if key_hidden.shape != hidden.shape:
        raise ValueError("key and query sequences must have equal shape")
    norm_q = layer.input_layernorm(hidden[:, -1:])
    norm_k = layer.input_layernorm(key_hidden)
    hd = attn.head_dim
    q = attn.q_norm(attn.q_proj(norm_q).view(1, 1, -1, hd)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(norm_k).view(1, hidden.shape[1], -1, hd)).transpose(1, 2)
    cos, sin = eng.text.rotary_emb(hidden, positions)
    q, _ = apply_rotary_pos_emb(q, q, cos[:, -1:], sin[:, -1:])
    k, _ = apply_rotary_pos_emb(k, k, cos, sin)
    k = repeat_kv(k, attn.num_key_value_groups)
    logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * attn.scaling
    return logits.softmax(-1).mean(1)[0, 0, int(img_start):int(img_end)]


def _inputs(eng, prep, V, DS):
    device = next(eng.text.parameters()).device
    ids = prep["ids"].to(device)
    start, end = int(prep["img_start"]), int(prep["img_end"])
    n = end - start
    if ids.ndim != 2 or ids.shape[0] != 1 or n != int(prep["n_vis"]):
        raise ValueError("expected one image and one contiguous visual span")
    if V.shape[0] != n or len(DS) != 3 or any(d.shape != V.shape for d in DS):
        raise ValueError("expected all native V and three aligned DeepStack streams")
    h = eng.inner.get_input_embeddings()(ids).clone()
    h[:, start:end] = V.to(device=device, dtype=h.dtype).unsqueeze(0)
    if "pos3d" in prep:
        positions = prep["pos3d"].to(device)
    else:
        positions = eng.positions_full(prep)[0].to(device)
    if positions.shape != (3, 1, ids.shape[1]):
        raise ValueError("positions must have native [3,1,L] shape")
    return h, positions, [d.to(device=device, dtype=h.dtype) for d in DS], start, end


def _keep_sequence(length, start, end, selected, device):
    keep = torch.ones(length, dtype=torch.bool, device=device)
    keep[start:end] = False
    keep[start + selected] = True
    return keep.nonzero(as_tuple=True)[0]


def _layer(eng, h, positions, DS, start, end, layer_idx, selected, cache):
    """Compact a layer's actual input/cache, then scatter persistent states."""
    keep = _keep_sequence(h.shape[1], start, end, selected, h.device)
    compact = h[:, keep].contiguous()
    pos = positions[:, :, keep].contiguous()
    n = compact.shape[1]
    cp = torch.arange(n, device=h.device)
    # This unpadded SDPA prefill uses the SAME no-mask causal kernel as native
    # text.forward (query_len > 1 => is_causal=True).  An explicit additive
    # triangle changes the CUDA kernel and bf16 accumulation path.
    out = eng.text.layers[layer_idx](
        compact, position_embeddings=eng.text.rotary_emb(compact, pos),
        attention_mask=None, position_ids=pos[0],
        past_key_values=cache, use_cache=cache is not None, cache_position=cp)
    if isinstance(out, tuple):
        out = out[0]
    if layer_idx < len(DS):
        vmask = ((keep >= start) & (keep < end)).unsqueeze(0)
        out = eng.text._deepstack_process(out, vmask, DS[layer_idx][selected])
    full = h.clone()
    full[:, keep] = out
    return full


@dataclass
class RoutePreparation:
    hidden: torch.Tensor
    reference: torch.Tensor
    positions: torch.Tensor
    DS: list
    cache: object
    first_keep: torch.Tensor
    first_scores: torch.Tensor
    img_start: int
    img_end: int
    K: int
    route_layers: tuple
    prefix_ms: float
    cache_lengths: list


@torch.no_grad()
def prepare_route(eng, prep, V, DS, K=256, route_layers=(2, 7)):
    """Execute the common baseline prefix, stopping BEFORE decision two."""
    first, second = map(int, route_layers)
    if not (0 <= first < second < len(eng.text.layers)):
        raise ValueError("routing layers must be ordered valid zero-based indices")
    h, pos, ds, start, end = _inputs(eng, prep, V, DS)
    n = end - start
    if not (1 <= int(K) <= n):
        raise ValueError("K outside visual sequence")
    all_vis = torch.arange(n, device=h.device)
    cache = DynamicCache(config=eng.text.config)
    selected = all_vis
    _sync(h.device)
    t0 = time.perf_counter()
    reference = first_keep = first_scores = None
    for i in range(second):
        if i == first:
            reference = h[0, start:end].clone()
            first_scores = attention_scores(eng, i, h, pos, start, end)
            first_keep = _top(first_scores, K)
            selected = first_keep
        h = _layer(eng, h, pos, ds, start, end, i, selected, cache)
    _sync(h.device)
    return RoutePreparation(h, reference, pos, ds, cache, first_keep,
                            first_scores, start, end, int(K), (first, second),
                            (time.perf_counter() - t0) * 1000,
                            _lengths(cache, len(eng.text.layers)))


@torch.no_grad()
def prepare_dense_reference(eng, prep, V, DS, target_layer=7):
    """Separate full-token reference BEFORE target_layer; no route cache."""
    h, pos, ds, start, end = _inputs(eng, prep, V, DS)
    target_layer = int(target_layer)
    if not (0 <= target_layer < len(eng.text.layers)):
        raise ValueError("invalid dense reference depth")
    selected = torch.arange(end-start, device=h.device)
    _sync(h.device)
    t0 = time.perf_counter()
    for i in range(target_layer):
        h = _layer(eng, h, pos, ds, start, end, i, selected, None)
    _sync(h.device)
    return dict(hidden=h, target_layer=target_layer, positions=pos,
                img_start=start, img_end=end,
                elapsed_ms=(time.perf_counter()-t0)*1000,
                purpose="offline dense visual keys only; not execution-state oracle")


def _decode_one(eng, token, cache, position):
    h = eng.inner.get_input_embeddings()(token)
    positions = torch.full((3, 1, 1), int(position), device=h.device, dtype=torch.long)
    pe = eng.text.rotary_emb(h, positions)
    for i, layer in enumerate(eng.text.layers):
        old_len = int(cache.get_seq_length(i))
        cp = torch.tensor([old_len], device=h.device, dtype=torch.long)
        # Layer-specific cache positions are essential.  For a single query
        # native SDPA sets is_causal=False; every cached key is valid.  No
        # padded rows exist in this batch-one diagnostic, so mask=None is exact.
        h = layer(h, position_embeddings=pe, attention_mask=None,
                  position_ids=positions[0], past_key_values=cache,
                  cache_position=cp, use_cache=True)
        if isinstance(h, tuple):
            h = h[0]
    return eng.model.lm_head(eng.text.norm(h))[:, -1]


def _generate(eng, logits, cache, decode_start, max_new_tokens):
    ids, positions = [], []
    token_logits = []
    stop = "max_new_tokens"
    for step in range(int(max_new_tokens)):
        tok = int(logits.argmax(-1).item())
        token_logits.append(logits.detach().clone())
        if tok in eng.eos_ids:
            stop = "eos"
            break
        ids.append(tok)
        if len(ids) >= max_new_tokens:
            break
        positions.append(int(decode_start + step))
        logits = _decode_one(eng, torch.tensor([[tok]], device=logits.device),
                             cache, decode_start + step)
    text = eng.tok.decode(ids, skip_special_tokens=True,
                          clean_up_tokenization_spaces=False)
    return text, ids, positions, token_logits, stop


@torch.no_grad()
def run_route(eng, prepared, variant="baseline", dense_reference=None,
              max_new_tokens=32, stop_after_route=False, shortlist_mult=2,
              force_second_keep=None, execution_ltc=True,
              return_logits=False):
    """Run one arm from an immutable shared prefix.

    Variants: baseline (stale scores); dense_full (dense deferred visual
    keys); dense_2k (the EXACT SAME full-key scores, with selection restricted
    to baseline top-2K); ltc_2k (temporary
    LTC correction of deferred shortlist keys).  Queries/nonvisual keys
    always come from the common baseline trajectory.  All arms apply
    post-selection LTC using A = first_keep intersect baseline_second_keep.
    """
    aliases = {"stale": "baseline", "ltc_shortlist": "ltc_2k"}
    variant = aliases.get(variant, variant)
    if variant not in ("baseline", "dense_full", "dense_2k", "ltc_2k"):
        raise ValueError(f"unknown routing variant {variant!r}")
    if int(max_new_tokens) < 1 or int(shortlist_mult) < 1:
        raise ValueError("positive token cap and shortlist multiplier required")
    p = prepared
    start, end, K = p.img_start, p.img_end, p.K
    n = end-start
    _sync(p.hidden.device)
    t0 = time.perf_counter()
    cache = copy.deepcopy(p.cache)
    h = p.hidden.clone()
    if _lengths(p.cache, len(eng.text.layers)) != p.cache_lengths:
        raise RuntimeError("shared preparation cache was mutated")
    _sync(h.device)
    clone_done = time.perf_counter()
    second = p.route_layers[1]
    stale_scores = attention_scores(eng, second, h, p.positions, start, end)
    baseline_keep = _top(stale_scores, K)
    anchors = p.first_keep[torch.isin(p.first_keep, baseline_keep)]
    deferred_mask = torch.ones(n, device=h.device, dtype=torch.bool)
    deferred_mask[p.first_keep] = False
    deferred = deferred_mask.nonzero(as_tuple=True)[0]
    shortlist = _top(stale_scores, min(n, int(shortlist_mult)*K))
    xy = _coords(p.positions, start, end)
    scores = stale_scores
    rerank_targets = torch.empty(0, device=h.device, dtype=torch.long)
    if variant != "baseline":
        key_hidden = h.clone()
        if variant.startswith("dense"):
            if dense_reference is None or int(dense_reference["target_layer"]) != second:
                raise ValueError("dense arm needs a separately prepared same-depth reference")
            if (dense_reference["img_start"], dense_reference["img_end"]) != (start, end):
                raise ValueError("dense reference visual span differs")
            if not torch.equal(dense_reference["positions"], p.positions):
                raise ValueError("dense reference native positions differ")
            # Both dense diagnostics must have identical per-head softmax
            # denominators.  dense_2k restricts selection, not key refresh.
            rerank_targets = deferred
            dense_h = dense_reference["hidden"].to(h.device)
            key_hidden[:, start+rerank_targets] = dense_h[:, start+rerank_targets]
        else:
            rerank_targets = shortlist[deferred_mask[shortlist]]
            key_hidden[0, start+rerank_targets] = ltc_states(
                p.reference, h[0, start:end], rerank_targets, anchors, xy)
        scores = attention_scores(eng, second, h, p.positions, start, end,
                                  key_hidden=key_hidden)
    candidates = (torch.arange(n, device=h.device) if variant in ("baseline", "dense_full")
                  else shortlist)
    selected = candidates[_top(scores[candidates], K)].sort().values
    if force_second_keep is not None:
        selected = _indices(n, h.device, force_second_keep)
        if len(selected) != K:
            raise ValueError("forced second decision must retain exactly K")
    readmitted = selected[deferred_mask[selected]]
    before = h[0, start:end].clone()
    if execution_ltc and readmitted.numel():
        h[0, start+readmitted] = ltc_states(p.reference, before,
                                           readmitted, anchors, xy)
    _sync(h.device)
    route_done = time.perf_counter()
    diag = dict(
        variant=variant, n_vis=n, K=K, route_layers=list(p.route_layers),
        route_layers_zero_based=True, official_four_stage_reproduction=False,
        first_keep=p.first_keep.tolist(), baseline_second_keep=baseline_keep.tolist(),
        second_keep=selected.tolist(), common_anchors=anchors.tolist(),
        shortlist=shortlist.tolist(), rerank_targets=rerank_targets.tolist(),
        reentered=readmitted.tolist(), execution_ltc=bool(execution_ltc),
        corrected_count=int(len(readmitted)) if execution_ltc and len(anchors) else 0,
        empty_anchor_correction_skipped=bool(execution_ltc and len(readmitted) and not len(anchors)),
        selection_changed=int(K-torch.isin(selected, baseline_keep).sum().item()),
        deferred_key_replacements_only=True, nonvisual_keys_unchanged=True,
        dense_2k_uses_identical_dense_full_scores=variant.startswith("dense"),
        query_unchanged=True, softmax_over_all_prompt_keys=True,
        decode_start=int(p.positions.max().item())+1,
        prepared_cache_lengths_before=list(p.cache_lengths),
        native_ds_original_lengths=[int(d.shape[0]) for d in p.DS],
        ds_injection_lengths=[n if i < p.route_layers[0] else K for i in range(3)],
        all_original_text_positions_preserved=True,
        dense_reference_is_execution_state=False,
        ltc=dict(topm=8,spatial_lambda=.15,tau=.07,gamma=.8,norm_preserve=False),
    )
    timings = dict(shared_prefix_ms=p.prefix_ms, cache_clone_ms=(clone_done-t0)*1000,
                   second_route_ms=(route_done-clone_done)*1000,
                   dense_reference_ms=(dense_reference or {}).get("elapsed_ms",0))
    if stop_after_route:
        return dict(text=None, token_ids=[], diag=diag, timings=timings,
                    stale_scores=stale_scores.detach(), scores=scores.detach())
    for i in range(second, len(eng.text.layers)):
        h = _layer(eng, h, p.positions, p.DS, start, end, i, selected, cache)
    logits = eng.model.lm_head(eng.text.norm(h[:, -1:]))[:, -1]
    diag["prefill_cache_lengths"] = _lengths(cache, len(eng.text.layers))
    _sync(h.device)
    prefill_done = time.perf_counter()
    text, ids, dec_pos, history, stop = _generate(
        eng, logits, cache, diag["decode_start"], max_new_tokens)
    _sync(h.device)
    done = time.perf_counter()
    diag.update(decode_positions=dec_pos, stop_reason=stop,
                truncated=stop == "max_new_tokens", generated_tokens=len(ids),
                final_cache_lengths=_lengths(cache,len(eng.text.layers)),
                prepared_cache_lengths_after=_lengths(p.cache,len(eng.text.layers)))
    if diag["prepared_cache_lengths_after"] != p.cache_lengths:
        raise RuntimeError("arm execution mutated reusable preparation cache")
    timings.update(remainder_prefill_ms=(prefill_done-route_done)*1000,
                   decode_ms=(done-prefill_done)*1000,
                   arm_wall_ms=(done-t0)*1000)
    out = dict(text=text, token_ids=ids, diag=diag, timings=timings)
    if return_logits:
        out.update(first_logits=logits.detach(), decode_logits=history)
    return out


@torch.no_grad()
def identity_gate(eng, prep, V, DS, max_new_tokens=8):
    """Full stock-layer walk versus NativeEngine full prefill/decode."""
    h, pos, ds, start, end = _inputs(eng, prep, V, DS)
    cache = DynamicCache(config=eng.text.config)
    all_vis = torch.arange(end-start, device=h.device)
    for i in range(len(eng.text.layers)):
        h = _layer(eng,h,pos,ds,start,end,i,all_vis,cache)
    logits = eng.model.lm_head(eng.text.norm(h[:, -1:]))[:, -1]
    native = eng.prefill(prep,V,DS,None)
    native_logits = native.logits.reshape_as(logits)
    delta = float((native_logits.float()-logits.float()).abs().max())
    cache_equal = _lengths(cache,len(eng.text.layers)) == _lengths(native.cache,len(eng.text.layers))
    _, ids, dec_positions, history, _ = _generate(eng,logits,cache,int(pos.max())+1,max_new_tokens)
    # Independent native text.forward decode, following NativeEngine's EOS
    # and original-full-prompt position convention.
    ref_ids, ref_history = [], []
    rlog = native_logits
    for step in range(int(max_new_tokens)):
        tok = int(rlog.argmax(-1).item())
        ref_history.append(rlog.detach().clone())
        if tok in eng.eos_ids:
            break
        ref_ids.append(tok)
        if len(ref_ids) >= max_new_tokens:
            break
        cp = int(native.lk)+step
        rp = int(pos.max())+1+step
        rout = eng.text(input_ids=torch.tensor([[tok]],device=h.device),
                        attention_mask=torch.ones(1,cp+1,device=h.device,dtype=torch.long),
                        position_ids=torch.full((3,1,1),rp,device=h.device,dtype=torch.long),
                        past_key_values=native.cache,use_cache=True)
        rlog = eng.model.lm_head(rout.last_hidden_state[:, -1:])[:, -1]
    dd = max((float((a.float()-b.float()).abs().max()) for a,b in zip(history,ref_history)),default=0.)
    same_argmax = bool(torch.equal(logits.argmax(-1),native_logits.argmax(-1)))
    return dict(pass_all=bool(delta==0 and ids==ref_ids and dd==0 and cache_equal),
                first_logit_max_delta=delta,first_argmax_equal=same_argmax,
                decode_tokens_equal=ids==ref_ids,decode_max_delta=dd,
                cache_lengths_equal=cache_equal,ds_lengths=[len(d) for d in ds],
                manual_token_ids=ids,native_token_ids=ref_ids,
                native_positions=True,stock_layers_unmodified=True)


@torch.no_grad()
def no_reentry_gate(eng, prep, V, DS, K=256, route_layers=(2,7),max_new_tokens=4):
    """Forced identical second choice equals one-shot compact execution."""
    p = prepare_route(eng,prep,V,DS,K,route_layers)
    a = run_route(eng,p,force_second_keep=p.first_keep,
                  max_new_tokens=max_new_tokens,return_logits=True)
    # A separate layer walk performs one irreversible L2 choice.  This also
    # independently checks compact slots and variable layer cache lengths.
    h,pos,ds,start,end = _inputs(eng,prep,V,DS)
    cache = DynamicCache(config=eng.text.config)
    all_vis = torch.arange(end-start,device=h.device)
    for i in range(len(eng.text.layers)):
        keep = all_vis if i < route_layers[0] else p.first_keep
        h = _layer(eng,h,pos,ds,start,end,i,keep,cache)
    logits = eng.model.lm_head(eng.text.norm(h[:, -1:]))[:, -1]
    before_lengths = _lengths(cache,len(eng.text.layers))
    _, ids, dec_positions, history, _ = _generate(eng,logits,cache,int(pos.max())+1,max_new_tokens)
    delta = float((a["first_logits"].float()-logits.float()).abs().max())
    dd = max((float((x.float()-y.float()).abs().max()) for x,y in zip(a["decode_logits"],history)),default=0.)
    lengths_equal = a["diag"]["prefill_cache_lengths"] == before_lengths
    expected_lengths = [int(prep["ids"].shape[1]) if i < route_layers[0]
                        else int(prep["ids"].shape[1])-int(prep["n_vis"])+int(K)
                        for i in range(len(eng.text.layers))]
    expected_cache_ok = before_lengths == expected_lengths
    positions_equal = a["diag"]["decode_positions"] == dec_positions
    independent = None
    if hasattr(eng, "run_llm_forward"):
        def select_once(ctx):
            keep = _keep_sequence(ctx["hidden"].shape[1], start, end,
                                  p.first_keep, ctx["hidden"].device)
            return keep, {"gate": "same_fixed_L2_choice"}
        single = eng.run_llm_forward(
            prep,V,DS,None,prune_layers={route_layers[0]:("before",select_once)})
        native_delta = float((single.logits.reshape_as(logits).float()-logits.float()).abs().max())
        independent = dict(prefill_max_delta=native_delta,
                           prefill_bitwise_equal=native_delta==0,
                           cache_lengths=_lengths(single.cache,len(eng.text.layers)),
                           decode_comparison_applicable=False,
                           reason="NativeEngine single pruning compacts earlier dense KV caches; CatchUp preserves them")
    native_ok = independent is None or independent["prefill_bitwise_equal"]
    return dict(pass_all=bool(delta==0 and dd==0 and a["token_ids"]==ids and lengths_equal
                             and expected_cache_ok and positions_equal and native_ok
                             and not a["diag"]["reentered"]),
                first_logit_max_delta=delta,decode_max_delta=dd,
                decode_tokens_equal=a["token_ids"]==ids,cache_lengths_equal=lengths_equal,
                expected_cache_lengths=expected_lengths,expected_cache_lengths_equal=expected_cache_ok,
                decode_positions_equal=positions_equal,decode_positions=dec_positions,
                ds_injection_lengths=a["diag"]["ds_injection_lengths"],
                independent_native_single_prune=independent,
                reentered_count=len(a["diag"]["reentered"]),
                preparation_cache_unchanged=p.cache_lengths==_lengths(p.cache,len(eng.text.layers)))
