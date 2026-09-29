"""PACE port to Qwen3-VL (EMNLP 2026 Findings) -- official repo jjL357/PACE
@ 240b2206, E0 prereg §4.1.

Port of the official Qwen2.5-VL machinery:
  * APC (adaptive_pixel_compressor.py, verbatim math): a depth-1 ViT preview
    estimates information density -> a patch-aligned compressed resolution;
    the FULL vision pass then runs on the compressed image (this, not the
    decoder token count, is where PACE saves encoder time);
  * DDAE (modeling_pace_qwen2_5_vl.py, DDAE extraction section): at LLM layer
    2, fuse the previous layer's semantic attention (head- and query-mean over
    the visual span) with the vision attention score using the official
    std-ratio softmax weights (fusion_temperature), keep the top tokens.

Qwen3-VL adaptations per prereg §4.1: patch unit 32 px (16 px patch x 2x2
merge); no window attention (the preview drops get_window_index, all blocks
full attention); fast_pos_embed_interpolate re-runs on the compressed grid
(the stock vision walk does this from grid_thw); DeepStack streams shrink with
the DDAE keep index (engine-side compaction).

Budget alignment (frozen in the amendment before any number): the DDAE stage
keeps exactly K tokens; APC's retention stays data-adaptive (the method's own
behaviour) and the realized ViT patch count / token count distribution is
reported per prereg §4.1.3.
"""

from __future__ import annotations

import io
import math
from base64 import b64decode

import torch
import torch.nn.functional as F
from PIL import Image


# --------------------------------------------------------------------------
# official APC math (adaptive_pixel_compressor.py)
# --------------------------------------------------------------------------
def compressed_resolution(height, width, patch_size, retention_ratio):
    original_patch_h = math.ceil(height / patch_size)
    original_patch_w = math.ceil(width / patch_size)
    original_tokens = original_patch_h * original_patch_w
    target_tokens = max(1, round(original_tokens * retention_ratio))
    aspect = height / width
    ideal_w = math.sqrt(target_tokens / aspect)
    ideal_h = ideal_w * aspect
    candidates = (
        (max(1, math.floor(ideal_h)), max(1, math.floor(ideal_w))),
        (max(1, math.floor(ideal_h)), max(1, math.ceil(ideal_w))),
        (max(1, math.ceil(ideal_h)), max(1, math.floor(ideal_w))),
        (max(1, math.ceil(ideal_h)), max(1, math.ceil(ideal_w))),
    )
    ph, pw = min(candidates, key=lambda it: (
        abs(it[0] * it[1] - target_tokens), abs(it[0] / it[1] - aspect)))
    return ph * patch_size, pw * patch_size


def _normalize(features):
    features = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(features.reshape(features.shape[0], -1), p=2, dim=-1,
                       eps=1e-12)


def apc_score(features, global_weight=0.6, detail_fraction=0.1,
              detail_scale=1.5, minimum_retention=0.05):
    feats = _normalize(features)
    n = feats.shape[0]
    if n <= 1:
        return minimum_retention, 0.0, 0.0
    fsum = feats.sum(dim=0)
    self_sim = feats.square().sum()
    pair = (torch.dot(fsum, fsum) - self_sim) / (n * (n - 1))
    global_density = (1.0 - pair).clamp(0.0, 1.0)
    bg = F.normalize((fsum / n).unsqueeze(0), p=2, dim=-1, eps=1e-12)
    dist = torch.linalg.vector_norm(feats - bg, dim=-1)
    detail_n = min(n, max(1, math.ceil(n * detail_fraction)))
    local_detail = (dist.topk(detail_n).values.mean() / detail_scale).clamp(0.0, 1.0)
    retention = (global_weight * global_density
                 + (1 - global_weight) * local_detail).clamp(minimum_retention, 1.0)
    return retention.item(), global_density.item(), local_detail.item()


# --------------------------------------------------------------------------
# Qwen3 vision passes
# --------------------------------------------------------------------------
def preview_features(visual, pixel_values, grid_thw, depth=1):
    """Official ShallowFeaturePreview without the window-attention branch
    (Qwen3 ViT is all-full-attention): first `depth` blocks over the patch
    sequence, pre-merger."""
    hidden_states = visual.patch_embed(pixel_values)
    hidden_states = hidden_states + visual.fast_pos_embed_interpolate(grid_thw)
    rotary = visual.rot_pos_emb(grid_thw)
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary = rotary.reshape(seq_len, -1)
    emb = torch.cat((rotary, rotary), dim=-1)
    pos_emb = (emb.cos(), emb.sin())
    cu = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                 grid_thw[:, 0]).cumsum(0, dtype=torch.int32)
    cu = F.pad(cu, (1, 0), value=0)
    for bi in range(depth):
        hidden_states = visual.blocks[bi](hidden_states, cu_seqlens=cu,
                                          position_embeddings=pos_emb)
        if isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
    return hidden_states


def vision_pass_with_score(visual, pixel_values, grid_thw):
    """Full vision pass; also returns the last block's received-attention
    score per MERGED token (head-mean, query-sum; PACE's vision_attention_score)."""
    from model.baselines.visionzip import visual_forward_with_visionzip
    V, DS, attn_mean, _ = visual_forward_with_visionzip(visual, pixel_values,
                                                        grid_thw)
    return V, DS, attn_mean


# --------------------------------------------------------------------------
# image loading under the harness convention
# --------------------------------------------------------------------------
def _load_expanded_image(engine, message, side=1024):
    from vlmeval.vlm.qwen3_vl.model_fixed_res import expand2square
    content = engine.vlm._prepare_content(message, dataset=None)
    imgs = [c for c in content if c.get("type") == "image"]
    assert len(imgs) == 1, "PACE port supports single-image messages"
    src = imgs[0]["image"]
    if src.startswith("data:image") or src.startswith("/9j") or len(src) > 200:
        img = Image.open(io.BytesIO(b64decode(src.split(",")[-1])))
    else:
        img = Image.open(src)
    mean = getattr(engine.vlm.processor.image_processor, "image_mean",
                   [0.5, 0.5, 0.5])
    bg = tuple(int(x * 255) for x in mean)
    return expand2square(img.convert("RGB"), bg).resize((side, side))


def _process(engine, img):
    text = engine.vlm.processor.apply_chat_template(
        engine.vlm._build_messages([{"type": "text", "value": "x"}]
                                   if False else []),
        tokenize=False, add_generation_prompt=True) if False else None
    return img


def pace_prepare(engine, message, dataset_name, K):
    """APC-fronted preprocessing + vision pass. Returns a native prep dict."""
    from vlmeval.vlm.qwen3_vl.model_fixed_res import expand2square
    eng = engine
    img = _load_expanded_image(eng, message, side=1024)

    # --- APC preview at the full 1024 grid (patch 16 -> 4096 patch tokens)
    img0 = img
    text = eng.vlm.processor.apply_chat_template(
        [{"role": "user",
          "content": [{"type": "image", "image": ""}, {"type": "text", "text": "x"}]}],
        tokenize=False, add_generation_prompt=True)
    proc = eng.vlm.processor
    inputs0 = proc(text=text, images=[img0], videos=None, do_resize=False,
                   return_tensors="pt")
    dev = next(eng.model.parameters()).device
    pv0 = inputs0["pixel_values"].type(eng.inner.visual.dtype).to(dev)
    g0 = inputs0["image_grid_thw"].to(dev)
    with torch.no_grad():
        feats = preview_features(eng.inner.visual, pv0, g0, depth=1)
        retention, gd, ld = apc_score(feats)

    # --- compressed resolution, patch unit 32 (merged token)
    h2, w2 = compressed_resolution(img.height, img.width, 32, retention)
    img2 = img.resize((w2, h2), Image.BICUBIC)

    # official preprocessing path on the compressed image
    inputs = eng.vlm._processor_inputs(eng.vlm._build_messages(message,
                                                               dataset=dataset_name))
    # replace the pixel input with the compressed image (same prompt text)
    from qwen_vl_utils import process_vision_info
    messages = eng.vlm._build_messages(message, dataset=dataset_name)
    text = proc.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    images, videos, vk = process_vision_info(messages, image_patch_size=16,
                                             return_video_kwargs=True,
                                             return_video_metadata=True)
    inputs2 = proc(text=text, images=[img2], videos=None, do_resize=False,
                   return_tensors="pt", **(vk or {}))
    inputs2 = eng.vlm._move_inputs_to_model(inputs2)

    ids = inputs2["input_ids"]
    pos = (ids[0] == eng.image_token_id).nonzero(as_tuple=True)[0]
    prep = dict(inputs=inputs2, ids=ids, has_image=True, message=message,
                ds=dataset_name, img_start=int(pos[0]),
                img_end=int(pos[-1]) + 1, n_vis=int(pos[-1]) - int(pos[0]) + 1,
                gthw=inputs2["image_grid_thw"],
                pv=inputs2["pixel_values"].type(eng.inner.visual.dtype),
                apc=dict(retention=retention, global_density=gd,
                         local_detail=ld, side=(h2, w2)))
    with torch.no_grad():
        V, DS, vis_score = vision_pass_with_score(eng.inner.visual, prep["pv"],
                                                  prep["gthw"])
    return prep, V, DS, vis_score


# --------------------------------------------------------------------------
# DDAE
# --------------------------------------------------------------------------
def build_ddae(K, vis_score, n_vis, extraction_layer=2,
               fusion_temperature=0.5):
    """prereq §4.1: DDAE fuses the PREVIOUS LLM layer's semantic attention
    with the vision score (official std-ratio weights) and keeps K tokens."""
    stash = {}

    def record(ctx):
        stash["attn"] = ctx["attn_weights"]
        return ctx["vis_mask"].nonzero(as_tuple=True)[0], {}, None

    def ddae(ctx):
        attn = stash["attn"].float()                       # [1, H, L, L]
        vis = ctx["vis_mask"]
        vis_idx = vis.nonzero(as_tuple=True)[0]
        llm = attn[0].mean(0)[:, vis_idx].mean(0)          # head+query mean
        llm_n = (llm - llm.min()) / (llm.max() - llm.min() + 1e-6)
        if vis_score is not None:
            vs = vis_score.float().to(llm.device)
            if vs.shape[0] != llm_n.shape[0]:
                # preview token count != merged count; interpolate to align
                vs = torch.nn.functional.interpolate(
                    vs.view(1, 1, -1), size=llm_n.shape[0], mode="nearest")[0, 0]
            vs_n = (vs - vs.min()) / (vs.max() - vs.min() + 1e-6)
            w = torch.softmax(torch.stack([llm_n.std(), vs_n.std()]) /
                              fusion_temperature, dim=0)
            score = w[0] * llm_n + w[1] * vs_n
        else:
            score = llm_n
        keep_num = min(K, int(vis.sum()))
        top = score.topk(keep_num).indices
        keep_vis = vis_idx[top.sort().values]
        text_idx = (~vis).nonzero(as_tuple=True)[0]
        keep_local = torch.sort(torch.cat([text_idx, keep_vis])).values
        return keep_local, dict(n_vis_keep=keep_num), None

    return {extraction_layer - 1: ("after", record),
            extraction_layer: ("before", ddae)}


# --------------------------------------------------------------------------
# full arm
# --------------------------------------------------------------------------
def generate(engine, message, dataset_name, K, deepstack, pos,
             max_new_tokens, timings):
    """PACE arm entry: returns the same result dict as NativeEngine.generate."""
    import time as _t
    from model.native_qwen3 import PrefillState
    eng = engine
    wall0 = _t.perf_counter()
    prep, V, DS, vis_score = pace_prepare(eng, message, dataset_name, K)
    timings["image_preprocess_ms"] = (_t.perf_counter() - wall0) * 1e3
    # the preview + full pass already ran inside pace_prepare; vision_ms is
    # reported as the compressed-grid full pass (the last vision call)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    tm, ts = eng.instruction_embeds(message, dataset_name)
    pl = build_ddae(K, vis_score, prep["n_vis"])
    st = eng.run_llm_forward(prep, V, DS, None, deepstack=deepstack, pos=pos,
                             prune_layers=pl)
    ev[1].record()
    torch.cuda.synchronize()
    timings["llm_prefill_ms"] = ev[0].elapsed_time(ev[1])
    timings["vision_ms"] = float("nan")   # measured inside preprocess window
    timings["selector_ms"] = 0.0
    timings["ttft_ms"] = (_t.perf_counter() - wall0) * 1e3
    gen_ids, text_out = eng.decode(st, max_new_tokens)
    timings["decode_wall_ms"] = (_t.perf_counter() - wall1) * 1e3 \
        if (wall1 := _t.perf_counter()) else 0
    meta = eng.invariants(prep, st, V, DS, n_decode=len(gen_ids))
    meta["apc"] = prep["apc"]
    return dict(text=text_out, gen_ids=gen_ids, keep_idx=st.keep_idx,
                meta=meta, timings=timings, state=st)
