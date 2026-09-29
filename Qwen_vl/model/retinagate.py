"""M12 RetinaGate: pre-encoder sparse ViT path for Qwen3-VL (E0 engine).

Everything here attacks the VISION ENCODER, not the LLM.  The unit of
selection is one native 2x2 spatial-merge group (= one merged token).  The
processor emits ``pixel_values`` rows in group-major order
``(t, gy, gx, iy, ix)`` -- the same order ``rot_pos_emb`` and
``fast_pos_embed_interpolate`` produce -- so a kept group's 4 patches are the
consecutive rows ``g*4 .. g*4+3`` and the merged-token index of a group is
exactly its group index.  That gives an exact, order-preserving sparse path:

    keep_groups [G]  ->  keep_patch = g*4 + {0,1,2,3}   (group-contiguous)
    patch_embed(pv[keep_patch])            (Conv3d acts per patch: exact)
    + pos_embeds[keep_patch]               (interpolated on the FULL grid,
                                            then indexed: original 2-D pos)
    rotary = rot_pos_emb(gthw)[keep_patch] (same)
    cu_seqlens from kept per-image counts  (full attention among kept patches)
    blocks(...)                            (real compute reduction)
    merger / deepstack mergers             (view(-1, D*4) stays valid because
                                            kept patches are group-contiguous)

At keep-all this is bit-identical to the stock ``visual`` forward (gate
m12_correctness.py).  Downstream, ``NativeEngine.prefill`` consumes
``(keep_idx=keep_groups, V_sel, DS_sel)`` unchanged: merged features,
DeepStack streams and the FULL-sequence 3-D mRoPE positions all shrink with
the same index vector, decode positions continue from ``prefill_max_pos``.

The selector itself is deliberately pre-encoder-cheap: from the normalized
``pixel_values`` rows we rebuild the per-patch mean RGB (the rows ARE the
image at patch resolution), invert the processor normalization, and compute
opponent-channel center-surround energy on the patch grid.  No network, no
extra forward.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# group / patch geometry
# ---------------------------------------------------------------------------
def group_counts(gthw: torch.Tensor, ms: int) -> torch.Tensor:
    return (gthw[:, 1] // ms) * (gthw[:, 2] // ms)


def patch_index_of_groups(keep_groups: torch.Tensor, gthw: torch.Tensor,
                          ms: int) -> torch.Tensor:
    """Global patch indices of every patch of every kept group.

    ``keep_groups`` indexes the CONCATENATED per-image merged sequence
    (same coordinate system as merged features / E0 keep_idx).
    """
    n_img = int(gthw.shape[0])
    gc = group_counts(gthw, ms)
    pc = gthw.prod(-1).long()
    g_off = torch.cumsum(gc, 0) - gc           # first group idx of each image
    p_off = torch.cumsum(pc, 0) - pc           # first patch row of each image
    img_all = torch.repeat_interleave(
        torch.arange(n_img, device=gthw.device), gc)
    keep_groups = keep_groups.to(gthw.device).long()
    img = img_all[keep_groups]
    local = keep_groups - g_off[img]
    intra = torch.arange(ms * ms, device=gthw.device)
    return (p_off[img, None] + local[:, None] * (ms * ms) + intra[None, :]
            ).reshape(-1)


# ---------------------------------------------------------------------------
# sparse visual forward (the real pre-encoder reduction)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sparse_visual_forward(visual, pv: torch.Tensor, gthw: torch.Tensor,
                          keep_groups: torch.Tensor,
                          timings: Optional[Dict] = None):
    """Run the vision tower on ONLY the kept merge-groups' patches.

    Returns ``(V_sel [G, out_D], DS_sel list of [G, D])`` in ascending group
    order -- directly consumable by ``NativeEngine.prefill`` via
    ``keep_idx=keep_groups, V_sel=..., DS_sel=...``.
    """
    ms = visual.spatial_merge_size
    dev = pv.device

    def tick(name, t0):
        if timings is not None:
            torch.cuda.synchronize()
            timings[name] = timings.get(name, 0.0) + \
                (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    keep_patch = patch_index_of_groups(keep_groups, gthw, ms)
    n_kept = int(keep_patch.numel())
    x = visual.patch_embed(pv[keep_patch].to(visual.patch_embed.proj.weight.dtype))
    tick("vit_patch_embed_ms", t0)

    t0 = time.perf_counter()
    # O(N) prep: interpolate / build on the FULL grid once, then index.
    pos_embeds_full = visual.fast_pos_embed_interpolate(gthw)
    rotary_full = visual.rot_pos_emb(gthw)
    x = x + pos_embeds_full[keep_patch].to(x.dtype)
    emb = torch.cat((rotary_full[keep_patch], rotary_full[keep_patch]), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    # per-image kept lengths -> cu_seqlens (full attention within each image)
    gc = group_counts(gthw, ms)
    img_all = torch.repeat_interleave(
        torch.arange(int(gthw.shape[0]), device=dev), gc)
    img = img_all[keep_groups.to(dev).long()]
    kept_per_img = torch.zeros(int(gthw.shape[0]), dtype=torch.long, device=dev)
    kept_per_img.scatter_add_(0, img, torch.ones_like(keep_groups, device=dev))
    cu = torch.cat([torch.zeros(1, dtype=torch.int32, device=dev),
                    kept_per_img.cumsum(0).to(torch.int32)]) * (ms * ms)
    tick("vit_pos_prep_ms", t0)

    t0 = time.perf_counter()
    deepstack_feature_lists = []
    for layer_num, blk in enumerate(visual.blocks):
        x = blk(x, cu_seqlens=cu, position_embeddings=position_embeddings)
        if layer_num in visual.deepstack_visual_indexes:
            ds_merger = visual.deepstack_merger_list[
                visual.deepstack_visual_indexes.index(layer_num)]
            deepstack_feature_lists.append(ds_merger(x))
    tick("vit_blocks_ms", t0)

    t0 = time.perf_counter()
    V_sel = visual.merger(x)
    tick("vit_merger_ms", t0)

    if timings is not None:
        timings["vit_input_patches"] = n_kept
        timings["vit_groups_kept"] = int(keep_groups.numel())
    return V_sel, deepstack_feature_lists


def vision_flopsgf(n_patch: int, cfg) -> float:
    """Analytic ViT FLOPs (GF) for n_patch input patches (single image)."""
    D = cfg.hidden_size
    H = cfg.num_heads
    dh = D // H
    L = cfg.depth
    I = cfg.intermediate_size
    lin = 2 * n_patch * D * (3 * D + D) + 2 * n_patch * D * (2 * I)
    attn = 2 * 2 * H * n_patch * n_patch * dh
    return L * (lin + attn) / 1e9


# ---------------------------------------------------------------------------
# retina signal (cheap, pre-encoder, from pixel_values rows)
# ---------------------------------------------------------------------------
def pv_to_rgb_maps(pv: torch.Tensor, gthw: torch.Tensor, ms: int,
                   image_mean, image_std, tps: int = 2) -> Dict[int, torch.Tensor]:
    """Per-image raw-RGB patch map ``[H, W, 3]`` (H = grid_h patches).

    ``pv`` rows are normalized group-major patches; invert the per-channel
    affine to recover raw means (contrast metrics are affine-invariant per
    channel, but the cross-channel opponents RG/BY are not, so we invert).
    Row layout is ``(channel, temporal_patch_size, ph, pw)`` -- note the
    temporal axis of the ROW is ``temporal_patch_size`` (2 for single
    images), not ``grid_thw``'s t.
    """
    mean = torch.tensor(image_mean, device=pv.device, dtype=torch.float32)
    std = torch.tensor(image_std, device=pv.device, dtype=torch.float32)
    ph = pw = 16  # patch size; rows are [C*t*ph*pw], channel-major blocks
    out = {}
    off = 0
    for i in range(int(gthw.shape[0])):
        t, h, w = (int(v) for v in gthw[i])
        n = t * h * w
        rows = pv[off:off + n].float()           # [n, C*tps*ph*pw]
        # channel blocks: layout (channel, temporal, ph, pw) per the processor
        ch = rows.reshape(h * w, 3, tps, ph * pw).mean(dim=(2, 3))  # [n, 3]
        raw = ch * std[None, :] + mean[None, :]  # de-normalize
        m = raw.reshape(h // ms, w // ms, ms, ms, 3)
        m = m.permute(0, 2, 1, 3, 4).reshape(h, w, 3)  # (gy,iy),(gx,ix) -> y,x
        out[i] = m
        off += n
    return out


def center_surround_energy(x: torch.Tensor, multi_scale: bool = True) -> torch.Tensor:
    """|x - avg3(x)| + 0.5*|x - avg5(x)| on a [H, W] map (patch resolution).

    ``multi_scale=False`` keeps only the 3x3 term (single-scale ablation).
    """
    x4 = x[None, None, :, :]
    e = (x4 - F.avg_pool2d(x4, 3, stride=1, padding=1)).abs()
    if multi_scale:
        e = e + 0.5 * (x4 - F.avg_pool2d(x4, 5, stride=1, padding=2)).abs()
    return e[0, 0]


def retina_channels(rgb_map: torch.Tensor, ms: int, single_scale: bool = False):
    """Y / RG / BY center-surround energies aggregated to the group grid."""
    y = 0.299 * rgb_map[..., 0] + 0.587 * rgb_map[..., 1] + 0.114 * rgb_map[..., 2]
    rg = rgb_map[..., 0] - rgb_map[..., 1]
    by = rgb_map[..., 2] - 0.5 * (rgb_map[..., 0] + rgb_map[..., 1])
    e = {}
    for name, m in (("Y", y), ("RG", rg), ("BY", by)):
        e[name] = _group_pool(center_surround_energy(m, multi_scale=not single_scale),
                              ms)
    return e


def _group_pool(patch_map: torch.Tensor, ms: int) -> torch.Tensor:
    """Mean-pool a [H, W] patch map onto the (H/ms, W/ms) group grid."""
    h, w = patch_map.shape
    return F.avg_pool2d(patch_map[None, None], ms).reshape(h // ms, w // ms)


def retina_event_score(rgb_map: torch.Tensor, ms: int,
                       lam_rg: float = 0.5, lam_by: float = 0.5,
                       single_scale: bool = False):
    """Per-image-normalized opponent-channel event score on the group grid.

    Returns (event [Mh, Mw], per-channel dict) with each channel scaled by
    its own mean (per-image normalization; no learned parameters).
    """
    e = retina_channels(rgb_map, ms, single_scale=single_scale)
    norm = {k: v / (v.mean() + 1e-6) for k, v in e.items()}
    event = norm["Y"] + lam_rg * norm["RG"] + lam_by * norm["BY"]
    return event, e


def variance_score(rgb_map: torch.Tensor, ms: int) -> torch.Tensor:
    """Local variance of luminance within each group."""
    y = 0.299 * rgb_map[..., 0] + 0.587 * rgb_map[..., 1] + 0.114 * rgb_map[..., 2]
    h, w = y.shape
    g = y.reshape(h // ms, ms, w // ms, ms).permute(0, 2, 1, 3)
    var = g.var(dim=(-2, -1))
    return var


def sobel_score(rgb_map: torch.Tensor, ms: int) -> torch.Tensor:
    """Sobel gradient-magnitude energy of luminance, grouped."""
    y = 0.299 * rgb_map[..., 0] + 0.587 * rgb_map[..., 1] + 0.114 * rgb_map[..., 2]
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=y.device)[None, None]
    ky = kx.transpose(-1, -2)
    y4 = y[None, None]
    gx = F.conv2d(y4, kx, padding=1)
    gy = F.conv2d(y4, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)[0, 0]
    return _group_pool(mag, ms)


# ---------------------------------------------------------------------------
# group selection (exact budget, raster-ascending, duplicate-free)
# ---------------------------------------------------------------------------
def uniform_lattice(n_groups: int, mh: int, mw: int, k: int) -> torch.Tensor:
    """2-D lattice covering: stride s.t. count >= k, then even subsample."""
    if k >= n_groups:
        return torch.arange(n_groups)
    best = None
    for sy in range(1, mh + 1):
        for sx in range(1, mw + 1):
            rows = len(range(0, mh, sy))
            cols = len(range(0, mw, sx))
            n = rows * cols
            if n >= k and (best is None or n < best[0]):
                best = (n, sy, sx)
    n, sy, sx = best
    rows = torch.arange(0, mh, sy)
    cols = torch.arange(0, mw, sx)
    grid = (rows[:, None] * mw + cols[None, :]).reshape(-1)
    if n > k:  # even deterministic subsample of the lattice
        sel = torch.linspace(0, n - 1, k).round().long().unique()
        while sel.numel() < k:  # linspace can collide after rounding
            sel = torch.linspace(0, n - 1, k + (k - sel.numel())) \
                .round().long().unique()
        grid = grid[sel[:k]]
    return grid.sort().values


def base_plus_event(event: torch.Tensor, k: int, base_frac: float = 0.32):
    """S_base (lattice) + fill with top event score.  Exact budget k."""
    dev = event.device
    mh, mw = event.shape
    n_groups = mh * mw
    k = min(k, n_groups)
    n_base = min(int(round(k * base_frac)), k)
    base = uniform_lattice(n_groups, mh, mw, n_base).to(dev)
    rest_budget = k - n_base
    mask = torch.ones(n_groups, dtype=torch.bool, device=dev)
    mask[base] = False
    cand = torch.nonzero(mask).reshape(-1)
    scores = event.reshape(-1)[cand]
    n_rest = min(rest_budget, cand.numel())
    top = cand[torch.topk(scores, n_rest).indices]
    keep = torch.cat([base, top]).sort().values
    return keep, base


# ---------------------------------------------------------------------------
# pre-ViT selector wrapper for the E0 engine
# ---------------------------------------------------------------------------
def select_groups(mode: str, k: int, prep: dict, eng,
                  lam_rg: float = 0.5, lam_by: float = 0.5,
                  base_frac: float = 0.32, seed: Optional[int] = None,
                  timing: Optional[Dict] = None):
    """Return (keep_groups, info).  Budget in MERGED tokens (= groups)."""
    gthw, ms = prep["gthw"], eng.inner.visual.spatial_merge_size
    dev = prep["pv"].device
    info = {}
    keeps = []
    t0 = time.perf_counter()
    off_g = 0
    rgb_maps = pv_to_rgb_maps(prep["pv"], gthw, ms,
                              eng.vlm.processor.image_processor.image_mean,
                              eng.vlm.processor.image_processor.image_std,
                              tps=eng.inner.visual.patch_embed.temporal_patch_size)
    for i in range(int(gthw.shape[0])):
        mh = int(gthw[i, 1]) // ms
        mw = int(gthw[i, 2]) // ms
        n = mh * mw
        kk = min(k, n)
        if kk >= n:
            keeps.append(torch.arange(off_g, off_g + n, device=dev))
            off_g += n
            continue
        if mode in ("random",):
            g = torch.Generator().manual_seed(
                (seed if seed is not None else 0) + 1000 * off_g)
            keeps.append(off_g + torch.randperm(n, generator=g)[:kk].sort().values)
        elif mode in ("uniform",):
            keeps.append(off_g + uniform_lattice(n, mh, mw, kk))
        else:
            rgb = rgb_maps[i]
            if mode == "retinagate":
                event, _ = retina_event_score(rgb, ms, lam_rg, lam_by)
                keep, base = base_plus_event(event, kk, base_frac)
                info.setdefault("n_base", []).append(int(base.numel()))
                keeps.append(off_g + keep)
            elif mode == "retinagate_s1":
                event, _ = retina_event_score(rgb, ms, lam_rg, lam_by,
                                              single_scale=True)
                keep, base = base_plus_event(event, kk, base_frac)
                info.setdefault("n_base", []).append(int(base.numel()))
                keeps.append(off_g + keep)
            elif mode == "event":
                event, _ = retina_event_score(rgb, ms, lam_rg, lam_by)
                keeps.append(off_g + event.reshape(-1).topk(kk).indices.sort().values)
            elif mode == "event_s1":
                event, _ = retina_event_score(rgb, ms, lam_rg, lam_by,
                                              single_scale=True)
                keeps.append(off_g + event.reshape(-1).topk(kk).indices.sort().values)
            elif mode == "graydog":
                e = _group_pool(center_surround_energy(
                    0.299 * rgb[..., 0] + 0.587 * rgb[..., 1]
                    + 0.114 * rgb[..., 2]), ms)
                keeps.append(off_g + e.reshape(-1).topk(kk).indices.sort().values)
            elif mode == "sobel":
                e = sobel_score(rgb, ms)
                keeps.append(off_g + e.reshape(-1).topk(kk).indices.sort().values)
            elif mode == "variance":
                e = variance_score(rgb, ms)
                keeps.append(off_g + e.reshape(-1).topk(kk).indices.sort().values)
            else:
                raise KeyError(f"unknown pre-ViT selector mode {mode!r}")
        off_g += n
    if timing is not None:
        torch.cuda.synchronize()
        timing["gate_ms"] = (time.perf_counter() - t0) * 1e3
    return torch.cat([k.to(dev) for k in keeps]), info


@torch.no_grad()
def rg_generate(eng, message, dataset_name=None, k: int = 512,
                mode: str = "retinagate", max_new_tokens: int = 2048,
                ignore_eos: bool = False, timings: Optional[Dict] = None,
                seed: Optional[int] = None, base_frac: float = 0.32,
                lam_rg: float = 0.5, lam_by: float = 0.5):
    """Pre-encoder arm: gate -> sparse ViT -> stock prefill -> decode.

    Mirrors ``NativeEngine.generate`` but the vision tower sees only the
    kept groups' patches.  ``timings`` keys stay compatible with the E0
    perf harness, plus ``vit_*`` stage breakdown and ``gate_ms``.
    """
    timings = timings if timings is not None else {}
    wall0 = time.perf_counter()
    prep = eng.prepare(message, dataset_name)
    timings["image_preprocess_ms"] = (time.perf_counter() - wall0) * 1e3

    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    keep_groups, info = select_groups(
        mode, k, prep, eng, lam_rg=lam_rg, lam_by=lam_by,
        base_frac=base_frac, seed=seed, timing=timings)
    keep_groups = keep_groups.to(prep["pv"].device)
    st_vit = {}
    V_sel, DS_sel = sparse_visual_forward(
        eng.inner.visual, prep["pv"], prep["gthw"], keep_groups,
        timings=st_vit)
    ev[1].record()
    torch.cuda.synchronize()
    timings["vision_ms"] = ev[0].elapsed_time(ev[1])
    for k_, v_ in st_vit.items():
        timings[k_] = v_

    evp = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    evp[0].record()
    st = eng.prefill(prep, None, None, keep_groups, deepstack=True,
                     pos="mrope3d", V_sel=V_sel, DS_sel=DS_sel)
    evp[1].record()
    torch.cuda.synchronize()
    timings["llm_prefill_ms"] = evp[0].elapsed_time(evp[1])
    timings["ttft_ms"] = (time.perf_counter() - wall0) * 1e3
    wall1 = time.perf_counter()
    gen_ids, text = eng.decode(st, max_new_tokens, ignore_eos=ignore_eos)
    timings["decode_wall_ms"] = (time.perf_counter() - wall1) * 1e3
    torch.cuda.synchronize()
    meta = eng.invariants(prep, st, None, None, n_decode=len(gen_ids))
    meta["n_groups_kept"] = int(keep_groups.numel())
    meta["info"] = {k_: v_ for k_, v_ in info.items()}
    return dict(text=text, gen_ids=gen_ids, keep_idx=st.keep_idx, meta=meta,
                timings=timings, state=st)
