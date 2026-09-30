"""M13 shared plumbing: DeepStack branch ablation + ViT compression-depth
oracle (report: reports/m13_hierarchy_compression_oracle.md).

Everything reuses the E0 native engine (``model/native_qwen3.py``) and the
M12 pre-encoder sparse path (``model/retinagate.py``).  Two custom vision
forwards live here:

* ``visual_forward_ds`` -- the STOCK vision walk with per-stage timing and
  per-branch DeepStack merger enable flags.  With all branches enabled it is
  bit-identical to ``visual(pv, gthw)``; a disabled branch's merger is
  genuinely skipped (never computed-then-zeroed).
* ``depth_sparse_forward`` -- dense contextualization for the first L blocks,
  then one compression event on native 2x2 merge-group atoms (drop or
  scale-preserving merge), then the remaining blocks on the compressed
  sequence.  Compression at L=0 with mode='drop' is the M12 pre-encoder
  sparse path (bit-identical to ``rg.sparse_visual_forward``).

The LLM side never changes structurally: prefill consumes
``(keep_idx, V_sel, DS_sel)`` exactly like M12; disabled DeepStack branches
are flagged zero rows that a patched ``_deepstack_process`` skips outright.
"""

from __future__ import annotations

import json
import os
import sys
from time import perf_counter

import torch

DISC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "discovery")
sys.path.insert(0, DISC_DIR)
import common  # noqa: E402  (path bootstrap must precede vlmeval imports)

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "m13")
ACC_DIR = os.path.join(OUT_DIR, "acc")
OCR_PANEL = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]

DS_INDEXES = (8, 16, 24)          # Qwen3-VL-8B vision config
N_BLOCKS = 27

# DeepStack branch ablation settings: (name, ds8, ds16, ds24)
DS_SETTINGS = [
    ("native", 1, 1, 1),
    ("A_no8",   0, 1, 1),
    ("B_no16",  1, 0, 1),
    ("C_no24",  1, 1, 0),
    ("D_only24", 0, 0, 1),
    ("E_only16", 0, 1, 0),
    ("F_only8",  1, 0, 0),
    ("G_none",   0, 0, 0),
]

# Compression-depth oracle settings: (L, mode) at K=512; L=0 drop == M12
DEPTH_SETTINGS = [(L, m) for L in (0, 4, 8, 12, 16, 20) for m in ("drop", "merge")]


# ---------------------------------------------------------------------------
# patched DeepStack injection: skip flagged (disabled) branches on the LLM side
# ---------------------------------------------------------------------------
_DS_SKIP_ATTR = "_m13_skip"


def install_ds_skip_patch(eng) -> None:
    """Patch ``text._deepstack_process`` so branch rows flagged with
    ``_m13_skip`` are NOT injected (not even added as zeros)."""
    text = eng.text
    if getattr(text, "_m13_patched", False):
        return
    orig = text._deepstack_process

    def _patched(hidden_states, visual_pos_masks, visual_embeds):
        if getattr(visual_embeds, _DS_SKIP_ATTR, False):
            return hidden_states
        return orig(hidden_states, visual_pos_masks, visual_embeds)

    text._deepstack_process = _patched
    text._m13_patched = True


def make_ds_streams(eng, ds_list, ds_on, n_rows, device):
    """Per-branch [n_rows, D] streams for prefill: real rows for enabled
    branches, flagged zero rows for disabled ones (skipped at injection)."""
    D = eng.text.config.hidden_size
    out = []
    for i, ds in enumerate(ds_list):
        if ds_on[i] and ds is not None:
            out.append(ds)
        else:
            z = torch.zeros(n_rows, D, device=device, dtype=torch.bfloat16)
            setattr(z, _DS_SKIP_ATTR, True)
            out.append(z)
    return out


# ---------------------------------------------------------------------------
# timing helper (sync'd wall clock, like model/retinagate.py's tick)
# ---------------------------------------------------------------------------
def _tick(timings, name, t0):
    if timings is not None:
        torch.cuda.synchronize()
        timings[name] = timings.get(name, 0.0) + (perf_counter() - t0) * 1e3


# ---------------------------------------------------------------------------
# Part A: stock vision walk with per-stage timing + branch skipping
# ---------------------------------------------------------------------------
@torch.no_grad()
def visual_forward_ds(visual, pv: torch.Tensor, gthw: torch.Tensor,
                      ds_on=(1, 1, 1), timings=None):
    """Stock forward, stage-timed; disabled DeepStack mergers are skipped.

    Returns (V [N_merged, D], ds_list of 3 [N_merged, D] or None entries).
    With ds_on=(1,1,1) the output is bit-identical to ``visual(pv, gthw)``.
    """
    def tick(name, t0):
        _tick(timings, name, t0)

    t0 = perf_counter()
    x = visual.patch_embed(pv.to(visual.patch_embed.proj.weight.dtype))
    pos_embeds = visual.fast_pos_embed_interpolate(gthw)
    x = x + pos_embeds
    rotary_pos_emb = visual.rot_pos_emb(gthw)
    seq_len, _ = x.size()
    x = x.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    cu_seqlens = torch.repeat_interleave(
        gthw[:, 1] * gthw[:, 2], gthw[:, 0]).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
    tick("vit_patch_embed_pos_ms", t0)

    ds_lists = [None, None, None]
    for layer_num, blk in enumerate(visual.blocks):
        seg = ("blocks_0_7" if layer_num < 8 else
               "blocks_8_15" if layer_num < 16 else
               "blocks_16_23" if layer_num < 24 else "blocks_24_26")
        t0 = perf_counter()
        x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
        tick(f"vit_{seg}_ms", t0)
        if layer_num in visual.deepstack_visual_indexes:
            i = visual.deepstack_visual_indexes.index(layer_num)
            if ds_on[i]:
                t0 = perf_counter()
                ds_lists[i] = visual.deepstack_merger_list[i](x)
                tick(f"vit_ds_merger_{DS_INDEXES[i]}_ms", t0)
    t0 = perf_counter()
    V = visual.merger(x)
    tick("vit_main_merger_ms", t0)
    return V, ds_lists


# ---------------------------------------------------------------------------
# Part B: merge-partner assignment (depends only on the keep set)
# ---------------------------------------------------------------------------
def build_merge_partner(gthw, ms, keep_groups, dev):
    """Spatial-nearest assignment of removed groups to survivor groups on the
    per-image (gy, gx) grid.  Returns (rem_patch [R*4], surv_slot [R],
    counts [K]): ``surv_slot`` indexes the SURVIVOR ORDER (= ascending
    keep_groups), ``counts[s]`` = number of removed groups assigned to
    survivor s.  Computed once per sample, reused across depths/modes.
    """
    from model.retinagate import group_counts
    gc = group_counts(gthw, ms)
    n_img = int(gthw.shape[0])
    g_off = torch.cumsum(gc, 0) - gc
    p_off = torch.cumsum(gthw.prod(-1).long(), 0) - gthw.prod(-1).long()
    keep_groups = keep_groups.to(dev).long()
    K = int(keep_groups.numel())

    rem_patch_parts, surv_slot_parts = [], []
    counts = torch.zeros(K, dtype=torch.long, device=dev)
    for i in range(n_img):
        mh = int(gthw[i, 1]) // ms
        mw = int(gthw[i, 2]) // ms
        n = mh * mw
        lo, hi = int(g_off[i]), int(g_off[i]) + n
        kg = keep_groups[(keep_groups >= lo) & (keep_groups < hi)] - lo
        keep_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        keep_mask[kg] = True
        rem = torch.nonzero(~keep_mask).reshape(-1)          # [R] local
        if rem.numel() == 0:
            continue
        surv = kg                                            # [S] local, ascending
        gy, gx = rem // mw, rem % mw
        sy, sx = surv // mw, surv % mw
        d2 = (gy[:, None] - sy[None, :]).float() ** 2 + \
             (gx[:, None] - sx[None, :]).float() ** 2
        nearest_local = surv[d2.argmin(dim=1)]               # [R] local
        nearest_global = nearest_local + lo
        slot = torch.searchsorted(keep_groups, nearest_global)
        counts.index_add_(0, slot, torch.ones_like(slot))
        intra = torch.arange(ms * ms, device=dev)
        rem_patch_parts.append(
            (p_off[i] + rem[:, None] * (ms * ms) + intra[None, :]).reshape(-1))
        surv_slot_parts.append(slot)
    if not rem_patch_parts:
        z = torch.zeros(0, dtype=torch.long, device=dev)
        return z, z, counts
    return (torch.cat(rem_patch_parts), torch.cat(surv_slot_parts), counts)


@torch.no_grad()
def depth_sparse_forward(visual, pv: torch.Tensor, gthw: torch.Tensor,
                         L: int, keep_groups: torch.Tensor,
                         mode: str = "drop", partner=None, timings=None):
    """Dense contextualization through blocks ``0..L-1`` on ALL patches, then
    one compression event on 2x2 merge-group atoms, then the remaining
    blocks.  L=0 + mode='drop' subsets patches BEFORE patch_embed (the exact
    M12 pre-encoder path); L=0 + 'merge' embeds everything first (the merge
    needs all patch features).

    Returns (V_sel [K, D], DS_sel list of 3 [K, D]).
    """
    from model.retinagate import patch_index_of_groups

    def tick(name, t0):
        _tick(timings, name, t0)

    ms = visual.spatial_merge_size
    dev = pv.device
    keep_groups = keep_groups.to(dev).long()
    K = int(keep_groups.numel())
    keep_patch = patch_index_of_groups(keep_groups, gthw, ms)

    pre_subset = (L == 0 and mode == "drop")
    t0 = perf_counter()
    if pre_subset:
        x = visual.patch_embed(
            pv[keep_patch].to(visual.patch_embed.proj.weight.dtype))
    else:
        x = visual.patch_embed(pv.to(visual.patch_embed.proj.weight.dtype))
    # position prep always on the FULL grid, then index (O(N), exact)
    pos_embeds_full = visual.fast_pos_embed_interpolate(gthw)
    rotary_full = visual.rot_pos_emb(gthw)
    if pre_subset:
        x = x + pos_embeds_full[keep_patch].to(x.dtype)
        rot = rotary_full[keep_patch]
    else:
        x = x + pos_embeds_full.to(x.dtype)
        rot = rotary_full
    emb = torch.cat((rot, rot), dim=-1)
    cu_full = torch.repeat_interleave(
        gthw[:, 1] * gthw[:, 2], gthw[:, 0]).cumsum(dim=0, dtype=torch.int32)
    cu_full = torch.nn.functional.pad(cu_full, (1, 0), value=0)
    tick("vit_patch_embed_pos_ms", t0)

    # ---- dense contextualization: blocks 0..L-1 on the full sequence ----
    ds_full = {}
    t0 = perf_counter()
    for layer_num in range(L):
        x = visual.blocks[layer_num](x, cu_seqlens=cu_full,
                                     position_embeddings=(emb.cos(), emb.sin()))
        if layer_num in visual.deepstack_visual_indexes:
            i = visual.deepstack_visual_indexes.index(layer_num)
            ds_full[i] = visual.deepstack_merger_list[i](x)   # [N_merged, D]
    tick(f"vit_dense_blocks_0_{max(L-1,0)}_ms", t0)

    # ---- compression event (group atoms) ----
    t0 = perf_counter()
    if pre_subset:
        x_c = x                                  # already compressed
    elif mode == "drop":
        x_c = x[keep_patch]
    elif mode == "merge":
        if partner is None:
            partner = build_merge_partner(gthw, ms, keep_groups, dev)
        rem_patch, surv_slot, counts = partner
        n_patch = int(x.shape[0])
        xk = x[keep_patch].to(x.dtype)           # [K*4, D], intra-slot aligned
        if rem_patch.numel() > 0:
            summed = torch.zeros(K * ms * ms, x.shape[-1], device=dev,
                                 dtype=torch.float32)   # fp32 accumulation
            # rem_patch rows are group-major with intra slot 0..3 consecutive,
            # so a per-group slot expands by repeat_interleave
            surv_row = surv_slot.repeat_interleave(ms * ms)
            intra = rem_patch % (ms * ms)
            summed.index_add_(0, surv_row * (ms * ms) + intra,
                              x[rem_patch].to(torch.float32))
            c = counts.repeat_interleave(ms * ms).to(x.dtype)   # [K*4]
            x_c = (xk + summed.to(x.dtype)) / (1 + c).unsqueeze(1)
        else:
            x_c = xk
    else:
        raise KeyError(mode)
    tick(f"vit_compress_{mode}_ms", t0)

    # survivors keep their own patch rotary positions (emb is already
    # keep-subset in the pre_subset path, full otherwise)
    cos = emb.cos() if pre_subset else emb.cos()[keep_patch]
    sin = emb.sin() if pre_subset else emb.sin()[keep_patch]
    position_embeddings = (cos, sin)
    cu = _kept_cu(gthw, keep_groups, ms, dev)

    # ---- remaining blocks on the compressed sequence ----
    ds_lists = [None, None, None]
    t0 = perf_counter()
    for layer_num in range(L, N_BLOCKS):
        x_c = visual.blocks[layer_num](x_c, cu_seqlens=cu,
                                       position_embeddings=position_embeddings)
        if layer_num in visual.deepstack_visual_indexes:
            i = visual.deepstack_visual_indexes.index(layer_num)
            ds_lists[i] = visual.deepstack_merger_list[i](x_c)   # [K, D]
    tick(f"vit_sparse_blocks_{L}_26_ms", t0)

    t0 = perf_counter()
    V_sel = visual.merger(x_c)                     # [K, D]
    tick("vit_main_merger_ms", t0)

    # DS streams: mergers before L ran dense -> subset by group; after L -> K rows
    DS_sel = []
    for i in range(3):
        if i in ds_full:
            DS_sel.append(ds_full[i][keep_groups])
        else:
            assert ds_lists[i] is not None and int(ds_lists[i].shape[0]) == K
            DS_sel.append(ds_lists[i])
    if timings is not None:
        timings["vit_groups_kept"] = K
        timings["vit_dense_depth"] = L
    return V_sel, DS_sel


def _kept_cu(gthw, keep_groups, ms, dev):
    """cu_seqlens over kept patches (groups are per-image-contiguous in the
    concatenated group coordinate system)."""
    from model.retinagate import group_counts
    gc = group_counts(gthw, ms)
    img_all = torch.repeat_interleave(
        torch.arange(int(gthw.shape[0]), device=dev), gc)
    img = img_all[keep_groups.to(dev).long()]
    kept_per_img = torch.zeros(int(gthw.shape[0]), dtype=torch.long, device=dev)
    kept_per_img.scatter_add_(0, img, torch.ones_like(keep_groups, device=dev))
    cu = torch.cat([torch.zeros(1, dtype=torch.int32, device=dev),
                    kept_per_img.cumsum(0).to(torch.int32)]) * (ms * ms)
    return cu


# ---------------------------------------------------------------------------
# shard IO + scoring (same layout as E0/M12)
# ---------------------------------------------------------------------------
def shard_path(arm_id: str, K: int, ds: str) -> str:
    d = os.path.join(ACC_DIR, arm_id, f"K{K}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{ds}.json")


def load_shard(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"records": {}}


def save_shard(path, shard):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(shard, f, indent=1)
    os.replace(tmp, path)


def score_shard(arm_id: str, K: int, ds: str):
    """Score one shard with the official VLMEvalKit rules (M12 path)."""
    from vlmeval.dataset import build_dataset as vlmeval_build
    import pandas as pd

    path = shard_path(arm_id, K, ds)
    shard = load_shard(path)
    done = [int(k) for k in shard["records"]]
    if not done:
        return None
    dataset = vlmeval_build(ds)
    data = dataset.data
    sub = data.iloc[sorted(done)].copy()
    for col in ("image",):
        if col in sub.columns:
            sub = sub.drop(columns=[col])
    sub["prediction"] = [shard["records"][str(i)]["prediction"] for i in sorted(done)]
    sub["truncated"] = [shard["records"][str(i)]["truncated"] for i in sorted(done)]
    tsv = path.replace(".json", "_pred.tsv")
    sub.to_csv(tsv, sep="\t", index=False)
    try:
        res = dataset.evaluate(tsv)
    except Exception as e:
        print(f"[score FAIL] {arm_id} K={K} {ds}: {e}", flush=True)
        return None
    if hasattr(res, "to_dict"):
        res = res.to_dict()
    per_q = None
    for src, sep in ((path.replace(".json", "_pred_acc.csv"), ","),
                     (path.replace(".json", "_pred_results.tsv"), "\t")):
        if os.path.exists(src):
            d = pd.read_csv(src, sep=sep)
            col = "eval_score" if "eval_score" in d.columns else \
                  ("score" if "score" in d.columns else None)
            if col is not None and "index" in d.columns:
                per_q = {str(int(r["index"])): float(r[col])
                         for _, r in d.iterrows()}
            break
    summary = dict(arm=arm_id, K=K, ds=ds, n=len(done),
                   official={k: (float(v) if isinstance(v, (int, float)) else str(v))
                             for k, v in res.items()},
                   per_question=per_q)
    with open(path.replace(".json", "_score.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[scored] {arm_id} K={K} {ds}: {res}", flush=True)
    return summary


def headline(ds: str, official: dict):
    """Official headline number for a dataset (M12 convention)."""
    if ds == "OCRBench":
        return official.get("Final Score")
    if ds == "DocVQA_VAL":
        return official.get("ANLS")
    return official.get("VQA score")
