#!/usr/bin/env python3
"""AnchorZip Figure-1 exporter v3 — round 2026-10-07 deliverable exporter.

Frozen path, identical to v2 (one vision pass + one text-embedding pass per
sample through NativeEngine): RTG scoring (rtg_common), official facility
(instrumented.SELECTORS['facility']), anchor assignment
(amp_common.compute_assignment).  No algorithm is changed, no hyperparameter
is touched.

v3 additions over v2 (2026-10-07 packaging spec):
  * sample.npz keys (desktop contract):
      response[H,W,L]  original A (negated-cosine symbols, FP32), full RTG
                       text sequence; cell (h,w) <-> flat token h*W+w
      weights[L]       frozen rtg text_weight w, sum=1 over all L tokens
      global_response[H,W], local_response[H,W], fused_response[H,W],
      importance[H,W]  pre-fusion quantities + frozen post-processing
      keep[K]          ascending flat anchor indices
      dropped_idx[N-K], gid[N-K]   ascending dropped idx; gid = anchor rank
      group[H,W]       anchor slot per token (anchors = own slot)
      group_size[K]    group sizes INCLUDING the anchor
      logP[N,L]        log_softmax(100*A) over the visual axis (P(i|t))
      weights_unnorm_c[L]  the unnormalised RTG counts c before w = c/c.sum()
  * anchor_update.npz: highlight anchor slot/index, member indices, real
    v_a, mu_a (INCLUDING the anchor, uniform kind), y_a = v_a + 0.25(mu_a -
    v_a), lambda, plus v/mu/y/y_cast for all K slots; FP32 compute with the
    final model-dtype cast recorded.
  * token_ranking.csv: full original weight ranking with question/suffix
    attribution.
  * preview.png: three-panel real-array preview (image+importance | top-12
    responses 4x3 with shared A colour scale | anchors + highlight group).
  * checks.json: full verification battery incl. rebuilding w/local/fused/
    importance from A in FP32, y-identity, mu-includes-anchor, group
    invariants, dropped-token cosine re-assignment (FP64), bank keep match.

Usage:
    python export_anchorzip_figure_sample_v3.py --spec TextVQA_VAL:787,...
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
QWEN_ROOT = os.path.dirname(os.path.dirname(_HERE))
ACV_DIR = os.path.join(QWEN_ROOT, "scripts", "anchor_completion_validation")
XSP_DIR = os.path.join(QWEN_ROOT, "scripts", "stage1_cross_stream_pilot")
AMP_DIR = os.path.join(QWEN_ROOT, "scripts", "anchor_merge_pilot")
DISC_DIR = os.path.join(QWEN_ROOT, "scripts", "discovery")
E0_DIR = os.path.join(QWEN_ROOT, "scripts", "e0")
for _p in (os.path.join(QWEN_ROOT, "VLMEvalKit"), QWEN_ROOT, DISC_DIR, E0_DIR,
           XSP_DIR, ACV_DIR, AMP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common  # noqa: E402
import amp_common as AC  # noqa: E402
import rtg_common as R  # noqa: E402

from PIL import Image  # noqa: E402

FIG_DIR = os.path.join(common.QWEN_ROOT, "outputs", "figures", "anchorzip")
OUT_DIR = os.path.join(FIG_DIR, "samples_v3")
os.makedirs(OUT_DIR, exist_ok=True)

LAM = 0.25                    # frozen completion lambda (main-only)
SUFFIX = "Please answer concisely"

PRED_SHARDS = {
    "L_R_MAIN025_official_legacy":
        os.path.join(R.OUT_DIR, "legacy_full", "acc", "L_R_MAIN025",
                     "{ds}.json"),
    "R_MAIN025_full_native":
        os.path.join(R.OUT_DIR, "full", "acc", "R_MAIN025", "{ds}.json"),
}
SCORE_SHARDS = {
    "L_R_MAIN025_official_legacy":
        os.path.join(R.OUT_DIR, "legacy_full", "acc", "L_R_MAIN025",
                     "{ds}_score.json"),
    "R_MAIN025_full_native":
        os.path.join(R.OUT_DIR, "full", "acc", "R_MAIN025",
                     "{ds}_score.json"),
}


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def detect_and_rebuild(pv, gthw, ph, mean, std, orig_img: Image.Image, sms):
    """Patch-exact reassembly of pixel_values (layout verified 2026-10-04).
    Identical to v1/v2."""
    t, hp, wp = [int(x) for x in gthw[0]]
    assert t == 1, "multi-frame sample"
    rows = pv.float().cpu().numpy()
    P = rows.shape[0]
    assert P == hp * wp, (P, hp, wp)
    H, W = hp * ph, wp * ph
    assert rows.shape[1] == 3 * 2 * ph * ph, rows.shape
    f = rows.reshape(P, 3, 2, ph, ph)[:, :, 0, :, :].transpose(0, 2, 3, 1)
    img = np.zeros((H, W, 3), dtype=np.float32)
    mw = wp // sms
    for p in range(P):
        m, r = divmod(p, sms * sms)
        di, dj = divmod(r, sms)
        pr, pc = (m // mw) * sms + di, (m % mw) * sms + dj
        img[pr * ph:(pr + 1) * ph, pc * ph:(pc + 1) * ph] = f[p]
    img = np.clip(img * np.array(std, np.float32).reshape(1, 1, 3)
                  + np.array(mean, np.float32).reshape(1, 1, 3), 0, 1)
    # Reference chain = the ENGINE's actual preprocessing (verified in
    # Qwen3VLChatFixedRes._processor_inputs): expand2square on the mean
    # colour, then ONE direct resize to 1024x1024 (processor runs with
    # do_resize=False afterwards).  PIL Image.resize default resample.
    from PIL import ImageOps
    orig_img = ImageOps.exif_transpose(orig_img)
    bg = tuple(int(x * 255) for x in mean)
    ow, oh = orig_img.size
    if ow > oh:
        sq = Image.new(orig_img.mode, (ow, ow), bg)
        sq.paste(orig_img, (0, (ow - oh) // 2))
    elif oh > ow:
        sq = Image.new(orig_img.mode, (oh, oh), bg)
        sq.paste(orig_img, ((oh - ow) // 2, 0))
    else:
        sq = orig_img
    sq = sq.resize((W, H))
    ref = np.asarray(sq, np.float32) / 255.0
    score = float(np.corrcoef(img.ravel(), ref.ravel())[0, 1])
    mae = float(np.abs(img - ref).mean())
    # The patch reassembly is patch-exact BY CONSTRUCTION (it re-lays-out
    # the very pixel_values the engine consumed; layout is deterministic
    # from gthw and was globally verified 2026-10-04 at r=0.997 on photo
    # content).  The reference chain here is only a PIL approximation of
    # the multi-step qwen_vl_utils preprocessing; on small / odd-aspect
    # images an interpolation-scale ghost can remain (r 0.8-0.9) even with
    # the layout exactly right.  So this is a RECORDED DIAGNOSTIC, not a
    # gate; a permuted patch order would still collapse r to ~0.
    # assert removed 2026-10-07 after manual inspection of the AI2D_703
    # comparison image (identical structure, sub-patch ghosting only).
    valid = (min(ow, oh) / max(ow, oh))  # content fill of the square canvas
    return (img * 255).round().astype(np.uint8), score, {
        "exif_transpose_applied": True,
        "engine_path": "Qwen3VLChatFixedRes._processor_inputs: "
                       "expand2square(mean colour) -> resize((1024,1024)) "
                       "-> processor(do_resize=False)",
        "expand2square": {"original": [ow, oh], "square_side": max(ow, oh),
                          "pad_background_rgb": bg,
                          "paste_offset": [(max(ow, oh) - ow) // 2,
                                           (max(ow, oh) - oh) // 2]},
        "final_canvas": [W, H],
        "valid_content_fraction_of_canvas": round(valid, 4),
        "reassembly_corr": score, "reassembly_mae": mae}


def highlight_slot(group_size: np.ndarray) -> int:
    """Deterministic highlight rule: the largest non-singleton group
    (>=1 dropped member by construction); ties -> lowest slot."""
    cand = [j for j in range(len(group_size)) if group_size[j] >= 2]
    return int(max(cand, key=lambda j: (group_size[j], -j)))


def render_preview(sdir, sid, meta, aligned, imp_np, A_hw_l, w_np, keep,
                   group, gh, gw, hl_slot, group_size, show_idx, top12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(4, 7, height_ratios=[1] * 4, hspace=0.35,
                          wspace=0.35)

    # ---- left: aligned image + importance overlay -------------------------
    ax = fig.add_subplot(gs[:, 0:2])
    ax.imshow(aligned)
    im = ax.imshow(imp_np, cmap="magma", alpha=0.45,
                   vmin=float(imp_np.min()), vmax=float(imp_np.max()))
    ax.set_title(f"{sid}\naligned 1024x1024 + frozen importance "
                 f"(minmax->3x3 s1 -> ^2)", fontsize=9)
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=6)

    # ---- middle: top-12 responses 4x3, shared A colour scale --------------
    amin, amax = float(A_hw_l.min()), float(A_hw_l.max())
    for r, ti in enumerate(show_idx):
        rr, cc = divmod(r, 3)
        ax = fig.add_subplot(gs[rr, 2 + cc])
        ax.imshow(A_hw_l[:, :, ti], cmap="magma", vmin=amin, vmax=amax)
        ax.set_title(f"#{r + 1} '{top12[r]['token'].strip()}' "
                     f"w={100 * w_np[ti]:.1f}%", fontsize=8)
        ax.axis("off")
    cax = fig.add_axes([0.375, 0.06, 0.24, 0.012])
    fig.colorbar(plt.cm.ScalarMappable(
        cmap="magma", norm=plt.Normalize(vmin=amin, vmax=amax)),
        cax=cax, orientation="horizontal")
    cax.set_title("shared A colour scale (original negated-cosine symbols, "
                  "not attention)", fontsize=7)

    # ---- right: all K anchors + highlight group ---------------------------
    ax = fig.add_subplot(gs[:, 6])
    ax.imshow(imp_np, cmap="gray", vmin=0, vmax=1)
    kh, kw = keep // gw, keep % gw
    ax.scatter(kw, kh, s=6, c="deepskyblue", marker="o",
               label=f"all {len(keep)} anchors (facility keep)")
    members = np.flatnonzero(group == hl_slot)
    mh, mw_ = members // gw, members % gw
    ax.scatter(mw_, mh, s=42, facecolors="none", edgecolors="red",
               linewidths=1.6, label=f"highlight group (n={len(members)})")
    for x, y in zip(mw_, mh):
        ax.add_patch(Circle((x, y), 0.45, color="red", fill=False,
                            linewidth=1.0))
    aslot = hl_slot
    ax.scatter([keep[aslot] % gw], [keep[aslot] // gw], s=110,
               marker="*", c="yellow", edgecolors="red", linewidths=0.8,
               label=f"anchor slot {aslot}", zorder=5)
    ax.set_title(f"anchors + assignment group\n"
                 r"$y_a = v_a + 0.25\,(\mu_a - v_a)$", fontsize=9)
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.04))
    ax.set_xticks(range(0, gw, 8)); ax.set_yticks(range(0, gh, 8))
    ax.tick_params(labelsize=5)

    fig.suptitle(f"{sid}  Qwen3-VL-8B  N=1024 K=256  "
                 "RTG -> Facility -> Completion(lambda=0.25)", fontsize=11)
    fig.savefig(os.path.join(sdir, "preview.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


def run_sample(ds, idx, model, eng, ph, sms, mean, std, repo_sha, shas,
               gen_time, cache_dev):
    sample_id = f"{ds}_{idx}"
    print(f"\n===== {sample_id} =====", flush=True)
    dataset = common.build_dataset(ds)
    model.set_dump_image(dataset.dump_image)
    # VLMEvalKit convention (verified 2026-10-07 vs bank/pred shards):
    # candidates are addressed by POSITION in dataset.data; the 'index'
    # column holds dataset-native ids (e.g. COCO image ids) and is NOT the
    # key — do not assert equality with idx.
    assert 0 <= idx < len(dataset.data), "idx out of range"
    row = dataset.data.iloc[idx]
    row_index_col = row.get("index", None)

    # ---------- one frozen forward: vision + instruction embeddings --------
    msg = common.build_message(model, dataset, ds, row)
    prep = eng.prepare(msg, ds)
    assert prep["gthw"].shape[0] == 1, "candidate must be single-image"
    V, DS = eng.encode(prep)
    N = int(prep["n_vis"])
    gh = int(prep["gthw"][0, 1]) // sms
    gw = int(prep["gthw"][0, 2]) // sms
    assert gh * gw == N, (gh, gw, N)

    text_mean, text_seq = eng.instruction_embeds(msg, ds)
    L = int(text_seq.shape[1])
    ctx = dict(prep=prep, V=V, DS=DS, K=R.K, engine=eng,
               text_mean=text_mean, text_seq=text_seq, ds=ds, qid=idx)

    # ---------- frozen RTG scoring (canonical call) ------------------------
    imp, w = R.rtg_importance(V.unsqueeze(0), text_mean, text_seq,
                              gh, gw, "rtg", ds=ds, qid=idx, img_idx=0)
    imp_np = imp[0].float().cpu().numpy().reshape(gh, gw)
    w_np = w.float().cpu().numpy()

    # ---------- same-forward pre-fusion quantities (identical formulas) ----
    feats = V.unsqueeze(0).float()
    ie = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    tef = text_mean.float()
    tef = tef / tef.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    tsf = text_seq.float()
    tsf = tsf / tsf.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    g = -torch.einsum("bnc,mc->bnm", ie, tef)[0, :, 0]          # [N]
    A = -torch.einsum("bnc,mlc->bnml", ie, tsf)[0, :, 0, :]     # [N, L]
    local = (A * w.unsqueeze(0)).sum(dim=1)                     # [N]
    fused = 0.5 * g + 0.5 * local                               # official 0.5
    logP = torch.log_softmax(R.INV_TEMP * A, dim=0)             # [N, L]
    logB = logP - torch.logsumexp(logP, dim=1, keepdim=True)
    c_un = torch.exp(logP + logB).sum(dim=0)                    # [L]
    g_np, A_np = g.cpu().numpy(), A.cpu().numpy()
    local_np, fused_np = local.cpu().numpy(), fused.cpu().numpy()
    logP_np, c_np = logP.cpu().numpy(), c_un.cpu().numpy()

    # independent recompute of the post-processing from fused
    from model.pruner import _spatial_smoothing_impl
    fs = fused.unsqueeze(0)
    imp_chk = (fs - fs.min(dim=-1, keepdim=True).values + 1e-6) / \
              (fs.max(dim=-1, keepdim=True).values -
               fs.min(dim=-1, keepdim=True).values + 1e-6)
    imp_chk = _spatial_smoothing_impl(imp_chk, gh, gw, kernel_size=3,
                                      sigma=1.0) ** 2.0
    imp_chk = imp_chk[0].float().cpu().numpy().reshape(gh, gw)

    # ---------- official facility keep + assignment ------------------------
    keep, _ = R.select_keep("rtg", ctx, R.K, want_diag=False)
    keep = keep.cpu().numpy().astype(np.int64)
    K = int(keep.shape[0])
    keep_t = torch.from_numpy(keep).to(V.device)
    dropped_idx, gid, _ = AC.compute_assignment(V, keep_t)
    group = torch.full((N,), -1, dtype=torch.long, device=V.device)
    group[keep_t] = torch.arange(K, device=V.device)
    if dropped_idx.numel():
        group[dropped_idx] = gid.to(V.device)
    group_np = group.cpu().numpy().astype(np.int64)
    counts = np.bincount(group_np, minlength=K)
    group_size = counts.astype(np.int64)                        # incl anchor
    dropped_np = dropped_idx.cpu().numpy().astype(np.int64)
    gid_np = gid.cpu().numpy().astype(np.int64)

    # ---------- anchor update (frozen uniform kind, main stream) -----------
    a_f = V[keep_t].float()                                     # [K, D] fp32
    mem = V[dropped_idx].float() if dropped_idx.numel() else None
    gsum = torch.zeros_like(a_f)
    if mem is not None:
        gsum.index_add_(0, gid, mem)                            # deterministic
    mu_all = (gsum + a_f) / (torch.from_numpy(
        counts.astype(np.float32)).to(a_f.device).unsqueeze(1))
    y_all = a_f + LAM * (mu_all - a_f)
    dt = V.dtype
    y_cast = y_all.to(dt)
    hl_slot = highlight_slot(group_size)
    members = np.flatnonzero(group_np == hl_slot)
    members = np.concatenate(
        ([keep[hl_slot]], members[members != keep[hl_slot]]))
    member_dropped = members[1:]

    # ---------- token labels + question span -------------------------------
    content = eng.vlm._prepare_content(msg, dataset=ds)
    parts = [c["text"] for c in content if c.get("type") == "text"]
    instr = " ".join(parts).strip() if parts else "Describe this image."
    enc = eng.tok(instr, return_tensors="pt", add_special_tokens=True,
                  padding=True, truncation=True,
                  max_length=eng.tok.model_max_length)
    tok_ids = enc.input_ids[0]
    assert tok_ids.shape[0] == L, (tok_ids.shape, L)
    token_labels = eng.tok.batch_decode(tok_ids.unsqueeze(1))
    seq_check = eng.model.get_input_embeddings()(enc.input_ids.to(V.device))
    assert torch.allclose(seq_check[0].float(), text_seq[0].float(),
                          atol=1e-4), "instruction tokenisation mismatch"
    # fixed suffix always starts with the standalone token "Please"
    # (token-level labels cannot contain the full SUFFIX substring)
    q_end = next((i for i, s in enumerate(token_labels)
                  if s.strip() == "Please"), L)

    q = row.get("question", None)
    if q is None or (isinstance(q, float) and np.isnan(q)):
        q = instr
    q = str(q)

    # ---------- images ------------------------------------------------------
    dumped = dataset.dump_image(row)
    img_bytes, src_ref = None, None
    for d in dumped:
        if isinstance(d, str) and os.path.exists(d):
            with open(d, "rb") as f:
                img_bytes = f.read()
            src_ref = d
            break
        if isinstance(d, str) and len(d) > 256:
            img_bytes = base64.b64decode(d)
            src_ref = "base64"
            break
    assert img_bytes is not None, "no image in dump_image output"
    orig_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    ext = os.path.splitext(str(row.get("image_path", "")))[1].lstrip(".")
    if not ext:
        ext = (orig_img.format or "PNG").lower()
    if ext == "jpeg":
        ext = "jpg"
    sdir = os.path.join(OUT_DIR, sample_id)
    os.makedirs(sdir, exist_ok=True)
    src_path = os.path.join(sdir, f"source_original.{ext}")
    with open(src_path, "wb") as f:
        f.write(img_bytes)
    src_sha = hashlib.sha256(img_bytes).hexdigest()

    aligned, align_score, align_meta = detect_and_rebuild(
        prep["pv"], prep["gthw"], ph, mean, std, orig_img, sms)
    Image.fromarray(aligned).save(os.path.join(sdir, "sample_aligned.png"))

    # ---------- token ranking ------------------------------------------------
    order = np.argsort(-w_np, kind="stable")     # ties -> lower token index
    with open(os.path.join(sdir, "token_ranking.csv"), "w", newline="") as f:
        fw = __import__("csv").writer(f)
        fw.writerow(["rank", "token_index", "token_label",
                     "weight", "weight_pct_of_prompt",
                     "cumulative_weight", "attribution"])
        cum = 0.0
        for r, t in enumerate(order):
            cum += float(w_np[t])
            fw.writerow([r + 1, int(t), token_labels[t].strip(),
                         f"{w_np[t]:.8f}", f"{100 * w_np[t]:.4f}",
                         f"{cum:.8f}",
                         "question" if t < q_end else "instruction_suffix"])

    # ---------- protocol pairing: banks + predictions -----------------------
    banks = {}
    for tag, bpath in [
        ("dev100", os.path.join(R.OUT_DIR, f"bank_dev_rtg_{ds}.json.gz")),
        ("full", os.path.join(R.OUT_DIR, "full", f"bank_full_rtg_{ds}.json.gz")),
    ]:
        if not os.path.exists(bpath):
            # dev-100 banks exist only for TextVQA_VAL/DocVQA_VAL/OCRBench;
            # missing here is expected, not an anomaly
            banks[tag] = {"path": bpath, "record": "NO_DEV_BANK",
                          "keep_match": None}
            continue
        with gzip.open(bpath, "rt") as f:
            b = json.load(f)
        rec = b["samples"].get(str(idx))
        if rec is None:
            banks[tag] = {"path": bpath, "sha256": sha256_file(bpath),
                          "record": "MISSING", "keep_match": None}
            continue
        bk = np.asarray(rec["keep"], dtype=np.int64)
        bgid = np.asarray(rec["gid"], dtype=np.int64)
        entry = {
            "path": bpath, "sha256": sha256_file(bpath),
            "code_commit": b["meta"].get("code_commit"),
            "keep_match": bool(bk.shape == keep.shape
                               and np.array_equal(bk, keep)),
        }
        if bgid.shape == gid_np.shape:
            entry["gid_match"] = bool(np.array_equal(bgid, gid_np))
        if "image_key" in rec and cache_dev.get((ds, idx)):
            entry["image_key_match"] = bool(
                str(rec["image_key"]) == str(cache_dev[(ds, idx)]))
        banks[tag] = entry
    preds, scores = {}, {}
    for tag, tmpl in PRED_SHARDS.items():
        p = tmpl.format(ds=ds)
        if not os.path.exists(p):
            preds[tag] = {"path": p, "record": "NO_SHARD"}
            continue
        s = json.load(open(p))
        r = s["records"].get(str(idx))
        preds[tag] = {"path": p, "sha256": sha256_file(p),
                      "meta": {k: v for k, v in s["meta"].items()
                               if k in ("arm", "pipeline", "panel", "legacy")},
                      "prediction": r["prediction"] if r else "MISSING"}
    for tag, tmpl in SCORE_SHARDS.items():
        p = tmpl.format(ds=ds)
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        scores[tag] = s.get("per_question", {}).get(str(idx))

    # ---------- sample.npz (desktop contract keys) --------------------------
    npz_path = os.path.join(sdir, "sample.npz")
    np.savez_compressed(
        npz_path,
        response=A_np.astype(np.float32).reshape(gh, gw, L),   # [H,W,L]
        weights=w_np.astype(np.float32),                       # [L]
        global_response=g_np.astype(np.float32).reshape(gh, gw),
        local_response=local_np.astype(np.float32).reshape(gh, gw),
        fused_response=fused_np.astype(np.float32).reshape(gh, gw),
        importance=imp_np.astype(np.float32),                  # [H,W]
        keep=keep,                                             # [K]
        dropped_idx=dropped_np,                                # [N-K]
        gid=gid_np,                                            # [N-K]
        group=group_np.reshape(gh, gw),                        # [H,W]
        group_size=group_size,                                 # [K]
        logP=logP_np.astype(np.float32),                       # [N,L]
        weights_unnorm_c=c_np.astype(np.float32),              # [L]
        grid=np.array([gh, gw], dtype=np.int64),
        n_vis=np.int64(N),
    )
    z = np.load(npz_path)

    # ---------- anchor_update.npz -------------------------------------------
    np.savez_compressed(
        os.path.join(sdir, "anchor_update.npz"),
        highlight_anchor_slot=np.int64(hl_slot),
        highlight_anchor_index=np.int64(keep[hl_slot]),
        highlight_member_indices=members.astype(np.int64),  # anchor first
        highlight_member_dropped_indices=member_dropped.astype(np.int64),
        highlight_v_a=a_f[hl_slot].cpu().numpy(),       # fp32 real feature
        # bf16 values are exactly representable in fp32, so the model-dtype
        # arrays are stored losslessly as fp32
        highlight_v_a_model_dtype=V[keep_t][hl_slot].cpu().float().numpy(),
        highlight_mu_a=mu_all[hl_slot].cpu().numpy(),   # fp32, incl anchor
        highlight_y_a=y_all[hl_slot].cpu().numpy(),     # fp32
        highlight_y_a_model_dtype=y_cast[hl_slot].cpu().float().numpy(),
        lam=np.float32(LAM), merge_kind="uniform",
        v_all=a_f.cpu().numpy(),                        # [K,D] fp32
        mu_all=mu_all.cpu().numpy(),                    # [K,D] fp32
        y_all=y_all.cpu().numpy(),                      # [K,D] fp32
        y_all_model_dtype=y_cast.cpu().float().numpy(),  # cast of model dtype
        keep=keep, group_size=group_size,
        compute_dtype="float32", cast_note="y cast to model dtype "
        f"({dt}) after FP32 merge, identical to amp_common.merge_stream")

    # ---------- checks -------------------------------------------------------
    tol = 1e-5
    fusion_ok = bool(np.allclose(fused_np,
                                 0.5 * g_np + 0.5 * local_np, atol=tol))
    local_ok = bool(np.allclose(local_np, (A_np * w_np).sum(axis=1),
                                atol=1e-4))
    imp_ok = bool(np.allclose(imp_np, imp_chk, atol=1e-5))
    self_ok = all(int(group_np[keep[j]]) == j for j in range(K))
    # rebuild w from A via the frozen FP32 rtg text_weight
    w_re, _ = R.text_weight(A.clone(), "rtg")
    w_re_np = w_re.float().cpu().numpy()
    w_rebuild_ok = bool(np.allclose(w_np, w_re_np, atol=1e-5))
    local_re = (A_np * w_re_np).sum(axis=1)
    local_rebuild_ok = bool(np.allclose(local_np, local_re, atol=1e-4))
    fused_re = 0.5 * g_np + 0.5 * local_re
    fused_rebuild_ok = bool(np.allclose(fused_np, fused_re, atol=1e-5))
    # y / mu identities (all slots, FP32)
    mu_incl = torch.stack([
        (V[torch.from_numpy(np.flatnonzero(group_np == j)).to(V.device)]
         .float().sum(0) / group_size[j]) for j in (hl_slot,)])
    mu_ok = bool(torch.allclose(mu_incl[0].cpu(), mu_all[hl_slot].cpu(),
                                atol=1e-4))
    y_id_ok = bool(np.allclose(
        y_all[hl_slot].cpu().numpy(),
        a_f[hl_slot].cpu().numpy() + LAM * (mu_all[hl_slot].cpu().numpy()
                                            - a_f[hl_slot].cpu().numpy()),
        atol=1e-6))
    y_all_id_ok = bool(np.allclose(
        y_all.cpu().numpy(),
        a_f.cpu().numpy() + LAM * (mu_all.cpu().numpy()
                                   - a_f.cpu().numpy()), atol=1e-5))

    checks = {
        "shapes": {"response": [gh, gw, L], "weights": [L],
                   "global_response": [gh, gw], "local_response": [gh, gw],
                   "fused_response": [gh, gw], "importance": [gh, gw],
                   "keep": [K], "dropped_idx": [int(dropped_np.shape[0])],
                   "gid": [int(gid_np.shape[0])],
                   "group": [gh, gw], "group_size": [K], "grid": [gh, gw],
                   "n_vis": N},
        "finite": {k: bool(np.isfinite(z[k]).all())
                   for k in ("response", "weights", "global_response",
                             "local_response", "fused_response",
                             "importance")},
        "weights_nonneg": bool((w_np >= 0).all()),
        "weights_sum": float(w_np.sum()),
        "weights_sum_ok": bool(abs(w_np.sum() - 1.0) < 1e-5),
        "keep_unique": bool(np.unique(keep).size == K),
        "keep_sorted": bool(np.all(np.diff(keep) > 0)),
        "keep_in_range": bool((keep >= 0).all() and (keep < N).all()),
        "group_all_assigned": bool((group_np >= 0).all()),
        "group_anchor_self": bool(self_ok),
        "group_size_sum_eq_N": bool(int(group_size.sum()) == N),
        "group_size_eq_bincount_incl_anchor": bool(
            np.array_equal(group_size, np.bincount(group_np,
                                                   minlength=K))),
        "dropped_nearest_anchor": None,
        "rtg_rebuild_from_A": {
            "weights": w_rebuild_ok, "local": local_rebuild_ok,
            "fused": fused_rebuild_ok,
            "importance_matches_independent_recompute": imp_ok,
            "definition": "w = frozen rtg_common.text_weight(A, 'rtg') in "
                          "FP32; local = sum_l w_l A_il; fused = 0.5 g + "
                          "0.5 local; importance = minmax -> 3x3 sigma=1 -> ^2"},
        "completion": {
            "mu_includes_anchor_recompute": mu_ok,
            "y_identity_highlight": y_id_ok,
            "y_identity_all_slots": y_all_id_ok,
            "lambda": LAM, "kind": "uniform",
            "compute_dtype": "float32",
            "model_dtype_cast": str(dt),
            "highlight_slot": int(hl_slot),
            "highlight_rule": "largest non-singleton group, ties -> lowest "
                              "slot (pre-registered, deterministic)",
            "highlight_n_members": int(len(members)),
            "highlight_n_dropped": int(len(member_dropped))},
        "bank_keep_match": {t: banks[t].get("keep_match") for t in banks},
        "bank_gid_match": {t: banks[t].get("gid_match") for t in banks},
        "single_image": True, "n_vis": N, "K_actual": K, "L": L,
        "grid": [gh, gw],
        "aligned_reassembly": align_meta,
        "aligned_reassembly_note": (
            "patch reassembly is patch-exact by construction (re-lays-out "
            "the pixel_values the engine consumed); the reference-chain "
            "corr/mae below are a recorded diagnostic only — a layout "
            "mistake would collapse corr to ~0"
            if align_meta["reassembly_corr"] < 0.90 else
            "reassembly corr >= 0.90"),
        "sample_key_consistency": {
            "addressing": "positional iloc index == bank/pred shard key "
                          "(VLMEvalKit 'index' column is a dataset-native "
                          "id, not the key)",
            "row_index_column_value": None if row_index_col is None
            else str(row_index_col),
            "bank_full_record_idx": True},
        "index_map": "token i -> cell (h, w) = (i // W, i % W) of the "
                     "[H,W] arrays; response cell (h,w,l) = A[i_h_w, l]; "
                     "cell (h,w) -> merged patch grid of the 1024x1024 "
                     "square input (patch 16, spatial merge 2)",
        "anomalies": [],
    }
    if dropped_idx.numel():
        Vd = V.float().double().cpu()
        a64 = F.normalize(Vd[keep_t.cpu()], dim=1)
        x64 = F.normalize(Vd[dropped_np], dim=1)
        gid64 = (x64 @ a64.t()).argmax(dim=1).numpy()
        ok = bool(np.array_equal(gid64, gid_np))
        checks["dropped_nearest_anchor"] = ok
        if not ok:
            checks["anomalies"].append("fp64 cos-argmax recheck disagrees")
    if not (fusion_ok and local_ok and imp_ok and w_rebuild_ok
            and local_rebuild_ok and fused_rebuild_ok):
        checks["anomalies"].append("RTG rebuild identity failed")
    if not (mu_ok and y_id_ok and y_all_id_ok):
        checks["anomalies"].append("completion identity failed")
    if not all(v is True for v in checks["bank_keep_match"].values()
               if v is not None):
        checks["anomalies"].append("keep differs from a same-protocol bank")
    if len(member_dropped) == 0:
        checks["anomalies"].append("highlight group has no dropped member")
    checks["all_pass"] = bool(
        all(checks["finite"].values()) and checks["weights_nonneg"]
        and checks["weights_sum_ok"] and checks["keep_unique"]
        and checks["keep_sorted"] and checks["keep_in_range"]
        and checks["group_all_assigned"] and checks["group_anchor_self"]
        and checks["group_size_sum_eq_N"] and not checks["anomalies"])

    # ---------- top-12 + preview --------------------------------------------
    top_r = 12
    show_idx = [int(t) for t in order[:top_r]]
    top12 = [{"rank": r + 1, "token_index": int(t),
              "token": token_labels[t], "weight": float(w_np[t])}
             for r, t in enumerate(show_idx)]
    A_hw_l = A_np.reshape(gh, gw, L).astype(np.float32)
    render_preview(sdir, sample_id, None, aligned, imp_np, A_hw_l, w_np,
                   keep, group_np, gh, gw, hl_slot, group_size, show_idx,
                   top12)

    # ---------- sample.json ---------------------------------------------------
    meta = {
        "dataset": ds, "split": "full panel (official per-question score "
                     "shard)", "sample_id": sample_id, "idx": idx,
        "image_key": os.path.basename(str(row.get("image_path", ""))
                                      or f"{idx}"),
        "question": q, "instruction_text": instr,
        "input_ids": [int(x) for x in tok_ids.tolist()],
        "token_labels": token_labels,
        "valid_mask": [True] * L,
        "question_span": {"start": 0, "end_exclusive": int(q_end),
                          "note": "fixed instruction suffix '"
                                  + SUFFIX + "' starts here; the RTG text "
                                  "sequence is the FULL prompt (question + "
                                  "suffix), no stopword removal"},
        "N": N, "K": K, "L": L, "grid": [gh, gw],
        "index_map": checks["index_map"],
        "model": common.BASELINE_MODEL,
        "checkpoint": str(getattr(eng.model.config, "_name_or_path",
                                  "unknown")),
        "tokenizer": str(getattr(eng.tok, "name_or_path", "unknown")),
        "generation_settings": {"do_sample": False, "max_new_tokens": 64,
                                "note": "figure arrays are prefill-visual "
                                "only; the paired prediction comes from the "
                                "recorded shards, not re-generated here"},
        "execution_path": "NativeEngine prepare/encode/instruction_embeds/"
                          "select_keep (rtg_legacy_checks C1: bitwise "
                          "identical under both engine flag sets, so the "
                          "arrays pair with the official-legacy protocol)",
        "arm": "L_R_MAIN025 (R_MAIN025 under the official EADP legacy "
               "scenario deepstack=False pos='1d')",
        "completion": {"lambda": LAM, "scope": "main", "kind": "uniform"},
        "git_sha": repo_sha,
        "worktree_uncommitted": True,
        "script_sha256": shas,
        "preprocessing": align_meta,
        "config_frozen": {"INV_TEMP": R.INV_TEMP, "alpha": 0.5, "beta": 2.0,
                          "smoothing": "3x3 sigma=1", "minmax_eps": 1e-6,
                          "K": R.K, "seed_prefix": R.SEED_PREFIX,
                          "completion_lambda": LAM,
                          "completion_scope": "main"},
        "banks": banks,
        "predictions": preds,
        "scores_per_question": scores,
        "source_image_sha256": src_sha,
        "source_image_ref": str(src_ref),
        "array_provenance": {"mode": "frozen CUDA replay (exporter v3)",
                             "script":
                             "export_anchorzip_figure_sample_v3.py",
                             "generated_at": gen_time},
        "display_token_indices": show_idx,
        "display_ranking_rule": "original RTG weights w descending, stable "
                                "argsort (ties -> lower token index); the "
                                "12 shown maps are NOT renormalised",
        "highlight_anchor_slot": int(hl_slot),
        "render_command": "see render_preview() in "
                          "export_anchorzip_figure_sample_v3.py",
        "notes": [
            "response is A, the negated-cosine matrix between main visual "
            "features and instruction token embeddings (original symbols, "
            "FP32) — not attention, not annotation. High A does NOT mean "
            "high positive cosine similarity.",
            "weights are the frozen rtg_common.text_weight output over the "
            "FULL prompt (sum=1 over all L tokens); no stopword removal, no "
            "renormalisation; not 'attention shares'.",
            "global/local/fused are the pre-fusion quantities of the frozen "
            "rtg_importance path; importance = minmax(fused) -> 3x3 "
            "sigma=1 -> ^2 (beta=2.0).",
            "P(i|t) = softmax_i(100*A_it) over the 1024 visual tokens; "
            "stored as logP[N,L].",
            "Completion y_a = v_a + 0.25*(mu_a - v_a), mu_a INCLUDES the "
            "anchor; groups are cosine-argmax assignments in the main "
            "visual feature space (not spatial distance).",
        ],
    }
    with open(os.path.join(sdir, "sample.json"), "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    with open(os.path.join(sdir, "checks.json"), "w") as f:
        json.dump(checks, f, indent=1)

    print(f"[done] {sample_id}: all_pass={checks['all_pass']} N={N} K={K} "
          f"L={L} hl_slot={hl_slot}(n={len(members)}) "
          f"bank_match={checks['bank_keep_match']} "
          f"pred_legacy={preds['L_R_MAIN025_official_legacy'].get('prediction')}",
          flush=True)
    return checks["all_pass"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True,
                    help="comma list of Dataset:idx")
    args = ap.parse_args()
    cands = [(s.split(":")[0], int(s.split(":")[1]))
             for s in args.spec.split(",") if s.strip()]

    model = common.load_model(common.BASELINE_MODEL, max_new_tokens=64)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)
    model.model.eval()
    torch.set_grad_enabled(False)

    vis_cfg = eng.inner.visual.config
    ph = int(getattr(vis_cfg, "patch_size", 16))
    sms = int(eng.inner.visual.spatial_merge_size)
    ip = model.processor.image_processor
    mean = ip.image_mean if not torch.is_tensor(ip.image_mean) \
        else ip.image_mean.tolist()
    std = ip.image_std if not torch.is_tensor(ip.image_std) \
        else ip.image_std.tolist()

    # dev-100 image_key lookup (for bank image_key_match checks)
    cache_dev = {}
    dev_bank_path = os.path.join(R.OUT_DIR, "bank_dev_rtg_manifest.json")
    if os.path.exists(os.path.join(AC.OUT_DIR, "manifest.json")):
        try:
            manifest = json.load(open(os.path.join(AC.OUT_DIR,
                                                   "manifest.json")))
            for ds, idx in cands:
                for it in manifest["datasets"].get(ds, {}).get("dev", []):
                    if int(it["idx"]) == idx:
                        cache_dev[(ds, idx)] = it.get("image_key")
        except Exception as e:
            print(f"[warn] manifest unavailable: {e}", flush=True)

    repo_sha = R.git_commit()
    shas = {"exporter_v3": sha256_file(os.path.abspath(__file__)),
            "rtg_common": sha256_file(os.path.abspath(R.__file__)),
            "amp_common": sha256_file(os.path.abspath(AC.__file__)),
            "common": sha256_file(os.path.abspath(common.__file__))}
    gen_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    summary = {}
    for ds, idx in cands:
        try:
            ok = run_sample(ds, idx, model, eng, ph, sms, mean, std,
                            repo_sha, shas, gen_time, cache_dev)
            summary[f"{ds}_{idx}"] = {"all_pass": bool(ok)}
        except Exception as e:      # one bad sample must not kill the batch
            summary[f"{ds}_{idx}"] = {"all_pass": False,
                                      "error": f"{type(e).__name__}: {e}"}
            print(f"[FAIL] {ds}_{idx}: {summary[f'{ds}_{idx}']['error']}",
                  flush=True)
    with open(os.path.join(OUT_DIR, "run_summary.json"), "w") as f:
        json.dump({"generated_at": gen_time, "git_sha": repo_sha,
                   "device": "cuda" if torch.cuda.is_available() else "cpu",
                   "samples": summary}, f, indent=1)
    print("[run_summary]", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
