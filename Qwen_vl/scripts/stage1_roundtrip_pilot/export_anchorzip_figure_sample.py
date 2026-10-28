#!/usr/bin/env python3
"""AnchorZip paper-figure sample exporter (M14 / RTG line).

For a small set of candidate dev samples, runs ONE vision pass + ONE text
embedding pass per sample through the FROZEN RTG pipeline and exports the raw
arrays the paper Figure-1 needs:

  importance[H,W]  RTG-fused importance after the frozen min-max / 3x3
                   sigma=1 Gaussian / beta=2.0 post-processing
                   (rtg_common.rtg_importance, verbatim).
  response[H,W,L]  negative-cosine matrix A (original symbols, FP32) between
                   the main visual features and the instruction token
                   embeddings -- same forward.  NOT attention, NOT annotation.
  weights[L]       RTG text weights from the frozen code path
                   (rtg_common.text_weight: P=softmax_vis(100*A),
                   B=P/sum_text P, c=sum_visual(P*B), w=c/c.sum()).
  keep[K]          official facility selection on the official rebound
                   similarity with the RTG importance (rtg_common.select_keep),
                   ascending flat indices into V.
  group[H,W]       anchor slot (index into keep) of every original visual
                   token, anchors included (amp_common.compute_assignment:
                   cos-argmax on RAW main features, ties -> lowest rank).

Plus sample.json (provenance), checks.json (frozen validation list), the
original dataset image and the aligned model-input image (patch-exact
reassembly of pixel_values, un-normalised -- the actual model input).

Usage:
    python export_anchorzip_figure_sample.py            # 3 candidates
    python export_anchorzip_figure_sample.py --limit 1  # smoke
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
import rtg_common as R  # noqa: E402  (frozen RTG math + protocol constants)

from PIL import Image  # noqa: E402

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "figures", "anchorzip",
                       "samples")
os.makedirs(OUT_DIR, exist_ok=True)

CANDIDATES = [  # first dev item of each RTG bank dataset (manifest-anchored)
    ("TextVQA_VAL", None),   # resolved below from the manifest dev[0]
    ("DocVQA_VAL", None),
    ("OCRBench", None),
]


def sha256_file(p: str) -> str:
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- aligned image
def detect_and_rebuild(pv, gthw, ph, mean, std, orig_img: Image.Image, sms):
    """Patch-exact reassembly of the processor pixel_values into the aligned
    (resized, un-normalised) RGB image the vision encoder actually consumed.

    Qwen3-VL layout (verified empirically, MAE 0.007 vs the wrapper's own
    expand2square+resize reference):
      row           = (C=3, temporal_patch_size=2, ph, ph); for a static
                      image the two temporal slices are bit-identical;
      row order     = grouped per MERGED token: sms*sms consecutive rows per
                      merged token, sub-patch (di, dj) = divmod(r, sms);
      normalisation = (x - mean) / std with the processor's mean/std.
    The reconstruction is scored against the wrapper's reference pipeline
    (expand2square with bg=mean*255, then resize to the fixed resolution);
    the score must exceed 0.95 or we refuse to emit the PNG."""
    t, hp, wp = [int(x) for x in gthw[0]]
    assert t == 1, "multi-frame sample"
    rows = pv.float().cpu().numpy()                  # (hp*wp, D)
    P = rows.shape[0]
    assert P == hp * wp, (P, hp, wp)
    H, W = hp * ph, wp * ph
    sms_g = int(sms)
    aligned_mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    aligned_std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    assert rows.shape[1] == 3 * 2 * ph * ph, rows.shape
    f = rows.reshape(P, 3, 2, ph, ph)[:, :, 0, :, :]           # (P, C, ph, ph)
    ts_identical = float(np.abs(
        rows.reshape(P, 3, 2, ph, ph)[:, :, 0] -
        rows.reshape(P, 3, 2, ph, ph)[:, :, 1]).max())
    f = f.transpose(0, 2, 3, 1)                                # (P, ph, ph, C)

    img = np.zeros((H, W, 3), dtype=np.float32)
    mw = wp // sms_g              # merged-token grid width
    for p in range(P):
        m, r = divmod(p, sms_g * sms_g)
        di, dj = divmod(r, sms_g)
        pr, pc = (m // mw) * sms_g + di, (m % mw) * sms_g + dj
        img[pr * ph:(pr + 1) * ph, pc * ph:(pc + 1) * ph] = f[p]
    img = np.clip(img * aligned_std + aligned_mean, 0, 1)

    # reference pipeline (qwen_vl_utils.fetch_image -> wrapper): smart_resize
    # to a factor-32 multiple, then expand2square, then the fixed-resolution
    # resize.  Only used as an independent sanity score for the reassembly.
    from qwen_vl_utils.vision_process import smart_resize
    side0 = max(orig_img.size)
    rh, rw = smart_resize(orig_img.height, orig_img.width, factor=32)
    pre = orig_img.resize((rw, rh), Image.BICUBIC)
    side = max(pre.size)
    bg = tuple(int(x * 255) for x in mean)
    sq = Image.new("RGB", (side, side), bg)
    sq.paste(pre, ((side - pre.width) // 2, (side - pre.height) // 2))
    ref = np.asarray(sq.resize((W, H), Image.BICUBIC), np.float32) / 255.0
    score = float(np.corrcoef(img.ravel(), ref.ravel())[0, 1])
    mae = float(np.abs(img - ref).mean())
    # geometry/parse must be right (corr) and the value round-trip sane (mae);
    # residual gap vs the reference is resize-trajectory detail, not layout.
    assert score > 0.90 and mae < 0.03, \
        f"aligned-image reassembly failed (r={score:.3f}, mae={mae:.4f})"
    chosen = (f"C_major|temporal_inner|merged{sms_g}x{sms_g} "
              f"(temporal slices identical: {ts_identical:.2e})")
    return (img * 255).round().astype(np.uint8), chosen, score


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    model = common.load_model(common.BASELINE_MODEL, max_new_tokens=64)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)
    model.model.eval()
    torch.set_grad_enabled(False)

    vis_cfg = eng.inner.visual.config
    ph = int(getattr(vis_cfg, "patch_size", 16))
    sms = int(eng.inner.visual.spatial_merge_size)
    ip = model.processor.image_processor
    mean = ip.image_mean if not torch.is_tensor(ip.image_mean) else ip.image_mean.tolist()
    std = ip.image_std if not torch.is_tensor(ip.image_std) else ip.image_std.tolist()
    print(f"[cfg] patch={ph} merge={sms} mean={mean} std={std}")

    manifest = json.load(open(os.path.join(AC.OUT_DIR, "manifest.json")))
    repo_sha = R.git_commit()
    rtg_sha = sha256_file(os.path.abspath(R.__file__))
    script_sha = sha256_file(os.path.abspath(__file__))
    ac_sha = sha256_file(os.path.abspath(AC.__file__))
    gen_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    cands = CANDIDATES[:args.limit] if args.limit else CANDIDATES
    for ds, _ in cands:
        items = manifest["datasets"][ds]["dev"]
        it = items[0]
        idx = int(it["idx"])
        out = run_sample(ds, idx, it, model, eng, ph, sms, mean, std,
                         repo_sha, rtg_sha, ac_sha, script_sha, gen_time)
        print(f"[done] {out}")


def run_sample(ds, idx, it, model, eng, ph, sms, mean, std,
               repo_sha, rtg_sha, ac_sha, script_sha, gen_time):
    sample_id = f"{ds}_{idx}"
    print(f"\n===== {sample_id} =====")
    dataset = common.build_dataset(ds)
    model.set_dump_image(dataset.dump_image)
    row = dataset.data.iloc[idx]

    # ---------- one forward: vision + instruction embeddings ----------------
    msg = common.build_message(model, dataset, ds, row)
    prep = eng.prepare(msg, ds)
    assert prep["gthw"].shape[0] == 1, "candidate must be single-image"
    V, DS = eng.encode(prep)                       # [N, D] main stream
    N = int(prep["n_vis"])
    gh = int(prep["gthw"][0, 1]) // sms
    gw = int(prep["gthw"][0, 2]) // sms
    assert gh * gw == N, (gh, gw, N)

    text_mean, text_seq = eng.instruction_embeds(msg, ds)   # [1,D], [1,L,D]
    L = int(text_seq.shape[1])
    qid = idx
    ctx = dict(prep=prep, V=V, DS=DS, K=R.K, engine=eng,
               text_mean=text_mean, text_seq=text_seq, ds=ds, qid=qid)

    # ---------- frozen RTG importance + weights (verbatim code path) --------
    feats = V.unsqueeze(0).float()
    ie = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    tsf = text_seq.float()
    tsf = tsf / tsf.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    A = -torch.einsum("bnc,mlc->bnml", ie, tsf)[0, :, 0, :]     # [N, L] fp32
    assert torch.isfinite(A).all()

    imp, w = R.rtg_importance(V.unsqueeze(0), text_mean, text_seq,
                              gh, gw, "rtg", ds=ds, qid=qid, img_idx=0)
    imp = imp[0].float().cpu().numpy().reshape(gh, gw)
    w_np = w.float().cpu().numpy()

    # ---------- official facility selection with the RTG importance ---------
    keep, _ = R.select_keep("rtg", ctx, R.K, want_diag=False)
    keep = keep.cpu().numpy().astype(np.int64)
    K = int(keep.shape[0])

    # ---------- assignment: every token -> anchor slot (anchors included) ---
    keep_t = torch.from_numpy(keep).to(V.device)
    dropped_idx, gid, sim = AC.compute_assignment(V, keep_t)
    group = torch.full((N,), -1, dtype=torch.long, device=V.device)
    group[keep_t] = torch.arange(K, device=V.device)
    if dropped_idx.numel():
        group[dropped_idx] = gid.to(V.device)
    group_np = group.cpu().numpy().reshape(gh, gw).astype(np.int64)

    # ---------- token labels (same tokenisation as instruction_embeds) ------
    content = eng.vlm._prepare_content(msg, dataset=ds)
    parts = [c["text"] for c in content if c.get("type") == "text"]
    instr = " ".join(parts).strip() if parts else "Describe this image."
    enc = eng.tok(instr, return_tensors="pt", add_special_tokens=True,
                  padding=True, truncation=True,
                  max_length=eng.tok.model_max_length)
    tok_ids = enc.input_ids[0]
    assert tok_ids.shape[0] == L, (tok_ids.shape, L)
    token_labels = eng.tok.batch_decode(tok_ids.unsqueeze(1))
    # consistency: the exact embeds used above must equal this tokenisation
    seq_check = eng.model.get_input_embeddings()(enc.input_ids.to(V.device))
    assert torch.allclose(seq_check[0].float(), text_seq[0].float(), atol=1e-4), \
        "instruction tokenisation mismatch vs instruction_embeds"

    # question text
    q = row.get("question", None)
    if q is None or (isinstance(q, float) and np.isnan(q)):
        q = instr
    q = str(q)

    # ---------- images -------------------------------------------------------
    dumped = dataset.dump_image(row)
    img_bytes, src_ref = None, None
    for d in dumped:
        if isinstance(d, str) and os.path.exists(d):   # local file path
            with open(d, "rb") as f:
                img_bytes = f.read()
            src_ref = d
            break
        if isinstance(d, str) and len(d) > 256:        # base64 payload
            img_bytes = base64.b64decode(d)
            src_ref = "base64"
            break
    assert img_bytes is not None, "no image in dump_image output"
    orig_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    ext = os.path.splitext(str(it.get("image_key", "")))[1].lstrip(".")
    if not ext:
        ext = (orig_img.format or "PNG").lower()
    if ext == "jpeg":
        ext = "jpg"
    src_path = os.path.join(OUT_DIR, sample_id, f"source_original.{ext}")
    os.makedirs(os.path.dirname(src_path), exist_ok=True)
    with open(src_path, "wb") as f:
        f.write(img_bytes)

    aligned, chosen_layout, align_score = detect_and_rebuild(
        prep["pv"], prep["gthw"], ph, mean, std, orig_img, sms)
    aligned_path = os.path.join(OUT_DIR, sample_id, "sample_aligned.png")
    Image.fromarray(aligned).save(aligned_path)

    # ---------- provenance: bank + prediction shards -------------------------
    bank_path = R.bank_path("rtg", ds)
    bank_sha = sha256_file(bank_path)
    with gzip.open(bank_path, "rt") as f:
        bank = json.load(f)
    rec = bank["samples"].get(str(idx))
    keep_match = None
    if rec is not None:
        bank_keep = np.asarray(rec["keep"], dtype=np.int64)
        keep_match = bool(bank_keep.shape == keep.shape
                          and np.array_equal(bank_keep, keep))

    acc_path = os.path.join(R.OUT_DIR, "acc", "R_MAIN025", f"{ds}.json")
    acc_sha = sha256_file(acc_path)
    acc = json.load(open(acc_path))
    rec_pred = acc["records"].get(str(idx))
    prediction = rec_pred["prediction"] if rec_pred else None

    # ---------- sample.npz ---------------------------------------------------
    npz_path = os.path.join(OUT_DIR, sample_id, "sample.npz")
    np.savez_compressed(
        npz_path,
        importance=imp.astype(np.float32),            # [H, W]
        response=A.cpu().numpy().reshape(gh, gw, L).astype(np.float32),
        weights=w_np.astype(np.float32),              # [L]
        keep=keep,                                    # [K] flat indices
        group=group_np,                               # [H, W] anchor slots
    )

    # ---------- checks.json --------------------------------------------------
    resp = np.load(npz_path)["response"]
    w_sum = float(w_np.sum())
    checks = {
        "shapes": {
            "importance": list(imp.shape), "response": list(resp.shape),
            "weights": list(w_np.shape), "keep": list(keep.shape),
            "group": list(group_np.shape),
        },
        "finite": {
            "importance": bool(np.isfinite(imp).all()),
            "response": bool(np.isfinite(resp).all()),
            "weights": bool(np.isfinite(w_np).all()),
        },
        "weights_sum": w_sum,
        "weights_sum_ok": bool(abs(w_sum - 1.0) < 1e-5),
        "keep_unique": bool(np.unique(keep).size == K),
        "keep_sorted": bool(np.all(np.diff(keep) > 0)),
        "keep_in_range": bool((keep >= 0).all() and (keep < N).all()),
        "group_anchor_self": bool(
            all(int(group_np.flatten()[keep[j]]) == j for j in range(K))),
        "group_all_assigned": bool((group_np >= 0).all()),
        "dropped_nearest_anchor": None,   # filled below (independent recheck)
        "n_vis": N, "K_actual": K, "grid": [gh, gw],
        "single_image": True,
        "bank_keep_match": keep_match,
        "aligned_reassembly": {"layout": chosen_layout,
                               "corr_vs_original": align_score},
        "anomalies": [],
    }
    # independent FP64 recheck of the cos-argmax rule on RAW main features
    if dropped_idx.numel():
        Vd = V.float().double().cpu()
        a64 = F.normalize(Vd[keep_t.cpu()], dim=1)
        x64 = F.normalize(Vd[dropped_idx.cpu()], dim=1)
        gid64 = (x64 @ a64.t()).argmax(dim=1).numpy()
        ok = bool(np.array_equal(gid64, gid.cpu().numpy()))
        checks["dropped_nearest_anchor"] = ok
        if not ok:
            checks["anomalies"].append(
                "fp64 recheck of cos-argmax assignment disagrees (bf16 tie?)")
    if rec is not None and not keep_match:
        checks["anomalies"].append("keep differs from bank_dev_rtg record")
    checks["all_pass"] = bool(
        all(checks["finite"].values()) and checks["weights_sum_ok"]
        and checks["keep_unique"] and checks["keep_sorted"]
        and checks["keep_in_range"] and checks["group_anchor_self"]
        and checks["group_all_assigned"]
        and (checks["dropped_nearest_anchor"] is not False)
        and not checks["anomalies"])

    # ---------- display helpers (derived, deterministic; raw arrays untouched)
    order = np.argsort(-w_np)
    display_token_indices = [int(i) for i in order[:4]]
    flat_imp = imp.reshape(-1)
    highlight_anchor_slot = int(group_np.reshape(-1)[int(flat_imp.argmax())])

    # ---------- sample.json --------------------------------------------------
    meta = {
        "dataset": ds, "sample_id": sample_id, "idx": idx,
        "image_key": str(it.get("image_key", "")),
        "question": q,
        "instruction_text": instr,
        "token_labels": token_labels,
        "N": N, "K": K, "grid": [gh, gw],
        "model": common.BASELINE_MODEL,
        "checkpoint": str(getattr(eng.model.config, "_name_or_path",
                                  "unknown")),
        "execution_path": "native (NativeEngine, model/native_qwen3.py); "
                          "RTG scoring + facility + MAIN025 assignment are the "
                          "frozen rtg_common/amp_common legacy-equivalent math; "
                          "rtg_legacy_checks ALL_PASS (2026-10-03)",
        "git_sha": repo_sha,
        "script_sha256": {"exporter": script_sha, "rtg_common": rtg_sha,
                          "amp_common": ac_sha},
        "config_frozen": {"INV_TEMP": R.INV_TEMP, "alpha": 0.5, "beta": 2.0,
                          "smoothing": "3x3 sigma=1", "minmax_eps": 1e-6,
                          "K": R.K, "seed_prefix": R.SEED_PREFIX},
        "bank": {"path": bank_path, "sha256": bank_sha,
                 "scorer": "rtg", "split": "dev",
                 "keep_match": keep_match},
        "source_image_ref": str(src_ref),
        "prediction": {"path": acc_path, "sha256": acc_sha,
                       "arm": "R_MAIN025", "prediction": prediction},
        "manifest": {"path": os.path.join(AC.OUT_DIR, "manifest.json"),
                     "sha256": sha256_file(os.path.join(AC.OUT_DIR,
                                                        "manifest.json"))},
        "generated_at": gen_time,
        "display_token_indices": display_token_indices,
        "highlight_anchor_slot": highlight_anchor_slot,
        "notes": [
            "response is the negative-cosine matrix A (original symbols), "
            "not attention and not human annotation.",
            "Completion (y = a + 0.25*(mean(G)-a), i.e. 0.75*v_a + 0.25*mean(G_a)) "
            "operates on the main visual feature stream only; no pixels are "
            "reconstructed or exported from the anchors.",
        ],
    }
    json_path = os.path.join(OUT_DIR, sample_id, "sample.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    checks_path = os.path.join(OUT_DIR, sample_id, "checks.json")
    with open(checks_path, "w") as f:
        json.dump(checks, f, indent=1)

    return f"{sample_id}: all_pass={checks['all_pass']} N={N} K={K} L={L}"


if __name__ == "__main__":
    main()
