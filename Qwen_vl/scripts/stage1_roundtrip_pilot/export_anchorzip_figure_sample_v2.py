#!/usr/bin/env python3
"""AnchorZip Figure-1 exporter v2 — extended arrays + protocol pairing.

Frozen path (identical to v1, one vision pass + one text-embedding pass per
sample through NativeEngine): RTG scoring (rtg_common), official facility
(instrumented.SELECTORS['facility'] on _sim_visual_impl), anchor assignment
(amp_common.compute_assignment).  No algorithm is changed.

v2 additions over v1:
  * sample.npz now carries the full pre-fusion quantities from the same
    forward: response=A[N,L] (original negated-cosine symbols), weights=w[L],
    global_response=g[N] (-cos vs text mean), local_response=(A*w).sum(1)[N],
    fused_response=0.5g+0.5local[N], post-processed importance[H,W]
    (min-max -> 3x3 sigma=1 -> beta=2 square), keep[K], group[N] (flat token
    index; cell (h,w) = (idx // gw, idx % gw)).
  * checks: fusion identity, local identity, independent importance
    recompute, keep match vs BOTH bank_dev_rtg (dev-100) and
    full/bank_full_rtg (full panel), prediction pairing across the three
    protocol chains.
  * protocol pairing in sample.json: the prediction cited for the figure is
    legacy_full/acc/L_R_MAIN025 (official EADP legacy scenario,
    deepstack=False pos='1d'); full/acc/R_MAIN025 and dev-100 acc/R_MAIN025
    are recorded as cross-protocol corroboration.  The visual arrays are
    prefill-independent (rtg_legacy_checks C1: bank records bitwise
    identical under both flag sets), so one array set pairs with all three.

Usage:
    python export_anchorzip_figure_sample_v2.py \
        --spec TextVQA_VAL:787,OCRBench:403[,TextVQA_VAL:1536,...]
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

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "figures", "anchorzip",
                       "samples_v2")
os.makedirs(OUT_DIR, exist_ok=True)

PRED_SHARDS = {   # protocol chain -> path template
    "L_R_MAIN025_official_legacy":
        os.path.join(R.OUT_DIR, "legacy_full", "acc", "L_R_MAIN025",
                     "{ds}.json"),
    "R_MAIN025_full_native":
        os.path.join(R.OUT_DIR, "full", "acc", "R_MAIN025", "{ds}.json"),
    "R_MAIN025_dev100_native":
        os.path.join(R.OUT_DIR, "acc", "R_MAIN025", "{ds}.json"),
}
SCORE_SHARDS = {
    "L_R_MAIN025_official_legacy":
        os.path.join(R.OUT_DIR, "legacy_full", "acc", "L_R_MAIN025",
                     "{ds}_score.json"),
    "R_MAIN025_full_native":
        os.path.join(R.OUT_DIR, "full", "acc", "R_MAIN025", "{ds}_score.json"),
    "R_MAIN025_dev100_native":
        os.path.join(R.OUT_DIR, "acc", "R_MAIN025", "{ds}_score.json"),
}


def sha256_file(p: str) -> str:
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def detect_and_rebuild(pv, gthw, ph, mean, std, orig_img: Image.Image, sms):
    """Patch-exact reassembly of pixel_values (layout verified 2026-10-04,
    corr 0.997 vs the wrapper reference chain).  Identical to v1."""
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
    assert score > 0.90 and mae < 0.03, \
        f"aligned-image reassembly failed (r={score:.3f}, mae={mae:.4f})"
    return (img * 255).round().astype(np.uint8), score


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

    manifest = json.load(open(os.path.join(AC.OUT_DIR, "manifest.json")))
    repo_sha = R.git_commit()
    shas = {"exporter_v2": sha256_file(os.path.abspath(__file__)),
            "rtg_common": sha256_file(os.path.abspath(R.__file__)),
            "amp_common": sha256_file(os.path.abspath(AC.__file__)),
            "common": sha256_file(os.path.abspath(common.__file__))}
    gen_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    summary = {}
    for ds, idx in cands:
        items = {int(it["idx"]): it for it in manifest["datasets"][ds]["dev"]}
        if idx not in items:
            summary[f"{ds}_{idx}"] = {"all_pass": False,
                                      "error": "not in manifest dev split"}
            continue
        try:
            ok = run_sample(ds, idx, items[idx], model, eng, ph, sms, mean,
                            std, repo_sha, shas, gen_time)
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


def run_sample(ds, idx, it, model, eng, ph, sms, mean, std,
               repo_sha, shas, gen_time):
    sample_id = f"{ds}_{idx}"
    print(f"\n===== {sample_id} =====", flush=True)
    dataset = common.build_dataset(ds)
    model.set_dump_image(dataset.dump_image)
    row = dataset.data.iloc[idx]

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
    g_np, A_np = g.cpu().numpy(), A.cpu().numpy()
    local_np, fused_np = local.cpu().numpy(), fused.cpu().numpy()

    # independent recomputation of the post-processing from fused
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
    suffix = "Please answer concisely"
    q_end = next((i for i, s in enumerate(token_labels)
                  if suffix in s), L)

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
    ext = os.path.splitext(str(it.get("image_key", "")))[1].lstrip(".")
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

    aligned, align_score = detect_and_rebuild(
        prep["pv"], prep["gthw"], ph, mean, std, orig_img, sms)
    Image.fromarray(aligned).save(os.path.join(sdir, "sample_aligned.png"))

    # ---------- protocol pairing: banks + predictions -----------------------
    banks = {}
    for tag, bpath in [
        ("dev100", os.path.join(R.OUT_DIR, f"bank_dev_rtg_{ds}.json.gz")),
        ("full", os.path.join(R.OUT_DIR, "full", f"bank_full_rtg_{ds}.json.gz")),
    ]:
        with gzip.open(bpath, "rt") as f:
            b = json.load(f)
        rec = b["samples"].get(str(idx))
        if rec is None:
            banks[tag] = {"path": bpath, "sha256": sha256_file(bpath),
                          "record": "MISSING", "keep_match": None}
            continue
        bk = np.asarray(rec["keep"], dtype=np.int64)
        banks[tag] = {
            "path": bpath, "sha256": sha256_file(bpath),
            "code_commit": b["meta"]["code_commit"],
            "keep_match": bool(bk.shape == keep.shape
                               and np.array_equal(bk, keep)),
        }
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

    # ---------- sample.npz --------------------------------------------------
    npz_path = os.path.join(sdir, "sample.npz")
    np.savez_compressed(
        npz_path,
        response=A_np.astype(np.float32),             # [N, L]
        weights=w_np.astype(np.float32),              # [L]
        global_response=g_np.astype(np.float32),      # [N]
        local_response=local_np.astype(np.float32),   # [N]
        fused_response=fused_np.astype(np.float32),   # [N]
        importance=imp_np.astype(np.float32),         # [H, W]
        keep=keep,                                    # [K] flat token idx
        group=group_np,                               # [N] anchor slot
        grid=np.array([gh, gw], dtype=np.int64),
        n_vis=np.int64(N),
    )
    z = np.load(npz_path)

    # ---------- checks ------------------------------------------------------
    tol = 1e-5
    fusion_ok = bool(np.allclose(fused_np,
                                 0.5 * g_np + 0.5 * local_np, atol=tol))
    local_ok = bool(np.allclose(local_np, (A_np * w_np).sum(axis=1),
                                atol=1e-4))
    imp_ok = bool(np.allclose(imp_np, imp_chk, atol=1e-5))
    self_ok = all(int(group_np[keep[j]]) == j for j in range(K))
    checks = {
        "shapes": {"response": list(A_np.shape), "weights": list(w_np.shape),
                   "global_response": [N], "local_response": [N],
                   "fused_response": [N], "importance": [gh, gw],
                   "keep": [K], "group": [N], "grid": [gh, gw],
                   "n_vis": N},
        "finite": {k: bool(np.isfinite(z[k]).all())
                   for k in ("response", "weights", "global_response",
                             "local_response", "fused_response",
                             "importance")},
        "weights_sum": float(w_np.sum()),
        "weights_sum_ok": bool(abs(w_np.sum() - 1.0) < 1e-5),
        "keep_unique": bool(np.unique(keep).size == K),
        "keep_sorted": bool(np.all(np.diff(keep) > 0)),
        "keep_in_range": bool((keep >= 0).all() and (keep < N).all()),
        "group_all_assigned": bool((group_np >= 0).all()),
        "group_anchor_self": bool(self_ok),
        "fusion_fused_eq_half_g_half_local": fusion_ok,
        "fusion_local_eq_weighted_A": local_ok,
        "importance_matches_recompute_from_fused": imp_ok,
        "dropped_nearest_anchor": None,
        "bank_keep_match": {t: banks[t].get("keep_match")
                            for t in banks},
        "single_image": True, "n_vis": N, "K_actual": K, "L": L,
        "grid": [gh, gw], "aligned_reassembly_corr": align_score,
        "index_map": "token i -> cell (h, w) = (i // gw, i % gw); "
                     "cell (h,w) -> merged patch grid of the 1024x1024 "
                     "square input",
        "anomalies": [],
    }
    if dropped_idx.numel():
        Vd = V.float().double().cpu()
        a64 = F.normalize(Vd[keep_t.cpu()], dim=1)
        x64 = F.normalize(Vd[dropped_idx.cpu()], dim=1)
        gid64 = (x64 @ a64.t()).argmax(dim=1).numpy()
        ok = bool(np.array_equal(gid64, gid.cpu().numpy()))
        checks["dropped_nearest_anchor"] = ok
        if not ok:
            checks["anomalies"].append("fp64 cos-argmax recheck disagrees")
    if not (fusion_ok and local_ok and imp_ok):
        checks["anomalies"].append("fusion/importance identity failed")
    if not all(v is True for v in checks["bank_keep_match"].values()
               if v is not None):
        checks["anomalies"].append("keep differs from a same-protocol bank")
    checks["all_pass"] = bool(
        all(checks["finite"].values()) and checks["weights_sum_ok"]
        and checks["keep_unique"] and checks["keep_sorted"]
        and checks["keep_in_range"] and checks["group_all_assigned"]
        and checks["group_anchor_self"] and fusion_ok and local_ok
        and imp_ok
        and (checks["dropped_nearest_anchor"] is not False)
        and not checks["anomalies"])

    # ---------- sample.json -------------------------------------------------
    meta = {
        "dataset": ds, "sample_id": sample_id, "idx": idx,
        "image_key": str(it.get("image_key", "")),
        "question": q, "instruction_text": instr,
        "token_labels": token_labels,
        "question_span": {"start": 0, "end_exclusive": int(q_end),
                          "note": "fixed instruction suffix starts here"},
        "N": N, "K": K, "L": L, "grid": [gh, gw],
        "index_map": checks["index_map"],
        "model": common.BASELINE_MODEL,
        "checkpoint": str(getattr(eng.model.config, "_name_or_path",
                                  "unknown")),
        "method": "RTG -> original Facility -> main-only Completion "
                  "(lambda=0.25); figure arrays are the prefill-visual path "
                  "(prepare/encode/instruction_embeds/select_keep), which "
                  "rtg_legacy_checks C1 verified bitwise-identical under "
                  "both engine flag sets",
        "protocol_pairing": {
            "figure_arrays": "this export (native engine visual path; "
                             "flag-independent, see above)",
            "prediction_primary": "L_R_MAIN025 = R_MAIN025 under the "
                                  "official EADP legacy scenario "
                                  "(deepstack=False, pos='1d'); shard "
                                  "legacy_full/acc/L_R_MAIN025",
            "prediction_corroboration": ["R_MAIN025 full native panel",
                                         "R_MAIN025 dev-100 native panel"],
            "protocol_audit_20261004": "dev-100 acc/R_MAIN025 is native "
                                       "DEFAULT flags (deepstack=True, "
                                       "pos=mrope3d) — the earlier report "
                                       "mislabelled it 'legacy'; corrected "
                                       "here. Official legacy results live "
                                       "in legacy_full/acc/L_R_MAIN025.",
        },
        "git_sha": repo_sha,
        "script_sha256": shas,
        "config_frozen": {"INV_TEMP": R.INV_TEMP, "alpha": 0.5, "beta": 2.0,
                          "smoothing": "3x3 sigma=1", "minmax_eps": 1e-6,
                          "K": R.K, "seed_prefix": R.SEED_PREFIX,
                          "completion_lambda": 0.25,
                          "completion_scope": "main"},
        "banks": banks,
        "predictions": preds,
        "scores_per_question": scores,
        "source_image_sha256": src_sha,
        "source_image_ref": str(src_ref),
        "manifest_sha256": sha256_file(os.path.join(AC.OUT_DIR,
                                                    "manifest.json")),
        "generated_at": gen_time,
        "notes": [
            "response is A, the negated-cosine matrix between main visual "
            "features and instruction token embeddings (original symbols, "
            "FP32) — not attention, not annotation.",
            "weights are the frozen rtg_common.text_weight output over the "
            "FULL prompt (sum=1 over all L tokens); no stopword removal, no "
            "renormalisation; not 'attention shares'.",
            "global/local/fused are the pre-fusion quantities of the frozen "
            "rtg_importance path; importance = minmax(fused) -> 3x3 "
            "sigma=1 -> ^2 (beta=2.0).",
            "Completion y=a+0.25*(mean(G)-a) acts on the main visual stream "
            "at merge time; it is not part of these arrays.",
        ],
    }
    with open(os.path.join(sdir, "sample.json"), "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    with open(os.path.join(sdir, "checks.json"), "w") as f:
        json.dump(checks, f, indent=1)

    print(f"[done] {sample_id}: all_pass={checks['all_pass']} N={N} K={K} "
          f"L={L} bank_match={checks['bank_keep_match']} "
          f"pred_legacy={preds['L_R_MAIN025_official_legacy'].get('prediction')}")
    return checks["all_pass"]


if __name__ == "__main__":
    main()
