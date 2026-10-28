#!/usr/bin/env python3
"""AnchorZip Figure-1 CPU screening (no model forward, no GPU).

Screens the frozen-method (RTG -> original Facility -> main-only Completion
lambda=0.25, arm R_MAIN025 on Qwen3) dev-100 pools of TextVQA_VAL /
DocVQA_VAL / OCRBench using ONLY existing on-disk artifacts:

  - bank_dev_rtg_{ds}.json.gz      keep/gid/gsize per dev sample
  - acc/R_MAIN025/{ds}.json        frozen-method predictions
  - acc/R_MAIN025/{ds}_score.json  official per-question scores
  - LMUData {ds}.tsv               question / answer / image bytes

Outputs (under outputs/figures/anchorzip/delivery_20261004/):
  screening.csv   every checked dev sample: metrics + decision + reject reason
  candidates.csv  ranked recommended candidates
  inventory.csv   artifact inventory (paths, sha256, sizes, counts)

No stopword removal, no renormalisation anywhere: RTG weights are reported
verbatim from the exported arrays (sample.npz) for the three GPU-exported
samples; this script computes NO weights (that needs a forward pass).
"""
from __future__ import annotations

import ast
import base64
import csv
import gzip
import hashlib
import io
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
QWEN_ROOT = os.path.dirname(os.path.dirname(_HERE))
OUT_DIR = os.path.join(QWEN_ROOT, "outputs", "stage1_roundtrip_pilot")
ACV_DIR = os.path.join(QWEN_ROOT, "outputs", "anchor_completion_validation")
AMP_DIR = os.path.join(QWEN_ROOT, "outputs", "anchor_merge_pilot")
FIG_DIR = os.path.join(QWEN_ROOT, "outputs", "figures", "anchorzip")
DELIV_DIR = os.path.join(FIG_DIR, "delivery_20261004")
os.makedirs(DELIV_DIR, exist_ok=True)

LMD = "/media/disk2/YZX/LMUData"
DATASETS = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]
GRID = 32  # merged-token grid (1024 tokens / 2x2 merge over 64x64 patches)

# screening thresholds (pre-registered here, before looking at results)
MIN_SIDE = 400          # original short-side pixels (figure legibility)
MIN_FILL = 0.55         # content fill after square padding (grid alignment)
MIN_SCORE = 0.9         # per-question official score (method working)
GMEAN_LO, GMEAN_HI = 1.0, 8.0   # mean group size window
GMAX = 32               # no mega-blobs
KEEP_ENT_LO = 0.35      # keep-map spatial entropy (normalized) not degenerate


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_entropy(p: np.ndarray) -> float:
    p = p / max(p.sum(), 1e-12)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(p.size))


def img_metrics(raw: bytes) -> dict:
    im = Image.open(io.BytesIO(raw))
    w, h = im.size
    side = max(w, h)
    fill = (w * h) / (side * side)          # content fraction after square pad
    return {"W": w, "H": h, "min_side": min(w, h), "fill": round(fill, 3),
            "bytes": len(raw), "mode": im.mode, "format": im.format}


def load_image_bytes(val, ds: str, image_path) -> bytes:
    """Dedup-TSV convention: base64 inline, else a key into images/<ds>/."""
    s = str(val)
    if s.startswith(("/9j", "iVBOR", "iVBORw", "R0lGOD")) or len(s) > 256:
        return base64.b64decode(s)
    name = str(image_path) if image_path is not None and not pd.isna(
        image_path) else f"{s}.jpg"
    p = os.path.join(LMD, "images", ds, name)
    if not os.path.exists(p):
        p = os.path.join(LMD, "images", ds, f"{s}.jpg")
    with open(p, "rb") as f:
        return f.read()


def main() -> None:
    manifest = json.load(open(os.path.join(AMP_DIR, "manifest.json")))
    rows = []
    for ds in DATASETS:
        dev = manifest["datasets"][ds]["dev"]
        bank = json.load(gzip.open(os.path.join(
            OUT_DIR, f"bank_dev_rtg_{ds}.json.gz"), "rt"))
        shard = json.load(open(os.path.join(OUT_DIR, "acc", "R_MAIN025",
                                            f"{ds}.json")))
        scores = json.load(open(os.path.join(OUT_DIR, "acc", "R_MAIN025",
                                             f"{ds}_score.json")))
        assert scores["arm"] == "R_MAIN025" and shard["meta"]["arm"] == "R_MAIN025"
        df = pd.read_csv(os.path.join(LMD, f"{ds}.tsv"), sep="\t")
        df = df.set_index("index")

        for it in dev:
            idx = int(it["idx"])
            rec = bank["samples"].get(str(idx))
            pred = shard["records"].get(str(idx))
            reasons = []
            if rec is None or pred is None:
                rows.append({"dataset": ds, "idx": idx,
                             "decision": "REJECT",
                             "reject_reason": "missing bank/prediction record"})
                continue
            trow = df.loc[int(it.get("tsv_index", idx))] \
                if "tsv_index" in it else None
            # manifest idx is the dataset row index used by VLMEvalKit;
            # locate the TSV row by position (dataset.build == reset_index order)
            if idx < len(df):
                trow = df.iloc[idx]
            q = str(trow["question"])
            img = img_metrics(load_image_bytes(trow["image"], ds,
                                               trow.get("image_path")))

            keep = np.asarray(rec["keep"], dtype=np.int64)
            gsize = np.asarray(rec["gsize"], dtype=np.int64)
            kh, kw = keep // GRID, keep % GRID
            kmap = np.zeros((GRID, GRID))
            np.add.at(kmap, (kh, kw), 1.0)
            keep_ent = norm_entropy(kmap.ravel())
            gmean, gmax = float(gsize.mean()), int(gsize.max())
            g_ent = norm_entropy(np.bincount(gsize, minlength=64).astype(float))
            score = float(scores["per_question"].get(str(idx), np.nan))

            if img["min_side"] < MIN_SIDE:
                reasons.append(f"low_res(min_side={img['min_side']})")
            if img["fill"] < MIN_FILL:
                reasons.append(f"padding_heavy(fill={img['fill']:.2f})")
            if not (score >= MIN_SCORE):
                reasons.append(f"prediction_not_correct(score={score:.2f})")
            if not (GMEAN_LO <= gmean <= GMEAN_HI):
                reasons.append(f"degenerate_groups(gmean={gmean:.2f})")
            if gmax > GMAX:
                reasons.append(f"mega_blob(gmax={gmax})")
            if keep_ent < KEEP_ENT_LO:
                reasons.append(f"keep_degenerate(ent={keep_ent:.2f})")

            rows.append({
                "dataset": ds, "idx": idx,
                "image_key": str(it.get("image_key", "")),
                "question": q[:160].replace("\t", " "),
                "prediction": str(pred["prediction"])[:60],
                "official_score": round(score, 4),
                "W": img["W"], "H": img["H"],
                "min_side": img["min_side"], "fill": img["fill"],
                "img_bytes": img["bytes"],
                "keep_spatial_entropy": round(keep_ent, 4),
                "group_mean": round(gmean, 3), "group_max": gmax,
                "group_size_entropy": round(g_ent, 4),
                "decision": "PASS" if not reasons else "REJECT",
                "reject_reason": "; ".join(reasons) if reasons else "",
            })

    scr = pd.DataFrame(rows)
    scr.to_csv(os.path.join(DELIV_DIR, "screening.csv"), index=False)
    n_pass = int((scr.decision == "PASS").sum())
    print(f"screened {len(scr)} rows, PASS {n_pass}")

    # ---- rank the PASS pool into candidates -------------------------------
    pool = scr[scr.decision == "PASS"].copy()
    if len(pool):
        # composite: correctness, then structure spread, then sharpness
        pool["rank_score"] = (
            2.0 * pool.official_score.fillna(0)
            + 1.0 * pool.keep_spatial_entropy
            + 0.5 * pool.group_size_entropy
            + 0.3 * np.log1p(pool.min_side) / np.log(4000)
        )
        pool = pool.sort_values("rank_score", ascending=False)
    pool.to_csv(os.path.join(DELIV_DIR, "candidates.csv"), index=False)
    print(pool[["dataset", "idx", "official_score", "keep_spatial_entropy",
                "group_mean", "group_max", "min_side", "rank_score"]]
          .head(12).to_string(index=False))

    # ---- inventory ---------------------------------------------------------
    inv = []
    def add(path, kind, note=""):
        if os.path.exists(path):
            inv.append({"artifact": kind, "path": path,
                        "bytes": os.path.getsize(path),
                        "sha256": sha256_file(path) if
                        os.path.getsize(path) < 200 << 20 else
                        "(>200MB, not hashed)", "note": note})
        else:
            inv.append({"artifact": kind, "path": path, "bytes": None,
                        "sha256": "MISSING", "note": note})
    for ds in DATASETS:
        add(os.path.join(OUT_DIR, f"bank_dev_rtg_{ds}.json.gz"),
            "rtg_bank", f"keep/gid/gsize for dev-100 of {ds}")
        add(os.path.join(OUT_DIR, "acc", "R_MAIN025", f"{ds}.json"),
            "R_MAIN025_predictions", "frozen-method predictions (legacy path)")
        add(os.path.join(OUT_DIR, "acc", "R_MAIN025", f"{ds}_score.json"),
            "R_MAIN025_official_scores", "per-question official scores")
        add(os.path.join(OUT_DIR, "acc", "R_MAIN025", f"{ds}_pred.tsv"),
            "R_MAIN025_pred_tsv", "VLMEvalKit-format prediction dump")
    add(os.path.join(AMP_DIR, "manifest.json"), "dev_manifest",
        "dev/confirm split, seed-anchored")
    for ds in DATASETS:
        add(os.path.join(LMD, f"{ds}.tsv"), "lmudata_tsv", ds)
    for sid in ["TextVQA_VAL_43", "DocVQA_VAL_180", "OCRBench_4"]:
        for f in ["sample.npz", "sample.json", "checks.json",
                  "sample_aligned.png"]:
            add(os.path.join(FIG_DIR, "samples", sid, f),
                "gpu_exported_sample", f"{sid}/{f}")
    pd.DataFrame(inv).to_csv(os.path.join(DELIV_DIR, "inventory.csv"),
                             index=False)
    print("inventory rows:", len(inv))


if __name__ == "__main__":
    main()
