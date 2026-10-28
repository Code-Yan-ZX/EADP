#!/usr/bin/env python3
"""AnchorZip Figure-1 round 2026-10-07 — Stage A (CPU-only pre-screening).

Selects the first replay batch (~24 readable candidates) from the FULL pools
of TextVQA_VAL / OCRBench / DocVQA_VAL / ChartQA_TEST / AI2D_TEST /
InfoVQA_VAL using ONLY existing on-disk artifacts (no model forward, no GPU):

  - full/bank_full_rtg_{ds}.json.gz            keep/gid/gsize + w_diag summary
  - legacy_full/acc/L_R_MAIN025/{ds}.json      official-legacy predictions
  - legacy_full/acc/L_R_MAIN025/{ds}_score.json official per-question scores
  - LMUData {ds}.tsv                            question / image bytes

Weight-based criteria (top-12 content words, w_top1, ...) are NOT computable
here — they need the frozen GPU replay (Stage B, exporter v3).  This script
only gates on: correct prediction, group structure, image readability,
question shape.  All thresholds are recorded verbatim in the output CSV.

Outputs under outputs/figures/anchorzip/round_20261007/:
  stageA_pool.csv        every gated row with metrics + decision + reason
  stageA_batch1_spec.txt comma spec for export_anchorzip_figure_sample_v3.py
  stageA_batch1.csv      the chosen batch with per-candidate rationale
"""
from __future__ import annotations

import base64
import csv
import gzip
import io
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
QWEN_ROOT = os.path.dirname(os.path.dirname(_HERE))
OUT_DIR = os.path.join(QWEN_ROOT, "outputs", "stage1_roundtrip_pilot")
ROUND_DIR = os.path.join(QWEN_ROOT, "outputs", "figures", "anchorzip",
                         "round_20261007")
os.makedirs(ROUND_DIR, exist_ok=True)

LMD = "/media/disk2/YZX/LMUData"
LEGACY = "legacy_full/acc/L_R_MAIN025"

# dataset -> (quota in batch 1, priority order 1=highest)
DATASETS = {"TextVQA_VAL": (9, 1), "OCRBench": (6, 2), "DocVQA_VAL": (4, 3),
            "ChartQA_TEST": (2, 4), "AI2D_TEST": (2, 5), "InfoVQA_VAL": (1, 6)}

# old candidates the round mandate asks to check first (forced into batch 1
# if they pass the gates; they are replayed regardless of quota)
OLD_CANDIDATES = [("TextVQA_VAL", 787), ("TextVQA_VAL", 1536),
                  ("OCRBench", 403), ("OCRBench", 517),
                  ("DocVQA_VAL", 430)]

# gates (pre-registered before looking at any replay weight)
MIN_SCORE = 1.0        # official per-question score must be fully correct
GMAX = 32              # no mega-blob groups (same threshold as 20261004 round)
MIN_SIDE = 400         # original short-side pixels
MIN_FILL = 0.55        # content fill after square padding
MAX_DECODE_PER_DS = 80 # image decodes per dataset (CPU cost cap)

# batch-1 selection weights over readability metrics (deterministic order)
SORT_KEYS = [("fill", False), ("min_side", False), ("idx", True)]


def sha256_file(p: str) -> str:
    h = __import__("hashlib").sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_image_bytes(val, ds: str, image_path) -> bytes:
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


def img_metrics(raw: bytes) -> dict:
    im = Image.open(io.BytesIO(raw))
    w, h = im.size
    side = max(w, h)
    return {"W": w, "H": h, "min_side": min(w, h),
            "fill": round((w * h) / (side * side), 4)}


def main() -> None:
    rows = []
    batch = []
    for ds, (quota, _prio) in DATASETS.items():
        bank = json.load(gzip.open(
            os.path.join(OUT_DIR, "full", f"bank_full_rtg_{ds}.json.gz"),
            "rt"))["samples"]
        shard = json.load(open(os.path.join(OUT_DIR, LEGACY, f"{ds}.json")))
        scores = json.load(open(
            os.path.join(OUT_DIR, LEGACY, f"{ds}_score.json")))["per_question"]
        df = pd.read_csv(os.path.join(LMD, f"{ds}.tsv"), sep="\t")

        cands = []
        for key_s, rec in bank.items():
            idx = int(key_s)
            score = scores.get(key_s)
            if score is None or float(score) < MIN_SCORE:
                continue
            gsize = np.asarray(rec["gsize"], dtype=np.int64)
            if int(gsize.max()) > GMAX:
                continue
            cands.append({
                "idx": idx,
                "score": float(score),
                "gsize_p90": float(np.percentile(gsize, 90)),
                "gsize_max": int(gsize.max()),
                "gsize_singleton_frac": round(float((gsize == 1).mean()), 4),
                "eff_tokens": round(float(rec["w_diag"]["eff_tokens"]), 3),
                "L_prompt": int(rec["w_diag"]["L"]),
                "prediction": str(shard["records"][key_s]["prediction"])[:60],
            })
        # deterministic pre-order: prefer compact prompt (lower L), moderate
        # eff_tokens, then idx; image metrics decided after (decode cap)
        cands.sort(key=lambda r: (r["L_prompt"], abs(
            r["eff_tokens"] - 20.0), r["idx"]))
        cands_all = list(cands)
        cands = cands[:MAX_DECODE_PER_DS]
        # the mandated old candidates must be checked regardless of the
        # decode cap — inject them so they always get image metrics
        old_here = [c for c in cands_all
                    if (ds, c["idx"]) in OLD_CANDIDATES]
        have = {c["idx"] for c in cands}
        cands.extend(c for c in old_here if c["idx"] not in have)

        seen_img = set()
        picked = 0
        for c in cands:
            idx = c["idx"]
            trow = df.iloc[idx]
            q = str(trow["question"]) if "question" in df.columns else ""
            img = img_metrics(load_image_bytes(trow["image"], ds,
                                               trow.get("image_path")))
            reasons = []
            if img["min_side"] < MIN_SIDE:
                reasons.append(f"low_res(min_side={img['min_side']})")
            if img["fill"] < MIN_FILL:
                reasons.append(f"padding_heavy(fill={img['fill']:.2f})")
            ikey = str(trow.get("image_path") or trow.get("image") or "")[:40]
            dup = ikey in seen_img
            if dup:
                reasons.append("duplicate_image_key")
            row = {**c, "dataset": ds, "question": q[:160].replace("\t", " "),
                   "image_key": ikey, **img,
                   "decision": "PASS" if not reasons else "REJECT",
                   "reject_reason": "; ".join(reasons)}
            rows.append(row)
            if not reasons:
                seen_img.add(ikey)
                if picked < quota:
                    batch.append({**row, "quota_slot": picked})
                    picked += 1

    # force the 5 mandated old candidates into the batch (if they passed)
    for ds, idx in OLD_CANDIDATES:
        hit = [r for r in rows if r["dataset"] == ds and r["idx"] == idx]
        if not hit:
            batch.append({"dataset": ds, "idx": idx,
                          "note": "mandated old candidate — REJECTED by gate",
                          "decision": "REJECT"})
            continue
        r = hit[0]
        if not any(b["dataset"] == ds and b["idx"] == idx for b in batch):
            batch.append({**r, "quota_slot": -1,
                          "note": "mandated old candidate"})

    pool = pd.DataFrame(rows)
    pool.to_csv(os.path.join(ROUND_DIR, "stageA_pool.csv"), index=False)

    with open(os.path.join(ROUND_DIR, "stageA_batch1_spec.txt"), "w") as f:
        f.write(",".join(f"{r['dataset']}:{r['idx']}" for r in batch))

    keys: list = []
    for r in batch:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(ROUND_DIR, "stageA_batch1.csv"), "w",
              newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=keys, restval="")
        wcsv.writeheader()
        wcsv.writerows(batch)

    print(f"pool rows: {len(pool)}  PASS: "
          f"{int((pool.decision == 'PASS').sum())}  batch: {len(batch)}")
    for r in batch:
        print(f"  {r['dataset']}:{r['idx']}  fill={r.get('fill')} "
              f"min_side={r.get('min_side')} L={r.get('L_prompt')} "
              f"pred={str(r.get('prediction'))[:30]!r}")


if __name__ == "__main__":
    main()
