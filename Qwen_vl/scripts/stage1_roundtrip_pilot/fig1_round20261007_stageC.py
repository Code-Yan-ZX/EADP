#!/usr/bin/env python3
"""AnchorZip Figure-1 round 2026-10-07 — Stage C (metrics from real arrays).

Reads the GPU-replayed samples under outputs/figures/anchorzip/samples_v3/
and computes, from sample.npz ONLY, the round-mandate screening columns:

  - original top-12 tokens by RTG weights w (no stopword removal, no
    renormalisation), with question content / function / template
    classification (fixed rule recorded below);
  - w_top1, L*w_top1 (multiple over uniform 1/L), content-vs-function gap,
    top-12 total weight, content-word counts in top-3 / top-12;
  - real A range / std; pairwise Pearson correlations of the 12 shown
    response maps;
  - P(i|t) = softmax_i(100*A_it): normalised spatial entropy and top-10%
    position mass (both per shown token, aggregated);
  - group-size p90 / max / singleton fraction (structural constants N/K are
    noted but NOT used as quality metrics);
  - verification status + decision.

Classification rule (fixed, pre-registered): tokens at or after the fixed
instruction suffix ("Please answer ...") are 'template'; earlier tokens
whose stripped lowercase label is punctuation or in FUNCTION_WORDS are
'function'; everything else is 'content'.  Token-level (subwords inherit
nothing); the rule and word list ship with this file.

Usage:
    python fig1_round20261007_stageC.py            # all exported samples
    python fig1_round20261007_stageC.py SID ...    # subset
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

FIG_DIR = "/media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/figures/anchorzip"
SV3 = os.path.join(FIG_DIR, "samples_v3")
ROUND_DIR = os.path.join(FIG_DIR, "round_20261007")
SUFFIX = "Please answer concisely"

FUNCTION_WORDS = {
    "what", "which", "where", "who", "whom", "whose", "when", "is", "are",
    "was", "were", "the", "a", "an", "of", "on", "in", "at", "to", "for",
    "with", "by", "from", "and", "or", "this", "that", "these", "those",
    "it", "its", "as", "be", "been", "does", "do", "did", "can", "could",
    "please", "answer", "concisely", "short", "words", "phrases", "possible",
    "given", "there", "their", "his", "her", "how", "many", "much", "than",
    "into", "about", "up", "down", "out", "over", "under", "not", "no",
    "yes", "you", "your", "he", "she", "they", "them", "we", "us", "will",
    "would", "should", "may", "might", "must", "has", "have", "had",
}
PUNCT = set(".,?！';:\"()[]{}<>/\\|`~@#$%^&*+-=_\n\t")


def classify(label: str, t: int, q_end: int) -> str:
    if t >= q_end:
        return "template"
    s = label.strip().lower().strip(".,?!\n")
    if not s or all(ch in PUNCT for ch in s):
        return "function"
    if s in FUNCTION_WORDS:
        return "function"
    return "content"


def norm_entropy(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(p.size))


def softmax100(row: np.ndarray) -> np.ndarray:
    z = 100.0 * row.astype(np.float64)
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


def main() -> None:
    sids = sys.argv[1:]
    if not sids:
        sids = sorted(d for d in os.listdir(SV3)
                      if os.path.isdir(os.path.join(SV3, d)))
    rows = []
    for sid in sids:
        d = os.path.join(SV3, sid)
        try:
            npz = np.load(os.path.join(d, "sample.npz"))
            meta = json.load(open(os.path.join(d, "sample.json")))
            checks = json.load(open(os.path.join(d, "checks.json")))
        except Exception as e:
            rows.append({"sample_id": sid, "decision": "REJECT",
                         "reject_reason": f"unreadable export: {e}"})
            continue
        A = npz["response"].astype(np.float64)          # [H,W,L]
        H, W, L = A.shape
        w = npz["weights"].astype(np.float64)           # [L]
        labels = meta["token_labels"]
        # token-level labels cannot contain the full SUFFIX substring;
        # the fixed suffix always starts with the standalone token "Please"
        q_end = next((i for i, s in enumerate(labels)
                      if s.strip() == "Please"), L)
        flat = A.reshape(H * W, L)

        order = np.argsort(-w, kind="stable")
        top12 = order[:12]
        cls = [classify(labels[t], int(t), q_end) for t in top12]
        content_top = [i for i, c in enumerate(cls) if c == "content"]
        # content vs highest-weight non-content token within top-12
        nonc = [i for i, c in enumerate(cls) if c != "content"]
        w_top_content = float(w[top12[content_top[0]]]) if content_top else 0.0
        w_top_noncontent = float(w[top12[nonc[0]]]) if nonc else 0.0
        gap = w_top_content - w_top_noncontent

        # P(i|t) statistics over the shown tokens
        ents, mass10 = [], []
        ntop = max(1, int(round(0.10 * (H * W))))
        for t in top12:
            p = softmax100(flat[:, t])
            ents.append(norm_entropy(p))
            idx10 = np.argpartition(-p, ntop - 1)[:ntop]
            mass10.append(float(p[idx10].sum()))
        # pairwise correlation of the 12 shown response maps
        Rm = flat[:, top12]
        Rm = Rm - Rm.mean(axis=0, keepdims=True)
        sd = Rm.std(axis=0)
        C = (Rm.T @ Rm) / np.outer(sd, sd) / (H * W)
        iu = np.triu_indices(12, 1)
        pw = C[iu]

        gsize = npz["group_size"].astype(np.int64)
        N = int(gsize.sum())
        K = int(gsize.shape[0])
        # readability / fill from the preprocessing record
        pre = meta.get("preprocessing", {})
        img_w, img_h = None, None
        src = os.path.join(d, next((f for f in os.listdir(d)
                                    if f.startswith("source_original.")), ""))
        if os.path.exists(src):
            from PIL import Image
            iw, ih = Image.open(src).size
            img_w, img_h = iw, ih
            fill = round((iw * ih) / (max(iw, ih) ** 2), 4)
            min_side = min(iw, ih)
        else:
            fill, min_side = None, None

        top1 = int(top12[0])
        reasons = []
        if not checks.get("all_pass"):
            reasons.append("checks_not_all_pass:" +
                           ";".join(checks.get("anomalies", [])))
        if len(content_top) < 3:
            reasons.append(f"top12_content_lt_3({len(content_top)})")
        if content_top and min(content_top) > 5:
            reasons.append("no_content_word_in_top6")
        bkm = checks.get("bank_keep_match", {})
        if bkm and not all(v is True for v in bkm.values() if v is not None):
            reasons.append("bank_keep_mismatch")
        decision = "PASS" if not reasons else "REJECT"
        rows.append({
            "sample_id": sid,
            "dataset": meta["dataset"], "idx": meta["idx"],
            "question": meta["question"][:120].replace("\t", " "),
            "img_W": img_w, "img_H": img_h, "min_side": min_side,
            "fill": fill,
            "canvas_fill": pre.get("valid_content_fraction_of_canvas"),
            "protocol": "official_legacy L_R_MAIN025 (N=1024 K=256)",
            "N": N, "K": K, "L": L,
            "all_pass": checks.get("all_pass"),
            "bank_keep_match": json.dumps(bkm),
            "top12": json.dumps(
                [{"rank": i + 1, "tok": labels[int(t)].strip(),
                  "cls": cls[i], "w": round(float(w[int(t)]), 5)}
                 for i, t in enumerate(top12)], ensure_ascii=False),
            "top12_content_n": len(content_top),
            "top3_content_n": int(sum(c == "content" for c in cls[:3])),
            "w_top1": round(float(w[top1]), 6),
            "w_top1_label": labels[top1].strip(),
            "w_top1_class": cls[0],
            "L_w_top1_vs_uniform": round(L * float(w[top1]), 4),
            "top12_weight_sum": round(float(w[top12].sum()), 6),
            "w_top_content": round(w_top_content, 6),
            "w_top_noncontent": round(w_top_noncontent, 6),
            "content_minus_function_gap": round(gap, 6),
            "A_range": [round(float(flat.min()), 4),
                        round(float(flat.max()), 4)],
            "A_std": round(float(flat.std()), 5),
            "pairwise_corr_mean": round(float(pw.mean()), 4),
            "pairwise_corr_std": round(float(pw.std()), 4),
            "pairwise_corr_min": round(float(pw.min()), 4),
            "H_norm_median_top12": round(float(np.median(ents)), 4),
            "H_norm_min_top12": round(float(np.min(ents)), 4),
            "top10pct_mass_mean_top12": round(float(np.mean(mass10)), 4),
            "P_definition": "P(i|t)=softmax_i(100*A_it); entropy normalised "
                            "by ln(1024); top-10% mass = sum of P over the "
                            "103 highest-probability visual positions",
            "group_p90": round(float(np.percentile(gsize, 90)), 2),
            "group_max": int(gsize.max()),
            "group_singleton_frac": round(float((gsize == 1).mean()), 4),
            "highlight_slot": checks["completion"]["highlight_slot"],
            "highlight_n_dropped":
                checks["completion"]["highlight_n_dropped"],
            "decision": decision,
            "reject_reason": "; ".join(reasons),
        })

    keys: list = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(ROUND_DIR, "screening_round20261007.csv"), "w",
              newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=keys, restval="")
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"{len(rows)} rows -> screening_round20261007.csv")
    for r in rows:
        if r.get("decision") == "PASS":
            print(f"  PASS {r['sample_id']}  top3c={r['top3_content_n']} "
                  f"top12c={r['top12_content_n']} w1={r['w_top1']:.4f} "
                  f"({r['w_top1_label']!r},{r['w_top1_class']}) "
                  f"Lw1={r['L_w_top1_vs_uniform']} "
                  f"Hmed={r['H_norm_median_top12']} "
                  f"pw={r['pairwise_corr_mean']}±{r['pairwise_corr_std']}")


if __name__ == "__main__":
    main()
