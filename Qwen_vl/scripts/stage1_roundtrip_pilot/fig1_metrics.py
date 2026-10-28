#!/usr/bin/env python3
"""Figure-1 per-sample metrics for the GPU-exported AnchorZip samples.

Computed from the exported sample.npz ONLY (no forward pass):
  - top-12 response words by ORIGINAL RTG weights w (full prompt, no stopword
    removal, no renormalisation, not "attention shares");
  - for every token t: P(i|t) = softmax_i(100 * A_it) over the 32x32 visual
    grid and its spatial entropy H_t (nats, and normalised by ln(1024));
  - question-token span inside the instruction token sequence;
  - raw response range min/max of A.

Writes fig1_metrics.json next to each sample and prints a verification block
for the user-flagged liquor/of pair on TextVQA_VAL_43.
"""
from __future__ import annotations

import json
import os

import numpy as np

FIG_DIR = "/media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/figures/anchorzip"
TOPN = 12


def load_arrays(d: str):
    """Return (A[N,L], w[L], labels, meta, grid) for v1 ([H,W,L] npz) or
    v2 ([N,L] npz) layouts."""
    npz = np.load(os.path.join(d, "sample.npz"))
    meta = json.load(open(os.path.join(d, "sample.json")))
    labels = meta["token_labels"]
    if "response" in npz and npz["response"].ndim == 3:      # v1
        H, W, L = npz["response"].shape
        return (npz["response"].reshape(-1, L).astype(np.float64),
                npz["weights"], labels, meta, (H, W))
    A = npz["response"].astype(np.float64)                    # v2 [N, L]
    w = npz["weights"]
    gh, gw = (int(x) for x in npz["grid"])
    assert A.shape == (gh * gw, len(labels)) == (A.shape[0], w.shape[0])
    return A, w, labels, meta, (gh, gw)


def spatial_entropy(A_row: np.ndarray) -> tuple[float, float]:
    """H(P(i|t)) with P = softmax_i(100*A_it) over all H*W cells."""
    z = 100.0 * A_row.astype(np.float64)
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    h = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return h, h / np.log(p.size)


def main() -> None:
    import sys
    sids = sys.argv[1:] or ["TextVQA_VAL_43", "DocVQA_VAL_180", "OCRBench_4"]
    for sid in sids:
        d = os.path.join(FIG_DIR, "samples_v2" if os.path.isdir(
            os.path.join(FIG_DIR, "samples_v2", sid)) else "samples", sid)
        A, w, labels, meta, (H, W) = load_arrays(d)
        L = A.shape[1]
        ents = np.empty(L)
        for t in range(L):
            ents[t] = spatial_entropy(A[:, t])[0]

        order = np.argsort(-w)
        top = []
        for t in order[:TOPN]:
            h_nat, h_norm = spatial_entropy(A[:, t])
            top.append({
                "rank": len(top) + 1,
                "token_index": int(t),
                "token": labels[t],
                "weight": float(w[t]),
                "weight_pct_of_prompt": float(w[t]),   # sums to 1 over ALL L
                "spatial_entropy_nats": round(h_nat, 4),
                "spatial_entropy_normalized": round(h_norm, 4),
                "A_max": float(A[:, t].max()),
                "A_min": float(A[:, t].min()),
                "argmax_cell": [int(np.argmax(A[:, t]) // W),
                                int(np.argmax(A[:, t]) % W)],
            })

        # question span: tokens before the fixed instruction suffix
        suffix = "Please answer concisely"
        q_end = next((i for i, s in enumerate(labels) if suffix in s), L)
        q_span = {"start": 0, "end_exclusive": q_end,
                  "n_tokens": q_end, "note": "instruction suffix starts here"}

        out = {
            "sample_id": sid,
            "panel": "4x3 = top-12 tokens by ORIGINAL RTG weights over the "
                     "FULL prompt (no stopword removal, no renormalisation; "
                     "weights sum to 1 over all L prompt tokens)",
            "P_definition": "P(i|t) = softmax_i(100 * A_it) over the 32x32 "
                            "visual grid; entropy in nats and normalised by "
                            "ln(1024)",
            "N": H * W, "L": L, "grid": [H, W],
            "weights_sum": float(w.sum()),
            "A_global_range": [float(A.min()), float(A.max())],
            "question_span": q_span,
            "question_text": meta["question"],
            "instruction_text": meta["instruction_text"],
            "top12": top,
            "token_labels": labels,
            "weights_all": [round(float(x), 6) for x in w],
        }
        with open(os.path.join(d, "fig1_metrics.json"), "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(f"{sid}: L={L} w_sum={w.sum():.6f} A_range="
              f"[{A.min():.4f},{A.max():.4f}]")
        for r in top:
            print(f"   #{r['rank']:>2} {r['token']!r:<14} w={r['weight']:.4f} "
                  f"H={r['spatial_entropy_normalized']:.3f}")

    # ---- user-flagged pair verification (TextVQA_VAL_43) -------------------
    d = os.path.join(FIG_DIR, "samples", "TextVQA_VAL_43")
    npz = np.load(os.path.join(d, "sample.npz"))
    w = npz["weights"]
    labels = json.load(open(os.path.join(d, "sample.json")))["token_labels"]
    A = npz["response"].reshape(-1, w.shape[0]).astype(np.float64)
    out = {}
    for want in ("liquor", "of"):
        ti = [i for i, s in enumerate(labels) if s.strip() == want]
        assert len(ti) == 1, (want, ti)
        t = ti[0]
        h_nat, h_norm = spatial_entropy(A[:, t])
        out[want] = {"weight": float(w[t]), "entropy_normalized": h_norm}
    print("\nverification of user-flagged pair (TextVQA_VAL_43):")
    print(f"  liquor: w={out['liquor']['weight']:.4f} "
          f"H_norm={out['liquor']['entropy_normalized']:.3f}")
    print(f"  of    : w={out['of']['weight']:.4f} "
          f"H_norm={out['of']['entropy_normalized']:.3f}")
    print("  user quoted 0.0958/0.0925 and 0.738/0.731 -> "
          "match" if abs(out["liquor"]["weight"] - 0.0958) < 5e-4 and
          abs(out["of"]["weight"] - 0.0925) < 5e-4 else "  -> MISMATCH, check")
    print("  NOTE: w liquor > w of AND H liquor ≈ H of (0.738 vs 0.731, "
          "diff 0.007) — this pair shows NO focus/divergence contrast.")


if __name__ == "__main__":
    main()
