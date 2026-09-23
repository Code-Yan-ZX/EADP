"""
M2 figure: the accuracy-efficiency Pareto, and what each arm spends its
milliseconds on.

Panel A  macro downstream accuracy vs end-to-end TTFT (median). The Pareto plot
         the pre-registration requires. Error bars are the seed range on
         accuracy and the P10-P90 spread on TTFT. The shaded bands are the three
         pre-registered success regions relative to B1.
Panel B  macro downstream accuracy vs peak allocated GPU memory.
Panel C  the stage breakdown per arm, stacked, so it is visible where the
         milliseconds go and whether the L0-L4 full-token forward is repaid.

Reads m2_system_audit.json; writes figures/m2_pareto.png.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
import numpy as np                                                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR                                         # noqa: E402

C_BASE = "#333333"
C_CAND = "#1f77b4"
C_CTRL = "#d62728"
C_OK = "#2ca02c"

STAGE_LABELS = [
    ("image_preprocess_ms", "image preprocess", "#c7c7c7"),
    ("vision_encoder_ms", "vision encoder", "#8c8c8c"),
    ("eadp_scoring_ms", "EADP scoring", "#f0ad4e"),
    ("selector_ms", "selector", "#d9534f"),
    ("L0_L4_ms", "L0-L4 full-token", "#5bc0de"),
    ("scorer_ms", "LOCAL-MLP scorer", "#5cb85c"),
    ("token_compaction_ms", "compaction", "#9b59b6"),
    ("llm_forward_ms", "LLM forward", "#428bca"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="m2_system_audit.json")
    ap.add_argument("--out", default="m2_pareto.png")
    args = ap.parse_args()

    audit = json.load(open(os.path.join(OUTPUT_DIR, args.audit)))
    eff = audit["efficiency_table"]
    crit = audit["criteria"]
    sm = audit.get("seed_means") or {}

    def acc_of(tag):
        if tag in sm and sm[tag]:
            return sm[tag]["macro_seed_mean"], sm[tag]["macro_seed_range"]
        for key, v in audit["accuracy_table"].items():
            if v["arm"] == tag:
                return v["macro_pct"], [v["macro_pct"], v["macro_pct"]]
        return None, None

    order = [t for t in ("B0", "B1", "B2", "C0", "C1", "C2", "C3", "C1-R")
             if t in eff]
    pts = []
    for t in order:
        a, rng = acc_of(t)
        if a is None:
            continue
        e = eff[t]
        kind = ("base" if t.startswith("B") else
                "ctrl" if t.startswith("C1-") else "cand")
        pts.append(dict(tag=t, acc=a, rng=rng, ttft=e["ttft_wall_median_ms"],
                        p90=e["ttft_wall_p90_ms"], peak=e["peak_allocated_mb"],
                        kind=kind, e=e))

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.0))
    ax = axes[0]
    b1 = next((p for p in pts if p["tag"] == "B1"), None)
    if b1:
        ax.axvline(b1["ttft"], color=C_BASE, ls="--", lw=1.0, alpha=0.6)
        ax.axhline(b1["acc"], color=C_BASE, ls="--", lw=1.0, alpha=0.6)
        ax.annotate("B1 (incumbent)", (b1["ttft"], b1["acc"]),
                    textcoords="offset points", xytext=(6, -14), fontsize=8,
                    color=C_BASE)
    col = dict(base=C_BASE, cand=C_CAND, ctrl=C_CTRL)
    mrk = dict(base="s", cand="o", ctrl="^")
    for p in pts:
        ax.errorbar(p["ttft"], p["acc"],
                    yerr=[[p["acc"] - p["rng"][0]], [p["rng"][1] - p["acc"]]],
                    xerr=[[0], [max(0.0, p["p90"] - p["ttft"])]],
                    fmt=mrk[p["kind"]], color=col[p["kind"]], ms=9,
                    capsize=3, lw=1.2, zorder=3)
        ax.annotate(p["tag"], (p["ttft"], p["acc"]), textcoords="offset points",
                    xytext=(7, 5), fontsize=9, color=col[p["kind"]])
    ax.set_xlabel("end-to-end TTFT, median (ms)   [batch 1, A40, SDPA]")
    ax.set_ylabel("macro downstream accuracy (pts)")
    ax.set_title("A  accuracy–latency Pareto\n(error bars: seed range; "
                 "x-bar to P90)", fontsize=10)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for p in pts:
        ax.errorbar(p["peak"] / 1024.0, p["acc"],
                    yerr=[[p["acc"] - p["rng"][0]], [p["rng"][1] - p["acc"]]],
                    fmt=mrk[p["kind"]], color=col[p["kind"]], ms=9, capsize=3,
                    lw=1.2, zorder=3)
        ax.annotate(p["tag"], (p["peak"] / 1024.0, p["acc"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=9,
                    color=col[p["kind"]])
    ax.set_xlabel("peak torch.cuda.max_memory_allocated (GB)")
    ax.set_ylabel("macro downstream accuracy (pts)")
    ax.set_title("B  accuracy vs peak memory\n(weights dominate the footprint: "
                 "pruning moves tokens, not parameters)", fontsize=10)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ys = np.arange(len(order))
    left = np.zeros(len(order))
    for key, lab, c in STAGE_LABELS:
        vals = np.array([eff[t]["stages"].get(key, 0.0) for t in order])
        if vals.sum() <= 0:
            continue
        ax.barh(ys, vals, left=left, color=c, label=lab, height=0.68)
        left += vals
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{t}" for t in order])
    ax.invert_yaxis()
    ax.set_xlabel("mean stage time within the prefill cycle (ms)")
    ax.set_title("C  where the milliseconds go\n(image preprocessing is "
                 "wall-clock; the rest are CUDA-event brackets)", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")
    ax.grid(alpha=0.25, axis="x")

    fig.suptitle("M2 / GDEP — the system Pareto (held-out 150: 50 × "
                 "TextVQA_VAL / DocVQA_VAL / OCRBench; budget 256; "
                 f"conclusion {audit['conclusion']['code']})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "figures", args.out)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[saved] figures/{args.out}")


if __name__ == "__main__":
    main()
