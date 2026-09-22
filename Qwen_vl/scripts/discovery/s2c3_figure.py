"""
S2-C3 figure: does head retention convert into downstream accuracy?

Panel A is the stage's key plot -- x is the recall of the teacher's Top-8/16/32
inside the selected 256, y is macro accuracy through the identical Top-K@256
harness, one point per arm. The reference arms bracket the candidate line: the
EADP score and a random map sit at the bottom-left, the P1-G2 teacher at
(1.0, teacher). A target that buys head recall but does not move up this plot is
the GO-B outcome.

Panel B puts the head-recall change and the accuracy change on the same axis with
their paired-bootstrap CIs, so the reader can see which of the two moved and by
how much against the retraining noise.

Panel C is the validation curve (head_recall@8 and overlap256 against epoch) for
every arm and seed -- the check that the early-stopping criterion, which is
S2-C1's Top-256 overlap rather than a head metric, did not stop the head arms
before they had converged.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT  # noqa: E402

ARMS = ["BASE", "HEAD_BIN", "HEAD_MULTI", "HEAD_RANK"]
COLOR = {"BASE": "#444444", "HEAD_BIN": "#1f77b4", "HEAD_MULTI": "#2ca02c",
         "HEAD_RANK": "#d62728"}
MARK = {"BASE": "o", "HEAD_BIN": "s", "HEAD_MULTI": "^", "HEAD_RANK": "D"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="s2c3_retarget.json")
    ap.add_argument("--out", default="s2c3_head_retargeting.png")
    args = ap.parse_args()
    d = json.load(open(os.path.join(OUT, args.json)))
    R = d["references"]
    E, HB, DA = d["eval"]["table"], d["head_gain"], d["downstream_per_arm"]

    fig = plt.figure(figsize=(18.5, 11.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.32, wspace=0.26)

    # ---------------------------------------------------------------- panel A
    ks = [("head_recall8", "recall of teacher Top-8 in the selected 256"),
          ("head_recall16", "recall of teacher Top-16 in the selected 256"),
          ("head_recall32", "recall of teacher Top-32 in the selected 256")]
    for j, (mk, xlabel) in enumerate(ks):
        ax = fig.add_subplot(gs[0, j])
        # reference points
        ax.axhline(R["eadp_topk_macro"], ls=":", lw=1.2, color="#888888")
        ax.axhline(R["facility_macro"], ls="--", lw=1.2, color="#bbbbbb")
        ax.axhline(R["teacher_macro"], ls="-", lw=1.2, color="#000000", alpha=0.6)
        ax.text(0.02, R["eadp_topk_macro"] + 0.7, "EADP score @TopK", fontsize=8,
                color="#666666", transform=ax.get_yaxis_transform())
        ax.text(0.02, R["facility_macro"] + 0.7, "official EADP facility", fontsize=8,
                color="#999999", transform=ax.get_yaxis_transform())
        ax.text(0.02, R["teacher_macro"] + 0.7, "P1-G2 teacher", fontsize=8,
                color="#333333", transform=ax.get_yaxis_transform())
        ax.plot([1.0], [R["teacher_macro"]], marker="*", ms=17, color="#000000",
                zorder=5, label="P1-G2 teacher (head recall 1.0)")
        ax.plot([0.25], [R["eadp_topk_macro"]], marker="v", ms=9, color="#999999",
                zorder=5, label="random map @TopK")
        for arm in ARMS:
            if arm not in DA:
                continue
            x = HB[arm]["metrics"][mk]
            y = DA[arm]["macro_mean"]
            ax.errorbar([x], [y],
                        yerr=[[y - DA[arm]["macro_lo"]], [DA[arm]["macro_hi"] - y]],
                        xerr=[[HB[arm]["metrics"][mk]
                               - HB[arm]["seed_spread"][mk]["lo"]],
                              [HB[arm]["seed_spread"][mk]["hi"]
                               - HB[arm]["metrics"][mk]]],
                        marker=MARK[arm], ms=11, color=COLOR[arm], capsize=3,
                        lw=1.4, label=arm, zorder=6)
        # the cached S2-C1 baseline
        xb = E["S2C1_LIN_L4"][mk]
        ax.plot([xb], [R["s2c1_lin_l4_macro"]], marker="x", ms=13, mew=2.4,
                color="#111111", zorder=7, label="S2-C1 LIN_L4 (exact baseline)")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("macro accuracy (held-out 150)" if j == 0 else "", fontsize=9)
        ax.grid(alpha=0.25)
        if j == 0:
            ax.legend(fontsize=8, loc="lower right", framealpha=0.92)
    fig.suptitle("S2-C3  head retention vs downstream accuracy (Top-K @256, "
                 "held-out 150, seed mean; bars = 3-seed spread)", fontsize=12, y=0.97)

    # ---------------------------------------------------------------- panel B
    ax = fig.add_subplot(gs[1, 0])
    arms = [a for a in ARMS if a in HB and a != "BASE"]
    ypos = np.arange(len(arms))
    for n, arm in enumerate(arms):
        h = HB[arm]["vs"]["S2C1_LIN_L4"]["head_recall8"]
        a = DA[arm]
        ax.errorbar([h["delta"]], [n + 0.15], xerr=[[h["delta"] - h["ci"][0]],
                                                    [h["ci"][1] - h["delta"]]],
                    marker="o", color="#1f77b4", capsize=3, ms=8,
                    label="Δ head_recall@8" if n == 0 else None)
        dv = a["delta_vs_s2c1_lin_l4"]
        ax.errorbar([dv], [n - 0.15],
                    xerr=[[dv - a["ci_vs_s2c1_lin_l4"][0]],
                           [a["ci_vs_s2c1_lin_l4"][1] - dv]],
                    marker="s", color="#d62728", capsize=3, ms=8,
                    label="Δ macro accuracy" if n == 0 else None)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(d["min_meaningful_head_gain"], color="#1f77b4", ls=":",
               label=f"meaningful head gain (+{d['min_meaningful_head_gain']:.0f} pt)")
    ax.set_yticks(ypos)
    ax.set_yticklabels(arms, fontsize=9)
    ax.set_xlabel("change vs S2-C1 LIN_L4 (points), 95 % paired bootstrap CI", fontsize=9)
    ax.set_title("what moved: head recall, accuracy, or neither", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25, axis="x")

    # ---------------------------------------------------------------- panel C
    for n, arm in enumerate(ARMS):
        ax = fig.add_subplot(gs[1, 1 + (n // 2)]) if False else None
    ax = fig.add_subplot(gs[1, 1])
    for arm in ARMS:
        for tag, res in d["train"].items():
            if res["arm"] != arm:
                continue
            hist = res["val_history"]
            ax.plot([h["epoch"] for h in hist], [h["head_recall8"] for h in hist],
                    color=COLOR[arm], lw=1.1, alpha=0.75,
                    label=arm if tag.endswith("_s0") else None)
            ax.plot([res["best_epoch"]], [hist[res["best_epoch"]]["head_recall8"]],
                    marker="*", ms=13, color=COLOR[arm], mec="k", mew=0.5)
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel("validation head_recall@8", fontsize=9)
    ax.set_title("val head recall per epoch (★ = the epoch Top-256-overlap\n"
                 "early stopping picked)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ---------------------------------------------------------------- panel D
    ax = fig.add_subplot(gs[1, 2])
    res = d["rescue"]
    arms_r = [a for a in ARMS if f"{a}_s0" in res]
    w = 0.26
    for i, arm in enumerate(arms_r):
        r = np.mean([res[t]["rescued"] for t in res if t.startswith(arm + "_s")])
        s = np.mean([res[t]["still_wrong"] for t in res if t.startswith(arm + "_s")])
        b = np.mean([res[t]["newly_broken"] for t in res if t.startswith(arm + "_s")])
        ax.bar([i - w], [r], width=w, color="#2ca02c", label="rescued" if i == 0 else None)
        ax.bar([i], [s], width=w, color="#999999", label="still wrong" if i == 0 else None)
        ax.bar([i + w], [b], width=w, color="#d62728", label="newly broken" if i == 0 else None)
    ax.set_xticks(range(len(arms_r)))
    ax.set_xticklabels(arms_r, fontsize=9)
    ax.set_ylabel(f"instances (of {d['rescue_counts_class1']} student-wrong /"
                  " teacher-correct)", fontsize=8)
    ax.set_title("S2-C2 class-1 instances: what each target changes", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    path = os.path.join(OUT, "figures", args.out)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
