"""
S2-C5 figure: where the teacher's head tokens are, and which information access
recovers them.

Panel A  the C5-0 answer: the student rank of every missed teacher Top-8 token,
         for HEAD_RANK and LIN_L4, with the four candidate pools marked. The mass
         to the right of 256 is what "the misses are not at the frontier" means.
Panel B  the same as a coverage curve: how much of the teacher's head sits inside
         the student's top-C, for C in 256..1024. This is the ceiling of any
         scorer restricted to a top-C pool.
Panel C  the ladder: held-out head_recall@8 per arm with the paired CI against
         HEAD_RANK, plus the wide token-local ceiling.
Panel D  the wrong-image context control: honest minus wrong-image head_recall@8
         per contextual arm and seed, against the 1 pt attribution margin.
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
from s1_audit import OUT                                              # noqa: E402

C_REF, C_LOCAL, C_GLOBAL, C_SET, C_FLOOR = (
    "#000000", "#1f77b4", "#2ca02c", "#d62728", "#888888")
C_POOL = "#cccccc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c5")
    ap.add_argument("--out", default="s2c5_factorization.png")
    args = ap.parse_args()

    D0 = json.load(open(os.path.join(OUT, f"{args.tag}_c0_diagnosis.json")))
    F = json.load(open(os.path.join(OUT, f"{args.tag}_factorization.json")))
    C = json.load(open(os.path.join(OUT, f"{args.tag}_controls.json")))

    fig = plt.figure(figsize=(19.5, 10.5))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

    # ------------------------------------------------------------- panel A
    ax = fig.add_subplot(gs[0, 0])
    reps = {"HEAD_RANK": "HEAD_RANK_seedavg", "LIN_L4": "LIN_L4"}
    for name, key in reps.items():
        h8 = D0["arms"][key]["aggregate"]["head_recall8"]
        m = D0["arms"][key]["by_k"]["8"]["missed_token_rank"]
        # the rank *distribution* is summarised by its percentiles plus the
        # bucket fractions, so plot the bucket midpoints as a step curve
        b = D0["arms"][key]["by_k"]["8"]["missed_rank_buckets"]
        edges = [256, 320, 384, 512, 768, 1024]
        fr = [b[f"frac_le_{e}"] for e in edges]
        xs, ys = [256], [0.0]
        for e, f in zip(edges, fr):
            xs += [e, e]
            ys += [ys[-1], f]
        ax.step(xs, ys, where="post", lw=2.0,
                label=f"{name} (h8={h8:.3f}, median miss={m['median']:.0f})")
    for e in (320, 384, 512, 768):
        ax.axvline(e, color=C_POOL, lw=1.0, zorder=0)
        ax.text(e, 1.02, str(e), ha="center", va="bottom", fontsize=8,
                color="#666666")
    ax.axvline(256, color=C_REF, lw=1.6, ls="--")
    ax.text(262, 0.5, "selection frontier (256)", fontsize=8, color=C_REF,
            ha="left", rotation=90, va="center")
    ax.set_xlim(248, 1024)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("student absolute rank of the missed teacher-Top-8 token")
    ax.set_ylabel("cumulative fraction of missed head tokens")
    ax.set_title("A  where the misses live: not at the frontier", loc="left",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # ------------------------------------------------------------- panel B
    ax = fig.add_subplot(gs[0, 1])
    cs = sorted(int(k.split("@")[1]) for k in
                D0["arms"]["HEAD_RANK_seedavg"]["by_k"]["8"]
                ["candidate_oracle_coverage"])
    for name, key, c in (("HEAD_RANK", "HEAD_RANK_seedavg", C_REF),
                         ("LIN_L4", "LIN_L4", C_FLOOR)):
        cov = D0["arms"][key]["by_k"]["8"]["candidate_oracle_coverage"]
        ys = [cov[f"recall8@{x}"] for x in cs]
        ax.plot(cs, ys, "o-", color=c, lw=2.0, ms=4, label=name)
    for arm in ("LOCAL-MLP", "GLOBAL-CTX", "SET-CTX"):
        y = F["arms"][arm]["seed_mean_head_recall8"]
        ax.axhline(y, color={"LOCAL-MLP": C_LOCAL, "GLOBAL-CTX": C_GLOBAL,
                             "SET-CTX": C_SET}[arm], lw=1.4, ls=":",
                   label=f"{arm} achieved ({y:.3f})")
    ax.axhline(0.82, color="#ff7f0e", lw=1.4, ls="-.",
               label="deployable gate (0.82)")
    ax.set_xlabel("candidate pool size C (a scorer may only re-rank the student's top C)")
    ax.set_ylabel("teacher Top-8 recall@C")
    ax.set_title("B  candidate-oracle coverage: the ceiling of a frontier-only scorer",
                 loc="left", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.6, 1.02)

    # ------------------------------------------------------------- panel C
    ax = fig.add_subplot(gs[1, 0])
    ref = F["rule"]["reference_head_recall8"]
    order = ["LOCAL-MLP", "GLOBAL-CTX", "SET-CTX", "LOCAL-MLP-WIDE"]
    xs = np.arange(len(order))
    # The paired bootstrap CI is on the *difference* against the reference, so
    # the delta is what carries an interval; the absolute level is printed above
    # each point rather than given a CI it does not have.
    for i, arm in enumerate(order):
        a = F["arms"][arm]
        b = a["vs_reference"]
        col = {"LOCAL-MLP": C_LOCAL, "GLOBAL-CTX": C_GLOBAL, "SET-CTX": C_SET,
               "LOCAL-MLP-WIDE": C_FLOOR}[arm]
        ax.errorbar(i, b["mean"], yerr=[[b["mean"] - b["lo"]], [b["hi"] - b["mean"]]],
                    fmt="o", ms=10, color=col, capsize=5, lw=2.0)
        ax.text(i, b["hi"] + 0.002,
                f"h8={a['seed_mean_head_recall8']:.3f}\nΔ={b['mean']:+.3f}",
                ha="center", fontsize=8)
        ax.text(i + 0.16, b["mean"], f"{'✓' if b['passes'] else '✗'}",
                fontsize=10, va="center",
                color=col if b["passes"] else C_FLOOR)
    ax.axhline(0, color=C_REF, lw=2.0, ls="--",
               label=f"HEAD_RANK reference ({ref:.3f})")
    ax.axhline(F["rule"]["margin"], color="#ff7f0e", lw=1.4, ls=":",
               label=f"+1 pt attribution margin")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{a}\n({F['arms'][a]['n_params']:,}p)" for a in order],
                       fontsize=8)
    ax.set_ylim(-0.012, 0.068)
    ax.set_ylabel("Δ head_recall@8 vs HEAD_RANK\n(bar = paired 95% CI)")
    ax.set_title(f"C  the ladder — verdict: {F['verdict']}", loc="left",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # ------------------------------------------------------------- panel D
    ax = fig.add_subplot(gs[1, 1])
    width, drawn = 0.34, 0
    for arm, col in (("GLOBAL-CTX", C_GLOBAL), ("SET-CTX", C_SET)):
        u = F["arms"][arm]["uses_ctx"]
        if not u.get("available"):
            continue
        vals = [p["honest_minus_wrong"] for p in u["per_seed"]]
        xs = np.arange(len(vals)) + (drawn - 0.5) * (width + 0.06)
        ax.bar(xs, vals, width=width, color=col, label=f"{arm}")
        drawn += 1
    ax.axhline(0, color=C_REF, lw=1.6)
    ax.axhline(F["rule"]["margin"], color="#ff7f0e", lw=1.4, ls=":",
               label="1 pt attribution margin")
    ax.axhline(-F["rule"]["margin"], color="#ff7f0e", lw=1.4, ls=":")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([f"seed {s}" for s in (0, 1, 2)])
    ax.set_ylabel("honest − wrong-image head_recall@8")
    ax.set_title("D  wrong-image context control: is the context read at all?",
                 loc="left", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("S2-C5  token-local vs set-dependent factorization", fontsize=14,
                 fontweight="bold")
    path = os.path.join(OUT, "figures", args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[saved] figures/{args.out}")


if __name__ == "__main__":
    main()
