"""
S2-C6 figure: where the residual signal is, and whether it is there at all.

Panel A  the teacher-data learning curve: held-out teacher Top-8 recall@256 as a
         function of the number of fit images, with every seed drawn, the
         bootstrap CI of the mean, and the marginal 180 -> 240 segment called
         out. This is the panel that decides whether ~0.80 can be read as a
         representation ceiling.
Panel B  the representation arms: held-out R@8 (left axis, absolute) and the
         paired delta against each arm's *governing* reference (right of the
         line), so the architecture control L4+L4 is visible next to the
         trajectory arms it governs.
Panel C  the mechanism controls: honest minus wrong-image held-out R@8 for the
         dual arms' two substitutions and for the query arm's shuffled query,
         against the 1 pt attribution margin. A bar near zero means the arm was
         not reading the quantity that was substituted.
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

MARGIN = 0.01
C_LINE, C_PT, C_CTRL, C_QUERY = "#1f77b4", "#333333", "#d62728", "#2ca02c"
C_GOOD, C_BAD = "#2ca02c", "#999999"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c6")
    ap.add_argument("--out", default="s2c6_localization.png")
    args = ap.parse_args()

    A = json.load(open(os.path.join(OUT, f"{args.tag}_audit.json")))
    fig = plt.figure(figsize=(19.5, 10.5))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.24)

    # ------------------------------------------------------------- panel A
    ax = fig.add_subplot(gs[0, 0])
    curve = A["part_A"]["curve"]
    ns = sorted(int(k) for k in curve)
    mean = [curve[str(n)]["held_out"]["head_recall8"] for n in ns]
    per = [curve[str(n)]["per_seed_head_recall8"] for n in ns]
    for i, n in enumerate(ns):
        ax.scatter([n] * len(per[i]), per[i], s=34, color=C_PT, alpha=0.55,
                   zorder=3, label="seed" if i == 0 else None)
    ax.plot(ns, mean, "-o", color=C_LINE, lw=2.2, ms=8, zorder=4,
            label="mean held-out R@8")
    seg = A["part_A"]["marginal_180_to_240"]
    ax.annotate(
        f"180 -> 240: {seg['mean']:+.4f}\nCI [{seg['lo']:+.4f}, {seg['hi']:+.4f}]"
        f"\n{'RESOLVED GAIN' if seg['resolved'] else 'not resolved'}",
        xy=(ns[-1], mean[-1]), xytext=(0.42, 0.30), textcoords="axes fraction",
        fontsize=11, ha="left",
        arrowprops=dict(arrowstyle="->", color=C_PT, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.45", fc="#f4f4f4", ec="#bbbbbb"))
    ax.set_xlabel("fit images (nested, benchmark-stratified)")
    ax.set_ylabel("held-out teacher Top-8 recall@256")
    ax.set_title(f"A  teacher-data scaling -- verdict: {A['part_A']['verdict']}",
                 loc="left", fontsize=13)
    ax.set_xticks(ns)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=10)

    # ------------------------------------------------------------- panel B
    ax = fig.add_subplot(gs[0, 1])
    arms = A["part_B"]["arms"]
    order = [a for a in ("L2", "DELTA", "L4+L4", "L2+L4", "L4+DELTA")
             if a in arms]
    ref = arms["L4"]["held_out"]["head_recall8"] if "L4" in arms else None
    ys = np.arange(len(order))
    vals = [arms[a]["held_out"]["head_recall8"] for a in order]
    ax.barh(ys, vals, color=[C_CTRL if a == "L4+L4" else C_LINE for a in order],
            alpha=0.85, height=0.6)
    if ref is not None:
        ax.axvline(ref, color=C_PT, ls="--", lw=1.6,
                   label=f"L4 reference {ref:.4f}")
    for y, a in zip(ys, order):
        v = arms[a]["vs_reference"]
        col = C_GOOD if v["resolved"] else C_BAD
        ax.text(vals[order.index(a)] + 0.002, y,
                f"  {v['mean']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]"
                f" vs {arms[a]['governing_reference']}"
                + ("  RESOLVED" if v["resolved"] else
                   ("  below-margin" if v["below_margin"] else "")),
                va="center", fontsize=10, color=col)
    ax.set_yticks(ys)
    ax.set_yticklabels(order)
    ax.set_xlim(min(vals) - 0.03, max(vals) + 0.09)
    ax.set_xlabel("held-out teacher Top-8 recall@256")
    ax.set_title("B  representation arms, and the dual-architecture control",
                 loc="left", fontsize=13)
    ax.grid(alpha=0.25, axis="x")
    if ref is not None:
        ax.legend(loc="lower right", fontsize=10)

    # ------------------------------------------------------------- panel C
    ax = fig.add_subplot(gs[1, 0])
    rows = []
    for a, s in arms.items():
        for name, c in (s.get("controls") or {}).items():
            if name.startswith("SELF_CHECK"):
                continue
            rows.append((f"{a}\n{name.replace('WRONG_', '')}",
                         c["honest_minus_wrong"]))
    for a, s in A["part_C"]["arms"].items():
        for name, c in (s.get("controls") or {}).items():
            rows.append((f"{a}\n{name.replace('WRONG_', '')}",
                         c["honest_minus_wrong"]))
    if rows:
        ys = np.arange(len(rows))
        m = [r[1]["mean"] for r in rows]
        lo = [r[1]["mean"] - r[1]["lo"] for r in rows]
        hi = [r[1]["hi"] - r[1]["mean"] for r in rows]
        ax.barh(ys, m, xerr=[lo, hi], color=C_QUERY, alpha=0.85, height=0.6,
                error_kw=dict(lw=1.4, capsize=4))
        ax.axvline(0, color=C_PT, lw=1.4)
        ax.axvline(MARGIN, color=C_CTRL, ls="--", lw=1.4,
                   label=f"1 pt margin ({MARGIN})")
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=9)
        ax.set_xlabel("honest − wrong-image held-out R@8")
        ax.legend(loc="lower right", fontsize=10)
    else:
        ax.text(0.5, 0.5, "no controls", ha="center", transform=ax.transAxes)
    ax.set_title("C  mechanism controls: is the substituted quantity read?",
                 loc="left", fontsize=13)
    ax.grid(alpha=0.25, axis="x")

    # ------------------------------------------------------------- panel D
    ax = fig.add_subplot(gs[1, 1])
    lab = ["all", "TextVQA_VAL", "DocVQA_VAL", "OCRBench"]
    xs = np.arange(len(lab))
    for a, col, mk in (("L4", C_PT, "o"), ("L2+L4", C_QUERY, "s"),
                       ("L4+DELTA", C_CTRL, "^"), ("L4+QUERY", "#9467bd", "D")):
        s = arms.get(a) or A["part_C"]["arms"].get(a)
        if not s:
            continue
        by = s["by_benchmark_head_recall8"]
        vals = [s["held_out"]["head_recall8"]] + [by.get(d, np.nan) for d in lab[1:]]
        ax.plot(xs, vals, mk + "-", color=col, lw=1.9, ms=8, label=a)
    ax.set_xticks(xs)
    ax.set_xticklabels(lab, fontsize=10)
    ax.set_ylabel("held-out teacher Top-8 recall@256")
    ax.set_title("D  per-benchmark breakdown", loc="left", fontsize=13)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)
    ax.set_ylim(0.5, 1.0)

    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    path = os.path.join(OUT, "figures", args.out)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[saved] figures/{args.out}")


if __name__ == "__main__":
    main()
