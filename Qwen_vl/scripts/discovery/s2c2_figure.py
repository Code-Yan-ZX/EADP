"""
S2-C2 figure: the budget-preserving swap curves.

Four panels -- macro plus one per benchmark -- each showing accuracy against the
number of tokens swapped in from the teacher's disagreement set. Every point is
the same 256-token budget; only which tokens are held changes.
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
from s2c2_common import DS_ALL, load_case  # noqa: E402

PANELS = DS_ALL + ["macro"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="s2c2_diagnosis.json")
    ap.add_argument("--out", default="s2c2_swap_curves.png")
    args = ap.parse_args()
    d = json.load(open(os.path.join(OUT, args.json)))
    b = d["baselines"]
    macro = d["macro"]
    cases = load_case()
    m_frac = {ds: float(np.mean([len(c["T_only"]) for c in cases if c["ds"] == ds]))
              for ds in DS_ALL}
    m_frac["macro"] = float(np.mean([m_frac[ds] for ds in DS_ALL]))

    fig, grid = plt.subplots(2, 4, figsize=(19, 8.6), sharey=False,
                              gridspec_kw=dict(height_ratios=[1.15, 1.0]))
    axes = list(grid[0])
    for ax, panel in zip(axes, PANELS):
        ds = None if panel == "macro" else panel
        stud = macro["student"] if ds is None else b[ds]["student"]
        teach = macro["teacher"] if ds is None else b[ds]["teacher"]
        eadp = macro["eadp_topk"] if ds is None else b[ds]["eadp_topk"]

        if ds is None:
            def macro_of(fn):
                ks = sorted(set.intersection(*[set(fn(x)) for x in DS_ALL]))
                return ks, [float(np.mean([fn(x)[k] for x in DS_ALL])) for k in ks]
            kt, vt = macro_of(lambda x: d["curves"]["teacher"][x])
            # the band must be the sd *of the macro*, i.e. combine seeds first and
            # benchmarks second, not an average of per-benchmark sds
            kr = sorted(set.intersection(*[set(d["curves"]["random"][x]) for x in DS_ALL]))
            vr, vrs, vrlo = [], [], []
            for k in kr:
                # seed counts differ by k (some points have one seed, some three);
                # use however many are common to all three benchmarks
                ns = min(len(d["curves"]["random"][x][k]["vals"]) for x in DS_ALL)
                per_seed_macro = np.array(
                    [np.mean([d["curves"]["random"][x][k]["vals"][i] for x in DS_ALL])
                     for i in range(ns)])
                vr.append(float(per_seed_macro.mean()))
                vrs.append(float(per_seed_macro.mean() + per_seed_macro.std()))
                vrlo.append(float(per_seed_macro.mean() - per_seed_macro.std()))
            ka, va = macro_of(lambda x: d["curves"]["adversarial"][x])
            ksh, vsh = macro_of(lambda x: d["curves"]["shuffled"][x])
        else:
            kt = sorted(d["curves"]["teacher"][ds]); vt = [d["curves"]["teacher"][ds][k] for k in kt]
            kr = sorted(d["curves"]["random"][ds])
            vr = [d["curves"]["random"][ds][k]["mean"] for k in kr]
            vrs = [d["curves"]["random"][ds][k]["mean"] + d["curves"]["random"][ds][k]["std"] for k in kr]
            vrlo = [d["curves"]["random"][ds][k]["mean"] - d["curves"]["random"][ds][k]["std"] for k in kr]
            ka = sorted(d["curves"]["adversarial"][ds]); va = [d["curves"]["adversarial"][ds][k] for k in ka]
            ksh = sorted(d["curves"]["shuffled"][ds]); vsh = [d["curves"]["shuffled"][ds][k] for k in ksh]

        # k = 0 and k = full are the two endpoints, measured by the identity arms
        kt = kt + [m_frac[panel]]
        vt = vt + [teach]

        ax.axhspan(eadp, stud, color="0.88", zorder=0)
        ax.text(0.02, (eadp + stud) / 2, "EADP Top-K\n(student's floor)", fontsize=7,
                va="center", color="0.35", transform=ax.get_yaxis_transform())
        ax.axhline(teach, color="0.55", ls="--", lw=1.0, zorder=1)
        ax.text(0.98, teach, " teacher", fontsize=7, color="0.4", va="bottom", ha="right",
                transform=ax.get_yaxis_transform())
        ax.axhline(stud, color="0.75", ls=":", lw=1.0, zorder=1)

        if kr:
            ax.fill_between(kr, vrlo, vrs, color="#d95f02", alpha=0.22, zorder=2)
            ax.plot(kr, vr, color="#d95f02", marker="s", ms=3.5, lw=1.3,
                    label="random swap (3 seeds ± sd)", zorder=3)
        # the controls are measured at a handful of k only; drawing them out to
        # the full-swap point would imply measurements that do not exist
        ax.plot(ka, va, color="#7570b3", marker="^", ms=3.5, lw=1.3, ls="none",
                label="adversarial order", zorder=3)
        ax.plot(ksh, vsh, color="#1b9e77", marker="v", ms=3.5, lw=1.3, ls="none",
                label="shuffled map control", zorder=3)
        ax.plot(kt, vt, color="#111111", marker="o", ms=4.5, lw=2.0,
                label="teacher priority", zorder=4)

        ax.set_title(panel if ds else "macro (mean of 3)", fontsize=11)
        ax.set_xlabel("tokens swapped in from $T\\setminus S$  (budget fixed at 256)")
        if panel == "macro":
            ax.set_ylabel("accuracy")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xscale("log")
        ax.set_xlim(0.85, m_frac[panel] * 1.15)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
        ax.set_xlabel("tokens swapped in from $T\\setminus S$ (log scale; budget fixed at 256)")
    axes[0].legend(fontsize=8, loc="lower right", framealpha=0.95)
    for ax in grid[1]:
        ax.axis("off")            # the 2x4 grid created four spare axes in row 2
    fig.suptitle("S2-C2 — budget-preserving swap from the LIN_L4 student set S toward the "
                 "P1-G2 teacher set T (held-out 150, 50/benchmark)", fontsize=12)
    # --- second row: rescue rate on the 28 rescuable instances ---------------
    # The discrete reading of the same experiment: how many of the 28
    # "student wrong, teacher correct" instances have a correct answer again.
    # Less noisy than an ANLS mean, and it is the quantity a method would have
    # to move.
    axr = fig.add_subplot(2, 4, 5)
    rr = (d.get("rescue_rate") or {}).get("ALL", {})
    ks_r = sorted({int(a.split(":")[1].split("|")[0]) for a in rr if ":" in a})
    for kind, col, mk in (("teacher", "#111111", "o"), ("adversarial", "#7570b3", "^"),
                          ("shuffled", "#1b9e77", "v")):
        xs = [k for k in ks_r if f"{kind}:{k}" in rr]
        ys = [rr[f"{kind}:{k}"] for k in xs]
        lab = {"teacher": "teacher priority", "adversarial": "adversarial order",
               "shuffled": "shuffled map control"}[kind]
        if kind == "teacher":
            axr.plot(xs + [m_frac["macro"]], ys + [100], color=col, marker=mk, ms=4,
                     lw=1.8, label=lab, zorder=4)
        else:
            axr.plot(xs, ys, color=col, marker=mk, ms=4, lw=1.3, ls="none",
                     label=lab, zorder=3)
    axr.axhline(100, color="0.55", ls="--", lw=1.0)
    axr.set_title("rescue rate on the 28 student-wrong / teacher-correct instances",
                  fontsize=10)
    axr.set_xlabel("tokens swapped in from $T\\setminus S$")
    axr.set_ylabel("% of the 28 now correct")
    axr.grid(alpha=0.25, lw=0.5)
    axr.set_xscale("log")
    axr.set_xlim(0.85, m_frac["macro"] * 1.15)
    axr.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    axr.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
    axr.set_xlabel("tokens swapped in from $T\\setminus S$ (log scale)")
    axr.legend(fontsize=8, loc="lower right", framealpha=0.95)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUT, "figures", args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=170)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
