"""
S2-C4 figure: is the per-image oracle direction transferable, and if not, why not?

Panel A  the distribution of the whole cross-image transfer matrix -- every
         (image, other-image direction) pair -- with the arms that pick one
         column marked on it. The gap between the left mass and the marked arms
         is what "not transferable" means numerically.
Panel B  every arm's head_recall@8, sorted, against the three references that
         matter: no adaptation (GLOBAL), the S2-C3 trained shared direction, and
         the in-sample oracle SELF.
Panel C  the PCA spectrum of the fit-set directions, with the variance retained
         by a k-component basis and the recall that basis preserves.
Panel D  the two levels of the low-dimensional test: the oracle coefficient
         bound and the deployable ridge predictor, against k. The vertical gap
         between the curves is the predictability failure; the gap between the
         oracle curve and SELF is the low-dimensionality failure.
Panel E  per-image transfer: the mean and the best over the 240 candidate
         directions, per held-out image. A near-1.0 best with a ~0.55 mean says
         the right direction exists for almost every image but is not findable.
Panel F  pairwise cosine similarity of the oracle directions, within and across
         benchmarks -- the geometry that explains why.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                                 # noqa: E402
from s2c2_common import DS_ALL                                           # noqa: E402

C_ORACLE, C_SHARED, C_DEPLOY, C_FLOOR = "#000000", "#d62728", "#1f77b4", "#888888"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c4")
    ap.add_argument("--out", default="s2c4_direction_transfer.png")
    args = ap.parse_args()

    D = json.load(open(os.path.join(OUT, f"{args.tag}_direction.json")))
    T = json.load(open(os.path.join(OUT, f"{args.tag}_transfer.json")))
    P = json.load(open(os.path.join(OUT, f"{args.tag}_pca.json")))
    Z = np.load(os.path.join(OUT, f"{args.tag}_transfer.npz"))
    Zg = np.load(os.path.join(OUT, f"{args.tag}_geometry.npz"))

    self_r8 = D["rule_inputs"]["SELF_head_recall8"]
    global_r8 = D["rule_inputs"]["GLOBAL_head_recall8"]
    trained_r8 = D["rule_inputs"]["BEST_TRAINED_head_recall8"]
    dep_r8 = D["rule_inputs"]["BEST_DEPLOY_head_recall8"]

    fig = plt.figure(figsize=(19.5, 11.5))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.24)

    # ------------------------------------------------------------- panel A
    ax = fig.add_subplot(gs[0, 0])
    M = Z["M_recall8"].ravel()
    ax.hist(M, bins=60, color="#bcd4e6", edgecolor="#4a7fa5", lw=0.4)
    for x, c, lab, ls in [
            (M.mean(), C_FLOOR, f"random other image ({M.mean():.3f})", "-"),
            (global_r8, "#2ca02c", f"GLOBAL direction ({global_r8:.3f})", "--"),
            (dep_r8, C_DEPLOY, f"best deployable ({dep_r8:.3f})", "-"),
            (trained_r8, C_SHARED, f"S2-C3 trained ({trained_r8:.3f})", "-"),
            (self_r8, C_ORACLE, f"SELF oracle ({self_r8:.3f})", "-")]:
        ax.axvline(x, color=c, ls=ls, lw=2.0, label=lab)
    ax.set_xlabel("head_recall@8 of one image's direction on another image", fontsize=9)
    ax.set_ylabel(f"pairs (150 held-out images x 240 fit directions)", fontsize=9)
    ax.set_title("A  the cross-image transfer matrix is mostly low, with a real\n"
                 "spike at 1.0: a random other-image direction keeps 0.54",
                 fontsize=10)
    ax.legend(fontsize=7.6, loc="upper center")
    ax.grid(alpha=0.25)

    # ------------------------------------------------------------- panel B
    ax = fig.add_subplot(gs[0, 1])
    arms = sorted([a for a in D["arms"] if not a.get("optimistic")],
                  key=lambda a: a["head_recall8"])
    y = np.arange(len(arms))
    cols = [C_DEPLOY if a["source"] == "transfer" and
            a["arm"].startswith(("KNN", "NN__")) else
            ("#8c564b" if a["arm"].startswith("PCA_RIDGE") else
             ("#2ca02c" if a["arm"] == "GLOBAL" else "#bbbbbb"))
            for a in arms]
    ax.barh(y, [a["head_recall8"] for a in arms], color=cols, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels([a["arm"] for a in arms], fontsize=7.4)
    ax.axvline(trained_r8, color=C_SHARED, lw=2, label=f"S2-C3 trained ({trained_r8:.3f})")
    ax.axvline(global_r8, color="#2ca02c", lw=1.6, ls="--",
               label=f"GLOBAL / no adaptation ({global_r8:.3f})")
    ax.axvline(self_r8, color=C_ORACLE, lw=2, ls=":", label=f"SELF oracle ({self_r8:.3f})")
    ax.set_xlabel("head_recall@8 (held-out 150)", fontsize=9)
    ax.set_title("B  no deployable arm reaches the trained shared\n"
                 "direction, let alone the oracle", fontsize=10)
    ax.legend(fontsize=7.6, loc="lower right")
    ax.grid(alpha=0.25, axis="x")
    ax.set_xlim(0.55, 1.0)

    # ------------------------------------------------------------- panel C
    ax = fig.add_subplot(gs[0, 2])
    cum = Zg["pca_eigenvalues"].cumsum() / Zg["pca_eigenvalues"].sum()
    k = np.arange(1, len(cum) + 1)
    ax.plot(k, cum, color="#333333", lw=1.8, label="variance retained")
    ok = {int(a["arm"].split("_k")[1]): a["head_recall8"] / self_r8
          for a in P["arms"] if a["arm"].startswith("PCA_ORACLE_k")}
    ax.plot(sorted(ok), [ok[i] for i in sorted(ok)], marker="s", ms=6,
            color=C_ORACLE, lw=1.8, label="recall of SELF retained by the\n"
                                          "k-component oracle reconstruction")
    for t in (0.80, 0.95):
        ax.axhline(t, color="#999999", ls=":", lw=1)
        ax.text(len(cum) * 0.55, t + 0.012, f"{int(t*100)} %", fontsize=7.5,
                color="#777777")
    ax.set_xscale("log")
    ax.set_xlabel("number of fit-set PCA components (log)", fontsize=9)
    ax.set_ylabel("fraction", fontsize=9)
    eff = D["rule_inputs"]["pca_effective_rank"]
    d80 = D["rule_inputs"]["pca_dims_for_80pct"]
    ax.set_title(f"C  the directions are not low-dimensional:\n"
                 f"effective rank {eff:.1f}/240, {d80} components for 80 % variance",
                 fontsize=10)
    ax.legend(fontsize=7.4, loc="lower right")
    ax.grid(alpha=0.25, which="both")

    # ------------------------------------------------------------- panel D
    ax = fig.add_subplot(gs[1, 0])
    ks = sorted(k for k in ok)
    ax.plot(ks, [ok[i] * self_r8 for i in ks], marker="o", ms=6, color=C_ORACLE,
            lw=1.8, label="PCA_ORACLE: true coefficients (upper bound)")
    for dname, style in zip(("mean", "meanstd+delta"), ("--", "-")):
        curve = P["ridge_curves"][dname]
        ys = [np.mean(curve.get(str(i), curve.get(i))["test"]["head_recall8"])
              for i in ks]
        ax.plot(ks, ys, marker="^", ms=5, color="#8c564b", ls=style,
                lw=1.6, label=f"PCA_RIDGE from {dname} descriptor")
    ax.axhline(self_r8, color=C_ORACLE, ls=":", lw=1.6, label=f"SELF oracle ({self_r8:.3f})")
    ax.axhline(trained_r8, color=C_SHARED, ls="-.", lw=1.8,
               label=f"S2-C3 trained shared ({trained_r8:.3f})")
    ax.axhline(global_r8, color="#2ca02c", ls="--", lw=1.4,
               label=f"GLOBAL / no adaptation ({global_r8:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("number of PCA components k (log)", fontsize=9)
    ax.set_ylabel("head_recall@8", fontsize=9)
    ax.set_title("D  even the optimistic low-dimensional reconstruction\n"
                 "stops at 0.83, and the predictable part at 0.66", fontsize=10)
    ax.legend(fontsize=7.2, loc="lower right")
    ax.grid(alpha=0.25, which="both")

    # ------------------------------------------------------------- panel E
    ax = fig.add_subplot(gs[1, 1])
    pm = np.array(T["transfer_distribution"]["per_image_mean"])
    pb = np.array(T["transfer_distribution"]["per_image_best"])
    bins = np.linspace(0, 1, 41)
    ax.hist(pm, bins=bins, alpha=0.75, color="#1f77b4",
            label=f"mean over the 240 candidates (median {np.median(pm):.2f})")
    ax.hist(pb, bins=bins, alpha=0.55, color="#ff7f0e",
            label=f"best of the 240 candidates (median {np.median(pb):.2f})")
    ax.axvline(global_r8, color="#2ca02c", ls="--", lw=1.8,
               label=f"GLOBAL direction ({global_r8:.3f})")
    ax.axvline(dep_r8, color=C_DEPLOY, lw=1.8,
               label=f"best deployable ({dep_r8:.3f})")
    ax.set_xlabel("head_recall@8, per held-out image", fontsize=9)
    ax.set_ylabel("images (of 150)", fontsize=9)
    ax.set_title("E  the right direction exists for almost every image\n"
                 "(95 % have some candidate above 0.85) but is not identifiable",
                 fontsize=10)
    ax.legend(fontsize=7.4)
    ax.grid(alpha=0.25)

    # ------------------------------------------------------------- panel F
    ax = fig.add_subplot(gs[1, 2])
    off = Zg["cos_offdiag_fit"]
    ax.hist(off, bins=80, color="#cfcfcf", edgecolor="#8f8f8f", lw=0.3,
            label=f"all fit pairs (mean {off.mean():.3f})")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("cosine similarity between two images' oracle directions", fontsize=9)
    ax.set_ylabel("pairs", fontsize=9)
    g = D["geometry"]
    ax.set_title(f"F  direction geometry: mean cosine {g['cosine_mean_within']:.2f} "
                 f"(within benchmark {g['cosine_within_benchmark']:.2f},\n"
                 f"across {g['cosine_across_benchmark']:.2f}); "
                 f"PC1 carries only {g['evr_top5'][0]*100:.1f} % of the variance",
                 fontsize=10)
    ax.legend(fontsize=7.6)
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"S2-C4  per-image direction transfer — {D['verdict']}: the oracle is "
        f"strong (0.95 vs 0.71 unadapted) but its variation is not recoverable "
        f"from cheap forward information", fontsize=12.5, y=0.975)
    path = os.path.join(OUT, "figures", args.out)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
