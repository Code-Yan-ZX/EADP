"""
S3-A figure: the conditionality evidence in one view.

A  example delta matrices (12 candidates x 5 contexts), one instance per ds
B  delta(base) vs delta(weak) scatter, per benchmark, with the noise band
C  interaction variance share per benchmark with bootstrap CIs
D  rescue summary: dL and rescue rate per arm at k=16, per benchmark
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                   # noqa: E402
import s3a_common as C                                     # noqa: E402

CTX = ["base", "weak", "strong", "rand", "rand2"]


def main():
    cases = json.load(open(os.path.join(OUT, C.CASES_JSON)))["cases"]
    nll = json.load(open(os.path.join(OUT, "s3a_nll.json")))
    ana = json.load(open(os.path.join(OUT, "s3a_analysis.json")))
    rescue = (json.load(open(os.path.join(OUT, "s3a_rescue.json")))["runs"]
              if os.path.exists(os.path.join(OUT, "s3a_rescue.json")) else {})

    from s3a_analysis import per_instance_matrices
    mats = per_instance_matrices(nll, cases)
    by_key = {c["key"]: c for c in cases}

    fig, ax = plt.subplots(2, 3, figsize=(15, 9),
                           gridspec_kw=dict(width_ratios=[1.1, 1.1, 1.1,
                                                           1.1, 1.4, 1.4]))
    # --- A: three example heatmaps ------------------------------------------
    picks = []
    for ds in C.DS_ALL:
        cand = [(k, m) for k, m in mats.items() if m["ds"] == ds]
        cand.sort(key=lambda km: -(np.nanmax(np.nanmax(km[1]["M"], 1) -
                                             np.nanmin(km[1]["M"], 1))))
        picks.append(cand[0])
    for j, (key, m) in enumerate(picks):
        a = ax[0, j]
        im = a.imshow(m["M"], aspect="auto", cmap="RdBu_r",
                      vmin=-np.nanmax(np.abs(m["M"])),
                      vmax=np.nanmax(np.abs(m["M"])))
        a.set_xticks(range(len(CTX)), CTX, rotation=45, ha="right",
                     fontsize=8)
        c = by_key[key]
        a.set_yticks(range(len(m["P"])),
                     [f"{t}({c['grank'][str(t)]})" for t in m["P"]],
                     fontsize=6)
        a.set_title(f"{key} [{c['stratum']}]", fontsize=9)
        plt.colorbar(im, ax=a, fraction=0.046)

    # --- B: base vs weak scatter --------------------------------------------
    a = ax[1, 0]
    colors = {"TextVQA_VAL": "tab:blue", "DocVQA_VAL": "tab:green",
              "OCRBench": "tab:red"}
    for key, m in mats.items():
        c = by_key[key]
        a.scatter(m["M"][:, 0], m["M"][:, 1], s=10,
                  c=colors[m["ds"]], alpha=0.6)
    lim = max(np.nanmax(np.abs(m["M"])) for m in mats.values()) * 1.1
    a.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    a.set_xlim(-lim, lim)
    a.set_ylim(-lim, lim)
    a.axhline(0, lw=0.5, c="gray")
    a.axvline(0, lw=0.5, c="gray")
    a.set_xlabel("delta(i | S_base)")
    a.set_ylabel("delta(i | S_weak)")
    a.set_title("B  marginal utility moves with S\n(blue TVQA, green DocVQA, "
                "red OCRBench)", fontsize=9)

    # --- C: interaction share -----------------------------------------------
    a = ax[1, 1]
    xs = np.arange(len(C.DS_ALL))
    vals = [ana["part_a"][ds]["interaction_share"]["mean"] for ds in C.DS_ALL]
    lo = [v - ana["part_a"][ds]["interaction_share"]["lo"]
          for ds, v in zip(C.DS_ALL, vals)]
    hi = [ana["part_a"][ds]["interaction_share"]["hi"] - v
          for ds, v in zip(C.DS_ALL, vals)]
    a.bar(xs, vals, yerr=[lo, hi], color=[colors[ds] for ds in C.DS_ALL],
          alpha=0.8)
    a.set_xticks(xs, ["TextVQA", "DocVQA", "OCRBench"], fontsize=8)
    a.set_ylabel("interaction share of Var(delta)")
    a.set_title("C  token x context interaction", fontsize=9)

    # --- D/E: rescue summary (if available) ---------------------------------
    for col, metric in ((2, "dL"), (None, None)):
        if col is None:
            break
        a = ax[1, col]
        if not rescue:
            a.axis("off")
            a.text(0.5, 0.5, "D/E  rescue pending", ha="center")
            break
        arms = ["unary_k16", "cond_k16", "spatial_k16", "random_k16"]
        labels = ["unary", "cond", "spatial", "random"]
        w = 0.22
        for i, ds in enumerate(C.DS_ALL):
            keys = [k for k, v in rescue.items() if v["ds"] == ds]
            mv = []
            for arm in arms:
                vals_ = [rescue[k][arm][metric] for k in keys
                         if arm in rescue[k]]
                mv.append(np.mean(vals_) if vals_ else 0.0)
            a.bar(np.arange(len(arms)) + (i - 1) * w, mv, width=w,
                  color=colors[ds], label=ds.split("_")[0], alpha=0.85)
        a.set_xticks(np.arange(len(arms)), labels, fontsize=8)
        a.set_ylabel("mean dL (nats)")
        a.set_title("D  rescue at k=16", fontsize=9)
        a.legend(fontsize=7)

    fig.suptitle("S3-A  conditional utility of visual tokens "
                 "(delta = L(y|S) - L(y|S+i-r), pre-LLM delivery)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    path = os.path.join(OUT, "figures", "s3a_conditional_utility.png")
    fig.savefig(path, dpi=160)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
