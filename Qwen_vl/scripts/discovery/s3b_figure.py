"""
S3-B figure: the bundle evidence in one view.

A  prefix rescue rate vs k (both banks, per benchmark)  -- how deep you must go
B  the intervention ladder: hit rate by arm family       -- what "the bundle"
   actually buys over its substitutes
C  minimal rescue-group sizes k*                          -- the bundle inventory
D  within-instance structure AUC (T1) + matched-k prefix
   AUC (T3)                                               -- is it predictable?
E  gold-answer NLL along the prefix depth (both banks)
F  the crop probe: is a bundle a legible piece of evidence?
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                        # noqa: E402
import s3b_common as B                                          # noqa: E402
from s3b_structure import CANDIDATE_FEATURES                   # noqa: E402

DS_COL = {"TextVQA_VAL": "#1f77b4", "DocVQA_VAL": "#2ca02c",
          "OCRBench": "#d62728"}
LADDER = ["bundle", "loo", "substitute", "shift", "blockperm", "null_window",
          "null_uni", "null_out", "single", "removal_control"]
LADDER_LBL = {"bundle": "full bundle G", "loo": "G minus one member",
              "substitute": "one member swapped (same depth)",
              "shift": "window slid 1-4 ranks",
              "blockperm": "tail members swapped",
              "null_window": "k* tokens, same window (0 kept)",
              "null_uni": "k* tokens, whole missed pool",
              "null_out": "k* tokens, outside S+T",
              "single": "one member alone",
              "removal_control": "G, random removals"}


def main():
    ana = json.load(open(os.path.join(OUT, "s3b_analysis.json")))
    sweep = json.load(open(os.path.join(OUT, "s3b_sweep.json")))
    base = json.load(open(os.path.join(OUT, "s3b_base.json")))

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.2))
    fig.suptitle("S3-B — are the teacher's accuracy-critical tokens small bundles?",
                 fontsize=13, fontweight="bold")

    # ---------------- A: cumulative rescue depth --------------------------
    a = ax[0, 0]
    for bank, ls, lab in (("G", "-", "bank G (GDEP)"),
                          ("L", "--", "bank L (LIN_L4)")):
        d = ana["partA"][bank]
        n_tot = max(1, (d["n_base_wrong"] if bank == "G" else d["n_class"]))
        for ds, blk in sorted(d["per_ds"].items()):
            ys = [sum(1 for k in blk["kstars"] if k <= k_) / max(1, blk["n"])
                  for k_ in B.KS_SWEEP]
            a.plot(B.KS_SWEEP, ys, ls, color=DS_COL[ds], alpha=0.6, lw=1.3,
                   label=f"{lab} {ds.replace('_VAL', '')} (n={blk['n']})")
        ys = [sum(1 for k in d["kstars"] if k <= k_) / n_tot
              for k_ in B.KS_SWEEP]
        a.plot(B.KS_SWEEP, ys, "o" if bank == "G" else "s", color="k",
               lw=1.8, ms=4.5,
               label=f"bank {bank} pooled (of {n_tot} in class)")
    a.set_xscale("log")
    a.set_xticks(list(B.KS_SWEEP))
    a.set_xticklabels([str(k) for k in B.KS_SWEEP], fontsize=7, rotation=45)
    a.set_xlabel("k = teacher-missed tokens swapped in")
    a.set_ylabel("cumulative share rescued by k")
    a.set_title("A  how deep the teacher queue must go", fontsize=10)
    a.grid(alpha=0.3)
    a.legend(fontsize=5.8, ncol=2)

    # ---------------- B: the ladder ---------------------------------------
    b = ax[0, 1]
    rates = dict(ana["partB"]["family_hit_rates"])
    ys = np.arange(len(LADDER))
    for i, (bank, off) in enumerate((("G", -0.2), ("L", 0.2))):
        vals, los, his = [], [], []
        for f in LADDER:
            r = rates.get(f"{bank}|{f}")
            if not r or f == "substitute":
                if f == "substitute" and r:
                    vals.append(r["rate"]); los.append(np.nan); his.append(np.nan)
                else:
                    vals.append(np.nan); los.append(np.nan); his.append(np.nan)
                continue
            if not r:
                vals.append(np.nan); los.append(np.nan); his.append(np.nan)
                continue
            vals.append(r["rate"])
            los.append(r["rate"] - r["ci"][0])
            his.append(r["ci"][1] - r["rate"])
        b.barh(ys + off, vals, height=0.36, xerr=[los, his],
               error_kw=dict(lw=0.8, capsize=2),
               color=["#1f77b4", "#d62728"][i], alpha=0.8,
               label=f"bank {bank}")
    b.set_yticks(ys)
    b.set_yticklabels([LADDER_LBL[f] for f in LADDER], fontsize=7.5)
    b.invert_yaxis()
    b.axvline(1.0, color="k", lw=0.8, ls=":")
    b.set_xlabel("P(answer correct)")
    b.set_title("B  what the exact bundle buys over its substitutes",
                fontsize=10)
    b.legend(fontsize=8)
    b.grid(alpha=0.3, axis="x")

    # ---------------- C: k* inventory -------------------------------------
    c = ax[0, 2]
    for i, bank in enumerate(("G", "L")):
        ks = ana["partA"][bank]["kstars"]
        bins = np.arange(0.5, 36.5, 1)
        c.hist([k for k in ks if k <= 32], bins=bins, alpha=0.55 + 0.2 * i,
               color=("#1f77b4", "#d62728")[i],
               label=f"bank {bank}  n={len(ks)}"
                     f"{' (k*<=32 shown)' if max(ks) > 32 else ''}, median "
                     f"{np.median(ks) if ks else float('nan'):.1f}")
    c.axvline(4, color="k", ls=":", lw=1)
    c.set_xlabel("minimal rescue group size k*")
    c.set_ylabel("instances")
    pB = ana["partB"]
    t4 = ana.get("T4_depth_from_shape", {}).get("16", {}).get("top", [])
    rho = {f: e["rho"] for f, e in t4}
    c.set_title("C  bundle sizes $k^*$: group-only "
                f"{pB['group_only_rate']:.2f}, monadic {pB['monadic_rate']:.2f}"
                "\n"
                "a fixed top-16 probe predicts the depth needed"
                f" ($\\rho$ vs $k^*$: redundancy "
                f"{rho.get('ft_sim_sel_S', float('nan')):+.2f}, "
                f"edge {rho.get('ct_edge_mean', float('nan')):+.2f}$)",
                fontsize=7.4)
    c.legend(fontsize=8)
    c.grid(alpha=0.3)

    # ---------------- D: structure AUCs -----------------------------------
    d = ax[1, 0]
    t1 = [(f, v) for f, v in ana["T1_null_vs_null"]["rows"]
          if f in CANDIDATE_FEATURES
          and np.isfinite(v["mean_auc_oriented"])][:12]
    lbl = [f for f, _ in t1][::-1]
    yy = np.arange(len(lbl))
    mm = np.array([dict(t1)[f]["mean_auc_oriented"] for f in lbl])
    lo = np.array([dict(t1)[f]["auc_oriented_ci"][0] for f in lbl])
    hi = np.array([dict(t1)[f]["auc_oriented_ci"][1] for f in lbl])
    d.errorbar(mm, yy, xerr=[mm - lo, hi - mm], fmt="o", ms=5, lw=1,
               capsize=2, color="#2c7fb8", label="T1 within-instance\nnull vs null")
    t3pts = []
    for k, blk in sorted(ana["T3_prefix_hit_vs_miss"].items(), key=lambda x: int(x[0])):
        for f, e in blk["top"][:3]:
            if np.isfinite(e["loo_auc"]):
                t3pts.append((f, e["loo_auc"]))
    t3d = dict(t3pts)
    m3 = [t3d.get(f, np.nan) for f in lbl]
    ok = np.isfinite(m3)
    d.errorbar(np.array(m3)[ok], yy[ok], fmt="s", ms=5, color="#d95f0e",
               ls="none", label="T3 matched-k prefix\nout-of-fold (LOO-inst) AUC")
    d.axvline(0.5, color="k", lw=0.8)
    d.axvline(0.65, color="r", ls="--", lw=1)
    d.text(0.655, len(lbl) - 0.5, "clause-A bar", color="r", fontsize=7)
    d.set_yticks(yy)
    d.set_yticklabels(lbl, fontsize=7)
    d.set_xlim(0.35, 0.9)
    d.set_xlabel("oriented AUC (0.5 = no structure)")
    d.set_title("D  is bundle success predictable?\n"
                "(T1 within-instance is weak; T3 across-instance is strong)",
                fontsize=9)
    d.legend(fontsize=7, loc="lower right")
    d.grid(alpha=0.3, axis="x")

    # ---------------- E: NLL dose -----------------------------------------
    e = ax[1, 1]
    for bank, col in (("G", "#1f77b4"), ("L", "#d62728")):
        xs, ys = [], []
        for k in B.KS_SWEEP:
            if bank == "G":
                v = ana["partA"]["G"]["curve"][str(k)]
                if v["mean_dL"] is not None:
                    xs.append(k); ys.append(v["mean_dL"])
                continue
            src = sweep["L"]
            dl = [x["L0"] - x["curve_NLL"][str(k)] for x in src.values()
                  if str(k) in x["curve_NLL"]]
            if dl:
                xs.append(k); ys.append(float(np.mean(dl)))
        e.plot(xs, ys, "o-", color=col, label=f"bank {bank}")
    fl = [v["floor"] for v in base["recs"].values()]
    e.axhline(float(np.mean(fl)), color="gray", ls=":",
              label=f"mean bf16 floor {np.mean(fl):.2f} nats")
    e.set_xscale("log")
    e.set_xticks(list(B.KS_SWEEP))
    e.set_xticklabels([str(k) for k in B.KS_SWEEP], fontsize=7, rotation=45)
    e.set_xlabel("k")
    e.set_ylabel("mean gold-NLL drop (nats)")
    e.set_title("E  loss vs accuracy: both graded in k", fontsize=10)
    e.legend(fontsize=8)
    e.grid(alpha=0.3)

    # ---------------- F: crop probe ---------------------------------------
    f = ax[1, 2]
    if "crop" in ana:
        names = ["whole", "bundle", "null", "matched"]
        vals = [ana["crop"][n]["rate"] or 0 for n in names]
        err = [[(ana["crop"][n]["rate"] - ana["crop"][n]["ci"][0]) for n in names],
               [(ana["crop"][n]["ci"][1] - ana["crop"][n]["rate"]) for n in names]]
        f.bar(names, vals, yerr=err, capsize=3,
              color=["#7f7f7f", "#1f77b4", "#ff9896", "#aec7e8"])
        f.set_ylabel("P(answer correct | crop only)")
        bn = ana["crop"]["bundle_only_vs_null_only"]
        f.set_title(f"F  crop probe: bundle box "
                    f"{ana['crop']['bundle']['rate']:.2f} vs same-size random "
                    f"box {ana['crop']['matched']['rate']:.2f} "
                    f"(paired {bn['bundle_not_null']}/{bn['null_not_bundle']})",
                    fontsize=8.6)
        f.grid(alpha=0.3, axis="y")
    else:
        f.text(0.5, 0.5, "crop probe unavailable", ha="center")
        f.set_axis_off()
    path = os.path.join(OUT, "figures", "s3b_bundles.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
