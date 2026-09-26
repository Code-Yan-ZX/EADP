"""
M5 (Safe Removal) step 5 -- figures.

Four panels, each answering one of the brief's questions and nothing else:

    m5_trim_curve.png   macro against tokens retained, per family, with the
                        paired CI drawn -- the brief's §9 curve, and the one
                        place the "sweet spot" claim is visible or absent.
    m5_learning.png     the same curve for SAFE and for MAXRED, plus the
                        paired per-instance delta between them: what the
                        supervision is worth (brief §12).
    m5_probe_gate.png   the early gate (brief §3): AUROC for each arm, the
                        safe-precision of each arm's bottom-8, and the
                        critical-head formulation on the same features.
    m5_deletion.png     the teacher itself: d_i by sampling stratum, the
                        per-instance noise floor, and d against the three
                        features the baselines rank by.

Reads stored artefacts only (no GPU).

Usage
    python scripts/discovery/m5_figures.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUT_DIR                                    # noqa: E402
from m2_accuracy import DS_ORDER                                  # noqa: E402
from m5_accuracy import K_GRID, RULE_FAMILIES                     # noqa: E402
from m5_common import FEATURES, IMP_COL, RECON_COL, RED_COL       # noqa: E402

FIG = os.path.join(OUTPUT_DIR, "figures")
COLORS = dict(SAFE="#c0392b", MAXRED="#2471a3", LOWIMP="#1e8449",
              RECON="#b9770e", RANDOM="#7d3c98")


def load(tag):
    with open(os.path.join(OUTPUT_DIR, f"{tag}.json")) as f:
        return json.load(f)


def _curve(rows, fam):
    ks, ms, los, his = [], [], [], []
    for k in K_GRID:
        r = rows.get(f"{fam}-k{k}")
        if not r or "hits" not in r:
            continue
        v = r.get("vs_B2", {})
        ks.append(256 - k)
        ms.append(r["macro"])
        ci = v.get("ci", [0, 0])
        los.append(r["macro"] + ci[0] - v.get("delta", 0))
        his.append(r["macro"] + ci[1] - v.get("delta", 0))
    return np.array(ks), np.array(ms), np.array(los), np.array(his)


def fig_trim_curve(rows, rep):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    b2 = rows["B2"]["macro"]
    ax.axhline(b2, color="#555555", ls="--", lw=1.2,
               label=f"B2 (256 tok) = {b2:.2f}")
    ax.axhline(rows["B1"]["macro"], color="#999999", ls=":", lw=1.2,
               label=f"B1 EADP facility = {rows['B1']['macro']:.2f}")
    for fam in ("SAFE",) + RULE_FAMILIES:
        t, m, lo, hi = _curve(rows, fam)
        if not len(t):
            continue
        lw = 2.4 if fam == "SAFE" else 1.4
        ax.plot(t, m, "-o", color=COLORS[fam], lw=lw, ms=6, label=fam)
        ax.fill_between(t, lo, hi, color=COLORS[fam], alpha=0.12, lw=0)
        p = rep["peaks"].get(fam)
        if p:
            ax.annotate(f"{p['macro']:.1f}", (p["tokens"], p["macro"]),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, color=COLORS[fam])
    ax.set_xlabel("visual tokens retained (256 - k)")
    ax.set_ylabel("held-out 150 macro")
    ax.set_title("M5 Safe Removal -- accuracy against retained size\n"
                 "bands are paired bootstrap CIs against B2", fontsize=10)
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p = os.path.join(FIG, "m5_trim_curve.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[fig] {p}")


def fig_learning(rows, rep):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for fam in ("SAFE", "MAXRED", "LOWIMP", "RECON"):
        t, m, lo, hi = _curve(rows, fam)
        if not len(t):
            continue
        ax.plot(t, m, "-o", color=COLORS[fam], ms=5,
                lw=2.4 if fam == "SAFE" else 1.3, label=fam)
        ax.fill_between(t, lo, hi, color=COLORS[fam], alpha=0.10, lw=0)
    ax.axhline(rows["B2"]["macro"], color="#555", ls="--", lw=1.1)
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_xlabel("tokens retained")
    ax.set_ylabel("macro")
    ax.set_title("SAFE vs the training-free rules", fontsize=10)

    ax = axes[1]
    names, deltas, errs = [], [], []
    for fam in RULE_FAMILIES:
        key = f"SAFE-k8 - {fam}-k8"
        c = rep["learning_contrast"].get(key)
        if not c:
            continue
        names.append(f"vs {fam}")
        deltas.append(c["delta"])
        errs.append([c["delta"] - c["ci"][0], c["ci"][1] - c["delta"]])
    if names:
        y = np.arange(len(names))
        errs = np.array(errs).T
        ax.barh(y, deltas, color=COLORS["SAFE"], alpha=0.8)
        ax.errorbar(deltas, y, xerr=errs, fmt="none", ecolor="#222", capsize=3)
        ax.set_yticks(y, names)
        ax.axvline(0, color="#222", lw=1)
        ax.set_xlabel("SAFE-k8 minus rule (macro points)")
        ax.set_title("what the supervision is worth at k=8", fontsize=10)
        ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    p = os.path.join(FIG, "m5_learning.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[fig] {p}")


def fig_probe_gate(probe):
    b = probe["by_split"]["val"]
    arms = [a for a in ("rule:random", "rule:maxred", "rule:lowimp",
                        "rule:recon", "probe:lr", "probe:mlp", "probe:vis")
            if a in b]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    x = np.arange(len(arms))

    ax = axes[0]
    ax.bar(x, [b[a]["auroc_harmful_mean"] for a in arms],
           yerr=[b[a]["auroc_harmful_sd"] for a in arms],
           color=["#95a5a6"] * 4 + ["#c0392b"] * 3, capsize=3)
    ax.axhline(0.5, color="#222", ls="--", lw=1)
    ax.set_xticks(x, [a.split(":")[-1] for a in arms], rotation=20)
    ax.set_ylabel("AUROC (deletion harmful)")
    ax.set_title("val: can removal risk be predicted?", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    ax.bar(x, [b[a]["bottom_k8_safe_precision"] for a in arms],
           color=["#95a5a6"] * 4 + ["#c0392b"] * 3)
    ax.set_xticks(x, [a.split(":")[-1] for a in arms], rotation=20)
    ax.set_ylabel("P(true d <= eps) among the bottom-8")
    ax.set_title("val: precision of the 8 the arm would delete", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    hc = probe["head"]
    ks = sorted({hc[k]["K"] for k in hc})
    w = 0.35
    for j, split in enumerate(("fit", "val")):
        vals = [hc[f"head_K{K}_{split}"]["auroc"] for K in ks]
        ax.bar(np.arange(len(ks)) + (j - 0.5) * w, vals, w, label=split,
               color=["#7f8c8d", "#2c3e50"][j])
    ax.axhline(0.5, color="#222", ls="--", lw=1)
    ax.set_xticks(np.arange(len(ks)), [f"top-{K}" for K in ks])
    ax.set_ylabel("AUROC")
    ax.set_title("the critical-head formulation,\nsame features / same probe",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    p = os.path.join(FIG, "m5_probe_gate.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[fig] {p}")


def fig_deletion(teacher, probe):
    meas = teacher["meas"]
    keys = [k for k in teacher["done_keys"] if k in meas]
    strata = [s["stratum"] for k in keys for s in meas[k]["singles"]]
    d = np.array([s["d"] for k in keys for s in meas[k]["singles"]])
    eps = probe["thresholds"]["eps"]
    delta = probe["thresholds"]["delta"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    order = ["rand", "minred", "maxred", "highimp", "lowimp", "recon"]
    order = [s for s in order if s in set(strata)]
    data = [d[np.array(strata) == s] for s in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.axhline(0, color="#222", lw=1)
    ax.axhline(eps, color="#c0392b", ls="--", lw=1, label=f"eps={eps:.3f}")
    ax.axhline(delta, color="#8e44ad", ls=":", lw=1, label=f"delta={delta:.3f}")
    ax.set_ylabel("d_i = L(S0\\{i}) - L(S0)   [nats]")
    ax.set_title("deletion harm by sampling stratum", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    fm = np.array([meas[k]["floor_mean"] for k in keys])
    fx = np.array([meas[k]["floor_max"] for k in keys])
    bins = np.linspace(0, max(np.percentile(fx, 99), 1e-6), 40)
    ax.hist(fx, bins=bins, alpha=0.55, label="floor_max (per gold)",
            color="#95a5a6")
    ax.hist(fm, bins=bins, alpha=0.75, label="floor_mean (the one used)",
            color="#c0392b")
    ax.set_yscale("log")
    ax.set_xlabel("nats")
    ax.set_ylabel("instances")
    ax.set_title("the two noise floors are not the same quantity", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[2]
    X = np.array([[s["red"], s["imp"], s["recon"]] for k in keys
                  for s in meas[k]["singles"]])
    names = ["red_s0", "imp", "nn4_recon"]
    for j, nm in enumerate(names):
        ax.scatter(X[:, j], d, s=2, alpha=0.18, label=nm)
        ok = np.isfinite(X[:, j]) & np.isfinite(d)
        r = np.corrcoef(X[ok, j], d[ok])[0, 1]
        print(f"[corr] {nm:10s} pearson vs d = {r:+.3f}")
    ax.axhline(0, color="#222", lw=1)
    ax.set_xlabel("feature value")
    ax.set_ylabel("d_i")
    ax.set_title("the three features the rules rank by", fontsize=10)
    ax.legend(fontsize=8, markerscale=5)
    fig.tight_layout()
    p = os.path.join(FIG, "m5_deletion.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[fig] {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m5_analysis")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    rep = load(args.tag)
    m5 = load("m5_accuracy")
    m2 = load("m2_accuracy")
    from m5_analyze import rows_of, add_contrasts
    rows = rows_of(m5, m2)
    add_contrasts(rows)
    fig_trim_curve(rows, rep)
    fig_learning(rows, rep)
    if os.path.exists(os.path.join(OUTPUT_DIR, "m5_probe.json")):
        fig_probe_gate(load("m5_probe"))
    if os.path.exists(os.path.join(OUTPUT_DIR, "m5_teacher.json")):
        fig_deletion(load("m5_teacher"), load("m5_probe"))


if __name__ == "__main__":
    main()
