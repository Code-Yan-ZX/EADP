"""Stage-1 Cross-Stream Pilot — analysis (protocol §6).

Six-arm DEV table, pre-registered contrasts with paired image-cluster
bootstrap (20,000 resamples, seed 20261002), 0.5-point non-inferiority
screening rule, and Stage-1 diagnostics (importance distributions, per-stream
residuals / zero ratios, anchor overlap with EADP, group sizes).

Usage: python xsp_analyze.py
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import amp_common as AC
import amp_analyze as AN
import xsp_common as XC

ACC_DIR = os.path.join(XC.OUT_DIR, "acc")


def load_scores(arms):
    out = {}
    for arm in arms:
        for ds in XC.DS_LIST:
            p = os.path.join(ACC_DIR, arm, f"{ds}_score.json")
            if not os.path.exists(p):
                continue
            s = json.load(open(p))
            if s.get("per_question"):
                out[(arm, ds)] = s
    return out


def acc100(s):
    return 100.0 * float(np.mean([float(v) for v in s["per_question"].values()]))


def macro_of(scores, arm):
    vals = [acc100(scores[(arm, ds)]) for ds in XC.DS_LIST
            if (arm, ds) in scores]
    return float(np.mean(vals)) if len(vals) == len(XC.DS_LIST) else None


def bootstrap(scores, manifest, arm_a, arm_b, n_boot=XC.N_BOOT,
              seed=XC.BOOT_SEED):
    c = AN.paired_macro_bootstrap(scores, manifest, "dev", arm_a, arm_b,
                                  n_boot=n_boot, seed=seed)
    if c is None:
        return None
    c["margin"] = XC.MARGIN
    c["rule"] = ("new - old; CI_lo > -0.5 -> non_inferior_supported; "
                 "CI_hi < -0.5 -> exceeds_tolerance_loss; else uncertain")
    lo, hi = c["ci"][0], c["ci"][1]
    if lo > -XC.MARGIN:
        verdict = "non_inferior_supported"
    elif hi < -XC.MARGIN:
        verdict = "exceeds_tolerance_loss"
    else:
        verdict = "uncertain"
    c["verdict"] = verdict
    c["direction"] = ("positive" if lo > 0
                      else ("negative" if hi < 0 else "ci_crosses_zero"))
    return c


def rescue_break(scores, arm_a, arm_b):
    return AN.rescue_break(scores, arm_a, arm_b)


# ------------------------------------------------------------ diagnostics --
def bank_diag(arm):
    scorer = XC.ARMS[arm]["scorer"]
    out = {}
    for ds in XC.DS_LIST:
        bank = XC.load_bank(scorer, ds)
        ws, gs = [], []
        zero = [0, 0, 0]
        dmean = [[], [], []]
        n = 0
        for rec in bank["samples"].values():
            if not rec.get("gsize"):
                continue
            n += 1
            gs += rec["gsize"]
            if rec.get("w"):
                ws += rec["w"]
            for i, z in enumerate(rec.get("zero_norm", []) or []):
                zero[i] += z
            for i, dm in enumerate(rec.get("d_mean", []) or []):
                dmean[i].append(dm)
        if n == 0:
            continue
        w = np.array(ws) if ws else None
        g = np.array(gs)
        d = dict(n_samples=n,
                 group_size_mean=float(g.mean()),
                 group_size_p50=float(np.median(g)),
                 group_size_p90=float(np.percentile(g, 90)),
                 group_size_max=int(g.max()),
                 singleton_frac=float((g == 0).mean()))
        if w is not None:
            d["importance"] = dict(
                mean=float(w.mean()), p50=float(np.percentile(w, 50)),
                p90=float(np.percentile(w, 90)), p99=float(np.percentile(w, 99)),
                max=float(w.max()), min=float(w.min()))
        d["zero_norm_counts"] = zero
        d["d_mean_per_stream"] = [float(np.mean(x)) if x else None
                                  for x in dmean]
        out[ds] = d
    return out


def anchor_overlap(manifest):
    """Anchor overlap of each scorer's keep with the EADP keep, per dataset."""
    out = {}
    eadp_banks = {ds: XC.load_bank("eadp", ds) for ds in XC.DS_LIST}
    for arm in XC.ARMS:
        scorer = XC.ARMS[arm]["scorer"]
        row = {}
        for ds in XC.DS_LIST:
            if scorer == "eadp":
                row[ds] = dict(mean_inter=XC.K, mean_jaccard=1.0)
                continue
            bank = XC.load_bank(scorer, ds)
            inter, jac = [], []
            for it in manifest["datasets"][ds]["dev"]:
                key = str(it["idx"])
                a = set(eadp_banks[ds]["samples"][key]["keep"])
                b = set(bank["samples"][key]["keep"])
                if not a or not b:
                    continue
                inter.append(len(a & b))
                jac.append(len(a & b) / len(a | b))
            row[ds] = dict(mean_inter=float(np.mean(inter)),
                           mean_jaccard=float(np.mean(jac)),
                           n=len(inter))
        out[arm] = row
    return out


def scorer_changes_anchors(manifest):
    """Check the new scorers actually change the anchors (not just a heatmap)."""
    out = {}
    for ds in XC.DS_LIST:
        eadp = XC.load_bank("eadp", ds)
        for scorer in ("cross_stream", "uniform", "main_residual"):
            bank = XC.load_bank(scorer, ds)
            diff = [int(eadp["samples"][str(it["idx"])]["keep"]
                        != bank["samples"][str(it["idx"])]["keep"])
                    for it in manifest["datasets"][ds]["dev"]]
            out[f"{ds}/{scorer}"] = dict(
                n_changed=int(sum(diff)), n=len(diff))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis_dev.json")
    args = ap.parse_args()
    arms = list(XC.ARMS)
    manifest = json.load(open(os.path.join(AC.OUT_DIR, "manifest.json")))
    scores = load_scores(arms)

    table = {}
    for arm in arms:
        per = {ds: (acc100(scores[(arm, ds)]) if (arm, ds) in scores else None)
               for ds in XC.DS_LIST}
        table[arm] = dict(per_ds=per, macro=macro_of(scores, arm),
                          scorer=XC.ARMS[arm]["scorer"], lam=XC.ARMS[arm]["lam"])

    # pre-registered contrasts (protocol §6)
    pairs = [
        ("X_MAIN025", "E_MAIN025"),          # primary
        ("X_GATHER", "E_GATHER"),            # primary
        ("X_MAIN025", "U_MAIN025"),          # mechanism
        ("X_MAIN025", "V_MAIN025"),          # mechanism
        ("X_MAIN025", "X_GATHER"),           # mechanism
    ]
    contrasts = {}
    for a, b in pairs:
        c = bootstrap(scores, manifest, a, b)
        if c is None:
            contrasts[f"{a}_minus_{b}"] = dict(status="incomplete")
            continue
        rb = rescue_break(scores, a, b)
        c["rescue_break_per_ds"] = rb
        c["rescued"] = sum(v["rescued"] for v in rb.values())
        c["broken"] = sum(v["broken"] for v in rb.values())
        contrasts[f"{a}_minus_{b}"] = c
    # completion 2x2 interaction: (X_MAIN - X_GATH) - (E_MAIN - E_GATH)
    c1 = bootstrap(scores, manifest, "X_MAIN025", "X_GATHER")
    c2 = bootstrap(scores, manifest, "E_MAIN025", "E_GATHER")
    if c1 and c2:
        # point estimate of the interaction and a bootstrap on the
        # difference-of-differences (same resampling indices per draw is not
        # available across the two independent calls; report the point
        # estimate plus both CIs, and note this limitation explicitly)
        contrasts["completion_2x2_interaction"] = dict(
            point=c1["delta"] - c2["delta"],
            ci_main=[c1["ci"], c2["ci"]],
            note=("point = (X_MAIN025-X_GATHER) - (E_MAIN025-E_GATHER); no "
                  "joint CI because the two deltas use independent bootstrap "
                  "runs; treat as descriptive"))
    result = dict(round="stage1_cross_stream_pilot", split="dev", arms=arms,
                  boot=dict(n=XC.N_BOOT, seed=XC.BOOT_SEED),
                  margin=XC.MARGIN, table=table, contrasts=contrasts,
                  diagnostics=dict(bank={arm: bank_diag(arm) for arm in arms},
                                   anchor_overlap=anchor_overlap(manifest),
                                   anchors_changed=scorer_changes_anchors(
                                       manifest)),
                  manifest_sha={ds: manifest["datasets"][ds]["dev_sha256"]
                                for ds in XC.DS_LIST},
                  code_commit=XC.git_commit())

    out = os.path.join(XC.OUT_DIR, args.out)
    with open(out + ".tmp", "w") as f:
        json.dump(result, f, indent=1)
    os.replace(out + ".tmp", out)

    print("\n== Stage-1 cross-stream DEV panel ==")
    hdr = "| arm | scorer | " + " | ".join(XC.DS_LIST) + " | macro |"
    print(hdr)
    print("|---" * (len(XC.DS_LIST) + 3) + "|")
    for arm in arms:
        t = table[arm]
        row = " | ".join(f"{t['per_ds'][ds]:.2f}"
                         if t['per_ds'][ds] is not None else "—"
                         for ds in XC.DS_LIST)
        mac = f"{t['macro']:.3f}" if t["macro"] is not None else "—"
        print(f"| {arm} | {t['scorer']} | {row} | {mac} |")
    for k, c in contrasts.items():
        if "delta" not in c:
            print(f"\n{k}: {c}")
            continue
        print(f"\n{k}: delta={c['delta']:+.3f} "
              f"CI=[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] "
              f"verdict={c.get('verdict')} rescued={c.get('rescued')} "
              f"broken={c.get('broken')}")
    print(f"\nanchors_changed: {result['diagnostics']['anchors_changed']}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
