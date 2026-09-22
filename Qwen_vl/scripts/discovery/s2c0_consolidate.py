"""
S2-C0 step 4: merge the accuracy arms, compute the retained-gain ratios and the
gate, and write the single deliverable JSON.

Accuracy arms come from two places, and both are paired on dataset index:
  * the forward-proxy arms, run here on the S2-B pilot (``s2c0_pilot.json``);
  * the P1-G2 teacher + Top-K upper bound, already measured by S2-B on the frozen
    150 (``s2b_full.json``) -- restricted to the pilot indices, which is exact
    because generation is per-instance and the pilot is a subset of the frozen
    bank. Re-running it would produce the same numbers from the same harness.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from s1_audit import OUT
from s2b_analysis import DS, macro_bootstrap, paired

TEACHER_RUN = "topk|C3"           # S2-B's P1-G2 + TopK arm, on the frozen 150
TEACHER_ARM = "proxy|P1G2"        # the same map re-run here on the pilot
OFFICIAL_TOPK = "proxy|official"  # EADP's own score through the same Top-K selector
OFFICIAL_FILE = "diag_selectors_b256.json"
GATE_REGRESSION_PT = 2.0


def load_official():
    off_all = json.load(open(os.path.join(OUT, OFFICIAL_FILE)))["runs"]
    return {v["dataset"]: v for v in off_all.values() if v["selector"] == "facility"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c0")
    args = ap.parse_args()

    proxy = json.load(open(os.path.join(OUT, "s2c0_pilot.json")))["runs"]
    full = json.load(open(os.path.join(OUT, "s2b_full.json")))["runs"]
    off = load_official()

    # arms: {arm_name: {dataset: run}}
    arms = {}
    for k, v in full.items():
        name = f"{v['selector']}|{v['calibration']}"
        if name == "facility|C3":
            arms.setdefault(name, {})[v["dataset"]] = v
    for k, v in proxy.items():
        arms.setdefault(f"proxy|{v['calibration']}", {})[v["dataset"]] = v

    # Identity check: the teacher arm re-run here must reproduce S2-B's own
    # ``topk|C3`` arm on the pilot indices exactly. If it does, the S2-C0 runner
    # is neutral and the proxy numbers beside it are attributable to the maps.
    teach_ref = {v["dataset"]: v for k, v in full.items()
                 if f"{v['selector']}|{v['calibration']}" == TEACHER_RUN}
    identity = {}
    for d in DS:
        ri = {int(i): float(h) for i, h in zip(teach_ref[d]["idx"], teach_ref[d]["hits"])}
        mine = arms[TEACHER_ARM][d]
        keys = [int(i) for i in mine["idx"]]
        ref = float(np.mean([ri[k] for k in keys]) * 100)
        identity[d] = dict(s2b_topk_c3_subset=ref, s2c0_teacher=mine["acc_pct"],
                           identical=bool(abs(ref - mine["acc_pct"]) < 1e-9))
    print("[identity] teacher arm vs S2-B topk|C3 on the pilot: "
          + ", ".join(f"{d.split('_')[0]}={'OK' if v['identical'] else 'MISMATCH'}"
                      for d, v in identity.items()))

    # Official baseline restricted to the instances every arm actually ran. The
    # proxy arms ran the pilot (50/dataset) and the teacher arm ran the frozen 150,
    # so this intersection is exactly the pilot -- never the full-150 accuracy.
    off_sub = {}
    for d in DS:
        oi = {int(i): float(h) for i, h in zip(off[d]["idx"], off[d]["hits"])}
        keys = set(oi)
        for per_ds in arms.values():
            keys &= {int(i) for i in per_ds[d]["idx"]}
        keys = sorted(keys)
        off_sub[d] = dict(idx=keys, hits=[oi[i] for i in keys], dataset=d)
        print(f"[paired subset] {d}: n={len(keys)} of {len(off[d]['idx'])} frozen")

    base_acc = [float(np.mean(off_sub[d]["hits"]) * 100) for d in DS]
    base_macro = float(np.mean(base_acc))

    summary = {}
    for name, per_ds in sorted(arms.items()):
        accs, deltas = [], {}
        for d in DS:
            oh, ah, _ = paired(off_sub[d], per_ds[d])
            accs.append(float(ah.mean() * 100))
            deltas[d] = ah - oh
        bs = macro_bootstrap(deltas)
        summary[name] = dict(
            per_dataset={d: dict(n=len(deltas[d]),
                                 official=float(np.mean(off_sub[d]["hits"]) * 100),
                                 arm=float(accs[i]), delta=float(deltas[d].mean() * 100))
                         for i, d in enumerate(DS)},
            macro=float(np.mean(accs)), macro_delta=float(np.mean(accs) - base_macro),
            ci=[float(np.percentile(bs, 2.5) * 100), float(np.percentile(bs, 97.5) * 100)],
        )

    # Two retained-gain readings, because the brief's definition and the
    # selector-matched one answer different questions:
    #   as_specified  (proxy - official_facility) / (teacher_topk - official_facility)
    #                 -- the brief's formula. It mixes the selector change into the
    #                 numerator, so a proxy is charged for Top-K's own cost.
    #   sel_matched   (proxy - official_topk) / (teacher_topk - official_topk)
    #                 -- holds the selector fixed at Top-K for every map, so it
    #                 isolates score quality from selector quality.
    teach = summary[TEACHER_ARM]["macro_delta"]
    off_topk = summary.get(OFFICIAL_TOPK)
    teach_sel = (teach - off_topk["macro_delta"]) if off_topk else None
    for name, s in summary.items():
        s["retained_gain"] = float(s["macro_delta"] / teach) if abs(teach) > 1e-9 else None
        s["retained_gain_selector_matched"] = (
            float((s["macro_delta"] - off_topk["macro_delta"]) / teach_sel)
            if (off_topk and teach_sel and abs(teach_sel) > 1e-9) else None)
        s["regressions_gt_2pt"] = int(sum(
            1 for d in DS if s["per_dataset"][d]["delta"] < -GATE_REGRESSION_PT))

    # ---- aggregate the causal side from the analysis file ----
    A = json.load(open(os.path.join(OUT, "s2c0_proxy_analysis.json")))
    causal = A["causal_primary"]
    teacher_auroc = causal["P1_G2"]["auroc"]["mean"]

    gates = {}
    for arm in A["arms"]:
        c = causal[arm]
        rank_gain = -c["rank"]["delta"]
        auroc = c["auroc"]["mean"]
        acc_name = f"proxy|{arm}"
        acc = summary.get(acc_name)
        g1 = bool(auroc >= 0.65 or rank_gain >= 0.10)
        g2 = bool(acc is not None and acc["retained_gain"] is not None
                  and acc["retained_gain"] >= 0.50)
        g3 = bool(acc is not None and acc["regressions_gt_2pt"] <= 1)
        gates[arm] = dict(rank_gain=rank_gain, auroc=auroc,
                          auroc_vs_teacher_ratio=float(auroc / teacher_auroc),
                          teacher_spearman=A["agreement"][arm]["spearman_mean"],
                          teacher_top256=A["agreement"][arm]["top256_overlap"],
                          pilot_macro=(acc or {}).get("macro"),
                          macro_delta=(acc or {}).get("macro_delta"),
                          retained_gain=(acc or {}).get("retained_gain"),
                          retained_gain_selector_matched=(
                              acc or {}).get("retained_gain_selector_matched"),
                          regressions_gt_2pt=(acc or {}).get("regressions_gt_2pt"),
                          g1_causal=g1, g2_retained=g2, g3_regressions=g3,
                          passed=bool(g1 and g2 and g3))

    passed = {k: v for k, v in gates.items() if v["passed"]}
    best = max(gates, key=lambda k: (gates[k]["passed"], gates[k]["auroc"]))
    verdict = ("S2-C0 GO: early-layer forward influence recovers a meaningful fraction "
               "of the gradient teacher.") if passed else (
              "S2-C0 NO-GO: simple early-layer forward influence does not approximate "
              "the gradient sensitivity teacher.")

    out = {
        "verdict": verdict, "go": bool(passed),
        "official_macro": base_macro,
        "teacher_topk_macro": summary[TEACHER_ARM]["macro"],
        "teacher_topk_macro_delta": teach,
        "harness_identity_vs_s2b": identity,
        "teacher_causal_auroc": teacher_auroc,
        "summary": summary, "gates": gates, "best_arm": best,
        "analysis": A,
        "sanity": json.load(open(os.path.join(OUT, "s2c0_sanity.json"))),
        "timing": json.load(open(os.path.join(OUT, "s2c0_forward_proxy_meta.json"))),
    }
    path = os.path.join(OUT, f"{args.tag}_forward_proxy.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[saved] {path}")

    print("\n" + "=" * 100)
    print(f"PILOT ACCURACY (T=256, TopK, n=50/benchmark)   official macro {base_macro:.3f}")
    print(f"{'arm':22s}{'TextVQA':>10s}{'DocVQA':>10s}{'OCRBench':>10s}{'macro':>9s}"
          f"{'delta':>9s}{'retained':>10s}")
    for name, s in sorted(summary.items(), key=lambda kv: -kv[1]["macro"]):
        print(f"{name:22s}" + "".join(f"{s['per_dataset'][d]['arm']:10.3f}" for d in DS)
              + f"{s['macro']:9.3f}{s['macro_delta']:+9.3f}"
              + f"{(s['retained_gain'] or 0) * 100:9.1f}%")

    def num(v, fmt, width, suffix=""):
        return "n/a".rjust(width) if v is None else f"{v:{fmt}}{suffix}".rjust(width)

    print("\n" + "=" * 100)
    print("GATE (primary tier nNec<=256, n=8). Accuracy columns only exist for the arms")
    print("actually run through generation; the other layers are causal-only.")
    print(f"{'arm':10s}{'rankGain':>10s}{'AUROC':>8s}{'vsT':>7s}{'Spear':>7s}{'Top256':>8s}"
          f"{'macroD':>9s}{'retain':>8s}{'selMatch':>10s}{'regr':>6s}  verdict")
    for arm in A["arms"]:
        g = gates[arm]
        print(f"{arm:10s}{g['rank_gain']:10.4f}{g['auroc']:8.3f}"
              f"{g['auroc_vs_teacher_ratio']:7.2f}{g['teacher_spearman']:7.3f}"
              f"{g['teacher_top256']:8.3f}"
              + num(g["macro_delta"], "+.3f", 9)
              + num(g["retained_gain"], ".1%", 8)
              + num(g["retained_gain_selector_matched"], ".1%", 10)
              + num(g["regressions_gt_2pt"], "d", 6) + "  "
              + ("PASS" if g["passed"] else ("fail" if g["pilot_macro"] is not None
                                             else "not run")))
    print("\n" + verdict)
    print(f"best arm by (gate, AUROC): {best}")


if __name__ == "__main__":
    main()
