"""
S2-C1 step 4: held-out accuracy translation, retained teacher gain, and the gate.

Every arm here is measured on the SAME held-out 150 instances (the frozen bank's
every-3rd index -- which is the S2-C0 / S2-B pilot, so the baselines do not need
re-running; S2-C0 already verified its teacher arm reproduces S2-B's ``topk|C3``
on these exact indices to 1e-9).

Every arm uses the SAME selector: Top-K @ 256. Per the brief (section 8), the
retained-gain denominator is therefore the teacher *through Top-K*, never the
official facility selector:

    retained = (student_macro - eadp_topk_macro)
             / (teacher_topk_macro - eadp_topk_macro)

The official facility baseline is still reported, because gate condition 3
requires beating it -- but it is not the denominator.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT
from s2b_analysis import DS, macro_bootstrap, paired

GATE_RETAINED_GO = 0.40
GATE_RETAINED_STRONG = 0.60
GATE_RETAINED_NOGO = 0.20
GATE_BENCH_MARGIN = 2.0          # points a benchmark must gain to count as "clearly better"


def load_runs(path, selector=None):
    runs = json.load(open(os.path.join(OUT, path)))["runs"]
    out = {}
    for k, v in runs.items():
        if selector and v["selector"] != selector:
            continue
        # the Stage-1 baseline file predates the calibration tag and keys arms by
        # selector alone; the S2-C0/S2-B files carry an explicit calibration.
        tag = v.get("calibration", v["selector"])
        out.setdefault(tag, {})[v["dataset"]] = v
    return out


def summarize(name, per_ds, off_sub, base_acc):
    accs, deltas = [], {}
    for d in DS:
        oh, ah, _ = paired(off_sub[d], per_ds[d])
        accs.append(float(ah.mean() * 100))
        deltas[d] = ah - oh
    bs = macro_bootstrap(deltas)
    return dict(
        arm=name,
        per_dataset={d: dict(n=len(deltas[d]),
                             official_facility=float(np.mean(off_sub[d]["hits"]) * 100),
                             acc=float(accs[i]),
                             delta_vs_facility=float(deltas[d].mean() * 100))
                     for i, d in enumerate(DS)},
        macro=float(np.mean(accs)),
        macro_delta_vs_facility=float(np.mean(accs) - np.mean(base_acc)),
        ci_vs_facility=[float(np.percentile(bs, 2.5) * 100),
                        float(np.percentile(bs, 97.5) * 100)],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c1")
    ap.add_argument("--student-file", default="s2c1_pilot.json")
    ap.add_argument("--select", default=None,
                    help="comma-separated student arms to gate; default: every arm "
                         "in the student file")
    ap.add_argument("--extra", nargs="*", default=None,
                    help="extra run files to report but never gate "
                         "(the single robustness check lives here)")
    args = ap.parse_args()

    students = load_runs(args.student_file)
    s2c0 = load_runs("s2c0_pilot.json")
    off = load_runs("diag_selectors_b256.json", selector="facility")

    # extra diagnostic arms (e.g. the single robustness check). Reported beside
    # the gate, never part of it -- the gate set is fixed by --student-file.
    extras = {}
    for path in (args.extra or []):
        for tag, per_ds in load_runs(path).items():
            for d, v in per_ds.items():
                extras.setdefault(f"{tag}@{v['selector']}", {})[d] = v

    # restrict the official facility baseline to the held-out instances
    off_sub = {}
    for d in DS:
        oi = {int(i): float(h) for i, h in zip(off["facility"][d]["idx"],
                                               off["facility"][d]["hits"])}
        keys = set(oi)
        for src in (students, s2c0):
            for per_ds in src.values():
                keys &= {int(i) for i in per_ds[d]["idx"]}
        keys = sorted(keys)
        off_sub[d] = dict(idx=keys, hits=[oi[i] for i in keys], dataset=d)
    base_acc = [float(np.mean(off_sub[d]["hits"]) * 100) for d in DS]
    base_macro = float(np.mean(base_acc))
    print(f"[held-out] n={len(off_sub[DS[0]]['idx'])}/benchmark   "
          f"official facility macro = {base_macro:.3f}")

    summary, sources = {}, {}
    for prefix, src in (("s2c0", s2c0), ("student", students), ("extra", extras)):
        for a, per_ds in src.items():
            name = f"{prefix}|{a}"
            summary[name] = summarize(a, per_ds, off_sub, base_acc)
            sources[name] = per_ds

    # ---- selector-matched retained gain, Top-K for every arm ----------------
    teacher = summary["s2c0|P1G2"]
    eadp_topk = summary["s2c0|official"]
    eadp_topk_hits = {d: {int(i): float(h) for i, h
                          in zip(sources["s2c0|official"][d]["idx"],
                                 sources["s2c0|official"][d]["hits"])}
                      for d in DS}
    t_num = teacher["macro"] - eadp_topk["macro"]
    for name, s in summary.items():
        # paired bootstrap against the selector-matched reference, on the same
        # instances: this is the CI on the retained ratio's numerator.
        deltas = {}
        for d in DS:
            _, ah, idx = paired(off_sub[d], sources[name][d])
            deltas[d] = ah - np.array([eadp_topk_hits[d][int(i)] for i in idx])
        bs = macro_bootstrap(deltas)
        s["ci_vs_eadp_topk"] = [float(np.percentile(bs, 2.5) * 100),
                                float(np.percentile(bs, 97.5) * 100)]
        s["retained_vs_eadp_topk"] = float(
            (s["macro"] - eadp_topk["macro"]) / t_num) if abs(t_num) > 1e-9 else None
        s["delta_vs_eadp_topk"] = float(s["macro"] - eadp_topk["macro"])
        s["benchmarks_gt_eadp_topk_by_2pt"] = int(sum(
            1 for d in DS
            if s["per_dataset"][d]["acc"] - eadp_topk["per_dataset"][d]["acc"]
            > GATE_BENCH_MARGIN))

    # ---- gate ---------------------------------------------------------------
    gate_arms = args.select.split(",") if args.select else list(students)
    gates = {}
    for a in gate_arms:
        s = summary[f"student|{a}"]
        c1 = bool(s["retained_vs_eadp_topk"] is not None
                  and s["retained_vs_eadp_topk"] >= GATE_RETAINED_GO)
        c2 = bool(s["benchmarks_gt_eadp_topk_by_2pt"] >= 2)
        c3 = bool(s["macro"] > base_macro)
        gates[a] = dict(
            retained=s["retained_vs_eadp_topk"], condition1_retained_ge_40=c1,
            condition2_two_benchmarks=s["benchmarks_gt_eadp_topk_by_2pt"],
            condition2_pass=c2, condition3_beats_facility=s["macro"] - base_macro,
            condition3_pass=c3, passed=bool(c1 and c2 and c3),
        )

    best = max(gate_arms, key=lambda a: summary[f"student|{a}"]["macro"]) if gate_arms else None
    if best is None:
        verdict = "S2-C1 NO-GO"
    else:
        g = gates[best]
        r = g["retained"] or 0.0
        if r < GATE_RETAINED_NOGO:
            verdict = "S2-C1 NO-GO"
        elif (r >= GATE_RETAINED_STRONG) and g["passed"]:
            verdict = "S2-C1 STRONG GO"
        elif g["passed"]:
            verdict = "S2-C1 GO"
        else:
            verdict = "S2-C1 AMBIGUOUS"

    # ---- token-only vs query-conditioned (brief section 10) -----------------
    E = json.load(open(os.path.join(OUT, f"{args.tag}_eval.json")))
    T = json.load(open(os.path.join(OUT, f"{args.tag}_train.json")))
    family = {}
    for a, res in T["results"].items():
        fam = res["family"]
        family.setdefault(fam, {})[a] = dict(
            val_overlap256=res["val"]["overlap256"], val_auroc=res["val"]["auroc"],
            test_overlap256=res["test"]["overlap256"], test_auroc=res["test"]["auroc"],
            n_params=res["n_params"], layer=res["layer"],
        )
    fam_acc = {}
    for a in gate_arms:
        res = T["results"][a]
        fam_acc.setdefault(res["family"], {})[a] = summary[f"student|{a}"]["macro"]

    out = {
        "verdict": verdict, "go": bool(best is not None and gates[best]["passed"]),
        "primary_student": best,
        "selection_rule": ("the two generated student arms were chosen on the val split "
                           "(one per family, by val Top-256 overlap); the verdict is taken "
                           "on the better of those two on the held-out 150"),
        "official_facility_macro": base_macro,
        "eadp_topk_macro": eadp_topk["macro"],
        "teacher_topk_macro": teacher["macro"],
        "teacher_gain_vs_eadp_topk": t_num,
        "summary": summary, "gates": gates,
        "robustness": {a: summary[f"extra|{a}"] for a in extras},
        "family_comparison": {"per_arm": family, "accuracy_by_family": fam_acc,
                              "eval_tier1": E["held_out"]},
        "eval": E, "train": T["results"],
        "split_counts": T["split_counts"],
        "config": vars(args),
    }
    path = os.path.join(OUT, f"{args.tag}_distillation.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[saved] {path}")

    print("\n" + "=" * 108)
    print(f"HELD-OUT 150 ACCURACY (T=256, Top-K for every arm; official facility = {base_macro:.3f})")
    print(f"{'arm':22s}" + "".join(f"{d[:9]:>10s}" for d in DS)
          + f"{'macro':>9s}{'vs EADPk':>10s}{'vs facil':>10s}{'retained':>10s}{'>2pt':>6s}")
    for name in sorted(summary, key=lambda x: -summary[x]["macro"]):
        s = summary[name]
        r = s["retained_vs_eadp_topk"]
        print(f"{name:22s}" + "".join(f"{s['per_dataset'][d]['acc']:10.3f}" for d in DS)
              + f"{s['macro']:9.3f}{s['delta_vs_eadp_topk']:+10.3f}"
              + f"{s['macro_delta_vs_facility']:+10.3f}"
              + f"{(r * 100 if r is not None else float('nan')):9.1f}%"
              + f"{s['benchmarks_gt_eadp_topk_by_2pt']:6d}")

    print("\n" + "=" * 108)
    print("GATE (brief section 9)")
    for a, g in gates.items():
        print(f"  {a:22s} retained={g['retained'] * 100:6.1f}%  "
              f"c1(>=40%)={'Y' if g['condition1_retained_ge_40'] else 'n'}  "
              f"c2(2 benchmarks >EADP-TopK by 2pt)={g['condition2_two_benchmarks']}/3 "
              f"{'Y' if g['condition2_pass'] else 'n'}  "
              f"c3(beats facility)={'Y' if g['condition3_pass'] else 'n'}  "
              f"-> {'PASS' if g['passed'] else 'fail'}")
    print("\n" + "=" * 108)
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
