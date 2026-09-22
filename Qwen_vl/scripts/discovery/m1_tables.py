"""
M1 step 6: render the markdown tables from ``m1_audit.json``.

Every number in ``docs/scoring_search_m1.md`` comes from here rather than being
retyped, so the document cannot drift from the artifact it reports.

Usage
    python m1_tables.py --tag m1 > outputs/discovery/m1_tables.md
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                              # noqa: E402


def f(x, d=4):
    return "n/a" if x is None else f"{x:.{d}f}"


def sgn(x, d=4):
    return "n/a" if x is None else f"{x:+.{d}f}"


def seg_cell(s):
    if s is None:
        return "n/a"
    return f"{sgn(s['mean'])} [{sgn(s['lo'])}, {sgn(s['hi'])}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m1")
    args = ap.parse_args()
    a = json.load(open(os.path.join(OUT, f"{args.tag}_audit.json")))

    P = print
    P(f"# M1 tables — arm `{a['arm']}`, target HEAD_RANK, margin {a['margin']}, "
      f"{a['n_boot_draws']} bootstrap draws\n")
    P(f"Verdict: **{a['verdict']['verdict']}** — {a['verdict']['why']}\n")

    # ------------------------------------------------------- primary ladder --
    P("## Primary protocol — fixed steps (900 updates for every n)\n")
    P("| n | R@8 | R@16 | R@32 | ov256 | per-seed R@8 | selected point "
      "(0-29) | selected step | OV-selected R@8 | train R@8 | val R@8 |")
    P("|---|-----|------|------|-------|--------------|----------------------"
      "|---------------|-----------------|-----------|---------|")
    for x in a["protocols"]["fixed-step"]["ladder"]:
        P(f"| {x['n']} | **{f(x['R8'])}** | {f(x['R16'])} | {f(x['R32'])} | "
          f"{f(x['ov256'])} | "
          f"{' / '.join(f(v) for v in x['per_seed'])} | "
          f"{' / '.join(str(v) for v in x['per_seed_best_point'])} | "
          f"{' / '.join(str(v) for v in x['per_seed_best_step'])} | "
          f"{f(x['ov_selected_R8'])} | {f(x['train_R8'])} | {f(x['val_R8'])} |")
    P("")
    P("| segment | delta | 95 % CI | per-seed | label |")
    P("|---------|-------|---------|----------|-------|")
    for k, s in a["protocols"]["fixed-step"]["segments"].items():
        P(f"| {k} | {sgn(s['mean'])} | [{sgn(s['lo'])}, {sgn(s['hi'])}] | "
          f"{' / '.join(sgn(v) for v in s['per_seed'])} | {s['label']} |")
    P("")
    if "240->960" in a["protocols"]["fixed-step"]["segments"]:
        s = a["protocols"]["fixed-step"]["segments"]["240->960"]
        P(f"Largest scale vs the S2-C6 reference (240 → 960): "
          f"**{sgn(s['mean'])}** [{sgn(s['lo'])}, {sgn(s['hi'])}] — {s['label']}\n")

    # ----------------------------------------------------- per benchmark -----
    P("## Per-benchmark paired deltas (50 held-out images each, descriptive)\n")
    segs = list(a["protocols"]["fixed-step"]["per_benchmark"].keys())
    benches = list(a["protocols"]["fixed-step"]["per_benchmark"][segs[0]].keys())
    P("| segment | " + " | ".join(benches) + " |")
    P("|---------|" + "|".join(["------"] * len(benches)) + "|")
    for sg in segs:
        cells = []
        for b in benches:
            s = a["protocols"]["fixed-step"]["per_benchmark"][sg][b]
            cells.append(f"{sgn(s['mean'])} [{sgn(s['lo'])}, {sgn(s['hi'])}] "
                         f"{s['label']}")
        P(f"| {sg} | " + " | ".join(cells) + " |")
    P("")

    # --------------------------------------------------------- secondary -----
    if "fixed-epoch" in a["protocols"]:
        P("## Secondary protocol — fixed epoch (S2-C6's own budget)\n")
        P("| n | R@8 | R@16 | ov256 | per-seed R@8 | selected epoch | "
          "epochs run | steps run |")
        P("|---|-----|------|-------|--------------|----------------|"
          "------------|-----------|")
        for x in a["protocols"]["fixed-epoch"]["ladder"]:
            P(f"| {x['n']} | **{f(x['R8'])}** | {f(x['R16'])} | "
              f"{f(x['ov256'])} | "
              f"{' / '.join(f(v) for v in x['per_seed'])} | "
              f"{' / '.join(str(v) for v in x['per_seed_best_epoch'])} | "
              f"{' / '.join(str(v) for v in x['per_seed_epochs_run'])} | "
              f"{' / '.join(str(v) for v in x['per_seed_steps_run'])} |")
        P("")
        P("| segment | delta | 95 % CI | per-seed | label |")
        P("|---------|-------|---------|----------|-------|")
        for k, s in a["protocols"]["fixed-epoch"]["segments"].items():
            P(f"| {k} | {sgn(s['mean'])} | [{sgn(s['lo'])}, {sgn(s['hi'])}] | "
              f"{' / '.join(sgn(v) for v in s['per_seed'])} | {s['label']} |")
        P("")

    # -------------------------------------------------------------- Q1 -------
    q1 = a["Q1_step_decomposition"]
    P("## Q1 — how much of S2-C6's 60 → 240 rise is optimizer steps?\n")
    P("| n | C6 fixed-epoch R@8 | M1 fixed-step R@8 |")
    P("|---|--------------------|-------------------|")
    for n in ("60", "120", "180", "240"):
        P(f"| {n} | {f(q1['c6_published_fixed_epoch_curve'].get(n))} | "
          f"{f(q1['m1_fixed_step_curve'].get(n))} |")
    P("")
    if "fixed_step_60_to_240" in q1:
        P(f"* 60 → 240, fixed epoch (published): "
          f"**{sgn(q1['c6_60_to_240'])}**")
        P(f"* 60 → 240, fixed steps: **{sgn(q1['fixed_step_60_to_240'])}**")
        P(f"* difference (the part the extra updates can account for): "
          f"**{sgn(q1.get('step_attributable'))}**"
          + (f" = {100*q1['step_attributable_fraction']:.0f} % of the published "
             f"rise" if q1.get("step_attributable_fraction") is not None else ""))
        sg = q1.get("fixed_step_segment_60_240")
        if sg:
            P(f"* fixed-step 60 → 240 paired CI "
              f"[{sgn(sg['lo'])}, {sgn(sg['hi'])}] — {sg['label']}")
    P("")

    # -------------------------------------------------------------- Q5 -------
    P("## Q5 — train/validation gap across the ladder\n")
    P("| n | train R@8 | val R@8 | held-out R@8 | train − val | "
      "train − held-out |")
    P("|---|-----------|---------|--------------|-------------|"
      "------------------|")
    for r in a["Q5_train_val_gap"]:
        P(f"| {r['n']} | {f(r['train_R8'])} | {f(r['val_R8'])} | "
          f"{f(r['test_R8'])} | {sgn(r['train_minus_val'])} | "
          f"{sgn(r['train_minus_test'])} |")
    P("")

    # -------------------------------------------------------------- Q6 -------
    g = a["Q6_downstream_gate"]
    P("## Q6 — downstream gate (S2-C6 §6, re-applied unchanged)\n")
    P(f"* best primary arm: n = {g['best_n']}, held-out R@8 = {f(g['best_R8'])}")
    P(f"* threshold {g['threshold']} (aspirational {g['aspirational']})")
    if "vs_L4_reference" in g:
        s = g["vs_L4_reference"]
        P(f"* vs the 240 reference: {sgn(s['mean'])} "
          f"[{sgn(s['lo'])}, {sgn(s['hi'])}] — {s['label']}")
    P(f"* **gate opens: {g['opens']}** — no generation is run by M1 either way")
    P("")

    # ------------------------------------------------------------- G3 --------
    G = a["gate_G3_reproduction"]
    P("## Gate G3 — fixed-step n=240 against S2-C6's published A240\n")
    P("| seed | epochs compared | C6 epochs run | C6 H8 epoch | M1 H8 point | "
      "epoch match | M1 held-out R@8 | published | match |")
    P("|------|-----------------|---------------|-------------|-------------|"
      "-------------|-----------------|-----------|-------|")
    for r in G["per_seed"]:
        P(f"| {r['seed']} | {r['epochs_compared']} | {r['c6_epochs_run']} | "
          f"{r['h8_epoch_c6']} | {r['h8_epoch_m1']} | {r['epoch_match']} | "
          f"{f(r['m1_test_h8'])} | {f(r['published_test_h8'])} | "
          f"{r['matches_published']} |")
    P("")
    P(f"worst validation-curve |diff| over the compared epochs: "
      f"{G['worst_val_curve_abs_diff']:.1e} — passed: **{G['passed']}**")


if __name__ == "__main__":
    main()
