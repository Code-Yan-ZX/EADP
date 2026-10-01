"""Anchor-Merge Pilot — render markdown tables for the report from JSONs.

Usage: python amp_report.py --split dev --arms BASE,U025,U050,U100,S025 [--winner W]
Prints the results section (tables + contrasts) to stdout; the narrative in
docs/anchor_merge_pilot_report.md references these exact numbers.
"""

from __future__ import annotations

import argparse
import json
import os

import amp_common as AC


def load(split, arms):
    out = {}
    for arm in arms:
        for ds in AC.DS_LIST:
            import glob as _glob
            hits = sorted(_glob.glob(os.path.join(
                AC.OUT_DIR, "acc", split, arm, "K*", f"{ds}_score.json")))
            if hits:
                s = json.load(open(hits[-1]))
                if s.get("per_question"):
                    out[(arm, ds)] = s
    return out


def acc(s):
    return 100.0 * sum(float(v) for v in s["per_question"].values()) \
        / len(s["per_question"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--arms", required=True)
    args = ap.parse_args()
    arms = args.arms.split(",")
    ana = json.load(open(os.path.join(AC.OUT_DIR,
                                      f"analysis_{args.split}.json")))
    scores = load(args.split, arms)

    print(f"### {args.split} panel (per-question official rule, 0-100)\n")
    print("| arm | " + " | ".join(AC.DS_LIST) + " | macro |")
    print("|---" * (len(AC.DS_LIST) + 2) + "|")
    for arm in arms:
        t = ana["table"][arm]
        row = []
        for ds in AC.DS_LIST:
            v = t["per_ds"][ds]
            row.append(f"{v:.2f}" if v is not None else "—")
        mac = f"**{t['macro']:.3f}**" if t["macro"] is not None else "—"
        print(f"| {arm} | " + " | ".join(row) + f" | {mac} |")

    print("\n### paired contrasts (cluster bootstrap, 95% CI)\n")
    print("| contrast | Δ macro | 95% CI | rescued | broken | per-ds Δ |")
    print("|---|---:|---|---:|---:|---|")
    for k, c in ana.get("contrasts", {}).items():
        kname = k.replace("_vs_", " − ")
        per = ", ".join(f"{d} {v:+.2f}" for d, v in c["per_ds"].items())
        print(f"| {kname} | {c['delta']:+.3f} | "
              f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] | "
              f"{c.get('rescued', '—')} | {c.get('broken', '—')} | {per} |")

    print("\n### generation quality\n")
    print("| arm | ds | n | empty | truncated | degenerate-repeat |")
    print("|---|---|---:|---:|---:|---:|")
    for arm in arms:
        q = ana["table"][arm].get("quality", {})
        for ds, v in q.items():
            print(f"| {arm} | {ds} | {v['n']} | {v['empty']} | "
                  f"{v['truncated']} | {v['degenerate_repeat']} |")

    if ana.get("bank_stats"):
        print("\n### mechanism diagnostics (from the frozen bank)\n")
        print("| ds | groups/img | grp size p50/p90/max | singleton frac | "
              "‖m−a‖/‖a‖ mean (main) | cos(a,m) mean (main) | ‖m−a‖/‖a‖ mean (DS) |")
        print("|---|---|---|---:|---:|---:|---:|")
        for ds, s in ana["bank_stats"].items():
            print(f"| {ds} | {AC.K} | {s['group_size_p50']:.0f} / "
                  f"{s['group_size_p90']:.0f} / {s['group_size_max']} | "
                  f"{s['singleton_frac']:.3f} | {s['rnorm_main_mean']:.3f} | "
                  f"{s['cos_am_main_mean']:.3f} | {s['rnorm_ds_mean']:.3f} |")

    if "winner" in ana:
        print(f"\n**winner**: {ana['winner']}  ·  any_positive: "
              f"{ana.get('any_positive')}")


if __name__ == "__main__":
    main()
