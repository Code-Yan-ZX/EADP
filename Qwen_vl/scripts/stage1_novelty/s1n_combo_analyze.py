"""Stage-1 x Stage-2 combination analysis — the 2x2 table (CONFIRM 600).

Cells:
  A  EADP BASE            (amp round-2 confirm shards, reused)
  B  EADP + MAIN025       (amp round-2 confirm shards, reused)
  C  NEW-S1 (C)           (this round's confirm shards)
  D  NEW-S1 + MAIN025     (this round's combo shards)

Reads score files from both output trees, paired cluster bootstrap vs the
BASE cell for B, C, D (same manifest / questions / seeds).

Usage: python s1n_combo_analyze.py --split confirm
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import s1n_common as SC
import amp_common as AC
from amp_analyze import acc100, macro_of, paired_macro_bootstrap, rescue_break

ARM_DIRS = {   # arm -> (tree, arm-name-in-tree)
    "BASE": ("amp", "BASE"),
    "MAIN025": ("amp", "MAIN025"),
    "C": ("s1n", "C"),
    "C_M025": ("s1n", "C_M025"),
}


def load_scores(split, arms):
    out = {}
    for arm in arms:
        tree, name = ARM_DIRS[arm]
        root = AC.OUT_DIR if tree == "amp" else SC.ACC_DIR
        for ds in SC.DS_LIST:
            hits = sorted(glob.glob(os.path.join(
                root, "acc", split, name, "K*", f"{ds}_score.json")))
            if not hits:
                continue
            s = json.load(open(hits[-1]))
            if not s.get("per_question"):
                continue
            out[(arm, ds)] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="confirm", choices=["dev", "confirm"])
    args = ap.parse_args()
    arms = list(ARM_DIRS)
    manifest = SC.load_manifest()
    scores = load_scores(args.split, arms)

    table = {}
    for arm in arms:
        per = {ds: (acc100(scores[(arm, ds)]) if (arm, ds) in scores else None)
               for ds in SC.DS_LIST}
        table[arm] = dict(per_ds=per, macro=macro_of(scores, arm))

    contrasts = {}
    for arm in arms[1:]:
        c = paired_macro_bootstrap(scores, manifest, args.split, arm, "BASE")
        if c:
            rb = rescue_break(scores, arm, "BASE")
            c["rescued"] = sum(v["rescued"] for v in rb.values())
            c["broken"] = sum(v["broken"] for v in rb.values())
            contrasts[f"{arm}_vs_BASE"] = c

    result = dict(split=args.split, arms=arms, table=table,
                  contrasts=contrasts)
    out = os.path.join(SC.OUT_DIR, f"analysis_2x2_{args.split}.json")
    with open(out + ".tmp", "w") as f:
        json.dump(result, f, indent=1)
    os.replace(out + ".tmp", out)

    print(f"\n== 2x2 Stage1/Stage2 combination ({args.split} 600) ==")
    print("| cell | TextVQA | DocVQA | OCRBench | macro |")
    print("|---|---|---|---|---|")
    for arm in arms:
        t = table[arm]
        row = " | ".join(f"{t['per_ds'][ds]:.2f}"
                         if t['per_ds'][ds] is not None else "—"
                         for ds in SC.DS_LIST)
        mac = f"{t['macro']:.3f}" if t["macro"] is not None else "—"
        print(f"| {arm} | {row} | {mac} |")
    for k, c in contrasts.items():
        print(f"{k}: delta={c['delta']:+.3f} CI=[{c['ci'][0]:+.3f}, "
              f"{c['ci'][1]:+.3f}] rescued={c.get('rescued')} "
              f"broken={c.get('broken')} per_ds="
              f"{ {d: round(v, 2) for d, v in c['per_ds'].items()} }")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
