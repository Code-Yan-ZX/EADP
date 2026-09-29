"""M12 analysis: accuracy tables, paired bootstrap CIs, Pareto data.

Reads m12 acc shards (outputs/m12/acc/<arm>/K< budget>/<ds>_score.json with
per_question) plus E0 baseline shards (b0 K1024, b2 K256, rres* from e0 acc
dir when the m12 arm is absent) and produces:
  * main accuracy table per budget (official scores, n);
  * paired per-question contrasts vs b0 and vs rres at matched budget with
    block-bootstrap 95 % CIs (resample questions);
  * latency summary from m12_speed_oracle.json (median vit / ttft / gate).
Writes outputs/m12/m12_analysis.json and prints markdown tables.
"""

from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

QWEN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
OUT12 = os.path.join(QWEN_ROOT, "outputs", "m12")
OUTE0 = os.path.join(QWEN_ROOT, "outputs", "e0")
DS_LIST = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench", "ChartQA_TEST"]

# official scalar per dataset
def official_score(ds, off):
    if ds == "OCRBench":
        return off.get("Final Score")
    for k in ("vqa_score", "anls", "acc", "VQA score"):
        if k in off:
            return off[k]
    return None


def load_scores(root, arm, K, ds):
    p = os.path.join(root, "acc", arm, f"K{K}", f"{ds}_score.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    sc = official_score(ds, d.get("official", {}))
    return dict(arm=arm, K=K, ds=ds, n=d.get("n"), score=sc,
                per_q=d.get("per_question"))


def paired_bootstrap(a_perq, b_perq, n_boot=10000, seed=0):
    """Mean(a-b) over shared questions with bootstrap CI."""
    keys = sorted(set(a_perq) & set(b_perq))
    if not keys:
        return None
    da = np.array([a_perq[k] for k in keys], dtype=float)
    db = np.array([b_perq[k] for k in keys], dtype=float)
    diff = da - db
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n=n, mean_diff=float(diff.mean()),
                ci95=[float(lo), float(hi)],
                p_gt0=float((boots > 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT12, "m12_analysis.json"))
    args = ap.parse_args()

    oracle_path = os.path.join(OUT12, "m12_speed_oracle.json")
    oracle = json.load(open(oracle_path))["summary"] \
        if os.path.exists(oracle_path) else {}

    # auto-discover shards: m12 arms + E0 baselines (b0 / b2 / rres)
    entries = []
    acc_root = os.path.join(OUT12, "acc")
    if os.path.isdir(acc_root):
        for arm in sorted(os.listdir(acc_root)):
            for kdir in os.listdir(os.path.join(acc_root, arm)):
                if kdir.startswith("K"):
                    entries.append(dict(arm=arm, K=int(kdir[1:]),
                                        root="m12", key=arm, keyK=int(kdir[1:])))
    # E0 baselines (arm dir naming there: b0/K1024, b2/K256, rres256/K1024)
    for e0_arm, label, labK in [("b0", "b0", 1024), ("b2", "b2", 256),
                                ("rres256", "rres", 256)]:
        d = os.path.join(OUTE0, "acc", e0_arm, f"K{1024 if e0_arm != 'b2' else 256}")
        if os.path.isdir(d):
            entries.append(dict(arm=label, K=labK, root="e0",
                                key=e0_arm,
                                keyK=256 if e0_arm == "b2" else 1024))

    rows = []
    for entry in entries:
        root = OUT12 if entry["root"] == "m12" else OUTE0
        for ds in DS_LIST:
            r = load_scores(root, entry["key"], entry["keyK"], ds)
            if r:
                r["arm"], r["K"] = entry["arm"], entry["K"]
                rows.append(r)

    # grouped by ds
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["ds"], {})[f"{r['arm']}|K{r['K']}"] = r

    contrasts = []
    # paired contrasts vs b0 (K1024) per arm per ds
    for r in rows:
        b0 = by_ds.get(r["ds"], {}).get("b0|K1024")
        if b0 and b0.get("per_q") and r.get("per_q") and \
                not (r["arm"] == "b0"):
            c = paired_bootstrap(r["per_q"], b0["per_q"])
            if c:
                contrasts.append(dict(arm=r["arm"], K=r["K"], ds=r["ds"],
                                      vs="b0", **c))
    # vs rres at same K
    for r in rows:
        rr = by_ds.get(r["ds"], {}).get(f"rres|K{r['K']}")
        if rr and rr.get("per_q") and r.get("per_q") and r["arm"] != "rres":
            c = paired_bootstrap(r["per_q"], rr["per_q"])
            if c:
                contrasts.append(dict(arm=r["arm"], K=r["K"], ds=r["ds"],
                                      vs="rres", **c))

    # markdown table
    lines = ["| ds | arm | K | n | score | Δ vs b0 (95% CI) |",
             "|---|---|---|---|---|---|"]
    for ds in DS_LIST:
        for key, r in sorted(by_ds.get(ds, {}).items()):
            cs = [c for c in contrasts if c["ds"] == ds
                  and c["arm"] == r["arm"] and c["K"] == r["K"]
                  and c["vs"] == "b0"]
            dcell = ""
            if cs:
                c = cs[0]
                dcell = f"{c['mean_diff']:+.2f} [{c['ci95'][0]:+.2f},{c['ci95'][1]:+.2f}]"
            sc = r["score"]
            lines.append(f"| {ds} | {r['arm']} | {r['K']} | {r['n']} | "
                         f"{round(sc, 2) if sc is not None else '?'} | {dcell} |")
    table = "\n".join(lines)
    print(table)

    with open(args.out, "w") as f:
        json.dump(dict(rows=rows, contrasts=contrasts, oracle=oracle,
                       table=table), f, indent=1)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
