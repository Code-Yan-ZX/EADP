"""E0 S2 root-cause: OCRBench official-formula-on-subset check (no GPU).

The 09-30 BLOCKED note flagged "OCRBench norm 10.3-14.3 vs historical
~62-68".  Hypothesis: the official Final Score aggregates category
accuracies with fixed per-category thresholds and is only defined on the
full set; on the 164-row DEV subset it collapses.  This script recomputes
per-question hits with the round-1-validated scorer (verbatim official
rule, per-question) on the EXISTING on-disk predictions and compares.

Usage:
  python s2_ocrbench_subset_check.py
"""

from __future__ import annotations

import ast as _ast
import glob
import json
import os
import sys

DISC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "discovery")
sys.path.insert(0, DISC_DIR)
import common  # noqa: E402

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "e0")
ACC_DIR = os.path.join(OUT_DIR, "acc")

HISTORICAL = "sage_conf_B2 OCRBench per-q 62.08 (n=240, legacy config)"


def per_q_ocrbench(shard_path):
    """Verbatim official per-question rule (amp_accuracy shim, round-1
    validated 27/27): math branch no-lowercase, others lowercase both
    sides, strip + '\n'->' ' both sides before containment."""
    shard = json.load(open(shard_path))
    from vlmeval.dataset import build_dataset as _bd
    dso = _bd("OCRBench")
    per_q = {}
    for k, rec in shard["records"].items():
        row = dso.data.iloc[int(k)]
        predict = str(rec["prediction"]).strip()
        answers = _ast.literal_eval(row["answer"])
        hit = 0.0
        if row["category"] == "Handwritten Mathematical Expression Recognition":
            p2 = predict.replace("\n", " ").replace(" ", "")
            if any(a.strip().replace("\n", " ").replace(" ", "") in p2
                   for a in answers):
                hit = 1.0
        else:
            pr = predict.lower().strip().replace("\n", " ")
            if any(a.lower().strip().replace("\n", " ") in pr
                   for a in answers):
                hit = 1.0
        per_q[str(k)] = hit
    return per_q


def main():
    rows = []
    for shard_path in sorted(glob.glob(os.path.join(
            ACC_DIR, "*", "K*", "OCRBench.json"))):
        parts = shard_path.split(os.sep)
        arm_id, kdir = parts[-3], parts[-2]
        score_p = shard_path.replace(".json", "_score.json")
        official = None
        if os.path.exists(score_p):
            official = json.load(open(score_p)).get("official")
        try:
            per_q = per_q_ocrbench(shard_path)
        except Exception as e:
            print(f"[skip] {arm_id}/{kdir}: {e}")
            continue
        n = len(per_q)
        per_q_mean = 100.0 * sum(per_q.values()) / n
        norm_official = None
        if official:
            v = official.get("Final Score Norm",
                             official.get("Final Score"))
            if v is not None:
                norm_official = float(v)
        rows.append(dict(arm=arm_id, K=kdir, n=n,
                         official_subset_norm=norm_official,
                         per_q_mean=per_q_mean))
        print(f"{arm_id:12s} {kdir:6s} n={n:3d}  "
              f"official-on-subset norm={norm_official}  "
              f"per-q mean={per_q_mean:.2f}", flush=True)
    out = dict(historical_reference=HISTORICAL, rows=rows)
    with open(os.path.join(OUT_DIR, "s2_ocrbench_subset_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("[saved] s2_ocrbench_subset_check.json")


if __name__ == "__main__":
    main()
