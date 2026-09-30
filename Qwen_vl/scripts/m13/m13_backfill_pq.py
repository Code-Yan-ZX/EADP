"""Backfill per-question eval scores into *_score.json files from the
VLMEvalKit *_pred_results.xlsx detail files.  No model load, CPU only.

Usage: python m13_backfill_pq.py
"""

from __future__ import annotations

import glob
import json
import os

import pandas as pd

sys_dir = os.path.dirname(os.path.abspath(__file__))
ACC_DIR = os.path.join(os.path.dirname(os.path.dirname(sys_dir)), "outputs", "m13", "acc")


def main():
    n = 0
    for f in sorted(glob.glob(os.path.join(ACC_DIR, "*", "*", "*_score.json"))):
        with open(f) as fh:
            s = json.load(fh)
        if s.get("per_question"):
            continue
        xlsx = f.replace("_score.json", "_pred_results.xlsx")
        if not os.path.exists(xlsx):
            continue
        d = pd.read_excel(xlsx)
        if "eval_score" not in d.columns or "index" not in d.columns:
            continue
        s["per_question"] = {str(int(r["index"])): float(r["eval_score"])
                             for _, r in d.iterrows()}
        tmp = f + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(s, fh, indent=1)
        os.replace(tmp, f)
        n += 1
        print(f"[backfilled] {os.path.relpath(f, ACC_DIR)} ({len(s['per_question'])} qs)")
    print(f"[done] {n} score files updated")


if __name__ == "__main__":
    main()
