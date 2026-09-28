#!/usr/bin/env python3
"""
Collect MosaicPrune phase-1 results into one table.

Reads VLMEvalKit outputs from outputs/mosaic/ (accuracy), the offline token
statistics, and the efficiency profiles; writes mosaic_phase1_results.json and
prints a markdown main table.

Usage:  python scripts/discovery/mosaic_analyze.py
"""

from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

MOSAIC_DIR = os.path.join(common.QWEN_ROOT, "outputs", "mosaic")
EADP_DIR = os.path.join(common.QWEN_ROOT, "outputs", "eadp",
                        "Qwen3-VL-8B-EADP-256-a0.5-b2.0")
DISC_DIR = common.OUTPUT_DIR

DATASETS = ["DocVQA_VAL", "OCRBench", "TextVQA_VAL"]
# reference points from earlier runs of this repo (see report for provenance)
REFERENCES = {
    "baseline-1024": {},          # filled from outputs/mosaic if present
    "eadp-256": {"DocVQA_VAL": 61.14, "OCRBench": 623.0, "TextVQA_VAL": 71.04},
}


def read_accuracy(model_name: str, dataset_name: str, root: str):
    # VLMEvalKit writes either {root}/{model}_{ds}_acc.csv (flat work-dir) or
    # {root}/{model}/T*/{model}_{ds}_acc.csv (per-model timestamp dir).
    def candidates():
        yield os.path.join(root, f"{model_name}_{dataset_name}_score.json")
        yield os.path.join(root, f"{model_name}_{dataset_name}_acc.csv")
        model_dir = os.path.join(root, model_name)
        if os.path.isdir(model_dir):
            for entry in sorted(os.listdir(model_dir)):
                yield os.path.join(model_dir, entry,
                                   f"{model_name}_{dataset_name}_score.json")
                yield os.path.join(model_dir, entry,
                                   f"{model_name}_{dataset_name}_acc.csv")

    if dataset_name == "OCRBench":
        for p in candidates():
            if os.path.exists(p):
                with open(p) as f:
                    d = json.load(f)
                return float(d.get("Final Score",
                                   d.get("final_score", d.get("Overall", 0))))
        return None
    for p in candidates():
        if os.path.exists(p):
            break
    else:
        return None
    with open(p) as f:
        rows = list(csv.reader(f))
    if not rows:
        return None
    # formats seen in this repo: "val,Overall" header + one data row, or a
    # bare "Overall" header + one value row; take the first numeric cell of
    # the last non-empty row.
    for r in reversed(rows):
        for cell in r:
            try:
                return float(cell)
            except ValueError:
                continue
    return None


def main():
    out = {"datasets": DATASETS, "arms": {}}

    def grab(model_name, root, label):
        accs = {ds: read_accuracy(model_name, ds, root) for ds in DATASETS}
        if any(v is not None for v in accs.values()):
            out["arms"][label] = accs

    grab("Qwen3-VL-8B-Instruct-1024", MOSAIC_DIR, "baseline-1024")
    grab("Qwen3-VL-8B-EADP-256-a0.5-b2.0", EADP_DIR, "eadp-256")
    for K in (256, 128):
        for mode in ("uniform", "random", "dispersion"):
            grab(f"Qwen3-VL-8B-Mosaic-{K}-{mode}", MOSAIC_DIR,
                 f"mosaic-{K}-{mode}")
    # fall back to recorded references for the EADP row
    for ds, v in REFERENCES["eadp-256"].items():
        if out["arms"].get("eadp-256", {}).get(ds) is None:
            out["arms"].setdefault("eadp-256", {})[ds] = v
            out["arms"]["eadp-256"][ds + " (cited)"] = True

    # token statistics + efficiency, when present
    for name, rel in (("tokenstats", "mosaic/tokenstats.json"),
                      ("efficiency_baseline", "efficiency_profile.json"),
                      ("efficiency_mosaic", "mosaic/efficiency_mosaic.json")):
        p = os.path.join(DISC_DIR, rel)
        if os.path.exists(p):
            with open(p) as f:
                out[name] = json.load(f)

    dest = os.path.join(DISC_DIR, "mosaic", "mosaic_phase1_results.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dest}\n")

    header = "| arm | " + " | ".join(DATASETS) + " |"
    print(header)
    print("|" + "---|" * (len(DATASETS) + 1))
    order = ["baseline-1024", "eadp-256", "mosaic-256-uniform",
             "mosaic-256-random", "mosaic-256-dispersion",
             "mosaic-128-uniform", "mosaic-128-random", "mosaic-128-dispersion"]
    for arm in order:
        if arm not in out["arms"]:
            continue
        cells = []
        for ds in DATASETS:
            v = out["arms"][arm].get(ds)
            cells.append("-" if v is None else f"{v:.2f}")
        print(f"| {arm} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
