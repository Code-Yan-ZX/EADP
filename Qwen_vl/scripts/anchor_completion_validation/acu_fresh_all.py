"""Formal-method results on ALL qualified fresh rows (dispatch 四.1).

Uses the completed main-panel per-question scores of BASE and MAIN025,
restricted to exposure-fresh rows per dataset (whole image clusters).
OCRBench has zero fresh rows -> honestly marked 无法独立比较.
Paired cluster bootstrap 20000x seed=20261002, per-task + pooled.

Usage: python acu_fresh_all.py
"""

from __future__ import annotations

import json
import os

import numpy as np

import acu_common as AU

OUT = os.path.join(AU.OUT_DIR, "analysis_fresh_all.json")


def load_pq(panel: str, arm: str, ds: str):
    p = AU.shard_path(panel, arm, ds).replace(".json", "_score.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d.get("per_question") or None


def main():
    exp = json.load(open(os.path.join(AU.OUT_DIR, "exposure_manifest.json")))
    out = dict(note="fresh = 未用于本项目方案选择; 不声称排除预训练污染",
               seed=20261002, n_boot=20000, datasets={})
    rng = np.random.default_rng(20261002)
    pooled_draws = []
    pooled_deltas = []
    for ds in AU.DS_MAIN:
        rows = exp["datasets"][ds]["rows"]
        pq_b = load_pq("main", "BASE", ds)
        pq_m = load_pq("main", "MAIN025", ds)
        if not pq_b or not pq_m:
            out["datasets"][ds] = dict(status="missing_scores")
            continue
        fresh = [k for k, v in rows.items()
                 if v["cls"] == "fresh" and k in pq_b and k in pq_m]
        entry = dict(n_fresh_scored=len(fresh))
        if not fresh:
            entry["status"] = "no_fresh_unable_to_compare"
            out["datasets"][ds] = entry
            pooled_draws.append(None)
            continue
        acc_b = 100.0 * float(np.mean([float(pq_b[k]) for k in fresh]))
        acc_m = 100.0 * float(np.mean([float(pq_m[k]) for k in fresh]))
        diffs = np.array([float(pq_m[k]) - float(pq_b[k]) for k in fresh])
        clusters = np.array([rows[k]["image_key"] for k in fresh])
        uniq = np.unique(clusters)
        members = [np.where(clusters == u)[0] for u in uniq]
        draws = []
        for _ in range(20000):
            pick = rng.integers(0, len(uniq), len(uniq))
            idx = np.concatenate([members[i] for i in pick])
            draws.append(idx)
        vals = np.array([float(diffs[ix].mean()) for ix in draws]) * 100.0
        entry.update(
            n_images=len(uniq),
            acc_BASE=acc_b, acc_MAIN025=acc_m,
            delta=float(diffs.mean()) * 100.0,
            ci=[float(np.percentile(vals, 2.5)),
                float(np.percentile(vals, 97.5))])
        pooled_deltas.append(diffs)
        pooled_draws.append(draws)
        out["datasets"][ds] = entry
    # pooled over datasets with fresh rows (per-question weighting,
    # per-dataset resampling aligned by draw index)
    valid = [i for i, d in enumerate(pooled_draws) if d is not None]
    if valid:  # pooled over datasets that HAVE fresh rows
        ds_list = [AU.DS_MAIN[i] for i in valid]
        pv = np.array([
            np.concatenate([pooled_deltas[j][pooled_draws[j][b]]
                            for j in valid]).mean()
            for b in range(20000)]) * 100.0
        out["pooled_fresh"] = dict(
            delta=float(np.mean([pooled_deltas[j].mean()
                                 for j in valid])) * 100.0,
            ci=[float(np.percentile(pv, 2.5)),
                float(np.percentile(pv, 97.5))])
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
