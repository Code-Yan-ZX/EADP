"""R_MAIN025 full-panel analysis (GO addendum §2, preregistered).

Primary: R_MAIN025 - E_MAIN025 (= MAIN025 - BASE core full panel) equal-weight
macro with per-task detail; image-cluster bootstrap 20000x, seed=20261002;
non-inferiority margin 0.5 (CI lower bound > -0.5 supports non-inferiority);
DocVQA reported explicitly.  Does NOT modify the original BASE/MAIN025
protocol judgment.

Usage: python rtg_analyze_full.py
"""

from __future__ import annotations

import json
import os

import numpy as np

import rtg_common as RC

ACV_OUT = os.path.join(RC.common.QWEN_ROOT, "outputs",
                       "anchor_completion_validation")


def score_path(ds: str, arm: str, panel: str = "full"):
    if panel == "full" and arm == "R_MAIN025":
        return os.path.join(RC.OUT_DIR, "full", "acc", "R_MAIN025",
                            f"{ds}_score.json")
    return os.path.join(ACV_OUT, "acc", "main", arm, "K256",
                        f"{ds}_score.json")


def score_of(path: str):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return d if d.get("per_question") else None


def cluster_map(ds: str, keys):
    with __import__("gzip").open(
            os.path.join(ACV_OUT, f"bank_full_{ds}.json.gz"), "rt") as f:
        bank = json.load(f)
    return {k: bank[k].get("image_key") or f"row:{k}" for k in keys}


class Boot:
    def __init__(self, clusters, seed):
        uniq = np.unique(clusters)
        self.members = [np.where(clusters == u)[0] for u in uniq]
        self.n = len(uniq)
        self.rng = np.random.default_rng(seed)

    def draws(self, n):
        out = []
        for _ in range(n):
            pick = self.rng.integers(0, self.n, self.n)
            out.append(np.concatenate([self.members[i] for i in pick]))
        return out


def main():
    out = dict(n_boot=20000, seed=20261002,
               contrast="R_MAIN025-E_MAIN025 (full panel)")
    per_task, deltas, boots = {}, {}, {}
    ready = True
    for ds in RC.DS_LIST:
        sr = score_of(score_path(ds, "R_MAIN025"))
        se = score_of(score_path(ds, "E_MAIN025", panel="main"))
        if not sr or not se:
            ready = False
            break
        keys = sorted(set(sr["per_question"]) & set(se["per_question"]),
                      key=int)
        diffs = np.array([float(sr["per_question"][k]) -
                          float(se["per_question"][k]) for k in keys])
        cmap = cluster_map(ds, keys)
        clusters = np.array([cmap.get(k, f"row:{k}") for k in keys])
        per_task[ds] = dict(
            n=len(keys), n_images=len(set(clusters)),
            acc_r=100.0 * float(np.mean([sr["per_question"][k]
                                         for k in keys])),
            acc_e=100.0 * float(np.mean([se["per_question"][k]
                                         for k in keys])))
        deltas[ds] = diffs
        boots[ds] = Boot(clusters, 20261002)
    if not ready:
        out["status"] = "not_ready"
    else:
        idx = {ds: boots[ds].draws(20000) for ds in RC.DS_LIST}
        out["per_task"] = {}
        for ds in RC.DS_LIST:
            vals = np.array([deltas[ds][idx[ds][b]].mean() * 100.0
                             for b in range(20000)])
            out["per_task"][ds] = dict(
                **per_task[ds],
                delta=float(deltas[ds].mean()) * 100.0,
                ci=[float(np.percentile(vals, 2.5)),
                    float(np.percentile(vals, 97.5))])
        macro_draws = np.mean(np.stack(
            [np.array([deltas[ds][idx[ds][b]].mean() for b in range(20000)])
             for ds in RC.DS_LIST], axis=0), axis=0)
        out["macro"] = dict(
            delta=float(np.mean([deltas[ds].mean()
                                 for ds in RC.DS_LIST])) * 100.0,
            ci=[float(np.percentile(macro_draws, 2.5)) * 100.0,
                float(np.percentile(macro_draws, 97.5)) * 100.0])
        m = out["macro"]
        out["non_inferiority_margin_0.5"] = dict(
            ci_lower=m["ci"][0], supports=bool(m["ci"][0] > -0.5))
    path = os.path.join(RC.OUT_DIR, "full", "analysis_full.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1)[:1500])


if __name__ == "__main__":
    main()
