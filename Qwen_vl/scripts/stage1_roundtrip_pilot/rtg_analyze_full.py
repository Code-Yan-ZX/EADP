"""R_MAIN025 full-panel analysis (GO addendum §2 + dispatch 四.3/四.4).

Primary (preregistered, unchanged): R_MAIN025 - E_MAIN025, equal-weight
macro + per-task, image-cluster bootstrap 20000x seed=20261002,
non-inferiority margin 0.5 (CI lower > -0.5 supports non-inferiority),
DocVQA reported explicitly.
Supplementary (dispatch 四.4, registered): R_MAIN025 - E_GATHER — does the
complete method beat original EADP?  Supplementary, never overrides the
primary endpoint.

Usage: python rtg_analyze_full.py
"""

from __future__ import annotations

import json
import os

import numpy as np

import rtg_common as RC

ACV_OUT = os.path.join(RC.common.QWEN_ROOT, "outputs",
                       "anchor_completion_validation")


def score_path(ds: str, arm: str):
    if arm == "R_MAIN025":
        return os.path.join(RC.OUT_DIR, "full", "acc", "R_MAIN025",
                            f"{ds}_score.json")
    core = {"E_MAIN025": "MAIN025", "E_GATHER": "BASE"}.get(arm, arm)
    return os.path.join(ACV_OUT, "acc", "main", core, "K256",
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


def analyze_contrast(arm_a: str, arm_b: str, n_boot: int = 20000):
    """arm_a - arm_b paired analysis on the full panel."""
    per_task, deltas, boots = {}, {}, {}
    for ds in RC.DS_LIST:
        sa = score_of(score_path(ds, arm_a))
        sb = score_of(score_path(ds, arm_b))
        if not sa or not sb:
            return None
        keys = sorted(set(sa["per_question"]) & set(sb["per_question"]),
                      key=int)
        diffs = np.array([float(sa["per_question"][k]) -
                          float(sb["per_question"][k]) for k in keys])
        cmap = cluster_map(ds, keys)
        clusters = np.array([cmap.get(k, f"row:{k}") for k in keys])
        per_task[ds] = dict(
            n=len(keys), n_images=len(set(clusters)),
            acc_a=100.0 * float(np.mean([sa["per_question"][k]
                                         for k in keys])),
            acc_b=100.0 * float(np.mean([sb["per_question"][k]
                                         for k in keys])))
        deltas[ds] = diffs
        boots[ds] = Boot(clusters, 20261002)
    idx = {ds: boots[ds].draws(n_boot) for ds in RC.DS_LIST}
    out = dict(per_task={})
    for ds in RC.DS_LIST:
        vals = np.array([deltas[ds][idx[ds][b]].mean() * 100.0
                         for b in range(n_boot)])
        out["per_task"][ds] = dict(
            **per_task[ds],
            delta=float(deltas[ds].mean()) * 100.0,
            ci=[float(np.percentile(vals, 2.5)),
                float(np.percentile(vals, 97.5))])
    macro_draws = np.mean(np.stack(
        [np.array([deltas[ds][idx[ds][b]].mean() for b in range(n_boot)])
         for ds in RC.DS_LIST], axis=0), axis=0)
    out["macro"] = dict(
        delta=float(np.mean([deltas[ds].mean() for ds in RC.DS_LIST])) * 100.0,
        ci=[float(np.percentile(macro_draws, 2.5)) * 100.0,
            float(np.percentile(macro_draws, 97.5)) * 100.0])
    return out


def main():
    out = dict(n_boot=20000, seed=20261002)
    primary = analyze_contrast("R_MAIN025", "E_MAIN025")
    if primary is None:
        out["status"] = "not_ready (R_MAIN025 full scores incomplete)"
    else:
        out["primary_R_MAIN025-E_MAIN025"] = primary
        m = primary["macro"]
        out["primary_non_inferiority_margin_0.5"] = dict(
            ci_lower=m["ci"][0], supports=bool(m["ci"][0] > -0.5))
    supp = analyze_contrast("R_MAIN025", "E_GATHER")
    if supp is not None:
        out["supplementary_R_MAIN025-E_GATHER"] = supp
    path = os.path.join(RC.OUT_DIR, "full", "analysis_full.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("written:", path)
    for key in ("primary_R_MAIN025-E_MAIN025",
                "supplementary_R_MAIN025-E_GATHER"):
        if key in out and "macro" in out[key]:
            m = out[key]["macro"]
            print(f"  {key}: macro Δ={m['delta']:.3f} "
                  f"CI [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]")
            for ds, v in out[key]["per_task"].items():
                print(f"    {ds}: Δ={v['delta']:.3f} "
                      f"CI [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]")


if __name__ == "__main__":
    main()
