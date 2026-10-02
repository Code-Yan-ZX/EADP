"""P1 fresh-panel mechanism contrasts (dispatch P1).

Rows: the frozen fresh panel (TextVQA 200 / DocVQA 200; OCRBench none).
Arms: R_MAIN025 (from the full-panel per-question scores, same rows),
E_MAIN025 / E_GATHER (main-panel scores, same rows), R_GATHER / F_MAIN025 /
S_MAIN025 (fresh generation).  Cluster bootstrap 20000x seed=20261002.
"""

from __future__ import annotations

import json
import os

import numpy as np

import rtg_common as RC

ACV_OUT = os.path.join(RC.common.QWEN_ROOT, "outputs",
                       "anchor_completion_validation")
FRESH_ACC = os.path.join(RC.OUT_DIR, "fresh_acc")


def fresh_rows(ds: str):
    man = json.load(open(os.path.join(RC.OUT_DIR, "..",
                                      "anchor_completion_validation",
                                      "fresh_panel_manifest.json")))
    d = man["datasets"][ds]
    return [str(i) for i in d.get("rows", [])] if d["status"] == "ok" else []


def pq(arm: str, ds: str):
    if arm == "R_MAIN025":
        p = os.path.join(RC.OUT_DIR, "full", "acc", "R_MAIN025",
                         f"{ds}_score.json")
    elif arm in ("E_MAIN025", "E_GATHER"):
        core = {"E_MAIN025": "MAIN025", "E_GATHER": "BASE"}[arm]
        p = os.path.join(ACV_OUT, "acc", "main", core, "K256",
                         f"{ds}_score.json")
    else:
        p = os.path.join(FRESH_ACC, arm, f"{ds}_score.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d.get("per_question") or None


def main():
    exp = json.load(open(os.path.join(ACV_OUT, "exposure_manifest.json")))
    out = dict(n_boot=20000, seed=20261002, panel="fresh")
    # per-arm absolute scores
    out["arm_scores"] = {}
    for ds in ("TextVQA_VAL", "DocVQA_VAL"):
        rows = fresh_rows(ds)
        out["arm_scores"][ds] = {}
        for arm in ("E_GATHER", "E_MAIN025", "R_GATHER", "R_MAIN025",
                    "F_MAIN025", "S_MAIN025"):
            q = pq(arm, ds)
            if q is None:
                continue
            vals = [float(q[k]) for k in rows if k in q]
            out["arm_scores"][ds][arm] = dict(
                n=len(vals),
                acc=100.0 * float(np.mean(vals)) if vals else None)
    # contrasts
    def contrast(a, b):
        per_task, deltas, boots = {}, {}, {}
        for ds in ("TextVQA_VAL", "DocVQA_VAL"):
            qa, qb = pq(a, ds), pq(b, ds)
            if qa is None or qb is None:
                return None
            rows = fresh_rows(ds)
            keys = [k for k in rows if k in qa and k in qb]
            if not keys:
                return None
            diffs = np.array([float(qa[k]) - float(qb[k]) for k in keys])
            clusters = np.array([exp["datasets"][ds]["rows"][k]["image_key"]
                                 for k in keys])
            uniq = np.unique(clusters)
            members = [np.where(clusters == u)[0] for u in uniq]
            rng = np.random.default_rng(20261002)
            idx_sets = []
            for _ in range(20000):
                pick = rng.integers(0, len(uniq), len(uniq))
                idx_sets.append(np.concatenate(
                    [members[i] for i in pick]))
            vals = np.array([float(diffs[ix].mean()) for ix in idx_sets])
            per_task[ds] = dict(
                n=len(keys), delta=float(diffs.mean()) * 100.0,
                ci=[float(np.percentile(vals, 2.5)) * 100.0,
                    float(np.percentile(vals, 97.5)) * 100.0])
            deltas[ds] = diffs
            boots[ds] = (idx_sets,)
        mv = []
        for ds in ("TextVQA_VAL", "DocVQA_VAL"):
            mv.append(np.array([deltas[ds][boots[ds][0][b]].mean()
                                for b in range(20000)]))
        macro_draws = np.mean(np.stack(mv, axis=0), axis=0)
        return dict(
            per_task=per_task,
            macro=dict(
                delta=float(np.mean([deltas[ds].mean()
                                     for ds in per_task])) * 100.0,
                ci=[float(np.percentile(macro_draws, 2.5)) * 100.0,
                    float(np.percentile(macro_draws, 97.5)) * 100.0]))
    for a, b in (("R_MAIN025", "E_MAIN025"), ("R_MAIN025", "E_GATHER"),
                 ("R_MAIN025", "R_GATHER"), ("R_MAIN025", "F_MAIN025"),
                 ("R_MAIN025", "S_MAIN025")):
        c = contrast(a, b)
        if c:
            out[f"{a}-{b}"] = c
    path = os.path.join(RC.OUT_DIR, "analysis_fresh_contrasts.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("written:", path)
    for k, v in out.items():
        if isinstance(v, dict) and "macro" in v:
            m = v["macro"]
            print(f"  {k}: macro Δ={m['delta']:.3f} "
                  f"CI [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]")


if __name__ == "__main__":
    main()
