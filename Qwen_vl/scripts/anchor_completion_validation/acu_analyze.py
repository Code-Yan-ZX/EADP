"""Anchor Completion Validation — analysis & frozen statistics (protocol §8).

Primary judgment (verbatim from docs/anchor_merge_full_eval_prereg.md §3):
  pooled = per-question-weighted merge of the paired per-question diffs of
  the three main-panel benchmarks; image-cluster paired bootstrap 5000x,
  seed 20261002, two-sided 95%; improvement holds iff pooled CI excludes
  zero on the positive side AND all three per-task diffs point the same way.
Sensitivity: 20000 draws of the SAME seeded stream (nested: the first 5000
  draws are identical to the primary) — never overrides the primary.
All paired comparisons share one precomputed resampling index set per
(dataset, panel).  Supplementary contrasts (fresh panel) each get per-task
and equal-weight macro delta + CI; CI crossing zero is NOT equivalence.
Conditional BOTH025 trigger: mechanical per prereg §4 (pooled CI crosses
zero AND pooled point estimate < +0.5).

Usage: python acu_analyze.py            (analyzes whatever panels exist)
"""

from __future__ import annotations

import json
import os

import numpy as np

import acu_common as AU

BOOT_SEED = AU.BOOT_SEED
N_PRIMARY = AU.N_BOOT_PRIMARY
N_SENS = AU.N_BOOT_SENS


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_scores(panel: str, arm: str, ds: str):
    p = AU.shard_path(panel, arm, ds).replace(".json", "_score.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d if d.get("per_question") else dict(d, per_question={})


def load_cluster_map(panel: str, ds: str, keys):
    """image cluster per row: bank image_key (same rule as e0_plan)."""
    bank = AU.load_bank(ds)
    return {k: bank[k]["image_key"] for k in keys if k in bank}


def paired(panel: str, arm_a: str, arm_b: str, ds: str):
    """Per-question paired diffs (a-b) over common keys + cluster ids."""
    sa, sb = load_scores(panel, arm_a, ds), load_scores(panel, arm_b, ds)
    if not sa or not sb:
        return None
    keys = sorted(set(sa["per_question"]) & set(sb["per_question"]),
                  key=int)
    if not keys:
        return None
    diffs = np.array([float(sa["per_question"][k]) -
                      float(sb["per_question"][k]) for k in keys])
    cmap = load_cluster_map(panel, ds, keys)
    clusters = np.array([cmap.get(k, f"nokey:{k}") for k in keys])
    return dict(keys=keys, diffs=diffs, clusters=clusters,
                n=len(keys), n_images=len(set(clusters)),
                mean_a=100.0 * float(np.mean(
                    [sa["per_question"][k] for k in keys])),
                mean_b=100.0 * float(np.mean(
                    [sb["per_question"][k] for k in keys])))


# ---------------------------------------------------------------------------
# bootstrap machinery (one index set per (panel, ds), shared by contrasts)
# ---------------------------------------------------------------------------
class Boot:
    def __init__(self, clusters: np.ndarray, seed: int):
        uniq = np.unique(clusters)
        self.members = [np.where(clusters == u)[0] for u in uniq]
        self.n_clusters = len(uniq)
        self.rng = np.random.default_rng(seed)

    def draws(self, n: int):
        """List of index arrays: resampled cluster members concatenated."""
        out = []
        for _ in range(n):
            pick = self.rng.integers(0, self.n_clusters, self.n_clusters)
            idx = np.concatenate([self.members[i] for i in pick])
            out.append(idx)
        return out


def boot_ci(diffs: np.ndarray, idx_sets):
    vals = np.array([float(diffs[idx].mean()) for idx in idx_sets])
    return (float(np.mean(diffs)) * 100.0,
            float(np.percentile(vals, 2.5)) * 100.0,
            float(np.percentile(vals, 97.5)) * 100.0)


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
def analyze_main_panel(arms=("BASE", "MAIN025")):
    """prereg §3 primary + per-task + macro + sensitivity + trigger."""
    per_task = {}
    boots = {}
    for ds in AU.DS_MAIN:
        pr = paired("main", arms[1], arms[0], ds)   # new - old
        if pr is None:
            return None
        per_task[ds] = pr
        boots[ds] = Boot(pr["clusters"], BOOT_SEED)
    # shared index sets: 20000 draws, primary = first 5000 (nested)
    idx_main = {ds: boots[ds].draws(N_SENS) for ds in AU.DS_MAIN}
    prim = {ds: idx_main[ds][:N_PRIMARY] for ds in AU.DS_MAIN}

    res = dict(arms=list(arms), per_task={}, macro={}, pooled={},
               sensitivity={})
    deltas = {ds: per_task[ds]["diffs"] for ds in AU.DS_MAIN}
    for ds in AU.DS_MAIN:
        lo_d = per_task[ds]
        m, lo, hi = boot_ci(deltas[ds], prim[ds])
        res["per_task"][ds] = dict(
            n=lo_d["n"], n_images=lo_d["n_images"],
            acc_new=lo_d["mean_a"], acc_old=lo_d["mean_b"],
            delta=m, ci=[lo, hi])
        ms, los, his = boot_ci(deltas[ds], idx_main[ds])
        res["sensitivity"][ds] = dict(delta=ms, ci=[los, his])
    # equal-weight macro
    dsum = float(np.mean([deltas[ds].mean() for ds in AU.DS_MAIN])) * 100.0
    macro_vals, macro_sens = [], []
    for b in range(N_PRIMARY):
        macro_vals.append(np.mean([deltas[ds][prim[ds][b]].mean()
                                   for ds in AU.DS_MAIN]))
    for b in range(N_SENS):
        macro_sens.append(np.mean([deltas[ds][idx_main[ds][b]].mean()
                                   for ds in AU.DS_MAIN]))
    res["macro"] = dict(
        delta=dsum,
        ci=[float(np.percentile(macro_vals, 2.5)) * 100.0,
            float(np.percentile(macro_vals, 97.5)) * 100.0])
    res["sensitivity"]["macro"] = dict(
        delta=float(np.mean(macro_sens)) * 100.0,
        ci=[float(np.percentile(macro_sens, 2.5)) * 100.0,
            float(np.percentile(macro_sens, 97.5)) * 100.0])
    # pooled: per-question weighted merge; per-dataset resampling aligned
    # by draw index b, then concatenated (per-question weighting)
    pooled = np.concatenate([deltas[ds] for ds in AU.DS_MAIN])
    pooled_vals = np.array([
        np.concatenate([deltas[ds][prim[ds][b]] for ds in AU.DS_MAIN]).mean()
        for b in range(N_PRIMARY)])
    pm, plo, phi = (float(pooled.mean()) * 100.0,
                    float(np.percentile(pooled_vals, 2.5)) * 100.0,
                    float(np.percentile(pooled_vals, 97.5)) * 100.0)
    res["pooled"] = dict(delta=pm, ci=[plo, phi],
                         n=int(len(pooled)))
    pooled_sens = np.array([
        np.concatenate([deltas[ds][idx_main[ds][b]]
                        for ds in AU.DS_MAIN]).mean()
        for b in range(N_SENS)])
    res["sensitivity"]["pooled"] = dict(
        delta=float(pooled_sens.mean()) * 100.0,
        ci=[float(np.percentile(pooled_sens, 2.5)) * 100.0,
            float(np.percentile(pooled_sens, 97.5)) * 100.0])
    # prereg improvement rule + mechanical BOTH025 trigger
    directions = [float(np.mean(deltas[ds])) for ds in AU.DS_MAIN]
    res["improvement_holds"] = bool(plo > 0 and all(d > 0
                                                    for d in directions))
    res["both025_trigger"] = bool(plo <= 0 <= phi and pm < 0.5)
    return res


def rescue_break(panel: str, arm_a: str, arm_b: str, ds: str):
    pr = paired(panel, arm_a, arm_b, ds)
    if pr is None:
        return None
    d = pr["diffs"]
    return dict(rescue=int((d > 0.5).sum()), break_=int((d < -0.5).sum()))


def coverage(panel: str):
    """n questions / images per exposure class per task (formal method)."""
    exp = json.load(open(os.path.join(AU.OUT_DIR, "exposure_manifest.json")))
    out = {}
    for ds in AU.DS_MAIN:
        entry = dict.fromkeys(("full", "exposed", "fresh", "unknown"), 0)
        imgs = {c: set() for c in ("exposed", "fresh", "unknown")}
        for arm in ("BASE", "MAIN025"):
            shard = AU.load_shard(AU.shard_path(panel, arm, ds))
            entry["full"] = max(entry["full"], len(shard["records"]))
        rows = exp["datasets"][ds]["rows"]
        shard = AU.load_shard(AU.shard_path(panel, "BASE", ds))
        for k in shard["records"]:
            c = rows.get(k, {}).get("cls", "unknown")
            entry[c] = entry.get(c, 0) + 1
            imgs[c].add(rows.get(k, {}).get("image_key", f"unk:{k}"))
        entry["images"] = {c: len(s) for c, s in imgs.items()}
        out[ds] = entry
    return out


def analyze_fresh():
    """Supplementary contrasts on the fresh control panel."""
    out = dict(contrasts={}, per_task_scores={})
    for ds in AU.DS_MAIN:
        for arm in ("BASE", "MAIN025", "MAIN100", "MAIN_SIM025"):
            s = load_scores("fresh", arm, ds)
            if s:
                out["per_task_scores"].setdefault(ds, {})[arm] = dict(
                    n=len(s["per_question"]),
                    acc=100.0 * float(np.mean(
                        [float(v) for v in s["per_question"].values()])))
    for a, b in (("MAIN025", "MAIN100"), ("MAIN025", "MAIN_SIM025")):
        per_task, boots = {}, {}
        ok = True
        for ds in AU.DS_MAIN:
            pr = paired("fresh", a, b, ds)
            if pr is None:
                ok = False
                break
            per_task[ds] = pr
            boots[ds] = Boot(pr["clusters"], BOOT_SEED)
        if not ok:
            out["contrasts"][f"{a}-{b}"] = dict(
                status="unavailable",
                note="fresh panel missing for at least one task (OCRBench "
                     "has zero fresh rows -> 无法独立比较)")
            continue
        idx = {ds: boots[ds].draws(N_PRIMARY) for ds in AU.DS_MAIN}
        deltas = {ds: per_task[ds]["diffs"] for ds in AU.DS_MAIN}
        entry = dict(per_task={}, macro={})
        for ds in AU.DS_MAIN:
            m, lo, hi = boot_ci(deltas[ds], idx[ds])
            entry["per_task"][ds] = dict(
                n=per_task[ds]["n"], n_images=per_task[ds]["n_images"],
                delta=m, ci=[lo, hi])
        mv = [np.mean([deltas[ds][idx[ds][bi]].mean() for ds in AU.DS_MAIN])
              for bi in range(N_PRIMARY)]
        entry["macro"] = dict(
            delta=float(np.mean([deltas[ds].mean()
                                 for ds in AU.DS_MAIN])) * 100.0,
            ci=[float(np.percentile(mv, 2.5)) * 100.0,
                float(np.percentile(mv, 97.5)) * 100.0])
        out["contrasts"][f"{a}-{b}"] = entry
    return out


def main():
    out = {}
    main_res = analyze_main_panel()
    if main_res:
        out["main"] = main_res
        out["main"]["rescue_break"] = {
            ds: rescue_break("main", "MAIN025", "BASE", ds)
            for ds in AU.DS_MAIN}
        out["main"]["coverage"] = coverage("main")
    else:
        out["main"] = dict(status="not_ready")
    nonreg = {}
    for ds in AU.DS_NONREG:
        pr = paired("nonreg", "MAIN025", "BASE", ds)
        if pr is None:
            nonreg[ds] = dict(status="not_ready")
            continue
        boots = Boot(pr["clusters"], BOOT_SEED)
        m, lo, hi = boot_ci(pr["diffs"], boots.draws(N_PRIMARY))
        nonreg[ds] = dict(n=pr["n"], n_images=pr["n_images"],
                          acc_new=pr["mean_a"], acc_old=pr["mean_b"],
                          delta=m, ci=[lo, hi])
    out["nonreg"] = nonreg
    fresh = analyze_fresh()
    if fresh:
        out["fresh"] = fresh
    for name in ("main", "nonreg", "fresh"):
        path = os.path.join(AU.OUT_DIR, f"analysis_{name}.json")
        blob = {name: out.get(name)}
        with open(path, "w") as f:
            json.dump(blob, f, indent=1)
        print(f"written {path}")
    if main_res:
        p = main_res["pooled"]
        print(f"\nPRIMARY (prereg): pooled Δ={p['delta']:.3f} "
              f"CI [{p['ci'][0]:.3f}, {p['ci'][1]:.3f}] "
              f"improvement={main_res['improvement_holds']} "
              f"both025_trigger={main_res['both025_trigger']}")
        for ds in AU.DS_MAIN:
            t = main_res["per_task"][ds]
            print(f"  {ds}: Δ={t['delta']:.3f} CI [{t['ci'][0]:.3f}, "
                  f"{t['ci'][1]:.3f}] (n={t['n']})")


if __name__ == "__main__":
    main()
