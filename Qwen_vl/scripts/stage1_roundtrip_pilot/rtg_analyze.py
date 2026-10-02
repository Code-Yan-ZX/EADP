"""RTG pilot — analysis & frozen statistics (protocol §6).

Cluster bootstrap 20000x, seed=20261002, shared resampling indices across
ALL predefined contrasts; per-task + equal-weight macro delta/95% CI on the
0-100 scale.  Contrasts:
  R_MAIN025-E_MAIN025 (primary), R_GATHER-E_GATHER, R_MAIN025-F_MAIN025,
  R_MAIN025-S_MAIN025, R_MAIN025-R_GATHER, 2x2 interaction.
GO gate (mechanical, frozen): R_MAIN025-E_MAIN025 macro >= -0.5; DocVQA
point >= -1.0; R macro strictly > F and S macro; TTFT med increment <= 5%.

Usage: python rtg_analyze.py
"""

from __future__ import annotations

import json
import os

import numpy as np

import rtg_common as RC


def load_scores(arm: str, ds: str):
    p = RC.shard_path(arm, ds).replace(".json", "_score.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d if d.get("per_question") else None


AC_MANIFEST_DIR = os.path.join(RC.common.QWEN_ROOT, "outputs",
                               "anchor_merge_pilot")


def cluster_map(ds: str, keys):
    """image_key per row from the round-1 frozen bank (same image rule)."""
    with __import__("gzip").open(
            os.path.join(AC_MANIFEST_DIR, f"bank_dev_{ds}.json.gz"),
            "rt") as f:
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


def boot_ci(diffs, idx_sets):
    vals = np.array([float(diffs[idx].mean()) for idx in idx_sets])
    return (float(diffs.mean()) * 100.0,
            float(np.percentile(vals, 2.5)) * 100.0,
            float(np.percentile(vals, 97.5)) * 100.0)


def contrast(a: str, b: str, idx_sets, per_task, deltas):
    entry = dict(per_task={})
    for ds in RC.DS_LIST:
        m, lo, hi = boot_ci(deltas[ds], idx_sets[ds])
        pr = per_task[ds]
        entry["per_task"][ds] = dict(
            n=pr["n"], n_images=pr["n_images"],
            acc_a=pr["mean_a"], acc_b=pr["mean_b"],
            delta=m, ci=[lo, hi])
    mv = [np.mean([deltas[ds][idx_sets[ds][bi]].mean() for ds in RC.DS_LIST])
          for bi in range(len(idx_sets[RC.DS_LIST[0]]))]
    entry["macro"] = dict(
        delta=float(np.mean([deltas[ds].mean() for ds in RC.DS_LIST])) * 100.0,
        ci=[float(np.percentile(mv, 2.5)) * 100.0,
            float(np.percentile(mv, 97.5)) * 100.0])
    return entry


def build_paired(arm_a, arm_b):
    per_task, boots, deltas = {}, {}, {}
    for ds in RC.DS_LIST:
        sa, sb = load_scores(arm_a, ds), load_scores(arm_b, ds)
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
            mean_a=100.0 * float(np.mean([sa["per_question"][k]
                                          for k in keys])),
            mean_b=100.0 * float(np.mean([sb["per_question"][k]
                                          for k in keys])))
        deltas[ds] = diffs
        boots[ds] = Boot(clusters, RC.BOOT_SEED)
    idx_sets = {ds: boots[ds].draws(RC.N_BOOT) for ds in RC.DS_LIST}
    return per_task, deltas, idx_sets


def main():
    out = dict(n_boot=RC.N_BOOT, seed=RC.BOOT_SEED, contrasts={})
    # shared index sets per dataset across ALL contrasts (frozen rule)
    base = build_paired("R_MAIN025", "E_MAIN025")
    if base is None:
        out["status"] = "not_ready"
        with open(os.path.join(RC.OUT_DIR, "analysis.json"), "w") as f:
            json.dump(out, f, indent=1)
        print("analysis: not ready (missing score cells)")
        return
    per_task, deltas, idx_sets = base
    out["per_task_scores"] = {}
    for ds in RC.DS_LIST:
        out["per_task_scores"][ds] = {}
        for arm in ("E_GATHER", "E_MAIN025", "R_GATHER", "R_MAIN025",
                    "F_MAIN025", "S_MAIN025"):
            s = load_scores(arm, ds)
            if s:
                out["per_task_scores"][ds][arm] = dict(
                    n=len(s["per_question"]),
                    acc=100.0 * float(np.mean(
                        [float(v) for v in s["per_question"].values()])))
    out["contrasts"]["R_MAIN025-E_MAIN025"] = contrast(
        "R_MAIN025", "E_MAIN025", idx_sets, per_task, deltas)
    for a, b in (("R_GATHER", "E_GATHER"), ("R_MAIN025", "F_MAIN025"),
                 ("R_MAIN025", "S_MAIN025"), ("R_MAIN025", "R_GATHER")):
        pb = build_paired(a, b)
        if pb is None:
            out["contrasts"][f"{a}-{b}"] = dict(status="missing_cells")
            continue
        pt2, d2, _ = pb
        out["contrasts"][f"{a}-{b}"] = contrast(a, b, idx_sets, pt2, d2)
    # 2x2 interaction: (R_MAIN025-R_GATHER) - (E_MAIN025-E_GATHER)
    p3 = build_paired("E_MAIN025", "E_GATHER")
    if p3 is not None:
        pt3, d3, _ = p3
        inter = {}
        mv = []
        for ds in RC.DS_LIST:
            d_inter = deltas[ds] - d3[ds]
            m = float(d_inter.mean()) * 100.0
            vals = np.array([d_inter[idx_sets[ds][bi]].mean() * 100.0
                             for bi in range(RC.N_BOOT)])
            inter[ds] = dict(delta=m,
                             ci=[float(np.percentile(vals, 2.5)),
                                 float(np.percentile(vals, 97.5))])
            mv.append([d_inter[idx_sets[ds][bi]].mean() for bi in
                       range(RC.N_BOOT)])
        macro_vals = np.mean(np.stack(mv, axis=0), axis=0)
        out["contrasts"]["2x2_interaction"] = dict(
            per_task=inter,
            macro=dict(delta=float(macro_vals.mean()) * 100.0,
                       ci=[float(np.percentile(macro_vals, 2.5)) * 100.0,
                           float(np.percentile(macro_vals, 97.5)) * 100.0]))
    # diagnostics: effective tokens / weight stats from banks
    diag = {}
    for scorer in RC.SCORERS:
        effs, Ls = [], []
        for ds in RC.DS_LIST:
            try:
                bank = RC.load_bank(scorer, ds)
            except FileNotFoundError:
                continue
            for rec in bank["samples"].values():
                wd = rec.get("w_diag")
                if wd and wd.get("eff_tokens"):
                    effs.append(wd["eff_tokens"])
                    Ls.append(wd.get("L", 0))
        if effs:
            diag[scorer] = dict(eff_tokens_mean=float(np.mean(effs)),
                                eff_tokens_p50=float(np.median(effs)),
                                L_mean=float(np.mean(Ls)))
    out["weight_diagnostics"] = diag
    # anchors overlap vs the EADP bank (round-1 frozen b1)
    try:
        import gzip as _gz
        ov = {}
        for ds in RC.DS_LIST:
            with _gz.open(os.path.join(AC_MANIFEST_DIR,
                                       f"bank_dev_{ds}.json.gz"), "rt") as f:
                ebank = json.load(f)
            inter, union, n = 0, 0, 0
            for scorer in RC.SCORERS:
                b = RC.load_bank(scorer, ds)
                for key, rec in b["samples"].items():
                    if key not in ebank:
                        continue
                    e_set = set(ebank[key]["keep"])
                    s_set = set(rec["keep"])
                    inter += len(e_set & s_set)
                    union += len(e_set | s_set)
                    n += 1
            ov[ds] = dict(jaccard=round(inter / max(1, union), 4),
                          n_samples=n)
        out["anchors_overlap_vs_eadp"] = ov
    except Exception as e:  # noqa: BLE001
        out["anchors_overlap_vs_eadp"] = dict(error=str(e))
    # GO gate (mechanical; TTFT condition filled by perf step)
    c = out["contrasts"].get("R_MAIN025-E_MAIN025", {})
    go12 = None
    if c and "macro" in c:
        macro_d = c["macro"]["delta"]
        dv = c["per_task"]["DocVQA_VAL"]["delta"]
        f_m = out["contrasts"].get("R_MAIN025-F_MAIN025", {}).get("macro")
        s_m = out["contrasts"].get("R_MAIN025-S_MAIN025", {}).get("macro")
        go1 = macro_d >= -0.5
        go2 = dv >= -1.0
        # frozen rule: R macro strictly higher than F and S macros
        # (point estimates; CI crossing zero -> mechanism hint only)
        go3 = bool(f_m and s_m and f_m["delta"] > 0 and s_m["delta"] > 0)
        out["go_gate"] = dict(cond1_macro_ge_m05=go1, cond2_docvqa_ge_m1=go2,
                              cond3_beats_F_and_S=go3,
                              cond4_ttft_le_5pct=None, go=None,
                              macro_delta=macro_d, docvqa_delta=dv)
    # GO gate condition 4 from rtg_perf (if it exists)
    perf_p = os.path.join(RC.OUT_DIR, "rtg_perf.json")
    if os.path.exists(perf_p):
        pf = json.load(open(perf_p))
        if "go_gate" in out:
            out["go_gate"]["cond4_ttft_le_5pct"] = bool(
                pf.get("ttft_increment_pct", 99.0) <= 5.0)
            out["go_gate"]["ttft_increment_pct"] = pf.get("ttft_increment_pct")
            g = out["go_gate"]
            out["go_gate"]["go"] = bool(g["cond1_macro_ge_m05"]
                                        and g["cond2_docvqa_ge_m1"]
                                        and g["cond3_beats_F_and_S"]
                                        and g["cond4_ttft_le_5pct"])
    with open(os.path.join(RC.OUT_DIR, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("written analysis.json")
    for k, v in out["contrasts"].items():
        if "macro" in v:
            print(f"  {k}: macro Δ={v['macro']['delta']:.3f} "
                  f"CI [{v['macro']['ci'][0]:.3f}, {v['macro']['ci'][1]:.3f}]")


if __name__ == "__main__":
    main()
