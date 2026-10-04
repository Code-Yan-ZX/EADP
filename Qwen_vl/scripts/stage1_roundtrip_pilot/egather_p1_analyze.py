"""P1 analysis — L_R_MAIN025 vs E_GATHER (official_legacy, same protocol).

Per-task paired deltas and image/group-clustered bootstrap CIs.

E-side per-question sources:
  * TextVQA/DocVQA/OCRBench: audit-recovered official EADP predictions
    (rescore_detail/old_*_perq.csv.gz; byte-identical official code path,
    independent rescore diff=0 — see audit_base_gap_20261003).
  * ChartQA/AI2D/HallusionBench/MME/MMB-EN/MMB-CN/InfoVQA: L_E_GATHER
    official-legacy run (this round; same rows/prompts/scoring as
    L_R_MAIN025).

Pairing: row-aligned by dataset row index (both arms cover the full
official row sets; a per-task mean/coverage assertion is recorded).
Cluster bootstrap (5000 resamples, seed 20261004) resamples clusters —
image rows for VQA-style sets, figure groups for HallusionBench (fAcc),
image pairs for MME (acc+/acc pair rule), question groups for MMBench
(1292 circular groups) — and recomputes each arm's official aggregate
within the replicate.

NOTE on avg: deltas for the fixed 10-column average use the verified
paper formula (8 pct cols + OCR/10 + MME/20)/10.

Usage: python egather_p1_analyze.py
"""

from __future__ import annotations

import gzip
import json
import os
import pickle

import numpy as np
import pandas as pd

import rtg_common as RC

ROOT = os.path.join(RC.OUT_DIR, "legacy_full", "acc")
LR = os.path.join(ROOT, "L_R_MAIN025")
EG = os.path.join(ROOT, "L_E_GATHER")
OLD = ("/media/disk2/YZX/research/audit_base_gap_20261003/"
       "archive_recovery/20261003_r2/rescore_detail")
N_BOOT = 5000
SEED = 20261004

PCT_DS = ["TextVQA_VAL", "ChartQA_TEST", "AI2D_TEST", "HallusionBench",
          "MMBench_DEV_EN_V11", "MMBench_DEV_CN_V11", "DocVQA_VAL",
          "InfoVQA_VAL"]


def lr_perq(ds):
    d = json.load(open(os.path.join(LR, f"{ds}_score.json")))
    return d["per_question"], d["n"]


def eg_perq(ds):
    d = json.load(open(os.path.join(EG, f"{ds}_score.json")))
    return d["per_question"], d["n"]


def to_scores(pq):
    """per-question dict -> ordered list by int(row) (or grouped key)."""
    out = {}
    for k, v in pq.items():
        s = v["score"] if isinstance(v, dict) else float(v)
        out[k] = float(s)
    return out


def row_scores(pq):
    sc = to_scores(pq)
    return [sc[str(i)] for i in range(len(sc))]


def clusters_from_dataset(ds, n):
    """image cluster id per row for VQA-style datasets.

    Falls back to per-row clusters when no cheap non-binary image key
    column exists (recorded in the task note by the caller)."""
    from vlmeval.dataset import build_dataset as vb
    d = vb(ds)
    col = None
    for c in ("image_path", "image_id", "image_key", "file_name"):
        if c in d.data.columns:
            col = c
            break
    if col is None:
        return None                  # caller: per-row clusters
    vals = d.data[col].astype(str).tolist()[:n]
    while len(vals) < n:
        vals.append(str(len(vals)))
    uniq = {v: i for i, v in enumerate(sorted(set(vals)))}
    return [uniq[v] for v in vals]


def boot_mean_delta(a, b, cl, rng):
    """paired mean(a-b) CI, resampling clusters with replacement."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    cl = np.asarray(cl)
    d = a - b
    point = float(d.mean())
    uc = np.unique(cl)
    by = {u: np.where(cl == u)[0] for u in uc}
    stats = np.empty(N_BOOT)
    for r in range(N_BOOT):
        pick = rng.choice(uc, size=len(uc), replace=True)
        idx = np.concatenate([by[u] for u in pick])
        stats[r] = d[idx].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return dict(point=point, ci=[float(lo), float(hi)], n=int(len(d)),
                n_clusters=int(len(uc)))


def analyze_vqa_style(ds, eg_source):
    lr_pq, n_lr = lr_perq(ds)
    lr = row_scores(lr_pq)
    if eg_source == "old":
        f = os.path.join(OLD, f"old_{ds}_perq.csv.gz")
        old = pd.read_csv(f)
        e = old["score"].astype(float).tolist()
        eg_n = len(e)
    else:
        e_pq, eg_n = eg_perq(ds)
        e = row_scores(e_pq)
    assert n_lr == eg_n, f"{ds}: coverage mismatch {n_lr} vs {eg_n}"
    cl = clusters_from_dataset(ds, n_lr)
    note = ("E-side = audit-recovered official EADP predictions"
            if eg_source == "old" else "E-side = L_E_GATHER")
    if cl is None:
        cl = list(range(n_lr))
        note += "; clusters = per-row (no image key column available)"
    rng = np.random.default_rng(SEED)
    boot = boot_mean_delta(lr, e, cl, rng)
    return dict(n=n_lr, lr_mean=100.0 * float(np.mean(lr)),
                eg_mean=100.0 * float(np.mean(e)),
                delta_pct=100.0 * boot["point"], boot=boot, note=note)


def analyze_hallusion():
    lr_pq, n = lr_perq("HallusionBench")
    eg_pq, n_e = eg_perq("HallusionBench")
    assert n == n_e == 951

    def groups(pq):
        g_f, rows = {}, []
        for k, v in pq.items():
            key = f"{v['l2_category']}_{v['set_id']}_{v['figure_id']}"
            g_f.setdefault(key, []).append(float(v["score"]))
            rows.append((key, float(v["score"])))
        return g_f, rows

    gL, rowsL = groups(lr_pq)
    gE, rowsE = groups(eg_pq)
    assert list(gL) == list(gE)

    def facc(g):
        return 100.0 * float(np.mean([np.all(v) for v in g.values()]))

    point = facc(gL) - facc(gE)
    keys = list(gL)
    rng = np.random.default_rng(SEED)
    stats = np.empty(N_BOOT)
    for r in range(N_BOOT):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        fl = [np.all(gL[keys[i]]) for i in pick]
        fe = [np.all(gE[keys[i]]) for i in pick]
        stats[r] = (100.0 * float(np.mean(fl))
                    - 100.0 * float(np.mean(fe)))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return dict(n=n, lr_facc=facc(gL), eg_facc=facc(gE), delta=point,
                ci=[float(lo), float(hi)], n_clusters=len(keys),
                lr_aAcc=100.0 * float(np.mean([v["score"] for v in
                                               lr_pq.values()])))


def analyze_mme():
    lr_pq, n = lr_perq("MME")
    eg_pq, n_e = eg_perq("MME")
    assert n == n_e == 2374

    def pairs(pq):
        by_img = {}
        for k, v in pq.items():
            by_img.setdefault((v["category"], v["image_path"]),
                              []).append(float(v["score"]))
        return by_img

    pL, pE = pairs(lr_pq), pairs(eg_pq)
    assert list(pL) == list(pE)
    PER = ["OCR", "artwork", "celebrity", "color", "count", "existence",
           "landmark", "position", "posters", "scene"]
    cats = list({c for c, _ in pL})

    def totals(items):
        stats = {}
        for (cat, img), ss in items:
            stats.setdefault(cat, {})[img] = ss
        tot = 0.0
        for cat, g in stats.items():
            acc = np.mean([np.mean(v) for v in g.values()])
            accp = np.mean([np.prod(v) for v in g.values()])
            tot += (acc + accp) * 100.0
        return tot

    itemsL = list(pL.items())
    itemsE = list(pE.items())
    point = totals(itemsL) - totals(itemsE)
    rng = np.random.default_rng(SEED)
    stats = np.empty(N_BOOT)
    for r in range(N_BOOT):
        pick = rng.choice(len(itemsL), size=len(itemsL), replace=True)
        stats[r] = totals([itemsL[i] for i in pick]) - \
            totals([itemsE[i] for i in pick])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return dict(n=n, lr_total=totals(itemsL), eg_total=totals(itemsE),
                delta=point, ci=[float(lo), float(hi)],
                n_clusters=len(itemsL))


def _parse_official_overall(sj):
    v = sj["official"]["Overall"]
    if isinstance(v, dict):
        v = next(iter(v.values()))
    return 100.0 * float(str(v).split(":")[-1].strip(" {}'\""))


def analyze_mmb(ds):
    sjL = json.load(open(os.path.join(LR, f"{ds}_score.json")))
    sjE = json.load(open(os.path.join(EG, f"{ds}_score.json")))
    with open(os.path.join(LR, f"{ds}_pred_exact_matching_result.pkl"),
              "rb") as f:
        rL = pickle.load(f)
    with open(os.path.join(EG, f"{ds}_pred_exact_matching_result.pkl"),
              "rb") as f:
        rE = pickle.load(f)
    keys = sorted(set(rL) | set(rE))
    # groups missing from a pkl failed evaluation and are counted hit=0
    # by the official headline (bit-exact recompute verified in P0)
    hL = np.array([rL.get(k, {}).get("hit", 0) for k in keys],
                  dtype=float)
    hE = np.array([rE.get(k, {}).get("hit", 0) for k in keys],
                  dtype=float)
    point = float(hL.mean() - hE.mean())
    rng = np.random.default_rng(SEED)
    stats = np.empty(N_BOOT)
    for r in range(N_BOOT):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        stats[r] = hL[pick].mean() - hE[pick].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return dict(n_questions=len(keys),
                lr_headline=_parse_official_overall(sjL),
                lr_correct=int(hL.sum()), eg_correct=int(hE.sum()),
                delta_pct=100.0 * point, ci=[100.0 * float(lo),
                                             100.0 * float(hi)],
                note="clusters = circular question groups (1292); "
                     "groups that failed evaluation carry hit=0 in "
                     "both arms (official semantics)")


def main():
    out = dict(arm_lr="L_R_MAIN025", arm_eg="E_GATHER (official EADP)",
               pipeline="official_legacy", n_boot=N_BOOT, seed=SEED,
               tasks={})

    for ds in ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]:
        out["tasks"][ds] = analyze_vqa_style(ds, "old")
        print(f"[{ds}] delta={out['tasks'][ds]['delta_pct']:+.3f} "
              f"CI={out['tasks'][ds]['boot']['ci']}")
    for ds in ["ChartQA_TEST", "AI2D_TEST", "InfoVQA_VAL"]:
        out["tasks"][ds] = analyze_vqa_style(ds, "egather")
        print(f"[{ds}] delta={out['tasks'][ds]['delta_pct']:+.3f} "
              f"CI={out['tasks'][ds]['boot']['ci']}")
    out["tasks"]["HallusionBench"] = analyze_hallusion()
    out["tasks"]["MME"] = analyze_mme()
    print(f"[HallusionBench] fAcc delta="
          f"{out['tasks']['HallusionBench']['delta']:+.3f} "
          f"CI={out['tasks']['HallusionBench']['ci']}")
    print(f"[MME] total delta={out['tasks']['MME']['delta']:+.3f} "
          f"CI={out['tasks']['MME']['ci']}")
    for ds in ["MMBench_DEV_EN_V11", "MMBench_DEV_CN_V11"]:
        out["tasks"][ds] = analyze_mmb(ds)
        print(f"[{ds}] delta={out['tasks'][ds]['delta_pct']:+.3f} "
              f"CI={out['tasks'][ds]['ci']}")

    # fixed 10-column averages per verified formula
    def avg10(vals):
        return (sum(vals[c] for c in
                    ["TextVQA", "ChartQA", "AI2D", "HallusionBench",
                     "MMB_EN", "MMB_CN", "DocVQA", "InfoVQA"])
                + vals["OCRBench"] / 10.0 + vals["MME"] / 20.0) / 10.0

    t = out["tasks"]
    lr_vals = dict(TextVQA=t["TextVQA_VAL"]["lr_mean"],
                   ChartQA=t["ChartQA_TEST"]["lr_mean"],
                   AI2D=t["AI2D_TEST"]["lr_mean"],
                   HallusionBench=t["HallusionBench"]["lr_facc"],
                   MMB_EN=t["MMBench_DEV_EN_V11"]["lr_headline"],
                   MMB_CN=t["MMBench_DEV_CN_V11"]["lr_headline"],
                   DocVQA=t["DocVQA_VAL"]["lr_mean"],
                   InfoVQA=t["InfoVQA_VAL"]["lr_mean"],
                   OCRBench=t["OCRBench_VAL"]["lr_mean"]
                   if "OCRBench_VAL" in t else
                   t["OCRBench"]["lr_mean"] * 10.0,
                   MME=t["MME"]["lr_total"])
    eg_vals = dict(TextVQA=t["TextVQA_VAL"]["eg_mean"],
                   ChartQA=t["ChartQA_TEST"]["eg_mean"],
                   AI2D=t["AI2D_TEST"]["eg_mean"],
                   HallusionBench=t["HallusionBench"]["eg_facc"],
                   MMB_EN=100.0 * (t["MMBench_DEV_EN_V11"]["eg_correct"]
                                   / t["MMBench_DEV_EN_V11"]["n_questions"]),
                   MMB_CN=100.0 * (t["MMBench_DEV_CN_V11"]["eg_correct"]
                                   / t["MMBench_DEV_CN_V11"]["n_questions"]),
                   DocVQA=t["DocVQA_VAL"]["eg_mean"],
                   InfoVQA=t["InfoVQA_VAL"]["eg_mean"],
                   OCRBench=t["OCRBench"]["eg_mean"] * 10.0,
                   MME=t["MME"]["eg_total"])
    out["avg10"] = dict(
        formula="(8 pct + OCR/10 + MME/20)/10 (p0-audit verified)",
        lr=avg10(lr_vals), eg=avg10(eg_vals),
        delta=avg10(lr_vals) - avg10(eg_vals),
        lr_columns=lr_vals, eg_columns=eg_vals)
    p = os.path.join(RC.OUT_DIR, "legacy_full",
                     "p1_egather_vs_lrmain_analysis.json")
    with open(p + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(p + ".tmp", p)
    print(f"[saved] {p}")
    print(f"Avg10: L_R_MAIN025={out['avg10']['lr']:.3f} "
          f"E_GATHER={out['avg10']['eg']:.3f} "
          f"delta={out['avg10']['delta']:+.3f}")


if __name__ == "__main__":
    main()
