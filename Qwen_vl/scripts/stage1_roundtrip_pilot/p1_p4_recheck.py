"""P1/P4 results re-check (2026-10-07, closeout directive item 1).

Fixes applied vs the original analyses (egather_p1_analyze.py outputs):

  R1  P4 OCRBench E_GATHER baseline: the §13 table carried 620 while the
      audit per-question file (old_OCRBench_perq.csv.gz) means 0.6230 ->
      623 (the value §9/§12 used).  All P4 OCR effects recomputed with
      the single audited baseline.
  R2  P4 module effects (Completion / RTG / interaction) recomputed from
      ONE per-question source per arm, with image-cluster bootstrap CIs
      (TextVQA/DocVQA clusters reused; ChartQA upgraded from per-row to
      image clusters via md5 of the base64 image column).
  R3  InfoVQA paired delta: per-question OFFICIAL hit (hit_calculate
      semantics: 1-min(NLD) if >= 0.5 else 0, read from each arm's
      official _pred_results.xlsx eval_match) replacing the local
      extractor; verified to reproduce each arm's official headline.
  R4  ChartQA P1 CI upgraded to image clusters (was per-row).
  R5  MME (image-pair clusters) and MMBench (1292 circular groups)
      recomputed under the same framework for a single-source record.

All bootstrap: 5000 resamples, seed 20261007.  CPU only.
Usage: qwen3vl_clean python p1_p4_recheck.py
"""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
import pickle

import numpy as np
import pandas as pd

import rtg_common as RC

ROOT = os.path.join(RC.OUT_DIR, "legacy_full", "acc")
LR = os.path.join(ROOT, "L_R_MAIN025")
OLD = ("/media/disk2/YZX/research/audit_base_gap_20261003/"
       "archive_recovery/20261003_r2/rescore_detail")
N_BOOT = 5000
SEED = 20261007
OUT = os.path.join(RC.OUT_DIR, "legacy_full",
                   "p1_p4_recheck_20261008_cifix.json")

ARMS = ["L_E_GATHER", "L_E_MAIN025", "L_R_GATHER", "L_R_MAIN025"]


def arm_pq(arm, ds):
    """per-question dict {str(row_position): score} for an arm/dataset.

    Arms' score jsons key per_question by dataset row position (0..n-1,
    verified).  The audit per-question CSVs carry the vlmeval 'index'
    column, but their row ORDER is the dataset order, so they are keyed
    positionally here (same convention as egather_p1_analyze.row_scores).
    """
    if arm == "L_E_GATHER" and ds in ("TextVQA_VAL", "DocVQA_VAL",
                                      "OCRBench"):
        f = os.path.join(OLD, f"old_{ds}_perq.csv.gz")
        d = pd.read_csv(f)
        return {str(i): float(s) for i, s in enumerate(d["score"])}
    d = json.load(open(os.path.join(ROOT, arm, f"{ds}_score.json")))
    pq = d["per_question"]
    out = {str(k): float(v["score"] if isinstance(v, dict) else v)
           for k, v in pq.items()}
    if set(out) != {str(i) for i in range(len(out))}:
        # non-positional key space: sort by int and treat as positional
        out = {str(i): out[k] for i, k in
               enumerate(sorted(out, key=lambda x: int(x)))}
    return out


def image_clusters(ds, n):
    """cluster id per dataset row; md5 of base64 image when needed."""
    from vlmeval.dataset import build_dataset as vb
    d = vb(ds)
    col = None
    for c in ("image_path", "image_id", "image_key", "file_name"):
        if c in d.data.columns:
            col = c
            break
    if col is not None:
        vals = d.data[col].astype(str).tolist()[:n]
    elif "image" in d.data.columns:
        vals = [hashlib.md5(str(x).encode()).hexdigest()
                for x in d.data["image"].tolist()[:n]]
    else:
        return None
    while len(vals) < n:
        vals.append(str(len(vals)))
    uniq = {v: i for i, v in enumerate(sorted(set(vals)))}
    return [uniq[v] for v in vals]


def boot_delta(a, b, cl, rng):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    cl = np.asarray(cl)
    d = a - b
    uc = np.unique(cl)
    by = {u: np.where(cl == u)[0] for u in uc}
    stats = np.empty(N_BOOT)
    for r in range(N_BOOT):
        pick = rng.choice(uc, size=len(uc), replace=True)
        idx = np.concatenate([by[u] for u in pick])
        stats[r] = d[idx].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return dict(point=100.0 * float(d.mean()),
                ci=[100.0 * float(lo), 100.0 * float(hi)],
                n=int(len(d)), n_clusters=int(len(uc)))


def p4_effects():
    out = {}
    for ds, n in [("TextVQA_VAL", 5000), ("ChartQA_TEST", 2500),
                  ("DocVQA_VAL", 5349)]:
        pq = {a: arm_pq(a, ds) for a in ARMS}
        for a in ARMS:
            assert len(pq[a]) == n, (a, ds, len(pq[a]))
        rows = sorted(int(k) for k in pq[ARMS[0]])
        S = {a: np.array([pq[a][str(i)] for i in rows]) for a in ARMS}
        cl = image_clusters(ds, n)
        per_row = cl is None
        if per_row:
            cl = list(range(n))
        res = {"n": n,
               "clusters": "per-row" if per_row
               else f"image ({len(set(cl))} unique)"}
        eg = S["L_E_GATHER"]
        pairs = {
            "completion_effect": ("L_E_MAIN025", eg),
            "rtg_effect": ("L_R_GATHER", eg),
            "interaction": None,
        }
        rng = np.random.default_rng(SEED)
        # completion / RTG effects
        for k, (x, y) in list(pairs.items())[:2]:
            res[k] = boot_delta(S[x], y, cl, rng)
        # interaction = (R_MAIN025 - R_GATHER) - (E_MAIN025 - E_GATHER)
        d1 = S["L_R_MAIN025"] - S["L_R_GATHER"]
        d2 = S["L_E_MAIN025"] - eg
        res["interaction"] = boot_delta(d1, d2, cl, rng)
        # aggregate means for the table
        res["means"] = {a: 100.0 * float(S[a].mean()) for a in ARMS}
        out[ds] = res

    # OCRBench: aggregate point estimates (official Final Score semantics;
    # per-row 0/1 audit scores reproduce the official sum for the audit
    # arm).  Effects from the single audited baseline 623.
    ocr = {a: arm_pq(a, "OCRBench") for a in ARMS}
    M = {a: 100.0 * float(np.mean(list(ocr[a].values()))) for a in ARMS}
    out["OCRBench"] = {
        "n": 1000, "means": M,
        "audited_eg_baseline": M["L_E_GATHER"],
        "completion_effect": M["L_E_MAIN025"] - M["L_E_GATHER"],
        "rtg_effect": M["L_R_GATHER"] - M["L_E_GATHER"],
        "interaction": (M["L_R_MAIN025"] - M["L_R_GATHER"])
                       - (M["L_E_MAIN025"] - M["L_E_GATHER"]),
        "clusters": "per-row (OCRBench rows are independent questions)",
        "note": "§13's E_GATHER=620 was inconsistent with the audited "
                "per-question file (mean 0.6230 -> 623); corrected here.",
    }
    return out


def infovqa_official_pairing():
    """official per-question hit from each arm's _pred_results.xlsx."""
    import warnings
    warnings.filterwarnings("ignore")

    def official_hits(arm):
        f = os.path.join(ROOT, arm, "InfoVQA_VAL_pred_results.xlsx")
        d = pd.read_excel(f)
        hits = []
        for m in d["eval_match"]:
            nld = ast.literal_eval(m) if isinstance(m, str) else m
            nld = [float(x) for x in nld]
            best = 1.0 - min(nld)
            hits.append(best if best >= 0.5 else 0.0)
        return np.array(hits), d

    hl, dl = official_hits("L_R_MAIN025")
    he, de = official_hits("L_E_GATHER")
    assert len(hl) == len(he) == 2801
    head_l = _parse_overall(os.path.join(ROOT, "L_R_MAIN025",
                                         "InfoVQA_VAL_score.json"))
    head_e = _parse_overall(os.path.join(ROOT, "L_E_GATHER",
                                         "InfoVQA_VAL_score.json"))
    cl = image_clusters("InfoVQA_VAL", 2801)
    if cl is None:
        cl = list(range(2801))
    rng = np.random.default_rng(SEED)
    boot = boot_delta(hl, he, cl, rng)
    return {
        "n": 2801,
        "lr_official_mean": 100.0 * float(hl.mean()),
        "lr_headline": head_l,
        "lr_reproduces": bool(abs(100.0 * hl.mean() - head_l) <= 0.05),
        "eg_official_mean": 100.0 * float(he.mean()),
        "eg_headline": head_e,
        "eg_reproduces": bool(abs(100.0 * he.mean() - head_e) <= 0.05),
        "delta_pct": boot["point"], "ci": boot["ci"],
        "n_clusters": boot["n_clusters"],
        "note": "R3: official hit_calculate semantics from each arm's "
                "_pred_results.xlsx eval_match; replaces the local-"
                "extractor pairing in the original analysis.",
    }


def _parse_overall(path):
    # Overall is already a percent number in the score json (verified:
    # ChartQA Overall '42.64'); do NOT rescale.
    v = json.load(open(path))["official"]["Overall"]
    if isinstance(v, str) and v.startswith("{"):
        v = ast.literal_eval(v)
        v = next(iter(v.values()))
    return float(str(v).split(":")[-1].strip(" {}'\""))


def chartqa_reclustered():
    pqL = arm_pq("L_R_MAIN025", "ChartQA_TEST")
    pqE = arm_pq("L_E_GATHER", "ChartQA_TEST")
    rows = sorted(int(k) for k in pqL)
    a = np.array([pqL[str(i)] for i in rows])
    b = np.array([pqE[str(i)] for i in rows])
    cl = image_clusters("ChartQA_TEST", len(rows))
    rng = np.random.default_rng(SEED)
    boot = boot_delta(a, b, cl, rng)
    return {"n": len(rows), "clusters": f"image ({boot['n_clusters']} "
                                        "unique, md5 of base64)",
            "lr_mean": 100.0 * float(a.mean()),
            "eg_mean": 100.0 * float(b.mean()),
            "delta_pct": boot["point"], "ci": boot["ci"],
            "note": "R4: was per-row in the original analysis."}


def mme_recheck():
    def pairs(arm):
        d = json.load(open(os.path.join(
            ROOT, arm, "MME_score.json")))["per_question"]
        by = {}
        for k, v in d.items():
            by.setdefault((v["category"], v["image_path"]),
                          []).append(float(v["score"]))
        return by

    pL, pE = pairs("L_R_MAIN025"), pairs("L_E_GATHER")
    assert list(pL) == list(pE)

    def totals(items):
        # BUGFIX 2026-10-08: keying by (cat, img) alone silently OVERWROTE
        # image pairs picked more than once by the bootstrap (duplicate
        # clusters collapsed, CI too narrow).  Keying by occurrence index
        # keeps multiplicity; identical for the point estimate (unique keys).
        g = {}
        for i, ((cat, img), ss) in enumerate(items):
            g.setdefault(cat, {})[(img, i)] = ss
        tot = 0.0
        for cat, dg in g.items():
            acc = np.mean([np.mean(v) for v in dg.values()])
            accp = np.mean([np.prod(v) for v in dg.values()])
            tot += (acc + accp) * 100.0
        return tot

    itL, itE = list(pL.items()), list(pE.items())
    rng = np.random.default_rng(SEED)
    stats = np.empty(N_BOOT)
    for r in range(N_BOOT):
        pick = rng.choice(len(itL), size=len(itL), replace=True)
        stats[r] = totals([itL[i] for i in pick]) \
            - totals([itE[i] for i in pick])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"n": len(itL) * 2, "n_clusters": len(itL),
            "lr_total": totals(itL), "eg_total": totals(itE),
            "delta": totals(itL) - totals(itE),
            "ci": [float(lo), float(hi)],
            "note": "R5: image-pair clusters (recomputed under the "
                    "recheck seed for a single-source record)."}


def mmb_recheck(ds):
    def hits(arm):
        with open(os.path.join(ROOT, arm,
                               f"{ds}_pred_exact_matching_result.pkl"),
                  "rb") as f:
            r = pickle.load(f)
        return r

    rL, rE = hits("L_R_MAIN025"), hits("L_E_GATHER")
    # full official denominator: 1292 circular groups (index % 1e6 over
    # all 4876 V11 rows); evaluation-failed groups carry hit=0 both arms
    from vlmeval.dataset import build_dataset as vb
    idx = vb(ds).data["index"].astype("int64").tolist()
    full = sorted({int(i) % 1000000 for i in idx})
    keys = sorted(set(full) | set(rL) | set(rE))
    hL = np.array([rL.get(k, {}).get("hit", 0) for k in keys], float)
    hE = np.array([rE.get(k, {}).get("hit", 0) for k in keys], float)
    rng = np.random.default_rng(SEED)
    stats = np.empty(N_BOOT)
    for r in range(N_BOOT):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        stats[r] = hL[pick].mean() - hE[pick].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"n_groups": len(keys), "lr_correct": int(hL.sum()),
            "eg_correct": int(hE.sum()),
            "delta_pct": 100.0 * float(hL.mean() - hE.mean()),
            "ci": [100.0 * float(lo), 100.0 * float(hi)],
            "note": "R5: full official circular-group denominator (hit=0 for "
                    "evaluation-failed groups, official semantics)."}


def main():
    out = {
        "meta": {"seed": SEED, "n_boot": N_BOOT,
                 "fixes": ["R1 ocrbench 620->623",
                           "R2 p4 effects unified+cluster CI",
                           "R3 infovqa official ANLS pairing",
                           "R4 chartqa image clusters",
                           "R5 mme/mmb single-source recompute"]},
        "p4_effects": p4_effects(),
        "infovqa_official_pairing": infovqa_official_pairing(),
        "chartqa_reclustered": chartqa_reclustered(),
        "mme": mme_recheck(),
        "mmbench_en": mmb_recheck("MMBench_DEV_EN_V11"),
        "mmbench_cn": mmb_recheck("MMBench_DEV_CN_V11"),
    }
    import numpy as _np
    json.dump(out, open(OUT, "w"), indent=1, default=lambda o: bool(o) if isinstance(o, _np.bool_) else float(o))
    print("written", OUT)
    # console summary
    print("\n== P4 effects (corrected) ==")
    for ds, r in out["p4_effects"].items():
        if ds == "OCRBench":
            print(f"OCRBench: EG={r['audited_eg_baseline']:.1f} "
                  f"C={r['completion_effect']:+.1f} "
                  f"R={r['rtg_effect']:+.1f} I={r['interaction']:+.1f}")
        else:
            print(f"{ds}: C={r['completion_effect']['point']:+.3f} "
                  f"{r['completion_effect']['ci']} "
                  f"R={r['rtg_effect']['point']:+.3f} "
                  f"{r['rtg_effect']['ci']} "
                  f"I={r['interaction']['point']:+.3f} "
                  f"{r['interaction']['ci']} [{r['clusters']}]")
    iv = out["infovqa_official_pairing"]
    print(f"\nInfoVQA official: lr={iv['lr_official_mean']:.2f} "
          f"(headline {iv['lr_headline']}, repro={iv['lr_reproduces']}) "
          f"eg={iv['eg_official_mean']:.2f} "
          f"(headline {iv['eg_headline']}, repro={iv['eg_reproduces']}) "
          f"delta={iv['delta_pct']:+.3f} CI={iv['ci']}")
    c = out["chartqa_reclustered"]
    print(f"ChartQA reclustered: delta={c['delta_pct']:+.3f} CI={c['ci']}")
    print(f"MME: delta={out['mme']['delta']:+.2f} CI={out['mme']['ci']}")
    for k in ("mmbench_en", "mmbench_cn"):
        m = out[k]
        print(f"{k}: delta={m['delta_pct']:+.3f} CI={m['ci']} "
              f"({m['n_groups']} groups)")


if __name__ == "__main__":
    main()
