"""M12 analysis: accuracy tables + paired bootstrap contrasts.

Auto-discovers m12 shards (outputs/m12/acc) and the E0 baseline shards
(b0 K1024, b2 K256, rres256=side512 K256 in outputs/e0/acc).  Reuses
e0_analyze's official headline extraction and per-question backfill
(_pred_results.xlsx); paired contrasts vs b0 (K1024) and vs rres at the
same budget, question-level bootstrap 95% CI.

Writes outputs/m12/m12_analysis.json; prints a markdown table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

QWEN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(QWEN_ROOT, "scripts", "e0"))
from e0_analyze import official_main  # noqa: E402

OUT12 = os.path.join(QWEN_ROOT, "outputs", "m12")
OUTE0 = os.path.join(QWEN_ROOT, "outputs", "e0")
DS_LIST = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench", "ChartQA_TEST"]


def _per_q_from_detail(score_path: str, ds: str):
    """Per-question scores reproducing hit_calculate exactly.

    The _pred_results detail stores eval_match = list of per-gt scores
    (for DocVQA: anls DISTANCES).  Row order is the shard's sorted done
    positions 1:1 (the submission order is preserved by the evaluator).
    """
    import ast as _ast
    import pandas as pd
    tsv = score_path.replace("_score.json", "_pred.tsv")
    shard_p = score_path.replace("_score.json", ".json")
    if ds == "OCRBench":  # no per-question official dump; recompute the rule
        shard = json.load(open(shard_p))
        from vlmeval.dataset import build_dataset as _bd
        dso = _bd(ds)
        per_q = {}
        for k, rec in shard["records"].items():
            row = dso.data.iloc[int(k)]
            pred = str(rec["prediction"]).lower().strip()
            answers = _ast.literal_eval(row["answer"])
            hit = 0.0
            if row["category"] == "Handwritten Mathematical Expression Recognition":
                p2 = pred.replace("\n", " ").replace(" ", "")
                if any(a.strip().replace("\n", " ").replace(" ", "") in p2
                       for a in answers):
                    hit = 1.0
            elif any(a.lower().strip().replace("\n", " ") in pred
                     for a in answers):
                hit = 1.0
            per_q[str(k)] = hit
        return per_q
    for det in (tsv.replace(".tsv", "_results.xlsx"),
                tsv.replace(".tsv", "_results.tsv")):
        if not os.path.exists(det):
            continue
        d = pd.read_excel(det) if det.endswith("xlsx") else pd.read_csv(det, sep="\t")
        if "eval_match" not in d.columns:
            continue
        shard = json.load(open(shard_p))
        positions = sorted(int(k) for k in shard["records"])
        if len(positions) != len(d):
            return None
        per_q = {}
        for pos, (_, r) in zip(positions, d.iterrows()):
            m = r["eval_match"]
            if isinstance(m, str):
                m = _ast.literal_eval(m)
            m = list(m)
            if ds == "TextVQA_VAL":
                hit = float(np.mean(m))
            elif ds == "DocVQA_VAL":
                md = float(np.min(m))    # anls distance to the best gt
                hit = 0.0 if 1 - md < 0.5 else 1 - md
            elif ds == "ChartQA_TEST":
                hit = float(np.max(m))
            else:
                hit = float(np.mean(m))
            per_q[str(pos)] = hit
        return per_q
    return None


def load_entry(root, key_arm, key_K, out_arm, out_K, ds):
    d = os.path.join(root, "acc", key_arm, f"K{key_K}")
    p = os.path.join(d, f"{ds}_score.json")
    if not os.path.exists(p):
        return None
    s = json.load(open(p))
    try:
        off = s.get("official", {})
        if ds == "ChartQA_TEST":  # e0 headline order is dict-order fragile
            off = {"test_augmented": off.get("test_augmented",
                                             list(off.values())[-1])}
        head = official_main(ds, off)
        if ds == "OCRBench":    # report raw points (max = n), not the /10 norm
            head = float(off["Final Score"])
    except Exception:
        head = None
    pq = s.get("per_question")
    if not pq:
        try:
            pq = _per_q_from_detail(p, ds)
        except Exception:
            pq = None
    # consistency: mean(per_q) must reproduce the official headline
    if pq and head is not None:
        m = 100 * float(np.mean([float(v) for v in pq.values()]))
        ref = head
        if ds == "OCRBench":    # headline is raw points out of n
            ref = 100.0 * head / s.get("n", 164)
        if abs(m - ref) > 1.5:
            pq = None
    return dict(arm=out_arm, K=out_K, ds=ds, n=s.get("n"), score=head,
                per_q=pq)


def boot_ci(a, b, n_boot=10000, seed=0):
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    da = np.array([float(a[k]) for k in keys])
    db = np.array([float(b[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(da), size=(n_boot, len(da)))
    boots = (da - db)[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n=len(da), mean_diff=float((da - db).mean()),
                ci95=[float(lo), float(hi)], p_gt0=float((boots > 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT12, "m12_analysis.json"))
    args = ap.parse_args()

    entries = []
    acc_root = os.path.join(OUT12, "acc")
    if os.path.isdir(acc_root):
        for arm in sorted(os.listdir(acc_root)):
            for kdir in os.listdir(os.path.join(acc_root, arm)):
                if kdir.startswith("K"):
                    entries.append((OUT12, arm, int(kdir[1:]), arm,
                                    int(kdir[1:])))
    # E0 rres arm ids are SIDE lengths: rres512 = side 512 = 256 merged
    # tokens (RRES = {256: [512], 64: [256]} in e0_accuracy).
    for key_arm, out_arm, out_K in [("b0", "b0", 1024), ("b2", "b2", 256),
                                    ("rres512", "rres", 256)]:
        if os.path.isdir(os.path.join(OUTE0, "acc", key_arm)):
            entries.append((OUTE0, key_arm, 1024 if key_arm != "b2" else 256,
                            out_arm, out_K))

    rows = []
    for root, ka, kK, oa, oK in entries:
        for ds in DS_LIST:
            r = load_entry(root, ka, kK, oa, oK, ds)
            if r:
                rows.append(r)

    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["ds"], {})[f"{r['arm']}|K{r['K']}"] = r

    contrasts = []
    for r in rows:
        if r["arm"] == "b0" or not r.get("per_q"):
            continue
        for vs_arm, vs_K in (("b0", 1024), ("rres", r["K"])):
            ref = by_ds.get(r["ds"], {}).get(f"{vs_arm}|K{vs_K}")
            if ref and ref.get("per_q"):
                c = boot_ci(r["per_q"], ref["per_q"])
                if c:
                    contrasts.append(dict(arm=r["arm"], K=r["K"], ds=r["ds"],
                                          vs=vs_arm, vsK=vs_K, **c))

    lines = ["| ds | arm | K | n | score | Δ vs b0 (95% CI) | Δ vs rres@K (95% CI) |",
             "|---|---|---|---|---|---|---|"]
    for ds in DS_LIST:
        for key, r in sorted(by_ds.get(ds, {}).items(),
                             key=lambda kv: (kv[1]["K"], kv[1]["arm"])):
            def cell(vs):
                cs = [c for c in contrasts if c["ds"] == ds
                      and c["arm"] == r["arm"] and c["K"] == r["K"]
                      and c["vs"] == vs]
                if not cs:
                    return ""
                c = cs[0]
                scale = r["n"] if ds == "OCRBench" else 1.0  # raw points
                return (f"{scale * c['mean_diff']:+.1f} "
                        f"[{scale * c['ci95'][0]:+.1f},{scale * c['ci95'][1]:+.1f}]")
            lines.append(f"| {ds} | {r['arm']} | {r['K']} | {r['n']} | "
                         f"{round(r['score'], 2) if r['score'] is not None else '?'} "
                         f"| {cell('b0')} | {cell('rres')} |")
    table = "\n".join(lines)
    print(table)

    with open(args.out, "w") as f:
        json.dump(dict(rows=[{k: v for k, v in r.items()} for r in rows],
                       contrasts=contrasts, table=table), f, indent=1)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
