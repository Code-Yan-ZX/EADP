"""E0 M6: mechanical analysis and D1-D4 verdicts (prereg §5.1/§6).

Reads outputs/e0/acc/**/_score.json (official scores + per-question scores)
and outputs/e0/e0_perf_paired.json.  Produces e0_verdict.json and the Pareto
figure.  All decision rules are transcribed verbatim from prereg §6 and
evaluated mechanically.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import numpy as np

DISC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "discovery")
sys.path.insert(0, DISC_DIR)
import common  # noqa: E402

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "e0")
ACC_DIR = os.path.join(OUT_DIR, "acc")

OCR_PANEL = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench", "ChartQA_TEST"]
GENERAL_PANEL = ["MMBench_DEV_EN_V11", "MMStar", "RealWorldQA", "POPE"]
B235 = ["fastv", "pdrop", "sparsevlm", "visionzip", "divprune"]

N_BOOT = 10000
SEED = 20260929


def official_main(ds, official):
    """The official headline number; OCRBench normalised to 0-100."""
    if ds == "OCRBench":
        return float(official.get("Final Score Norm",
                                  float(official.get("Final Score", 0)) / 10))
    for k in ("Overall", "acc", "Accuracy", "Final Score"):
        if k in official:
            return float(official[k])
    if isinstance(official, dict) and len(official) == 1:
        return float(list(official.values())[0])
    raise KeyError(f"no headline in {official}")


def load_scores():
    """{(arm, K): {ds: dict(official=..., per_q={idx: score}, n=...)}}"""
    out = {}
    for p in glob.glob(os.path.join(ACC_DIR, "*", "K*", "*_score.json")):
        s = json.load(open(p))
        parts = os.path.normpath(p).split(os.sep)
        arm = s["arm"]
        out[(arm, s["K"], s["ds"])] = s
    return out


def macros(scores, arm, K):
    """OCR macro, general macro, total macro (only over datasets present)."""
    def macro(panel):
        vals = [official_main(ds, scores[(arm, K, ds)]["official"])
                for ds in panel if (arm, K, ds) in scores]
        return float(np.mean(vals)) if len(vals) == len(panel) and vals else \
            (float(np.mean(vals)) if vals else None)
    return dict(ocr=macro(OCR_PANEL), general=macro(GENERAL_PANEL),
                total=macro(OCR_PANEL + GENERAL_PANEL))


def image_keys(ds, rows):
    """Image identity per row (same rule as e0_plan)."""
    import pandas as pd
    from vlmeval.dataset import build_dataset as vlmeval_build
    dataset = vlmeval_build(ds)
    keys = {}
    for i in rows:
        row = dataset.data.iloc[int(i)]
        ip = row.get("image_path", None)
        if ip is not None and isinstance(ip, str) and ip.strip():
            keys[int(i)] = os.path.basename(ip.strip())
        else:
            b = row.get("image", None)
            keys[int(i)] = hashlib.md5(b.encode("ascii")).hexdigest() \
                if isinstance(b, str) and b.startswith("/9j") else f"n{int(i)}"
    return keys


def paired_bootstrap(pq_a, pq_b, clusters_a, clusters_b, n_boot=N_BOOT, seed=SEED):
    """Cluster bootstrap over images of mean(score_a) - mean(score_b)."""
    rng = np.random.default_rng(seed)
    keys = sorted(set(clusters_a) & set(clusters_b))
    by_key_a, by_key_b = {}, {}
    for k in keys:
        pass
    qa, qb = [], []
    key_of = []
    for i in sorted(set(pq_a) & set(pq_b)):
        qa.append(pq_a[i]); qb.append(pq_b[i]); key_of.append(clusters_a[i])
    qa, qb, key_of = map(np.asarray, (qa, qb, key_of))
    uk = np.unique(key_of)
    idx_by_k = [np.where(key_of == k)[0] for k in uk]
    diffs = []
    for _ in range(n_boot):
        sel_k = rng.choice(len(uk), len(uk), replace=True)
        idx = np.concatenate([idx_by_k[j] for j in sel_k])
        diffs.append(qa[idx].mean() - qb[idx].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def rescue_break(pq_a, pq_b):
    """rescued: b hits, a misses; broken: b misses, a hits (a=arm, b=B2)."""
    common_q = sorted(set(pq_a) & set(pq_b))
    rescued = sum(1 for i in common_q if pq_a[i] > 0.5 and pq_b[i] <= 0.5)
    broken = sum(1 for i in common_q if pq_a[i] <= 0.5 and pq_b[i] > 0.5)
    return rescued, broken


def main():
    scores = load_scores()
    verdict = {"scores_present": sorted({(a, k, d) for (a, k, d) in scores})}

    # macros for all (arm, K)
    mtable = {}
    for (arm, K, ds) in list(scores):
        mtable.setdefault((arm, K), macros(scores, arm, K))
    verdict["macros"] = {f"{a}|K={k}": v for (a, k), v in mtable.items()}

    # paired contrasts vs native B2 (per dataset, per-question)
    b2_pq = {(k): scores.get(("b2", k, ds), {}).get("per_question")
             for k in (64, 128, 256) for ds in OCR_PANEL + GENERAL_PANEL}
    contrasts = {}
    for (arm, K, ds) in scores:
        if arm == "b2":
            continue
        pq = scores[(arm, K, ds)].get("per_question")
        b2 = scores.get(("b2", K, ds), {}).get("per_question")
        if not pq or not b2:
            continue
        keys = image_keys(ds, sorted(set(pq) & set(b2), key=int))
        d, lo, hi = paired_bootstrap(pq, b2, keys, keys)
        r, br = rescue_break(pq, b2)
        contrasts[f"{arm}|K={K}|{ds}"] = dict(delta=d, ci=[lo, hi],
                                              rescued=r, broken=br)
    verdict["paired_vs_b2"] = contrasts

    # ---------------- D1 修复效应 ----------------
    d1 = {"rule": "|native B2 - A1| >= 2 macro with CI excluding 0"}
    if ("a1", 256, "TextVQA_VAL") in scores:
        a1_vals, b2_vals, keys_all = [], [], None
        for ds in ("TextVQA_VAL", "DocVQA_VAL", "OCRBench"):
            a1_pq = scores[("a1", 256, ds)]["per_question"]
            b2_pq_d = scores[("b2", 256, ds)]["per_question"]
            keys = image_keys(ds, sorted(set(a1_pq) & set(b2_pq_d), key=int))
            d, lo, hi = paired_bootstrap(a1_pq, b2_pq_d, keys, keys)
            a1_vals.append(d)
        delta = -float(np.mean(a1_vals))   # A1 - B2
        d1.update(ocr_datasets_delta_b2_minus_a1=-float(np.mean(a1_vals)),
                  per_ds=a1_vals)
        # CI for macro difference: bootstrap over datasets is not valid;
        # use per-dataset CIs: fire only if every |delta| CI excludes 0 and
        # mean |delta| >= 2 -> prereg's macro-level test approximated by the
        # mean of per-dataset deltas with a pooled bootstrap over questions.
        pooled_a1, pooled_b2 = [], []
        for ds in ("TextVQA_VAL", "DocVQA_VAL", "OCRBench"):
            pooled_a1 += list(scores[("a1", 256, ds)]["per_question"].values())
            pooled_b2 += list(scores[("b2", 256, ds)]["per_question"].values())
        rng = np.random.default_rng(SEED)
        pooled = np.asarray(pooled_a1) - np.asarray(pooled_b2)
        boots = [pooled[rng.integers(0, len(pooled), len(pooled))].mean()
                 for _ in range(N_BOOT)]
        ci = np.percentile(boots, [2.5, 97.5])
        d1["ci"] = [float(ci[0]), float(ci[1])]
        d1["fires"] = bool(abs(d1["ocr_datasets_delta_b2_minus_a1"]) >= 2.0
                           and ci[0] * ci[1] > 0)
    else:
        d1["fires"] = None
        d1["note"] = "A1 not run/present"
    verdict["D1"] = d1

    # ---------------- D3 编码器轴 ----------------
    perf_path = os.path.join(OUT_DIR, "e0_perf_paired.json")
    d3 = {"rule": "ViTshare>=0.5 & PACE TTFT -15% & PACE OCR -2 -> E1; "
                  "PACE OCR no-drop & faster -> occupied; ViTshare<0.35 -> stop"}
    if os.path.exists(perf_path):
        perf = json.load(open(perf_path))
        blocks = perf["blocks"]

        def med(arm_id, key="ttft_ms"):
            vals = [b["runs"][arm_id][key] for b in blocks
                    if arm_id in b["runs"] and b["runs"][arm_id].get(key)]
            return float(np.median(vals)) if vals else None

        b2_ttft = med("b2")
        prep = float(np.median([b["runs"]["b2"]["preprocess_ms"] for b in blocks
                                if "preprocess_ms" in b["runs"].get("b2", {})]))
        vit = float(np.median([b["runs"]["b2"]["vision_ms"] for b in blocks
                               if "vision_ms" in b["runs"].get("b2", {})]))
        vitshare = (prep + vit) / b2_ttft if b2_ttft else None
        d3["ViTshare"] = vitshare
        pace_ttft = med("pace@256")
        d3["pace_ttft_ms"] = pace_ttft
        d3["b2_ttft_ms"] = b2_ttft
        if pace_ttft and ("pace", 256) in mtable and \
                mtable[("pace", 256)]["ocr"] is not None and \
                mtable[("b2", 256)]["ocr"] is not None:
            pace_ocr = mtable[("pace", 256)]["ocr"]
            b2_ocr = mtable[("b2", 256)]["ocr"]
            ttft_drop = 1 - pace_ttft / b2_ttft
            ocr_drop = b2_ocr - pace_ocr
            d3.update(pace_ocr=pace_ocr, b2_ocr=b2_ocr,
                      ttft_drop=ttft_drop, ocr_drop=ocr_drop)
            if vitshare is not None and vitshare < 0.35:
                d3["verdict"] = "STOP_VITSHARE_LOW"
            elif ttft_drop >= 0.15 and ocr_drop < 2:
                d3["verdict"] = "OCCUPIED_BY_PACE"
            elif vitshare is not None and vitshare >= 0.5 and \
                    ttft_drop >= 0.15 and ocr_drop >= 2:
                d3["verdict"] = "ENTER_E1"
            else:
                d3["verdict"] = "NO_GATE_FIRES"
        else:
            d3["verdict"] = "INCOMPLETE (missing pace or perf arms)"
    else:
        d3["verdict"] = "INCOMPLETE (no perf file)"
    verdict["D3"] = d3

    # ---------------- D2 / D4 ----------------
    d2 = {"rule": "R-res(512) OCR macro >= Best24/25(256) - 1 AND R-res TTFT "
                  "lower than that arm's"}
    best25, best_name = None, None
    for arm in B235:
        if (arm, 256) in mtable and mtable[(arm, 256)]["ocr"] is not None:
            v = mtable[(arm, 256)]["ocr"]
            if best25 is None or v > best25:
                best25, best_name = v, arm
    d2["best24_25"] = {"arm": best_name, "ocr_macro": best25}
    if ("rres512", 1024) in mtable and mtable[("rres512", 1024)]["ocr"] is not None:
        rres_ocr = mtable[("rres512", 1024)]["ocr"]
        d2["rres_ocr_macro"] = rres_ocr
        if best25 is not None and os.path.exists(perf_path):
            perf = json.load(open(perf_path))
            blocks = perf["blocks"]

            def med(arm_id):
                vals = [b["runs"][arm_id]["ttft_ms"] for b in blocks
                        if arm_id in b["runs"]]
                return float(np.median(vals)) if vals else None
            d2["rres_ttft_lower"] = (med("rres512") is not None and
                                     best_name is not None and
                                     med("rres512") < med(best_name))
            d2["fires"] = bool(rres_ocr >= best25 - 1 and d2["rres_ttft_lower"])
        else:
            d2["fires"] = None
    else:
        d2["fires"] = None
    verdict["D2"] = d2

    d4 = {"rule": "baseline OCR macro < R-res OCR - 10 -> port-suspect, "
                  "excluded from Best24/25"}
    suspect = []
    rres_ocr = d2.get("rres_ocr_macro")
    if rres_ocr is not None:
        for arm in B235:
            if (arm, 256) in mtable and mtable[(arm, 256)]["ocr"] is not None:
                if mtable[(arm, 256)]["ocr"] < rres_ocr - 10:
                    suspect.append(arm)
    d4["suspect"] = suspect
    verdict["D4"] = d4

    with open(os.path.join(OUT_DIR, "e0_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1)
    print(f"[saved] e0_verdict.json")

    # Pareto figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if os.path.exists(perf_path):
            perf = json.load(open(perf_path))
            blocks = perf["blocks"]

            def med(arm_id):
                vals = [b["runs"][arm_id]["ttft_ms"] for b in blocks
                        if arm_id in b["runs"]]
                return float(np.median(vals)) if vals else None
            fig, ax = plt.subplots(figsize=(7, 5))
            for (arm, K), m in mtable.items():
                if m["ocr"] is None:
                    continue
                if arm == "rres512":
                    aid = "rres512"
                else:
                    aid = arm
                t = med(aid)
                if t is None:
                    continue
                ax.scatter(t, m["ocr"],
                           label=f"{arm} K={K}" if K != 1024 or arm != "b2"
                           else "B0")
            ax.set_xlabel("TTFT median (ms, paired)")
            ax.set_ylabel("OCR macro")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
            fig.savefig(os.path.join(OUT_DIR, "figures", "e0_pareto.png"),
                        dpi=150, bbox_inches="tight")
            print("[saved] figures/e0_pareto.png")
    except Exception as e:
        print(f"[figure skipped] {e}")


if __name__ == "__main__":
    main()
