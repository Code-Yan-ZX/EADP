#!/usr/bin/env python3
"""
Turn the discovery JSON artifacts into markdown tables for the report.

Reads whatever exists under outputs/discovery/ and prints only the sections it
has data for, so it is safe to run mid-study.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

OUT = common.OUTPUT_DIR


def jload(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """Mean(a) - Mean(b) with a 95% bootstrap CI over the paired differences."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------------------------------------------------------------------------
def _get(store, key):
    """Fetch a per-dataset aggregate, tolerating the '_mean_mean' double suffix."""
    vals = []
    for ds, d in store.items():
        if not isinstance(d, dict):
            continue
        for cand in (key, key + "_mean"):
            if cand in d:
                vals.append(d[cand])
                break
    return float(np.mean(vals)) if vals else float("nan")


def _stage(store, stage):
    vals = [
        store[ds]["stages"][stage]["mean"]
        for ds in store
        if isinstance(store[ds], dict) and stage in store[ds].get("stages", {})
    ]
    return float(np.mean(vals)) if vals else float("nan")


def table14(prof):
    """Reproduce the paper's Table 14 framing."""
    print("\n## Part 1 - efficiency profile (Qwen3-VL-8B, 1024x1024)\n")
    models = prof["models"]
    base = models.get("baseline", {})
    if not base:
        print("_baseline missing_")
        return

    base_prefill = _get(base, "prefill_ms_mean")
    base_total = _get(base, "prefill_total_ms_mean")
    base_vision = _get(base, "vision_ms_mean")
    base_e2e = _get(base, "e2e_ms_mean")
    base_mem = _get(base, "peak_mem_mb_mean")
    base_flops = _get(base, "flops_g_mean")

    stages = ["global_guidance", "dense_guidance", "score_fusion",
              "smoothing", "polarization", "facility_location"]
    header = ["Retained", "Baseline prefill", "Global", "Dense", "Fusion",
              "Smooth", "Polar", "FacilityLoc", "Total prune",
              "Pruned prefill", "Prune+prefill", "vs base"]
    rows = []
    for label in ("eadp128", "eadp256", "eadp512"):
        m = models.get(label)
        if not m:
            continue
        st = {s: _stage(m, s) for s in stages}
        tot = _get(m, "prune_ms_mean")
        pre = _get(m, "prefill_ms_mean")
        rows.append([
            label.replace("eadp", ""),
            f"{base_prefill:.2f}",
            *[f"{st[s]:.2f}" for s in stages],
            f"{tot:.2f}", f"{pre:.2f}", f"{tot + pre:.2f}",
            f"{base_prefill / (tot + pre):.2f}x",
        ])
    print(md_table(header, rows))

    print(f"\nBaseline: LLM prefill {base_prefill:.2f} ms | vision tower {base_vision:.2f} ms | "
          f"full prefill {base_total:.2f} ms | e2e {base_e2e:.2f} ms | "
          f"FLOPs {base_flops:.0f} G | peak mem {base_mem:.0f} MB\n")

    print("### Table 14 trend check vs the paper (their machine / ours)\n")
    paper = {
        128: {"fl": 37.28, "tot": 39.09, "pre": 77.60},
        256: {"fl": 74.26, "tot": 76.05, "pre": 122.63},
        512: {"fl": 143.73, "tot": 145.53, "pre": 178.97},
    }
    rows = []
    for tok, p in paper.items():
        m = models.get(f"eadp{tok}")
        if not m:
            continue
        fl, tot, pre = _stage(m, "facility_location"), _get(m, "prune_ms_mean"), _get(m, "prefill_ms_mean")
        rows.append([
            tok,
            f"{p['fl']:.2f} / {fl:.2f} ({fl / p['fl']:.2f}x)",
            f"{p['tot']:.2f} / {tot:.2f} ({tot / p['tot']:.2f}x)",
            f"{p['pre']:.2f} / {pre:.2f} ({pre / p['pre']:.2f}x)",
            f"{p['fl'] / p['tot'] * 100:.1f}% / {fl / tot * 100:.1f}%",
        ])
    print(md_table(["Retained", "FacilityLoc ms (paper/ours)", "Total prune (paper/ours)",
                    "Pruned prefill (paper/ours)", "FL share (paper/ours)"], rows))
    print(f"\nBaseline prefill: paper 320.51 ms / ours {base_prefill:.2f} ms "
          f"= {base_prefill / 320.51:.2f}x machine ratio\n")

    print("### FLOPs / memory / token counts\n")
    rows = []
    for label in ("baseline", "eadp128", "eadp256", "eadp512"):
        m = models.get(label)
        if not m:
            continue
        rows.append([
            label, f"{_get(m, 'visual_tokens_kept_mean'):.0f}",
            f"{_get(m, 'seq_len_mean'):.0f}", f"{_get(m, 'flops_g_mean'):.0f}",
            f"{_get(m, 'peak_mem_mb_mean'):.0f}", f"{_get(m, 'e2e_ms_mean'):.1f}",
            f"{_get(m, 'gen_tokens_mean'):.1f}",
        ])
    print(md_table(["Config", "vis tokens kept", "seq len", "FLOPs (G)",
                    "peak mem (MB)", "e2e (ms)", "gen tokens"], rows))


# ---------------------------------------------------------------------------
def selector_tables(diag):
    print("\n## Part 2 - selector comparison (identical importance scoring)\n")
    recs_all = list(diag["runs"].values())
    datasets = sorted({r["dataset"] for r in recs_all})
    per = {}
    for r in recs_all:
        per.setdefault((r["budget"], r.get("sim_mode", "rebound"), r["selector"]), {})[
            r["dataset"]
        ] = r

    print("### accuracy and cost by budget / similarity kernel / selector\n")
    header = ["Budget", "sim", "Selector", *datasets, "mean", "select ms", "prune ms"]
    rows = []
    for (budget, sim, sel) in sorted(per):
        rs = [per[(budget, sim, sel)][d] for d in datasets if d in per[(budget, sim, sel)]]
        if not rs:
            continue
        rows.append([
            budget, sim, sel,
            *[f"{per[(budget, sim, sel)][d]['acc_pct']:.2f}" if d in per[(budget, sim, sel)]
              else "-" for d in datasets],
            f"{np.mean([x['acc_pct'] for x in rs]):.2f}",
            f"{np.mean([x['select_ms_mean'] for x in rs]):.2f}",
            f"{np.mean([x['prune_ms_mean'] for x in rs]):.2f}",
        ])
    print(md_table(header, rows))

    baseline_key = (256, "rebound", "facility")
    if baseline_key not in per:
        return
    ref = per[baseline_key]
    ref_hits = np.concatenate([np.array(ref[d]["hits"]) for d in datasets if d in ref])

    print("\n### paired bootstrap vs official `facility` @256 rebound "
          "(95% CI on mean-accuracy difference)\n")
    rows = []
    for (budget, sim, sel) in sorted(per):
        if (budget, sim, sel) == baseline_key:
            continue
        rs = [per[(budget, sim, sel)][d] for d in datasets if d in per[(budget, sim, sel)]]
        hits = np.concatenate([np.array(x["hits"]) for x in rs])
        if len(hits) != len(ref_hits):
            continue
        d, lo, hi = paired_bootstrap(hits * 100, ref_hits * 100)
        sig = "**yes**" if (lo > 0 or hi < 0) else "no"
        rows.append([budget, sim, sel, f"{d:+.2f}", f"[{lo:+.2f}, {hi:+.2f}]", sig])
    print(md_table(["Budget", "sim", "Selector", "delta acc vs facility", "95% CI",
                    "significant"], rows))

    print("\n### per-dataset paired difference (TopK-family minus facility @256)\n")
    rows = []
    for (budget, sim, sel) in sorted(per):
        if budget != 256 or sim != "rebound" or sel == "facility":
            continue
        cells = []
        for ds in datasets:
            if ds not in per[(budget, sim, sel)] or ds not in ref:
                cells.append("-")
                continue
            a = np.array(per[(budget, sim, sel)][ds]["hits"]) * 100
            b = np.array(ref[ds]["hits"]) * 100
            d, lo, hi = paired_bootstrap(a, b)
            star = "*" if (lo > 0 or hi < 0) else ""
            cells.append(f"{d:+.2f}{star}")
        rows.append([sel, *cells])
    print(md_table(["Selector", *[f"{d} (delta)" for d in datasets]], rows))
    print("\n`*` = 95 % bootstrap CI excludes zero. TextVQA/DocVQA are VQA-score/ANLS "
          "x100; OCRBench is % correct.")


# ---------------------------------------------------------------------------
def overlap_table(ov):
    print("\n## Part 2b - selection-set overlap (no generation)\n")
    for ds, info in ov["datasets"].items():
        print(f"\n### {ds}  (n={info['n']})\n")
        ks = sorted(info["iou_mean"])
        rows = [[k.replace("|", " vs "), f"{info['iou_mean'][k]:.3f}"] for k in ks]
        print(md_table(["pair", "IoU"], rows))
        print("\nF(S) objective (higher = better coverage x importance):\n")
        print(md_table(["selector", "F(S)", "mean importance-rank pct"],
                       [[s, f"{info['objective_mean'][s]:.4f}",
                         f"{info['mean_rank_pct'][s]:.4f}"] for s in info["objective_mean"]]))
        print(f"\nsimilarity stats: {info['sim_stats']}\n")


# ---------------------------------------------------------------------------
def taxonomy_table(cls):
    print("\n## Part 3 - A/B/C/D failure taxonomy\n")
    summary = cls["summary"]
    datasets = sorted({r["dataset"] for r in cls["rows"]})
    rows = []
    for k in sorted(summary):
        rows.append([k, *[summary[k].get(d, 0) for d in datasets],
                     sum(summary[k].values())])
    print(md_table(["class", *datasets, "total"], rows))

    tot = sum(sum(summary[k].values()) for k in summary)
    print("\nshares:\n")
    print(md_table(["class", "share"],
                   [[k, f"{100 * sum(summary[k].values()) / tot:.1f}%"] for k in sorted(summary)]))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+",
                    default=["prof", "diag", "overlap", "taxonomy"])
    args = ap.parse_args()

    if "prof" in args.which:
        p = jload("efficiency_profile.json")
        if p:
            table14(p)
    if "diag" in args.which:
        merged = {"runs": {}}
        for name in ("diag_selectors_b256.json", "diag_selectors_b256_rebound.json",
                     "diag_selectors_b128_rebound.json", "diag_selectors_b256_clamp.json",
                     "diag_selectors_b128-256_rebound.json"):
            d = jload(name)
            if d:
                merged["runs"].update(d["runs"])
        if merged["runs"]:
            selector_tables(merged)
    if "overlap" in args.which:
        o = jload("diag_overlap_b256_rebound.json")
        if o:
            overlap_table(o)
    if "taxonomy" in args.which:
        c = jload("fa_classification.json")
        if c:
            taxonomy_table(c)


if __name__ == "__main__":
    main()
