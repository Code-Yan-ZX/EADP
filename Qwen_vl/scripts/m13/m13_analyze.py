"""M13 analysis: consolidate profiling, DS-ablation and depth-oracle shards
into the report tables + paired per-sample recovery analysis.

Outputs outputs/m13/m13_analysis.json and prints markdown tables.

Usage: python m13_analyze.py
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys

import torch  # noqa: F401  (imported for env parity only)

DISC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "discovery")
M13_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DISC_DIR)
sys.path.insert(0, M13_DIR)
import common  # noqa: E402
import m13_common as mc  # noqa: E402

OUT_DIR = mc.OUT_DIR
DS_PANEL = mc.OCR_PANEL


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def headline_from_score(score_json):
    """Headline number for a dataset from a *_score.json (M12 convention).

    OCRBench reports 'Final Score' directly; TextVQA/DocVQA nested result
    dicts are summarized by the mean of per-question eval scores x100 (the
    same protocol as M12's analysis).
    """
    if score_json is None:
        return None
    ds = score_json["ds"]
    official = score_json.get("official", {})
    v = mc.headline(ds, official)
    if isinstance(v, (int, float)):
        return float(v)
    per_q = score_json.get("per_question")
    if per_q:
        return 100.0 * sum(per_q.values()) / len(per_q)
    return None


def med(records, key):
    vals = [r[key] for r in records.values() if r.get(key) is not None]
    return statistics.median(vals) if vals else None


def timing_med(shard):
    """Median per-record vision / prefill / ttft, plus vit sub-stages."""
    recs = shard["records"]
    out = {k: med(recs, k) for k in ("vision_ms", "llm_prefill_ms", "ttft_ms")}
    vit_keys = set()
    for r in recs.values():
        if isinstance(r.get("vit"), dict):
            vit_keys.update(r["vit"].keys())
    for k in vit_keys:
        out[f"vit_{k}"] = statistics.median(
            [r["vit"][k] for r in recs.values()
             if isinstance(r.get("vit"), dict) and k in r["vit"]])
    return out


# ---------------------------------------------------------------------------
def table_ds_profile():
    p = load_json(os.path.join(OUT_DIR, "m13_ds_profile.json"))
    if p is None:
        return None, "profile json missing"
    s = p["summary"]
    lines = ["| stage | median ms | mean ms | std |",
             "|---|---|---|---|"]
    for k in p["stage_keys"] + ["vision_ms", "llm_prefill_ms"]:
        v = s[k]
        lines.append(f"| {k} | {v['median']:.2f} | {v['mean']:.2f} | {v['std']:.2f} |")
    lines.append(f"| **DS mergers total** | **{s['ds_mergers_total_median']:.2f}** | | |")
    return dict(summary=s), "\n".join(lines)


def table_ds_ablation(names=None):
    rows = []
    for name, d8, d16, d24 in mc.DS_SETTINGS:
        if names and name not in names:
            continue
        row = dict(setting=name, DS8=d8, DS16=d16, DS24=d24, scores={},
                   timings={})
        for ds in DS_PANEL:
            sc = load_json(mc.shard_path(name, 1024, ds).replace(".json", "_score.json"))
            row["scores"][ds] = headline(ds, sc["official"]) if sc else None
        shard = mc.load_shard(mc.shard_path(name, 1024, DS_PANEL[0]))
        if shard["records"]:
            row["timings"] = timing_med(shard)
            row["n"] = len(shard["records"])
        rows.append(row)
    lines = ["| setting | DS8 | DS16 | DS24 | TextVQA | DocVQA | OCRBench |"
             " vision ms | TTFT ms |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sc, t = r["scores"], r.get("timings", {})
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else "—"
        lines.append(
            f"| {r['setting']} | {r['DS8']} | {r['DS16']} | {r['DS24']} | "
            f"{fmt(sc.get('TextVQA_VAL'))} | {fmt(sc.get('DocVQA_VAL'))} | "
            f"{fmt(sc.get('OCRBench'))} | "
            f"{t.get('vision_ms', float('nan')):.1f} | "
            f"{t.get('ttft_ms', float('nan')):.1f} |")
    return rows, "\n".join(lines)


def table_depth(K=512, depths=(0, 4, 8, 12, 16, 20), modes=("drop", "merge")):
    # native reference
    ref = {}
    for ds in DS_PANEL:
        sc = load_json(os.path.join(common.QWEN_ROOT, "outputs", "e0", "acc",
                                    "b0", "K1024",
                                    f"{ds}_score.json".replace(f"{ds}_", f"{ds}_")))
    rows = []
    for L in depths:
        for m in modes:
            aid = f"L{L}_{m}"
            row = dict(arm=aid, L=L, mode=m, K=K, scores={}, timings={})
            for ds in DS_PANEL:
                sc = load_json(mc.shard_path(aid, K, ds).replace(".json", "_score.json"))
                row["scores"][ds] = headline(ds, sc["official"]) if sc else None
            shard = mc.load_shard(mc.shard_path(aid, K, DS_PANEL[0]))
            if shard["records"]:
                row["timings"] = timing_med(shard)
                row["n"] = len(shard["records"])
            rows.append(row)
    lines = ["| arm | L | mode | TextVQA | DocVQA | OCRBench | ViT ms |"
             " dense ms | sparse ms | compress ms | TTFT ms |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sc, t = r["scores"], r.get("timings", {})
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else "—"
        fmt1 = lambda v: f"{v:.1f}" if isinstance(v, float) else "—"
        lines.append(
            f"| {r['arm']} | {r['L']} | {r['mode']} | "
            f"{fmt(sc.get('TextVQA_VAL'))} | {fmt(sc.get('DocVQA_VAL'))} | "
            f"{fmt(sc.get('OCRBench'))} | {fmt1(t.get('vision_ms'))} | "
            f"{fmt1(t.get('vit_vit_dense_blocks_0_%s_ms' % (r['L'] - 1)) if r['L'] > 0 else t.get('vit_vit_dense_blocks_0_0_ms'))} | "
            f"{fmt1(t.get('vit_vit_sparse_blocks_%d_26_ms' % r['L']))} | "
            f"{fmt1(t.get('vit_vit_compress_%s_ms' % r['mode']))} | "
            f"{fmt1(t.get('ttft_ms'))} |")
    return rows, "\n".join(lines)


# ---------------------------------------------------------------------------
# paired per-sample analysis
# ---------------------------------------------------------------------------
def per_q_scores(arm, K, ds):
    sc = load_json(mc.shard_path(arm, K, ds).replace(".json", "_score.json"))
    return (sc or {}).get("per_question") or {}


def paired_recovery(arm_a, K_a, arm_b, K_b, ds):
    """Per-sample deltas a - b with counts of fixed / broken / flipped."""
    qa, qb = per_q_scores(arm_a, K_a, ds), per_q_scores(arm_b, K_b, ds)
    shared = sorted(set(qa) & set(qb), key=int)
    deltas = [qa[i] - qb[i] for i in shared]
    fixed = sum(1 for d in deltas if d > 0.01)
    broken = sum(1 for d in deltas if d < -0.01)
    net = sum(deltas)
    return dict(n=len(shared), fixed=fixed, broken=broken, net=net,
                mean_delta=net / max(1, len(shared)))


def main():
    out = {}
    prof, prof_md = table_ds_profile()
    out["ds_profile"] = prof
    abl, abl_md = table_ds_ablation()
    out["ds_ablation"] = abl
    dep, dep_md = table_depth()
    out["depth"] = dep

    print("## DS profile\n", prof_md, "\n", sep="")
    print("## DS branch ablation\n", abl_md, "\n", sep="")
    print("## Compression depth oracle (K=512)\n", dep_md, "\n", sep="")

    # drop-vs-merge per depth, per dataset
    dm = {}
    for L in (0, 4, 8, 12, 16, 20):
        for ds in DS_PANEL:
            d = mc.headline  # noqa
            sc_d = load_json(mc.shard_path(f"L{L}_drop", 512, ds).replace(".json", "_score.json"))
            sc_m = load_json(mc.shard_path(f"L{L}_merge", 512, ds).replace(".json", "_score.json"))
            if sc_d and sc_m:
                vd, vm = headline(ds, sc_d["official"]), headline(ds, sc_m["official"])
                dm[f"L{L}_{ds}"] = dict(drop=vd, merge=vm, delta=vm - vd)
    out["drop_vs_merge"] = dm

    # paired recovery: L8/L12 merge vs L0 drop, and vs b0
    pr = {}
    for ds in DS_PANEL:
        pr[ds] = {}
        for arm, K in (("L0_drop", 512), ("L8_merge", 512), ("L12_merge", 512),
                       ("L8_drop", 512), ("L12_drop", 512)):
            for base, Kb in (("L0_drop", 512), ("b0", 1024)):
                key = f"{arm}_vs_{base}"
                pr[ds][key] = paired_recovery(arm, K, base, Kb, ds)
    out["paired"] = pr

    # reference numbers (b0, M12 rres/variance@512)
    ref = {}
    for ds in DS_PANEL:
        ref[ds] = {}
        for arm, root, K in (("b0", "e0/acc/b0", 1024),
                             ("m12_rres512", "m12/acc/rres", 512),
                             ("m12_variance512", "m12/acc/variance", 512)):
            p = os.path.join(common.QWEN_ROOT, "outputs", root,
                             f"K{K}", f"{ds}_score.json")
            sc = load_json(p)
            if sc:
                ref[ds][arm] = headline(ds, sc["official"])
    out["reference"] = ref
    print("## Reference\n", json.dumps(ref, indent=1), "\n", sep="")

    with open(os.path.join(OUT_DIR, "m13_analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[saved] {os.path.join(OUT_DIR, 'm13_analysis.json')}")


if __name__ == "__main__":
    main()
