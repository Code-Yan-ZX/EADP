"""Anchor-Merge Pilot — analysis (protocol §6-§8).

Per-dataset accuracy = 100 x mean(per-question score) over the ACTUAL evaluated
count (OCRBench denominator = evaluated n, never 1000).  Macro = equal-weight
mean over the three datasets.  Paired cluster bootstrap (by image identity,
within dataset) on the macro delta, 95% CI.  Winner rule: max macro among
U025/U050/U100/S025; ties -> U025 first, then the simpler arm.

Usage: python amp_analyze.py --split dev --arms BASE,U025,U050,U100,S025
       python amp_analyze.py --split confirm --arms BASE,U025,NORM,SHUF --winner U025
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os

import numpy as np

import amp_common as AC

ACC_DIR = os.path.join(AC.OUT_DIR, "acc")


def load_scores(split, arms):
    out = {}
    for arm in arms:
        for ds in AC.DS_LIST:
            p = os.path.join(ACC_DIR, split, arm, "K256", f"{ds}_score.json")
            if not os.path.exists(p):
                continue
            s = json.load(open(p))
            if not s.get("per_question"):
                continue
            out[(arm, ds)] = s
    return out


def acc100(s):
    return 100.0 * float(np.mean([float(v) for v in s["per_question"].values()]))


def macro_of(scores, arm):
    vals = [acc100(scores[(arm, ds)]) for ds in AC.DS_LIST
            if (arm, ds) in scores]
    return float(np.mean(vals)) if len(vals) == len(AC.DS_LIST) else None


def image_clusters(manifest, split, ds):
    return {str(it["idx"]): it["image_key"]
            for it in manifest["datasets"][ds][split]}


def paired_macro_bootstrap(scores, manifest, split, arm_a, arm_b,
                           n_boot=AC.N_BOOT, seed=AC.BOOT_SEED):
    """Cluster bootstrap of macro(score_a) - macro(score_b), paired by
    question, resampled by image cluster within each dataset."""
    rng = np.random.default_rng(seed)
    per_ds = {}
    for ds in AC.DS_LIST:
        if (arm_a, ds) not in scores or (arm_b, ds) not in scores:
            return None
        pq_a = scores[(arm_a, ds)]["per_question"]
        pq_b = scores[(arm_b, ds)]["per_question"]
        keys = image_clusters(manifest, split, ds)
        common_q = sorted(set(pq_a) & set(pq_b), key=int)
        qa = np.array([float(pq_a[q]) for q in common_q])
        qb = np.array([float(pq_b[q]) for q in common_q])
        cl = np.array([keys[q] for q in common_q])
        uk = np.unique(cl)
        idx_by_k = [np.where(cl == k)[0] for k in uk]
        per_ds[ds] = (qa, qb, idx_by_k, len(uk))
    deltas = np.zeros(n_boot)
    for b in range(n_boot):
        dsd = []
        for ds in AC.DS_LIST:
            qa, qb, idx_by_k, nk = per_ds[ds]
            sel = rng.integers(0, nk, nk)
            idx = np.concatenate([idx_by_k[j] for j in sel])
            dsd.append(qa[idx].mean() - qb[idx].mean())
        deltas[b] = float(np.mean(dsd))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    point = float(np.mean([per_ds[ds][0].mean() - per_ds[ds][1].mean()
                           for ds in AC.DS_LIST]))
    # report in accuracy points (0-100 scale, same as the macro table)
    return dict(delta=100.0 * point, ci=[100.0 * float(lo), 100.0 * float(hi)],
                per_ds={ds: 100.0 * float(per_ds[ds][0].mean()
                                          - per_ds[ds][1].mean())
                        for ds in AC.DS_LIST})


def rescue_break(scores, arm_a, arm_b):
    """arm_a vs arm_b: rescued = b<=0.5 & a>0.5; broken = b>0.5 & a<=0.5
    (the 0.5 threshold applies to the continuous ANLS score as well)."""
    out = {}
    for ds in AC.DS_LIST:
        if (arm_a, ds) not in scores or (arm_b, ds) not in scores:
            continue
        pa = scores[(arm_a, ds)]["per_question"]
        pb = scores[(arm_b, ds)]["per_question"]
        common = set(pa) & set(pb)
        r = sum(1 for q in common if pa[q] > 0.5 >= pb[q])
        br = sum(1 for q in common if pb[q] > 0.5 >= pa[q])
        out[ds] = dict(rescued=r, broken=br)
    return out


def run_quality(scores, arm, split):
    out = {}
    for ds in AC.DS_LIST:
        if (arm, ds) not in scores:
            continue
        shard = os.path.join(ACC_DIR, split, arm, "K256", f"{ds}.json")
        if not os.path.exists(shard):
            continue
        sh = json.load(open(shard))
        recs = list(sh["records"].values())
        n = len(recs)
        out[ds] = dict(
            n=n,
            empty=sum(1 for r in recs if r.get("degeneracy", {}).get("empty")),
            truncated=sum(1 for r in recs if r.get("truncated")),
            degenerate_repeat=sum(1 for r in recs
                                  if r.get("degeneracy", {}).get("repeat4", 0)
                                  >= 10))
    return out


def bank_stats(split):
    stats = {}
    for ds in AC.DS_LIST:
        p = AC.bank_path(split, ds)
        if not os.path.exists(p):
            continue
        with gzip.open(p, "rt") as f:
            bank = json.load(f)
        gs, rn, ca, rnd, cad = [], [], [], [], []
        for rec in bank.values():
            g = rec.get("gsize") or []
            if not g:
                continue
            gs += g
            rn += rec["rnorm_main"]
            ca += rec["cos_am_main"]
            rnd += rec["rnorm_ds"]
            cad += rec["cos_am_ds"]
        g = np.array(gs)
        stats[ds] = dict(
            n_samples=len(bank),
            group_size_mean=float(g.mean()), group_size_p50=float(np.median(g)),
            group_size_p90=float(np.percentile(g, 90)),
            group_size_max=int(g.max()),
            singleton_frac=float((g == 0).mean()),
            rnorm_main_mean=float(np.mean(rn)), rnorm_main_p90=float(np.percentile(rn, 90)),
            cos_am_main_mean=float(np.mean(ca)),
            rnorm_ds_mean=float(np.mean(rnd)), cos_am_ds_mean=float(np.mean(cad)))
    return stats


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "confirm"])
    ap.add_argument("--arms", required=True)
    ap.add_argument("--winner", default=None,
                    help="confirm split: the frozen winner arm")
    args = ap.parse_args()
    arms = args.arms.split(",")
    manifest = json.load(open(os.path.join(AC.OUT_DIR, "manifest.json")))
    scores = load_scores(args.split, arms)

    table = {}
    for arm in arms:
        per = {ds: (acc100(scores[(arm, ds)]) if (arm, ds) in scores else None)
               for ds in AC.DS_LIST}
        table[arm] = dict(per_ds=per, macro=macro_of(scores, arm),
                          quality=run_quality(scores, arm, args.split))

    result = dict(split=args.split, arms=arms, table=table,
                  bank_stats=bank_stats(args.split),
                  manifest_sha={ds: dict(
                      dev=manifest["datasets"][ds]["dev_sha256"],
                      confirm=manifest["datasets"][ds]["confirm_sha256"])
                      for ds in AC.DS_LIST})

    base = "BASE"
    contrasts = {}
    for arm in arms:
        if arm == base:
            continue
        c = paired_macro_bootstrap(scores, manifest, args.split, arm, base)
        if c:
            contrasts[f"{arm}_vs_{base}"] = c
            rb = rescue_break(scores, arm, base)
            contrasts[f"{arm}_vs_{base}"]["rescued"] = sum(
                v["rescued"] for v in rb.values())
            contrasts[f"{arm}_vs_{base}"]["broken"] = sum(
                v["broken"] for v in rb.values())
            contrasts[f"{arm}_vs_{base}"]["rescue_break_per_ds"] = rb
    result["contrasts"] = contrasts

    # winner rule (dev split, or explicit for confirm)
    if args.split == "dev" and all(table[a]["macro"] is not None
                                   for a in AC.CANDIDATES):
        best = max(AC.CANDIDATES, key=lambda a: table[a]["macro"])
        ties = [a for a in AC.CANDIDATES
                if abs(table[a]["macro"] - table[best]["macro"]) < 1e-9]
        if len(ties) > 1:
            best = "U025" if "U025" in ties else ties[0]
        result["winner"] = best
        result["winner_rule"] = ("max DEV macro among candidates; ties -> "
                                 "U025 then simpler (frozen)")
        result["any_positive"] = bool(
            any(table[a]["macro"] > table["BASE"]["macro"]
                for a in AC.CANDIDATES))
    elif args.winner:
        for ctrl in ("NORM", "SHUF"):
            if ctrl in arms and args.winner in arms:
                c = paired_macro_bootstrap(scores, manifest, args.split,
                                           args.winner, ctrl)
                if c:
                    contrasts[f"{args.winner}_vs_{ctrl}"] = c

    out = os.path.join(AC.OUT_DIR, f"analysis_{args.split}.json")
    with open(out + ".tmp", "w") as f:
        json.dump(result, f, indent=1)
    os.replace(out + ".tmp", out)

    # markdown table
    print(f"\n== {args.split} panel ==")
    hdr = "| arm | " + " | ".join(AC.DS_LIST) + " | macro |"
    print(hdr)
    print("|---" * (len(AC.DS_LIST) + 2) + "|")
    for arm in arms:
        t = table[arm]
        row = " | ".join(f"{t['per_ds'][ds]:.2f}"
                         if t["per_ds"][ds] is not None else "—"
                         for ds in AC.DS_LIST)
        mac = f"{t['macro']:.3f}" if t["macro"] is not None else "—"
        print(f"| {arm} | {row} | {mac} |")
    for k, c in contrasts.items():
        print(f"\n{k}: delta={c['delta']:+.3f} "
              f"CI=[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] "
              f"rescued={c.get('rescued')} broken={c.get('broken')} "
              f"per_ds={ {d: round(v, 2) for d, v in c['per_ds'].items()} }")
    if "winner" in result:
        print(f"\nwinner: {result['winner']}  "
              f"any_positive={result.get('any_positive')}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
