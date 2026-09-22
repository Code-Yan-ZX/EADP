"""
S2-B analysis: accuracy translation, paired bootstrap, error transitions.

Baseline is the Stage-1 official ``facility @256`` arm stored in
``diag_selectors_b256.json`` (identical frozen indices and generation code) —
reused, never recomputed. Arms are aligned to it by dataset index, so a pilot
subset automatically compares against the same instances.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import sample_indices
from s1_audit import BLOCK, OUT

DS = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]
CORRECT = 0.5          # hit >= 0.5 counts as correct (Stage-1's "hard error" cut)
GRID = 32
rng = np.random.default_rng(0)
NBOOT = 10000


def block_of(tok):
    r, c = divmod(int(tok), GRID)
    return (r // BLOCK) * (GRID // BLOCK) + (c // BLOCK)


def paired(official, arm):
    """Align on dataset index -> (hits_official, hits_arm, indices)."""
    oi = {int(i): float(h) for i, h in zip(official["idx"], official["hits"])}
    ii = [(int(i), float(h)) for i, h in zip(arm["idx"], arm["hits"]) if int(i) in oi]
    idx = np.array([x[0] for x in ii])
    ah = np.array([x[1] for x in ii])
    oh = np.array([oi[i] for i in idx])
    return oh, ah, idx


def macro_bootstrap(deltas_per_ds):
    """Stratified paired bootstrap over instances, macro = mean of dataset means."""
    ds_use = [d for d in DS if d in deltas_per_ds and len(deltas_per_ds[d])]
    out = np.empty(NBOOT)
    for b in range(NBOOT):
        out[b] = np.mean([rng.choice(deltas_per_ds[d], len(deltas_per_ds[d]),
                                     replace=True).mean() for d in ds_use])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="s2b_pilot.json")
    ap.add_argument("--tag", default="s2b_pilot")
    args = ap.parse_args()

    off_all = json.load(open(os.path.join(OUT, "diag_selectors_b256.json")))["runs"]
    off = {v["dataset"]: v for v in off_all.values() if v["selector"] == "facility"}
    A = json.load(open(os.path.join(OUT, args.arms)))["runs"]
    # run keys are "b<budget>|<selector>|<calibration>|<dataset>"; group by arm
    recs = {}
    for k, v in A.items():
        recs.setdefault(f"{v['selector']}|{v['calibration']}", {})[v["dataset"]] = v

    Z = np.load(os.path.join(OUT, "s2b_gradient_scores.npz"))
    try:
        O = np.load(os.path.join(OUT, "s2b_official_selection.npz"))
    except FileNotFoundError:
        O = None

    # Restrict the official baseline to the instances the arms actually ran
    # (the pilot is a 50-per-dataset subset; comparing it against the full-150
    # accuracy would not be paired).
    # only datasets every arm actually ran (a partially-written results file is
    # fine: arms are appended one at a time)
    common_ds = [d for d in DS if all(d in recs[k] for k in recs)]
    if common_ds != DS:
        print(f"[warn] arms cover only {common_ds}; restricting the report to those")

    off_sub = {}
    for d in common_ds:
        oi = {int(i): float(h) for i, h in zip(off[d]["idx"], off[d]["hits"])}
        keys = set(oi)
        for k in recs:
            keys &= {int(i) for i in recs[k][d]["idx"]}
        keys = sorted(keys)
        off_sub[d] = dict(idx=keys, hits=[oi[i] for i in keys], dataset=d)
        print(f"[paired subset] {d}: n={len(keys)} of {len(off[d]['idx'])} frozen")
    off = off_sub

    # ---------------- accuracy table --------------------------------------
    print("\n" + "=" * 116)
    print(f"ACCURACY (T=256) — {args.arms}")
    print(f"{'arm':30s}" + "".join(f"{d[:9]:>13s}" for d in common_ds) + f"{'macro':>10s}")
    base_acc = [float(np.mean(off[d]["hits"]) * 100) for d in common_ds]
    print(f"{'official EADP facility':30s}" + "".join(f"{a:13.3f}" for a in base_acc)
          + f"{np.mean(base_acc):10.3f}")

    summary = {}
    for k in sorted(recs):
        accs, deltas, pooled = [], {}, []
        for d in common_ds:
            oh, ah, _ = paired(off[d], recs[k][d])
            accs.append(float(ah.mean() * 100))
            deltas[d] = ah - oh
            pooled.append(ah - oh)
        pooled = np.concatenate(pooled)
        bs = macro_bootstrap(deltas)
        mac, mac_d = float(np.mean(accs)), float(np.mean(accs) - np.mean(base_acc))
        summary[k] = dict(
            per_dataset={d: dict(n=len(deltas[d]),
                                 official=float(np.mean(off[d]["hits"]) * 100),
                                 arm=float(accs[i]),      # accs are already in points
                                 delta=float(deltas[d].mean() * 100))
                         for i, d in enumerate(common_ds)},
            macro=mac, macro_delta=mac_d,
            ci=[float(np.percentile(bs, 2.5) * 100), float(np.percentile(bs, 97.5) * 100)],
            pooled_delta=float(pooled.mean() * 100),
            win=int(np.sum(pooled > 1e-9)), tie=int(np.sum(np.abs(pooled) <= 1e-9)),
            loss=int(np.sum(pooled < -1e-9)))
        s = summary[k]
        print(f"{k:30s}" + "".join(f"{a:13.3f}" for a in accs) + f"{mac:10.3f}")
        print(f"{'  Δ vs official':30s}"
              + "".join(f"{a - b:+13.3f}" for a, b in zip(accs, base_acc))
              + f"{mac_d:+10.3f}")

    print("\n" + "=" * 116)
    print("PAIRED BOOTSTRAP (10 000 resamples, stratified over datasets; macro = mean of 3)")
    print(f"{'arm':30s}{'macro Δ':>10s}{'95% CI':>22s}{'win':>7s}{'tie':>7s}{'loss':>7s}"
          f"{'pooled Δ':>11s}")
    for k in sorted(summary):
        s = summary[k]
        ci = f"[{s['ci'][0]:+.3f},{s['ci'][1]:+.3f}]"
        print(f"{k:30s}{s['macro_delta']:+10.3f}{ci:>22s}{s['win']:7d}{s['tie']:7d}"
              f"{s['loss']:7d}{s['pooled_delta']:+11.3f}")

    # ---------------- selection diagnostics --------------------------------
    print("\n" + "=" * 116)
    print("SELECTION DIAGNOSTICS vs the official EADP selection on the same instance")
    print(f"{'arm':30s}{'overlap/256':>13s}{'blocks used':>13s}{'top256 blocks':>15s}")
    diag = {}
    for k in sorted(recs):
        ov, bl, tb = [], [], []
        for d in common_ds:
            v = recs[k][d]
            for i, si in zip(v["idx"], v["select_idx"]):
                key = f"{d}_{int(i)}"
                if si is None or O is None or f"sel__{key}" not in O.files:
                    continue
                si = np.array(si)
                off_si = O[f"sel__{key}"]
                ov.append(len(set(si.tolist()) & set(off_si.tolist())) / len(off_si))
                bl.append(len({block_of(t) for t in si}))
                tb.append(len({block_of(t) for t in np.argsort(-Z[key])[:256]}))
        if ov:
            diag[k] = dict(overlap=float(np.mean(ov)), blocks=float(np.mean(bl)),
                           top256_blocks=float(np.mean(tb)), n=len(ov))
            print(f"{k:30s}{np.mean(ov):13.3f}{np.mean(bl):13.2f}{np.mean(tb):15.2f}")

    # ---------------- error transitions ------------------------------------
    print("\n" + "=" * 116)
    print(f"ERROR TRANSITIONS (correct = hit >= {CORRECT})")
    trans = {}
    for k in sorted(recs):
        agg = dict(fixed=0, broken=0, both_correct=0, both_wrong=0)
        per_ds = {}
        for d in common_ds:
            oh, ah, _ = paired(off[d], recs[k][d])
            oc, ac = oh >= CORRECT, ah >= CORRECT
            c = dict(fixed=int(np.sum(~oc & ac)), broken=int(np.sum(oc & ~ac)),
                     both_correct=int(np.sum(oc & ac)), both_wrong=int(np.sum(~oc & ~ac)),
                     n=len(oh))
            per_ds[d] = c
            for kk in ("fixed", "broken", "both_correct", "both_wrong"):
                agg[kk] += c[kk]
        trans[k] = dict(total=agg, per_dataset=per_ds)
        print(f"  {k:30s} fixed={agg['fixed']:3d}  broken={agg['broken']:3d}  "
              f"both_correct={agg['both_correct']:3d}  both_wrong={agg['both_wrong']:3d}")
        for d in common_ds:
            c = per_ds[d]
            print(f"      {d:12s} n={c['n']:3d}  fixed={c['fixed']:3d} broken={c['broken']:3d} "
                  f"both_correct={c['both_correct']:3d} both_wrong={c['both_wrong']:3d}")

    json.dump({"summary": summary, "diagnostics": diag, "transitions": trans},
              open(os.path.join(OUT, f"{args.tag}_analysis.json"), "w"), indent=1)
    print(f"\n[saved] {os.path.join(OUT, f'{args.tag}_analysis.json')}")


if __name__ == "__main__":
    main()
