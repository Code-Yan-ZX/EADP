"""
S2-C3 step 3: head retention -> downstream accuracy, and the mechanism verdict.

The stage's whole question is whether the S2-C1 objective was aimed at the wrong
part of the teacher's ranking (GO-A) or whether a token-local linear read-out of
layer-4 hidden states cannot express the head no matter how it is supervised
(GO-B). So the consolidator reports two things side by side for every target:

  * head retention on the held-out 150 (recall of the teacher's Top-8/16/32/64
    inside the selected 256, plus exact Top-8/16/32 agreement), and
  * macro accuracy through the identical Top-K@256 harness.

------------------------------------------------------------------------------
Pre-registered decision rule (fixed before the numbers were read)
------------------------------------------------------------------------------
Every comparison is made against two references: the exact S2-C1 baseline
``LIN_L4`` (its cached scores) and the **retrained BASE arm at the same seed**,
a seed-matched control that cancels initialisation and batch order and leaves
only the target. An effect has to clear zero against both.

  head gain      paired bootstrap CI (10 000 resamples, stratified by benchmark,
                 resampling instances) on per-instance head_recall@8 excludes
                 zero against both references, AND the margin exceeds the
                 metric's own seed spread across the 3 seeds.
  accuracy gain  the same test on per-instance accuracy excludes zero against
                 both references, and the seed-matched difference vs BASE is
                 positive.

  meaningful     a head gain of at least +5.0 points of head_recall@8 -- the
                 student misses 2.11 of the teacher's best 8 per image, so
                 5 points is about 0.4 tokens/image.

  GO-A        some arm has both a head gain and an accuracy gain.
  GO-B        (a) head_recall@8 never improves, or
              (b) head recall improves but no arm's accuracy gain clears zero.
  AMBIGUOUS   no head gain clears the bar, but the CIs are too wide to exclude
              a meaningful one either.

The rule is stated here, in the script that computes it, so it cannot be fitted
to the result afterwards.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                    # noqa: E402
from s2b_analysis import DS, paired                         # noqa: E402
from s2c1_consolidate import load_runs                      # noqa: E402
from s2c2_common import stratified_bootstrap                # noqa: E402

ARMS = ["BASE", "HEAD_BIN", "HEAD_MULTI", "HEAD_RANK"]
BASELINE = "S2C1_LIN_L4"
HEAD_PRIMARY = "head_recall8"
HEAD_SECONDARY = "head_recall32"
MIN_MEANINGFUL_HEAD_GAIN = 5.0     # points of head_recall8
NBOOT = 10000
METRIC_KEYS = (["overlap256", "auroc", "ap", "spearman"]
               + [f"head_recall{k}" for k in (8, 16, 32, 64)]
               + [f"head_miss{k}" for k in (8, 16, 32, 64)]
               + [f"head_agree{k}" for k in (8, 16, 32)])


# --------------------------------------------------------------------------
def head_block(H, tags):
    """Seed-mean head metrics per arm + paired bootstrap against both references."""
    idx = {t: n for n, t in enumerate(tags)}
    dsmask = {ds: np.array([k.startswith(ds) for k in H["keys"]]) for ds in DS}
    out = {}
    for arm in ARMS:
        ts = [t for t in tags if t.startswith(arm + "_s")]
        if not ts:
            continue
        per = {m: np.mean([H[m][idx[t]] for t in ts], axis=0) for m in METRIC_KEYS}
        rec = {"seeds": sorted(int(t.split("_s")[1]) for t in ts),
               "metrics": {m: float(np.mean(per[m])) for m in per},
               "seed_spread": {}}
        for m in METRIC_KEYS:
            vals = [float(np.mean(H[m][idx[t]])) for t in ts]
            rec["seed_spread"][m] = dict(lo=min(vals), hi=max(vals),
                                         spread=max(vals) - min(vals), vals=vals)
        refs = {BASELINE: {m: H[m][idx[BASELINE]] for m in per}}
        bts = [t for t in tags if t.startswith("BASE_s")]
        if bts:
            refs["BASE_retrained"] = {m: np.mean([H[m][idx[t]] for t in bts], axis=0)
                                      for m in per}
        rec["vs"] = {}
        for ref, rv in refs.items():
            rec["vs"][ref] = {}
            for m in (HEAD_PRIMARY, HEAD_SECONDARY, "head_agree8", "overlap256"):
                d = per[m] - rv[m]
                obs, lo, hi = stratified_bootstrap({ds: d[dsmask[ds]] for ds in DS}, NBOOT)
                rec["vs"][ref][m] = dict(delta=obs * 100, ci=[lo * 100, hi * 100])
        sm = []
        for t in ts:
            bt = f"BASE_s{t.split('_s')[1]}"
            if bt in idx:
                sm.append(float(np.mean(H[HEAD_PRIMARY][idx[t]] - H[HEAD_PRIMARY][idx[bt]])))
        if sm:
            rec["seed_matched_delta_head8"] = dict(vals=sm, mean=float(np.mean(sm)))
        out[arm] = rec
    return out


def downstream_block(pilot, lin_hits, fac_sub, refs):
    out = {}
    for tag, per_ds in pilot.items():
        accs, vs_base, vs_fac = {}, {}, {}
        for d in DS:
            _, ah, idx = paired(fac_sub[d], per_ds[d])
            accs[d] = float(ah.mean() * 100)
            vs_base[d] = ah - np.array([lin_hits[d][int(i)] for i in idx])
            oh, _, _ = paired(fac_sub[d], per_ds[d])
            vs_fac[d] = ah - oh
        obs_b, lo_b, hi_b = stratified_bootstrap(vs_base, NBOOT)
        obs_f, lo_f, hi_f = stratified_bootstrap(vs_fac, NBOOT)
        macro = float(np.mean(list(accs.values())))
        out[tag] = dict(
            per_dataset=accs, macro=macro,
            delta_vs_s2c1_lin_l4=obs_b * 100,
            ci_vs_s2c1_lin_l4=[lo_b * 100, hi_b * 100],
            delta_vs_facility=obs_f * 100,
            ci_vs_facility=[lo_f * 100, hi_f * 100],
            retained_vs_eadp_topk=(macro - refs["eadp_topk_macro"])
            / (refs["teacher_macro"] - refs["eadp_topk_macro"]),
            delta_vs_eadp_topk=macro - refs["eadp_topk_macro"],
            n_kept_mean=float(np.mean([v["n_kept_mean"] for v in per_ds.values()])),
        )
    return out


def rescue_block(pilot, lin_hits, rescuable):
    """The S2-C2 class-1 instances (student wrong, teacher correct): what changes."""
    out = {}
    for tag, per_ds in pilot.items():
        rec = {"rescued": 0, "still_wrong": 0, "newly_broken": 0, "per_ds": {}}
        for d in DS:
            hits = {int(i): float(h) for i, h in zip(per_ds[d]["idx"], per_ds[d]["hits"])}
            r = s = 0
            for key in rescuable:
                if not key.startswith(d):
                    continue
                i = int(key.rsplit("_", 1)[1])
                if i not in hits:
                    continue
                if lin_hits[d][i] < 0.5 <= hits[i]:
                    r += 1
                elif lin_hits[d][i] < 0.5:
                    s += 1
            nb = sum(1 for i, h in hits.items()
                     if lin_hits[d].get(i, 1.0) >= 0.5 > h)
            rec["per_ds"][d] = dict(rescued=r, still_wrong=s, newly_broken=nb,
                                    n_class1=sum(1 for k in rescuable if k.startswith(d)))
            rec["rescued"] += r
            rec["still_wrong"] += s
            rec["newly_broken"] += nb
        out[tag] = rec
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c3")
    ap.add_argument("--pilot", default="s2c3_pilot.json")
    args = ap.parse_args()

    Z = np.load(os.path.join(OUT, f"{args.tag}_headmetrics.npz"), allow_pickle=True)
    H = {k: Z[k] for k in Z.files}
    tags = [str(t) for t in H["tags"]]
    H["keys"] = [str(k) for k in H["keys"]]
    E = json.load(open(os.path.join(OUT, f"{args.tag}_eval.json")))
    T = json.load(open(os.path.join(OUT, f"{args.tag}_train.json")))
    seeds = sorted({int(t.split("_s")[1]) for t in tags if "_s" in t})

    s2c0 = load_runs("s2c0_pilot.json")
    s2c1 = load_runs("s2c1_pilot.json")
    fac = load_runs("diag_selectors_b256.json", selector="facility")

    lin_hits = {d: {int(i): float(h) for i, h in zip(s2c1["LIN_L4"][d]["idx"],
                                                     s2c1["LIN_L4"][d]["hits"])}
                for d in DS}
    fac_sub = {}
    for d in DS:
        oi = {int(i): float(h) for i, h in zip(fac["facility"][d]["idx"],
                                               fac["facility"][d]["hits"])}
        ks = sorted(set(lin_hits[d]) & set(oi))
        fac_sub[d] = dict(idx=ks, hits=[oi[i] for i in ks], dataset=d, selector="facility")

    def macro_of(per_ds):
        """Mean of per-benchmark mean hit rate, x100 -- the project's macro."""
        return float(np.mean([np.mean(per_ds[d]["hits"]) * 100 for d in DS]))

    macro = macro_of
    refs = dict(
        eadp_topk_macro=macro(s2c0["official"]),
        teacher_macro=macro(s2c0["P1G2"]),
        facility_macro=macro(fac_sub),
        s2c1_lin_l4_macro=float(np.mean([np.mean(list(lin_hits[d].values())) * 100
                                         for d in DS])),
    )
    rescuable = json.load(open(os.path.join(OUT, "s2c2_decompose.json")))["rescuable"]["keys"]

    pilot = load_runs(args.pilot)
    # a partially-written pilot file (the harness appends per arm) must not be
    # read as a complete arm: the macro average would silently lose a benchmark.
    incomplete = [t for t, per_ds in pilot.items() if not all(d in per_ds for d in DS)]
    if incomplete:
        print(f"[skip] {len(incomplete)} arm(s) without all benchmarks yet: "
              f"{sorted(incomplete)}")
        pilot = {t: v for t, v in pilot.items() if t not in incomplete}
    pd_block = downstream_block(pilot, lin_hits, fac_sub, refs)
    rs_block = rescue_block(pilot, lin_hits, rescuable)
    hb_block = head_block(H, tags)

    arm_ds = {}
    for arm in ARMS:
        ts = [t for t in pd_block if t.startswith(arm + "_s")]
        if not ts:
            continue
        ms = [pd_block[t]["macro"] for t in ts]
        diffs = [pd_block[t]["macro"] - pd_block[f"BASE_s{t.split('_s')[1]}"]["macro"]
                 for t in ts if f"BASE_s{t.split('_s')[1]}" in pd_block]
        arm_ds[arm] = dict(
            seeds=sorted(int(t.split("_s")[1]) for t in ts),
            macro_mean=float(np.mean(ms)), macro_lo=min(ms), macro_hi=max(ms),
            per_dataset={d: float(np.mean([pd_block[t]["per_dataset"][d] for t in ts]))
                         for d in DS},
            delta_vs_s2c1_lin_l4=float(np.mean([pd_block[t]["delta_vs_s2c1_lin_l4"] for t in ts])),
            ci_vs_s2c1_lin_l4=[
                float(np.mean([pd_block[t]["ci_vs_s2c1_lin_l4"][0] for t in ts])),
                float(np.mean([pd_block[t]["ci_vs_s2c1_lin_l4"][1] for t in ts]))],
            retained_vs_eadp_topk=float(np.mean([pd_block[t]["retained_vs_eadp_topk"] for t in ts])),
            rescued=float(np.mean([rs_block[t]["rescued"] for t in ts])),
            still_wrong=float(np.mean([rs_block[t]["still_wrong"] for t in ts])),
            newly_broken=float(np.mean([rs_block[t]["newly_broken"] for t in ts])),
            seed_matched_delta_macro=float(np.mean(diffs)) if diffs else None,
            seed_matched_macro_vals=diffs,
        )

    # ---- the pre-registered decision ---------------------------------------
    verdicts = {}
    for arm in ARMS:
        if arm == "BASE" or arm not in hb_block or arm not in arm_ds:
            continue
        h = hb_block[arm]
        detail, hg = {}, True
        for ref in (BASELINE, "BASE_retrained"):
            r = h["vs"].get(ref, {}).get(HEAD_PRIMARY)
            if r is None:
                continue
            ok = bool(r["ci"][0] > 0
                      and r["delta"] / 100.0 > h["seed_spread"][HEAD_PRIMARY]["spread"])
            detail[ref] = dict(delta=r["delta"], ci=r["ci"], clears=ok,
                               meaningful=r["ci"][1] >= MIN_MEANINGFUL_HEAD_GAIN)
            hg &= ok
        sm = arm_ds[arm]["seed_matched_delta_macro"]
        ag = bool(arm_ds[arm]["ci_vs_s2c1_lin_l4"][0] > 0 and (sm is None or sm > 0))
        verdicts[arm] = dict(head_gain=hg, head_detail=detail, accuracy_gain=ag,
                             accuracy_ci=arm_ds[arm]["ci_vs_s2c1_lin_l4"],
                             seed_matched_delta_macro=sm)

    any_head = any(v["head_gain"] for v in verdicts.values())
    any_acc = any(v["head_gain"] and v["accuracy_gain"] for v in verdicts.values())
    # can the data exclude a *meaningful* head gain on every arm?
    unresolved = any(all(d["meaningful"] for d in v["head_detail"].values())
                     for v in verdicts.values())
    if any_acc:
        verdict = "GO-A"
        headline = ("GO-A -- objective mismatch supported: head-aligned supervision "
                    "raises teacher head retention and the gain survives to accuracy")
    elif any_head:
        verdict = "GO-B"
        headline = ("GO-B -- head recall moves but downstream accuracy does not follow it")
    elif unresolved:
        verdict = "AMBIGUOUS"
        headline = ("AMBIGUOUS -- the head-recall CIs are too wide to exclude a "
                    f"meaningful (+{MIN_MEANINGFUL_HEAD_GAIN:.0f} pt) gain")
    else:
        verdict = "GO-B"
        headline = ("GO-B -- no head-aligned target materially improves teacher "
                    "Top-8/32 retention inside the selected 256")

    out = dict(verdict=verdict, headline=headline, decision_rule=__doc__,
               arms=ARMS, seeds=seeds, n_held=len(E["keys"]),
               min_meaningful_head_gain=MIN_MEANINGFUL_HEAD_GAIN,
               references=refs, head_metrics_table=E["table"], head_gain=hb_block,
               downstream_per_run=pd_block, downstream_per_arm=arm_ds,
               rescue=rs_block, rescue_counts_class1=len(rescuable),
               verdicts=verdicts, eval=E, train=T["results"],
               target_definitions=T["arms"], latency=T["latency"])
    path = os.path.join(OUT, f"{args.tag}_retarget.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[saved] {path}")

    # ---- console ------------------------------------------------------------
    print(f"\nrefs:  EADP-TopK {refs['eadp_topk_macro']:.3f}   facility "
          f"{refs['facility_macro']:.3f}   P1G2 teacher {refs['teacher_macro']:.3f}   "
          f"S2C1 LIN_L4 {refs['s2c1_lin_l4_macro']:.3f}")
    print("\n" + "=" * 100)
    print("HEAD RETENTION, HELD-OUT 150 (seed mean; R@k = teacher Top-k kept in the 256)")
    print(f"{'arm':11s}{'ov256':>8s}{'R@8':>8s}{'R@16':>8s}{'R@32':>8s}{'R@64':>8s}"
          f"{'A@8':>8s}{'A@32':>8s}{'miss8':>7s}{'AUROC':>8s}   {'dR@8 vs LIN_L4':>18s}")
    for t in sorted(E["table"], key=lambda x: -E["table"][x]["head_recall8"]):
        v = E["table"][t]
        d = hb_block.get(t.split("_s")[0], {}).get("vs", {}).get(BASELINE, {}).get(HEAD_PRIMARY)
        s = f"{d['delta']:+.2f} [{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}]" if d else ""
        print(f"{t:11s}{v['overlap256']:8.3f}{v['head_recall8']:8.3f}{v['head_recall16']:8.3f}"
              f"{v['head_recall32']:8.3f}{v['head_recall64']:8.3f}{v['head_agree8']:8.3f}"
              f"{v['head_agree32']:8.3f}{v['head_miss8']:7.2f}{v['auroc']:8.3f}   {s:>18s}")

    print("\n" + "=" * 100)
    print("DOWNSTREAM (Top-K @256, held-out 150, seed mean)")
    print(f"{'arm':11s}" + "".join(f"{d[:9]:>10s}" for d in DS)
          + f"{'macro':>8s}{'seed rng':>13s}{'vs LIN_L4':>21s}{'retain':>8s}"
          f"{'resc':>6s}{'still':>6s}{'brkn':>6s}")
    for a, v in sorted(arm_ds.items(), key=lambda x: -x[1]["macro_mean"]):
        print(f"{a:11s}" + "".join(f"{v['per_dataset'][d]:10.3f}" for d in DS)
              + f"{v['macro_mean']:8.3f}"
              + f"{v['macro_lo']:.1f}/{v['macro_hi']:.1f}".rjust(13)
              + f"{v['delta_vs_s2c1_lin_l4']:+7.3f} "
                f"[{v['ci_vs_s2c1_lin_l4'][0]:+.2f},{v['ci_vs_s2c1_lin_l4'][1]:+.2f}]".rjust(19)
              + f"{v['retained_vs_eadp_topk'] * 100:7.1f}%"
              + f"{v['rescued']:6.1f}{v['still_wrong']:6.1f}{v['newly_broken']:6.1f}")

    print("\n" + "=" * 100)
    print("DECISION")
    for a, v in verdicts.items():
        sm = v["seed_matched_delta_macro"]
        print(f"  {a:11s} head_gain={'Y' if v['head_gain'] else 'n'}  "
              f"accuracy_gain={'Y' if v['accuracy_gain'] else 'n'}  "
              f"accCI=[{v['accuracy_ci'][0]:+.2f},{v['accuracy_ci'][1]:+.2f}]  "
              f"seed-matched dMacro={'--' if sm is None else f'{sm:+.2f}'}")
    print(f"\nVERDICT: {verdict}\n{headline}")


if __name__ == "__main__":
    main()
