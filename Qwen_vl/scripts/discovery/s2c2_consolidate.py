"""
S2-C2 step 5: build every table the report needs, from the JSONs written by
steps 1-4. No model, no GPU.

Recovery is reported under two denominators, because they answer different
questions and S2-C1 used the second:

  raw gap      (acc(k) - acc_student) / (acc_teacher - acc_student)
               "how much of the *swap's* span has been covered"
  EADP prize   (acc(k) - acc_eadp_topk) / (acc_teacher - acc_eadp_topk)
               the S2-C1 retention denominator, comparable to its 45.0 %
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT  # noqa: E402
from s2c2_common import DS_ALL, stratified_bootstrap  # noqa: E402


def _load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def interp_k(ks, vals, target, base, teacher):
    """First k at which the interpolated curve reaches `target` recovery."""
    frac = [(v - base) / (teacher - base) for v in vals]
    for i in range(1, len(ks)):
        if frac[i - 1] < target <= frac[i]:
            f0, f1 = frac[i - 1], frac[i]
            w = (target - f0) / (f1 - f0 + 1e-12)
            return float(ks[i - 1] + w * (ks[i] - ks[i - 1]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap-json", default="s2c2_swap.json")
    ap.add_argument("--identity-json", default="s2c2_identity.json")
    ap.add_argument("--rescue-json", default="s2c2_rescue.json")
    ap.add_argument("--out", default="s2c2_diagnosis.json")
    args = ap.parse_args()

    from s2c2_common import load_case
    _cases = load_case()
    dec = _load("s2c2_decompose.json")
    swap = _load(args.swap_json) or {"runs": {}}
    ident = _load(args.identity_json) or {"runs": {}}
    resc = _load(args.rescue_json) or {"runs": {}}
    plan = _load("s2c2_rescue_plan.json")
    char = _load("s2c2_characterize.json")

    runs = dict(swap["runs"])
    runs.update(ident["runs"])

    baselines = {}
    for ds in DS_ALL:
        s = runs[f"identity_S|{ds}"]["acc_pct"]
        t = runs[f"identity_T|{ds}"]["acc_pct"]
        e = dec["baselines_topk"][ds]["eadp_topk"]
        baselines[ds] = dict(student=s, teacher=t, eadp_topk=e,
                             raw_gap=t - s, prize=t - e)
    macro = {k: float(np.mean([baselines[d][k] for d in DS_ALL]))
             for k in ("student", "teacher", "eadp_topk", "raw_gap", "prize")}

    # ---- per-instance hits for every arm -----------------------------------
    hits = {}
    for rk, r in runs.items():
        arm, ds = rk.rsplit("|", 1)
        for n, i in enumerate(r["idx"]):
            hits.setdefault(arm, {}).setdefault(ds, {})[i] = float(r["hits"][n])

    # a common per-dataset instance order for paired deltas.  The grid is filled
    # in over several invocations, so an arm may be partially complete; only
    # arms holding all three benchmarks enter the paired comparison.
    order = {ds: sorted(hits["identity_S"][ds]) for ds in DS_ALL}
    incomplete = []
    for arm in list(hits):
        if any(ds not in hits[arm] for ds in DS_ALL):
            incomplete.append(arm)
            del hits[arm]
            continue
        for ds in DS_ALL:
            assert sorted(hits[arm][ds]) == order[ds], f"{arm}|{ds} instance mismatch"
    if incomplete:
        print(f"[skip] {len(incomplete)} incomplete arms: {sorted(incomplete)}")

    def arm_delta(arm, ref="identity_S"):
        """Paired per-instance deltas vs `ref`, macro-averaged with a CI."""
        by_ds = {}
        for ds in DS_ALL:
            a = np.array([hits[arm][ds][i] for i in order[ds]])
            b = np.array([hits[ref][ds][i] for i in order[ds]])
            by_ds[ds] = (a - b) * 100.0
        obs, lo, hi = stratified_bootstrap(by_ds)
        return dict(obs=obs, lo=lo, hi=hi,
                    per_ds={ds: float(v.mean()) for ds, v in by_ds.items()})

    # ---- per-instance outcome counts (improved / tied / worsened) ----------
    outcomes = {}
    for arm in sorted(hits):
        if "|" in arm:
            continue
        per_ds, tot = {}, [0, 0, 0]
        for ds in DS_ALL:
            a = np.array([hits[arm][ds][i] for i in order[ds]])
            b = np.array([hits["identity_S"][ds][i] for i in order[ds]])
            d = a - b
            c = [int((d > 1e-9).sum()), int((np.abs(d) <= 1e-9).sum()),
                 int((d < -1e-9).sum())]
            per_ds[ds] = c
            tot = [tot[j] + c[j] for j in range(3)]
        outcomes[arm] = dict(improved=tot[0], tied=tot[1], worsened=tot[2],
                             per_ds=per_ds)

    # ---- answer flips ------------------------------------------------------
    # The mean hides the shape of the effect: on DocVQA the per-instance median
    # delta is 0.00 and the mean moves because a minority of answers flip from
    # wrong to right.  Count the flips explicitly rather than only the mean.
    flips = {}
    for arm in sorted(hits):
        if "|" in arm:
            continue
        up = dn = 0
        per_ds = {}
        for ds in DS_ALL:
            a = {i: hits[arm][ds][i] for i in order[ds]}
            b = {i: hits["identity_S"][ds][i] for i in order[ds]}
            u = sum(1 for i in order[ds] if a[i] >= 0.5 > b[i])
            d_ = sum(1 for i in order[ds] if b[i] >= 0.5 > a[i])
            per_ds[ds] = [u, d_]
            up += u; dn += d_
        flips[arm] = dict(wrong_to_right=up, right_to_wrong=dn, per_ds=per_ds)

    # ---- rescue rate on the rescuable class -------------------------------
    _cls1 = {c["key"] for c in _cases
             if c["hit_student"] < 0.5 <= c["hit_teacher"]}
    dec_class = [k for k in dec["rescuable"]["keys"] if k in _cls1]
    rescue_rate = {}
    for ds in DS_ALL + ["ALL"]:
        keys = [k for k in dec_class if ds == "ALL" or k.startswith(ds + "_")]
        if not keys:
            continue
        row = {}
        for arm in hits:
            if "|" in arm or arm.startswith(("addteacher", "remworst", "F_rev",
                                             "identity")):
                continue
            vals = []
            for k in keys:
                d, i = k.rsplit("_", 1)
                if int(i) in hits[arm].get(d, {}):
                    vals.append(float(hits[arm][d][int(i)]))
            if len(vals) == len(keys):
                row[arm] = float(np.mean([v >= 0.5 for v in vals]) * 100)
        rescue_rate[ds] = row

    # ---- the recovery curves ----------------------------------------------
    def family(kind, seed=None):
        out = {}
        for ds in DS_ALL:
            accs = {}
            for arm in hits:
                if not arm.startswith(f"{kind}:"):
                    continue
                parts = arm.split(":")
                if seed is not None:
                    if len(parts) < 3 or parts[2] != f"s{seed}":
                        continue
                elif len(parts) > 2:
                    continue
                accs[int(parts[1])] = float(np.mean([hits[arm][ds][i] for i in order[ds]]) * 100)
            out[ds] = accs
        return out

    ks = sorted(int(a.split(":")[1]) for a in hits if a.startswith("teacher:"))
    curves = {}
    for kind in ("teacher", "adversarial", "shuffled"):
        fam = family(kind)
        curves[kind] = {ds: {k: fam[ds][k] for k in ks if k in fam[ds]}
                        for ds in DS_ALL}
    rand = {ds: {} for ds in DS_ALL}
    for s in (0, 1, 2):
        fam = family("random", seed=s)
        for ds in DS_ALL:
            for k, v in fam[ds].items():
                rand[ds].setdefault(k, []).append(v)
    curves["random"] = {ds: {k: dict(mean=float(np.mean(v)), std=float(np.std(v)),
                                     n=len(v), vals=v)
                             for k, v in sorted(rand[ds].items())} for ds in DS_ALL}

    # ---- recovery fractions and the k needed for 25/50/75 % ---------------
    levels = {}
    for ds in DS_ALL:
        b = baselines[ds]
        row = {"teacher": {}, "random": {}, "adversarial": {}, "shuffled": {}}
        kk = sorted(curves["teacher"][ds])
        vals = [curves["teacher"][ds][k] for k in kk]
        row["curve_k"] = kk
        row["curve_acc"] = vals
        row["recovery_raw"] = [(v - b["student"]) / b["raw_gap"] for v in vals]
        row["recovery_prize"] = [(v - b["eadp_topk"]) / b["prize"] for v in vals]
        # endpoints: k = 0 (student) and k = m (the full swap, which IS the
        # teacher set).  The x for the endpoint must be the true |T_only| of
        # this benchmark, not the last grid point -- |T_only| varies by
        # instance, so the curve's right edge is its mean.
        m_ds = float(np.mean([len(c["T_only"]) for c in _cases if c["ds"] == ds]))
        kk_ext = [0] + kk + [m_ds]
        vv_ext = [b["student"]] + vals + [b["teacher"]]
        row["k_mean_T_only"] = m_ds
        row["k_for"] = {}
        for tgt in (0.25, 0.50, 0.75):
            row["k_for"][f"raw_{int(tgt*100)}"] = interp_k(kk_ext, vv_ext, tgt,
                                                           b["student"], b["teacher"])
            row["k_for"][f"prize_{int(tgt*100)}"] = interp_k(kk_ext, vv_ext, tgt,
                                                             b["eadp_topk"], b["teacher"])
        levels[ds] = row

    # ---- marginal value along the teacher curve ---------------------------
    marginal = {}
    for ds in DS_ALL:
        kk = levels[ds]["curve_k"]; vv = levels[ds]["curve_acc"]
        b = baselines[ds]
        pts = [(0, b["student"])] + list(zip(kk, vv)) + [(max(kk) + 8, b["teacher"])]
        marginal[ds] = [dict(k0=pts[i][0], k1=pts[i + 1][0],
                             dk=pts[i + 1][0] - pts[i][0],
                             dacc=pts[i + 1][1] - pts[i][1],
                             per_token=(pts[i + 1][1] - pts[i][1]) / (pts[i + 1][0] - pts[i][0]))
                        for i in range(len(pts) - 1)]

    # ---- recovery curve vs random control, as a paired delta ---------------
    control = {}
    for ds in DS_ALL:
        kk = sorted(curves["random"][ds])
        for k in kk:
            if k not in curves["teacher"][ds]:
                continue
            arm_t, arm_r0 = f"teacher:{k}", f"random:{k}:s0"
            if arm_r0 not in hits or ds not in hits[arm_r0]:
                continue
            a = np.array([hits[arm_t][ds][i] for i in order[ds]])
            b_ = np.array([hits[arm_r0][ds][i] for i in order[ds]])
            control[f"{k}|{ds}"] = dict(
                teacher=float(a.mean() * 100), random=float(b_.mean() * 100),
                delta=float((a - b_).mean() * 100))
    for ds in DS_ALL:
        for k in sorted(curves["shuffled"][ds]):
            arm = f"shuffled:{k}"
            if arm not in hits or ds not in hits[arm]:
                continue
            a = np.array([hits[f"teacher:{k}"][ds][i] for i in order[ds]])
            b_ = np.array([hits[arm][ds][i] for i in order[ds]])
            control[f"{k}|{ds}"]["shuffled"] = float(b_.mean() * 100)
            control[f"{k}|{ds}"]["delta_vs_shuffled"] = float((a - b_).mean() * 100)

    # ---- F_rev identity check ---------------------------------------------
    frev = {}
    for ds in DS_ALL:
        m = int(round(np.mean([len(r["T_only"]) for r in
                               __import__("s2c2_common", fromlist=["x"]).load_case()
                               if r["ds"] == ds])))
        for k in (16, 32, 64, 96):
            arm = f"F_rev:{k}"
            if arm not in hits or ds not in hits[arm]:
                continue
            frev[f"{k}|{ds}"] = dict(acc=float(np.mean(
                [hits[arm][ds][i] for i in order[ds]]) * 100),
                expected_teacher_k=m - k, mean_m=m)

    # ---- leave-one-out ablation summary -----------------------------------
    loo = {}
    if plan and resc.get("runs"):
        rruns = resc["runs"]
        def hit_of(arm, key):
            r = rruns.get(f"{arm}|{key.rsplit('_', 1)[0]}")
            if not r or key not in r["keys"]:
                return None
            return float(r["hits"][r["keys"].index(key)])
        rows = []
        for rec in plan["rescuable"]:
            key, k = rec["key"], rec["ref_k"]
            if k is None or not rec["block_js"]:
                continue
            base = hit_of(f"teacher:{k}", key)
            if base is None:
                continue
            blk = [hit_of(f"loo:{j}", key) for j in rec["block_js"]]
            blk = [h for h in blk if h is not None]
            early_j = [j for j in range(0, max(rec["prev_k"] or 0, 1))]
            early_j = early_j[:3]
            early = [hit_of(f"looearly:{j}", key) for j in early_j]
            early = [h for h in early if h is not None]
            rows.append(dict(
                key=key, ds=rec["ds"], k=k, block=rec["block"],
                n_block_ablations=len(blk),
                block_breaks=int(sum(h < 0.5 for h in blk)),
                early_ablations=len(early),
                early_breaks=int(sum(h < 0.5 for h in early)),
                base_ok=bool(base is not None and base >= 0.5)))
        loo = dict(
            n_instances=len(rows), rows=rows,
            total_block_ablations=int(sum(r["n_block_ablations"] for r in rows)),
            total_block_breaks=int(sum(r["block_breaks"] for r in rows)),
            total_early_ablations=int(sum(r["early_ablations"] for r in rows)),
            total_early_breaks=int(sum(r["early_breaks"] for r in rows)),
            instances_with_any_block_break=int(sum(r["block_breaks"] > 0 for r in rows)),
            instances_with_any_early_break=int(sum(r["early_breaks"] > 0 for r in rows)),
        )

    out = dict(baselines=baselines, macro=baselines_macro(macro), curves=curves,
               outcomes=outcomes, rescue_rate=rescue_rate, loo=loo, flips=flips,
               levels=levels, marginal=marginal, control=control, frev=frev,
               arm_deltas={a: arm_delta(a) for a in sorted(hits)
                           if a.split(":")[0] in ("teacher", "random", "adversarial",
                                                  "shuffled", "F_rev")},
               decompose=dec, rescue_plan=plan, characterize=char,
               rescue_runs={k: dict(acc_pct=v["acc_pct"], n=v["n"])
                            for k, v in resc.get("runs", {}).items()})
    path = os.path.join(OUT, args.out)
    json.dump(out, open(path, "w"), indent=1)
    print(f"[saved] {path}")

    # ---- console -----------------------------------------------------------
    print(f"\nbaselines  student {macro['student']:.3f}  teacher {macro['teacher']:.3f}  "
          f"EADP-TopK {macro['eadp_topk']:.3f}  raw gap {macro['raw_gap']:.3f}  "
          f"prize {macro['prize']:.3f}")
    print(f"\n{'ds':14s}{'k':>5s}{'acc':>9s}{'rec_raw':>9s}{'rec_prize':>10s}"
          f"{'rand':>8s}{'adv':>8s}{'shuf':>8s}")
    for ds in DS_ALL:
        for k in levels[ds]["curve_k"]:
            r = curves["random"][ds].get(k, {}).get("mean")
            a = curves["adversarial"][ds].get(k)
            s = curves["shuffled"][ds].get(k)
            i = levels[ds]["curve_k"].index(k)
            print(f"{ds:14s}{k:5d}{levels[ds]['curve_acc'][i]:9.2f}"
                  f"{levels[ds]['recovery_raw'][i]:9.3f}"
                  f"{levels[ds]['recovery_prize'][i]:10.3f}"
                  f"{(f'{r:.2f}' if r else '-'):>8s}"
                  f"{(f'{a:.2f}' if a else '-'):>8s}"
                  f"{(f'{s:.2f}' if s else '-'):>8s}")
    if loo:
        print(f"\nleave-one-out at the rescue level: {loo['n_instances']} instances, "
              f"{loo['total_block_ablations']} block ablations "
              f"({loo['total_block_breaks']} broke the rescue); "
              f"control {loo['total_early_ablations']} early ablations "
              f"({loo['total_early_breaks']} broke it)")
    print("\nk needed for 25/50/75 % of the raw student->teacher gap:")
    for ds in DS_ALL:
        kf = levels[ds]["k_for"]
        mm = float(np.mean([len(c["T_only"]) for c in _cases if c["ds"] == ds]))
        print(f"  {ds:14s} raw[25]={kf['raw_25']} raw[50]={kf['raw_50']} "
              f"raw[75]={kf['raw_75']}   (|T_only|~{mm:.0f})")


def baselines_macro(m):
    return m


if __name__ == "__main__":
    main()
