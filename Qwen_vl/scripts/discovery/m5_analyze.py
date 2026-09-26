"""
M5 (Safe Removal) step 4 -- tables, controls and the frozen verdict.

Reads stored artefacts only (no GPU).  Four things it is careful about:

1. **The verdict reads ONE pre-registered arm**, `m5_accuracy.PRIMARY_ARM`
   (`SAFE-k8`), frozen before the grid ran: k=8 is where M4's `EVICT-r8` lead
   was measured, and the lead came from a `maxred` eviction, so `MAXRED-k8` is
   the exact training-free rule the method has to beat.
2. **Every threshold is printed with the resolution it needs.**  M3-v0 measured
   the paired macro SE on this 150 at ~2-4 points and the MDE at 80 % power at
   7.0; M4 measured it at 2.9 and 8.2.  A +2 point estimate is therefore NOT
   evidence on its own, and every delta below is printed with its paired
   bootstrap CI and the McNemar p on discordant pairs.
3. **The learning is priced separately from the rule.**  `SAFE - MAXRED` at the
   same k is the contrast that decides whether answer-conditioned supervision
   contributes anything (brief §12).  If it does not, the method is a
   training-free redundancy cleanup and is reported as one.
4. **The arms break the 256-token budget on purpose.**  Every M5 arm returns
   `256 - k` tokens.  That is the point of the round -- the brief's §9 asks
   whether the prescribed budget is the optimal retained size -- so the token
   count is printed beside every score and no arm is ever compared to B2
   without it.

Usage
    python scripts/discovery/m5_analyze.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUT_DIR                                       # noqa: E402
from m2_accuracy import DS_ORDER                                     # noqa: E402
from m3_analyze import mcnemar_p, paired                             # noqa: E402
from m5_accuracy import K_GRID, PRIMARY_ARM, RULE_FAMILIES           # noqa: E402

B2_MACRO = 59.88
B1_MACRO = 61.10
B0_MACRO = 75.625
# The brief's §10 gates, verbatim.
GATE_STRONG_DELTA = 2.0
GATE_PROMISING_DELTA = 0.0        # "accuracy does not drop", pushed further
STABLE_MARGIN = 2.0


def load(tag):
    with open(os.path.join(OUTPUT_DIR, f"{tag}.json")) as f:
        return json.load(f)


def macro_of(r):
    return float(np.mean([r["per_benchmark"][ds]["acc_pct"] for ds in DS_ORDER]))


def rows_of(m5, m2):
    rows = {}
    for k in ("B0", "B1", "B2"):
        r = m2["arms"].get(k)
        if not r or "error" in r:
            continue
        rows[k] = dict(arm=k, macro=macro_of(r), kind="baseline", k=0,
                       tokens=1024 if k == "B0" else 256,
                       per_ds={ds: r["per_benchmark"][ds]["acc_pct"]
                               for ds in DS_ORDER},
                       hits={ds: r["per_benchmark"][ds]["hits"] for ds in DS_ORDER},
                       predictions=r["predictions"])
    for k, r in m5["arms"].items():
        if "error" in r:
            rows[k] = dict(arm=k, error=r["error"][-400:])
            continue
        kk = int(r["spec"].get("k", 0) or 0)
        rows[k] = dict(arm=k, macro=macro_of(r),
                       kind="identity" if kk == 0 else "arm",
                       k=kk, tokens=256 - kk, spec=r["spec"],
                       trim_ms=r.get("trim_ms_median"),
                       wall=r.get("wall_seconds"),
                       per_ds={ds: r["per_benchmark"][ds]["acc_pct"]
                               for ds in DS_ORDER},
                       hits={ds: r["per_benchmark"][ds]["hits"] for ds in DS_ORDER},
                       predictions=r["predictions"],
                       per_image_meta=r.get("per_image_meta"))
    return rows


def add_contrasts(rows):
    refs = [k for k in ("B2", "B1", "B0") if k in rows and "hits" in rows[k]]
    for k, r in rows.items():
        if "hits" not in r or k in ("B0", "B1", "B2"):
            continue
        for ref in refs:
            d = paired(r["hits"], rows[ref]["hits"])
            d["mcnemar_p"] = mcnemar_p(d["fixed"], d["broken"])
            r[f"vs_{ref}"] = d
        r["n_pred_changed_vs_B2"] = int(sum(
            1 for x, y in zip(r["predictions"], rows["B2"]["predictions"])
            if x.strip() != y.strip()))


def add_pairwise(rows):
    """SAFE - RULE at the same k: what the learning is worth (brief §12)."""
    for fam in RULE_FAMILIES:
        for k in K_GRID:
            a, b = f"SAFE-k{k}", f"{fam}-k{k}"
            if a not in rows or b not in rows:
                continue
            if "hits" not in rows[a] or "hits" not in rows[b]:
                continue
            d = paired(rows[a]["hits"], rows[b]["hits"])
            d["mcnemar_p"] = mcnemar_p(d["fixed"], d["broken"])
            rows[a][f"vs_{b}"] = d
            rows[a][f"vs_{b}_pred_changed"] = int(sum(
                1 for x, y in zip(rows[a]["predictions"], rows[b]["predictions"])
                if x.strip() != y.strip()))


def _stable(delta, ci):
    return bool(delta >= STABLE_MARGIN and (ci[0] > 0 or ci[1] < 0))


def family_curve(rows, fam):
    out = []
    for k in K_GRID:
        key = f"{fam}-k{k}" if fam != "SAFE" else f"SAFE-k{k}"
        r = rows.get(key)
        if not r or "hits" not in r:
            continue
        out.append(dict(k=k, tokens=256 - k, macro=r["macro"],
                        delta=r.get("vs_B2", {}).get("delta"),
                        ci=r.get("vs_B2", {}).get("ci"),
                        fix=r.get("vs_B2", {}).get("fixed"),
                        brk=r.get("vs_B2", {}).get("broken"),
                        trim_ms=r.get("trim_ms")))
    return out


def peak(curve):
    if not curve:
        return None
    return max(curve, key=lambda r: r["macro"])


def print_table(rows):
    print(f"\n{'arm':16s} {'tok':>4s} {'TextVQA':>8s} {'DocVQA':>8s} "
          f"{'OCR':>7s} {'macro':>7s} {'dB2':>7s} {'CI':>16s} {'p':>6s} "
          f"{'fix/brk':>8s} {'ms':>6s}")
    for k in ("B0", "B1", "B2"):
        r = rows.get(k)
        if not r or "hits" not in r:
            continue
        print(f"{k:16s} {r['tokens']:>4d} {r['per_ds']['TextVQA_VAL']:>8.3f} "
              f"{r['per_ds']['DocVQA_VAL']:>8.3f} {r['per_ds']['OCRBench']:>7.3f} "
              f"{r['macro']:>7.3f}")
    for name in sorted(k for k in rows if rows[k].get("kind") == "arm"):
        r = rows[name]
        v = r.get("vs_B2", {})
        ci = v.get("ci", [float("nan")] * 2)
        print(f"{name:16s} {r['tokens']:>4d} {r['per_ds']['TextVQA_VAL']:>8.3f} "
              f"{r['per_ds']['DocVQA_VAL']:>8.3f} {r['per_ds']['OCRBench']:>7.3f} "
              f"{r['macro']:>7.3f} {v.get('delta', float('nan')):>+7.2f} "
              f"[{ci[0]:>+6.2f},{ci[1]:>+6.2f}] {v.get('mcnemar_p', float('nan')):>6.3f} "
              f"{v.get('fixed', 0):>3d}/{v.get('broken', 0):<3d} "
              f"{(r.get('trim_ms') or 0):>6.2f}")


def gate(rows, m5, probe=None):
    """The brief's §10 gate, applied to the pre-registered primary arm.

    Every clause of the brief's five branches is evaluated separately and
    printed, so a verdict can be read as "which clauses held" rather than as a
    label.  `learnable` is the one clause that cannot be read off the accuracy
    grid: it is the brief's §3 early gate -- whether the probe's bottom-risk
    tokens are safe more often than the training-free rules' -- and it comes
    from `m5_probe.json`.
    """
    a = rows.get(PRIMARY_ARM)
    out = dict(primary_arm=PRIMARY_ARM, applied_to=None)
    if not a or "hits" not in a:
        return dict(out, error="primary arm missing or failed")
    v = a["vs_B2"]
    mx = rows.get(f"MAXRED-k{a['k']}")
    vm = a.get(f"vs_MAXRED-k{a['k']}", {})
    rd = rows.get(f"RANDOM-k{a['k']}-s0") or rows.get(f"RANDOM-k{a['k']}")
    vrd = a.get(f"vs_RANDOM-k{a['k']}", {})

    # clause 1: is the safe-removal signal learnable at all?
    learn = dict(checked=False, reason="no m5_probe.json")
    if probe:
        b = probe["by_split"]["val"]
        def g(arm, key="bottom_k8_safe_precision"):
            return b.get(arm, {}).get(key)
        p_safe = g("probe:mlp")
        r_safe = {r: g(f"rule:{r}") for r in ("maxred", "lowimp", "random")}
        best_rule = max((x for x in r_safe.values() if x is not None),
                        default=None)
        learn = dict(probe_bottom8_safe_precision=p_safe, rules=r_safe,
                     best_rule=best_rule,
                     probe_auroc=g("probe:mlp", "auroc_harmful_mean"),
                     maxred_auroc=g("rule:maxred", "auroc_harmful_mean"),
                     checked=True,
                     passed=bool(p_safe is not None and best_rule is not None
                                 and p_safe > best_rule))

    # clause 5: pre-LLM and forward-only.  A property of the implementation,
    # asserted rather than measured, and stated as such.
    prellm = dict(mode="prellm", decoder_layers_before_pruning=0,
                  backward_passes_at_inference=0, teacher_at_inference=False,
                  passed=True)

    out.update(applied_to=a["arm"], macro=a["macro"], tokens=a["tokens"],
               delta_b2=v["delta"], ci_b2=v["ci"], p_b2=v["mcnemar_p"],
               fix=v["fixed"], brk=v["broken"],
               maxred_macro=(mx["macro"] if mx and "hits" in mx else None),
               maxred_delta_b2=(mx.get("vs_B2", {}).get("delta")
                                if mx and "hits" in mx else None),
               delta_vs_maxred=vm.get("delta"), ci_vs_maxred=vm.get("ci"),
               p_vs_maxred=vm.get("mcnemar_p"),
               random_macro=(rd["macro"] if rd and "hits" in rd else None),
               delta_vs_random=vrd.get("delta"), ci_vs_random=vrd.get("ci"),
               learnable=learn, prellm=prellm)

    strong = bool(
        learn.get("passed")                                   # (1) learnable
        and v["delta"] >= GATE_STRONG_DELTA                   # (2) >= B2 + 2
        and _stable(v["delta"], v["ci"])
        and (vm.get("delta") or -1) >= GATE_STRONG_DELTA      # (3) beats MAXRED
        and a["tokens"] < 256                                 # (4) fewer tokens
        and prellm["passed"])                                 # (5) deployable
    promising = bool(
        v["delta"] >= GATE_PROMISING_DELTA                    # accuracy holds
        and (vm.get("delta") or -1) > 0                       # beats MAXRED
        and a["tokens"] < 256)                                # at fewer tokens
    heuristic = bool(mx is not None and "hits" in mx
                     and (mx.get("vs_B2", {}).get("delta") or -1) > 0
                     and (vm.get("delta") is None or vm["delta"] <= 0))
    out.update(strong=strong, promising=promising, heuristic_only=heuristic)
    out["verdict"] = ("STRONG" if strong else "PROMISING" if promising
                      else "HEURISTIC-ONLY" if heuristic else "REFUTED")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m5_accuracy")
    ap.add_argument("--out", default="m5_analysis")
    args = ap.parse_args()

    m5 = load(args.tag)
    m2 = load("m2_accuracy")
    probe = load("m5_probe") if os.path.exists(
        os.path.join(OUTPUT_DIR, "m5_probe.json")) else None
    teacher = load("m5_teacher") if os.path.exists(
        os.path.join(OUTPUT_DIR, "m5_teacher.json")) else None

    rows = rows_of(m5, m2)
    add_contrasts(rows)
    add_pairwise(rows)

    rep = dict(primary_arm=PRIMARY_ARM, gates=m5.get("gates", {}),
               verdict=gate(rows, m5, probe))
    print_table(rows)

    rep["curves"] = {fam: family_curve(rows, fam)
                     for fam in ("SAFE",) + RULE_FAMILIES}
    rep["peaks"] = {fam: peak(c) for fam, c in rep["curves"].items()}
    print("\n-- the trim curve (macro vs tokens retained) --")
    for fam, c in rep["curves"].items():
        s = "  ".join(f"{r['tokens']}tok:{r['macro']:.2f}" for r in c)
        p = rep["peaks"][fam]
        print(f"{fam:8s} {s}   peak {p['tokens']}tok ({p['macro']:.2f})"
              if p else f"{fam:8s} (no arms)")

    # ---- the contrast that prices the learning -----------------------------
    print("\n-- SAFE - RULE at the same k (what the supervision is worth) --")
    rep["learning_contrast"] = {}
    for fam in RULE_FAMILIES:
        for k in K_GRID:
            a = rows.get(f"SAFE-k{k}")
            key = f"vs_{fam}-k{k}"
            if not a or key not in a:
                continue
            d = a[key]
            rep["learning_contrast"][f"SAFE-k{k} - {fam}-k{k}"] = dict(
                delta=d["delta"], ci=d["ci"], p=d["mcnemar_p"],
                fixed=d["fixed"], broken=d["broken"],
                pred_changed=a.get(key + "_pred_changed"))
            print(f"SAFE-k{k} - {fam}-k{k}: {d['delta']:+.2f} "
                  f"[{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}] p={d['mcnemar_p']:.3f} "
                  f"{d['fixed']}fix/{d['broken']}brk "
                  f"({a.get(key + '_pred_changed')}/150 preds moved)")

    if teacher:
        rep["teacher"] = teacher_summary(teacher)
        print("\n-- group deletion control (brief §2B) --")
        for k, v in rep["teacher"]["group"].items():
            print(f"{k:28s} D={v['D_mean']:+.3f}  "
                  f"corr(mean single d of the group, D)={v['corr']:+.3f}")
    if probe:
        rep["probe"] = probe_summary(probe)
        print("\n-- the early gate (brief §3) --")
        for split in ("fit", "val"):
            b = rep["probe"]["by_split"][split]
            print(f"[{split}] " + "  ".join(
                f"{a.split(':')[-1]}={b[a]['auroc_harmful_mean']:.3f}"
                for a in ("rule:random", "rule:maxred", "rule:lowimp",
                          "rule:recon", "probe:lr", "probe:mlp", "probe:vis")))
        hc = rep["probe"]["head"]
        print("[head] " + "  ".join(
            f"K={hc[k]['K']}/{hc[k]['split']}: AUROC {hc[k]['auroc']:.3f} "
            f"recall@16 {hc[k]['recall_at_16']:.3f}"
            for k in sorted(hc)))

    from m2_gdep import dump_json
    dump_json(f"{args.out}.json", rep)
    print(f"\n[verdict] {rep['verdict']['verdict']}")


def teacher_summary(teacher):
    """Does a single-token safe label predict the GROUP's harm?"""
    groups, singles = {}, {}
    for k, m in teacher["meas"].items():
        singles[k] = {int(s["token"]): s["d"] for s in m["singles"]}
        for g in m["groups"]:
            groups.setdefault((g["k"], g["rule"]), []).append(
                dict(key=k, D=g["D"], tokens=g["tokens"], hit=g.get("hit")))
    out = dict(n_instances=len(teacher["done_keys"]),
               floors=dict(mean_mean=float(np.mean(
                   [m["floor_mean"] for m in teacher["meas"].values()])),
                   median_mean=float(np.median(
                       [m["floor_mean"] for m in teacher["meas"].values()])),
                   p90_mean=float(np.percentile(
                       [m["floor_mean"] for m in teacher["meas"].values()], 90)),
                   max_rep=float(max(m["floor_rep"]
                                     for m in teacher["meas"].values()))),
               group={})
    for (k, rule), gs in sorted(groups.items()):
        Ds = np.array([g["D"] for g in gs])
        pred = np.array([np.mean([singles[g["key"]][t] for t in g["tokens"]])
                         for g in gs])
        ok = np.isfinite(pred) & np.isfinite(Ds)
        corr = (float(np.corrcoef(pred[ok], Ds[ok])[0, 1])
                if ok.sum() > 2 else float("nan"))
        row = dict(n=int(len(Ds)), D_mean=float(Ds.mean()),
                   D_median=float(np.median(Ds)),
                   D_frac_positive=float(np.mean(Ds > 0)),
                   corr=corr,
                   sum_single_mean=float(pred.mean()),
                   corr_sumsingle=(float(np.corrcoef(pred[ok] * k, Ds[ok])[0, 1])
                                   if ok.sum() > 2 else float("nan")))
        hits = [g["hit"] for g in gs if g.get("hit") is not None]
        if hits:
            row["hit_rate"] = float(np.mean(hits))
        out["group"][f"k{k}-{rule}"] = row
    return out


def probe_summary(probe):
    out = dict(thresholds=probe["thresholds"], d_stats=probe["d_stats"],
               head=probe["head_comparison"], by_split={},
               by_target={})
    for tname in ("harmful", "notsafe"):
        rs = [r for r in probe["probes"] if r.get("target") == tname]
        if not rs:
            continue
        out["by_target"][tname] = {
            f"{r['arm']}|{r['split']}": dict(
                auroc_harmful=r["auroc_harmful"], auroc_safe=r["auroc_safe"],
                spearman_d=r["spearman_d"],
                bottom8_safe_precision=r["bottom"][1]["safe_precision"],
                bottom8_harm_rate=r["bottom"][1]["harm_rate"])
            for r in rs if r["seed"] == 0}
    allrows = probe["probes"] + probe["rules"]
    for split in ("fit", "val"):
        block = {}
        for arm in sorted({r["arm"] for r in allrows}):
            # The primary target only; the `notsafe` fits are reported in full
            # inside m5_probe.json and summarised separately below.
            rs = [r for r in allrows if r["arm"] == arm and r["split"] == split
                  and r.get("target", "harmful") == "harmful"]
            if not rs:
                continue
            block[arm] = dict(
                n=rs[0]["n"],
                auroc_harmful_mean=float(np.mean([r["auroc_harmful"] for r in rs])),
                auroc_harmful_sd=float(np.std([r["auroc_harmful"] for r in rs])),
                auroc_safe_mean=float(np.mean([r["auroc_safe"] for r in rs])),
                spearman_d_mean=float(np.mean([r["spearman_d"] for r in rs])),
                # The per-instance twins: the probe ranks one image's 256
                # tokens at a time, so these are the protocol it runs under.
                auroc_harmful_per_instance=float(np.mean(
                    [r["per_instance"]["auroc_harmful"] for r in rs])),
                spearman_d_per_instance=float(np.mean(
                    [r["per_instance"]["spearman_d"] for r in rs])),
                bottom_k8_safe_precision=float(np.mean(
                    [r["bottom"][1]["safe_precision"] for r in rs])),
                bottom_k8_harm_rate=float(np.mean(
                    [r["bottom"][1]["harm_rate"] for r in rs])),
                bottom_k8_mean_d=float(np.mean(
                    [r["bottom"][1]["mean_d"] for r in rs])))
        out["by_split"][split] = block
    return out


if __name__ == "__main__":
    main()
