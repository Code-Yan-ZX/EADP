"""
S2-C2 step 1: teacher/student disagreement decomposition (CPU only, no model).

Answers, from the S2-C1 caches alone:

  * how much of the teacher's Top-256 the student recovers, per benchmark;
  * how the four correctness quadrants are distributed;
  * for the class that matters -- student wrong, teacher correct -- how much
    disagreement there is to spend, i.e. the budget a counterfactual swap has to
    work with;
  * whether the cached teacher/student baselines reproduce the S2-C1 report
    numbers (harness identity check before any new generation is run).

Writes ``s2c2_decompose.json``.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT  # noqa: E402
from s2c2_common import (BUDGET, CLASSES, DS_ALL, FEATURES_META, PILOT_STUDENT,  # noqa: E402
                         PILOT_TEACHER, STUDENT_SCORES, TEACHER_SCORES, cls_name,
                         load_case, load_runs, macro)


def main():
    cases = load_case()
    pil, stu = load_runs()

    out = {"budget": BUDGET, "n_total": len(cases),
           "per_dataset": {ds: {} for ds in DS_ALL}, "overall": {}}

    # ---- 0. cache-identity checks -----------------------------------------
    checks = {}
    T = np.load(os.path.join(OUT, TEACHER_SCORES))
    S = np.load(os.path.join(OUT, STUDENT_SCORES))
    ok = 0
    for c in cases[:30]:                      # spot-check the rank-order claim
        ts = T[f"{c['ds']}_{c['idx']}"]
        ss = S[f"LIN_L4__{c['ds']}_{c['idx']}"]
        ok += int(np.array_equal(np.array(c["T"]), np.argsort(-ts, kind="stable")[:BUDGET]))
        ok += int(np.array_equal(np.array(c["S"]), np.argsort(-ss, kind="stable")[:BUDGET]))
    checks["select_idx_is_descending_score_order"] = f"{ok}/{2 * 30}"
    checks["cached_macro_teacher"] = macro(
        {ds: np.array([c["hit_teacher"] for c in cases if c["ds"] == ds]) for ds in DS_ALL})
    checks["cached_macro_student"] = macro(
        {ds: np.array([c["hit_student"] for c in cases if c["ds"] == ds]) for ds in DS_ALL})
    checks["reported_s2c1"] = {"teacher": 75.150, "student": 57.318}
    out["cache_checks"] = checks

    # ---- 1. overlap structure ---------------------------------------------
    for ds in DS_ALL + ["ALL"]:
        slot = out["overall"] if ds == "ALL" else out["per_dataset"][ds]
        sub = cases if ds == "ALL" else [c for c in cases if c["ds"] == ds]
        nC = np.array([len(c["C"]) for c in sub], dtype=float)
        nTo = np.array([len(c["T_only"]) for c in sub], dtype=float)
        nSo = np.array([len(c["S_only"]) for c in sub], dtype=float)
        slot["overlap"] = {
            "n": len(sub),
            "mean_C": float(nC.mean()), "mean_T_only": float(nTo.mean()),
            "mean_S_only": float(nSo.mean()),
            "overlap_frac": float(nC.mean() / BUDGET),
            "T_only_frac": float(nTo.mean() / BUDGET),
            "S_only_frac": float(nSo.mean() / BUDGET),
            "min_C": float(nC.min()), "max_C": float(nC.max()),
        }

    # ---- 2. quadrants ------------------------------------------------------
    for ds in DS_ALL + ["ALL"]:
        slot = out["overall"] if ds == "ALL" else out["per_dataset"][ds]
        sub = cases if ds == "ALL" else [c for c in cases if c["ds"] == ds]
        cnt = {k: 0 for k in CLASSES}
        for c in sub:
            cnt[cls_name(c["hit_teacher"], c["hit_student"])] += 1
        hit_t = np.array([c["hit_teacher"] for c in sub])
        hit_s = np.array([c["hit_student"] for c in sub])
        rec = dict(
            n=len(sub), counts=cnt,
            frac={k: cnt[k] / len(sub) for k in CLASSES},
            mean_hit_teacher=float(hit_t.mean()), mean_hit_student=float(hit_s.mean()),
            acc_teacher=float((hit_t >= 0.5).mean() * 100),
            acc_student=float((hit_s >= 0.5).mean() * 100),
            # graded gap on the same instances (the S2-C1 "retained gain" is
            # defined against EADP Top-K, this is the raw student->teacher gap)
            graded_gap=float((hit_t.mean() - hit_s.mean()) * 100),
        )
        # disagreement available to a budget-preserving swap, by class
        rec["mean_T_only_by_class"] = {
            k: float(np.mean([len(c["T_only"]) for c in sub
                              if cls_name(c["hit_teacher"], c["hit_student"]) == k]))
            for k in CLASSES if cnt[k]}
        slot["quadrants"] = rec

    # ---- 3. baselines to beat (cached, no new generation) ------------------
    base = {}
    for ds in DS_ALL:
        rT = pil[f"b{BUDGET}|topk|P1G2|{ds}"]
        rS = stu[f"b{BUDGET}|topk|LIN_L4|{ds}"]
        base[ds] = dict(teacher_topk=rT["acc_pct"], student_topk=rS["acc_pct"])
    rO = pil[f"b{BUDGET}|topk|official|{ds}"]
    for ds in DS_ALL:
        base[ds]["eadp_topk"] = pil[f"b{BUDGET}|topk|official|{ds}"]["acc_pct"]
        base[ds]["teacher_shuffled_topk"] = pil[f"b{BUDGET}|topk|P1G2+shuf7|{ds}"]["acc_pct"]
    out["baselines_topk"] = base
    out["baselines_macro"] = {
        "teacher": float(np.mean([base[d]["teacher_topk"] for d in DS_ALL])),
        "student": float(np.mean([base[d]["student_topk"] for d in DS_ALL])),
        "eadp_topk": float(np.mean([base[d]["eadp_topk"] for d in DS_ALL])),
        "teacher_shuffled": float(np.mean([base[d]["teacher_shuffled_topk"] for d in DS_ALL])),
    }

    # ---- 3b. head agreement: recall of the teacher's *top-k* --------------
    # The headline overlap (0.562) is a Top-256 number.  Section 4 shows the
    # value sits in the teacher's first few dozen ranks, so what matters is
    # agreement at the *head* of the ranking, not at the budget boundary.
    HEAD_KS = (8, 16, 32, 64, 128, 256)
    head = {}
    for ds in DS_ALL + ["ALL"]:
        sub = cases if ds == "ALL" else [c for c in cases if c["ds"] == ds]
        rec = {}
        for k in HEAD_KS:
            vals = [len(set(c["T"][:k]) & set(c["S"][:k])) / k for c in sub]
            within = [len(set(c["T"][:k]) & set(c["S"])) / k for c in sub]
            rec[str(k)] = dict(
                topk_agreement=float(np.mean(vals)),      # what the student puts at the top
                in_selection=float(np.mean(within)),      # what it keeps anywhere in 256
                chance=k / BUDGET / 4)                    # k/1024
        # the S2-C0 attention proxies, for the same reading
        for arm in ("A_L2", "C_L2"):
            try:
                P = np.load(os.path.join(OUT, "s2c0_forward_proxy.npz"))
                vals = {}
                for k in HEAD_KS:
                    v = []
                    for c in sub:
                        key = f"{arm}__{c['ds']}_{c['idx']}"
                        if key not in P.files:
                            continue
                        o = np.argsort(-P[key], kind="stable")
                        v.append(len(set(c["T"][:k]) & set(o[:k])) / k)
                    vals[str(k)] = float(np.mean(v)) if v else None
                rec[arm] = vals
            except FileNotFoundError:
                pass
        head[ds] = rec
    out["head_agreement"] = head

    # ---- 4. the rescuable set ---------------------------------------------
    res = [c for c in cases if cls_name(c["hit_teacher"], c["hit_student"])
           == "student_wrong_teacher_correct"]
    out["rescuable"] = {
        "n": len(res),
        "per_ds": {ds: sum(1 for c in res if c["ds"] == ds) for ds in DS_ALL},
        "keys": [c["key"] for c in res],
        "mean_T_only": float(np.mean([len(c["T_only"]) for c in res])) if res else 0.0,
        "mean_S_only": float(np.mean([len(c["S_only"]) for c in res])) if res else 0.0,
        "mean_hit_student": float(np.mean([c["hit_student"] for c in res])) if res else 0.0,
        "mean_hit_teacher": float(np.mean([c["hit_teacher"] for c in res])) if res else 0.0,
    }

    # ---- 5. feature-plan index map (for step 4 characterization) -----------
    meta = json.load(open(os.path.join(OUT, FEATURES_META)))
    jmap = {p["key"]: j for j, p in enumerate(meta["plan"])}
    missing = [c["key"] for c in cases if c["key"] not in jmap]
    out["feature_plan"] = {"n_plan": len(meta["plan"]),
                           "held_out_covered": len(cases) - len(missing),
                           "missing": missing,
                           "splits_of_held_out": sorted(
                               {meta["plan"][jmap[c["key"]]]["split"] for c in cases
                                if c["key"] in jmap})}

    path = os.path.join(OUT, "s2c2_decompose.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)

    # ---- console summary ---------------------------------------------------
    print(f"[cache] select_idx order check: {checks['select_idx_is_descending_score_order']}")
    print(f"[cache] teacher macro {checks['cached_macro_teacher']:.3f} "
          f"(S2-C1 report 75.150)   student macro {checks['cached_macro_student']:.3f} "
          f"(S2-C1 report 57.318)")
    print("\noverlap / |C| of 256:")
    for ds in DS_ALL:
        o = out["per_dataset"][ds]["overlap"]
        print(f"  {ds:14s} |C|={o['mean_C']:6.2f} ({o['overlap_frac']:.3f})  "
              f"|T_only|={o['mean_T_only']:6.2f}  |S_only|={o['mean_S_only']:6.2f}")
    print("\nquadrants (rows = benchmark):")
    hdr = "".join(f"{k[:22]:>24s}" for k in CLASSES)
    print(f"  {'':14s}{hdr}")
    for ds in DS_ALL + ["ALL"]:
        q = (out["overall"] if ds == "ALL" else out["per_dataset"][ds])["quadrants"]
        print(f"  {ds:14s}" + "".join(f"{q['counts'][k]:>24d}" for k in CLASSES))
    print("\ngraded gap (teacher - student, points):")
    for ds in DS_ALL + ["ALL"]:
        q = (out["overall"] if ds == "ALL" else out["per_dataset"][ds])["quadrants"]
        print(f"  {ds:14s} teacher {q['acc_teacher']:6.2f}  student {q['acc_student']:6.2f}  "
              f"gap {q['graded_gap']:6.2f}")
    print("\nhead agreement -- share of the teacher's top-k that the student also ranks top-k:")
    hk = sorted((int(x) for x in out["head_agreement"]["ALL"] if x.isdigit()))
    print("  " + "".join(f"{k:>9d}" for k in hk))
    for label, key in (("LIN_L4 top-k", "topk_agreement"),
                       ("kept anywhere in 256", "in_selection")):
        print(f"  {label:26s}" + "".join(
            f"{out['head_agreement']['ALL'][str(k)][key]:9.3f}" for k in hk))
    print(f"  {'chance':26s}" + "".join(f"{k / BUDGET / 4:9.3f}" for k in hk))
    print(f"\nrescuable (student wrong, teacher correct): n={out['rescuable']['n']} "
          f"{out['rescuable']['per_ds']}, mean |T_only|={out['rescuable']['mean_T_only']:.1f}")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
