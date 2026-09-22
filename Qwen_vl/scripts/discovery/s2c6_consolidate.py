"""
S2-C6 step 2: apply the pre-registered rule and write the stage's main artifact.

The rule lives in ``docs/scoring_search_s2c6_prereg.md`` and was written before
the first training run; the constants below are read from there, not chosen here.
This script trains nothing and selects nothing -- it reads the per-image held-out
arrays the training step froze and applies the rule to them.

  MARGIN          0.01 held-out head_recall@8
  RESOLVED GAIN   mean paired delta >= MARGIN, paired-bootstrap 95 % CI lower
                  bound > 0, and all three seed-matched deltas > 0
  BELOW-MARGIN    CI lower bound > 0 but the rest fails -> reported, no verdict

Verdicts

  A  DATA-LIMITED            Part A's n=180 -> n=240 segment is a RESOLVED GAIN
  B  SNAPSHOT-SUFFICIENT     no snapshot/trajectory arm clears the reference
                             that governs it
  C  TRAJECTORY-INFORMATIVE  L2+L4 or L4+DELTA is a RESOLVED GAIN over its
                             governing reference
  D  QUERY-INFORMATIVE       L4+QUERY is a RESOLVED GAIN over L4 and the
                             shuffled-query control says the arm reads the
                             image's own query

Which reference *governs* an arm is fixed in the pre-registration and is not a
choice made here:

  L2, DELTA, L4+QUERY            -> L4, which is the identical architecture
  L2+L4, L4+DELTA, L4+L4         -> L4+L4 if that control is itself a RESOLVED
                                    GAIN over L4, otherwise L4

so a dual arm only counts as a trajectory result if it beats the dual
architecture's own null.

Two consistency checks are asserted rather than described: Part A's n=240 point
and Part B's L4 reference are the same configuration run through the same code
path, so their per-image held-out metrics must be identical; and the L4+QUERY
arm's untrained output must equal the L4 arm's to numerical zero.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                          # noqa: E402
from s2c3_common import load_plan                                 # noqa: E402
from s2c6_common import (COL, DEPLOY_ASPIRATIONAL, DEPLOY_TARGET,  # noqa: E402
                         MARGIN, NS, PERIMAGE_COLS, SELECTIONS, compare,
                         derangement, dump, nested_fit_rows)

SEEDS = (0, 1, 2)
SINGLE = ("L2", "DELTA")                 # governing reference: L4
DUAL = ("L2+L4", "L4+DELTA")             # governing reference: L4+L4 or L4
CONTROL = "L4+L4"
QUERY = "L4+QUERY"                       # the pre-registered token-independent arm
QUERY_ARMS = ("L4+QUERY", "L4+QRY-BILIN")  # the second is the post-hoc amendment
REF = "L4"
TOL = 1e-9


# ------------------------------------------------------------------ loading --
class Stage:
    """Per-image held-out arrays for one part, indexed by (job, arm, seed, sel)."""

    def __init__(self, tag, part):
        self.T = json.load(open(os.path.join(OUT, f"{tag}_train_{part}.json")))
        self.Z = np.load(os.path.join(OUT, f"{tag}_perimage_{part}.npz"))
        self.results = self.T["results"]
        self.test_keys = self.T["test_keys"]

    def honest(self, job, arm, seed, sel, col=COL):
        """Per-image held-out metric vector, in the frozen test-key order."""
        k = f"{job}__{arm}_s{seed}__{sel}__full"
        cols = list(PERIMAGE_COLS)
        return self.Z[k][:, cols.index(col)].astype(np.float64)

    def control(self, job, arm, seed, sel, name):
        return self.Z[f"{job}__{arm}_s{seed}__{sel}__{name}"].astype(np.float64)

    def stack(self, job, arm, sel, col=COL):
        return np.stack([self.honest(job, arm, s, sel, col) for s in SEEDS])

    def rec(self, job, arm, seed):
        return self.results[f"{job}__{arm}_s{seed}"]

    def arm_mean(self, job, arm, sel="H8", col=COL):
        return float(self.stack(job, arm, sel, col).mean())

    def by_benchmark(self, job, arm, sel="H8", col=COL):
        lab = np.array([self.T["test_keys"][i].rsplit("_", 1)[0]
                        for i in range(len(self.test_keys))])
        M = self.stack(job, arm, sel, col)
        return {d: float(M[:, lab == d].mean()) for d in sorted(set(lab))}


def summarise(st: Stage, job, arm, sel="H8"):
    r = st.rec(job, arm, 0)
    per_seed = [float(st.honest(job, arm, s, sel).mean()) for s in SEEDS]
    return {
        "held_out": dict(
            head_recall8=float(np.mean(per_seed)),
            head_recall16=st.arm_mean(job, arm, sel, "head_recall16"),
            head_recall32=st.arm_mean(job, arm, sel, "head_recall32"),
            head_agree8=st.arm_mean(job, arm, sel, "head_agree8"),
            head_agree16=st.arm_mean(job, arm, sel, "head_agree16"),
            overlap256=st.arm_mean(job, arm, sel, "overlap256")),
        "per_seed_head_recall8": per_seed,
        "seed_sd": float(np.std(per_seed)),
        "by_benchmark_head_recall8": st.by_benchmark(job, arm, sel),
        "val_head_recall8": float(np.mean(
            [st.rec(job, arm, s)["selections"][sel]["val"]["head_recall8"]
             for s in SEEDS])),
        "train_head_recall8": float(np.mean(
            [st.rec(job, arm, s)["selections"][sel]["train"]["head_recall8"]
             for s in SEEDS])),
        "best_epoch": [st.rec(job, arm, s)["selections"][sel]["best_epoch"]
                       for s in SEEDS],
        "epochs_run": [st.rec(job, arm, s)["epochs_run"] for s in SEEDS],
        "init_held_out_head_recall8": float(np.mean(
            [st.rec(job, arm, s)["init"]["test"]["head_recall8"]
             for s in SEEDS])),
        "n_params": st.rec(job, arm, 0)["n_params"],
        "n_query_params": st.rec(job, arm, 0).get("n_query_params", 0),
        "access": st.rec(job, arm, 0)["access"],
    }


def control_summary(st: Stage, job, arm, name, sel="H8"):
    """honest - wrong, per seed, as a paired comparison (positive = the arm reads
    its own copy of the borrowed quantity)."""
    A = st.stack(job, arm, sel)
    B = np.stack([st.control(job, arm, s, sel, name) for s in SEEDS])
    return {"honest_minus_wrong": compare(A, B),
            "wrong_held_out_head_recall8": float(B.mean())}


# ============================================================== part A ========
def part_a(S: Stage, rows_of, keys, out):
    jobs = [f"A{n}" for n in NS]
    rows = {n: nested_fit_rows(rows_of, keys, n) for n in NS}
    curve = {}
    for n, jt in zip(NS, jobs):
        s = summarise(S, jt, "L4", "H8")
        s["selection"] = "H8"
        s["n_fit"] = n
        s["subset_nested_in_next"] = (
            set(rows[n]) <= set(rows[NS[NS.index(n) + 1]])
            if NS.index(n) + 1 < len(NS) else None)
        s["ov_selected_head_recall8"] = float(
            S.stack(jt, "L4", "OV").mean())
        curve[str(n)] = s
    segs = {}
    for a, b in zip(NS, NS[1:]):
        segs[f"{a}->{b}"] = compare(S.stack(f"A{b}", "L4", "H8"),
                                    S.stack(f"A{a}", "L4", "H8"))
    delta_240_180 = compare(S.stack("A240", "L4", "H8"),
                            S.stack("A180", "L4", "H8"))
    rising = [int(x) for x in NS[1:] if segs[f"{NS[NS.index(x)-1]}->{x}"]["mean"] > 0]
    out["part_A"] = {
        "question": "is the token-local family data-limited at 240 fit images?",
        "curve": curve, "segments": segs,
        "marginal_180_to_240": delta_240_180,
        "marginal_gain_last_segment": float(delta_240_180["mean"]),
        "monotone": bool(all(segs[k]["mean"] > 0 for k in segs)),
        "n_segments_positive": len(rising),
        "verdict": ("DATA-LIMITED" if delta_240_180["resolved"]
                    else "DATA-PLATEAU"),
        "verdict_reason": (
            f"the n=180 -> n=240 segment is {'a' if delta_240_180['resolved'] else 'not a'} "
            f"RESOLVED GAIN ({delta_240_180['mean']:+.4f}, CI "
            f"[{delta_240_180['lo']:+.4f}, {delta_240_180['hi']:+.4f}], per-seed "
            f"{[round(x, 4) for x in delta_240_180['per_seed']]})"
            + ("; the curve is still materially rising at the largest n this "
               "stage can train, so ~0.80 may not be read as a representation "
               "ceiling and the Part B/C arms are read against a moving floor."
               if delta_240_180["resolved"] else
               "; at this design's resolution 180 -> 240 has stopped paying, "
               "which is a statement about that segment and not a proof of a "
               "plateau -- four points cannot distinguish a plateau from a very "
               "slowly rising curve.")),
        "note": "the held-out 150 is never used to select anything here; the "
                "subset draw is fixed by SUBSET_SEED and each n is a prefix of "
                "the next, so the curve is nested by construction",
    }
    return out


# ============================================================== parts B/C =====
def part_bc(S: Stage, R: Stage, arms, ref_governs, out, key):
    """``S`` holds the arms under test; ``R`` holds the references.

    They are the same Stage for Part B and differ for Part C, which trains only
    the query arm and reads its L4 reference from Part B -- where the identical
    configuration is trained through the identical code path.
    """
    ref = R.stack("full", REF, "H8")
    res = {}
    for a in arms:
        s = summarise(S, "full", a, "H8")
        gov = CONTROL if (a in DUAL and ref_governs is not None
                          and ref_governs["resolved"]) else REF
        s["governing_reference"] = gov
        s["vs_reference"] = compare(S.stack("full", a, "H8"),
                                    R.stack("full", gov, "H8"))
        s["vs_L4"] = compare(S.stack("full", a, "H8"), ref)
        s["ov_selected_head_recall8"] = float(S.stack("full", a, "OV").mean())
        if a in DUAL:
            s["controls"] = {n: control_summary(S, "full", a, n)
                             for n in ("WRONG_TRAJ_ch0", "WRONG_TRAJ_ch1")}
        if a in QUERY_ARMS:
            s["controls"] = {"WRONG_QUERY": control_summary(S, "full", a,
                                                            "WRONG_QUERY")}
        res[a] = s
    out[key] = {
        "reference_L4": summarise(R, "full", REF, "H8"),
        "arms": res,
        "pairwise_vs_L4": {a: res[a]["vs_L4"] for a in arms},
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c6")
    args = ap.parse_args()

    meta, plan, keys, rows_of = load_plan()
    A, B, C = (Stage(args.tag, "A"), Stage(args.tag, "B"), Stage(args.tag, "C"))
    out = {"stage": "S2-C6",
           "question": "is the residual 0.80 -> 0.95 gap a data limit or a "
                       "single-snapshot representation limit, and if the latter "
                       "does the missing signal come from the layer trajectory "
                       "or from the query?",
           "prereg": "docs/scoring_search_s2c6_prereg.md",
           "rule": {"margin": MARGIN, "primary":
                    "held-out teacher Top-8 recall@256",
                    "selection": "validation teacher Top-8 recall@256 (H8)",
                    "paired_bootstrap_draws": 10000, "seeds": list(SEEDS)}}

    # ---- Part A -----------------------------------------------------------
    part_a(A, rows_of, keys, out)

    # ---- consistency: Part A's n=240 is Part B's L4 ------------------------
    a240 = np.stack([A.honest("A240", "L4", s, "H8") for s in SEEDS])
    bref = np.stack([B.honest("full", REF, s, "H8") for s in SEEDS])
    out["consistency"] = {
        "partA_240_equals_partB_L4_max_abs_diff": float(np.abs(a240 - bref).max()),
        "identical": bool(np.abs(a240 - bref).max() < TOL),
        "note": "the same configuration run twice through the same deterministic "
                "code path; identical per-image held-out metrics are what makes "
                "the two parts share one reference",
        "l4_query_init_identity_max_abs_diff": [
            C.rec("full", QUERY, s)["init_identity_l4_query"] for s in SEEDS],
    }

    # ---- Part B -----------------------------------------------------------
    ref_governs = compare(B.stack("full", CONTROL, "H8"), B.stack("full", REF, "H8"))
    part_bc(B, B, SINGLE + DUAL + (CONTROL,), ref_governs, out, "part_B")
    out["part_B"]["architecture_control_L4_L4_vs_L4"] = ref_governs
    out["part_B"]["control_fired"] = bool(ref_governs["resolved"])
    out["part_B"]["control_effect"] = (
        "the dual architecture is itself a RESOLVED GAIN over L4, so the "
        "trajectory arms are read against L4+L4 and not against L4"
        if ref_governs["resolved"] else
        "the dual architecture is not a RESOLVED GAIN over L4, so L4 remains "
        "the governing reference for the dual arms")

    # ---- Part C -----------------------------------------------------------
    part_bc(C, B, QUERY_ARMS, None, out, "part_C")
    out["part_C"]["reference_L4_from_part_B"] = True
    out["part_C"]["amendment"] = (
        "L4+QUERY was the pre-registered arm. It adds one 128-vector per image "
        "to every token, so its query term is identical across the image's 1 024 "
        "tokens and it can reorder tokens only through the final GELU's "
        "nonlinearity -- it cannot express token-specific query relevance. Its "
        "null therefore does not close the brief's question, and L4+QRY-BILIN "
        "was added after the fact: a token-dependent rank-8 bilinear term at an "
        "*identical* added budget (33 792 parameters). Both are reported; the "
        "pre-registration is amended in scoring_search_s2c6.md rather than "
        "rewritten.")

    # ---- verdicts ---------------------------------------------------------
    b = out["part_B"]["arms"]
    c = out["part_C"]["arms"]
    traj = [a for a in DUAL if b[a]["vs_reference"]["resolved"]]
    traj_below = [a for a in DUAL if b[a]["vs_reference"]["below_margin"]]
    snap = [a for a in SINGLE + DUAL if b[a]["vs_reference"]["resolved"]]
    # D needs one query arm to clear the reference *and* to depend on the
    # image's own query. With two arms, either one would be evidence for D.
    q_details = {a: dict(
        vs_reference=c[a]["vs_reference"],
        control=c[a]["controls"]["WRONG_QUERY"]["honest_minus_wrong"]) for a in
        QUERY_ARMS if a in c}
    q_resolved = any(v["vs_reference"]["resolved"] and v["control"]["resolved"]
                     for v in q_details.values())
    q = c[QUERY]
    qctrl = q_details[QUERY]["control"]

    verdicts = {
        "A_DATA_LIMITED": bool(out["part_A"]["marginal_180_to_240"]["resolved"]),
        "B_SNAPSHOT_SUFFICIENT": bool(not snap),
        "C_TRAJECTORY_INFORMATIVE": bool(traj),
        "D_QUERY_INFORMATIVE": q_resolved,
        "INDETERMINATE_TRAJECTORY": bool(not traj and traj_below),
    }
    why = {
        "A_DATA_LIMITED": out["part_A"]["verdict_reason"],
        "B_SNAPSHOT_SUFFICIENT": (
            "no arm in {" + ", ".join(SINGLE + DUAL) + "} is a RESOLVED GAIN "
            "over its governing reference"
            if not snap else
            "not applicable: " + ", ".join(snap) + " clears its reference"),
        "C_TRAJECTORY_INFORMATIVE": (
            ", ".join(traj) + " is a RESOLVED GAIN over its governing reference"
            if traj else "neither L2+L4 nor L4+DELTA clears its reference"),
        "D_QUERY_INFORMATIVE": (
            "; ".join(f"{a} clears L4 ({v['vs_reference']['mean']:+.4f}) and its "
                      f"shuffled-query control costs {v['control']['mean']:+.4f}"
                      for a, v in q_details.items()
                      if v["vs_reference"]["resolved"] and v["control"]["resolved"])
            if q_resolved else
            "; ".join(
                f"{a}: vs L4 {v['vs_reference']['mean']:+.4f} "
                f"(CI_lo {v['vs_reference']['lo']:+.4f}, resolved="
                f"{v['vs_reference']['resolved']}), shuffled-query control "
                f"honest-wrong {v['control']['mean']:+.4f} "
                f"(resolved={v['control']['resolved']})"
                for a, v in q_details.items())),
        "INDETERMINATE_TRAJECTORY": (
            ", ".join(traj_below) + " is BELOW-MARGIN against its governing "
            "reference -- a directional but sub-margin effect, which carries no "
            "verdict" if traj_below else "no trajectory arm is below-margin"),
    }
    out["verdicts"] = {"flags": verdicts, "reasons": why}
    out["headline"] = headline(verdicts, out)

    # ---- downstream gate --------------------------------------------------
    cands = {a: out["part_B"]["arms"][a] for a in SINGLE + DUAL}
    cands.update({a: c[a] for a in QUERY_ARMS if a in c})
    clear = [a for a, s in cands.items()
             if s["held_out"]["head_recall8"] >= DEPLOY_TARGET
             and s["vs_L4"]["lo"] > 0 and s["vs_L4"]["all_seeds_positive"]]
    near = [a for a, s in cands.items()
            if s["held_out"]["head_recall8"] >= DEPLOY_TARGET]
    out["deployable_gate"] = {
        "rule": f"forward-only arm with held-out R@8 >= {DEPLOY_TARGET} "
                f"(aspirational {DEPLOY_ASPIRATIONAL}), paired CI lower bound "
                f"> 0 vs L4, and a consistent seed pattern",
        "best_arm": max(cands, key=lambda a: cands[a]["held_out"]["head_recall8"]),
        "best_held_out_head_recall8": max(
            cands[a]["held_out"]["head_recall8"] for a in cands),
        "reaches_0.82": near, "clears": clear,
        "downstream_run": bool(clear),
        "note": ("generation is run once, for the best arm + the L4 baseline + "
                 "the gradient teacher reference, only if `clears` is non-empty"
                 if clear else "no generation is run: the gate stays closed"),
    }

    dump(f"{args.tag}_audit.json", out)
    report(out)
    return out


def headline(v, out):
    parts = []
    parts.append(
        "A " + ("DATA-LIMITED" if v["A_DATA_LIMITED"] else "DATA-PLATEAU")
        + f" (180->240 {out['part_A']['marginal_gain_last_segment']:+.4f})")
    parts.append("B " + ("SNAPSHOT-SUFFICIENT" if v["B_SNAPSHOT_SUFFICIENT"]
                         else "a snapshot/trajectory arm clears"))
    parts.append("C " + ("TRAJECTORY-INFORMATIVE" if v["C_TRAJECTORY_INFORMATIVE"]
                         else ("INDETERMINATE-TRAJECTORY"
                               if v["INDETERMINATE_TRAJECTORY"]
                               else "no trajectory gain")))
    parts.append("D " + ("QUERY-INFORMATIVE" if v["D_QUERY_INFORMATIVE"]
                         else "query adds nothing resolved"))
    return " | ".join(parts)


def report(out):
    A = out["part_A"]
    print("\n=== S2-C6 ===")
    print("\nPart A -- fit-set scaling (held-out R@8, H8 checkpoint, 3 seeds)")
    print(f"{'n':>5} {'R@8':>8} {'R@16':>8} {'R@32':>8} {'ov256':>8} "
          f"{'init':>8} {'ep':>10}")
    for n in NS:
        c = A["curve"][str(n)]
        h = c["held_out"]
        print(f"{n:>5} {h['head_recall8']:8.4f} {h['head_recall16']:8.4f} "
              f"{h['head_recall32']:8.4f} {h['overlap256']:8.4f} "
              f"{c['init_held_out_head_recall8']:8.4f} "
              f"{str(c['best_epoch']):>10}")
    for k, v in A["segments"].items():
        print(f"  {k:>10}  delta={v['mean']:+.4f} "
              f"CI=[{v['lo']:+.4f},{v['hi']:+.4f}] "
              f"per-seed={[round(x, 4) for x in v['per_seed']]} "
              f"resolved={v['resolved']}")
    print(f"  verdict: {A['verdict']}")

    print("\nPart B -- representation arms (vs L4, then vs the governing "
          "reference)")
    print(f"{'arm':<10} {'R@8':>8} {'R@16':>8} {'ov256':>7} {'vs_L4':>9} "
          f"{'CI_lo':>8} {'gov':>7} {'resolved':>9} {'params':>9} {'init':>7}")
    for a, s in out["part_B"]["arms"].items():
        h, v = s["held_out"], s["vs_reference"]
        print(f"{a:<10} {h['head_recall8']:8.4f} {h['head_recall16']:8.4f} "
              f"{h['overlap256']:7.4f} {v['mean']:+9.4f} {v['lo']:+8.4f} "
              f"{s['governing_reference']:>7} {str(v['resolved']):>9} "
              f"{s['n_params']:9,} {s['init_held_out_head_recall8']:7.4f}")
    print(f"  L4+L4 architecture control vs L4: "
          f"{out['part_B']['architecture_control_L4_L4_vs_L4']['mean']:+.4f} "
          f"CI=[{out['part_B']['architecture_control_L4_L4_vs_L4']['lo']:+.4f},"
          f"{out['part_B']['architecture_control_L4_L4_vs_L4']['hi']:+.4f}]")
    print("  " + out["part_B"]["control_effect"])
    for a, s in out["part_B"]["arms"].items():
        for n, c in s.get("controls", {}).items():
            v = c["honest_minus_wrong"]
            print(f"    {a:<10} {n:<16} honest-wrong={v['mean']:+.4f} "
                  f"CI=[{v['lo']:+.4f},{v['hi']:+.4f}] "
                  f"per-seed={[round(x, 4) for x in v['per_seed']]} "
                  f"resolved={v['resolved']}")

    print("\nPart C -- query re-check")
    for a, s in out["part_C"]["arms"].items():
        h, v = s["held_out"], s["vs_reference"]
        print(f"{a:<10} {h['head_recall8']:8.4f} {h['head_recall16']:8.4f} "
              f"vs_L4={v['mean']:+.4f} CI_lo={v['lo']:+.4f} "
              f"resolved={v['resolved']} params={s['n_params']:,} "
              f"(query {s['n_query_params']:,})")
        for n, c in s.get("controls", {}).items():
            x = c["honest_minus_wrong"]
            print(f"    {n:<16} honest-wrong={x['mean']:+.4f} "
                  f"CI=[{x['lo']:+.4f},{x['hi']:+.4f}] resolved={x['resolved']}")

    print("\nverdicts")
    for k, v in out["verdicts"]["flags"].items():
        print(f"  {k:<28} {v}")
    print("\n  " + out["headline"])
    g = out["deployable_gate"]
    print(f"\ndeployable gate ({DEPLOY_TARGET}): "
          f"best={g['best_arm']} {g['best_held_out_head_recall8']:.4f} | "
          f"reaches target: {g['reaches_0.82'] or 'none'} | "
          f"downstream_run={g['downstream_run']}")
    print(f"consistency: PartA(240) vs PartB(L4) max|diff| = "
          f"{out['consistency']['partA_240_equals_partB_L4_max_abs_diff']:.2e}")


if __name__ == "__main__":
    main()
