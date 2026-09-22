"""
M1 step 4: apply the pre-registered rule and write the stage artifact.

Reads only what ``m1_train.py`` wrote plus the published S2-C6 records. Computes
no new model, trains nothing, selects nothing.

What it produces
    gate G3   the fixed-step n=240 rung against S2-C6's A240 arm, epoch by epoch
    Q1        the 60 -> 240 rise decomposed against a fixed update budget
    Q2, Q3    the primary fixed-step ladder 240 -> 480 -> 960 and its segments
    Q4        per-seed and per-benchmark consistency
    Q5        the train/validation gap across the ladder
    Q6        the S2-C6 downstream gate, re-applied unchanged
    verdict   the ordered prereg 5.3 conclusion, first match wins

Usage
    python m1_consolidate.py --tag m1
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                              # noqa: E402
from m1_common import (ALL_NS, COL, DS_ORDER, M1_NS, MARGIN, NBOOT,    # noqa: E402
                       N_VAL_POINTS, SEEDS, compare, dump)

SPLIT_ORDER = DS_ORDER                       # held-out is grouped in this order


def environment():
    """What the numbers were produced on, recorded rather than assumed."""
    import platform
    import subprocess
    import torch

    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  check=False).stdout.strip()
        except Exception as e:                                    # pragma: no cover
            return f"unavailable: {e}"

    return dict(
        python=platform.python_version(), torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        deterministic_algorithms="torch.use_deterministic_algorithms(True)",
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        host=platform.node(),
        git_commit=sh(["git", "-C", os.path.dirname(OUT), "rev-parse", "HEAD"]),
        git_branch=sh(["git", "-C", os.path.dirname(OUT), "rev-parse",
                       "--abbrev-ref", "HEAD"]),
        libc=platform.libc_ver()[0],
    )


# ------------------------------------------------------------------ helpers --
def load(tag, protocol):
    p = os.path.join(OUT, f"{tag}_train_{protocol}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    z = np.load(os.path.join(OUT, f"{tag}_perimage_{protocol}.npz"))
    return d, z


def per_image(z, protocol, n, seed, sel="H8", block="honest"):
    k = f"m1_{protocol}_n{n}_L4_s{seed}__{sel}__{block}"
    if k not in z.files:
        raise KeyError(f"{k} missing (have {len(z.files)} entries)")
    return z[k]


def matrix(z, protocol, n, sel="H8"):
    """(n_seeds, 150) per-image held-out head_recall@8, seeds in SEEDS order."""
    return np.stack([per_image(z, protocol, n, s, sel) for s in SEEDS])


def segment(z, protocol, n1, n2, sel="H8"):
    return compare(matrix(z, protocol, n2, sel), matrix(z, protocol, n1, sel))


def bench_slice(i):
    """Held-out block for benchmark i (50 images each, in plan order)."""
    return slice(i * 50, (i + 1) * 50)


def sel_of(d, protocol, n, seed, sel):
    return d["results"][f"m1_{protocol}_n{n}_L4_s{seed}"]["selections"][sel]


# ------------------------------------------------------- gate G3: n=240 ------
def gate_g3(d, z, tag):
    """The fixed-step n=240 rung must reproduce S2-C6's A240 arm exactly.

    Prereg 3.3: one n=240 epoch is 30 updates, which is exactly one M1 validation
    interval and exactly one reshuffled queue, so M1's validation point k *is*
    S2-C6's epoch k. Anything other than agreement means the fixed-step protocol
    is not a controlled re-parameterisation of the published run, and the
    data/step decomposition would be uninterpretable.
    """
    ref = json.load(open(os.path.join(OUT, "s2c6_train_A.json")))
    rep = {"per_seed": [], "epochs_compared": 0}
    worst = 0.0
    for seed in SEEDS:
        r = ref["results"][f"A240__L4_s{seed}"]
        m = d["results"][f"m1_fixed-step_n240_L4_s{seed}"]
        c6h, m1h = r["val_history"], m["val_history"]
        k = min(len(c6h), len(m1h))
        for i in range(k):
            for col in ("head_recall8", "head_recall16", "overlap256"):
                worst = max(worst, abs(float(c6h[i][col]) - float(m1h[i][col])))
        e_ref = r["selections"]["H8"]["best_epoch"]
        e_m1 = m["selections"]["H8"]["best_point"]
        h_ref = r["selections"]["H8"]["test"]["head_recall8"]
        h_m1 = m["selections"]["H8"]["test"]["head_recall8"]
        pub = {0: 0.7992, 1: 0.8133, 2: 0.8017}[seed]
        rep["per_seed"].append(dict(
            seed=seed, epochs_compared=k, c6_epochs_run=r["epochs_run"],
            m1_val_points=len(m1h),
            h8_epoch_c6=e_ref, h8_epoch_m1=e_m1, epoch_match=bool(e_ref == e_m1),
            c6_test_h8=h_ref, m1_test_h8=h_m1,
            published_test_h8=pub,
            matches_published=bool(abs(h_m1 - pub) < 5e-5)))
        rep["epochs_compared"] = max(rep["epochs_compared"], k)
    rep["worst_val_curve_abs_diff"] = worst
    rep["passed"] = bool(worst == 0.0 and
                         all(r["epoch_match"] and r["matches_published"]
                             for r in rep["per_seed"]))
    return rep


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m1")
    args = ap.parse_args()
    tag = args.tag

    prim = load(tag, "fixed-step")
    sec = load(tag, "fixed-epoch")
    assert prim is not None, "run m1_train.py --protocol fixed-step first"
    d, z = prim

    audit = {"stage": "M1", "arm": "L4 (LOCAL-MLP)", "tag": tag,
             "protocols": {}, "margin": MARGIN, "nboot": NBOOT,
             "n_boot_draws": NBOOT, "seeds": list(SEEDS),
             "held_out_n": 150, "val_n": 60,
             "environment": environment(),
             "prereg": "docs/scoring_search_m1_prereg.md",
             "source_data": dict(
                 original_cache="s2c1_feats_L4.npy (465,1024,4096) fp16, untouched",
                 extension_cache="m1_feats_L4_extra.npy (720,1024,4096) fp16",
                 teacher="s2b_gradient_scores.npz (450) + "
                         "m1_gradient_scores_extra.npz (720)",
                 plan="m1_plan.json (extraset_seed 20260924)")}

    # ------------------------------------------------------------ gate G3 ----
    audit["gate_G3_reproduction"] = gate_g3(d, z, tag)
    print(f"[gate G3] passed={audit['gate_G3_reproduction']['passed']} "
          f"worst val-curve |diff|={audit['gate_G3_reproduction']['worst_val_curve_abs_diff']:.1e}")

    # ----------------------------------------------------- primary ladder ----
    nl = list(d["config"]["ns"])
    audit["protocols"]["fixed-step"] = dict(
        config=d["config"], ns=nl, runs={}, ladder=[], segments={})

    for n in nl:
        row = []
        for seed in SEEDS:
            h8 = d["results"][f"m1_fixed-step_n{n}_L4_s{seed}"]["selections"]["H8"]
            ov = d["results"][f"m1_fixed-step_n{n}_L4_s{seed}"]["selections"]["OV"]
            row.append(dict(seed=seed, best_point=h8["best_point"],
                            best_step=h8["best_step"],
                            n_val_points=h8["n_val_points"],
                            val_h8=h8["val"][COL],
                            test_h8=h8["test"]["head_recall8"],
                            test_h16=h8["test"]["head_recall16"],
                            test_h32=h8["test"]["head_recall32"],
                            test_ov=h8["test"]["overlap256"],
                            train_h8=h8["train"]["head_recall8"],
                            ov_point=ov["best_point"],
                            ov_test_h8=ov["test"]["head_recall8"]))
        m = np.mean([r["test_h8"] for r in row])
        audit["protocols"]["fixed-step"]["runs"][str(n)] = row
        audit["protocols"]["fixed-step"]["ladder"].append(dict(
            n=n, R8=float(m),
            R16=float(np.mean([r["test_h16"] for r in row])),
            R32=float(np.mean([r["test_h32"] for r in row])),
            ov256=float(np.mean([r["test_ov"] for r in row])),
            per_seed=[r["test_h8"] for r in row],
            per_seed_best_point=[r["best_point"] for r in row],
            per_seed_best_step=[r["best_step"] for r in row],
            ov_selected_R8=float(np.mean([r["ov_test_h8"] for r in row])),
            train_R8=float(np.mean([r["train_h8"] for r in row])),
            val_R8=float(np.mean([r["val_h8"] for r in row]))))
        print(f"[fixed-step n={n:4d}] R@8={m:.4f} "
              f"per-seed={[round(r['test_h8'], 4) for r in row]} "
              f"sel-point={[r['best_point'] for r in row]}")

    for a, b in zip(nl, nl[1:]):
        s = segment(z, "fixed-step", a, b)
        audit["protocols"]["fixed-step"]["segments"][f"{a}->{b}"] = s
        print(f"[segment {a}->{b}] {s['mean']:+.4f} "
              f"[{s['lo']:+.4f}, {s['hi']:+.4f}] {s['label']} "
              f"per-seed={[round(x, 4) for x in s['per_seed']]}")

    # per-benchmark blocks (descriptive: 50 images each)
    pb = {}
    for a, b in zip(nl, nl[1:]):
        pb[f"{a}->{b}"] = {}
        for i, ds in enumerate(SPLIT_ORDER):
            sl = bench_slice(i)
            A, B = matrix(z, "fixed-step", b), matrix(z, "fixed-step", a)
            r = compare(A[:, sl], B[:, sl])
            pb[f"{a}->{b}"][ds] = r
            print(f"  [bench {a}->{b} {ds}] {r['mean']:+.4f} "
                  f"[{r['lo']:+.4f}, {r['hi']:+.4f}] {r['label']}")
    audit["protocols"]["fixed-step"]["per_benchmark"] = pb

    # --------------------------------------------------- secondary ladder ----
    if sec is not None:
        d2, z2 = sec
        audit["protocols"]["fixed-epoch"] = dict(
            config=d2["config"], ns=list(d2["config"]["ns"]), ladder=[],
            segments={}, runs={})
        for n in d2["config"]["ns"]:
            row = []
            for seed in SEEDS:
                h8 = d2["results"][f"m1_fixed-epoch_n{n}_L4_s{seed}"]["selections"]["H8"]
                r0 = d2["results"][f"m1_fixed-epoch_n{n}_L4_s{seed}"]
                row.append(dict(seed=seed, best_epoch=h8["best_point"],
                                epochs_run=r0["epochs_run"],
                                steps_run=r0["steps_run"],
                                val_h8=h8["val"][COL],
                                test_h8=h8["test"]["head_recall8"],
                                test_h16=h8["test"]["head_recall16"],
                                test_h32=h8["test"]["head_recall32"],
                                test_ov=h8["test"]["overlap256"],
                                train_h8=h8["train"]["head_recall8"],
                                params_at_selection_steps=h8["best_step"]))
            m = np.mean([r["test_h8"] for r in row])
            audit["protocols"]["fixed-epoch"]["runs"][str(n)] = row
            audit["protocols"]["fixed-epoch"]["ladder"].append(dict(
                n=n, R8=float(m),
                R16=float(np.mean([r["test_h16"] for r in row])),
                R32=float(np.mean([r["test_h32"] for r in row])),
                ov256=float(np.mean([r["test_ov"] for r in row])),
                per_seed=[r["test_h8"] for r in row],
                per_seed_best_epoch=[r["best_epoch"] for r in row],
                per_seed_epochs_run=[r["epochs_run"] for r in row],
                per_seed_steps_run=[r["steps_run"] for r in row],
                per_seed_steps_at_selection=[r["params_at_selection_steps"]
                                             for r in row],
                train_R8=float(np.mean([r["train_h8"] for r in row])),
                val_R8=float(np.mean([r["val_h8"] for r in row]))))
            print(f"[fixed-epoch n={n:4d}] R@8={m:.4f} "
                  f"per-seed={[round(r['test_h8'], 4) for r in row]} "
                  f"sel-epoch={[r['best_epoch'] for r in row]} "
                  f"steps={[r['steps_run'] for r in row]}")
        ns2 = list(d2["config"]["ns"])
        for a, b in zip(ns2, ns2[1:]):
            s = segment(z2, "fixed-epoch", a, b)
            audit["protocols"]["fixed-epoch"]["segments"][f"{a}->{b}"] = s
            print(f"[segment {a}->{b}] {s['mean']:+.4f} "
                  f"[{s['lo']:+.4f}, {s['hi']:+.4f}] {s['label']}")

    # ------------------------------------------------------------- Q1 --------
    # C6's published curve, and the same contrast under a fixed update budget.
    c6_curve = {60: 0.7644, 120: 0.7853, 180: 0.7922, 240: 0.8047}
    fs_curve = {x["n"]: x["R8"]
                for x in audit["protocols"]["fixed-step"]["ladder"]}
    q1 = {"c6_published_fixed_epoch_curve": c6_curve,
          "m1_fixed_step_curve": {str(k): v for k, v in fs_curve.items()},
          "c6_60_to_240": c6_curve[240] - c6_curve[60],
          "fixed_step_60_to_240":
              fs_curve.get(240, float("nan")) - fs_curve.get(60, float("nan")),
          "note": "the published 60 -> 240 rise is the fixed-epoch number; the "
                  "fixed-step number holds the update count constant, so the "
                  "difference between the two is the part of the rise that the "
                  "extra optimizer updates can account for"}
    if 60 in fs_curve and 240 in fs_curve:
        q1["step_attributable"] = float(q1["c6_60_to_240"]
                                        - q1["fixed_step_60_to_240"])
        q1["step_attributable_fraction"] = float(
            q1["step_attributable"] / q1["c6_60_to_240"])
        q1["fixed_step_segment_60_240"] = segment(z, "fixed-step", 60, 240)
    audit["Q1_step_decomposition"] = q1
    print(f"[Q1] C6 curve 60->240 = {q1['c6_60_to_240']:+.4f}; "
          f"fixed-step 60->240 = {q1.get('fixed_step_60_to_240', float('nan')):+.4f}")

    # -------------------------------------------------------- verdict --------
    v = {"rule": "prereg 5.3, ordered, first match wins"}
    seg = audit["protocols"]["fixed-step"]["segments"]
    d_ab = seg.get("240->480")
    d_bc = seg.get("480->960")
    finite = lambda s: (s is not None and np.isfinite(s["mean"]))

    # 1. endpoint check over the whole primary grid
    endpoint = []
    for n in nl:
        for r in audit["protocols"]["fixed-step"]["runs"][str(n)]:
            if r["best_point"] >= N_VAL_POINTS - 1:
                endpoint.append(dict(n=n, seed=r["seed"],
                                     best_point=r["best_point"],
                                     n_val_points=r["n_val_points"]))
    v["endpoint_at_final_val_point"] = endpoint

    if endpoint:
        v["verdict"] = "ENDPOINT-UNRESOLVED"
        v["why"] = ("at least one primary run selects its H8 checkpoint at the "
                    "final validation point, so the fixed budget rather than the "
                    "fit pool is the binding constraint")
    elif finite(d_bc) and d_bc["lo"] > 0:
        v["verdict"] = "DATA-RESPONSIVE"
        v["why"] = (f"the 480 -> 960 paired CI lower bound is {d_bc['lo']:+.4f} "
                    f"> 0: doubling the fit pool still buys a measurable "
                    f"held-out gain at a fixed update budget")
    elif finite(d_bc) and d_bc["hi"] < MARGIN:
        v["verdict"] = "EVIDENCE-OF-SATURATION"
        v["why"] = (f"the 480 -> 960 paired CI upper bound is {d_bc['hi']:+.4f} "
                    f"< MARGIN = {MARGIN}: the data exclude a >= 1 pt gain from "
                    f"doubling the fit pool at a fixed update budget")
    else:
        v["verdict"] = "INCONCLUSIVE"
        v["why"] = ("the 480 -> 960 interval spans both zero and MARGIN, so the "
                    "primary contrast is unresolved; that is not evidence of a "
                    "plateau and is not reported as one")
    v["segment_240_480"] = d_ab
    v["segment_480_960"] = d_bc
    if 240 in fs_curve and 960 in fs_curve:
        v["segment_240_960"] = segment(z, "fixed-step", 240, 960)
        v["largest_vs_240"] = v["segment_240_960"]
    audit["verdict"] = v
    print(f"[verdict] {v['verdict']}  ({v['why']})")

    # ------------------------------------------------------------- Q5 --------
    q5 = []
    for x in audit["protocols"]["fixed-step"]["ladder"]:
        q5.append(dict(n=x["n"], train_R8=x["train_R8"], val_R8=x["val_R8"],
                       test_R8=x["R8"],
                       train_minus_val=x["train_R8"] - x["val_R8"],
                       train_minus_test=x["train_R8"] - x["R8"]))
    audit["Q5_train_val_gap"] = q5
    for r in q5:
        print(f"[Q5 n={r['n']:4d}] train={r['train_R8']:.4f} val={r['val_R8']:.4f} "
              f"test={r['test_R8']:.4f} gap={r['train_minus_test']:+.4f}")

    # ------------------------------------------------------------- Q6 --------
    best = max(audit["protocols"]["fixed-step"]["ladder"], key=lambda x: x["R8"])
    gate = dict(
        best_n=best["n"], best_R8=best["R8"],
        best_R8_ci_lower=(v.get("segment_240_960", {}) or {}).get("lo"),
        threshold=0.82, aspirational=0.85,
        opens=bool(best["R8"] >= 0.82))
    if best["n"] != 240:
        s = segment(z, "fixed-step", 240, best["n"])
        gate["vs_L4_reference"] = s
        gate["opens"] = bool(s["lo"] > 0 and best["R8"] >= 0.82)
    audit["Q6_downstream_gate"] = gate
    print(f"[Q6] best n={best['n']} R@8={best['R8']:.4f} gate_opens={gate['opens']}")

    audit["config"] = d["config"]
    dump(f"{tag}_audit.json", audit)
    return audit


if __name__ == "__main__":
    main()
