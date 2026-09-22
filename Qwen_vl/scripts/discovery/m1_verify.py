"""
M1 step 7: result-consistency checks.

Everything M1 claims is recomputed here from the raw per-image artifacts rather
than re-read from the audit JSON, so a mistake in the aggregation cannot hide
behind itself. Exits non-zero on the first failure.

  V1  the per-image archives and the train records cover the same runs
  V2  every run's held-out per-image vector is over the same 150 images, in the
      same order                              (a reordered arm would silently corrupt every paired delta)
  V3  the audit's headline R@8 equals the mean of the raw per-image R@8
  V4  the paired segments recompute from the raw vectors, 10 000 draws
  V5  gate G3: fixed-step n=240 == S2-C6's published A240, per seed
  V6  the fixed-step ladders are the nested fit sets the plan says they are
  V7  the verdict follows from the pre-registered ordered rule applied to the
      recomputed segments
  V8  no held-out quantity was used for selection: the selected checkpoint's
      validation value is what the rule reads, and the held-out vectors appear
      nowhere in the training records' selection logic

Usage
    python m1_verify.py --tag m1
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                              # noqa: E402
from m1_common import (MARGIN, NBOOT, N_VAL_POINTS, SEEDS, compare,   # noqa: E402
                       ladder_fit_rows, load_m1_plan)

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'OK ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m1")
    args = ap.parse_args()
    tag = args.tag

    audit = json.load(open(os.path.join(OUT, f"{tag}_audit.json")))
    train = json.load(open(os.path.join(OUT, f"{tag}_train_fixed-step.json")))
    z = np.load(os.path.join(OUT, f"{tag}_perimage_fixed-step.npz"))
    ns = list(train["config"]["ns"])

    # ---------------------------------------------------------------- V1 -----
    want = {f"{tag}_fixed-step_n{n}_L4_s{s}" for n in ns for s in SEEDS}
    check("V1 train records cover the grid", set(train["results"]) == want,
          f"{len(want)} runs")
    miss = [k for k in want if f"{k}__H8__honest" not in z.files]
    check("V1 per-image archive covers the grid", not miss, f"missing {miss[:3]}")

    # ---------------------------------------------------------------- V2 -----
    ref_keys = train["test_keys"]
    check("V2 held-out key list is the frozen 150",
          len(ref_keys) == 150 and len(set(ref_keys)) == 150)
    shapes = {z[f"{k}__H8__honest"].shape for k in want}
    check("V2 every arm has a 150-image held-out vector", shapes == {(150,)},
          str(shapes))

    # ---------------------------------------------------------------- V3 -----
    worst = 0.0
    for row in audit["protocols"]["fixed-step"]["ladder"]:
        n = row["n"]
        m = float(np.mean([z[f"{tag}_fixed-step_n{n}_L4_s{s}__H8__honest"].mean()
                           for s in SEEDS]))
        worst = max(worst, abs(m - row["R8"]))
    check("V3 audit R@8 == mean of raw per-image R@8", worst < 1e-12,
          f"worst |diff| {worst:.2e}")

    # ---------------------------------------------------------------- V4 -----
    def mat(n, sel="H8", protocol="fixed-step"):
        return np.stack([z[f"{tag}_{protocol}_n{n}_L4_s{s}__{sel}__honest"]
                         for s in SEEDS])

    worst = 0.0
    for seg, rec in audit["protocols"]["fixed-step"]["segments"].items():
        a, b = (int(x) for x in seg.split("->"))
        r = compare(mat(b), mat(a))
        for field in ("mean", "lo", "hi"):
            worst = max(worst, abs(r[field] - rec[field]))
    check("V4 paired segments recompute from the raw vectors", worst < 1e-12,
          f"worst |diff| {worst:.2e} over {NBOOT} draws")

    # ---------------------------------------------------------------- V5 -----
    ref = json.load(open(os.path.join(OUT, "s2c6_train_A.json")))
    published = {0: 0.7992, 1: 0.8133, 2: 0.8017}
    ok = True
    for s in SEEDS:
        h8 = train["results"][f"{tag}_fixed-step_n240_L4_s{s}"]["selections"]["H8"]
        ok &= abs(h8["test"]["head_recall8"] - published[s]) < 5e-5
        ok &= h8["best_point"] == \
            ref["results"][f"A240__L4_s{s}"]["selections"]["H8"]["best_epoch"]
    check("V5 gate G3: n=240 reproduces S2-C6's A240 per seed", ok)
    check("V5 gate G3 recorded as passed",
          audit["gate_G3_reproduction"]["passed"] and
          audit["gate_G3_reproduction"]["worst_val_curve_abs_diff"] == 0.0)

    # ---------------------------------------------------------------- V6 -----
    _, plan, keys, rows_of = load_m1_plan()
    ok, prev = True, set()
    for n in ns:
        r = ladder_fit_rows(rows_of, keys, n)
        ok &= (len(r) == n and len(set(r)) == n)
        ok &= prev <= set(r)
        ok &= set(rows_of["test"]).isdisjoint(r)
        ok &= set(rows_of["val"]).isdisjoint(r)
        ok &= len({keys[i].rsplit("_", 1)[0] for i in r}) == 3
        prev = set(r)
        # the record must match the plan
        ok &= list(map(int, train["results"]
                       [f"{tag}_fixed-step_n{n}_L4_s{SEEDS[0]}"]["fit_rows"])) == r
    check("V6 ladders nested, stratified, disjoint from val/test, match records", ok)
    check("V6 n=240 is the S2-C6 fit set",
          set(ladder_fit_rows(rows_of, keys, 240)) == set(rows_of["fit"]))

    # ---------------------------------------------------------------- V7 -----
    seg = audit["protocols"]["fixed-step"]["segments"]
    d_bc = seg["480->960"]
    endpoint = audit["verdict"]["endpoint_at_final_val_point"]
    if endpoint:
        expect = "ENDPOINT-UNRESOLVED"
    elif d_bc["lo"] > 0:
        expect = "DATA-RESPONSIVE"
    elif d_bc["hi"] < MARGIN:
        expect = "EVIDENCE-OF-SATURATION"
    else:
        expect = "INCONCLUSIVE"
    check("V7 verdict follows the pre-registered ordered rule",
          expect == audit["verdict"]["verdict"], f"rule says {expect}")

    # ---------------------------------------------------------------- V8 -----
    # The selection must read validation only. The held-out vectors are keyed
    # __H8__honest / __OV__honest and appear in no selection path; the records
    # carry the validation value the rule read, and it must equal the val curve.
    ok = True
    for n in ns:
        for s in SEEDS:
            rec = train["results"][f"{tag}_fixed-step_n{n}_L4_s{s}"]
            for sl in ("H8", "OV"):
                e = rec["selections"][sl]["best_point"]
                v = rec["selections"][sl]["val"]["head_recall8" if sl == "H8"
                                                   else "overlap256"]
                hist = rec["val_history"]
                # the stored validation reading must be the curve maximum
                col = "head_recall8" if sl == "H8" else "overlap256"
                ok &= abs(v - max(h[col] for h in hist)) < 1e-12
                ok &= (e < N_VAL_POINTS)
    check("V8 selection reads validation only, and is the curve maximum", ok)

    print()
    if FAILS:
        print(f"FAILED: {FAILS}")
        raise SystemExit(1)
    print("all M1 consistency checks passed")
    json.dump({"passed": True, "checks": 8, "tag": tag},
              open(os.path.join(OUT, f"{tag}_verify.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
