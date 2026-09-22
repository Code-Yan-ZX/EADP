"""
M1 step 0: build and freeze the extension instance list, then audit it.

Writes ``m1_plan.json`` -- the 720 extension instances, benchmark-major in draw
order, each with the benchmark, the dataset row index and its draw rank. Nothing
downstream re-derives this list: ``m1_common.load_m1_plan`` reads it back, so the
row space every M1 arm sees is the one frozen here.

Read-only with respect to the existing results. Run this before
``m1_features.py`` and ``m1_teacher.py``.

Checks printed here (prereg §3.3, gate G1a):

  * every extension row is outside its benchmark's frozen bank (so val and test
    cannot move) and outside its S2-A causal cases;
  * the extension rows are distinct and benchmark-stratified 240 / 240 / 240;
  * ``c6_fit_rows`` here reproduces ``s2c6_common.nested_fit_rows`` exactly for
    every S2-C6 n -- an independent implementation of the published definition;
  * the M1 ladder is nested, S(60) c S(120) c S(180) c S(240) c S(480) c S(960),
    and equals the S2-C6 fit set at n = 240.

Usage
    python m1_plan.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import sample_indices                                     # noqa: E402
from s1_audit import OUT                                              # noqa: E402
from s2c6_common import nested_fit_rows as c6_nested                  # noqa: E402
from s2c3_common import load_plan                                     # noqa: E402
from m1_common import (ALL_NS, C6_NS, DS_ORDER, EXTRASET_SEED,        # noqa: E402
                       M1_PLAN, N_EXTRA_PER_DS, build_extension,
                       c6_fit_rows, causal_indices, dump, ladder_fit_rows,
                       load_m1_plan)


def main():
    # n_total per benchmark comes from the frozen bank itself: the bank is an
    # evenly spaced 150-sample of range(n_total), so its last index is
    # round((n_total-1) * 149/150) and n_total is recovered by search.
    meta = json.load(open(os.path.join(OUT, "s2c1_features.json")))
    n_total = {}
    for ds in DS_ORDER:
        top = max(p["idx"] for p in meta["plan"] if p["ds"] == ds)
        for cand in range(top, top + 200):
            if max(sample_indices(cand, 150)) == top and \
                    len(sample_indices(cand, 150)) == 150:
                n_total[ds] = cand
                break
        assert ds in n_total, f"{ds}: could not recover n_total"
        assert sample_indices(n_total[ds], 150) == sorted(
            p["idx"] for p in meta["plan"] if p["ds"] == ds
            and p["split"] in ("fit", "val", "test")), f"{ds}: bank mismatch"
    print("[sizes]", {k: v for k, v in n_total.items()})

    class _DS:
        def __init__(self, n):
            self.data = list(range(n))

    datasets = {ds: _DS(n_total[ds]) for ds in DS_ORDER}
    instances = build_extension(datasets)

    # --- G1a: the draw never touches the frozen bank or the causal cases -----
    cases = json.load(open(os.path.join(OUT, "s2a_gradient_viability.json")))["cases"]
    causal = causal_indices(cases)
    per_ds = {}
    for p in instances:
        ds = p["ds"]
        bank = set(sample_indices(n_total[ds], 150))
        assert p["idx"] not in bank, f"{p['key']} re-draws a frozen bank row"
        assert p["idx"] not in causal[ds], f"{p['key']} re-draws a causal row"
        per_ds.setdefault(ds, []).append(p["idx"])
    for ds in DS_ORDER:
        assert len(per_ds[ds]) == N_EXTRA_PER_DS, f"{ds}: {len(per_ds[ds])}"
        assert len(set(per_ds[ds])) == N_EXTRA_PER_DS, f"{ds}: duplicate rows"
        print(f"[draw] {ds}: {len(per_ds[ds])} rows, "
              f"{len(set(per_ds[ds]))} distinct, "
              f"free pool {n_total[ds] - 150 - len(causal[ds])}")

    dump(M1_PLAN, dict(extraset_seed=EXTRASET_SEED,
                       n_extra_per_ds=N_EXTRA_PER_DS,
                       n_total=n_total,
                       ds_order=DS_ORDER,
                       instances=instances))

    # --- G1a cont.: the ladders, cross-checked against S2-C6 ------------------
    _, plan, keys, rows_of = load_plan()
    for n in C6_NS:
        a = c6_fit_rows(rows_of, keys, n)
        b = c6_nested(rows_of, keys, n)
        assert a == b, f"n={n}: independent C6 subset implementations disagree"
    print(f"[gate G1a] c6_fit_rows == s2c6_common.nested_fit_rows for n in {C6_NS}")

    # the M1 plan appends the extension rows, so load the combined view and
    # re-check that the original-465 prefix is untouched
    _, cplan, ckeys, crows = load_m1_plan()
    assert ckeys[:465] == keys, "combined plan perturbed the original row order"
    assert crows["fit"] == rows_of["fit"]
    assert crows["val"] == rows_of["val"]
    assert crows["test"] == rows_of["test"]

    seen = set()
    for n in ALL_NS:
        r = ladder_fit_rows(crows, ckeys, n)
        assert len(r) == n, f"n={n}: {len(r)} rows"
        assert len(set(r)) == n, f"n={n}: duplicate rows"
        assert seen <= set(r), f"n={n}: ladder is not nested"
        seen = set(r)
        nbench = {ds: sum(1 for i in r if ckeys[i].startswith(ds + "_"))
                  for ds in DS_ORDER}
        assert set(nbench.values()) == {n // 3}, f"n={n}: not stratified {nbench}"
        print(f"[ladder] n={n:4d}  {nbench}  held-out={len(crows['test'])} "
              f"val={len(crows['val'])}")
    assert set(ladder_fit_rows(crows, ckeys, 240)) == set(crows["fit"]), \
        "n=240 is not the S2-C6 fit set"
    print("[gate G1a] ladder nested, benchmark-stratified, n=240 == S2-C6 fit set")


if __name__ == "__main__":
    main()
