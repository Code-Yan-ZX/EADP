"""
SAGE step 0 -- the fresh, locked confirmation split (decision 11).

240 rows per benchmark, drawn from outside the frozen bank (the evenly spaced
150), the M1 extension (720) and the S2-A causal cases -- the same protocol as
m1_common.build_extension, under a NEW seed, so the panel has never carried a
single number of any earlier stage or of this one.

Writes sage_plan.json. Nothing downstream re-derives the list.

Usage
    python scripts/discovery/sage_plan.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                        # noqa: E402
from common import OUTPUT_DIR, sample_indices                        # noqa: E402
from sage_common import DS_ORDER, F_PLAN, N_CONF_PER_DS, SAGE_SEED   # noqa: E402

M1_PLAN = "m1_plan.json"


def main():
    plan = json.load(open(os.path.join(OUTPUT_DIR, M1_PLAN)))
    ext_by_ds = {ds: set() for ds in DS_ORDER}
    for p in plan["instances"]:
        ext_by_ds[p["ds"]].add(int(p["idx"]))

    from m1_common import causal_indices
    cases = json.load(open(os.path.join(
        OUTPUT_DIR, "s2a_gradient_viability.json")))["cases"]
    causal = causal_indices(cases)

    out = []
    for i, ds in enumerate(DS_ORDER):
        dataset = common.build_dataset(ds)
        n_total = len(dataset.data)
        used = set(sample_indices(n_total, 150)) | ext_by_ds[ds] | set(causal[ds])
        free = sorted(set(range(n_total)) - used)
        assert len(free) >= N_CONF_PER_DS, f"{ds}: only {len(free)} free rows"
        rng = np.random.default_rng(SAGE_SEED + 303 * (i + 1))
        perm = rng.permutation(len(free))
        for r, j in enumerate(perm[:N_CONF_PER_DS]):
            out.append(dict(key=f"{ds}_{free[j]}", ds=ds, idx=int(free[j]),
                            conf_rank=r, source="sage_confirmation"))
        print(f"[{ds}] n_total {n_total}, used {len(used)}, "
              f"free {len(free)} -> took {N_CONF_PER_DS}")

    keys = [p["key"] for p in out]
    assert len(set(keys)) == len(keys) == N_CONF_PER_DS * len(DS_ORDER)
    dump = dict(seed=SAGE_SEED, n_per_ds=N_CONF_PER_DS, instances=out,
                excludes=dict(bank="evenly spaced 150 (sample_indices)",
                              extension="m1_plan.json",
                              causal="s2a_gradient_viability.json"))
    path = os.path.join(OUTPUT_DIR, F_PLAN)
    with open(path + ".tmp", "w") as f:
        json.dump(dump, f, indent=1)
    os.replace(path + ".tmp", path)
    print(f"[saved] {F_PLAN}: {len(out)} instances "
          f"({ {ds: sum(1 for p in out if p['ds'] == ds) for ds in DS_ORDER} })")


if __name__ == "__main__":
    main()
