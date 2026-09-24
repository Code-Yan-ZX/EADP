"""
S3-B step 0 (CPU): freeze the case bank.

Writes ``s3b_cases.json``: for both bases, the 150 held-out instances with
their S / T / T_only / S_only / C lists (rank-ordered), the frozen per-instance
outcomes the rescue classes are defined on, and the bank-L rescue inventory
recomputed from the S2-C2 FINE grid (no GPU: those arms were already run).

Everything downstream reads this file instead of recomputing sets, so a
re-run cannot silently drift from the frozen protocol.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3b_common as B                                           # noqa: E402
from s1_audit import OUT                                         # noqa: E402


def main():
    oc = B.frozen_outcomes()
    banks = {}
    for bank in ("G", "L"):
        recs = []
        for r in B.load_bank(bank):
            recs.append(dict(key=r["key"], ds=r["ds"], idx=r["idx"],
                             row=r["row"], m=r["m"],
                             S=r["S"], T=r["T"], T_only=r["T_only"],
                             S_only=r["S_only"], C=r["C"],
                             hits=oc[r["key"]]))
        banks[bank] = recs
    # identity check between banks: same teacher set, same instances
    g = {r["key"]: r for r in banks["G"]}
    l = {r["key"]: r for r in banks["L"]}
    assert set(g) == set(l) and all(g[k]["T"] == l[k]["T"] for k in g)
    bundles_L = B.s2c2_bundles()

    out = dict(
        config=dict(budget=B.BUDGET, ks_sweep=list(B.KS_SWEEP),
                    ks_ext=list(B.KS_EXT), ks_max=B.KS_MAX,
                    thr=B.THR, seed_arms=B.SEED_ARMS,
                    n_randwin=B.N_RANDWIN, n_randt=B.N_RANDT,
                    n_rando=B.N_RANDO, shifts=list(B.SHIFTS),
                    n_blockperm=B.N_BLOCKPERM, n_remrand=B.N_REMRAND,
                    bank_G="GDEP LOCAL-MLP n960 s2",
                    bank_L="LIN_L4 (S2-C1 pilot sets)"),
        summary=dict(
            n=len(g),
            m_mean={b: float(np.mean([r["m"] for r in banks[b]]))
                    for b in banks},
            bank_L_bundles=dict(
                n=len(bundles_L),
                n_kstar=sum(1 for v in bundles_L.values()
                            if v["kstar"] is not None),
                n_kstar_le32=sum(1 for v in bundles_L.values()
                                 if v["kstar"] and v["kstar"] <= B.KS_MAX),
                kstars=sorted(v["kstar"] for v in bundles_L.values()
                              if v["kstar"] is not None)),
        ),
        banks=banks, bundles_L=bundles_L)
    path = os.path.join(OUT, B.CASES_JSON)
    tmp = path + ".tmp"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, path)
    print(f"[saved] {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    print("[bank L inventory]", out["summary"]["bank_L_bundles"])
    for k, v in sorted(bundles_L.items()):
        if v["kstar"]:
            print(f"  {k:20s} k*={v['kstar']:3d}  (S2-C2 coarse said "
                  f"{v['first_coarse']})  m={v['m']}")


if __name__ == "__main__":
    main()
