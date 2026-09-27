"""
M9 — Deferred Boundary Adjudication, Phase 0.5: the oracle boundary ceiling.

The brief's stop-or-go gate that runs before any model experiment:

    Boundary = B2 greedy tail-r  ∪  cos_s0c top-32 reserve
    Oracle   = the r boundary members with the highest P1-G2 teacher score
               (tail tokens compete directly -- they are in S0 and the
               teacher score is defined on all 1024 tokens)
    Final    = S_core ∪ (the r oracle picks), |Final| = 256

If a perfect adjudicator confined to this boundary cannot place a rescue at a
teacher rank inside (or near) the band that M3-v0 measured as converting
(3.5-15.5 -> +7.7..+14.9 macro; 73.7-131.9 -> ~0), the restricted formulation
itself has no ceiling and M9 stops here. No model is loaded.

Anchors (M3-v0 / M6 / M8, held-out panels):
    oracle rescue rank 3.5 (r=8)   -> +7.7 macro
    oracle rescue rank 15.5 (r=32) -> +14.9 macro
    learned-student rank 73.7-131.9 -> ~0 (matched by random)
    P1 (cos_s0c top-32) pool oracle at r=8: mean teacher rank 31.3

Everything from cached banks: m6_eapd_order.npz (greedy order), m5_bank.npz
(s0, g2, cos_s0c), on the held-out 210 (val+test) headline and all 450.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_gdep import OUTPUT_DIR  # noqa: E402

RS = 8
POOL = 32
R_LIST = (4, 8, 16)


def load():
    bank = np.load(os.path.join(OUTPUT_DIR, "m5_bank.npz"), allow_pickle=True)
    ordr = np.load(os.path.join(OUTPUT_DIR, "m6_eapd_order.npz"),
                   allow_pickle=True)
    assert (bank["key"] == ordr["key"]).all()
    key = bank["key"]
    ds = bank["ds"]
    split = bank["split"]
    s0 = bank["s0"].astype(np.int64)
    order = ordr["order"].astype(np.int64)
    g2 = bank["g2"].astype(np.float64)
    fi = int(np.where(bank["feature_names"] == "cos_s0c")[0][0])
    cos = bank["X"][:, :, fi]
    # gate: order[:256] sorted == recorded s0 (M7's G-order)
    assert (np.sort(order[:, :256], axis=1) == np.sort(s0, axis=1)).all()
    return key, ds, split, s0, order, g2, cos


def instance_metrics(order_i, g2_i, cos_i, r, pool=POOL):
    n = order_i.shape[0]
    s0 = order_i[:256]
    in_s0 = np.zeros(n, dtype=bool)
    in_s0[s0] = True
    dropped = np.flatnonzero(~in_s0)
    tail = order_i[256 - r:256]
    # reserve: cos_s0c oriented LOW, among dropped
    cos_d = cos_i[dropped]
    reserve = dropped[np.argsort(cos_d, kind="stable")[:pool]]
    boundary = np.concatenate([tail, reserve])
    # teacher rank within D (0 = teacher's best dropped token)
    torder = dropped[np.argsort(-g2_i[dropped], kind="stable")]
    trank = np.full(n, -1, dtype=np.int64)
    trank[torder] = np.arange(torder.shape[0])

    # oracle adjudication: r boundary members with the highest teacher score
    picks = boundary[np.argsort(-g2_i[boundary], kind="stable")[:r]]
    is_res = ~in_s0[picks]
    swapped = picks[is_res]
    k = int(is_res.sum())
    best_res_rank = int(trank[reserve].min())
    # teacher-head recall of the swap (denominator r, comparable to prior
    # rescue tables; only swapped-in reserves can land in the dropped head)
    head = torder[:r]
    recall = float(np.isin(swapped, head).sum()) / r if k else 0.0
    dmass = float(g2_i[picks].sum() - g2_i[tail].sum())
    return dict(k_swap=k, best_res_rank=best_res_rank,
                swap_ranks=[int(t) for t in trank[swapped]],
                recall=recall, dmass=dmass)


def summarize(rows, r):
    ks = np.array([x["k_swap"] for x in rows])
    ranks = [t for x in rows for t in x["swap_ranks"]]
    br = np.array([x["best_res_rank"] for x in rows])
    recall = np.array([x["recall"] for x in rows])
    dm = np.array([x["dmass"] for x in rows])
    swapped = [x for x in rows if x["k_swap"] > 0]
    all_swapped_ranks = np.array([t for x in swapped for t in x["swap_ranks"]]
                                 or [np.nan])
    return dict(
        r=r, n=len(rows),
        frac_any_swap=float((ks > 0).mean()),
        frac_full_swap=float((ks == r).mean()),
        mean_k_swap=float(ks.mean()),
        mean_best_res_rank=float(br.mean()),
        median_best_res_rank=float(np.median(br)),
        p90_best_res_rank=float(np.percentile(br, 90)),
        mean_swap_rank_all=float(np.nanmean(all_swapped_ranks))
        if len(all_swapped_ranks) else None,
        mean_swap_rank_when_swapped=float(np.mean(ranks)) if ranks else None,
        median_swap_rank_when_swapped=float(np.median(ranks)) if ranks else None,
        head_recall_at_r=float(recall.mean()),
        mean_teacher_mass_gain=float(dm.mean()),
    )


def main():
    key, ds, split, s0, order, g2, cos = load()
    panels = {"heldout210": split != "fit", "all450": np.ones(len(key), bool)}
    out = {"config": dict(pool=POOL, r_list=list(R_LIST),
                          reserve_rule="cos_s0c LOW top-32 among dropped"),
           "panels": {}}
    for pname, sel in panels.items():
        idx = np.flatnonzero(sel)
        for r in R_LIST:
            rows = [instance_metrics(order[i], g2[i], cos[i], r)
                    for i in idx]
            out["panels"].setdefault(pname, {})[f"r{r}"] = summarize(rows, r)
            print(f"[{pname}] r={r}: " + json.dumps(
                out["panels"][pname][f"r{r}"], indent=None))
    path = os.path.join(OUTPUT_DIR, "m9_oracle_boundary.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
