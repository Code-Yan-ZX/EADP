"""
M6 step 2 -- Phase 1, the critical-miss enrichment test.

This is the gate.  It asks one question and answers it without any generation:

    Do tokens where EADP disagrees with the other cheap proxies enrich the set
    the gradient teacher would have kept -- more than RANDOM, more than
    EADP-NEXT, and more than the best single proxy?

Definitions, all inherited from M3-v0 so the numbers are comparable:

    D        = [1024] \\ S0, the 768 tokens B2 dropped        (per instance)
    tr(i)    = teacher rank of i within D, 0 = the teacher's best dropped token
    head_q   = { i in D : tr(i) < q }                        the oracle-critical set
    recall@k = |pick_k ∩ head_q| / q, computed per instance, then averaged

`recall@k` against `head_k` is exactly M3-v0's `overlap@r`, so a number here is
directly comparable to the 0.2365 the M3 student reached and the 0.0208 chance
rate.  Nothing is pooled across instances.

Every rule returns exactly k indices from D.  The only quantity fitted anywhere
in this stage is each proxy's *sign*, fixed on the fit split (240) and never on
the split it is reported on.

Usage
    python scripts/discovery/m6_phase1.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m6_common import (AUX, BUDGET, CONTROL, DS_ORDER, HOLDOUT, INCUMBENT, KS,
                       PRIMARY, RANDOM_SEEDS, Dropped, aggregate, by_benchmark,
                       dump_json, eadp_next_local, fit_orientations, load_bank,
                       load_eapd_order, paired_bootstrap, per_instance, r_disagree,
                       r_maxfusion, r_minfusion, r_oracle_union, r_random,
                       r_random_in_pool, r_single, r_union)            # noqa: E402


def build_rules(bank, dropped, orient, eapd_order, k):
    """Every rule for one k.  Returns {name: picks} with picks[i] local to D_i."""
    n = dropped.n
    rules = {}

    for s in RANDOM_SEEDS:
        rules[f"R0-random-s{s}"] = [r_random(dropped, i, k, s) for i in range(n)]

    rules["R1-EADP-next"] = [eadp_next_local(eapd_order, dropped, i, k)
                             for i in range(n)]

    for nm in PRIMARY:
        rules[f"R2-{nm}"] = [r_single(bank, dropped, orient, i, nm, k)
                             for i in range(n)]

    # union over the fit-ranked proxy members
    order_by_fit = sorted(PRIMARY, key=lambda m: -orient[m]["fit_recall"])
    for m in (2, 3, len(PRIMARY)):
        mem = order_by_fit[:m]
        rules[f"R3-union-top{m}"] = [r_union(bank, dropped, orient, i, k, mem)
                                     for i in range(n)]

    for kind in ("D1", "D2", "D3", "D4"):
        rules[f"R4-{kind}"] = [r_disagree(bank, dropped, orient, i, k, kind)
                               for i in range(n)]
    # D1 restricted to the strongest AUXILIARIES only (the incumbent `imp` is
    # not an auxiliary): if disagreement has value, adding weak proxies should
    # not be what destroys it
    aux_by_fit = sorted(AUX, key=lambda m: -orient[m]["fit_recall"])
    for m in (2, 3):
        mem = tuple(aux_by_fit[:m])
        rules[f"R4-D1-top{m}"] = [r_disagree(bank, dropped, orient, i, k, "D1",
                                             members=mem) for i in range(n)]

    rules["R5-maxfusion"] = [r_maxfusion(bank, dropped, orient, i, k, AUX)
                             for i in range(n)]
    rules["R5-minfusion"] = [r_minfusion(bank, dropped, orient, i, k, AUX)
                             for i in range(n)]
    rules["R6-ORACLE-union"] = [r_oracle_union(bank, dropped, orient, i, k, AUX)
                                for i in range(n)]
    for s in RANDOM_SEEDS[:5]:
        rules[f"R6b-random-in-pool-s{s}"] = [
            r_random_in_pool(bank, dropped, orient, i, k, AUX, s) for i in range(n)]
    return rules, order_by_fit


def main():
    bank = load_bank()
    dropped = Dropped(bank)
    eapd_order = load_eapd_order(bank)
    n = bank["n"]
    print(f"[bank] {n} instances, {bank['split'].count('fit')} fit / "
          f"{bank['split'].count('val')} val / {bank['split'].count('test')} test")

    orient = fit_orientations(bank, dropped)
    print("[orientation, fitted on fit only]")
    for nm, o in orient.items():
        print(f"    {nm:14s} sign={o['sign']:+d}  fit_recall@16={o['fit_recall']:.4f}")

    fit_idx = np.array([i for i in range(n) if bank["split"][i] == "fit"])
    hold_idx = np.array([i for i in range(n) if bank["split"][i] in HOLDOUT])
    test_idx = np.array([i for i in range(n) if bank["split"][i] == "test"])
    panels = {"all450": np.arange(n), "holdout210": hold_idx, "test150": test_idx}

    out = dict(orientation=orient, panels={k: int(v.size) for k, v in panels.items()},
               ks=list(KS), grid={})

    for k in KS:
        q = k
        rules, order_by_fit = build_rules(bank, dropped, orient, eapd_order, k)
        pi = {name: per_instance(p, dropped, q) for name, p in rules.items()}

        # the random control's spread over its 20 seeds
        rnd_names = [f"R0-random-s{s}" for s in RANDOM_SEEDS]
        rnd_rec = np.stack([pi[nm]["recall"] for nm in rnd_names])
        rnd_mean = rnd_rec.mean(0)

        # best single proxy, chosen on FIT
        singles = [f"R2-{nm}" for nm in PRIMARY]
        best_single = max(singles, key=lambda s: pi[s]["recall"][fit_idx].mean())

        refs = {"R0-random": rnd_mean,
                "R1-EADP-next": pi["R1-EADP-next"]["recall"],
                best_single: pi[best_single]["recall"]}

        cell = dict(k=k, q=q, best_single=best_single,
                    order_by_fit=list(order_by_fit),
                    random_seed_spread=dict(
                        lo=float(rnd_rec.mean(1).min()),
                        hi=float(rnd_rec.mean(1).max())),
                    rules={})
        for name in rules:
            e = dict()
            for pname, idx in panels.items():
                a = aggregate(pi[name], idx)
                a["per_benchmark"] = by_benchmark(
                    dict(recall=pi[name]["recall"], precision=pi[name]["precision"],
                         teacher_rank=pi[name]["teacher_rank"],
                         picked=pi[name]["picked"]), bank)
                e[pname] = a
            for rname, rvec in refs.items():
                if name == rname:
                    continue
                b = paired_bootstrap(pi[name]["recall"][hold_idx], rvec[hold_idx])
                e.setdefault("vs", {})[rname] = b
            cell["rules"][name] = e
        out["grid"][str(k)] = cell
        print(f"[k={k}] done  best_single={best_single}  "
              f"random spread [{cell['random_seed_spread']['lo']:.4f}, "
              f"{cell['random_seed_spread']['hi']:.4f}]")

    dump_json("m6_phase1.json", out)
    print("[saved] m6_phase1.json")


if __name__ == "__main__":
    main()
