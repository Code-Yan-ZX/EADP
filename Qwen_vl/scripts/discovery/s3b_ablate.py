"""
S3-B step 3 (GPU): Part B -- is the bundle actually indivisible?

For every minimal rescue group G = T_only[:k*] with k* <= 32 (both banks) the
frozen arm list of docs/s3b_prereg_rules.md section 4 is measured, official
accuracy and gold NLL together:

    full        F(k*)                                    (re-measured: a
                                                          cross-process
                                                          determinism check)
    single_j    only member j is added                   sufficiency
    loo_j       every member except j                    necessity
    randwin_r   k* tokens drawn from the same teacher-rank window, other
                members of which are NOT in G            the key null
    randT_r     k* tokens from the whole missed pool
    randO_r     k* tokens from outside S union T         "any extra pixels"
    shift_a     the same-size window slid down the ordering
    blockperm_r the window kept, the tail members swapped for the next ranks
    remrand_r   G kept, but k* RANDOM student tokens dropped (removal control)

Every arm except remrand uses the identical F(k*) removal set, so bundle vs
null is instance-, size- and removal-matched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                                    # noqa: E402
from s1_audit import OUT                                         # noqa: E402
import s3b_common as B                                           # noqa: E402
import s3b_gpu as G                                              # noqa: E402


def bundles_of(sweep, bundles_L):
    """(bank, key, kstar) for every minimal rescue group within the cap."""
    out = []
    for key, e in sorted(sweep["G"].items()):
        if e.get("kstar") and e["kstar"] <= B.KS_MAX:
            out.append(("G", key, int(e["kstar"])))
    for key, b in sorted(bundles_L.items()):
        if b.get("kstar") and b["kstar"] <= B.KS_MAX:
            out.append(("L", key, int(b["kstar"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s3b_ablate")
    ap.add_argument("--banks", default="G,L")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cases = json.load(open(os.path.join(OUT, B.CASES_JSON)))
    sweep = json.load(open(os.path.join(OUT, "s3b_sweep.json")))
    recs = {b: {r["key"]: r for r in cases["banks"][b]} for b in ("G", "L")}
    want = set(args.banks.split(","))
    todo = [t for t in bundles_of(sweep, cases["bundles_L"]) if t[0] in want]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[Part B] {len(todo)} bundles")

    out_path = os.path.join(OUT, f"{args.tag}.json")
    results = dict(config=dict(ks_max=B.KS_MAX, thr=B.THR, gen_cap=G.GEN_CAP,
                               n_randwin=B.N_RANDWIN, n_randt=B.N_RANDT,
                               n_rando=B.N_RANDO, shifts=list(B.SHIFTS),
                               n_blockperm=B.N_BLOCKPERM,
                               n_remrand=B.N_REMRAND, seed_arms=B.SEED_ARMS),
                   done=[], runs={})
    if os.path.exists(out_path):
        prev = json.load(open(out_path))
        if prev["config"] != results["config"]:
            raise RuntimeError("existing s3b_ablate.json under a different config")
        results["done"], results["runs"] = prev["done"], prev["runs"]
        print(f"[resume] {len(results['done'])} bundles done")
    todo = [t for t in todo if f"{t[0]}|{t[1]}" not in results["done"]]

    datasets = {ds: common.build_dataset(ds) for ds in B.DS_ALL}
    model, h = G.load_stack()
    t0 = time.time()
    for bank, key, kstar in todo:
        rec = recs[bank][key]
        try:
            prep, ans, row, golds = G.instance_prep(h, model, datasets, rec)
            arms = B.partb_plan(rec, kstar)
            src = sweep[bank].get(key) or sweep["G"].get(key) or sweep["L"].get(key)
            ent = dict(bank=bank, key=key, ds=rec["ds"], kstar=kstar,
                       m=rec["m"], floor=(src or {}).get("floor"),
                       arms={})
            print(f"\n[{bank}|{key}] k*={kstar} m={rec['m']} "
                  f"arms={len(arms)}")
            for a in arms:
                m = G.score_set(h, prep, ans, rec, a["adds"], a["drops"], row)
                ent["arms"][a["name"]] = dict(
                    kind=a["kind"], hit=m["hit"], L=m["L"],
                    pred=m["pred"][:200], adds=[int(t) for t in a["adds"]],
                    drops=[int(t) for t in a["drops"]])
                flag = "*" if m["hit"] >= B.THR else " "
                print(f"   {flag}{a['name']:12s} {a['kind']:16s} "
                      f"hit={m['hit']:.2f} L={m['L']:.3f} "
                      f"pred={m['pred'][:34]!r}")
            results["runs"][f"{bank}|{key}"] = ent
            results["done"].append(f"{bank}|{key}")
        except Exception:
            traceback.print_exc()
            print(f"[skip] {bank}|{key}")
            continue
        tmp = out_path + ".tmp"
        json.dump(results, open(tmp, "w"))
        os.replace(tmp, out_path)
        print(f"[{len(results['done'])} bundles] {bank}|{key} "
              f"{time.time() - t0:.0f}s")

    # quick console summary
    full_hits = [v["arms"]["full"]["hit"] for v in results["runs"].values()]
    print(f"\n[Part B] {len(full_hits)} bundles; full-arm hit mean "
          f"{np.mean(full_hits):.3f} (determinism vs Part A reported in analysis)")
    G.require_complete(results, len(todo), os.path.basename(out_path))
    print(f"[saved] {out_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
