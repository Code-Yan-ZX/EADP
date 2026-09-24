"""
S3-B step 3c (GPU): the substitution arms — membership vs rank depth.

Why this stage exists, stated as an amendment to the frozen Part-B table
(docs/s3b_prereg_rules.md section 4) rather than a hidden change:

The frozen null family `randwin_r` draws its k* tokens from
`T_only[:W] \\ G`, i.e. it is **fully disjoint** from the bundle. Combined with
`loo_j` (bundle member j replaced by the STUDENT token it had displaced), the
measured ladder so far is

    keep all k* members      1.00
    keep k*-1 members        0.76      (loo)
    keep 0 members, same window  0.33  (randwin)

which cannot separate "these tokens" from "this many top ranks": loo both
removes a member *and* puts a low-value student token back, while randwin
changes membership and rank depth together. The substitution arms below change
membership at CONSTANT depth, which is the comparison the bundle hypothesis
actually needs:

    subst{j}|{off}  adds = (G \\ {T_only[j]}) \\cup {T_only[k*+off]}
                    drops = F(k*)'s drops without the one belonging to j

so exactly one bundle member is traded for a teacher-missed token that sits
just OUTSIDE the bundle (offset 1, 2 or 4 ranks deeper), for j at the head,
middle and tail of the bundle. Read it against the ladder:

    subst ~ full   -> the bundle is a rank-depth dose, not a set of members
    subst ~ loo    -> members are individually load-bearing
    subst in between -> graded, order-dominated
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

OFFSETS = (1, 2, 4)


def subst_arms(rec, kstar):
    k = int(kstar)
    if k < 2:
        return []
    adds0, drops0 = B.prefix_arms(rec, k)
    js = sorted({0, k // 2, k - 1})
    arms = []
    for j in js:
        for off in OFFSETS:
            idx = k - 1 + off                      # 0-based position in T_only
            if idx >= rec["m"]:
                continue
            # the substitute is a TEACHER-missed token, so the same k student
            # tokens stay dropped and the budget holds: |adds| = |drops| = k.
            arms.append(dict(name=f"subst{j}|{off}",
                             adds=adds0[:j] + adds0[j + 1:] + [rec["T_only"][idx]],
                             drops=drops0,
                             kind="substitute", j=j, off=off,
                             sub_rank=idx))
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s3b_subst")
    args = ap.parse_args()

    cases = json.load(open(os.path.join(OUT, B.CASES_JSON)))
    ablate = json.load(open(os.path.join(OUT, "s3b_ablate.json")))
    recs = {b: {r["key"]: r for r in cases["banks"][b]} for b in ("G", "L")}
    todo = sorted((v["bank"], v["key"]) for v in ablate["runs"].values())

    out_path = os.path.join(OUT, f"{args.tag}.json")
    results = dict(config=dict(offsets=list(OFFSETS), thr=B.THR,
                               gen_cap=G.GEN_CAP), done=[], runs={})
    if os.path.exists(out_path):
        prev = json.load(open(out_path))
        if prev["config"] != results["config"]:
            raise RuntimeError("existing s3b_subst.json under a different config")
        results["done"], results["runs"] = prev["done"], prev["runs"]
        print(f"[resume] {len(results['done'])} done")
    todo = [t for t in todo if f"{t[0]}|{t[1]}" not in results["done"]]

    datasets = {ds: common.build_dataset(ds) for ds in B.DS_ALL}
    model, h = G.load_stack()
    t0 = time.time()
    for bank, key in todo:
        rec = recs[bank][key]
        kstar = ablate["runs"][f"{bank}|{key}"]["kstar"]
        try:
            prep, ans, row, golds = G.instance_prep(h, model, datasets, rec)
            ent = dict(bank=bank, key=key, ds=rec["ds"], kstar=kstar, arms={})
            for a in subst_arms(rec, kstar):
                m = G.score_set(h, prep, ans, rec, a["adds"], a["drops"], row)
                ent["arms"][a["name"]] = dict(
                    kind=a["kind"], j=a["j"], off=a["off"], sub_rank=a["sub_rank"],
                    hit=m["hit"], L=m["L"], pred=m["pred"][:200],
                    adds=[int(t) for t in a["adds"]],
                    drops=[int(t) for t in a["drops"]])
            hits = [v["hit"] >= B.THR for v in ent["arms"].values()]
            print(f"[{bank}|{key}] k*={kstar} subst hits "
                  f"{sum(hits)}/{len(hits)}")
            results["runs"][f"{bank}|{key}"] = ent
            results["done"].append(f"{bank}|{key}")
        except Exception:
            traceback.print_exc()
            print(f"[skip] {bank}|{key}")
            continue
        tmp = out_path + ".tmp"
        json.dump(results, open(tmp, "w"))
        os.replace(tmp, out_path)
        print(f"[{len(results['done'])}/{len(todo)}] {time.time() - t0:.0f}s")
    G.require_complete(results, len(todo), os.path.basename(out_path))
    print(f"[saved] {out_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
