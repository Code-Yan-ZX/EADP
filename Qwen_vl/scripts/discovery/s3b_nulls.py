"""
S3-B step 3b (GPU): the comparison class the verdict needs -- null sets on
instances the prefix does NOT rescue.

docs/s3b_prereg_rules.md section 5 makes a candidate structure variable prove
two things: it separates a bundle from its same-window alternatives (measured
inside s3b_ablate.py), AND it separates *successful* from *unsuccessful*
size-matched sets across instances. The second clause needs matched sets on the
non-rescued instances, which is this stage: for every bank-G instance whose
prefix sweep reached k = 32 without ever hitting, the same randwin / randT null
draws are measured at the two grid points k = 8 and k = 16 (the sizes where
bundles are common), with the identical removal rule.

They also give the base rate of "a random handful of missed teacher tokens
happens to fix the answer" on instances where the teacher's own order does not,
which is the denominator every Part-B rate should be read against.
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

KS = (8, 16)
N_DRAW = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s3b_nulls")
    args = ap.parse_args()

    cases = json.load(open(os.path.join(OUT, B.CASES_JSON)))
    sweep = json.load(open(os.path.join(OUT, "s3b_sweep.json")))
    recs = {r["key"]: r for r in cases["banks"]["G"]}
    todo = sorted(k for k, e in sweep["G"].items() if not e.get("kstar"))
    print(f"[nulls] {len(todo)} non-rescued bank-G instances")

    out_path = os.path.join(OUT, f"{args.tag}.json")
    results = dict(config=dict(ks=list(KS), n_draw=N_DRAW, thr=B.THR,
                               gen_cap=G.GEN_CAP, seed_arms=B.SEED_ARMS),
                   done=[], runs={})
    if os.path.exists(out_path):
        prev = json.load(open(out_path))
        if prev["config"] != results["config"]:
            raise RuntimeError("existing s3b_nulls.json under a different config")
        results["done"], results["runs"] = prev["done"], prev["runs"]
        print(f"[resume] {len(results['done'])} done")
    todo = [k for k in todo if k not in results["done"]]

    datasets = {ds: common.build_dataset(ds) for ds in B.DS_ALL}
    model, h = G.load_stack()
    t0 = time.time()
    for key in todo:
        rec = recs[key]
        try:
            prep, ans, row, golds = G.instance_prep(h, model, datasets, rec)
            ent = dict(key=key, ds=rec["ds"], m=rec["m"],
                       floor=sweep["G"][key]["floor"], ks={})
            for k in KS:
                if k > rec["m"]:
                    continue
                adds0, drops0 = B.prefix_arms(rec, k)
                W = B.window(rec, k)
                Gset = set(adds0)
                win_pool = [t for t in rec["T_only"][:W] if t not in Gset]
                uni_pool = [t for t in rec["T_only"] if t not in Gset]
                arm = {"prefix": dict(adds=adds0, drops=drops0,
                                      kind="prefix_unsuccess")}
                for r in range(N_DRAW):
                    pick = B._rng(rec, f"nrandwin{k}|{r}").choice(
                        win_pool, size=k, replace=False)
                    arm[f"randwin{r}"] = dict(adds=[int(t) for t in pick],
                                              drops=drops0, kind="null_window")
                for r in range(N_DRAW):
                    pick = B._rng(rec, f"nrandT{k}|{r}").choice(
                        uni_pool, size=k, replace=False)
                    arm[f"randT{r}"] = dict(adds=[int(t) for t in pick],
                                            drops=drops0, kind="null_uni")
                ent["ks"][str(k)] = {}
                for name, a in arm.items():
                    m = G.score_set(h, prep, ans, rec, a["adds"], a["drops"], row)
                    ent["ks"][str(k)][name] = dict(
                        kind=a["kind"], hit=m["hit"], L=m["L"],
                        pred=m["pred"][:200],
                        adds=[int(t) for t in a["adds"]],
                        drops=[int(t) for t in a["drops"]])
                hits = {n: v["hit"] >= B.THR for n, v in ent["ks"][str(k)].items()}
                print(f"[{key}] k={k:2d} prefix={hits['prefix']} "
                      f"nulls_hit={sum(v for n, v in hits.items() if n != 'prefix')}"
                      f"/{len(hits) - 1}")
            results["runs"][key] = ent
            results["done"].append(key)
        except Exception:
            traceback.print_exc()
            print(f"[skip] {key}")
            continue
        tmp = out_path + ".tmp"
        json.dump(results, open(tmp, "w"))
        os.replace(tmp, out_path)
        print(f"[{len(results['done'])}/{len(todo)}] {key} {time.time() - t0:.0f}s")
    G.require_complete(results, len(todo), os.path.basename(out_path))
    print(f"[saved] {out_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
