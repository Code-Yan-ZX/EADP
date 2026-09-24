"""
S3-B step 3d (GPU): the rank-band ladder — where along the teacher's ordering
does an accuracy flip actually live?

The bundle-vs-null comparisons answer "is THIS set special". This stage answers
the complementary question the SCORE-ORDER-ONLY verdict needs: if the value is
carried by rank position rather than by group membership, then sets of the same
size drawn from successive teacher-rank bands should decay smoothly, with no
band-specific cliff and no bundle-shaped bump.

For every bundle instance (both banks), size k* sets are drawn from dyadic bands
of the missed queue T_only --

    band b covers T_only positions [2^b, 2^(b+1)),  b = 0..6
    band 0'  = T_only[:k*] \\ G   (the bundle's own ranks, its members excluded:
                                  the same-rank-band control for `randwin`)
    band out = image tokens outside S union T

with 3 draws per band (bands with fewer than k* eligible tokens are skipped),
all under the identical F(k*) removal rule. The curve P(hit) vs band, against
the prefix point (P = 1 at band "top k*"), is the shape of the payload.
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

N_DRAW = 3
BANDS = range(0, 7)


def band_sets(rec, kstar):
    k = int(kstar)
    adds0, drops0 = B.prefix_arms(rec, k)
    G = set(adds0)
    m = rec["m"]
    sT = set(rec["T"])
    out_pool = [t for t in range(B.N_VIS)
                if t not in set(rec["S"]) and t not in sT]
    bands = []
    for b in BANDS:
        lo, hi = 2 ** b, 2 ** (b + 1)
        pool = [t for t in rec["T_only"][lo:hi]]
        bands.append((f"pos_{lo}_{hi}", pool))
    bands.append(("outside", out_pool))
    arms = []
    for name, pool in bands:
        if len(pool) < k:
            continue
        for r in range(N_DRAW):
            pick = B._rng(rec, f"band{name}|{r}").choice(pool, size=k,
                                                         replace=False)
            arms.append(dict(name=f"{name}|{r}", kind="band", band=name,
                             adds=[int(t) for t in pick], drops=drops0))
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s3b_bands")
    args = ap.parse_args()

    cases = json.load(open(os.path.join(OUT, B.CASES_JSON)))
    ablate = json.load(open(os.path.join(OUT, "s3b_ablate.json")))
    recs = {b: {r["key"]: r for r in cases["banks"][b]} for b in ("G", "L")}
    todo = sorted((v["bank"], v["key"]) for v in ablate["runs"].values()
                  if v["kstar"] >= 2)

    out_path = os.path.join(OUT, f"{args.tag}.json")
    results = dict(config=dict(n_draw=N_DRAW, bands=[f"pos_{2**b}_{2**(b+1)}"
                                                     for b in BANDS],
                               thr=B.THR, gen_cap=G.GEN_CAP),
                   done=[], runs={})
    if os.path.exists(out_path):
        prev = json.load(open(out_path))
        if prev["config"] != results["config"]:
            raise RuntimeError("existing s3b_bands.json under a different config")
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
            pos = {t: i for i, t in enumerate(rec["T_only"])}
            for a in band_sets(rec, kstar):
                m = G.score_set(h, prep, ans, rec, a["adds"], a["drops"], row)
                rk = [pos[t] for t in a["adds"] if t in pos]
                ent["arms"][a["name"]] = dict(
                    band=a["band"], hit=m["hit"], L=m["L"],
                    pred=m["pred"][:120],
                    rank_mean=float(np.mean(rk)) if rk else float("nan"),
                    adds=[int(t) for t in a["adds"]],
                    drops=[int(t) for t in a["drops"]])
            results["runs"][f"{bank}|{key}"] = ent
            results["done"].append(f"{bank}|{key}")
            print(f"[{bank}|{key}] k*={kstar} arms={len(ent['arms'])} "
                  f"hits={sum(v['hit'] >= B.THR for v in ent['arms'].values())}")
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
