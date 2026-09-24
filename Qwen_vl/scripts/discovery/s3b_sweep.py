"""
S3-B step 2 (GPU): Part A -- the minimal rescue groups.

bank G (the current student). Every instance whose F(0) generation is wrong
(s3b_base.json) is swept over the frozen prefix grid k = 1,2,3,4,6,8,12,16,24,32
in ascending order, stopping at the first k with official hit >= 0.5. That k is
k*, and G = T_only[:k*] is the instance's minimal rescue group under the
budget-frozen F(k) protocol. Instances not rescued by 32 are extended to
k = 48, 64 and recorded k* = nil. Both endpoints of every k point are measured:
official accuracy (primary) and teacher-forced gold NLL (secondary).

bank L (S2-C2's student). Its accuracy curve is already on disk at the fine
grid, so Part A costs nothing; the NLL along the same grid is added here, plus a
re-generation of every bundle's own k* point (B2b) to confirm the frozen hit
labels under this stage's generation cap.
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


def ks_for(rec, kstar):
    """the k points bank L needs an NLL for: the frozen grid, capped just past
    the rescue, plus the immediate neighbours of k*."""
    ks = [k for k in B.KS_SWEEP if k <= max(B.KS_MAX, (kstar or B.KS_MAX))]
    if kstar:
        ks += [k for k in (kstar - 1, kstar, kstar + 1) if 1 <= k <= rec["m"]]
    return sorted(set(ks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s3b_sweep")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-L", action="store_true")
    args = ap.parse_args()

    cases = json.load(open(os.path.join(OUT, B.CASES_JSON)))
    base = json.load(open(os.path.join(OUT, "s3b_base.json")))
    bundles_L = cases["bundles_L"]
    recsG = {r["key"]: r for r in cases["banks"]["G"]}
    recsL = {r["key"]: r for r in cases["banks"]["L"]}

    # ---- class: bank-G instances wrong at F(0) in THIS regime -------------
    todo_G = sorted(k for k, e in base["recs"].items()
                    if e["banks"]["G"]["hit"] < B.THR)
    todo_L = sorted(k for k, v in bundles_L.items() if v["kstar"])
    if args.limit:
        todo_G = todo_G[:args.limit]

    out_path = os.path.join(OUT, f"{args.tag}.json")
    results = dict(config=dict(ks_sweep=list(B.KS_SWEEP), ks_ext=list(B.KS_EXT),
                               thr=B.THR, gen_cap=G.GEN_CAP),
                   done_G=[], done_L=[], G={}, L={})
    if os.path.exists(out_path):
        prev = json.load(open(out_path))
        if prev["config"] != results["config"]:
            raise RuntimeError("existing s3b_sweep.json under a different config")
        results.update({k: prev[k] for k in ("done_G", "done_L", "G", "L")})
        print(f"[resume] G {len(results['done_G'])}/{len(todo_G)}  "
              f"L {len(results['done_L'])}/{len(todo_L)}")

    datasets = {ds: common.build_dataset(ds) for ds in B.DS_ALL}
    model, h = G.load_stack()
    t0 = time.time()

    # ------------------------------------------------------------------ bank G
    for key in [k for k in todo_G if k not in results["done_G"]]:
        rec = recsG[key]
        try:
            prep, ans, row, golds = G.instance_prep(h, model, datasets, rec)
            floor = base["recs"][key]["floor"]
            ent = dict(key=key, ds=rec["ds"], m=rec["m"], floor=floor,
                       L0=base["recs"][key]["banks"]["G"]["L"],
                       hit0=base["recs"][key]["banks"]["G"]["hit"],
                       pred0=base["recs"][key]["banks"]["G"]["pred"],
                       curve={}, ext={}, kstar=None, kstar_ext=None)
            print(f"\n[{key}] m={rec['m']} L0={ent['L0']:.3f} "
                  f"B1={rec['hits']['B1']:.2f} B0={rec['hits']['B0']:.2f} "
                  f"T={rec['hits']['identity_T']:.2f}")
            for k in list(B.KS_SWEEP) + list(B.KS_EXT):
                if k > rec["m"]:
                    continue
                adds, drops = B.prefix_arms(rec, k)
                m = G.score_set(h, prep, ans, rec, adds, drops, row)
                tag = "curve" if k in B.KS_SWEEP else "ext"
                ent[tag][str(k)] = dict(hit=m["hit"], L=m["L"],
                                         pred=m["pred"][:200],
                                         added=adds, dropped=drops)
                print(f"   k={k:3d} hit={m['hit']:.2f} L={m['L']:.3f} "
                      f"pred={m['pred'][:40]!r}")
                if m["hit"] >= B.THR:
                    if k in B.KS_SWEEP:
                        ent["kstar"] = int(k)
                    else:
                        ent["kstar_ext"] = int(k)
                    break
            results["G"][key] = ent
            results["done_G"].append(key)
        except Exception:
            traceback.print_exc()
            print(f"[skip] {key}")
            continue
        tmp = out_path + ".tmp"
        json.dump(results, open(tmp, "w"))
        os.replace(tmp, out_path)
        print(f"[G {len(results['done_G'])}/{len(todo_G)}] {key} "
              f"k*={ent['kstar']}  {time.time() - t0:.0f}s")

    # ------------------------------------------------------------------ bank L
    if not args.skip_L:
        swap = json.load(open(os.path.join(OUT, "s2c2_rescue.json")))["runs"]
        ref_pred = {}
        for arm, r in swap.items():
            if not arm.startswith("teacher:"):
                continue
            k = int(arm.split(":")[1].split("|")[0])
            for key, pred, hit in zip(r["keys"], r["predictions"], r["hits"]):
                ref_pred[(key, k)] = (pred, float(hit))
        for key in [k for k in todo_L if k not in results["done_L"]]:
            rec = recsL[key]
            b = bundles_L[key]
            try:
                prep, ans, row, golds = G.instance_prep(h, model, datasets, rec)
                floor = base["recs"][key]["floor"]
                ent = dict(key=key, ds=rec["ds"], m=rec["m"], floor=floor,
                           kstar=b["kstar"], L0=base["recs"][key]["banks"]["L"]["L"],
                           hit0=base["recs"][key]["banks"]["L"]["hit"],
                           hit_frozen=b["curve"].get("0", b["hit_student"]),
                           curve_NLL={}, regen={},
                           curve_hit_frozen=b["curve"])
                for k in ks_for(rec, b["kstar"]):
                    adds, drops = B.prefix_arms(rec, k)
                    sel = B.apply_arms(rec, adds, drops)
                    L = h.nll(prep, sel, ans)[0]
                    ent["curve_NLL"][str(k)] = float(L)
                # B2b: re-generate the bundle point, compare with S2-C2
                for k in [b["kstar"]]:
                    if (key, k) not in ref_pred:
                        continue
                    adds, drops = B.prefix_arms(rec, k)
                    m = G.score_set(h, prep, ans, rec, adds, drops, row)
                    pr, hr = ref_pred[(key, k)]
                    ent["regen"][str(k)] = dict(
                        hit=m["hit"], hit_frozen=hr, L=m["L"],
                        pred=m["pred"][:200],
                        text_prefix_match=bool(
                            m["pred"].startswith(pr[:len(m["pred"])])),
                        hit_match=bool(abs(m["hit"] - hr) < 1e-6))
                results["L"][key] = ent
                results["done_L"].append(key)
            except Exception:
                traceback.print_exc()
                print(f"[skip L] {key}")
                continue
            tmp = out_path + ".tmp"
            json.dump(results, open(tmp, "w"))
            os.replace(tmp, out_path)
            print(f"[L {len(results['done_L'])}/{len(todo_L)}] {key} "
                  f"k*={b['kstar']} {time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- summary
    ks = [v["kstar"] for v in results["G"].values()]
    print("\n[Part A bank G] swept", len(results["G"]),
          "rescued<=32:", sum(1 for x in ks if x),
          "median k*:", (np.median([x for x in ks if x])
                         if any(ks) else None),
          "k* nil but rescued 33-64:",
          sum(1 for v in results["G"].values()
              if not v["kstar"] and v.get("kstar_ext")))
    kl = [v["kstar"] for v in results["L"].values()]
    print("[Part A bank L] bundles", len(kl), "median", np.median(kl),
          "regen agreement:",
          f"{sum(1 for v in results['L'].values() for r in v['regen'].values() if r['hit_match'])}"
          f"/{sum(len(v['regen']) for v in results['L'].values())}")
    print(f"[saved] {out_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
