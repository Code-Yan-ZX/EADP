"""
S2-A analysis: score the gradient saliency maps with the S1 metric code and
apply Gate A / Gate B.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import BLOCK, MAPS, OUT, spatial_smooth
from scoring_search_s1 import evaluate

METHODS = ["A1_G1", "A1_G2", "A1_G1L1", "P1_G1", "P1_G2", "P2_G1", "P2_G2"]
LABEL = {"A1_G1": "A1 grad-norm", "A1_G2": "A1 grad*input", "A1_G1L1": "A1 grad-L1",
         "P1_G1": "P1(top1) grad-norm", "P1_G2": "P1(top1) grad*input",
         "P2_G1": "P2(margin) grad-norm", "P2_G2": "P2(margin) grad*input"}
TIERS = [("loc64", "nNec=64", lambda n: n == 64),
         ("loc256", "nNec<=256 (PRIMARY)", lambda n: n <= 256),
         ("all", "all causal", lambda n: True)]


def main():
    D = json.load(open(os.path.join(OUT, "s2a_gradient_viability.json")))
    recs, scores = D["records"], D["scores"]

    cases = {}
    for key, r in recs.items():
        needed = np.array(r["needed"])
        cases[key] = dict(key=key, ds=r["ds"], idx=r["idx"], needed=needed,
                          nnec=int(len(needed) * BLOCK * BLOCK))

        # baselines
        off = np.load(os.path.join(MAPS, f"probe_{key}.npz"))["importance"]
        b = np.load(os.path.join(MAPS, f"{key}_b256.npz"))
        cases[key]["scores"] = {m: np.asarray(v, dtype=np.float64)
                                for m, v in scores[key].items()}
        cases[key]["scores"]["official"] = off
        cases[key]["scores"]["s1_best"] = b["global_sim"].reshape(-1).astype(np.float64)

    metrics = {k: {m: evaluate(c, s) for m, s in c["scores"].items()}
               for k, c in cases.items()}

    def agg(method, fn, field="mean_rank_pct"):
        ks = [k for k, c in cases.items() if fn(c["nnec"])]
        v = np.array([metrics[k][method][field] for k in ks], dtype=float)
        o = np.array([metrics[k]["official"][field] for k in ks], dtype=float)
        d = v - o
        return dict(n=len(ks), mean=float(np.nanmean(v)), official=float(np.nanmean(o)),
                    delta=float(np.nanmean(d)),
                    improved=int(np.sum(d < -1e-9)), worsened=int(np.sum(d > 1e-9)))

    # ---------------- per-case table, primary tier -------------------------
    prim = [k for k, c in cases.items() if c["nnec"] <= 256]
    print("=" * 118)
    print("PER-CASE mean necessary-token rank percentile (lower = better), primary set nNec<=256")
    hdr = f"{'case':20s}{'nNec':>5s}{'official':>10s}{'s1_best':>9s}" + \
          "".join(f"{m:>10s}" for m in METHODS)
    print(hdr)
    for k in sorted(prim, key=lambda k: (cases[k]["ds"], cases[k]["idx"])):
        c = cases[k]
        row = f"{k:20s}{c['nnec']:5d}{metrics[k]['official']['mean_rank_pct']:10.4f}" \
              f"{metrics[k]['s1_best']['mean_rank_pct']:9.4f}"
        for m in METHODS:
            row += f"{metrics[k][m]['mean_rank_pct']:10.4f}"
        print(row)

    # ---------------- aggregates ------------------------------------------
    for tk, tn, fn in TIERS:
        print("\n" + "=" * 118)
        print(f"AGGREGATE -- {tn}   (delta vs official EADP; negative = better)")
        print(f"{'method':22s}{'n':>3s}{'mean':>9s}{'med':>9s}{'delta':>9s}"
              f"{'impr':>6s}{'wors':>6s}{'AUROC':>8s}{'dAUROC':>8s}{'AP':>7s}"
              f"{'R@128':>7s}{'R@256':>7s}{'R@512':>7s}")
        order = ["official", "s1_best"] + METHODS
        for m in order:
            a = agg(m, fn)
            au = agg(m, fn, "block_auroc")
            ap = agg(m, fn, "block_ap")
            r128, r256, r512 = (agg(m, fn, f"recall@{k}") for k in (128, 256, 512))
            med = np.nanmean([metrics[k][m]["median_rank_pct"]
                              for k, c in cases.items() if fn(c["nnec"])])
            tag = LABEL.get(m, m)
            print(f"{tag:22s}{a['n']:3d}{a['mean']:9.4f}{med:9.4f}{a['delta']:+9.4f}"
                  f"{a['improved']:6d}{a['worsened']:6d}{au['mean']:8.3f}{au['delta']:+8.3f}"
                  f"{ap['mean']:7.3f}{r128['mean']:7.3f}{r256['mean']:7.3f}{r512['mean']:7.3f}")

    # ---------------- gates -------------------------------------------------
    print("\n" + "=" * 118)
    print("GATE A -- answer-conditioned gradient must reach ALL of: dRank >= 0.10,")
    print("         block AUROC >= 0.65, >= 6/8 cases improved")
    a1 = [m for m in METHODS if m.startswith("A1")]
    gA, best = False, None
    for m in a1:
        a = agg(m, lambda n: n <= 256)
        au = agg(m, lambda n: n <= 256, "block_auroc")
        ok = (-a["delta"] >= 0.10) and (au["mean"] >= 0.65) and (a["improved"] >= 6)
        print(f"  {LABEL[m]:24s} dRank={-a['delta']:+.4f} (need 0.10)  "
              f"AUROC={au['mean']:.3f} (need 0.65)  improved={a['improved']}/8 (need 6)  "
              f"-> {'PASS' if ok else 'fail'}")
        if best is None or -a["delta"] > -agg(best, lambda n: n <= 256)["delta"]:
            best = m
        gA = gA or ok

    print("\nGATE B -- answer-free proxy (only if Gate A passes):")
    if not gA:
        print("  not evaluated: Gate A did not pass, so the brief stops at S2-A NO-GO.")
    for m in [x for x in METHODS if x[0] == "P"]:
        a = agg(m, lambda n: n <= 256)
        au = agg(m, lambda n: n <= 256, "block_auroc")
        ok = (au["mean"] >= 0.60 or -a["delta"] >= 0.08) and a["improved"] >= 6
        print(f"  {LABEL[m]:24s} dRank={-a['delta']:+.4f}  AUROC={au['mean']:.3f}  "
              f"improved={a['improved']}/8  -> {'PROMISING' if ok else 'chance-level'}")

    print("\n" + "=" * 118)
    print("EFFICIENCY (per case, unpruned 1024-token forward; fwd = A2 prompt-only forward)")
    fw = [r["t_A2_fwd_ms"] for r in recs.values()]
    bw = [r.get("t_P1_bwd_ms", np.nan) for r in recs.values()]
    a1f = [r.get("t_A1_fwd_ms", np.nan) for r in recs.values()]
    a1b = [r.get("t_A1_bwd_ms", np.nan) for r in recs.values()]
    pk = [r.get("P1_peak_mem_mb", np.nan) for r in recs.values()]
    print(f"  prompt-only forward   {np.nanmean(fw):8.1f} ms")
    print(f"  one backward          {np.nanmean(bw):8.1f} ms")
    print(f"  A1 forward (w/ answer){np.nanmean(a1f):8.1f} ms")
    print(f"  A1 backward           {np.nanmean(a1b):8.1f} ms")
    print(f"  peak memory           {np.nanmean(pk):8.0f} MB")
    print(f"  reference: EADP scoring ~0.9 ms, facility selector ~40 ms, "
          f"unpruned prefill 228 ms (Stage-1)")

    json.dump({"cases": {k: {"ds": c["ds"], "idx": c["idx"], "nnec": c["nnec"]}
                         for k, c in cases.items()},
               "metrics": metrics,
               "gate_a_pass": bool(gA), "best_a1": best},
              open(os.path.join(OUT, "s2a_metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
