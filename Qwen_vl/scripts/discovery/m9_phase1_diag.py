"""
M9 Phase 1 diagnostics -- the three questions the gate must answer beyond the
mean-swap-rank table:

  D1  per-benchmark direction: is the L4/A2 CMC cell better than cos_s0c on
      TextVQA, DocVQA and OCRBench separately?
  D2  significance: paired bootstrap over the 210 instances of
      (CMC - cos_s0c) on mean swap teacher rank and on head recall@r
  D3  positive-case identification: on instances where the oracle boundary
      adjudication actually swaps in at least one teacher-top-r dropped token
      (an oracle-correction-positive case), P(the arm lands >= 1 such token).

The pre-registered primary cell for everything downstream is
    cmc_mn | A2_last4 | L4 | adj      (declared BEFORE reading D1-D3;
    ties with the best cell of the grid at r=8: 78.6)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_gdep import OUTPUT_DIR                                        # noqa: E402
from m9_common import load_banks                                      # noqa: E402
from m9_phase1_eval import Panel, METRICS, L_MAP, R_LIST              # noqa: E402

PRIMARY = ("cmc_mn", "A2_last4", 4)
NBOOT = 4000
RNG = np.random.default_rng(20260927)


def picks_for(P, j, r, kind):
    bnd = P.bnd(j, r)
    if kind == "oracle":
        g2b = P.banks["g2"][int(P.rows[j])][bnd]
        return bnd[np.argsort(-g2b)[:r]]
    if kind == "cos":
        # oriented like everywhere else: LOW cosine to the S0 centroid first
        return bnd[np.argsort(P.banks["cos"][int(P.rows[j])][bnd])[:r]]
    if kind == "cmc":
        m, q, L = PRIMARY
        sig = P.signal(j, r, m, q, L_MAP[L])
        return bnd[np.argsort(-sig)[:r]]
    if kind == "cmc_resc":
        m, q, L = PRIMARY
        sig = P.signal(j, r, m, q, L_MAP[L])
        return P.reserve[j][np.argsort(-sig[r:])[:r]]
    if kind == "cos_resc":
        return P.reserve[j][
            np.argsort(P.banks["cos"][int(P.rows[j])][P.reserve[j]])[:r]]
    raise KeyError(kind)


def main():
    P = Panel()
    n = len(P.rows)
    ds = np.array([str(x) for x in P.banks["ds"][P.rows]])
    out = {}
    for r in R_LIST:
        arms = {k: [picks_for(P, j, r, k) for j in range(n)]
                for k in ("oracle", "cos", "cmc", "cmc_resc", "cos_resc")}
        # per-instance stats
        def inst_stats(picks):
            rec, ranks, pos = [], [], []
            for j, p in enumerate(picks):
                sw = p[np.isin(p, P.reserve[j])]
                dropped = np.flatnonzero(P.tr[j] >= 0)
                torder = dropped[np.argsort(P.tr[j][dropped])]
                rec.append(np.isin(sw, torder[:r]).sum() / r)
                ranks.append(P.tr[j][sw] if len(sw) else
                             np.array([], dtype=np.int64))
                pos.append(int(len(sw) and (P.tr[j][sw] < r).any()))
            return np.array(rec), ranks, np.array(pos)

        def pooled_mean(ranks_list, idx):
            """pooled mean swap rank over resampled instances (the metric the
            eval tables report)."""
            allr = np.concatenate([ranks_list[i] for i in idx
                                   if len(ranks_list[i])])
            return allr.mean() if allr.size else np.nan
        st = {k: inst_stats(v) for k, v in arms.items()}
        oracle_pos = st["oracle"][2].astype(bool)   # oracle lands a head token
        d = dict(
            r=r,
            n_positive=int(oracle_pos.sum()),
            positive_rate=float(oracle_pos.mean()),
            P_hit_given_positive={
                k: float(st[k][2][oracle_pos].mean()) for k in
                ("cos", "cmc", "cmc_resc", "cos_resc")},
            per_benchmark={},
            bootstrap={},
        )
        for b in sorted(set(ds)):
            sel = ds == b
            d["per_benchmark"][b] = {
                k: dict(recall=float(st[k][0][sel].mean()),
                        mean_swap_rank=float(pooled_mean(
                            st[k][1], np.flatnonzero(sel))))
                for k in ("cos", "cmc", "cmc_resc", "cos_resc")}
        # paired bootstrap cmc vs cos (both framings), pooled-mean metric
        for mkey, a, b in (("cmc-cos", "cmc", "cos"),
                           ("cmc_resc-cos_resc", "cmc_resc", "cos_resc")):
            dr = st[a][0] - st[b][0]
            rec_boot = [dr[RNG.integers(0, n, n)].mean() for _ in range(NBOOT)]
            diffs = []
            for _ in range(NBOOT):
                idx = RNG.integers(0, n, n)
                diffs.append(pooled_mean(st[a][1], idx)
                             - pooled_mean(st[b][1], idx))
            d["bootstrap"][mkey] = dict(
                recall_diff=float(dr.mean()),
                recall_ci=[float(x) for x in
                           np.percentile(rec_boot, [2.5, 97.5])],
                rank_diff_mean=float(np.nanmean(diffs)),
                rank_ci=[float(x) for x in np.percentile(diffs, [2.5, 97.5])],
            )
        out[f"r{r}"] = d
        print(f"r={r}: positive cases {d['n_positive']}/{n} "
              f"({d['positive_rate']:.2f});  P(hit|pos): "
              + "  ".join(f"{k}={v:.3f}" for k, v in
                          d["P_hit_given_positive"].items()))
        for b, v in d["per_benchmark"].items():
            print(f"   {b:12s} recall cos {v['cos']['recall']:.3f} "
                  f"cmc {v['cmc']['recall']:.3f}  |  rank cos "
                  f"{v['cos']['mean_swap_rank']:.1f} "
                  f"cmc {v['cmc']['mean_swap_rank']:.1f}")
        for mkey, v in d["bootstrap"].items():
            print(f"   {mkey}: rank diff {v['rank_diff_mean']:.1f} "
                  f"CI {v['rank_ci']}")
    path = os.path.join(OUTPUT_DIR, "m9_phase1_diag.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
