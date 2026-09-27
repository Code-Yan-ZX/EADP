"""
M9 Phase 1 evaluation -- boundary adjudication quality, offline.

Every arm decides, per instance, which r of the boundary (tail-r u reserve-32)
survive. Two framings for every signal:

    adj   full-boundary adjudication: rank ALL 32+r boundary members by the
          score, keep top-r. Tail members compete through their own score.
    resc  rescue framing (the M6/M8-comparable one): always evict the r tail
          tokens and rescue the top-r reserves by the score. The cos_s0c resc
          cell must reproduce M8's mean teacher rank ~118.9 -- that is the
          cross-check that the panel, pool and metric are the M8 ones.

Captured-signal aggregations (fixed, no search):
    metric  att_mean | avn | cmc_mn | cmc_nm
    qagg    A1 = last question row; A2 = mean of last 4 question rows;
            A3 = mean of all question rows   (suffix rows 1..nq)
    L       1 / 2 / 4 / 8 captured layers (0-based)
    persist max over layers <= L of the A2 signal (cmc_mn / cmc_nm only)

Gate (pre-registered in the brief): r=8 CMC mean swap teacher rank < 60,
ideally < 40-50, clearly better than cos_s0c (~119); ~100-150 -> REFUTED.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_gdep import OUTPUT_DIR                                        # noqa: E402
from m9_common import load_banks                                      # noqa: E402

R_LIST = (4, 8, 16)
L_MAP = {1: (0,), 2: (0, 1), 4: (0, 1, 2, 3), 8: (0, 1, 2, 3, 4, 5, 6, 7)}
QAGG = ("A1_last", "A2_last4", "A3_all")
METRICS = ("att_mean", "avn", "cmc_mn", "cmc_nm")
N_SEEDS = 20


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


class Panel:
    """Per-instance static quantities, computed once."""

    def __init__(self):
        z = np.load(os.path.join(OUTPUT_DIR, "m9_phase1_scores.npz"),
                    allow_pickle=False)
        self.z = z
        banks = load_banks()
        key2row = {str(k): i for i, k in enumerate(banks["key"])}
        self.rows = np.array([key2row[str(k)] for k in z["key"]])
        self.banks = banks
        n = len(self.rows)
        self.qlens = z["qlens"]
        self.nq = z["nq"]
        self.reserve = z["reserve"]
        self.tail16 = z["tail"]
        self.scores = z["scores"].astype(np.float64)
        # teacher rank within the dropped set, -1 for S0 members
        self.tr = np.full((n, 1024), -1, dtype=np.int64)
        for j, row in enumerate(self.rows):
            order = banks["order"][int(row)]
            in_s0 = np.zeros(1024, bool)
            in_s0[order[:256]] = True
            dropped = np.flatnonzero(~in_s0)
            self.tr[j, dropped] = np.argsort(
                np.argsort(-banks["g2"][int(row)][dropped]))
        assert (self.nq >= 1).all(), "instance without question tokens"

    def bnd(self, j, r):
        tail_r = self.tail16[j, 16 - r:]
        return np.concatenate([tail_r, self.reserve[j]])

    def signal(self, j, r, metric, qagg, lidx):
        """score per boundary member of instance j, order [tail_r, reserve].
        Plain variant: the LAST layer in lidx (per-layer values)."""
        Q, nq = int(self.qlens[j]), int(self.nq[j])
        m = self.scores[j, lidx[-1], METRICS.index(metric), :Q, :]
        if qagg == "A1_last":
            v = m[nq]                       # last question row
        elif qagg == "A2_last4":
            v = m[max(1, nq - 3):nq + 1].mean(0)
        else:
            v = m[1:nq + 1].mean(0)
        # capture columns: 16 tail + 32 reserve -> this r's boundary order
        return np.concatenate([v[16 - r:16], v[16:]])

    def arm(self, r, picks_by_inst):
        ksw, ovl, resc = [], [], []
        rec = []
        for j, picks in enumerate(picks_by_inst):
            is_res = np.isin(picks, self.reserve[j])
            ksw.append(int(is_res.sum()))
            sw = picks[is_res]
            resc.extend(self.tr[j, sw].tolist())
            ovl.append(len(np.intersect1d(picks, self.oracle(j, r))) / r)
            dropped = np.flatnonzero(self.tr[j] >= 0)
            torder = dropped[np.argsort(self.tr[j][dropped])]
            rec.append(np.isin(sw, torder[:r]).sum() / r)
        sr = np.array(resc, dtype=np.float64)
        ksw = np.array(ksw)
        return dict(mean_k_swap=float(ksw.mean()),
                    frac_any_swap=float((ksw > 0).mean()),
                    head_recall_at_r=float(np.mean(rec)),
                    mean_swap_rank=float(sr.mean()) if len(sr) else None,
                    median_swap_rank=float(np.median(sr)) if len(sr) else None,
                    oracle_overlap=float(np.mean(ovl)))

    def oracle(self, j, r):
        bnd = self.bnd(j, r)
        g2b = self.banks["g2"][int(self.rows[j])][bnd]
        return bnd[np.argsort(-g2b)[:r]]

    def with_rho(self, arm_res, sigs_res, j_sorted=None):
        """attach mean Spearman(signal, teacher) over the 32 reserves."""
        rhos = []
        for j, sig in enumerate(sigs_res):
            tr = self.tr[j, self.reserve[j]].astype(np.float64)
            rhos.append(spearman(sig, -tr))
        arm_res = dict(arm_res)
        arm_res["mean_rho_P"] = float(np.nanmean(rhos))
        return arm_res


def main():
    P = Panel()
    n = len(P.rows)
    rng = np.random.default_rng(0)
    results = {}
    for r in R_LIST:
        res = {}
        bnd = [P.bnd(j, r) for j in range(n)]

        # --- offline arms ---------------------------------------------------
        res["oracle_boundary"] = P.arm(r, [
            bnd[j][np.argsort(-P.banks["g2"][int(P.rows[j])][bnd[j]])[:r]]
            for j in range(n)])
        res["incumbent_tail"] = P.arm(r, [P.tail16[j, 16 - r:]
                                          for j in range(n)])
        rand_arms = []
        for s in range(N_SEEDS):
            picks = [rng.permutation(bnd[j])[:r] for j in range(n)]
            rand_arms.append(P.arm(r, picks))
        res["random"] = dict(
            mean_k_swap=float(np.mean([a["mean_k_swap"]
                                       for a in rand_arms])),
            head_recall_at_r=float(np.mean([a["head_recall_at_r"]
                                            for a in rand_arms])),
            mean_swap_rank=float(np.mean([a["mean_swap_rank"]
                                          for a in rand_arms])),
            n_seeds=N_SEEDS)

        # cos_s0c (score = -cos, oriented low)
        cos_adj = []
        cos_resc = []
        sigs_res = []
        for j in range(n):
            cb = -P.banks["cos"][int(P.rows[j])][bnd[j]]
            cos_adj.append(bnd[j][np.argsort(-cb)[:r]])
            cp = -P.banks["cos"][int(P.rows[j])][P.reserve[j]]
            picks = P.reserve[j][np.argsort(-cp)[:r]]
            cos_resc.append(picks)
            sigs_res.append(cp)
        res["cos_adj"] = P.arm(r, cos_adj)
        res["cos_resc"] = P.with_rho(P.arm(r, cos_resc), sigs_res)

        # --- captured-signal arms -------------------------------------------
        for L, lidx in L_MAP.items():
            for metric in METRICS:
                for qagg in QAGG:
                    sigs = [P.signal(j, r, metric, qagg, lidx)
                            for j in range(n)]
                    picks_adj = [bnd[j][np.argsort(-sigs[j])[:r]]
                                 for j in range(n)]
                    picks_resc = [P.reserve[j][
                        np.argsort(-sigs[j][r:])[:r]] for j in range(n)]
                    res[f"{metric}|{qagg}|L{L}|adj"] = P.arm(r, picks_adj)
                    res[f"{metric}|{qagg}|L{L}|resc"] = P.with_rho(
                        P.arm(r, picks_resc),
                        [sigs[j][r:] for j in range(n)])
            for metric in ("cmc_mn", "cmc_nm"):
                # persistent: max over layers <= L of the A2 signal
                sigs_max = []
                for j in range(n):
                    Q, nq = int(P.qlens[j]), int(P.nq[j])
                    per_layer = []
                    for li in lidx:
                        q = P.scores[j, li, METRICS.index(metric),
                                     :Q, :][max(1, nq - 3):nq + 1].mean(0)
                        per_layer.append(np.concatenate(
                            [q[16 - r:16], q[16:]]))
                    sigs_max.append(np.stack(per_layer).max(0))
                picks_adj = [bnd[j][np.argsort(-sigs_max[j])[:r]]
                             for j in range(n)]
                picks_resc = [P.reserve[j][
                    np.argsort(-sigs_max[j][r:])[:r]] for j in range(n)]
                res[f"{metric}|maxL|L{L}|adj"] = P.arm(r, picks_adj)
                res[f"{metric}|maxL|L{L}|resc"] = P.with_rho(
                    P.arm(r, picks_resc),
                    [sigs_max[j][r:] for j in range(n)])
        results[f"r{r}"] = res

    path = os.path.join(OUTPUT_DIR, "m9_phase1_eval.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"[saved] {path}")

    for r in R_LIST:
        print(f"\n=== r={r}: mean swap teacher rank / head recall@r ===")
        ranked = sorted(((k, v) for k, v in results[f"r{r}"].items()
                         if isinstance(v, dict) and "mean_swap_rank" in v
                         and v["mean_swap_rank"] is not None),
                        key=lambda x: x[1]["mean_swap_rank"])
        for k, v in ranked:
            rho = v.get("mean_rho_P")
            med = v.get("median_swap_rank")
            ovl = v.get("oracle_overlap")
            ksw = v.get("mean_k_swap")
            print(f"  {k:32s} rank {v['mean_swap_rank']:7.1f} "
                  f"med {(med if med is not None else float('nan')):6.1f} "
                  f"recall {v['head_recall_at_r']:.3f} "
                  f"k_swap {(ksw if ksw is not None else float('nan')):5.2f} "
                  f"ovl {(ovl if ovl is not None else float('nan')):.3f} "
                  f"rho {('%+.3f' % rho) if rho is not None else '    -'}")


if __name__ == "__main__":
    main()
