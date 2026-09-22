"""
S2-A controls: is the gradient saliency real, or a confound?

  C1  is G2 = sum_d |g_i,d * v_i,d| merely the input feature magnitude ||v_i||?
  C2  do the necessary blocks cluster at one spatial position, so that a position
      prior would score as well?
  C3  is the metric simply easy (random score should sit at AUROC ~0.5)?
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import BLOCK, MAPS, OUT
from scoring_search_s1 import evaluate

rng = np.random.default_rng(0)
METHODS = ["A1_G2", "A1_G1", "P1_G2", "P1_G1"]
TIERS = {"loc64": lambda n: n == 64, "loc256": lambda n: n <= 256}


def main():
    D = json.load(open(os.path.join(OUT, "s2a_gradient_viability.json")))
    recs = D["records"]
    keys = sorted(recs)
    Z = np.load(os.path.join(OUT, "s2a_input_norm.npz"))
    norms = {k: Z[k].astype(np.float64) for k in Z.files}

    for tk, fn in TIERS.items():
        sel = [k for k in keys if fn(len(recs[k]["needed"]) * BLOCK * BLOCK)]
        print(f"\n=== tier {tk}  (n={len(sel)}) ===")
        print(f"{'score':28s}{'meanRank':>10s}{'AUROC':>8s}{'AP':>8s}")

        def show(name, fn_score):
            ms, aus, aps = [], [], []
            for k in sel:
                rec = dict(needed=np.array(recs[k]["needed"]))
                r = evaluate(rec, fn_score(k))
                ms.append(r["mean_rank_pct"])
                aus.append(r["block_auroc"])
                aps.append(r["block_ap"])
            print(f"{name:28s}{np.mean(ms):10.4f}{np.nanmean(aus):8.3f}{np.nanmean(aps):8.3f}")

        for m in METHODS:
            show(m, lambda k, m=m: np.asarray(D["scores"][k][m], dtype=np.float64))
        show("CONTROL ||v|| only", lambda k: norms[k])
        show("CONTROL random", lambda k: rng.random(1024))
        show("CONTROL shuffled ||v||", lambda k: rng.permutation(norms[k]))

        cs = [np.corrcoef(np.asarray(D["scores"][k]["A1_G2"], dtype=np.float64), norms[k])[0, 1]
              for k in sel]
        print(f"  corr(A1_G2, ||v||): mean {np.mean(cs):+.3f} "
              f"range [{np.min(cs):+.3f}, {np.max(cs):+.3f}]")
        cg = [np.corrcoef(np.asarray(D["scores"][k]["A1_G1"], dtype=np.float64), norms[k])[0, 1]
              for k in sel]
        print(f"  corr(A1_G1, ||v||): mean {np.mean(cg):+.3f}")

        allb = np.concatenate([np.array(recs[k]["needed"]) for k in sel])
        cnt = np.bincount(allb, minlength=16)
        print(f"  necessary-block histogram over 16 blocks: {cnt.tolist()}")
        print(f"  distinct blocks hit {int((cnt > 0).sum())}/16; "
              f"most frequent covers {100 * cnt.max() / cnt.sum():.1f}% of positives")


if __name__ == "__main__":
    main()
