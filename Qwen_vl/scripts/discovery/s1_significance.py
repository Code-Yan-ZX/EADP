"""Paired bootstrap over instances for the S1 Top-3 configs (per-instance statistic)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT

rng = np.random.default_rng(0)
D = json.load(open(os.path.join(OUT, "s1_scoring_search.json")))
off, res, keys, nnec = D["official"], D["results"], D["rec_keys"], D["rec_nnec"]
by = {(r["norm"], r["alpha"], r["lam"]): r for r in res}
nnec = np.array(nnec)
TIERS = {"loc64": nnec == 64, "loc128": nnec <= 128, "loc256": nnec <= 256,
         "all": np.ones(len(nnec), bool)}

# Top-3 by primary-tier delta (ties broken by the broader tiers)
ranked = sorted(res, key=lambda r: np.mean(
    [r["per"][k]["mean_rank_pct"] - off[k]["mean_rank_pct"]
     for k, n in zip(keys, nnec) if n == 64]))
top3 = []
seen = set()
for r in ranked:
    c = (r["norm"], r["alpha"], r["lam"])
    if c in seen:
        continue
    seen.add(c)
    top3.append(r)
    if len(top3) == 3:
        break

print("Top-3 by primary tier (nNec == 64):")
for r in top3:
    print(f"  norm={r['norm']:10s} alpha={r['alpha']:.2f} lam={r['lam']:.2f}")

print(f"\n{'config':32s} {'tier':8s} {'n':>3s} {'official':>9s} {'cand':>9s} "
      f"{'delta':>8s} {'95% CI':>20s} {'impr/wors':>10s}")
for r in top3:
    for t, m in TIERS.items():
        ks = [k for k, mm in zip(keys, m) if mm]
        o = np.array([off[k]["mean_rank_pct"] for k in ks])
        c = np.array([r["per"][k]["mean_rank_pct"] for k in ks])
        d = c - o
        bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(20000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        cid = (f"[{lo:+.4f},{hi:+.4f}]")
        tag = f"{str((r['norm'], r['alpha'], r['lam'])):32s}"
        print(f"{tag} {t:8s} {len(d):3d} {o.mean():9.4f} {c.mean():9.4f} "
              f"{d.mean():+8.4f} {cid:>20s} "
              f"{int((d < -1e-9).sum())}/{int((d > 1e-9).sum()):<4d}")
    print()

# sign test on the largest tier, for the best config
from math import comb
r = top3[0]
d = np.array([r["per"][k]["mean_rank_pct"] - off[k]["mean_rank_pct"] for k in keys])
n_imp = int((d < -1e-9).sum())
n = int((np.abs(d) > 1e-9).sum())
p = sum(comb(n, i) for i in range(n_imp, n + 1)) / 2 ** n
print(f"sign test on all 15 causal cases for the Top-1 config: "
      f"{n_imp}/{n} improved, two-sided p = {2*min(p, 1-p):.3f}")

# how much of the gain survives at the *best achievable* ranking quality
print("\nceiling of this family:")
for t, m in TIERS.items():
    ks = [k for k, mm in zip(keys, m) if mm]
    au = [res_i["per"][k]["block_auroc"] for res_i in res for k in ks]
    au = [x for x in au if x == x]
    best = max(np.mean([res_i["per"][k]["block_auroc"] for k in ks]) for res_i in res)
    off_au = np.mean([off[k]["block_auroc"] for k in ks])
    print(f"  {t:8s} block AUROC: official {off_au:.3f}  best-in-family {best:.3f}")
