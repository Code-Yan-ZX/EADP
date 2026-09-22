"""Render the S1 tables: per-case deltas, alpha/lambda profiles, AUROC/AP, Recall@k."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import BLOCK, OUT

D = json.load(open(os.path.join(OUT, "s1_scoring_search.json")))
off, res, keys, nnec = D["official"], D["results"], D["rec_keys"], D["rec_nnec"]
tier = lambda n: {"loc64": n == 64, "loc128": n <= 128, "loc256": n <= 256, "all": True}
TIERS = ["loc64", "loc128", "loc256", "all"]

by = {(r["norm"], r["alpha"], r["lam"]): r for r in res}


def aggr(r, t, field="mean_rank_pct"):
    ks = [k for k, n in zip(keys, nnec) if tier(n)[t]]
    v = [r["per"][k][field] for k in ks]
    o = [off[k][field] for k in ks]
    v = [x for x in v if x == x]
    o = [x for x in o if x == x]
    return (np.mean(v) if v else np.nan, np.mean(o) if o else np.nan, len(v))


print("=" * 108)
print("A. alpha / lambda profile at norm=none  (mean necessary rank pct, lower = better)")
for t in TIERS:
    print(f"\n  -- tier {t} --")
    print(f"  {'':6s}" + "".join(f"{'lam=%.2f' % l:>12s}" for l in [0, .25, .5, .75, 1.0]))
    for a in [0, .25, .5, .75, 1.0]:
        row = f"  a={a:<4.2f}"
        for l in [0, .25, .5, .75, 1.0]:
            r = by.get(("none", a, l))
            if r is None:
                row += f"{'-':>12s}"
                continue
            m, o, n = aggr(r, t)
            row += f"{m:>7.4f}({m - o:+.3f})"
        print(row)
    m, o, n = aggr(by[("none", .5, 1.0)], t)
    print(f"  official: {m:.4f}   (n={n})")

print("\n" + "=" * 108)
print("B. per-case deltas of the best configs on the primary tier (delta = candidate - official;"
      "\n   negative = necessary tokens move UP in rank = better)")
cands = [("none", 1.0, 0.0), ("none", 1.0, 0.25), ("none", 0.75, 0.0), ("ds_robust", 0.75, 0.0),
         ("img_z", 0.75, 0.0), ("img_mm", 0.75, 0.0), ("none", 0.5, 1.0)]
short = [k for k in keys if dict(zip(keys, nnec))[k] == 64]
print(f"\n  {'config':30s}" + "".join(f"{k.split('_')[-1][:8]:>10s}" for k in short) + f"{'mean':>9s}")
for c in cands:
    r = by[c]
    ds = [r["per"][k]["mean_rank_pct"] - off[k]["mean_rank_pct"] for k in short]
    print(f"  {str(c):30s}" + "".join(f"{x:+10.4f}" for x in ds) + f"{np.mean(ds):+9.4f}")
print(f"  {'official (abs)':30s}" + "".join(f"{off[k]['mean_rank_pct']:10.4f}" for k in short))
print(f"\n  cases: " + ", ".join(f"{k}={dict(zip(keys,nnec))[k]}" for k in short))

print("\n" + "=" * 108)
print("C. block-level AUROC / AP  (block = mean of its 64 tokens; label = occlusion-necessary)")
for t in ["loc64", "all"]:
    print(f"\n  -- tier {t} --")
    print(f"  {'config':30s} {'AUROC':>8s} {'vs off':>8s} {'AP':>8s} {'vs off':>8s}")
    rows = sorted(res, key=lambda r: -np.nan_to_num(aggr(r, t, "block_auroc")[0]))
    seen = set()
    for r in rows:
        c = (r["norm"], r["alpha"], r["lam"])
        if c in seen:
            continue
        seen.add(c)
        au, auo, n = aggr(r, t, "block_auroc")
        ap, apo, _ = aggr(r, t, "block_ap")
        if len(seen) > 6:
            break
        print(f"  {str(c):30s} {au:8.3f} {au-auo:+8.3f} {ap:8.3f} {ap-apo:+8.3f}")
    auo, _, _ = aggr(by[(("none"), .5, 1.0)], t, "block_auroc")
    apo, _, _ = aggr(by[(("none"), .5, 1.0)], t, "block_ap")
    print(f"  {'official':30s} {auo:8.3f} {'':>8s} {apo:8.3f}")

print("\n" + "=" * 108)
print("D. Recall@k for necessary tokens")
for t in TIERS:
    print(f"\n  -- tier {t} --")
    print(f"  {'config':30s}" + "".join(f"{'R@%d' % k:>10s}" for k in (128, 256, 512)))
    for c in cands:
        r = by[c]
        row = f"  {str(c):30s}"
        for k in (128, 256, 512):
            m, o, n = aggr(r, t, f"recall@{k}")
            row += f"{m:>10.3f}"
        print(row)
    row = f"  {'official':30s}"
    for k in (128, 256, 512):
        m, o, n = aggr(by[("none", .5, 1.0)], t, f"recall@{k}")
        row += f"{m:>10.3f}"
    print(row)

print("\n" + "=" * 108)
print("E. non-affine arm (per-image rank normalization), best per tier")
for t in TIERS:
    rows = [(r, aggr(r, t)) for r in res if r["norm"] == "img_rank"]
    rows.sort(key=lambda x: x[1][0])
    r, (m, o, n) = rows[0]
    print(f"  {t:8s} best img_rank a={r['alpha']:.2f} l={r['lam']:.2f}  "
          f"mean={m:.4f} delta={m-o:+.4f}")
