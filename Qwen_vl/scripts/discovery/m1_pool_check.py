"""
M1 step 2b: is the extension pool comparable to the pool it extends?

The ladder's whole reading rests on "n changes and nothing else". Extending the
fit set with 720 images that are systematically harder, longer, or differently
supervised would produce a flat curve for a reason that has nothing to do with
data quantity. So the extension draw is compared against the 240 rows it is
appended to, on quantities that are properties of the *instance* and of the
*teacher map*, not of any model M1 trains:

    seq_len        prompt length (images with more text are longer)
    g2_sum         total teacher gradient-mass
    g2_max         peak teacher gradient-mass
    g2_top32_mass  fraction of the map's mass inside its own Top-32 -- how
                   concentrated the supervision is
    h4_norm        mean |h4| over the image's visual tokens
    top1_token     the token the P1 objective maximised; its distribution is a
                   proxy for answer format (OCRBench answers are not DocVQA's)

Reported as a two-sample comparison per benchmark and pooled, with a KS statistic
and a rank-biserial effect size, so "comparable" is measured rather than assumed.
CPU only; reads the caches.

Usage
    python m1_pool_check.py --tag m1
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                              # noqa: E402
from s2c3_common import TEACHER                                       # noqa: E402
from m1_common import (DS_ORDER, EXTRA_TEACHER, M1Store, dump,        # noqa: E402
                       load_m1_plan)


def ks_stat(a, b):
    """Two-sample Kolmogorov-Smirnov statistic (no scipy dependency)."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    grid = np.concatenate([a, b])
    ca = np.searchsorted(a, grid, side="right") / len(a)
    cb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(ca - cb).max())


def ks_critical(n1, n2, alpha=0.05):
    """Two-sample KS critical value, asymptotic formula 1.36*sqrt((n1+n2)/(n1*n2))."""
    return float(1.36 * np.sqrt((n1 + n2) / (n1 * n2)))


def rank_biserial(a, b):
    """P(a > b) - P(a < b), estimated from ranks; 0 == no shift."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    allv = np.concatenate([a, b])
    r = np.argsort(np.argsort(allv, kind="stable"), kind="stable").astype(float)
    ra = r[:len(a)].sum()
    u = ra - len(a) * (len(a) - 1) / 2.0
    return float(2.0 * u / (len(a) * len(b)) - 1.0)


def per_image(keys, G, stats_rows):
    seq = {r["key"]: r for r in stats_rows}
    out = []
    for k in keys:
        g = np.asarray(G[k], dtype=np.float64)
        o = np.argsort(-g)
        tot = g.sum()
        row = dict(key=k, g2_sum=float(tot), g2_max=float(g.max()),
                   g2_top32_mass=float(g[o[:32]].sum() / tot) if tot > 0 else 0.0)
        if k in seq:
            row["seq_len"] = int(seq[k]["seq_len"])
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m1")
    args = ap.parse_args()

    _, plan, keys, rows_of = load_m1_plan()
    store = M1Store()
    G = {}
    for f in (TEACHER, EXTRA_TEACHER):
        z = np.load(os.path.join(OUT, f))
        for k in z.files:
            G[k] = z[k]

    # h4 norm per image, read off the combined row space (no model needed)
    fit_rows = rows_of["fit"]
    n240 = [keys[i] for i in fit_rows]
    extra_rows = [i for ds in DS_ORDER for i in rows_of["extra"][ds]]
    ext = [keys[i] for i in extra_rows]

    seq_meta = json.load(open(os.path.join(OUT, "m1_features_extra.json")))["instances"]
    A = per_image(n240, G, [])
    B = per_image(ext, G, seq_meta)
    for r, row in zip(extra_rows, B):
        row["h4_norm"] = float(np.abs(np.asarray(store.rows([r])[0],
                                                 dtype=np.float32)).mean())
    for r, row in zip(fit_rows, A):
        row["h4_norm"] = float(np.abs(np.asarray(store.rows([r])[0],
                                                 dtype=np.float32)).mean())

    cols = ("g2_sum", "g2_max", "g2_top32_mass", "h4_norm")
    rep = {"n_original_fit": len(A), "n_extension": len(B),
           "columns": {}, "groups": {}}

    def block(name, a_rows, b_rows):
        d = {}
        for c in cols:
            a = [r[c] for r in a_rows if c in r]
            b = [r[c] for r in b_rows if c in r]
            if not a or not b:
                continue
            crit = ks_critical(len(a), len(b))
            d[c] = dict(mean_original=float(np.mean(a)), mean_extension=float(np.mean(b)),
                        median_original=float(np.median(a)),
                        median_extension=float(np.median(b)),
                        ks=ks_stat(a, b), ks_critical_5pct=crit,
                        ks_significant=bool(ks_stat(a, b) > crit),
                        n_original=len(a), n_extension=len(b),
                        rank_biserial=rank_biserial(b, a),
                        rel_shift=float((np.mean(b) - np.mean(a))
                                        / (abs(np.mean(a)) + 1e-12)))
        rep["columns"][name] = d
        print(f"[pool {name}]")
        for c, v in d.items():
            print(f"  {c:14s} orig={v['mean_original']:.4g} "
                  f"ext={v['mean_extension']:.4g}  rel={v['rel_shift']:+.3f}  "
                  f"KS={v['ks']:.3f} (crit {v['ks_critical_5pct']:.3f}, "
                  f"sig={v['ks_significant']})  "
                  f"rank-biserial={v['rank_biserial']:+.3f}")
        rep.setdefault("any_significant", {})[name] = any(
            v["ks_significant"] for v in d.values())

    block("all", A, B)
    for i, ds in enumerate(DS_ORDER):
        a = [r for r in A if r["key"].startswith(ds + "_")]
        b = [r for r in B if r["key"].startswith(ds + "_")]
        block(ds, a, b)

    # the extension is used as a nested prefix, so the first 80/ds (which feed
    # n=480) are compared separately from the last 160/ds (which feed only 960)
    prefixes, tails = {}, {}
    for ds in DS_ORDER:
        sel = [keys[i] for i in rows_of["extra"][ds]]
        n = len(sel)
        byk = {r["key"]: r for r in B}
        prefixes[ds] = [byk[k] for k in sel[:80]]
        tails[ds] = [byk[k] for k in sel[80:]]
    block("extension_first80_n480", A, [r for ds in DS_ORDER for r in prefixes[ds]])
    block("extension_last160_n960", A, [r for ds in DS_ORDER for r in tails[ds]])

    # answer-format proxy: how many distinct top-1 tokens, and the entropy
    ref = json.load(open(os.path.join(OUT, "s2b_gradient_scores_meta.json")))
    em = json.load(open(os.path.join(OUT, "m1_gradient_scores_extra_meta.json")))
    for name, meta, ks in (("original_fit", ref, n240), ("extension", em, ext)):
        toks = [meta[k]["top1_token_id"] for k in ks if k in meta]
        _, cnt = np.unique(toks, return_counts=True)
        p = cnt / cnt.sum()
        rep.setdefault("top1", {})[name] = dict(
            n=len(toks), n_distinct=int(len(cnt)),
            entropy_bits=float(-(p * np.log2(p)).sum()))
        print(f"[top1 {name}] n={len(toks)} distinct={len(cnt)} "
              f"entropy={rep['top1'][name]['entropy_bits']:.3f} bits")

    dump(f"{args.tag}_pool_check.json", rep)
    print("any quantity significantly different at 5 %: "
          f"{rep['any_significant']}")


if __name__ == "__main__":
    main()
