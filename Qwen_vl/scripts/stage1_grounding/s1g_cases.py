"""Stage-1 grounding discovery — mechanism analysis (report §10).

From the per-sample diag jsonl (tokens + entropy + grounding stat + kept
mask) produced by s1g_accuracy --diag:

 1. Population view: correlation between entropy and grounding stat over
    all captured tokens; how often the grounding stat would flip the
    official near-one-hot entropy choice inside the kept set.
 2. Canonical cases: tokens where entropy is HIGH but grounding is STRONG
    (entropy blind spot) and tokens where entropy is LOW but grounding is
    WEAK (dispersion without evidence) — with corpus-wide frequencies.
 3. Failure/success cases at the question level: samples where the new
    arm's keep set diverges most from BASE AND the question outcome
    flipped (rescued / broken), grounded in the diag token stats.

Usage: python s1g_cases.py --diag <path.jsonl> --arm B_G1 [--out cases_B_G1.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import s1g_common as SC

STOPWORDS = {"the", "a", "an", "of", "and", "to", "in", "is", "are", "this",
             "that", "please", "using", "with", "for", "on", "it", "as"}


def load_diag(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def population(rows):
    """Correlation between entropy(-log weight) and grounding z; flip rate."""
    ent_all, g_all, flip = [], [], 0
    n_sets = 0
    for r in rows:
        H = r.get("H")
        g = r.get("g")
        kept = r.get("kept")
        if H is None or g is None or not kept:
            continue
        H = np.array(H)[kept]
        g = np.array(g)[kept]
        if len(kept) < 2 or H.std() < 1e-9:
            continue
        ent_all.append(H)
        g_all.append(g)
        n_sets += 1
        # official near-one-hot choice = argmin H; grounding choice = argmax g
        if int(np.argmin(H)) != int(np.argmax(g)):
            flip += 1
    if not ent_all:
        return dict(n_sets=0)
    H = np.concatenate(ent_all)
    g = np.concatenate(g_all)
    return dict(
        n_sets=n_sets,
        n_tokens=int(H.size),
        corr_ent_g=float(np.corrcoef(H, g)[0, 1]),
        flip_rate=float(flip / n_sets))


def canonical(rows, topn=12):
    """Tokens extreme on (entropy, grounding) within kept sets."""
    hi_H_hi_g, hi_H_lo_g, lo_H_lo_g = [], [], []
    for r in rows:
        H = r.get("H")
        g = r.get("g")
        kept = r.get("kept")
        toks = r.get("tokens")
        if H is None or g is None or not kept or not toks:
            continue
        H = np.array(H)
        g = np.array(g)
        # z-score across ALL tokens for comparability across samples
        if g.std() < 1e-9 or H.std() < 1e-9:
            continue
        gz = (g - g.mean()) / g.std()
        for j in kept:
            t = toks[j].strip().lower() if j < len(toks) else ""
            if not t:
                continue
            rec = dict(token=toks[j], H=float(H[j]), g=float(g[j]),
                       ds=r["ds"], idx=r["idx"])
            if H[j] > np.median(H) + 0.5 * H.std() and gz[j] > 1.0:
                hi_H_hi_g.append(rec)     # entropy blind spot
            if gz[j] < -1.0 and H[j] < np.median(H) - 0.5 * H.std():
                lo_H_lo_g.append(rec)     # dispersion without evidence
    def agg(lst):
        by = {}
        for r in lst:
            k = r["token"].strip().lower()
            by.setdefault(k, []).append(r["g"])
        return sorted(((k, len(v), float(np.mean(v))) for k, v in by.items()),
                      key=lambda x: (-x[1], -x[2]))
    return dict(
        hi_entropy_strong_grounding=dict(count=len(hi_H_hi_g),
                                         top_tokens=agg(hi_H_hi_g)[:topn]),
        lo_entropy_weak_grounding=dict(count=len(lo_H_lo_g),
                                       top_tokens=agg(lo_H_lo_g)[:topn]))


def outcome_flips(arm, split="dev", tag=""):
    """Question-level rescued/broken between arm and BASE with keep-set
    divergence, from shards + score jsons."""
    import amp_analyze as AA
    import amp_common as AC
    scores = {}
    for ds in SC.DS_LIST:
        for a, base in ((arm, None), ("BASE", "BASE")):
            if a == "BASE":
                hits = sorted(glob.glob(os.path.join(
                    AC.OUT_DIR, "acc", split, "BASE", "K*",
                    f"{ds}_score.json")))
            else:
                hits = sorted(glob.glob(os.path.join(
                    SC.ACC_DIR, split, a + (f"_{tag}" if tag else ""),
                    "K*", f"{ds}_score.json")))
            if hits:
                s = json.load(open(hits[-1]))
                scores[(a, ds)] = s.get("per_question", {})
    flips = dict(rescued=[], broken=[])
    for ds in SC.DS_LIST:
        pa = scores.get((arm, ds), {})
        pb = scores.get(("BASE", ds), {})
        for q in set(pa) & set(pb):
            if pa[q] > 0.5 >= pb[q]:
                flips["rescued"].append(dict(ds=ds, idx=int(q),
                                             score_a=pa[q], score_b=pb[q]))
            elif pb[q] > 0.5 >= pa[q]:
                flips["broken"].append(dict(ds=ds, idx=int(q),
                                            score_a=pa[q], score_b=pb[q]))
    return flips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_diag(args.diag)
    out = dict(arm=args.arm, diag=args.diag, n_samples=len(rows),
               population=population(rows), canonical=canonical(rows),
               outcome_flips=outcome_flips(args.arm, args.split, args.tag))
    p = args.out or os.path.join(SC.DIAG_DIR, f"cases_{args.arm}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    pop = out["population"]
    print(f"[{args.arm}] sets={pop.get('n_sets')} tokens={pop.get('n_tokens')} "
          f"corr(H,g)={pop.get('corr_ent_g')} "
          f"flip_rate={pop.get('flip_rate')}")
    print(f"  blind-spot tokens: {out['canonical']['hi_entropy_strong_grounding']['top_tokens'][:6]}")
    print(f"  no-evidence tokens: {out['canonical']['lo_entropy_weak_grounding']['top_tokens'][:6]}")
    print(f"  rescued={len(out['outcome_flips']['rescued'])} "
          f"broken={len(out['outcome_flips']['broken'])}")
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
