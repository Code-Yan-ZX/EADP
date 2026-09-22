"""
S2-C1 step 3: teacher separability of the learned scorers on the held-out 150,
plus the causal-15 supplementary tier.

Tier 1 (the one that matters). On the held-out 150 -- which no selection decision
in this stage ever touched -- how well does each score predict the P1-G2
teacher's Top-256 membership?

    AUROC, AP, Top-256 overlap, recall of the teacher's Top-256,
    per-image Spearman (diagnostic only)

Compared against: the S2-C0 attention proxies A_L2 / C_L2, the official EADP
importance score, and a seeded random map. Per the brief, Spearman is reported
but never used to select: S2-C0 showed the layers that best agree with the
teacher are the ones that are least useful.

Tier 2 (supplementary, never gates). The S2-A causal metrics on the 15 causal
cases -- mean necessary rank, R@256, block AUROC -- for interpretation only.
S2-C0 already showed n=8 causal AUROC has no resolution for downstream accuracy.
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import BLOCK, OUT
from scoring_search_s1 import evaluate

KS = (128, 256, 512)
DS_ORDER = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def separability(score, y):
    """AUROC / AP / Top-256 overlap / recall against a hard Top-256 membership."""
    order = np.argsort(-score, kind="stable")
    truth = np.nonzero(y > 0.5)[0]
    inter = len(set(order[:256].tolist()) & set(truth.tolist()))
    r = np.argsort(np.argsort(score, kind="stable"), kind="stable").astype(np.float64) + 1
    n_pos, n_neg = len(truth), len(y) - len(truth)
    auroc = (r[truth].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    hit = np.isin(order, truth)
    prec = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    return dict(auroc=float(auroc), ap=float((prec * hit).sum() / n_pos),
                overlap256=inter / 256.0, recall256=inter / float(n_pos))


def seeded_random(key, n=1024):
    """Deterministic per-instance random map (zlib.crc32, not hash(): PYTHONHASHSEED
    randomizes str hashing per process, which would make the control unrepeatable)."""
    return np.random.default_rng(zlib.crc32(key.encode()) % (2 ** 31)).random(n)


def load_arms(keys, learned, Z, P0, OFF):
    """arm -> {key: score vector}, for every key available."""
    arms = {a: {} for a in learned + ["A_L2", "C_L2", "official", "random"]}
    for k in keys:
        for a in learned:
            kk = f"{a}__{k}"
            if kk in Z.files:
                arms[a][k] = Z[kk].astype(np.float64)
        for a in ("A_L2", "C_L2"):
            kk = f"{a}__{k}"
            if kk in P0.files:
                arms[a][k] = P0[kk].astype(np.float64)
        if f"imp__{k}" in OFF.files:
            arms["official"][k] = OFF[f"imp__{k}"].astype(np.float64)
        arms["random"][k] = seeded_random(k)
    return arms


def tier1(arms, G, keys):
    """Held-out separability, per instance then averaged (never pooling tokens)."""
    out = {}
    for arm, d in arms.items():
        kk = [k for k in keys if k in d]
        if not kk:
            continue
        y = {k: targets(G[k]) for k in kk}
        per_img = {k: separability(d[k], y[k]) for k in kk}
        sp = [spearman(d[k], G[k].astype(np.float64)) for k in kk]
        agg = {m: float(np.mean([per_img[k][m] for k in kk]))
               for m in ("auroc", "ap", "overlap256", "recall256")}
        agg["spearman"] = float(np.mean(sp))
        for ds in DS_ORDER:
            sub = [k for k in kk if k.startswith(ds)]
            if sub:
                agg[f"auroc|{ds}"] = float(np.mean([per_img[k]["auroc"] for k in sub]))
                agg[f"overlap256|{ds}"] = float(
                    np.mean([per_img[k]["overlap256"] for k in sub]))
        agg["n"] = len(kk)
        out[arm] = agg
    return out


def targets(g):
    y = np.zeros(1024, dtype=np.float32)
    y[np.argsort(-g.astype(np.float64), kind="stable")[:256]] = 1.0
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c1")
    args = ap.parse_args()

    feat = json.load(open(os.path.join(OUT, f"{args.tag}_features.json")))
    plan = {p["key"]: p for p in feat["plan"]}
    all_keys = sorted(plan)
    held = sorted([k for k, p in plan.items() if p["split"] == "test"])
    causes = sorted([k for k, p in plan.items() if p["causal"]])

    G = np.load(os.path.join(OUT, "s2b_gradient_scores.npz"))
    Z = np.load(os.path.join(OUT, f"{args.tag}_scores.npz"))
    learned = sorted({k.split("__", 1)[0] for k in Z.files})
    P0 = np.load(os.path.join(OUT, "s2c0_forward_proxy.npz"))
    OFF = np.load(os.path.join(OUT, "s2b_official_selection.npz"))

    arms_all = load_arms(all_keys, learned, Z, P0, OFF)
    held = [k for k in held if np.all([k in arms_all[a] for a in
                                       ("A_L2", "C_L2", "official")])]
    print(f"[tier 1] {len(held)} held-out instances with every baseline present")
    print(f"[tier 2] {len(causes)} causal cases")

    T1 = tier1(arms_all, G, held)
    print("\n" + "=" * 106)
    print(f"TIER 1 -- P1-G2 Top-256 separability on the held-out {len(held)}"
          f"   (chance Top-256 overlap = 0.250)")
    print(f"{'arm':10s}{'AUROC':>9s}{'AP':>9s}{'Top256':>9s}{'Recall':>9s}{'Spearman':>10s}"
          f"{'| TextVQA':>11s}{'DocVQA':>9s}{'OCR':>8s}")
    for a in sorted(T1, key=lambda x: -T1[x]["auroc"]):
        t = T1[a]
        print(f"{a:10s}{t['auroc']:9.3f}{t['ap']:9.3f}{t['overlap256']:9.3f}"
              f"{t['recall256']:9.3f}{t['spearman']:10.3f}"
              f"{t.get('auroc|TextVQA_VAL', float('nan')):11.3f}"
              f"{t.get('auroc|DocVQA_VAL', float('nan')):9.3f}"
              f"{t.get('auroc|OCRBench', float('nan')):8.3f}")

    # --------------------------------------------------------------- tier 2
    S2A = json.load(open(os.path.join(OUT, "s2a_gradient_viability.json")))["cases"]
    needed = {k: np.array(S2A[k]["needed"]) for k in causes}
    nnec = {k: int(len(needed[k]) * BLOCK * BLOCK) for k in causes}
    prim = [k for k in causes if nnec[k] <= 256]
    print(f"[causal] primary tier nNec<=256: {len(prim)} cases "
          f"(supplementary -- does not gate)")

    cmetrics = {}
    for k in causes:
        rec = dict(needed=needed[k])
        cmetrics[k] = {}
        for a, d in arms_all.items():
            if k in d and a != "random":
                cmetrics[k][a] = evaluate(rec, d[k])
        for ref in ("official", "P1_G2"):
            cmetrics[k][ref] = S2A[k]["metrics"][ref]

    def agg(method, field, ks):
        v = np.array([cmetrics[k][method][field] for k in ks], dtype=float)
        o = np.array([cmetrics[k]["official"][field] for k in ks], dtype=float)
        return dict(n=len(ks), mean=float(np.nanmean(v)),
                    delta=float(np.nanmean(v - o)))

    T2 = {}
    print(f"\n{'method':10s}{'rank':>9s}{'delta':>9s}{'AUROC':>8s}{'AP':>8s}"
          + "".join(f"{'R@' + str(k):>8s}" for k in KS))
    for m in ["official", "P1_G2"] + learned + ["A_L2", "C_L2"]:
        ks = [k for k in prim if m in cmetrics[k]]
        if not ks:
            continue
        r = agg(m, "mean_rank_pct", ks)
        T2[m] = dict(rank=r, auroc=agg(m, "block_auroc", ks), ap=agg(m, "block_ap", ks),
                     recall={k: agg(m, f"recall@{k}", ks) for k in KS})
        print(f"{m:10s}{r['mean']:9.4f}{r['delta']:+9.4f}"
              f"{T2[m]['auroc']['mean']:8.3f}{T2[m]['ap']['mean']:8.3f}"
              + "".join(f"{T2[m]['recall'][k]['mean']:8.3f}" for k in KS))

    json.dump({"held_out": dict(n=len(held), keys=held, arms=T1),
               "causal": dict(n=len(causes), primary=prim, methods=T2),
               "learned_arms": learned},
              open(os.path.join(OUT, f"{args.tag}_eval.json"), "w"), indent=1)
    print(f"\n[saved] {os.path.join(OUT, f'{args.tag}_eval.json')}")


if __name__ == "__main__":
    main()
