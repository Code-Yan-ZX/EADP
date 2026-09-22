"""
S2-C4 step 1: build and cache the per-image oracle directions.

For every instance with a teacher map (240 fit + 60 val + 150 held-out = 450; the
15 causal instances have no P1-G2 map and are excluded -- S2-C1's rule, unchanged):

  w32_i  mean-difference direction of the teacher's Top-32, unit-normalised
  w8_i   the same for the teacher's Top-8 (supplementary)

plus the cheap per-image descriptors used later to *select* a direction without
reading that image's teacher map:

  mean(h_L4)                 4096
  std(h_L4)                  4096
  mean+std                   8192
  meanstd+delta             12288   (delta = mean(h_L4) - mean(h_L2), S2-C1 cache)

and the fit-split standardisation statistics (mu, sd) that every quantity in this
stage is expressed in.

This step is the only one that touches the 3.9 GB feature memmap in full; the
artifacts it writes are ~40 MB, and every later step is CPU-only and reads them.

Sanity checks run here, printed and stored:
  * the in-sample recall of each oracle direction, which must reproduce S2-C3 §6
    (w32 -> 0.903, w8 -> 0.982) -- if it does not, this stage is not comparable
    to the one it is diagnosing;
  * descriptor finiteness and the fraction of near-zero-norm directions.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                            # noqa: E402
from s2c4_common import (DESCRIPTORS, DIRS, FEATS, HEAD_KS, K_HEAD,  # noqa: E402
                         K_HEAD8, LAYER, descriptors, direction, dump,
                         load_splits, metrics_single, standardizer,
                         teacher_orders)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c4")
    args = ap.parse_args()

    meta, plan, keys, rows_of = load_plan_rows()
    G = np.load(os.path.join(OUT, "s2b_gradient_scores.npz"))
    orders = teacher_orders(G, keys)
    H = np.load(os.path.join(OUT, FEATS), mmap_mode="r")
    H2 = np.load(os.path.join(OUT, f"s2c1_feats_L2.npy"), mmap_mode="r")

    # every instance whose teacher map exists; the causal 15 silently drop out
    have = np.array([k in G.files for k in keys])
    rows = np.nonzero(have)[0]
    split = np.array([plan[keys[i]]["split"] for i in rows])
    ds = np.array([plan[keys[i]]["ds"] for i in rows])
    print(f"instances with a teacher map: {len(rows)} "
          f"({dict(zip(*np.unique(split, return_counts=True)))})")

    t0 = time.time()
    mu, sd = standardizer(H, rows_of["fit"])
    # layer 2 gets its own statistics -- the delta descriptor subtracts two means
    # that only mean the same thing if each is expressed in its own standardisation
    mu2, sd2 = standardizer(H2, rows_of["fit"])
    print(f"[{time.time()-t0:6.1f}s] fit-split standardisation statistics (L2, L4)")

    d = H.shape[-1]
    W32 = np.zeros((len(rows), d), np.float32)
    W8 = np.zeros((len(rows), d), np.float32)
    desc = {name: [] for name in DESCRIPTORS}
    in_sample = []
    t0 = time.time()
    for n, i in enumerate(rows):
        key = keys[i]
        h = (H[i].astype(np.float32) - mu) / sd
        h2 = (H2[i].astype(np.float32) - mu2) / sd2
        w32 = direction(h, orders[key][:K_HEAD])
        w8 = direction(h, orders[key][:K_HEAD8])
        W32[n], W8[n] = w32, w8
        for name, v in descriptors(h, h2).items():
            desc[name].append(v)
        m8 = metrics_single(h @ w8, orders[key])
        in_sample.append(dict(key=key, r8=m8["head_recall8"],
                              r32=metrics_single(h @ w32,
                                                 orders[key])["head_recall32"],
                              agree8=m8["head_agree8"]))
        if (n + 1) % 90 == 0:
            print(f"[{time.time()-t0:6.1f}s] {n+1}/{len(rows)} directions")

    desc = {k: np.stack(v).astype(np.float32) for k, v in desc.items()}
    isr8 = np.array([r["r8"] for r in in_sample])
    isr32 = np.array([r["r32"] for r in in_sample])
    isa8 = np.array([r["agree8"] for r in in_sample])

    # ---- sanity: reproduce S2-C3 §6 on the held-out 150 -------------------
    te = split == "test"
    sanity = {
        "n": int(te.sum()),
        "in_sample_recall8": float(isr8[te].mean()),
        "in_sample_recall32": float(isr32[te].mean()),
        "in_sample_agree8": float(isa8[te].mean()),
        "s2c3_reference": {"recall8": 0.9817, "recall32": 0.9025, "agree8": 0.4267},
        "frac_zero_direction": float(np.mean([np.linalg.norm(w) < 1e-6
                                              for w in W32])),
    }
    print(f"\nin-sample oracle on the held-out {int(te.sum())} "
          f"(S2-C3 §6 reference in brackets):")
    print(f"  Top-32 direction -> recall@256 of teacher Top-32 = "
          f"{sanity['in_sample_recall32']:.4f}  [0.9025]")
    print(f"  Top-8  direction -> recall@256 of teacher Top-8  = "
          f"{sanity['in_sample_recall8']:.4f}  [0.9817]")
    print(f"  Top-8  direction -> exact Top-8 agreement        = "
          f"{sanity['in_sample_agree8']:.4f}  [0.4267]")

    np.savez_compressed(
        os.path.join(OUT, DIRS),
        W32=W32, W8=W8, mu=mu, sd=sd,
        rows=rows, split=split, ds=ds,
        keys=np.array([keys[i] for i in rows]),
        **{f"desc::{k}": v for k, v in desc.items()},
    )
    print(f"\n[saved] {DIRS}  ({os.path.getsize(os.path.join(OUT, DIRS))/1e6:.1f} MB)")

    dump(f"{args.tag}_build.json", dict(
        note=__doc__, n_instances=int(len(rows)),
        split_counts={s: int((split == s).sum()) for s in np.unique(split)},
        descriptor_dims={k: int(v.shape[1]) for k, v in desc.items()},
        sanity=sanity,
        in_sample_per_instance={k: r for k, r in zip(
            [keys[i] for i in rows], in_sample)} if len(rows) < 20 else None,
        in_sample_by_split={
            s: dict(
                recall8=float(isr8[split == s].mean()),
                recall32=float(isr32[split == s].mean()),
                agree8=float(isa8[split == s].mean()),
                n=int((split == s).sum()))
            for s in ("fit", "val", "test")},
        head_ks=list(HEAD_KS), k_head=K_HEAD, layer=LAYER,
    ))


def load_plan_rows():
    return load_splits()


if __name__ == "__main__":
    main()
