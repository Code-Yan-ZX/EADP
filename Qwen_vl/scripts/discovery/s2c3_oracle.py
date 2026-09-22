"""
S2-C3 supplementary: can the layer-4 token representation express the teacher's
head at all?

This is a diagnostic, not an arm -- nothing here is trained, nothing is selected
on, and the verdict does not depend on it. It exists because the trained arms
answer "can a *shared* linear direction, learned from 240 images, put the
teacher's head tokens on top", which conflates two very different failures:

  (i)  the token-local layer-4 representation does not contain the head
       information, so no linear read-out can express it (a representation wall);
  (ii) it does contain it, but one direction fitted across images cannot
       transfer to a new image (a generalisation wall).

The test uses the simplest closed-form linear read-out there is -- the
mean-difference (one-shot centroid) direction -- fitted **on the same image**,
which is deliberately optimistic. Three quantities:

  A  recall(teacher Top-8  |  w = teacher-head direction)      in-sample, the
     head scored with a direction centred on itself
  B  recall(random  8      |  w = that random set's direction) in-sample control,
     same procedure, arbitrary 8 tokens (5 seeds)
  C  recall(teacher Top-8  |  w = a random set's direction)     cross control

A well above C says centring on the head genuinely pulls the head up. A far below
B says the teacher's own best 8 tokens are *less* linearly self-consistent than
an arbitrary group of 8 -- which is the representation wall. A comparable to B
says the head is as expressible as anything, and the failure of the trained
scorer is one of transfer, not of representation.
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                    # noqa: E402
from s2c3_common import LAYER, TEACHER, load_plan, teacher_orders  # noqa: E402

BUDGET = 256
N_SEEDS = 5


def recall_in_topk(score, target_idx, k=BUDGET):
    """Fraction of `target_idx` that lands inside the score's top-k."""
    order = np.argsort(-score, kind="stable")[:k]
    tgt = np.asarray(target_idx)
    return len(set(order.tolist()) & set(tgt.tolist())) / float(len(tgt))


def direction(h, pos):
    """Mean-difference direction: mean of `pos` minus mean of everything else."""
    mask = np.zeros(len(h), dtype=bool)
    mask[pos] = True
    return h[mask].mean(0) - h[~mask].mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c3")
    args = ap.parse_args()

    meta, plan, keys, rows_of = load_plan()
    held = [keys[i] for i in rows_of["test"]]
    G = np.load(os.path.join(OUT, TEACHER))
    orders = teacher_orders(G, keys)
    H = np.load(os.path.join(OUT, f"s2c1_feats_L{LAYER}.npy"), mmap_mode="r")

    # the same fixed preprocessing the scorers use (fit-split mean/std)
    fit_rows = rows_of["fit"]
    s1 = np.zeros(H.shape[-1], dtype=np.float64)
    s2 = np.zeros(H.shape[-1], dtype=np.float64)
    n = 0
    for j in fit_rows:
        x = H[j].astype(np.float32)
        s1 += x.sum(axis=0, dtype=np.float64)
        s2 += np.square(x, dtype=np.float64).sum(axis=0)
        n += x.shape[0]
    mu = (s1 / n).astype(np.float32)
    sd = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 1e-12)).astype(np.float32)

    rows = {k: None for k in held}
    for i in rows_of["test"]:
        rows[keys[i]] = i

    out = {"A_teacher_dir_recall_teacher_head8": [], "A_agree8": [],
           "B_random_dir_recall_own8": [], "C_random_dir_recall_teacher_head8": [],
           "A32_teacher_dir_recall_teacher_head32": [], "B32_own32": []}
    rng = np.random.default_rng(0)
    for key in held:
        h = (H[rows[key]].astype(np.float32) - mu) / sd
        head8 = orders[key][:8]
        head32 = orders[key][:32]

        w = direction(h, head8)
        sc = h @ w
        out["A_teacher_dir_recall_teacher_head8"].append(recall_in_topk(sc, head8))
        o = np.argsort(-sc, kind="stable")[:8]
        out["A_agree8"].append(len(set(o.tolist()) & set(head8.tolist())) / 8.0)

        sc32 = h @ direction(h, head32)
        out["A32_teacher_dir_recall_teacher_head32"].append(recall_in_topk(sc32, head32))

        for _ in range(N_SEEDS):
            r = rng.choice(1024, 8, replace=False)
            w = direction(h, r)
            scr = h @ w
            out["B_random_dir_recall_own8"].append(recall_in_topk(scr, r))
            out["C_random_dir_recall_teacher_head8"].append(
                recall_in_topk(scr, head8))
            r32 = rng.choice(1024, 32, replace=False)
            out["B32_own32"].append(recall_in_topk(h @ direction(h, r32), r32))

    res = {k: dict(mean=float(np.mean(v)), std=float(np.std(v)), n=len(v))
           for k, v in out.items()}
    res["head_recall8_of_the_trained_BASE"] = None

    print("mean-difference (one-shot centroid) read-out of layer-4 hidden states,")
    print("fitted IN-SAMPLE on the held-out 150 (optimistic by construction):\n")
    print(f"  A   teacher's Top-8, teacher-head direction   "
          f"recall@256 = {res['A_teacher_dir_recall_teacher_head8']['mean']:.4f}   "
          f"top-8 agreement = {res['A_agree8']['mean']:.4f}")
    print(f"  B   arbitrary 8, own direction ({N_SEEDS} seeds)     "
          f"recall@256 = {res['B_random_dir_recall_own8']['mean']:.4f} "
          f"± {res['B_random_dir_recall_own8']['std']:.4f}")
    print(f"  C   teacher's Top-8, an arbitrary-8 direction "
          f"recall@256 = {res['C_random_dir_recall_teacher_head8']['mean']:.4f} "
          f"± {res['C_random_dir_recall_teacher_head8']['std']:.4f}")
    print(f"  A32 teacher's Top-32, teacher-head direction  "
          f"recall@256 = {res['A32_teacher_dir_recall_teacher_head32']['mean']:.4f}")
    print(f"  B32 arbitrary 32, own direction               "
          f"recall@256 = {res['B32_own32']['mean']:.4f} "
          f"± {res['B32_own32']['std']:.4f}")

    json.dump({"note": __doc__, "n": len(held), "seeds": N_SEEDS, "results": res,
               "per_instance": out},
              open(os.path.join(OUT, f"{args.tag}_oracle.json"), "w"), indent=1)
    print(f"\n[saved] {args.tag}_oracle.json")


if __name__ == "__main__":
    main()
