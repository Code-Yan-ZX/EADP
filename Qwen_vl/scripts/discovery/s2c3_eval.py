"""
S2-C3 step 2: head-oriented separability on the held-out 150.

Top-256 overlap is reported but demoted: the stage's primary metrics are the
recall of the teacher's Top-8/16/32/64 inside the selected 256 (what S2-C2 showed
the downstream value is paid in) and the exact Top-8/16/32 agreement (how close
the student's own ordering is to the teacher's at the head).

Every arm is scored on the identical 150 instances, and the cached S2-C1 LIN_L4
scores are recomputed through the same code path so that this stage's baseline
row is provably the same object S2-C2 measured (head_recall@8 = 0.736,
head_agree@8 = 0.192).
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                   # noqa: E402
from s2c3_common import (DS_ALL, N_VIS, TEACHER, aggregate,  # noqa: E402
                         head_metrics, load_plan, teacher_orders)

METRICS = (["overlap256", "auroc", "ap", "spearman"]
           + [f"head_recall{k}" for k in (8, 16, 32, 64)]
           + [f"head_miss{k}" for k in (8, 16, 32, 64)]
           + [f"head_agree{k}" for k in (8, 16, 32)])


def seeded_random(key, n=N_VIS):
    """Deterministic per-instance random map (crc32, not hash(): PYTHONHASHSEED)."""
    return np.random.default_rng(zlib.crc32(key.encode()) % (2 ** 31)).random(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c3")
    args = ap.parse_args()

    meta, plan, keys, rows_of = load_plan()
    held = [keys[i] for i in rows_of["test"]]
    G = np.load(os.path.join(OUT, TEACHER))
    orders = teacher_orders(G, keys)

    Z = np.load(os.path.join(OUT, f"{args.tag}_scores.npz"))
    C1 = np.load(os.path.join(OUT, "s2c1_scores.npz"))
    P0 = np.load(os.path.join(OUT, "s2c0_forward_proxy.npz"))
    OFF = np.load(os.path.join(OUT, "s2b_official_selection.npz"))

    # arm -> {key: score vector}
    arms = {}
    for k in Z.files:
        tag, key = k.split("__", 1)
        arms.setdefault(tag, {})[key] = Z[k].astype(np.float64)
    # the exact S2-C1 baseline, read from its own cache
    arms["S2C1_LIN_L4"] = {k.split("__", 1)[1]: C1[k].astype(np.float64)
                           for k in C1.files if k.startswith("LIN_L4__")}
    for a in ("A_L2", "C_L2"):
        arms[a] = {k.split("__", 1)[1]: P0[k].astype(np.float64)
                   for k in P0.files if k.startswith(a + "__")}
    arms["official"] = {k.split("__", 1)[1]: OFF[k].astype(np.float64)
                        for k in OFF.files if k.startswith("imp__")}
    arms["P1G2_teacher"] = {k: G[k].astype(np.float64) for k in G.files}
    arms["random"] = {k: seeded_random(k) for k in keys}

    held = [k for k in held if all(k in arms[a] for a in
                                   ("S2C1_LIN_L4", "A_L2", "C_L2", "official"))]
    tags = [t for t in arms if all(k in arms[t] for k in held)]
    print(f"[held-out] {len(held)} instances; {len(tags)} arms with full coverage")

    per = {t: {k: head_metrics(arms[t][k], orders[k]) for k in held} for t in tags}

    table = {}
    for t in tags:
        agg = aggregate([per[t][k] for k in held])
        for ds in DS_ALL:
            sub = [per[t][k] for k in held if k.startswith(ds)]
            for m in ("head_recall8", "head_recall32", "head_agree8", "overlap256"):
                agg[f"{m}|{ds}"] = float(np.mean([r[m] for r in sub]))
        agg["n"] = len(held)
        table[t] = agg

    # ---- identity check: the recomputed baseline must be S2-C2's numbers ----
    base = table["S2C1_LIN_L4"]
    print(f"\n[identity] S2C1_LIN_L4 recomputed here: "
          f"overlap256={base['overlap256']:.4f} head_recall8={base['head_recall8']:.4f} "
          f"head_agree8={base['head_agree8']:.4f}")
    print(f"[identity] S2-C2 measured:              "
          f"overlap256=0.5620 head_recall8=0.7360 head_agree8=0.1920")

    # ---- the table -----------------------------------------------------------
    hdr = (f"{'arm':14s}{'ov256':>8s}{'R@8':>8s}{'R@16':>8s}{'R@32':>8s}{'R@64':>8s}"
           f"{'A@8':>8s}{'A@16':>8s}{'A@32':>8s}{'miss8':>8s}{'AUROC':>8s}")
    print("\n" + "=" * len(hdr))
    print("HELD-OUT 150 -- teacher head retention inside the selected 256 "
          "(R@k) and exact top-k agreement (A@k)")
    print(hdr)
    order_tags = sorted(table, key=lambda t: -table[t]["head_recall8"])
    for t in order_tags:
        v = table[t]
        print(f"{t:14s}{v['overlap256']:8.3f}{v['head_recall8']:8.3f}"
              f"{v['head_recall16']:8.3f}{v['head_recall32']:8.3f}"
              f"{v['head_recall64']:8.3f}{v['head_agree8']:8.3f}"
              f"{v['head_agree16']:8.3f}{v['head_agree32']:8.3f}"
              f"{v['head_miss8']:8.2f}{v['auroc']:8.3f}")

    # ---- per-instance arrays for the paired bootstrap ------------------------
    npz = {m: np.array([[per[t][k][m] for k in held] for t in tags]) for m in METRICS}
    np.savez_compressed(os.path.join(OUT, f"{args.tag}_headmetrics.npz"),
                        keys=np.array(held), tags=np.array(tags), **npz)

    json.dump({"n": len(held), "keys": held, "tags": tags, "table": table,
               "identity_check": {"recomputed": {m: base[m] for m in
                                                 ("overlap256", "head_recall8",
                                                  "head_agree8")},
                                  "s2c2_measured": {"overlap256": 0.562,
                                                    "head_recall8": 0.736,
                                                    "head_agree8": 0.192}}},
              open(os.path.join(OUT, f"{args.tag}_eval.json"), "w"), indent=1)
    print(f"\n[saved] {args.tag}_eval.json + {args.tag}_headmetrics.npz")


if __name__ == "__main__":
    main()
