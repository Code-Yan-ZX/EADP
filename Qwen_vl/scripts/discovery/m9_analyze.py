"""
M9 Phase 3 analysis -- the bank150 screening tables.

Everything the brief asks the generation stage to answer:

  T1  Method | r | TextVQA | DocVQA | OCRBench | Macro | dOriginal-B2 | dD-B2
  T2  the decomposition chain Original-B2 -> D-B2 -> Random -> Cos -> Attn ->
      CMC -> Oracle, separating the TEMPORARY EXPOSURE effect (D-B2 - B2)
      from the ADJUDICATION effect (X - D-B2)
  T3  paired bootstrap contrasts (ATTN/CMC/ORACLE/COS/RND - D-B2, and CMC-ATTN)
  T4  paired flips vs D-B2
  G   the screening gate verdict (brief's two GO conditions + STOP conditions)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_gdep import OUTPUT_DIR                                        # noqa: E402
from m6_common import DS_ORDER                                        # noqa: E402

NBOOT = 6000
SEED = 20260927
ARMS_MAIN = ["DB2", "RND4", "RND8", "COS4", "COS8", "ATTN4", "ATTN8",
             "CMC4", "CMC8", "ORC4", "ORC8"]


def macro(rec):
    return float(np.mean([rec["per_benchmark"][ds]["acc_pct"]
                          for ds in DS_ORDER]))


def hits_vec(rec, ds):
    return np.array(rec["per_benchmark"][ds]["hits"], dtype=np.float64)


def macro_vec(rec):
    """per-instance macro contribution vector (mean of the 3 benchmarks'
    hits, aligned by key order within each benchmark block)."""
    per = [hits_vec(rec, ds) for ds in DS_ORDER]
    n = min(len(p) for p in per)
    return np.mean([p[:n] for p in per], axis=0)


def paired_bootstrap(a, b, seed=SEED, nboot=NBOOT):
    """macro delta a-b with paired CI over instances."""
    d = a - b
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(delta=float(d.mean()), ci=[float(lo), float(hi)],
                wins=int((d > 0).sum()), losses=int((d < 0).sum()))


def flips(rec_a, rec_b):
    """fixed/broken of a relative to b, per benchmark."""
    out = {}
    for ds in DS_ORDER:
        ha = np.round(hits_vec(rec_a, ds), 6)
        hb = np.round(hits_vec(rec_b, ds), 6)
        out[ds] = dict(fixed=int(((ha > hb)).sum()),
                       broken=int((ha < hb).sum()))
    out["total"] = dict(fixed=sum(v["fixed"] for k, v in out.items()
                                  if isinstance(v, dict)),
                        broken=sum(v["broken"] for k, v in out.items()
                                   if isinstance(v, dict)))
    return out


def main():
    z = json.load(open(os.path.join(OUTPUT_DIR, "m9_accuracy_bank.json")))
    arms = z["arms"]
    missing = [a for a in ["B2", "DB2"] + ARMS_MAIN if a not in arms]
    if missing:
        print(f"[warn] missing arms: {missing}")
    gates = {a: (arms[a].get("gate_B2_pass"), arms[a].get("gate_L0_pass"))
             for a in ("B2", "DB2L0") if a in arms}
    print("[gates]", gates)
    for a, (g1, g2) in gates.items():
        if g1 is False or g2 is False:
            raise SystemExit(f"GATE FAILED for {a}; refusing to analyse")

    b2, db2 = arms["B2"], arms["DB2"]
    m_b2, m_db2 = macro(b2), macro(db2)

    # T1 -------------------------------------------------------------------
    print("\n== T1 main table ==")
    t1 = []
    for arm in ["B2", "DB2"] + ARMS_MAIN:
        if arm not in arms or "error" in arms[arm]:
            continue
        r = arms[arm]
        m = macro(r)
        t1.append(dict(arm=arm,
                       **{ds: round(r["per_benchmark"][ds]["acc_pct"], 3)
                          for ds in DS_ORDER},
                       macro=round(m, 3),
                       d_orig=round(m - m_b2, 3),
                       d_db2=round(m - m_db2, 3)))
        print(f"  {arm:6s} " + "  ".join(
            f"{ds.split('_')[0]} {r['per_benchmark'][ds]['acc_pct']:7.3f}"
            for ds in DS_ORDER)
            + f"  macro {m:7.3f}  dOrig {m-m_b2:+.3f}  dDB2 {m-m_db2:+.3f}")

    # T2 decomposition -------------------------------------------------------
    print("\n== T2 decomposition ==")
    chain = [("B2 (original, no early pass)", m_b2),
             ("D-B2 (288 early, keep tail)", m_db2)]
    for arm in ["RND8", "COS8", "ATTN8", "CMC8", "ORC8"]:
        if arm in arms and "error" not in arms[arm]:
            chain.append((f"{arm}", macro(arms[arm])))
    prev = None
    for name, m in chain:
        step = "" if prev is None else f"  step {m-prev:+.3f}"
        print(f"  {name:34s} {m:7.3f}{step}")
        prev = m

    # T3 contrasts -----------------------------------------------------------
    print("\n== T3 paired bootstrap vs D-B2 (macro) ==")
    t3 = {}
    for arm in ARMS_MAIN:
        if arm not in arms or "error" in arms[arm]:
            continue
        if arm in ("DB2",):
            continue
        bs = paired_bootstrap(macro_vec(arms[arm]), macro_vec(db2))
        t3[arm] = bs
        print(f"  {arm:6s} vs DB2: {bs['delta']:+.3f} "
              f"CI [{bs['ci'][0]:+.3f}, {bs['ci'][1]:+.3f}] "
              f"({bs['wins']}/{bs['losses']})")
    if "CMC8" in arms and "ATTN8" in arms:
        bs = paired_bootstrap(macro_vec(arms["CMC8"]), macro_vec(arms["ATTN8"]))
        t3["CMC8-ATTN8"] = bs
        print(f"  CMC8 vs ATTN8: {bs['delta']:+.3f} "
              f"CI [{bs['ci'][0]:+.3f}, {bs['ci'][1]:+.3f}]")
    if "CMC4" in arms and "ATTN4" in arms:
        bs = paired_bootstrap(macro_vec(arms["CMC4"]), macro_vec(arms["ATTN4"]))
        t3["CMC4-ATTN4"] = bs
        print(f"  CMC4 vs ATTN4: {bs['delta']:+.3f} "
              f"CI [{bs['ci'][0]:+.3f}, {bs['ci'][1]:+.3f}]")
    bs_exp = paired_bootstrap(macro_vec(db2), macro_vec(b2))
    t3["DB2-B2"] = bs_exp
    print(f"  DB2 vs B2 (temporary exposure): {bs_exp['delta']:+.3f} "
          f"CI [{bs_exp['ci'][0]:+.3f}, {bs_exp['ci'][1]:+.3f}]")

    # T4 flips ---------------------------------------------------------------
    print("\n== T4 flips vs D-B2 ==")
    t4 = {}
    for arm in ARMS_MAIN:
        if arm not in arms or "error" in arms[arm] or arm == "DB2":
            continue
        f = flips(arms[arm], db2)
        t4[arm] = f
        print(f"  {arm:6s} fixed {f['total']['fixed']:3d} "
              f"broken {f['total']['broken']:3d} "
              f"net {f['total']['fixed']-f['total']['broken']:+d}")

    # G gate -----------------------------------------------------------------
    print("\n== screening gate ==")
    out = dict(t1=t1, t3=t3, t4=t4, gates=gates,
               exposure=dict(dB2_minus_B2=round(m_db2 - m_b2, 3)))
    best_sel = None
    for arm, r_ in (("ATTN8", 8), ("CMC8", 8), ("ATTN4", 4), ("CMC4", 4)):
        if arm in arms and "error" not in arms[arm]:
            d = macro(arms[arm]) - m_db2
            print(f"  {arm}: adjudication effect {d:+.3f}")
    json.dump(out, open(os.path.join(OUTPUT_DIR, "m9_analyze.json"), "w"),
              indent=1, default=float)
    print("[saved] m9_analyze.json")


if __name__ == "__main__":
    main()
