"""
S3-A shared definitions: does a visual token's downstream utility depend on
the retained set S, i.e. is u(i | S) really conditional?

This is a MECHANISM TEST, not a method. Everything analytic is built on frozen
caches; the only quantities that need a GPU are the teacher-forced answer loss
(`s3a_nll.py`) and the rescue generations (`s3a_rescue.py`).

Regime (stated because it is the whole design):
    Utilities are measured under PRE-LLM delivery. A retained set S is realised
    by splicing the vision tower's merged embeddings for exactly those indices
    into the prompt -- the same 256-token sets the S2-C2 indicator-map arms
    delivered through the incumbent selector, built directly here (gate
    S3A-G2 checks the two constructions agree). Dropped tokens never enter the
    LLM at all, so any difference between u(i | S1) and u(i | S2) can only come
    from the LLM's computation over the kept set -- exactly the conditional
    effect under test, and exactly the regime the final method must live in
    (L4 pruning was retired by the M2 TTFT measurement).

Frozen objects:
    gradient teacher g_i        ``s2b_gradient_scores.npz``  (P1-G2, L0-input
                                saliency on the unpruned 1024-token forward)
    student scores s_i          LOCAL-MLP n960 seed2 (the GDEP C1 scorer) read
                                off the published M1 L4 feature cache; top-256
                                of this IS S_base. Gate S3A-G1 verifies the
                                recomputed set equals the live GDEP engine's
                                select_idx on a probe sample.
    per-instance outcomes       ``m2_accuracy.json`` hits for B0 (full), B1
                                (incumbent EADP) and the three C1-P seeds
    held-out instance bank      the frozen test-150 of ``load_m1_plan()``
                                (== the m2_accuracy key list, verified equal)

Utility definition (implemented in s3a_nll.py):
    L(y | S)   mean teacher-forced NLL of the gold answer(s) under retained set
               S; answer tokens capped at 64 (the gradient teacher's cap);
               averaged over deduplicated gold answers.
    add-marginal    d(i | S)  = L(y | S) - L(y | S+{i} -{r}),  r = argmin_S s
    leave-marginal  d(j | S)  = L(y | S) - L(y | S-{j} +{f}),  f = neutral filler
    Both keep |S| = 256 exactly. The removed token's identity, student score
    and teacher rank are recorded with every measurement (S2-C2 found the
    remove side inert; S3-A keeps logging it so that finding is re-checked,
    plus an explicit removal-identity control).
"""
from __future__ import annotations

import ast
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from s1_audit import OUT                                          # noqa: E402,F401
from m1_common import load_m1_plan                                # noqa: E402,F401

DS_ALL = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]
BUDGET = 256
N_VIS = 1024
GRID_HW = 32
ANSWER_TOKEN_CAP = 64
THR = 0.5                     # hard right/wrong, same convention as S2-C2

W_CHURN = 32                  # context churn size
MAX_GOLDS = 6                 # deduplicated golds per instance, cost cap
SEED_SUBSET = 3001            # rng root for everything constructed here

TEACHER_NPZ = "s2b_gradient_scores.npz"
STUDENT_N_ARM, STUDENT_SEED = 960, 2      # the GDEP C1 scorer checkpoint
VIS_CACHE = "s3a_vis"         # per-instance merged vision embeddings (fp16)
CASES_JSON = "s3a_cases.json" # the frozen analytic case file (s3a_cases.py)


# ---------------------------------------------------------------------------
# frozen inputs
# ---------------------------------------------------------------------------
def load_backbone():
    """test-150 rows with teacher scores, student scores and the two top-256 sets.

    Student scores are recomputed on CPU from the published checkpoint
    (m1_fixed-step_n960_L4_s2__H8.pt + m1_stats_n960.npz + s2c1_feats_L4.npy);
    everything else is read straight from caches.
    """
    meta, plan, keys, rows_of = load_m1_plan()
    test_rows = rows_of["test"]
    Z = np.load(os.path.join(OUT, TEACHER_NPZ))
    assert set(Z.files) >= {keys[r] for r in test_rows}, "teacher cache too small"

    import torch
    from m2_gdep import build_scorer
    scorer, mu, sd = build_scorer(STUDENT_N_ARM, STUDENT_SEED, "cpu")
    H = np.load(os.path.join(OUT, "s2c1_feats_L4.npy"), mmap_mode="r")
    out = []
    with torch.no_grad():
        for r in test_rows:
            key = keys[r]
            h = torch.from_numpy(np.asarray(H[r], dtype=np.float32))
            x = ((h - mu) / sd).unsqueeze(0)
            s = scorer(x, None)[0].float().numpy()
            g = Z[key].astype(np.float64)
            S = np.argsort(-s, kind="stable")[:BUDGET]
            T = np.argsort(-g, kind="stable")[:BUDGET]
            ds, idx = key.rsplit("_", 1)
            out.append(dict(key=key, ds=ds, idx=int(idx), row=int(r),
                            teacher=g, student=s,
                            S=sorted(int(t) for t in S),
                            T=sorted(int(t) for t in T)))
    return out


def load_outcomes():
    """per-instance hits from the frozen M2 accuracy run."""
    d = json.load(open(os.path.join(OUT, "m2_accuracy.json")))
    keys = d["keys"]

    def hits(arm):
        a = d["arms"][arm]
        h = []
        for ds in DS_ALL:                       # arms store ds-major per_benchmark
            h += a["per_benchmark"][ds]["hits"]
        return np.asarray(h, dtype=float)

    H = {arm: hits(arm) for arm in
         ("B0", "B1", "C1-P|s0", "C1-P|s1", "C1-P|s2")}
    H["C1-P|mean"] = np.mean([H["C1-P|s0"], H["C1-P|s1"], H["C1-P|s2"]], axis=0)
    return {k: {key: float(H[k][n]) for n, key in enumerate(keys)} for k in H}


def golds_of(row):
    """deduplicated gold answer strings, capped at MAX_GOLDS."""
    a = row["answer"]
    if isinstance(a, str):
        try:
            a = ast.literal_eval(a)
        except (ValueError, SyntaxError):
            a = [a]
    if not isinstance(a, (list, tuple)):
        a = [a]
    seen, out = set(), []
    for x in a:
        x = str(x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out[:MAX_GOLDS]


# ---------------------------------------------------------------------------
# deterministic pool helpers
# ---------------------------------------------------------------------------
def even_spread(pool, k):
    """k elements at even positions of the pool (deterministic band coverage)."""
    pool = list(pool)
    if len(pool) <= k:
        return [int(t) for t in pool]
    pos = [int((j + 0.5) * len(pool) / k) for j in range(k)]
    return [int(pool[p]) for p in pos]


def rsample(pool, k, rng):
    pool = list(pool)
    assert len(pool) >= k
    return [int(pool[j]) for j in rng.choice(len(pool), size=k, replace=False)]


# ---------------------------------------------------------------------------
# subset selection (24 per benchmark, stratified on the FROZEN m2 outcomes)
# ---------------------------------------------------------------------------
# quotas: (name, cap). Predicates use hit(B1), hit(B0), hit(C1-P seed-mean).
#   break  = student wrong where the incumbent was right   (the OCRBench pain)
#   rescue = student right where the incumbent was wrong   (the TextVQA gain)
#   gw_b0  = student wrong, incumbent wrong, full model right   (rescue targets)
#   bw     = student, incumbent and full all wrong
#   ok     = all three right (stability control)
def quotas(ds):
    if ds == "TextVQA_VAL":
        return [("break", 1), ("rescue", 7), ("gw_b0", 1), ("bw", 8), ("ok", 7)]
    if ds == "DocVQA_VAL":
        return [("break", 9), ("rescue", 4), ("gw_b0", 7), ("bw", 4), ("ok", 0)]
    return [("break", 10), ("rescue", 6), ("gw_b0", 3), ("bw", 3), ("ok", 2)]


def select_subset(backbone, outcomes):
    picked = []
    for ds in DS_ALL:
        insts = [b for b in backbone if b["ds"] == ds]
        used = set()
        for name, cap in quotas(ds):
            got = 0
            for b in insts:
                if b["key"] in used or got >= cap:
                    continue
                g1 = outcomes["C1-P|mean"][b["key"]] >= THR
                b1 = outcomes["B1"][b["key"]] >= THR
                b0 = outcomes["B0"][b["key"]] >= THR
                ok = ((name == "break" and (not g1) and b1) or
                      (name == "rescue" and g1 and (not b1)) or
                      (name == "gw_b0" and (not g1) and (not b1) and b0) or
                      (name == "bw" and (not g1) and (not b1) and (not b0)) or
                      (name == "ok" and g1 and b1))
                if ok:
                    used.add(b["key"])
                    picked.append(dict(b, stratum=name))
                    got += 1
        want = sum(c for _, c in quotas(ds))
        have = len([p for p in picked if p["ds"] == ds])
        for b in insts:                     # spill (quota unreachable -> first unused)
            if have >= want:
                break
            if b["key"] not in used:
                used.add(b["key"])
                picked.append(dict(b, stratum="spill"))
                have += 1
    return picked


# ---------------------------------------------------------------------------
# fixed-budget contexts around S_base
# ---------------------------------------------------------------------------
def build_case(b, outcomes):
    """One analytic case: S_base, five 256-token contexts, candidate pools.

    Contexts (all exactly 256 tokens, all derived from S = S_base):
      base     S
      weak     drop the W_CHURN members of S with the HIGHEST teacher score,
               refill with neutral tokens (outside S and T).
      strong   drop the W_CHURN members of S with the LOWEST teacher score,
               add the top-W_CHURN of T \\ S (moves the set toward the teacher).
      rand     W_CHURN random members dropped / random outside tokens in.
      rand2    rand under a fresh draw (seed-robustness of the control).
    Refills never touch the add-candidate pool P, so every i in P lies outside
    every context and d(i | S) is well defined for all five S.

    Add-candidates P (12): 4 high-teacher-rank misses (teacher-set ranks
    33..96, which keeps them clear of the strong refill), 4 from the middle
    teacher band (global teacher rank 257..600, outside S), 4 from the low band
    (rank > 600). Chosen by even spread inside each band.

    Leave-candidates J (8): the S-members ranked 33..40 by teacher score within
    S -- they survive the weak drop (ranks 1..32) and the strong drop (the
    bottom 32) by construction, so j in J is retained in base/weak/strong and
    probed under three genuinely different contexts. rand/rand2 may drop an
    individual j; the code skips those.
    """
    key, ds, idx = b["key"], b["ds"], b["idx"]
    S = sorted(b["S"])
    g, s = b["teacher"], b["student"]
    gorder = np.argsort(-g, kind="stable")
    rank_of_g = {int(t): r for r, t in enumerate(gorder)}
    sT = set(int(t) for t in gorder[:BUDGET])
    T_only = [int(t) for t in gorder[:BUDGET] if int(t) not in set(S)]
    comp = [t for t in range(N_VIS) if t not in set(S) and t not in sT]
    rng = np.random.default_rng(SEED_SUBSET + idx)

    # ---- add-candidate pool P ------------------------------------------------
    hi = even_spread(T_only[32:96], 4)
    mid = even_spread([t for t in comp if 256 <= rank_of_g[t] < 600], 4)
    lo = even_spread([t for t in comp if rank_of_g[t] >= 600], 4)
    P = hi + mid + lo
    assert len(set(P)) == 12, (len(hi), len(mid), len(lo))
    sP = set(P)
    refill_pool = [t for t in comp if t not in sP]

    # ---- contexts -------------------------------------------------------------
    s_by_g = sorted(S, key=lambda t: -g[t])          # descending teacher score
    drop_weak = s_by_g[:W_CHURN]
    drop_strong = s_by_g[-W_CHURN:]
    add_strong = T_only[:W_CHURN]
    assert not (set(add_strong) & sP)

    def churn(S_set, drop, add):
        assert len(set(drop) & set(add)) == 0
        return sorted((set(S_set) - set(drop)) | set(add))

    contexts = {"base": S,
                "weak": churn(S, drop_weak, rsample(refill_pool, W_CHURN, rng)),
                "strong": churn(S, drop_strong, add_strong),
                "rand": churn(S, rsample(S, W_CHURN, rng),
                              rsample(refill_pool, W_CHURN, rng)),
                "rand2": churn(S, rsample(S, W_CHURN, rng),
                               rsample(refill_pool, W_CHURN, rng))}
    for name, c in contexts.items():
        assert len(c) == BUDGET and len(set(c)) == BUDGET, name
        assert not (sP & set(c)), f"candidate leaked into {name}"

    # ---- leave-candidates J ----------------------------------------------------
    J = s_by_g[W_CHURN:W_CHURN + 8]

    # neutral filler for leave-marginals; must sit outside S (comp guarantees it),
    # the NLL script re-picks on the rare collision with a context refill.
    filler = int(rng.choice(refill_pool))

    # removal-identity control: base context, 6 candidates x 3 random removals
    ri = {int(t): [int(x) for x in rng.choice(S, size=3, replace=False)]
          for t in P[:6]}

    return dict(key=key, ds=ds, idx=int(idx), row=int(b["row"]),
                stratum=b.get("stratum"),
                hit_B1=outcomes["B1"][key], hit_B0=outcomes["B0"][key],
                hit_G=outcomes["C1-P|mean"][key],
                S=S, contexts=contexts,
                drop_weak=[int(t) for t in drop_weak],
                drop_strong=[int(t) for t in drop_strong],
                add_strong=[int(t) for t in add_strong],
                P=P, P_bands={"hi": hi, "mid": mid, "lo": lo},
                J=[int(t) for t in J], filler=filler, removal_identity=ri,
                grank={int(t): int(rank_of_g[t]) for t in set(P) | set(J)})


def load_cases():
    """The frozen case file written by s3a_cases.py (CPU)."""
    with open(os.path.join(OUT, CASES_JSON)) as f:
        return json.load(f)["cases"]


def token_rc(t):
    """grid coordinates (row, col) of merged token index t on 32x32."""
    return divmod(int(t), GRID_HW)


def grid_dist(a, b_):
    ra, ca = token_rc(a)
    rb, cb = token_rc(b_)
    return float(np.hypot(ra - rb, ca - cb))
