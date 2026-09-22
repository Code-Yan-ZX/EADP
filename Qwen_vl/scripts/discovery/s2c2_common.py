"""
S2-C2 shared definitions.

Fixed analytical objects, all read from caches produced by S2-B / S2-C0 / S2-C1:

  T          the P1-G2 gradient teacher's Top-256        (``s2c0_pilot.json``, arm P1G2)
  S          the LIN_L4 student's Top-256                (``s2c1_pilot.json``, arm LIN_L4)
  C = T ∩ S  shared;  T_only = T − S;  S_only = S − T

Both arms' ``select_idx`` lists are stored in **descending score order** (the
Top-K selector returns ``torch.topk(..., sorted=True)``), which is verified
against the raw maps in ``s2c2_decompose.py``. So "teacher rank order inside
T_only" and "student rank order inside S_only" need no re-sorting: the cached
list order *is* the rank order.

Nothing here trains, scores, or re-selects anything with the LLM. The only
quantity that ever requires a GPU is the downstream answer, produced by the
unmodified harness in ``s2c0_run.generate_prediction``.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT  # noqa: E402

DS_ALL = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]
BUDGET = 256

PILOT_TEACHER = "s2c0_pilot.json"       # runs key: b256|topk|P1G2|{ds}
PILOT_STUDENT = "s2c1_pilot.json"       # runs key: b256|topk|LIN_L4|{ds}
TEACHER_SCORES = "s2b_gradient_scores.npz"   # keys '{ds}_{i}'
STUDENT_SCORES = "s2c1_scores.npz"           # keys 'LIN_L4__{ds}_{i}'
FEATURES_META = "s2c1_features.json"
FEAT_L2 = "s2c1_feats_L2.npy"
FEAT_L4 = "s2c1_feats_L4.npy"

WRONG_THRESHOLD = 0.5     # same "hard error" convention as Stage-1 §2


def _load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def load_runs():
    return _load(PILOT_TEACHER)["runs"], _load(PILOT_STUDENT)["runs"]


def load_case():
    """Per-instance record for the held-out 150: T, S, C, T_only, S_only, answers.

    Both cached arms ran on the identical instance list (verified in
    ``s2c2_decompose.py``); this function asserts it rather than assuming it.
    """
    pil, stu = load_runs()
    cases = []
    for ds in DS_ALL:
        rT = pil[f"b{BUDGET}|topk|P1G2|{ds}"]
        rS = stu[f"b{BUDGET}|topk|LIN_L4|{ds}"]
        assert rT["idx"] == rS["idx"], f"{ds}: teacher/student instance lists differ"
        for n, i in enumerate(rT["idx"]):
            T = list(rT["select_idx"][n])
            S = list(rS["select_idx"][n])
            sT, sS = set(T), set(S)
            cases.append(dict(
                key=f"{ds}_{i}", ds=ds, idx=int(i), n=int(n),
                T=T, S=S,
                C=[t for t in T if t in sS],                 # teacher-rank ordered
                T_only=[t for t in T if t not in sS],         # teacher-rank ordered
                S_only=[s for s in S if s not in sT],         # student-rank ordered
                pred_teacher=rT["predictions"][n],
                pred_student=rS["predictions"][n],
                hit_teacher=float(rT["hits"][n]),
                hit_student=float(rS["hits"][n]),
            ))
    return cases


def cls_name(hit_t: float, hit_s: float, thr: float = WRONG_THRESHOLD) -> str:
    """The brief's four quadrants, named by the student's outcome first."""
    t_ok, s_ok = hit_t >= thr, hit_s >= thr
    if not s_ok and t_ok:
        return "student_wrong_teacher_correct"      # class 1 -- the target
    if s_ok and t_ok:
        return "both_correct"                        # class 2
    if not s_ok and not t_ok:
        return "both_wrong"                          # class 3
    return "student_correct_teacher_wrong"           # class 4


CLASSES = ["student_wrong_teacher_correct", "both_correct", "both_wrong",
           "student_correct_teacher_wrong"]


def macro(hits_by_ds: dict) -> float:
    return float(np.mean([np.mean(v) * 100.0 for v in hits_by_ds.values()]))


def bootstrap_ci(deltas: np.ndarray, n_boot: int = 10000, seed: int = 0):
    """Paired bootstrap CI on the mean of per-instance differences."""
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boot = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def stratified_bootstrap(deltas_by_ds: dict, n_boot: int = 10000, seed: int = 0):
    """Bootstrap the macro average, resampling within each benchmark."""
    rng = np.random.default_rng(seed)
    arrs = {k: np.asarray(v, dtype=float) for k, v in deltas_by_ds.items()}
    ks = sorted(arrs)
    boot = np.zeros(n_boot)
    for k in ks:
        a = arrs[k]
        boot += a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    boot /= len(ks)
    obs = float(np.mean([arrs[k].mean() for k in ks]))
    return obs, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
