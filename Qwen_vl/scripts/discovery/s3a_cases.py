"""
S3-A step 0 (CPU): freeze the analytic cases.

Builds the stratified 24x3 subset, the five fixed-budget contexts per instance,
the candidate pools and the gold answers, and writes them once to
``s3a_cases.json``. Everything downstream reads that file -- no sampling or
context construction ever happens again.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                            # noqa: E402
from s1_audit import OUT                                 # noqa: E402
import s3a_common as C                                   # noqa: E402


def main():
    backbone = C.load_backbone()
    outcomes = C.load_outcomes()
    picked = C.select_subset(backbone, outcomes)
    print(f"[subset] {len(picked)} instances")

    datasets = {}
    cases = []
    for b in picked:
        if b["ds"] not in datasets:
            datasets[b["ds"]] = common.build_dataset(b["ds"])
        ds = datasets[b["ds"]]
        row = ds.data.iloc[b["idx"]]
        case = C.build_case(b, outcomes)
        case["question"] = str(row.get("question", ""))
        case["category"] = str(row.get("category", ""))
        import ast
        raw = row["answer"]
        case["golds"] = C.golds_of(row)
        try:
            parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
            case["n_golds_raw"] = len(parsed) if isinstance(parsed, (list, tuple)) else 1
        except (ValueError, SyntaxError):
            case["n_golds_raw"] = 1
        cases.append(case)

    # summary + sanity report
    Z = np.load(os.path.join(OUT, C.TEACHER_NPZ))
    by = {}
    for c in cases:
        by.setdefault(c["ds"], []).append(c)
    rep = {}
    for ds, cs in by.items():
        t_overlap = [len({int(t) for t in np.argsort(-Z[c["key"]], kind="stable")[:256]}
                         & set(c["S"])) for c in cs]
        rep[ds] = dict(
            n=len(cs),
            strata={k: sum(1 for c in cs if c["stratum"] == k)
                    for k in sorted(set(c["stratum"] for c in cs))},
            student_teacher_top256_overlap=float(np.mean(t_overlap)),
            weak_overlap_base=float(np.mean(
                [len(set(c["contexts"]["base"]) & set(c["contexts"]["weak"]))
                 for c in cs])),
            strong_overlap_base=float(np.mean(
                [len(set(c["contexts"]["base"]) & set(c["contexts"]["strong"]))
                 for c in cs])),
            n_golds_mean=float(np.mean([len(c["golds"]) for c in cs])))

    out = dict(config=dict(budget=C.BUDGET, w_churn=C.W_CHURN,
                           student=f"LOCAL-MLP n{C.STUDENT_N_ARM} s{C.STUDENT_SEED}",
                           teacher=C.TEACHER_NPZ, seed=C.SEED_SUBSET,
                           quotas={ds: C.quotas(ds) for ds in C.DS_ALL}),
               summary=rep, cases=cases)
    path = os.path.join(OUT, C.CASES_JSON)
    json.dump(out, open(path, "w"))
    print(f"[saved] {path}")
    for ds in C.DS_ALL:
        print(ds, json.dumps(rep[ds]))


if __name__ == "__main__":
    main()
