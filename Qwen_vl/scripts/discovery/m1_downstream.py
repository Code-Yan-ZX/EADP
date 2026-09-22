"""
M1 step 5: prepare -- but do not run -- the downstream selector comparison.

S2-C6's downstream gate (§6 of the M1 pre-registration, inherited verbatim) asks
whether a deployable forward-only arm beats the L4 reference with a paired CI
lower bound > 0 and reaches held-out R@8 >= 0.82. M1 answers that offline in
``m1_consolidate.py``. This file is the *next* step, and M1 deliberately stops
short of it: it assembles the comparison and checks that every artifact the
comparison needs is on disk and loadable, then prints the command. It produces no
comparison result under ``--check``, which is the only mode M1 runs.

The comparison it prepares
--------------------------
The learned scorer replaces the EADP importance signal; it does not replace the
selector. So the fair comparison holds the selector fixed and swaps only the
importance vector, for each of three selectors, at budget 256:

    topk      top-256 of the importance vector
    block8    block-parallel greedy on the facility-location objective, block 8
    facility  the official EADP greedy selection

    baseline  EADP importance -- already cached for the frozen 450 in
              s2b_official_selection.npz as imp__<key>, with the official
              facility selection as sel__<key>
    candidate the best M1 checkpoint applied to the cached L4 hidden states

Two halves, with different costs:

  * offline  the learned score is a function of cached L4 features, so the
             candidate's 256-token selection costs no forward pass at all; the
             baseline's importance is cached too. What is *not* cached is the
             token similarity matrix the coverage selectors need, which comes
             from the post-merger vision features (``_sim_visual_impl``,
             sim_mode "rebound") and needs one vision-tower pass per instance.
             ~150 instances, seconds each.
  * accuracy  actually running the pruned model on the held-out 150 under each
             selector -- this is the generation step, and it is NOT run here.

Usage
    python m1_downstream.py --check          # verify artifacts, print the command
    python m1_downstream.py --offline        # the selector-level comparison
    python m1_downstream.py --generate       # the accuracy run (not run by M1)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                              # noqa: E402
from s2c3_common import BUDGET, head_metrics, teacher_orders          # noqa: E402
from s2c6_models import build, n_params                               # noqa: E402
from m1_common import (ARM, TEACHER, TARGET_ARM, M1Store,             # noqa: E402
                       channel_stats_l4, load_m1_plan)

SELECTORS = ("topk", "block8", "facility")


def best_checkpoint():
    """The highest held-out R@8 primary checkpoint, read from the audit artifact."""
    p = os.path.join(OUT, "m1_audit.json")
    if not os.path.exists(p):
        raise SystemExit("run m1_consolidate.py first: no m1_audit.json")
    a = json.load(open(p))
    lad = a["protocols"]["fixed-step"]["ladder"]
    best = max(lad, key=lambda x: x["R8"])
    n = best["n"]
    runs = a["protocols"]["fixed-step"]["runs"][str(n)]
    seed = max(runs, key=lambda r: r["test_h8"])["seed"]
    tag = f"m1_fixed-step_n{n}_{ARM}_s{seed}__H8.pt"
    return dict(n=n, seed=seed, R8=best["R8"], file=tag,
                path=os.path.join(OUT, tag), verdict=a["verdict"]["verdict"])


def check(ck):
    """Every artifact the comparison needs, resolved and loadable. No results."""
    import torch
    _, plan, keys, rows_of = load_m1_plan()
    test_keys = [keys[i] for i in rows_of["test"]]
    need = [TEACHER, "s2b_official_selection.npz", "m1_gradient_scores_extra.npz"]

    rep = {"selected_checkpoint": ck, "n_held_out": len(test_keys), "checks": {}}
    rep["checks"]["checkpoint_exists"] = os.path.exists(ck["path"])
    if rep["checks"]["checkpoint_exists"]:
        sd = torch.load(ck["path"], map_location="cpu")
        m = build(ARM)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        # TokMLP stores a single weight vector w2 and bias b; the names follow
        # s2c5_models.LocalMLP. Any mismatch is reported rather than swallowed.
        rep["checks"]["checkpoint_loads"] = not missing and not unexpected
        rep["checks"]["n_params"] = n_params(m)
        rep["checks"]["state_keys"] = sorted(sd.keys())
        rep["checks"]["missing_keys"] = list(missing)
        rep["checks"]["unexpected_keys"] = list(unexpected)

    for f in need:
        p = os.path.join(OUT, f)
        rep["checks"][f"{f}_exists"] = os.path.exists(p)

    sel = np.load(os.path.join(OUT, "s2b_official_selection.npz"))
    rep["checks"]["baseline_importance_for_all_held_out"] = all(
        f"imp__{k}" in sel.files for k in test_keys)
    rep["checks"]["baseline_facility_sel_for_all_held_out"] = all(
        f"sel__{k}" in sel.files for k in test_keys)
    G = np.load(os.path.join(OUT, TEACHER))
    rep["checks"]["teacher_map_for_all_held_out"] = all(
        k in G.files for k in test_keys)
    rep["checks"]["baseline_budget"] = int(
        sel[f"sel__{test_keys[0]}"].shape[0])

    store = M1Store()
    H = store.orig[rows_of["test"][0]]
    rep["checks"]["l4_features_available"] = bool(H.shape == (1024, 4096))
    rep["checks"]["l4_feature_rows"] = len(rows_of["test"])

    try:
        from instrumented import SELECTORS as _S, _sim_visual_impl  # noqa: F401
        rep["checks"]["selector_functions_importable"] = all(
            s in _S for s in SELECTORS)
    except Exception as e:                                    # pragma: no cover
        rep["checks"]["selector_functions_importable"] = f"import failed: {e}"

    # the missing piece, stated rather than discovered later
    rep["checks"]["token_similarity_matrix_cached"] = False
    rep["note"] = ("the coverage selectors (block8, facility) need the token "
                   "similarity matrix from the post-merger vision features; it "
                   "is not cached anywhere in the S2 outputs and has to be "
                   "regenerated with one vision-tower pass per instance")
    rep["passed"] = all(v is True for k, v in rep["checks"].items()
                        if k not in ("n_params", "state_keys", "missing_keys",
                                     "unexpected_keys", "baseline_budget",
                                     "l4_feature_rows",
                                     "token_similarity_matrix_cached"))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--generate", action="store_true")
    args = ap.parse_args()

    ck = best_checkpoint()
    print(f"[best] n={ck['n']} seed={ck['seed']} R@8={ck['R8']:.4f} "
          f"verdict={ck['verdict']}")
    if args.check or not (args.offline or args.generate):
        rep = check(ck)
        for k, v in rep["checks"].items():
            print(f"  {'OK ' if v is True else '   '} {k} = {v}")
        print(f"\n[ready] {rep['passed']}")
        print(rep["note"])
        print("\nCommand the comparison would run (NOT run by M1):")
        print(f"  python scripts/discovery/m1_downstream.py --offline   "
              f"# selector-level, {ck['n']} fit, checkpoint seed {ck['seed']}")
        print(f"  python scripts/discovery/m1_downstream.py --generate  "
              f"# accuracy, needs the EADP-256 pipeline")
        return

    if args.offline:
        raise SystemExit(
            "offline selector comparison is prepared but deliberately not run "
            "by M1: the pre-registration (§6) says M1 records the command and "
            "the artifacts and does not execute the downstream comparison. "
            "Re-run without the guard once M2 is authorised.")
    if args.generate:
        raise SystemExit(
            "generation is out of scope for M1 by pre-registration (§6).")


if __name__ == "__main__":
    main()
