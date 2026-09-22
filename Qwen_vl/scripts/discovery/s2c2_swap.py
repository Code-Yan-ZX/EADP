"""
S2-C2 step 2: budget-preserving counterfactual swaps between the student set S
(LIN_L4 Top-256) and the teacher set T (P1-G2 Top-256).

The budget is **always exactly 256 tokens**. A swap of size k moves k tokens:
it adds k tokens taken from ``T_only = T - S`` and removes k tokens from
``S_only = S - T``. Note ``|T_only| == |S_only|`` identically (both sets have
256 members and share ``C``), which is what makes the reverse sweep a re-reading
of the forward curve rather than a new family -- see ``F_rev`` below.

Delivery. The harness never needs to know these are not scores: an **indicator
map** (1.0 on the desired set, 0.0 elsewhere) fed to the existing Top-K selector
returns exactly that set, because exactly 256 entries tie at the top. Nothing in
``model/pruner.py``, the scoring stages, or the generation path is touched --
only ``GradientPruner.override`` changes, exactly as in S2-B / S2-C0.

Orderings (the whole point of the experiment):
  teacher      add T_only by *descending teacher rank*, remove S_only by
               *ascending student rank*   (the student's weakest picks go first)
  adversarial  add T_only by *ascending teacher rank*, remove S_only by
               *descending student rank*  (the mirror ordering -- same endpoints)
  random       uniform samples from each side, fixed seeds
  shuffled     T_only ordering replaced by another image's teacher map, which
               is the content-free control: same value distribution, wrong image
  identity_S / identity_T / F_rev   mechanism checks (must reproduce caches)

Writes ``<tag>.json`` incrementally; safe to run in batches.
"""
import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import build_dataset, build_message, eadp_model_name, ensure_out_dir, sample_indices  # noqa: E402
from diag_selectors import generate_prediction  # noqa: E402
from s1_audit import OUT  # noqa: E402
from s2b_run import GradientPruner  # noqa: E402
from s2c2_common import (BUDGET, DS_ALL, STUDENT_SCORES, TEACHER_SCORES,  # noqa: E402
                         load_case)
from scoring import per_sample_hits  # noqa: E402

CS = 1000                                   # teacher-map pool stride for the shuffle control


def order_pool(case, kind, k, seed, shuffle_pool, shuffle_index):
    """Return (to_add, to_remove) -- each a list of exactly k token ids."""
    m = len(case["T_only"])
    k = min(k, m)
    if k == 0:
        return [], []
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng(0)
    sAll = set(case["S"]) | set(case["T"])

    if kind == "teacher":
        add = case["T_only"][:k]
        rem = case["S_only"][m - k:]
    elif kind == "adversarial":
        add = case["T_only"][m - k:]
        rem = case["S_only"][:k]
    elif kind == "random":
        add = [int(x) for x in rng.choice(case["T_only"], size=k, replace=False)]
        rem = [int(x) for x in rng.choice(case["S_only"], size=k, replace=False)]
    elif kind == "addteacher":
        # isolate the ADDITION half: teacher's best tokens in, but the tokens
        # removed from S are chosen at random rather than being its worst.
        add = case["T_only"][:k]
        rem = [int(x) for x in rng.choice(case["S_only"], size=k, replace=False)]
    elif kind == "remworst":
        # isolate the REMOVAL half: drop the student's worst-ranked picks and
        # refill with tokens that are neither student- nor teacher-selected, so
        # the refill carries no information from either side.
        neutral = [t for t in range(1024) if t not in sAll]
        add = [int(x) for x in rng.choice(neutral, size=k, replace=False)]
        rem = case["S_only"][m - k:]
    elif kind == "shuffled":
        # the teacher map of a *different* image, ranked on this image's tokens.
        # Keep the other map's own priority order but drop anything this image's
        # student already selected, so the swap stays exactly size k.
        j = (shuffle_index + CS) % len(shuffle_pool)
        other = shuffle_pool[j]
        assert other["key"] != case["key"]
        sS = set(case["S"])
        add = [int(t) for t in other["ranked"] if int(t) not in sS][:k]
        assert len(add) == k, f"shuffle pool exhausted ({len(add)} < {k})"
        rem = case["S_only"][m - k:]
    else:
        raise KeyError(kind)
    return list(add), list(rem)


def build_set(case, arm):
    """The 256-token counterfactual set for one arm."""
    kind = arm["kind"]
    sS, sT = set(case["S"]), set(case["T"])
    if kind == "identity_S":
        return list(case["S"])
    if kind == "identity_T":
        return list(case["T"])
    if kind == "F_rev":
        # from T: remove the teacher's *lowest-ranked* disagreement tokens and
        # add the student's *highest-ranked* ones.  Set-identical to
        # F_T(|T_only| - k) -- this arm is the empirical check of that identity.
        m = len(case["T_only"])
        k = min(arm["k"], m)
        return [t for t in case["T"] if t not in set(case["T_only"][m - k:])] \
            + case["S_only"][:k]
    add, rem = order_pool(case, kind, arm["k"], arm.get("seed"),
                          arm["_pool"], arm["_shuf_idx"])
    out = [int(s) for s in case["S"] if s not in set(rem)] + [int(a) for a in add]
    assert len(out) == BUDGET, f"{len(out)} tokens for {arm['name']}"
    assert len(set(out)) == BUDGET, "duplicate token in swap set"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="specs 'kind:k[:seed]', e.g. teacher:32 random:32:s0 "
                         "identity_S identity_T F_rev:32")
    ap.add_argument("--tag", default="s2c2_swap")
    ap.add_argument("--datasets", nargs="+", default=DS_ALL)
    ap.add_argument("--only-keys", default=None,
                    help="restrict to the instances listed in this JSON file "
                         "(a list of 'DS_idx' strings); the accuracy of such a run "
                         "is only comparable to other runs on the same subset")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    cases = load_case()
    case_by_key = {c["key"]: c for c in cases}
    if args.only_keys:
        keys = set(json.load(open(args.only_keys)))
        cases = [c for c in cases if c["key"] in keys]
        print(f"[subset] {len(cases)} instances")

    # the teacher-map pool for the shuffle control: every cached instance, keyed
    # in a fixed sorted order so the stride is reproducible.
    Z = np.load(os.path.join(OUT, TEACHER_SCORES))
    pool = []
    for key in sorted(Z.files):
        order = np.argsort(-Z[key], kind="stable")
        pool.append(dict(key=key, T_only=[int(t) for t in order[:BUDGET]],
                         ranked=[int(t) for t in order]))
    print(f"[shuffle pool] {len(pool)} teacher maps")

    arms = []
    for spec in args.arms:
        parts = spec.split(":")
        kind, k, seed = parts[0], None, None
        if len(parts) > 1:
            k = int(parts[1])
        if len(parts) > 2:
            seed = int(parts[2][1:]) if parts[2].startswith("s") else int(parts[2])
        name = spec
        arms.append(dict(name=name, kind=kind, k=k, seed=seed))

    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0),
                              max_new_tokens=args.max_new_tokens)
    pruner = GradientPruner(
        visual_token_num=BUDGET, alpha=0.5, beta=2.0,
        visual_dim=model.pruner.visual_dim,
        spatial_merge_size=model.pruner.spatial_merge_size,
        selector="topk", capture=True,
    ).to(next(model.model.parameters()).device)
    pruner.eval()
    model.pruner = pruner

    bank = {}
    for ds in args.datasets:
        dataset = build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        idx = sample_indices(len(dataset.data), 150, offset=0)[::3]   # the held-out 150
        keep = {c["idx"] for c in cases if c["ds"] == ds}
        idx = [i for i in idx if i in keep]
        bank[ds] = dict(idx=idx, rows=[dataset.data.iloc[i] for i in idx],
                        msgs=[build_message(model, dataset, ds, dataset.data.iloc[i])
                              for i in idx])
        print(f"[bank] {ds}: {len(idx)} instances")

    out_path = os.path.join(ensure_out_dir(), f"{args.tag}.json")
    results = {"config": dict(tag=args.tag, arms=args.arms,
                              only_keys=args.only_keys),
               "n_instances": len(cases), "runs": {}}
    if os.path.exists(out_path):
        prev = json.load(open(out_path))
        results["runs"] = prev.get("runs", {})
        print(f"[merge] {len(results['runs'])} existing arm-records")

    shuf_idx = {c["key"]: i for i, c in enumerate(cases)}

    for arm in arms:
        arm["_pool"] = pool
        arm["_shuf_idx"] = 0
        for ds in args.datasets:
            b = bank[ds]
            preds, meta, expect = [], [], []
            for n, m in enumerate(tqdm(b["msgs"], desc=f"{arm['name']}/{ds}", leave=False)):
                i = b["idx"][n]
                c = case_by_key[f"{ds}_{i}"]
                arm["_shuf_idx"] = shuf_idx[c["key"]]
                sel = build_set(c, arm)
                expect.append(sorted(sel))
                ind = np.zeros(1024, dtype=np.float32)
                ind[sel] = 1.0
                pruner.override = torch.from_numpy(ind).float().unsqueeze(0).to(
                    next(model.model.parameters()).device)
                try:
                    r = generate_prediction(model, m, ds)
                except Exception:
                    traceback.print_exc()
                    r = {"prediction": "", "n_kept": 0, "prune_ms": float("nan"),
                         "select_ms": float("nan")}
                preds.append(r["prediction"])
                got = pruner.last_capture.get("select_idx")
                meta.append(dict(n_kept=r["n_kept"],
                                 delivered_ok=(got is not None and
                                               sorted(got[0].cpu().numpy().tolist()) ==
                                               sorted(sel))))
            hits = per_sample_hits(ds, b["rows"], preds)
            rk = f"{arm['name']}|{ds}"
            results["runs"][rk] = {
                "arm": arm["name"], "kind": arm["kind"], "k": arm["k"],
                "seed": arm["seed"], "dataset": ds, "budget": BUDGET,
                "n": int(len(hits)), "acc_pct": float(np.mean(hits) * 100),
                "sum_hits": float(np.sum(hits)),
                "n_kept_mean": float(np.mean([x["n_kept"] for x in meta])),
                "delivery_ok_frac": float(np.mean([x["delivered_ok"] for x in meta])),
                "predictions": preds, "hits": [float(h) for h in hits],
                "idx": [int(i) for i in b["idx"]],
                "expect_sorted": expect,
            }
            print(f"{rk:34s} acc={results['runs'][rk]['acc_pct']:7.3f} "
                  f"kept={results['runs'][rk]['n_kept_mean']:6.1f} "
                  f"deliver_ok={results['runs'][rk]['delivery_ok_frac']:.3f}")
            tmp = out_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(results, f)
            os.replace(tmp, out_path)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
