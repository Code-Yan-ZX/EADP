"""
S3-A step 3 (GPU): the rescue control -- does CONDITIONAL marginal utility
convert into rescued answers better than the unary gradient score?

On instances whose S_base (the GDEP set delivered pre-LLM) generation is
wrong while the incumbent B1 or the full model B0 was right (the frozen
strata `break` / `gw_b0`), four arms add exactly k swap tokens under one
identical sequential rule: add one token, drop the current student-score
minimum, budget stays 256 at every step. Only the PICK rule differs:

    unary    highest frozen gradient-teacher score in the pool
    cond     highest MEASURED conditional marginal d(i | S_cur)
             = L(S_cur) - L(S_cur + i - r)   (greedy on the gold-answer NLL)
    spatial  farthest (maximin grid distance) from the current retained set
    random   seeded uniform pick

`cond` is deliberately given an ORACLE advantage: it optimizes the gold-
answer loss itself, which no other arm sees. If even that oracle fails to
beat the unary gradient order on answer rescue, the conditional-utility
route has no convertible value -- exactly the falsification the brief asks
for.

Endpoint per arm, k in {8, 16}: gold-NLL improvement dL = L(base) - L(set)
AND real greedy generation correctness (official per-instance score, the
same metric code the m2 arms used).

Candidate pool (48): 24 even-spread tokens over the top 48 of T_only,
12 from the middle teacher band (global rank 256..600, outside S and T),
12 from the low band (>= 600, outside S and T). All outside S_base.

Writes s3a_rescue.json incrementally (resume by completed keys).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                                   # noqa: E402
from common import eadp_model_name                              # noqa: E402
from s1_audit import OUT                                        # noqa: E402
import s3a_common as C                                          # noqa: E402
from s3a_nll import S3AHarness                                  # noqa: E402
from scoring import per_sample_hits                             # noqa: E402

KS = (8, 16)
POOLS = dict(hi=24, mid=12, lo=12)


# ---------------------------------------------------------------------------
def build_pool(case, teacher_scores):
    S = set(int(t) for t in case["S"])
    gorder = np.argsort(-teacher_scores, kind="stable")
    rank = {int(t): int(r) for r, t in enumerate(gorder)}
    T = set(int(t) for t in gorder[:C.BUDGET])
    T_only = [int(t) for t in gorder[:C.BUDGET] if int(t) not in S]
    comp = [t for t in range(C.N_VIS) if t not in S and t not in T]
    hi = C.even_spread(T_only[:max(48, POOLS["hi"] * 2)], POOLS["hi"])
    mid = C.even_spread([t for t in comp if 256 <= rank[t] < 600], POOLS["mid"])
    lo = C.even_spread([t for t in comp if rank[t] >= 600], POOLS["lo"])
    pool = hi + mid + lo
    assert len(pool) == sum(POOLS.values()), len(pool)
    assert len(set(pool)) == len(pool)
    assert not (set(pool) & S)
    bands = ["hi"] * len(hi) + ["mid"] * len(mid) + ["lo"] * len(lo)
    return dict(pool=[int(t) for t in pool], band=bands, rank=rank)


def swap_step(S, s_scores, add):
    """fixed-budget swap: add `add`, drop the current student-argmin."""
    r = int(min(S, key=lambda t: (float(s_scores[t]), t)))
    return sorted(set(S) - {r} | {int(add)}), r


def greedy_conditional(h, prep, ans, S0, pool, s_scores, max_k, log):
    S = list(S0)
    steps = []
    added = set()
    for step in range(max_k):
        cur = h.nll(prep, S, ans)[0]
        best, best_i = None, None
        for i in pool:
            if i in added:
                continue
            S_try, _ = swap_step(S, s_scores, i)
            gain = cur - h.nll(prep, S_try, ans)[0]
            if best is None or gain > best:
                best, best_i = gain, i
        if best_i is None:
            break
        S, r = swap_step(S, s_scores, best_i)
        added.add(best_i)
        steps.append(dict(S=list(S), added=int(best_i), removed=int(r),
                          gain=float(best)))
        log(f"    cond step {step + 1}: gain {best:+.4f} add {best_i}")
    return steps


def farthest_pick(S, cand):
    sS = set(S)
    best, bestd = None, -1.0
    for i in cand:
        if i in sS:
            continue
        d = min(C.grid_dist(i, t) for t in S)
        if d > bestd or (d == bestd and (best is None or i < best)):
            best, bestd = int(i), d
    return best


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keys", default=None,
                    help="JSON file: explicit list of instance keys")
    ap.add_argument("--tag", default="s3a_rescue")
    args = ap.parse_args()

    cases_doc = json.load(open(os.path.join(OUT, C.CASES_JSON)))
    cases = {c["key"]: c for c in cases_doc["cases"]}
    nll = json.load(open(os.path.join(OUT, "s3a_nll.json")))
    preds = nll["preds"]
    backbone = {b["key"]: b for b in C.load_backbone()}

    if args.keys:
        todo = json.load(open(args.keys))
    else:
        todo = []
        for key, p in preds.items():
            c = cases[key]
            rescuable = (c["hit_B1"] >= C.THR) or (c["hit_B0"] >= C.THR)
            if (p["hit"] < C.THR and rescuable
                    and c["stratum"] in ("break", "gw_b0")):
                todo.append(key)
    todo = sorted(todo)
    if args.limit:
        todo = todo[:args.limit]
    print(f"[rescue] {len(todo)} candidate instances")

    out_path = os.path.join(OUT, f"{args.tag}.json")
    results = {"config": dict(ks=list(KS), pools=POOLS,
                              rule="add-one drop-student-min per step"),
               "done": [], "runs": {}}
    if os.path.exists(out_path):
        prev = json.load(open(out_path))
        if prev.get("config") != results["config"]:
            raise RuntimeError("config mismatch with existing rescue file")
        results["done"] = prev.get("done", [])
        results["runs"] = prev.get("runs", {})
        print(f"[resume] {len(results['done'])} done")
    todo = [k for k in todo if k not in results["done"]]

    datasets = {ds: common.build_dataset(ds) for ds in C.DS_ALL}
    model = common.load_model(eadp_model_name(256, 0.5, 2.0),
                              max_new_tokens=2048)
    model.model.eval()
    torch.set_grad_enabled(False)
    h = S3AHarness(model)
    Z = np.load(os.path.join(OUT, C.TEACHER_NPZ))

    t0 = time.time()
    for key in todo:
        c = cases[key]
        b = backbone[key]
        ds_obj = datasets[c["ds"]]
        model.set_dump_image(ds_obj.dump_image)
        msg = common.build_message(model, ds_obj, c["ds"],
                                   ds_obj.data.iloc[c["idx"]])
        row = ds_obj.data.iloc[c["idx"]]
        try:
            prep = h.prep(msg, c["ds"], key)
            ans = h.answer_ids(c["golds"])
            pool = build_pool(c, Z[key])
            s_scores = b["student"]
            g_sorted = sorted(pool["pool"], key=lambda t: -float(Z[key][t]))

            Lb, _, _, _ = h.nll(prep, c["S"], ans)
            ids, text = h.generate(prep, c["S"])
            base_pred = h.vlm._post_process_response(text)
            base_hit = float(per_sample_hits(c["ds"], [row], [base_pred])[0])
            rec = dict(ds=c["ds"], stratum=c["stratum"], L_base=Lb,
                       base_pred=base_pred, base_hit=base_hit,
                       hit_B1=c["hit_B1"], hit_B0=c["hit_B0"], hit_G=c["hit_G"],
                       pool=pool["pool"], bands=pool["band"])
            print(f"\n[{key}] stratum={c['stratum']} L_base={Lb:.3f} "
                  f"base_hit={base_hit:.2f} (B1={c['hit_B1']:.2f} "
                  f"B0={c['hit_B0']:.2f} G={c['hit_G']:.2f})")

            def score_arm(name, S_final, added, removed):
                L = h.nll(prep, S_final, ans)[0]
                ids, text = h.generate(prep, S_final)
                pred = h.vlm._post_process_response(text)
                hit = float(per_sample_hits(c["ds"], [row], [pred])[0])
                rec[name] = dict(L=L, dL=float(Lb - L), pred=pred, hit=hit,
                                 rescued=bool(base_hit < C.THR <= hit),
                                 added=[int(x) for x in added],
                                 removed=[int(x) for x in removed])
                print(f"  {name:12s} dL={Lb - L:+.3f} hit={hit:.2f} "
                      f"pred={pred[:44]!r}")

            def sequential(name, pick):
                S = list(c["S"])
                added, removed = [], []
                snaps = {}
                for _ in range(max(KS)):
                    i = pick(S, set(added))
                    if i is None:
                        break
                    S, r = swap_step(S, s_scores, i)
                    added.append(int(i))
                    removed.append(int(r))
                    if len(added) in KS:
                        snaps[len(added)] = (list(S), list(added), list(removed))
                for k, (Sf, ad, rm) in snaps.items():
                    score_arm(f"{name}_k{k}", Sf, ad, rm)

            sequential("unary",
                       lambda S, hist: next((i for i in g_sorted
                                             if i not in hist and i not in set(S)),
                                            None))
            sequential("spatial",
                       lambda S, hist: farthest_pick(
                           S, [i for i in pool["pool"] if i not in hist]))
            rng = np.random.default_rng(C.SEED_SUBSET + c["idx"] + 777)
            sequential("random",
                       lambda S, hist: int(rng.choice(
                           [i for i in pool["pool"]
                            if i not in hist and i not in set(S)]))
                       if any(i not in hist and i not in set(S)
                              for i in pool["pool"]) else None)

            steps = greedy_conditional(h, prep, ans, c["S"], pool["pool"],
                                       s_scores, max(KS), print)
            for k in KS:
                if len(steps) >= k:
                    score_arm(f"cond_k{k}", steps[k - 1]["S"],
                              [s["added"] for s in steps[:k]],
                              [s["removed"] for s in steps[:k]])
            rec["cond_steps"] = steps
        except Exception:
            traceback.print_exc()
            print(f"[skip] {key}")
            continue
        results["runs"][key] = rec
        results["done"].append(key)
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f)
        os.replace(tmp, out_path)
        print(f"[done {len(results['done'])}] {key}  "
              f"elapsed {time.time() - t0:.0f}s")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
