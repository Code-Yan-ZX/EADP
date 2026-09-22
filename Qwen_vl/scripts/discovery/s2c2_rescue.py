"""
S2-C2 step 3: answer-rescue analysis.

Two jobs:

``--plan``   (CPU) read the coarse swap grid and build, for every instance in the
             rescuable class (student wrong, teacher correct), the recovery curve
             hit(k); locate the first k at which the answer becomes correct; and
             write the refinement plan (fine k grid + leave-one-out ablations)
             that ``--run`` then executes.

``--run``    (GPU) execute that plan on the rescuable subset only, with the same
             indicator-override harness as the main grid.

Leave-one-out ablation. At the rescue level ``k*`` the swap set is
``F(k*) = (S - S_only[m-k*:]) + T_only[:k*]``, i.e. the j-th teacher token that
was moved in, ``T_only[j]``, displaced the student token ``S_only[m-k*+j]``.
Ablating teacher rank ``j`` therefore means the budget-preserving set
``(F(k*) - {T_only[j]}) + {S_only[m-k*+j]}`` -- still exactly 256 tokens. Doing
this for every ``j`` in the *final block* (the tokens added between the last
still-wrong level and the first correct one) asks directly whether the rescue
rests on one token or on the block as a whole. Ablating a token from *earlier*
in the order is the control: if the answer is fragile to any removal, the
final-block finding means nothing.
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
from common import build_dataset, build_message, eadp_model_name, ensure_out_dir  # noqa: E402
from diag_selectors import generate_prediction  # noqa: E402
from s1_audit import OUT  # noqa: E402
from s2b_run import GradientPruner  # noqa: E402
from s2c2_common import BUDGET, DS_ALL, WRONG_THRESHOLD, cls_name, load_case  # noqa: E402
from scoring import per_sample_hits  # noqa: E402

COARSE_KS = [0, 8, 16, 32, 48, 64, 96, "full"]


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
def clean_ks(runs):
    """The teacher-priority k values that actually have results, ascending.

    Derived from the run keys rather than from the arm list in the config: the
    grid is filled in over several invocations, so the config of any one of them
    is not the full picture.
    """
    return sorted({int(k.split(":")[1].split("|")[0])
                   for k in runs if k.startswith("teacher:")})


def recovery_table(swap_json, cases, ks, teacher_full_rk="identity_T",
                   identity_json="s2c2_identity.json"):
    """Per-instance hit(k) for the teacher-priority family, plus the full-swap point.

    ``k = full`` is the teacher set T itself, which the identity arm measured;
    it lives in the identity file rather than the grid file.
    """
    runs = dict(swap_json["runs"])
    ip = os.path.join(OUT, identity_json)
    if os.path.exists(ip):
        runs.update(json.load(open(ip))["runs"])
    by = {}
    for ds in DS_ALL:
        for k in ks:
            r = runs.get(f"teacher:{k}|{ds}")
            if r is None:
                continue
            for n, i in enumerate(r["idx"]):
                by.setdefault(f"{ds}_{i}", {})[k] = float(r["hits"][n])
        # k = full == the teacher set itself == identity_T
        r = runs.get(f"{teacher_full_rk}|{ds}")
        if r is None:
            continue
        for n, i in enumerate(r["idx"]):
            by.setdefault(f"{ds}_{i}", {})["full"] = float(r["hits"][n])
    # k = 0 is the cached student run
    for c in cases:
        by.setdefault(c["key"], {})[0] = c["hit_student"]
    return by


def main_plan(args):
    swap = json.load(open(os.path.join(OUT, args.swap_json)))
    cases = load_case()
    case_by = {c["key"]: c for c in cases}
    ks = clean_ks(swap["runs"])
    by = recovery_table(swap, cases, ks, teacher_full_rk=args.full_arm)
    ks_all = sorted(ks)

    res = [c for c in cases if cls_name(c["hit_teacher"], c["hit_student"])
           == "student_wrong_teacher_correct"]
    plan = dict(ks=ks_all, rescuable=[], fine_ks=args.fine_ks)
    for c in res:
        curve = {k: by[c["key"]].get(k) for k in ks_all + ["full"]}
        # first level at which the answer is scored correct
        first = None
        prev = None
        for k in ks_all + ["full"]:
            h = curve.get(k)
            if h is None:
                continue
            if h >= WRONG_THRESHOLD:
                first = k
                break
            prev = k
        plan["rescuable"].append(dict(
            key=c["key"], ds=c["ds"], idx=c["idx"],
            m=len(c["T_only"]),
            hit_student=c["hit_student"], hit_teacher=c["hit_teacher"],
            curve={str(k): v for k, v in curve.items()},
            first_rescue_k=first,
            prev_k=prev,
            block=(None if first is None or prev is None or first == "full"
                   else first - prev),
            # teacher-rank indices whose *addition* coincides with the rescue;
            # these are the candidates for the leave-one-out ablation
            block_js=([] if first is None or prev is None or first == "full"
                      else list(range(prev, first))),
            pred_student=c["pred_student"],
            pred_teacher=c["pred_teacher"],
        ))
    for r in plan["rescuable"]:
        r["ref_k"] = r["first_rescue_k"] if isinstance(r["first_rescue_k"], int) else None
    n_resc = sum(1 for r in plan["rescuable"] if r["first_rescue_k"] is not None)
    plan["summary"] = dict(
        n_rescuable=len(res), n_rescued_by_full=sum(
            1 for r in plan["rescuable"] if r["curve"].get("full", 0) >= WRONG_THRESHOLD),
        n_rescued_by_grid=n_resc,
        blocks=[r["block"] for r in plan["rescuable"] if r["block"]],
    )
    path = os.path.join(OUT, "s2c2_rescue_plan.json")
    json.dump(plan, open(path, "w"), indent=1)
    print(f"[plan] rescuable n={len(res)}  rescued within the coarse grid: {n_resc}")
    for r in plan["rescuable"]:
        print(f"  {r['key']:18s} |T_only|={r['m']:3d}  "
              f"hit_s={r['hit_student']:.2f}->hit_t={r['hit_teacher']:.2f}  "
              f"first rescue k={r['first_rescue_k']} (prev {r['prev_k']}, block {r['block']})")
    print(f"[saved] {path}")


# ---------------------------------------------------------------------------
# running the refinement
# ---------------------------------------------------------------------------
def build_refined_set(case, arm, plan_rec):
    """Arms: 'teacher:<k>' | 'loo:<j>' | 'looearly:<j>' | 'blockrand:<k>:<seed>'."""
    parts = arm.split(":")
    kind = parts[0]
    T, S = list(case["T"]), list(case["S"])
    T_only, S_only = case["T_only"], case["S_only"]
    m = len(T_only)

    if kind == "teacher":
        k = min(int(parts[1]), m)
        return [s for s in S if s not in set(S_only[m - k:])] + T_only[:k]

    if kind in ("loo", "looearly"):
        j = int(parts[1])
        k = plan_rec["ref_k"]
        k = min(k, m)
        if j >= k:
            return None
        base = [s for s in S if s not in set(S_only[m - k:])] + T_only[:k]
        drop = T_only[j]
        fill = S_only[m - k + j]
        return [t for t in base if t != drop] + [fill]

    if kind == "blockrand":
        k = min(int(parts[1]), m)
        seed = int(parts[2][1:]) if parts[2].startswith("s") else int(parts[2])
        rng = np.random.default_rng(seed)
        # the top of the teacher's list is kept; only the tail (the last
        # `tail` ranks) is re-ordered, so this isolates "which tokens inside
        # the rescue block" from "how many tokens in total"
        tail = min(plan_rec.get("block") or 16, k)
        head = T_only[: k - tail]
        rest = list(T_only[k - tail:k])
        rng.shuffle(rest)
        add = head + rest
        return [s for s in S if s not in set(S_only[m - k:])] + add

    raise KeyError(kind)


def main_run(args):
    plan = json.load(open(os.path.join(OUT, "s2c2_rescue_plan.json")))
    cases = load_case()
    case_by = {c["key"]: c for c in cases}
    if args.auto:
        arms = []
        for k in plan["fine_ks"]:
            arms.append(f"teacher:{k}")
        for r in plan["rescuable"]:
            if r.get("ref_k"):
                arms.append(f"blockrand:{r['ref_k']}:s0")
        js = sorted({j for r in plan["rescuable"] for j in r["block_js"]})
        arms += [f"loo:{j}" for j in js]
        arms += [f"looearly:{j}" for j in (0, 1, 2)]
        seen, ordered = set(), []
        for a in arms:
            if a not in seen:
                seen.add(a); ordered.append(a)
        args.arms = ordered
        print(f"[auto] {len(args.arms)} arms")

    def eligible(arm):
        parts = arm.split(":")
        keys = []
        for r in plan["rescuable"]:
            if not args.all and (r["ref_k"] is None):
                continue
            if parts[0] == "loo" and int(parts[1]) not in r["block_js"]:
                continue
            if parts[0] == "looearly" and int(parts[1]) >= (r["prev_k"] or 0):
                continue
            keys.append(r["key"])
        return keys

    all_keys = [r["key"] for r in plan["rescuable"]
                if args.all or r.get("ref_k") is not None]
    if args.max_samples:
        all_keys = all_keys[: args.max_samples]
    keep = set(all_keys)
    cases = [c for c in cases if c["key"] in keep]
    print(f"[run] {len(cases)} instances")

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
    for ds in DS_ALL:
        dataset = build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        idx = sorted(c["idx"] for c in cases if c["ds"] == ds)
        bank[ds] = dict(idx=idx, rows=[dataset.data.iloc[i] for i in idx],
                        msgs=[build_message(model, dataset, ds, dataset.data.iloc[i])
                              for i in idx])

    out_path = os.path.join(ensure_out_dir(), f"{args.tag}.json")
    results = {"config": dict(arms=args.arms, fine_ks=plan["fine_ks"]), "runs": {}}
    if os.path.exists(out_path):
        results["runs"] = json.load(open(out_path)).get("runs", {})
        print(f"[merge] {len(results['runs'])} existing records")

    # arms already recorded are not re-run: the grid is filled in over several
    # invocations and re-computing an arm costs the same as computing it
    done = set()
    if os.path.exists(out_path):
        done = {k.split("|")[0] for k in results["runs"]}
    for arm in args.arms:
        if arm in done:
            continue
        if arm.startswith("loo") and not args.with_loo:
            continue
        for ds in DS_ALL:
            b = bank[ds]
            if not b["idx"]:
                continue
            elig = set(eligible(arm))
            msgs = [(n, msg) for n, msg in enumerate(b["msgs"])
                    if f"{ds}_{b['idx'][n]}" in elig]
            preds, ok = [], []
            for n, msg in tqdm(msgs, desc=f"{arm}/{ds}", leave=False):
                i = b["idx"][n]
                c = case_by[f"{ds}_{i}"]
                rec = next(r for r in plan["rescuable"] if r["key"] == c["key"])
                sel = build_refined_set(c, arm, rec)
                if sel is None:
                    preds.append(None); ok.append(False); continue
                ind = np.zeros(1024, dtype=np.float32)
                ind[sel] = 1.0
                pruner.override = torch.from_numpy(ind).float().unsqueeze(0).to(
                    next(model.model.parameters()).device)
                try:
                    r = generate_prediction(model, msg, ds)
                except Exception:
                    traceback.print_exc()
                    r = {"prediction": "", "n_kept": 0}
                preds.append(r["prediction"])
                got = pruner.last_capture.get("select_idx")
                ok.append(got is not None and
                          sorted(got[0].cpu().numpy().tolist()) == sorted(sel))
            sel_rows = [b["rows"][n] for n, _ in msgs]
            sel_idx = [b["idx"][n] for n, _ in msgs]
            valid = [p for p in preds if p is not None]
            rows = [r for r, p in zip(sel_rows, preds) if p is not None]
            hits = per_sample_hits(ds, rows, valid) if valid else np.array([])
            k = f"{arm}|{ds}"
            results["runs"][k] = dict(
                arm=arm, dataset=ds, n=int(len(hits)),
                acc_pct=float(np.mean(hits) * 100) if len(hits) else float("nan"),
                hits=[float(h) for h in hits],
                keys=[f"{ds}_{i}" for i, p in zip(sel_idx, preds) if p is not None],
                predictions=valid,
                delivery_ok_frac=float(np.mean([o for o, p in zip(ok, preds)
                                                if p is not None])) if valid else 0.0)
            print(f"{k:28s} n={len(hits):3d} acc={results['runs'][k]['acc_pct']:7.3f} "
                  f"ok={results['runs'][k]['delivery_ok_frac']:.2f}")
            json.dump(results, open(out_path, "w"))
    print(f"[saved] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--swap-json", default="s2c2_swap.json")
    ap.add_argument("--full-arm", default="identity_T")
    ap.add_argument("--tag", default="s2c2_rescue")
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--auto", action="store_true",
                    help="build the arm list from the plan: the whole fine k grid, "
                         "a block-shuffle arm at each rescue level, a leave-one-out "
                         "arm for every teacher rank inside a transition block, and "
                         "three early-rank ablations as the fragility control")
    # Dense where the transitions actually happen (the coarse grid put first
    # rescues at k = 2, 8, 16, 32, 48, 96), so the transition block -- the set of
    # tokens whose joint addition flips the answer -- is only 1-3 wide and the
    # leave-one-out test is sharp.  These arms run on the 28 rescuable instances
    # only, so the whole fine grid costs a few minutes.
    ap.add_argument("--fine-ks", nargs="*", type=int,
                    default=(list(range(1, 17)) + list(range(18, 33, 2))
                             + list(range(36, 97, 4))))
    ap.add_argument("--with-loo", action="store_true", default=True,
                    help="run the leave-one-out arms (on by default; they are cheap "
                         "because each one only touches the instances whose "
                         "transition block contains that teacher rank)")
    ap.add_argument("--all", action="store_true",
                    help="run on every rescuable instance, not only grid-rescued ones")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()
    if args.plan:
        main_plan(args)
    if args.run:
        main_run(args)


if __name__ == "__main__":
    main()
