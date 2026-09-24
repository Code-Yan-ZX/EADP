"""
M3-v0 step 3 -- the held-out 150 grid for MissGuard and its controls.

Every arm runs through the SAME engine the B0/B1/B2 numbers came from
(`m2_gdep.GDEPEngine`, mode `prellm`, selector `block8`): the only thing that
changes between an arm and B2 is the pruner's post-selection correction, applied
inside the pruner call -- i.e. inside the measured region, before any decoder
layer sees the sequence.

Arms
    MG0                r=0 identity: must reproduce the stored B2 arm hit-for-hit
    MG-<rule>-r<r>     learned student rescues r tokens, evicts r by <rule>
    OR-<rule>-r<r>     teacher rescues r, evicts r by <rule>          (the ceiling)
    OR-teacher-r<r>    teacher rescues r AND evicts by teacher        (double oracle)
    RND-<rule>-r<r>    random rescue, same eviction                   (content-free)

Pre-registered primary arm: **MG-lowimp-r16**.  Chosen before this grid ran, on
two grounds that are independent of any accuracy number: `lowimp` is the
cheapest deployable rule (it reads the EADP score the base selector already
computed and never touches the similarity submatrix), and the pilot's oracle
pair put `lowimp` eviction 0.20 macro from the *teacher* eviction oracle
(73.28 vs 73.47), which says the eviction axis is already saturated there.  All
other arms are secondary; the verdict reads the primary.

Usage
    python scripts/discovery/m3_accuracy.py --arms MG0 MG-lowimp-r16
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

import common                                                        # noqa: E402
from common import OUTPUT_DIR, eadp_model_name                       # noqa: E402
from m2_accuracy import DS_ORDER, MAX_NEW, heldout                   # noqa: E402
from m2_gdep import MODE_PRELLM, GDEPConfig, GDEPEngine, dump_json   # noqa: E402
from m3_common import (BASE_SELECTOR, BUDGET, D_VIS, FEATURES,       # noqa: E402
                       MissStudent, feature_index, install_missguard,
                       load_teacher, teacher_topk)
from scoring import per_sample_hits                                  # noqa: E402

PRIMARY_ARM = "MG-lowimp-r16"


def parse_arm(arm: str) -> dict:
    """Arm name -> a miss spec (see the module docstring)."""
    if arm in ("MG0", "B2"):
        return dict(source="none", r=0)
    src, rule, rr = arm.split("-")
    src = {"MG": "learned", "OR": "teacher", "RND": "random"}[src]
    return dict(source=src, rule=rule, r=int(rr[1:]))


def make_engine(model, miss: dict):
    """B2's engine, with the audit-and-correct pruner installed on BOTH hooks."""
    cfg = GDEPConfig(mode=MODE_PRELLM, budget=BUDGET, selector=BASE_SELECTOR,
                     tag="M3")
    eng = GDEPEngine.from_checkpoint(cfg, model=model, max_new_tokens=MAX_NEW)
    return install_missguard(eng, model, miss)


def load_student(tag: str, device):
    ck = torch.load(os.path.join(OUTPUT_DIR, f"{tag}.pt"), map_location="cpu",
                    weights_only=False)
    # A checkpoint records the features it was trained on.  Appending to
    # FEATURES must not invalidate an older student, so the checkpoint's list is
    # resolved to column indices instead of being compared for equality.
    ck["feature_idx"] = feature_index(ck["features"]).to(device)
    assert ck["feature_idx"].numel() == ck["d_hand"], \
        f"checkpoint d_hand {ck['d_hand']} != {len(ck['features'])} features"
    m = MissStudent(ck["d_hand"], ck["d_vis"])
    m.load_state_dict(ck["state_dict"])
    m.eval().to(device)
    for k in ("mu_hand", "sd_hand", "mu_vis", "sd_vis"):
        ck[k] = torch.from_numpy(np.asarray(ck[k], np.float32)).to(device)
    return ck, m


def stage(model, args):
    torch.set_grad_enabled(False)
    items = heldout(model)
    teacher = load_teacher()
    for it in items:
        assert it["key"] in teacher, it["key"]
    ck, student = load_student(args.student, next(model.model.parameters()).device)
    print(f"[student] {args.student}: {ck['config_name']}  target={ck['target']}  "
          f"z={ck['per_instance_z']}  val overlap@16 {ck['val_overlap16']:.4f}")

    bank = None
    if args.bank_gate:
        bz = np.load(os.path.join(OUTPUT_DIR, f"{args.bank_gate}.npz"),
                     allow_pickle=False)
        bank = {str(k): np.asarray(bz["s0"][i])
                for i, k in enumerate(bz["key"])}

    rec = dict(config=vars(args), n_instances=len(items), arms={},
               keys=[i["key"] for i in items],
               ds_order=[i["ds"] for i in items],
               primary_arm=PRIMARY_ARM,
               student=dict(tag=args.student, config=ck["config_name"],
                            target=ck["target"], per_instance_z=ck["per_instance_z"],
                            val_overlap16=ck["val_overlap16"],
                            features=ck["features"]),
               max_new_tokens=MAX_NEW)
    out_path = os.path.join(OUTPUT_DIR, f"{args.tag}.json")
    if args.resume and os.path.exists(out_path):
        prev = json.load(open(out_path)).get("arms", {})
        keep = {k: v for k, v in prev.items() if "error" not in v}
        rec["arms"].update(keep)
        print(f"[resume] keeping {sorted(keep)}")

    sfx = "" if args.student == "m3_miss" else "@" + args.student.replace("m3_miss_", "")
    for arm0 in args.arms:
        arm = arm0 + sfx          # the run key carries the student, so a second
        miss = parse_arm(arm0)    # seed cannot overwrite the first one's record
        if miss.get("source") == "learned":
            miss.update(student=student, mu_hand=ck["mu_hand"], sd_hand=ck["sd_hand"],
                        mu_vis=ck["mu_vis"], sd_vis=ck["sd_vis"],
                        feature_idx=ck["feature_idx"],
                        per_instance_z=bool(ck["per_instance_z"]))
        if miss.get("source") == "random":
            miss["rng_seed"] = args.rng_seed
        print(f"\n=== {arm}  {miss.get('source')} r={miss.get('r')} "
              f"rule={miss.get('rule')} ===", flush=True)
        t0 = time.time()
        try:
            eng = make_engine(model, miss)
            preds, meta, n_fail = [], [], 0
            for j, it in enumerate(items):
                if miss.get("source") == "teacher":
                    eng.pruner.miss["teacher"] = teacher[it["key"]]
                if miss.get("source") == "random":
                    eng.pruner.miss["key"] = it["key"]
                err = None
                try:
                    out = eng.run(it["msg"], it["ds"], MAX_NEW)
                    p, info = out["prediction"], out["info"]
                except Exception:
                    err = traceback.format_exc()[-800:]
                    traceback.print_exc()
                    p, info, out = "", {}, {}
                    n_fail += 1
                preds.append(p)
                ms = eng.pruner.read_miss_ms()
                lm = dict(eng.pruner.last_miss)
                meta.append(dict(key=it["key"], ds=it["ds"],
                                 n_kept=info.get("n_kept"),
                                 context_len=info.get("context_len"),
                                 n_decode=out.get("n_decode"),
                                 missguard_ms=ms,
                                 s0_idx=lm.get("s0_idx"),
                                 n_evicted=lm.get("n_evicted"),
                                 rescue_idx=lm.get("rescue_idx"),
                                 evict_idx=lm.get("evict_idx"),
                                 rescue_in_teacher256=lm.get("rescue_in_teacher256"),
                                 identity=lm.get("identity"), error=err))
                if (j + 1) % 25 == 0:
                    print(f"  {j+1}/{len(items)}  {time.time()-t0:.0f}s", flush=True)
            if n_fail:
                # A crashed instance is NOT a wrong answer: it must never be
                # persisted as an arm result (the pilot's six 0.2-macro arms
                # were exactly this).
                raise RuntimeError(
                    f"{n_fail}/{len(items)} instances raised; first traceback:\n"
                    + next(m["error"] for m in meta if m["error"]))
            hits = {}
            for ds in DS_ORDER:
                sel = [i for i, it in enumerate(items) if it["ds"] == ds]
                h = per_sample_hits(ds, [items[i]["row"] for i in sel],
                                    [preds[i] for i in sel])
                hits[ds] = [float(x) for x in h]
            macro = float(np.mean([np.mean(hits[ds]) for ds in DS_ORDER]))
            r = dict(arm=arm,
                     miss={k: v for k, v in miss.items()
                           if k not in ("student", "teacher", "mu_hand", "sd_hand",
                                        "mu_vis", "sd_vis", "feature_idx")},
                     per_benchmark={ds: dict(acc_pct=float(np.mean(hits[ds]) * 100),
                                             n=len(hits[ds]), hits=hits[ds])
                                    for ds in DS_ORDER},
                     macro_pct=macro * 100, predictions=preds, per_image_meta=meta,
                     wall_seconds=float(time.time() - t0),
                     missguard_ms_median=float(np.median(
                         [m["missguard_ms"] for m in meta])))
            rec["arms"][arm] = r
            print(f"  {arm}: TextVQA {r['per_benchmark']['TextVQA_VAL']['acc_pct']:.3f}"
                  f"  DocVQA {r['per_benchmark']['DocVQA_VAL']['acc_pct']:.3f}"
                  f"  OCRBench {r['per_benchmark']['OCRBench']['acc_pct']:.3f}"
                  f"  macro {r['macro_pct']:.3f}  ms {r['missguard_ms_median']:.3f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
            del eng
            torch.cuda.empty_cache()
        except Exception:
            traceback.print_exc()
            rec["arms"][arm] = dict(arm=arm, error=traceback.format_exc()[-2000:])
        dump_json(f"{args.tag}.json", rec)

    _gates(rec, bank, teacher)
    dump_json(f"{args.tag}.json", rec)
    print(f"\n[done] -> {args.tag}.json")
    return rec


def _gates(rec, bank, teacher):
    """Identity gates.  A gate that cannot run says so; it never passes silently."""
    g = {}
    # (1) r=0 must be the stored B2 arm, hit-for-hit AND string-for-string.
    path = os.path.join(OUTPUT_DIR, "m2_accuracy.json")
    b2 = json.load(open(path))["arms"].get("B2") if os.path.exists(path) else None
    mg0 = rec["arms"].get("MG0")
    if b2 and mg0 and "error" not in mg0:
        agree = sum(int((np.asarray(b2["per_benchmark"][ds]["hits"], float)
                         == np.asarray(mg0["per_benchmark"][ds]["hits"], float)).sum())
                    for ds in DS_ORDER)
        same = sum(1 for x, y in zip(b2["predictions"], mg0["predictions"]) if x == y)
        g["MG0_is_B2"] = dict(hit_agreement=f"{agree}/150",
                              prediction_string_agreement=f"{same}/150",
                              passed=bool(agree == 150))
    else:
        g["MG0_is_B2"] = dict(checked=False, reason="MG0 or stored B2 missing")

    # (2) live s0 must equal the bank's s0 for every instance of every arm
    #     that recorded one -- proves train-time and serve-time base sets agree.
    if bank is not None:
        bad, checked = [], 0
        for arm, r in rec["arms"].items():
            if "error" in r or r.get("miss", {}).get("r", 0) == 0:
                continue
            for m in r["per_image_meta"]:
                if m.get("s0_idx") is None:
                    continue
                checked += 1
                if not np.array_equal(np.sort(np.asarray(m["s0_idx"])),
                                      np.sort(bank[m["key"]])):
                    bad.append((arm, m["key"]))
        g["live_s0_equals_bank_s0"] = dict(checked=checked, mismatches=bad[:10],
                                           n_mismatch=len(bad),
                                           passed=bool(checked and not bad))
    else:
        g["live_s0_equals_bank_s0"] = dict(checked=False, reason="no --bank-gate")

    # (3) every OR arm's rescue set must BE the teacher's top-r dropped tokens.
    #     A mis-keyed teacher vector would turn the ceiling into a shuffle
    #     control, which is the one way the REFUTED branch could fire wrongly.
    bad, ties, checked = [], [], 0
    for arm, r in rec["arms"].items():
        if "error" in r or not arm.startswith("OR-"):
            continue
        rr = r["miss"]["r"]
        if rr == 0:
            continue
        for m in r["per_image_meta"]:
            if m.get("rescue_idx") is None or m.get("s0_idx") is None:
                continue
            checked += 1
            g2 = teacher[m["key"]]
            want = teacher_topk(g2, np.asarray(m["s0_idx"]), 1024)[:rr]
            got = np.asarray(m["rescue_idx"])
            # Tie-tolerant: what the claim needs is that the rescue is
            # teacher-OPTIMAL, and where the r-th and (r+1)-th teacher scores
            # are bit-identical the two arg-sorts may pick either token. Compare
            # the score multiset, and record index disagreement separately.
            same_scores = np.allclose(np.sort(g2[got]), np.sort(g2[want]), atol=0)
            if not same_scores:
                bad.append((arm, m["key"]))
            elif not np.array_equal(np.sort(got), np.sort(want)):
                ties.append((arm, m["key"]))
    g["oracle_rescue_is_teacher_topr"] = dict(
        checked=checked, n_not_teacher_optimal=len(bad), examples=bad[:5],
        n_tie_broken=len(ties), tie_examples=ties[:5],
        note="tie_broken instances have bit-identical teacher scores at the r-th "
             "boundary; numpy and torch arg-sort may pick either token, so the "
             "rescue is teacher-optimal but not index-identical there",
        passed=bool(checked and not bad))
    for k, v in g.items():
        print(f"[gate] {k}: {v.get('passed', v)}")
    rec["gates"] = g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["MG0"])
    ap.add_argument("--tag", default="m3_accuracy")
    ap.add_argument("--student", default="m3_miss")
    ap.add_argument("--bank-gate", default="m3_bank")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rng-seed", type=int, default=20260924)
    args = ap.parse_args()
    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0),
                              max_new_tokens=MAX_NEW)
    model.model.eval()
    stage(model, args)


if __name__ == "__main__":
    main()
