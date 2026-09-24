"""
M3-v2 step 4 -- the held-out 150 grid, for auditors that passed the head gate.

Runs only if `m3v2_proxy` passed the brief's §7 gate.  Every arm goes through
the SAME engine the B0/B1/B2 numbers came from (`m2_gdep.GDEPEngine`, mode
`prellm`, selector `block8`), the same held-out 150, the same greedy decode and
the same scorer; the only thing that differs between arms is the auditor.

Eviction is `maxred` for every non-identity arm, and r is 8 or 16 -- the two
values the brief fixes for this stage.  That is deliberate: the eviction rule
and r are held constant so the auditor is the only variable.

Arms
    MG0                     r=0 identity: must reproduce the stored B2 arm
    V0-maxred-r<r>          the v0 token-local student
    V2-<cfg>-maxred-r<r>    a HeadAuditor checkpoint (<cfg> = h1C_s0, ...)
    RND-maxred-r<r>         content-free rescue, same eviction
    OR-maxred-r<r>          teacher rescue, same eviction (the ceiling)

Usage
    python scripts/discovery/m3v2_accuracy.py --arms MG0 V2-h1C_s0-maxred-r16
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
from m3_common import (BASE_SELECTOR, BUDGET, MissStudent,           # noqa: E402
                       feature_index, install_missguard, load_teacher,
                       teacher_topk)
from m3v2_common import build_auditor, install_auditor               # noqa: E402
from scoring import per_sample_hits                                  # noqa: E402

EVICT = "maxred"


def parse_arm(arm: str) -> dict:
    """Arm name -> a miss spec.

    `MG0`            identity
    `RND-maxred-r8`  random rescue
    `OR-maxred-r16`  teacher rescue
    `V0-maxred-r16`  the v0 student (`m3_miss.pt`)
    `V2-h1C_s0-maxred-r16`  the HeadAuditor checkpoint `m3v2_auditor_h1C_s0.pt`
    """
    if arm in ("MG0", "B2"):
        return dict(source="none", r=0)
    # parse from the RIGHT: the rule and r are always the last two fields, and
    # the head may itself contain a '-' (nothing does today, but the checkpoint
    # name is free-form).
    parts = arm.split("-")
    rule, rr = parts[-2], int(parts[-1][1:])
    head, cfg = parts[0], "-".join(parts[1:-2])
    if head == "RND":
        return dict(source="random", rule=rule, r=rr)
    if head == "OR":
        return dict(source="teacher", rule=rule, r=rr)
    if head == "V0":
        return dict(source="learned", student_tag="m3_miss", rule=rule, r=rr)
    if head == "V2":
        return dict(source="learned2", ckpt=f"m3v2_auditor_{cfg}",
                    rule=rule, r=rr)
    raise KeyError(arm)


def load_v0(tag: str, device):
    ck = torch.load(os.path.join(OUTPUT_DIR, f"{tag}.pt"), map_location="cpu",
                    weights_only=False)
    ck["feature_idx"] = feature_index(ck["features"]).to(device)
    m = MissStudent(ck["d_hand"], ck["d_vis"])
    m.load_state_dict(ck["state_dict"])
    m.eval().to(device)
    for k in ("mu_hand", "sd_hand", "mu_vis", "sd_vis"):
        ck[k] = torch.from_numpy(np.asarray(ck[k], np.float32)).to(device)
    return ck, m


def load_v2(tag: str, device):
    ck = torch.load(os.path.join(OUTPUT_DIR, f"{tag}.pt"), map_location="cpu",
                    weights_only=False)
    m = build_auditor(ck, device)
    for k in ("mu_vis", "sd_vis", "mu_txt", "sd_txt", "mu_hand", "sd_hand"):
        ck[k] = torch.from_numpy(np.asarray(ck[k], np.float32)).to(device)
    return ck, m


def make_engine(model, miss: dict):
    cfg = GDEPConfig(mode=MODE_PRELLM, budget=BUDGET, selector=BASE_SELECTOR,
                     tag="M3V2")
    eng = GDEPEngine.from_checkpoint(cfg, model=model, max_new_tokens=MAX_NEW)
    return install_auditor(eng, model, miss)


def stage(model, args):
    torch.set_grad_enabled(False)
    items = heldout(model)
    teacher = load_teacher()
    for it in items:
        assert it["key"] in teacher, it["key"]
    dev = next(model.model.parameters()).device

    cache = {}
    bank = None
    if args.bank_gate:
        bz = np.load(os.path.join(OUTPUT_DIR, f"{args.bank_gate}.npz"),
                     allow_pickle=False)
        bank = {str(k): np.asarray(bz["s0"][i]) for i, k in enumerate(bz["key"])}

    rec = dict(config=vars(args), n_instances=len(items), arms={},
               keys=[i["key"] for i in items], ds_order=[i["ds"] for i in items],
               evict_rule=EVICT, max_new_tokens=MAX_NEW)
    out_path = os.path.join(OUTPUT_DIR, f"{args.tag}.json")
    if args.resume and os.path.exists(out_path):
        prev = json.load(open(out_path)).get("arms", {})
        keep = {k: v for k, v in prev.items() if "error" not in v}
        rec["arms"].update(keep)
        print(f"[resume] keeping {sorted(keep)}")

    for arm in args.arms:
        miss = parse_arm(arm)
        if miss.get("source") == "learned":
            tag = miss.pop("student_tag")
            ck, student = load_v0(tag, dev)
            miss.update(student=student, mu_hand=ck["mu_hand"], sd_hand=ck["sd_hand"],
                        mu_vis=ck["mu_vis"], sd_vis=ck["sd_vis"],
                        feature_idx=ck["feature_idx"],
                        per_instance_z=bool(ck["per_instance_z"]))
        elif miss.get("source") == "learned2":
            tag = miss.pop("ckpt")
            ck, student = load_v2(tag, dev)
            cache[tag] = (ck, student)
            miss.update(student=student, **{k: ck[k] for k in
                                            ("mu_vis", "sd_vis", "mu_txt", "sd_txt",
                                             "mu_hand", "sd_hand")})
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
                                 identity=lm.get("identity"), error=err))
                if (j + 1) % 25 == 0:
                    print(f"  {j+1}/{len(items)}  {time.time()-t0:.0f}s", flush=True)
            if n_fail:
                # A crashed instance is NOT a wrong answer (the M3-v0 pilot
                # persisted six crashed arms as 0.2 macro; never again).
                raise RuntimeError(
                    f"{n_fail}/{len(items)} instances raised; first traceback:\n"
                    + next(m["error"] for m in meta if m["error"]))
            hits = {}
            for ds in DS_ORDER:
                sel = [i for i, it in enumerate(items) if it["ds"] == ds]
                hits[ds] = [float(x) for x in per_sample_hits(
                    ds, [items[i]["row"] for i in sel], [preds[i] for i in sel])]
            macro = float(np.mean([np.mean(hits[ds]) for ds in DS_ORDER]))
            r = dict(arm=arm,
                     miss={k: v for k, v in miss.items()
                           if k not in ("student", "teacher", "mu_hand", "sd_hand",
                                        "mu_vis", "sd_vis", "mu_txt", "sd_txt",
                                        "feature_idx")},
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
    """The same identity gates the v0 grid ran, plus the v2-specific one."""
    g = {}
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
            if not np.allclose(np.sort(g2[got]), np.sort(g2[want]), atol=0):
                bad.append((arm, m["key"]))
            elif not np.array_equal(np.sort(got), np.sort(want)):
                ties.append((arm, m["key"]))
    g["oracle_rescue_is_teacher_topr"] = dict(
        checked=checked, n_not_teacher_optimal=len(bad), examples=bad[:5],
        n_tie_broken=len(ties), tie_examples=ties[:5],
        passed=bool(checked and not bad))

    # v2 only: a learned2 arm's live rescue must equal the rescue the frozen
    # checkpoint picks from the bank's dropped set.  This is the v2 analogue of
    # v0's `live_s0_equals_bank_s0`, and it is the gate that catches a pruner
    # whose standardisation or fp16 quantisation drifted from the trainer's.
    from m3v2_proxy import v2_scorer, v0_scores, _v0_scorer        # noqa: E402
    from m3v2_common import BankV2                                  # noqa: E402
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bkv2 = BankV2()
    idx_of = {k: i for i, k in enumerate(bkv2.key)}
    scorers, checked, bad2, tie2 = {}, 0, [], []
    for arm, r in rec["arms"].items():
        if "error" in r or r.get("miss", {}).get("source") not in ("learned", "learned2"):
            continue
        rr = r["miss"]["r"]
        spec = scorers.get(arm)
        if spec is None:
            if r["miss"]["source"] == "learned":
                ck = torch.load(os.path.join(OUTPUT_DIR, "m3_miss.pt"),
                                map_location="cpu", weights_only=False)
                m = MissStudent(ck["d_hand"], ck["d_vis"])
                m.load_state_dict(ck["state_dict"])
                ck["student"] = m
                spec = scorers[arm] = ("v0", _v0_scorer(bkv2, ck, dev))
            else:
                ck = torch.load(os.path.join(OUTPUT_DIR,
                                             f"{r['miss']['ckpt']}.pt"),
                                map_location="cpu", weights_only=False)
                spec = scorers[arm] = ("v2", v2_scorer(bkv2, ck, dev))
        kind, fn = spec
        for m in r["per_image_meta"]:
            i = idx_of.get(m["key"])
            if i is None or m.get("rescue_idx") is None:
                continue
            checked += 1
            if kind == "v0":
                sc = v0_scores(bkv2, i, fn, dev)
            else:
                sc = fn(i)
            d = bkv2.drop[i]
            got = np.sort(np.asarray(m["rescue_idx"]))
            want = np.sort(d[np.argsort(-sc, kind="stable")[:rr]])
            if np.array_equal(got, want):
                continue
            if np.allclose(np.sort(bkv2.g2[i][got]), np.sort(bkv2.g2[i][want]), atol=0):
                tie2.append((arm, m["key"]))
            else:
                bad2.append((arm, m["key"]))
    g["live_rescue_equals_bank_rescue"] = dict(
        checked=checked, n_mismatch=len(bad2), examples=bad2[:5],
        n_tie_broken=len(tie2), passed=bool(checked and not bad2))
    for k, v in g.items():
        print(f"[gate] {k}: {v.get('passed', v)}")
    rec["gates"] = g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["MG0"])
    ap.add_argument("--tag", default="m3v2_accuracy")
    ap.add_argument("--bank-gate", default="m3_bank_v1")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rng-seed", type=int, default=20260924)
    args = ap.parse_args()
    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0),
                              max_new_tokens=MAX_NEW)
    model.model.eval()
    stage(model, args)


if __name__ == "__main__":
    main()
