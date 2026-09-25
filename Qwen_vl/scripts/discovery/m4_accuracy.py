"""
M4-v0 step 2 -- the held-out 150 grid for REC and its controls.

Every arm runs through the SAME engine the B0/B1/B2 numbers came from
(`m2_gdep.GDEPEngine`, mode `prellm`, selector `block8`): the only thing that
changes between an arm and B2 is the pruner's post-selection treatment of the
tokens it rejected, applied inside the pruner call -- inside the measured
region, before any decoder layer sees the sequence.

Arms (all exactly 256 visual tokens unless marked)
    REC0              r=0 identity: must reproduce the stored B2 arm hit-for-hit
    REC-r8/16/32      residual-weighted capsules, spatial partition, tau=0.05
    MEAN-r16          uniform weights inside the cell   -- plain mean merge
    IMP-r16           EADP-importance weights, ESS-matched to REC-r16
    SHUF-r16          residual weights taken from ANOTHER instance (content-free)
    ANCH-r16          capsule = mean of the RETAINED tokens in the cell
    FPS-r16           residual weights, feature-space (farthest-point) grouping
    NORM-r16          residual weights + capsule rescaled to pooled-token norm
    DIAG-evict-r16    the eviction alone, nothing put back -- 240 tokens.
                      A diagnostic, never a candidate: it exists so that "what
                      the capsules add" separates from "what the eviction costs".

Pre-registered primary arm: **REC-r16**.  Fixed before the grid ran, on three
grounds that do not read an accuracy number: it is the midpoint of the r grid;
the offline diagnostic put its effective sample size at 10.0 of the 64 tokens it
pools (neither one-hot nor uniform); and r=16 is where M3-v0's teacher oracle
was strongest.  All other arms are secondary; the verdict reads the primary and
the controls around it.

Usage
    python scripts/discovery/m4_accuracy.py --arms REC0 REC-r16
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
from m3_common import evict_t                                        # noqa: E402
from m4_common import (BASE_SELECTOR, BUDGET, EVICT_RULE, N_VIS,     # noqa: E402
                       SHUF_OFFSET, TAU, install_rec, partition_ids)
from scoring import per_sample_hits                                  # noqa: E402

PRIMARY_ARM = "REC-r16"
UMAPS = "m4_offline_umaps.npz"
# tau_imp fixed by the offline ESS-matching rule (m4_offline.py): the importance
# softmax that concentrates on the same effective 10.04 tokens per cell as the
# residual softmax at tau=0.05.  Fixed before any generation ran.
TAU_IMP = 0.0361

# arm -> rec spec.  `r` comes from the name; everything else is frozen here.
ARM_SPEC = {
    "REC0":   dict(mode="none"),
    "REC":    dict(mode="capsule", weights="residual", assign="spatial"),
    "MEAN":   dict(mode="capsule", weights="mean", assign="spatial"),
    "IMP":    dict(mode="capsule", weights="imp", assign="spatial",
                   tau_imp=TAU_IMP),
    "SHUF":   dict(mode="capsule", weights="shuf", assign="spatial"),
    "ANCH":   dict(mode="capsule", weights="anchor_mean", assign="spatial"),
    "FPS":    dict(mode="capsule", weights="residual", assign="fps"),
    "NORM":   dict(mode="capsule", weights="residual", assign="spatial",
                   norm_restore=True),
    "EVICT":  dict(mode="evict"),
}


def parse_arm(arm: str) -> dict:
    """Arm name -> a rec spec (see ARM_SPEC)."""
    if arm in ("REC0", "B2"):
        return dict(ARM_SPEC["REC0"], r=0)
    parts = arm.split("-")
    if len(parts) != 2 or not parts[1].startswith("r"):
        raise KeyError(f"unparseable arm {arm!r}; expected <FAMILY>-r<k> "
                       f"with FAMILY in {sorted(ARM_SPEC)}")
    fam, rr = parts
    return dict(ARM_SPEC[fam], r=int(rr[1:]))


def make_engine(model, rec: dict):
    cfg = GDEPConfig(mode=MODE_PRELLM, budget=BUDGET, selector=BASE_SELECTOR,
                     tag="M4")
    eng = GDEPEngine.from_checkpoint(cfg, model=model, max_new_tokens=MAX_NEW)
    return install_rec(eng, model, rec)


def load_umaps():
    """Per-instance residual maps, keyed.  Only the SHUF arm reads this file."""
    path = os.path.join(OUTPUT_DIR, UMAPS)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing -- run m4_offline.py first; the SHUF arm must "
            f"never fall back to a silently different control")
    z = np.load(path, allow_pickle=False)
    return {str(k): np.asarray(z["u_r16"][i], dtype=np.float32)
            for i, k in enumerate(z["key"])}


def stage(model, args):
    torch.set_grad_enabled(False)
    items = heldout(model)
    umaps = load_umaps() if any(a.startswith("SHUF") for a in args.arms) else None
    keys = [it["key"] for it in items]
    missing = [k for k in keys if umaps is not None and k not in umaps]
    if missing:
        raise KeyError(f"{len(missing)} held-out keys absent from {UMAPS}: "
                       f"{missing[:5]}")

    rec_out = dict(config=vars(args), n_instances=len(items), arms={},
                   keys=keys, ds_order=[i["ds"] for i in items],
                   primary_arm=PRIMARY_ARM, tau=TAU, tau_imp=TAU_IMP,
                   evict_rule=EVICT_RULE, shuf_offset=SHUF_OFFSET,
                   max_new_tokens=MAX_NEW)
    out_path = os.path.join(OUTPUT_DIR, f"{args.tag}.json")
    if args.resume and os.path.exists(out_path):
        prev = json.load(open(out_path)).get("arms", {})
        keep = {k: v for k, v in prev.items() if "error" not in v}
        rec_out["arms"].update(keep)
        print(f"[resume] keeping {sorted(keep)}")

    for arm0 in args.arms:
        rec = parse_arm(arm0)
        if rec.get("weights") == "shuf":
            rec["shuf_u"] = None            # replaced per instance below
        print(f"\n=== {arm0}  mode={rec['mode']} r={rec.get('r')} "
              f"weights={rec.get('weights')} assign={rec.get('assign')} ===",
              flush=True)
        t0 = time.time()
        try:
            eng = make_engine(model, rec)
            preds, meta, n_fail = [], [], 0
            for j, it in enumerate(items):
                if rec.get("weights") == "shuf":
                    src = keys[(j + SHUF_OFFSET) % len(keys)]
                    eng.pruner.rec["shuf_u"] = umaps[src]
                    eng.pruner.rec["shuf_src"] = src
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
                lm = dict(eng.pruner.last_miss)
                meta.append(dict(key=it["key"], ds=it["ds"],
                                 n_kept=info.get("n_kept"),
                                 context_len=info.get("context_len"),
                                 n_decode=out.get("n_decode"),
                                 rec_ms=eng.pruner.read_miss_ms(),
                                 s0_idx=lm.get("s0_idx"),
                                 a_idx=lm.get("a_idx"),
                                 evict_idx=lm.get("evict_idx"),
                                 n_evicted=lm.get("n_evicted"),
                                 n_capsules=lm.get("n_capsules"),
                                 cap_stats=lm.get("cap_stats"),
                                 shuf_src=lm.get("shuf_src"),
                                 identity=lm.get("identity"), error=err))
                if (j + 1) % 25 == 0:
                    print(f"  {j+1}/{len(items)}  {time.time()-t0:.0f}s", flush=True)
            if n_fail:
                # A crashed instance is NOT a wrong answer: it must never be
                # persisted as an arm result (the M3 pilot's six 0.2-macro arms
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
            r = dict(arm=arm0, rec={k: v for k, v in rec.items()
                                    if k != "shuf_u"},
                     per_benchmark={ds: dict(acc_pct=float(np.mean(hits[ds]) * 100),
                                             n=len(hits[ds]), hits=hits[ds])
                                    for ds in DS_ORDER},
                     macro_pct=macro * 100, predictions=preds, per_image_meta=meta,
                     wall_seconds=float(time.time() - t0),
                     rec_ms_median=float(np.median([m["rec_ms"] for m in meta])))
            rec_out["arms"][arm0] = r
            print(f"  {arm0}: TextVQA {r['per_benchmark']['TextVQA_VAL']['acc_pct']:.3f}"
                  f"  DocVQA {r['per_benchmark']['DocVQA_VAL']['acc_pct']:.3f}"
                  f"  OCRBench {r['per_benchmark']['OCRBench']['acc_pct']:.3f}"
                  f"  macro {r['macro_pct']:.3f}  ms {r['rec_ms_median']:.3f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
            del eng
            torch.cuda.empty_cache()
        except Exception:
            traceback.print_exc()
            rec_out["arms"][arm0] = dict(arm=arm0,
                                         error=traceback.format_exc()[-2000:])
        dump_json(f"{args.tag}.json", rec_out)

    _gates(rec_out, model)
    dump_json(f"{args.tag}.json", rec_out)
    print(f"\n[done] -> {args.tag}.json")
    return rec_out


def _gates(rec_out, model):
    """Identity gates.  A gate that cannot run says so; it never passes silently."""
    g = {}
    arms = rec_out["arms"]

    # (1) r=0 must be the stored B2 arm, hit-for-hit AND string-for-string.
    path = os.path.join(OUTPUT_DIR, "m2_accuracy.json")
    b2 = json.load(open(path))["arms"].get("B2") if os.path.exists(path) else None
    rec0 = arms.get("REC0")
    if b2 and rec0 and "error" not in rec0:
        agree = sum(int((np.asarray(b2["per_benchmark"][ds]["hits"], float)
                         == np.asarray(rec0["per_benchmark"][ds]["hits"], float)).sum())
                    for ds in DS_ORDER)
        same = sum(1 for x, y in zip(b2["predictions"], rec0["predictions"]) if x == y)
        g["REC0_is_B2"] = dict(hit_agreement=f"{agree}/150",
                               prediction_string_agreement=f"{same}/150",
                               passed=bool(agree == 150))
    else:
        g["REC0_is_B2"] = dict(checked=False, reason="REC0 or stored B2 missing")

    # (2) the budget.  Every capsule arm must return exactly 256 visual tokens
    #     and its A must be exactly S0 minus the r evicted -- checked on the
    #     live record, so a partition or pooling bug that silently changed the
    #     budget cannot be read as an accuracy result.
    bad, checked, diag = [], 0, []
    for arm, r in arms.items():
        if "error" in r:
            continue
        rr = int(r["rec"].get("r", 0) or 0)
        is_evict = r["rec"]["mode"] == "evict"
        for m in r["per_image_meta"]:
            if rr == 0 or m.get("n_kept") is None:
                continue
            checked += 1
            # An evict-mode arm returns 256 - r by design; every other arm
            # returns exactly the budget.  Both are checked against their own
            # contract, so a partition or pooling bug that silently changed the
            # budget cannot be read as an accuracy result.
            want = BUDGET - rr if is_evict else BUDGET
            s0 = set(m["s0_idx"] or [])
            a = set(m["a_idx"] or [])
            ev = set(m["evict_idx"] or [])
            ok = (m["n_kept"] == want and len(s0) == BUDGET
                  and len(a) == BUDGET - rr and a <= s0
                  and a | ev == s0 and not (a & ev))
            if not ok:
                bad.append((arm, m["key"], m["n_kept"], len(a), len(ev)))
            if is_evict:
                diag.append(arm)
    g["budget_and_partition"] = dict(
        checked=checked, n_bad=len(bad), examples=bad[:5],
        diagnostic_arms=sorted(set(diag)),
        note="evict-mode arms are checked against 256-r on purpose; they are "
             "diagnostics, not candidates",
        passed=bool(checked and not bad))

    # (3) the capsules must be pooled from real tokens: an all-empty partition
    #     would make every capsule the same fallback vector and would score like
    #     a constant-image arm.  A cell can legitimately come up empty -- it
    #     means the maxred eviction happened to leave a whole cell inside A --
    #     and the fallback handles it, but it must stay rare, so the gate
    #     bounds the RATE rather than demanding zero.
    empt, ess, offenders = [], [], []
    for arm, r in arms.items():
        if "error" in r or r["rec"].get("mode") != "capsule":
            continue
        for m in r["per_image_meta"]:
            cs = m.get("cap_stats") or {}
            n_e = int(cs.get("n_empty", -1))
            empt.append(n_e)
            if n_e:
                offenders.append((arm, m["key"], n_e))
            if cs.get("ess") is not None:
                ess.append(float(cs["ess"]))
    rate = float(sum(empt) / len(empt)) if empt else 1.0
    g["capsules_non_degenerate"] = dict(
        n_empty_total=int(sum(empt)), n_checked=len(empt),
        n_instances_with_empty=len(offenders), examples=offenders[:5],
        empty_rate=rate, max_rate=0.01,
        ess_median=float(np.median(ess)) if ess else None,
        passed=bool(empt and rate <= 0.01))

    # (4) the live S0 must equal the bank's S0 -- proves train/serve-time and
    #     offline/online base sets agree, which every offline claim rests on.
    bank_path = os.path.join(OUTPUT_DIR, "m3_bank.npz")
    if os.path.exists(bank_path):
        bz = np.load(bank_path, allow_pickle=False)
        bank = {str(k): np.asarray(bz["s0"][i]) for i, k in enumerate(bz["key"])}
        bad2, n = [], 0
        for arm, r in arms.items():
            if "error" in r:
                continue
            for m in r["per_image_meta"]:
                if m.get("s0_idx") is None or m["key"] not in bank:
                    continue
                n += 1
                if not np.array_equal(np.sort(np.asarray(m["s0_idx"])),
                                      np.sort(bank[m["key"]])):
                    bad2.append((arm, m["key"]))
        g["live_s0_equals_bank_s0"] = dict(checked=n, n_mismatch=len(bad2),
                                           examples=bad2[:5],
                                           passed=bool(n and not bad2))
    else:
        g["live_s0_equals_bank_s0"] = dict(checked=False, reason="no bank")

    # (5) the SHUF control must actually be content-free: its source instance
    #     must never be the instance itself.
    self_src = [(a, m["key"]) for a, r in arms.items()
                for m in r.get("per_image_meta", [])
                if m.get("shuf_src") is not None and m["shuf_src"] == m["key"]]
    n_shuf = sum(1 for a, r in arms.items() for m in r.get("per_image_meta", [])
                 if m.get("shuf_src") is not None)
    g["shuf_is_content_free"] = dict(checked=n_shuf, n_self_source=len(self_src),
                                     passed=bool(n_shuf and not self_src))

    for k, v in g.items():
        print(f"[gate] {k}: {v.get('passed', v)}")
    rec_out["gates"] = g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["REC0"])
    ap.add_argument("--tag", default="m4_accuracy")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0),
                              max_new_tokens=MAX_NEW)
    model.model.eval()
    stage(model, args)


if __name__ == "__main__":
    main()
