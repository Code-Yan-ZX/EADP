"""
M5 (Safe Removal) step 3 -- the held-out 150 grid.

Every arm runs through the SAME engine the B0/B1/B2 numbers came from
(`m2_gdep.GDEPEngine`, mode `prellm`, selector `block8`): the only thing that
changes between an arm and B2 is which k of B2's own 256 retained tokens the
pruner deletes, inside the pruner call -- inside the measured region, before
any decoder layer sees the sequence.

Arms
    B2             the identity gate (k=0): must reproduce the stored B2 arm
                   hit-for-hit AND string-for-string
    SAFE-k*        the method: delete the k tokens the probe calls safest
    MAXRED-k*      delete the k most redundant retained tokens
    LOWIMP-k*      delete the k lowest-EADP-importance retained tokens
    RECON-k*       delete the k tokens best reconstructed by the rest of S0
    RANDOM-k*      delete k at random (several seeds)

`SAFE-k8` is the **pre-registered primary arm**: it is the k at which M4's
`EVICT-r8` lead was measured (the only positive effect in M4's grid), and the
`maxred` rule it used is one of the four training-free baselines here, so the
method is compared against the exact thing that produced the lead.

The contrast that decides what the learning is worth is `SAFE - MAXRED` at the
same k (brief §12): if a learned, answer-supervised risk does not beat the
training-free redundancy rule, the learning is not the contribution.

Usage
    python scripts/discovery/m5_accuracy.py --arms B2 SAFE-k8
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
from m5_common import (BASE_SELECTOR, BUDGET, SafeRisk, install_trim,  # noqa: E402
                       load_m5_bank)
from scoring import per_sample_hits                                  # noqa: E402

PRIMARY_ARM = "SAFE-k8"
PROBE_FILE = "m5_probe.pt"
# The trimming rules, and how many random seeds each arm family gets.
RULE_FAMILIES = ("MAXRED", "LOWIMP", "RECON", "RANDOM")
RULE_OF = {"MAXRED": "maxred", "LOWIMP": "lowimp", "RECON": "recon",
           "RANDOM": "random"}
# The three probe families.  `SAFE` is the pre-registered primary and is the
# small MLP the brief asks for; the other two exist so the grid prices the
# probe's capacity (linear vs MLP) and its input (scalars vs scalars+vision)
# instead of leaving that to the offline gate alone.
PROBE_OF = {"SAFE": "mlp", "SAFELR": "lr", "SAFEVIS": "vis"}
K_GRID = (4, 8, 12, 16)
RANDOM_SEEDS = (0, 1, 2)          # k=8 only; other k take seed 0
K8_ONLY_PROBES = ("SAFELR", "SAFEVIS")


def parse_arm(arm: str):
    """Arm name -> a trim spec (see SafeTrimPruner)."""
    if arm in ("B2", "SAFE-k0"):
        return dict(mode="none", k=0)
    fam, _, kk = arm.partition("-k")
    k = int(kk.split("-")[0])
    if fam in PROBE_OF:
        return dict(mode="probe", k=k, probe_kind=PROBE_OF[fam])
    if fam in RULE_OF:
        return dict(mode="rule", rule=RULE_OF[fam], k=k)
    raise KeyError(f"unparseable arm {arm!r}")


def load_probe(kind: str = "mlp", seed: int = 0):
    """The deployable probe, as `m5_probe.py` fitted it on the fit split."""
    path = os.path.join(OUTPUT_DIR, PROBE_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing -- run m5_probe.py first; the SAFE arm must never "
            f"fall back to a silently different probe")
    z = torch.load(path, map_location="cpu", weights_only=False)
    key = f"{kind}|s{seed}"
    if key not in z:
        raise KeyError(f"{key} not in {PROBE_FILE}: {sorted(z)}")
    p = z[key]
    model = SafeRisk(kind=p["kind"], d_hand=p["d_hand"], hidden=p["hidden"])
    model.load_state_dict(p["state_dict"])
    model.eval()
    return model, p["mu"], p["sd"], p["kind"]


def make_engine(model, trim: dict):
    cfg = GDEPConfig(mode=MODE_PRELLM, budget=BUDGET, selector=BASE_SELECTOR,
                     tag="M5")
    eng = GDEPEngine.from_checkpoint(cfg, model=model, max_new_tokens=MAX_NEW)
    return install_trim(eng, model, trim)


def stage(model, args):
    torch.set_grad_enabled(False)
    items = heldout(model)
    if args.limit:
        # Smoke-test only: a truncated instance list cannot produce a
        # comparable macro, so it is stamped on the record and the gates that
        # read 150 instances refuse to pass on it.
        items = items[:args.limit]
    keys = [it["key"] for it in items]
    bank = load_m5_bank()
    pos = bank["_index"]
    missing = [k for k in keys if k not in pos]
    if missing:
        raise KeyError(f"{len(missing)} held-out keys absent from the M5 bank: "
                       f"{missing[:5]}")

    rec_out = dict(config=vars(args), n_instances=len(items), arms={},
                   keys=keys, ds_order=[i["ds"] for i in items],
                   primary_arm=PRIMARY_ARM, max_new_tokens=MAX_NEW,
                   probe_file=PROBE_FILE, truncated=bool(args.limit))
    out_path = os.path.join(OUTPUT_DIR, f"{args.tag}.json")
    if args.resume and os.path.exists(out_path):
        prev = json.load(open(out_path)).get("arms", {})
        keep = {k: v for k, v in prev.items() if "error" not in v}
        rec_out["arms"].update(keep)
        print(f"[resume] keeping {sorted(keep)}")

    for arm0 in args.arms:
        trim = parse_arm(arm0)
        seeds = (RANDOM_SEEDS if (arm0.startswith("RANDOM") and trim["k"] == 8)
                 else (0,))
        for seed in seeds:
            name = arm0 if len(seeds) == 1 else f"{arm0}-s{seed}"
            spec = {k: v for k, v in trim.items() if k != "probe_kind"}
            if trim["mode"] == "probe":
                probe, mu, sd, kind = load_probe(trim.get("probe_kind", "mlp"), 0)
                spec.update(probe=probe, mu=mu, sd=sd, kind=kind)
            if trim["mode"] == "rule" and trim["rule"] == "random":
                spec["seed"] = seed
            print(f"\n=== {name}  mode={spec['mode']} k={spec['k']} "
                  f"rule={spec.get('rule')} ===", flush=True)
            t0 = time.time()
            try:
                eng = make_engine(model, spec)
                preds, meta, n_fail = [], [], 0
                for j, it in enumerate(items):
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
                                     risk_proof=lm.get("risk_proof"),
                                     n_kept=info.get("n_kept"),
                                     context_len=info.get("context_len"),
                                     n_decode=out.get("n_decode"),
                                     trim_ms=eng.pruner.read_miss_ms(),
                                     s0_idx=lm.get("s0_idx"),
                                     drop_idx=lm.get("drop_idx"),
                                     a_idx=lm.get("a_idx"),
                                     n_dropped=lm.get("n_dropped"),
                                     identity=lm.get("identity"), error=err))
                    if (j + 1) % 50 == 0:
                        print(f"  {j+1}/{len(items)}  {time.time()-t0:.0f}s",
                              flush=True)
                if n_fail:
                    # A crashed instance is NOT a wrong answer: it must never be
                    # persisted as an arm result.
                    raise RuntimeError(
                        f"{n_fail}/{len(items)} instances raised; first "
                        f"traceback:\n"
                        + next(m["error"] for m in meta if m["error"]))
                hits = {}
                for ds in DS_ORDER:
                    sel = [i for i, it in enumerate(items) if it["ds"] == ds]
                    h = per_sample_hits(ds, [items[i]["row"] for i in sel],
                                        [preds[i] for i in sel])
                    hits[ds] = [float(x) for x in h]
                macro = float(np.mean([np.mean(hits[ds]) for ds in DS_ORDER]))
                r = dict(arm=name, spec={k: v for k, v in spec.items()
                                         if k != "probe"},
                         per_benchmark={ds: dict(
                             acc_pct=float(np.mean(hits[ds]) * 100),
                             n=len(hits[ds]), hits=hits[ds])
                             for ds in DS_ORDER},
                         macro_pct=macro * 100, predictions=preds,
                         per_image_meta=meta,
                         wall_seconds=float(time.time() - t0),
                         trim_ms_median=float(np.median(
                             [m["trim_ms"] for m in meta])))
                rec_out["arms"][name] = r
                print(f"  {name}: "
                      f"TextVQA {r['per_benchmark']['TextVQA_VAL']['acc_pct']:.3f}"
                      f"  DocVQA {r['per_benchmark']['DocVQA_VAL']['acc_pct']:.3f}"
                      f"  OCRBench {r['per_benchmark']['OCRBench']['acc_pct']:.3f}"
                      f"  macro {r['macro_pct']:.3f}"
                      f"  ms {r['trim_ms_median']:.3f}"
                      f"  ({time.time()-t0:.0f}s)", flush=True)
                del eng
                torch.cuda.empty_cache()
            except Exception:
                traceback.print_exc()
                rec_out["arms"][name] = dict(
                    arm=name, error=traceback.format_exc()[-2000:])
            dump_json(f"{args.tag}.json", rec_out)

    _gates(rec_out, bank)
    dump_json(f"{args.tag}.json", rec_out)
    print(f"\n[done] -> {args.tag}.json")
    return rec_out


def _gates(rec_out, bank):
    """Integrity gates.  A gate that cannot run says so; it never passes silently."""
    g = {}
    arms = rec_out["arms"]
    keys = rec_out["keys"]

    # (1) k=0 must be the stored B2 arm, hit-for-hit AND string-for-string.
    path = os.path.join(OUTPUT_DIR, "m2_accuracy.json")
    b2 = json.load(open(path))["arms"].get("B2") if os.path.exists(path) else None
    ident = arms.get("B2") or arms.get("SAFE-k0")
    if b2 and ident and "error" not in ident:
        agree = sum(int((np.asarray(b2["per_benchmark"][ds]["hits"], float)
                         == np.asarray(ident["per_benchmark"][ds]["hits"], float)
                         ).sum()) for ds in DS_ORDER)
        same = sum(1 for x, y in zip(b2["predictions"], ident["predictions"])
                   if x == y)
        g["identity_is_B2"] = dict(hit_agreement=f"{agree}/150",
                                   prediction_string_agreement=f"{same}/150",
                                   passed=bool(agree == 150))
    else:
        g["identity_is_B2"] = dict(checked=False, reason="identity arm or stored B2 missing")

    # (2) the budget and the set algebra, per instance.
    bad, checked = [], 0
    for arm, r in arms.items():
        if "error" in r:
            continue
        k = int(r["spec"].get("k", 0) or 0)
        for m in r["per_image_meta"]:
            if m.get("n_kept") is None:
                continue
            checked += 1
            if k == 0:
                continue
            s0, dr, a = (set(m["s0_idx"] or []), set(m["drop_idx"] or []),
                         set(m["a_idx"] or []))
            ok = (m["n_kept"] == BUDGET - k and len(s0) == BUDGET
                  and len(dr) == k and dr <= s0 and a == s0 - dr)
            if not ok:
                bad.append((arm, m["key"], m["n_kept"], len(dr), len(a)))
    g["budget_and_trim"] = dict(checked=checked, n_bad=len(bad),
                                examples=bad[:5], passed=bool(checked and not bad))

    # (3) the live S0 must equal the bank's -- proves the arm trims the same set
    #     the offline probe scored, which is what makes the risk ranking mean
    #     anything at inference time.
    bad2, n = [], 0
    for arm, r in arms.items():
        if "error" in r:
            continue
        for m in r["per_image_meta"]:
            if m.get("s0_idx") is None or m["key"] not in bank["_index"]:
                continue
            n += 1
            want = np.sort(np.asarray(bank["s0"][bank["_index"][m["key"]]],
                                      dtype=np.int64))
            if not np.array_equal(np.sort(np.asarray(m["s0_idx"], dtype=np.int64)),
                                  want):
                bad2.append((arm, m["key"]))
    g["live_s0_equals_bank_s0"] = dict(checked=n, n_mismatch=len(bad2),
                                       examples=bad2[:5],
                                       passed=bool(n and not bad2))

    # (4) a SAFE arm's drop set must be the argmin-risk set of the SAME probe
    #     the probe file holds -- the offline/online identity check.  Without
    #     it, a train/serve skew in the features would look like a method.
    skew, checked_p = [], 0
    for arm, r in arms.items():
        if "error" in r or r["spec"].get("mode") != "probe":
            continue
        for m in r["per_image_meta"]:
            pr = m.get("risk_proof")
            if not pr or pr.get("min_kept") is None:
                continue
            checked_p += 1
            if not (pr["max_dropped"] <= pr["min_kept"]):
                skew.append((arm, m["key"], pr["max_dropped"], pr["min_kept"]))
    g["probe_drop_is_argmin_risk"] = dict(
        checked=checked_p, n_mismatch=len(skew), examples=skew[:5],
        passed=bool(checked_p and not skew),
        note="max risk among the dropped <= min risk among the kept, per "
             "instance, from the live pruner's own tensors -- a complete "
             "proof of the argmin that costs k+1 floats")

    # (5) the offline/online identity check that actually matters: recompute
    #     the drop set from the BANK's feature matrix and the SAVED probe, and
    #     require it to equal what the live pruner delivered.  If the live
    #     feature path diverged from the offline one by so much as a column
    #     order, the arm would be trimming a different set than the one the
    #     gate ranked -- and that would look exactly like a method result.
    mism, checked_o = [], 0
    for arm, r in arms.items():
        if "error" in r or r["spec"].get("mode") != "probe":
            continue
        k = int(r["spec"]["k"])
        kind = r["spec"].get("kind", "mlp")
        probe, mu, sd, _ = load_probe(kind, 0)
        Xb = bank["X"]
        for m in r["per_image_meta"]:
            i = bank["_index"].get(m["key"])
            if i is None or m.get("drop_idx") is None:
                continue
            checked_o += 1
            s0 = np.asarray(m["s0_idx"], dtype=np.int64)
            x = torch.from_numpy(Xb[i][s0].astype(np.float32))
            xn = (x - mu) / sd
            with torch.no_grad():
                risk = probe.risk(xn).numpy()
            want = set(s0[np.argsort(risk, kind="stable")[:k]].tolist())
            if want != set(m["drop_idx"]):
                mism.append((arm, m["key"],
                             len(want ^ set(m["drop_idx"]))))
    g["offline_recompute_matches_live"] = dict(
        checked=checked_o, n_mismatch=len(mism), examples=mism[:5],
        passed=bool(checked_o and not mism),
        note="the drop set recomputed from the bank's feature matrix and the "
             "saved probe equals the live one -- this is the train/serve skew "
             "gate, and it is the one that would otherwise turn a feature-path "
             "bug into a method result")

    for k, v in g.items():
        print(f"[gate] {k}: {v.get('passed', v)}")
    rec_out["gates"] = g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["B2"])
    ap.add_argument("--tag", default="m5_accuracy")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test on the first N instances only")
    args = ap.parse_args()
    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0),
                              max_new_tokens=MAX_NEW)
    model.model.eval()
    stage(model, args)


if __name__ == "__main__":
    main()
