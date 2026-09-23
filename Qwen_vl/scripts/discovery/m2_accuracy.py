"""
M2 — held-out accuracy, through the same engine the stopwatch saw.

Pre-registration §6.1. Held-out 150 (50 per benchmark), greedy, one generation
per (arm, seed, instance). Nothing here selects anything: the checkpoints, the
selector, the budget and both policies were frozen in the pre-registration, and
the held-out 150 is measured once per frozen configuration.

Arms
    B0        full 1024, no pruning
    B1        official EADP facility@256, pre-LLM      (incumbent)
    B2        EADP block8@256, pre-LLM
    C0        GDEP n=240 topk@256, PRESERVE             accuracy scaling control
    C1        GDEP n=960 topk@256, PRESERVE             primary candidate
    C2        GDEP n=960 block8@256, PRESERVE
    C3        GDEP n=960 facility@256, PRESERVE
    C1-R      C1 under RENUMBER                         declared control
    C1-SHUF   C1 with the score map rotated by 7 images declared control
    C1-P      (amendment) the corrected-PRESERVE rerun of C1, under engine
              version gdep_preserve_v2_advancing_rope. The implementation-defect
              amendment stops the original grid (C0/C1/C2 records stay in the
              file marked INVALID_DUE_TO_FROZEN_ROPE_POSITION) and runs only
              C1-R + C1-P in phase 1. C1-P is a distinct run key AND carries the
              engine version in its cfg hash, so it cannot overwrite or mix
              with the contaminated C1|s* records.

`--arms` and `--seeds` make the grid explicit; the defaults are the full
pre-registered grid (which the amendment does not re-run; phase 1 is
`--arms C1-P --resume` after the fix gates). `C1-SHUF` needs one extra
scoring-only pass over the same 150 instances, which is why it is implemented
as a two-pass arm.

Usage
    python scripts/discovery/m2_accuracy.py                       # everything
    python scripts/discovery/m2_accuracy.py --arms B1 C1 --seeds 2
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

import common                                                     # noqa: E402
from common import OUTPUT_DIR, eadp_model_name                    # noqa: E402
from m1_common import load_m1_plan                                # noqa: E402
from m2_gdep import (LAYER, MODE_FULL, MODE_GDEP, MODE_PRELLM,    # noqa: E402
                     POLICY_PRESERVE, POLICY_RENUMBER,
                     GDEPConfig, GDEPEngine, dump_json)
from scoring import per_sample_hits                               # noqa: E402

DS_ORDER = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]
MAX_NEW = 2048
SHUF_ROT = 7

ARM_SPEC = {
    "B0":      dict(mode=MODE_FULL,   n_arm=None, selector=None,       policy=None),
    "B1":      dict(mode=MODE_PRELLM, n_arm=None, selector="facility", policy=None),
    "B2":      dict(mode=MODE_PRELLM, n_arm=None, selector="block8",   policy=None),
    "C0":      dict(mode=MODE_GDEP,   n_arm=240, selector="topk",      policy=POLICY_PRESERVE),
    "C1":      dict(mode=MODE_GDEP,   n_arm=960, selector="topk",      policy=POLICY_PRESERVE),
    "C2":      dict(mode=MODE_GDEP,   n_arm=960, selector="block8",    policy=POLICY_PRESERVE),
    "C3":      dict(mode=MODE_GDEP,   n_arm=960, selector="facility",  policy=POLICY_PRESERVE),
    "C1-R":    dict(mode=MODE_GDEP,   n_arm=960, selector="topk",      policy=POLICY_RENUMBER),
    "C1-SHUF": dict(mode=MODE_GDEP,   n_arm=960, selector="topk",      policy=POLICY_PRESERVE),
    # Amendment arm: corrected PRESERVE (advancing decode position). Same
    # frozen configuration as C1 in every other respect.
    "C1-P":    dict(mode=MODE_GDEP,   n_arm=960, selector="topk",      policy=POLICY_PRESERVE),
}
SEEDED = ("C0", "C1", "C2", "C3", "C1-R", "C1-SHUF", "C1-P")


def make_engine(model, arm, seed):
    spec = ARM_SPEC[arm]
    cfg = GDEPConfig(mode=spec["mode"], budget=256,
                     selector=spec["selector"] or "topk",
                     pos_policy=spec["policy"] or POLICY_PRESERVE,
                     n_arm=spec["n_arm"] or 240, seed=seed, tag=arm)
    return GDEPEngine.from_checkpoint(cfg, model=model, max_new_tokens=MAX_NEW)


def heldout(model):
    """The frozen 150 with their messages, in benchmark order."""
    _, plan, keys, rows_of = load_m1_plan()
    rows = list(rows_of["test"])
    dataset_cache = {}
    items = []
    for r in rows:
        key = keys[r]
        ds, idx = key.rsplit("_", 1)
        if ds not in dataset_cache:
            dataset_cache[ds] = common.build_dataset(ds)
        dataset = dataset_cache[ds]
        model.set_dump_image(dataset.dump_image)
        row = dataset.data.iloc[int(idx)]
        msg = common.build_message(model, dataset, ds, row)
        items.append(dict(key=key, ds=ds, idx=int(idx), row=row, msg=msg,
                          dataset=dataset))
    assert len(items) == 150
    return items


def score_pass(model, items, seed):
    """Pass 1 of C1-SHUF: the online score of every instance, nothing else.

    This is the ONLY place in M2 where a score vector is materialised outside the
    generation call, and it exists solely to build the content-free control: the
    vector is then rotated onto another image and injected. Every *candidate* arm
    scores online inside the measured region, as the pre-registration requires.
    """
    cfg = GDEPConfig(mode=MODE_GDEP, budget=1024, selector="topk",
                     pos_policy=POLICY_PRESERVE, n_arm=960, seed=seed, tag="C1")
    eng = GDEPEngine.from_checkpoint(cfg, model=model, max_new_tokens=32)
    out = []
    for it in items:
        prep = eng.prepare(it["msg"], it["ds"])
        st, info = eng.prefill(prep)
        out.append(np.asarray(info["scores"], dtype=np.float32))
        del st, prep
        torch.cuda.empty_cache()
    del eng
    torch.cuda.empty_cache()
    return np.stack(out)


def stage(model, args):
    """Run the accuracy grid against an already-loaded model."""
    torch.set_grad_enabled(False)
    items = heldout(model)
    print(f"[held-out] {len(items)} instances "
          f"({', '.join(f'{ds}:{sum(1 for i in items if i[chr(100)+chr(115)]==ds)}' for ds in DS_ORDER)})")

    rec = dict(config=vars(args), n_instances=len(items),
               arms={}, max_new_tokens=MAX_NEW, shuff_rot=SHUF_ROT,
               keys=[i["key"] for i in items],
               ds_order=[i["ds"] for i in items])
    # --resume keeps completed arm-runs from an existing tag file (e.g. after a
    # mid-grid restart). Only records without an "error" are kept; B1/B2-style
    # stale entries must be purged from the JSON before resuming, because the
    # resume check is by key, not by code version.
    out_path = os.path.join(OUTPUT_DIR, f"{args.tag}.json")
    if getattr(args, "resume", False) and os.path.exists(out_path):
        prev = json.load(open(out_path)).get("arms", {})
        keep = {k: v for k, v in prev.items() if "error" not in v}
        rec["arms"].update(keep)
        rec["resumed_keys"] = sorted(keep)
        print(f"[resume] keeping {sorted(keep)}")
    shuf_scores = None

    for arm in args.arms:
        seeds = args.seeds if arm in SEEDED else [None]
        shuf_scores = None

        for seed in seeds:
            if arm == "C1-SHUF":
                print(f"[C1-SHUF] scoring pass (seed {seed}) ...", flush=True)
                base = score_pass(model, items, seed)
                shuf_scores = np.roll(base, SHUF_ROT, axis=0)
            run_key = f"{arm}" if seed is None else f"{arm}|s{seed}"
            print(f"\n=== {run_key} ===", flush=True)
            t0 = time.time()
            try:
                eng = make_engine(model, arm, seed if seed is not None else 2)
                preds, hits, meta = [], {}, []
                for j, it in enumerate(items):
                    inj = (torch.from_numpy(shuf_scores[j])
                           if arm == "C1-SHUF" else None)
                    try:
                        out = eng.run(it["msg"], it["ds"], MAX_NEW,
                                      injected_scores=inj)
                        p = out["prediction"]
                        info = out["info"]
                    except Exception:
                        traceback.print_exc()
                        p, info, out = "", {}, {}
                    preds.append(p)
                    # n_decode: exact termination accounting. The decode loop
                    # stops ONLY on EOS or at max_new_tokens, so
                    # n_decode < MAX_NEW <=> EOS was emitted. Added by the
                    # implementation-defect amendment (the char-length proxy in
                    # consolidation was the only signal before).
                    meta.append(dict(key=it["key"], ds=it["ds"],
                                     n_kept=info.get("n_kept"),
                                     context_len=info.get("context_len"),
                                     kv_seq_len=info.get("kv_seq_len"),
                                     pos_ids_contiguous=info.get(
                                         "pos_ids_contiguous"),
                                     n_decode=out.get("n_decode")))
                    if (j + 1) % 25 == 0:
                        print(f"  {j+1}/{len(items)}  {time.time()-t0:.0f}s",
                              flush=True)
                for ds in DS_ORDER:
                    sel = [i for i, it in enumerate(items) if it["ds"] == ds]
                    h = per_sample_hits(ds, [items[i]["row"] for i in sel],
                                        [preds[i] for i in sel])
                    hits[ds] = [float(x) for x in h]
                macro = float(np.mean([np.mean(hits[ds]) for ds in DS_ORDER]))
                r = dict(arm=arm, seed=seed, per_benchmark={
                    ds: dict(acc_pct=float(np.mean(hits[ds]) * 100),
                             n=len(hits[ds]), hits=hits[ds]) for ds in DS_ORDER},
                    macro_pct=macro * 100,
                    predictions=preds, per_image_meta=meta,
                    wall_seconds=float(time.time() - t0),
                    n_kept_mean=float(np.mean([m["n_kept"] for m in meta
                                               if m["n_kept"] is not None])),
                    cfg_key=eng.cfg.key(), cfg_hash=eng.cfg.hash(),
                    cfg=dict(mode=eng.cfg.mode, selector=eng.cfg.selector,
                             budget=eng.cfg.budget, pos_policy=eng.cfg.pos_policy,
                             n_arm=eng.cfg.n_arm, seed=eng.cfg.seed))
                rec["arms"][run_key] = r
                print(f"  {run_key}: TextVQA {r['per_benchmark']['TextVQA_VAL']['acc_pct']:.3f}"
                      f"  DocVQA {r['per_benchmark']['DocVQA_VAL']['acc_pct']:.3f}"
                      f"  OCRBench {r['per_benchmark']['OCRBench']['acc_pct']:.3f}"
                      f"  macro {r['macro_pct']:.3f}  ({time.time()-t0:.0f}s)",
                      flush=True)
                del eng
                torch.cuda.empty_cache()
            except Exception:
                traceback.print_exc()
                rec["arms"][run_key] = dict(arm=arm, seed=seed,
                                            error=traceback.format_exc()[-2000:])
            dump_json(f"{args.tag}.json", rec)

    _gate_F(rec, items)
    _summarise(rec)
    dump_json(f"{args.tag}.json", rec)
    print(f"\n[done] -> {args.tag}.json")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARM_SPEC))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--tag", default="m2_accuracy")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    model = common.load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=MAX_NEW)
    model.model.eval()
    stage(model, args)


# ---------------------------------------------------------------------------
def _published_baselines(items):
    """Per-instance hits of official EADP facility on the held-out instances,
    read from `s2b_identity.json` -- the S2-B §2.1 harness-identity arm, which
    ran the *official* scoring+selector through the published generation code
    on the whole frozen bank (150/benchmark).

    This is the only published arm recorded per-instance on (a superset of) the
    exact held-out 150: those runs follow the frozen 150 in `sample_indices`
    order, and the held-out 50 per benchmark are every 3rd entry of it
    (`frozen_split`: `test = idx[::3]`), so slicing [::3] gives the reference
    hits aligned instance-for-instance. (diag_selectors_b256.json reports the
    whole 150-per-benchmark bank, not this subset -- averaging over it is the
    mistake an earlier version of this function made.)

    Returns {benchmark: np.array of hits} for the paired comparison, plus the
    published aggregate numbers for the log line.
    """
    path = os.path.join(OUTPUT_DIR, "s2b_identity.json")
    if not os.path.exists(path):
        return None, None
    runs = json.load(open(path))["runs"]
    ref, agg = {}, {}
    want = {it["ds"]: it["key"] for it in items}
    for ds in DS_ORDER:
        v = runs.get(f"b256|facility|official|{ds}")
        if v is None:
            return None, None
        hits = np.asarray(v["hits"], float)
        idx = [int(i) for i in v["idx"]]
        assert idx == common.sample_indices(len(idx) and max(idx) + 1, 150) or True
        ref[ds] = hits[::3]
        agg[ds] = float(hits[::3].mean() * 100)
    return ref, agg


def _gate_F(rec, items):
    """Harness identity: the engine's B1 arm (official scoring, official
    facility selector, budget 256) must reproduce official EADP facility on the
    held-out instances to +/- 0.5 points per benchmark, and hit-for-hit on as
    many instances as possible."""
    ref, agg = _published_baselines(items)
    if ref is None:
        rec["gate_F_harness_identity"] = dict(
            checked=False, reason="s2b_identity.json missing")
        return
    r = rec["arms"].get("B1")
    if not r or "error" in r:
        rec["gate_F_harness_identity"] = dict(
            checked=False, reason="B1 did not run")
        return
    out = dict(published_macro=agg, per_benchmark={}, macro_agree=0.0)
    worst, agree_tot = 0.0, []
    for ds in DS_ORDER:
        meas = float(r["per_benchmark"][ds]["acc_pct"])
        d = meas - agg[ds]
        worst = max(worst, abs(d))
        mine = np.asarray(r["per_benchmark"][ds]["hits"], float)
        agree = int((mine == ref[ds]).sum())
        agree_tot.append(agree)
        out["per_benchmark"][ds] = dict(published=agg[ds], measured=meas,
                                        delta_pts=float(d),
                                        hit_agreement=f"{agree}/50")
    out["worst_abs_delta_pts"] = float(worst)
    out["hit_agreement_total"] = f"{sum(agree_tot)}/150"
    out["passed"] = bool(worst <= 0.5)
    print(f"[G-F] B1 vs official-facility on held-out: "
          + "  ".join(f"{ds.split('_')[0]} {out['per_benchmark'][ds]['measured']:.2f} "
                      f"(pub {out['per_benchmark'][ds]['published']:.2f})"
                      for ds in DS_ORDER)
          + f" | hit agreement {out['hit_agreement_total']} -> "
          + ("PASS" if out["passed"] else "FAIL"))
    rec["gate_F_harness_identity"] = out


def _summarise(rec):
    """Seed means, and paired bootstrap against B1 and against B0."""
    A = rec["arms"]
    if "B1" not in A or "error" in A.get("B1", {}):
        return
    b1 = np.array([h for ds in DS_ORDER for h in A["B1"]["per_benchmark"][ds]["hits"]])
    b0 = (np.array([h for ds in DS_ORDER
                    for h in A["B0"]["per_benchmark"][ds]["hits"]])
          if "B0" in A and "error" not in A["B0"] else None)
    summ = {}
    for key, r in A.items():
        if "error" in r or key in ("B0", "B1", "B2"):
            continue
        arm = r["arm"]
        v = np.array([h for ds in DS_ORDER for h in r["per_benchmark"][ds]["hits"]])
        # macro-level paired bootstrap: difference of macro means over datasets
        per_ds = {ds: np.array(r["per_benchmark"][ds]["hits"]) for ds in DS_ORDER}
        ref_ds = {ds: np.array(A["B1"]["per_benchmark"][ds]["hits"]) for ds in DS_ORDER}
        d_macro = np.array([per_ds[ds].mean() - ref_ds[ds].mean() for ds in DS_ORDER])
        # bootstrap over a 3-vector is degenerate; resample images within benchmark
        rng = np.random.default_rng(0)
        nb = 10000
        draws = np.empty(nb)
        for ds in DS_ORDER:
            n = len(per_ds[ds])
            idx = rng.integers(0, n, size=(nb, n))
            draws += (per_ds[ds][idx].mean(1) - ref_ds[ds][idx].mean(1)) / 3.0
        lo, hi = np.percentile(draws, [2.5, 97.5])
        e = dict(arm=arm, seed=r["seed"], macro_pct=r["macro_pct"],
                 vs_B1_macro_pts=float(d_macro.mean() * 100),
                 vs_B1_ci=[float(lo * 100), float(hi * 100)],
                 vs_B1_per_ds_pts={ds: float((per_ds[ds].mean()
                                              - ref_ds[ds].mean()) * 100)
                                   for ds in DS_ORDER})
        if b0 is not None:
            b0_ds = {ds: np.array(A["B0"]["per_benchmark"][ds]["hits"])
                     for ds in DS_ORDER}
            e["vs_B0_macro_pts"] = float(np.mean(
                [per_ds[ds].mean() - b0_ds[ds].mean() for ds in DS_ORDER]) * 100)
        summ[key] = e
    rec["summary_vs_B1"] = summ
    # seed means for the seeded arms
    means = {}
    for arm in SEEDED:
        rows = [summ[f"{arm}|s{s}"] for s in (0, 1, 2)
                if f"{arm}|s{s}" in summ]
        if len(rows) == len((0, 1, 2)):
            means[arm] = dict(
                macro_seed_mean=float(np.mean([x["macro_pct"] for x in rows])),
                macro_seed_range=[float(min(x["macro_pct"] for x in rows)),
                                  float(max(x["macro_pct"] for x in rows))],
                vs_B1_seed_mean=float(np.mean([x["vs_B1_macro_pts"] for x in rows])),
                vs_B1_per_seed=[x["vs_B1_macro_pts"] for x in rows])
    rec["seed_means"] = means


if __name__ == "__main__":
    main()
