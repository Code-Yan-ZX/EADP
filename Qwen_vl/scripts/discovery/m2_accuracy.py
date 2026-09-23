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

`--arms` and `--seeds` make the grid explicit; the defaults are the full
pre-registered grid.  `C1-SHUF` needs one extra scoring-only pass over the same
150 instances, which is why it is implemented as a two-pass arm.

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
}
SEEDED = ("C0", "C1", "C2", "C3", "C1-R", "C1-SHUF")


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
                        p, info = "", {}
                    preds.append(p)
                    meta.append(dict(key=it["key"], ds=it["ds"],
                                     n_kept=info.get("n_kept"),
                                     context_len=info.get("context_len"),
                                     kv_seq_len=info.get("kv_seq_len"),
                                     pos_ids_contiguous=info.get(
                                         "pos_ids_contiguous")))
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

    _gate_F(rec)
    _summarise(rec)
    dump_json(f"{args.tag}.json", rec)
    print(f"\n[done] -> {args.tag}.json")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARM_SPEC))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--tag", default="m2_accuracy")
    args = ap.parse_args()
    model = common.load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=MAX_NEW)
    model.model.eval()
    stage(model, args)


# ---------------------------------------------------------------------------
def _published_baselines():
    """The published pre-LLM held-out numbers, read from the Stage-1 records
    rather than retyped: `facility` from diag_selectors_b256.json (§5.2) and
    `block8` from diag_selectors_b256_rebound.json (§5.5). Both were produced on
    the identical frozen bank with the identical model name and budget."""
    files = {"B1": ("diag_selectors_b256.json", "facility"),
             "B2": ("diag_selectors_b256_rebound.json", "block8")}
    out = {}
    for arm, (fname, sel) in files.items():
        path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(path):
            continue
        runs = json.load(open(path))["runs"]
        ref = {}
        for ds in DS_ORDER:
            for k, v in runs.items():
                # key shapes differ between the two Stage-1 records
                # ("b256|sel|ds" vs "b256|sel|sim|ds"), so match on the
                # delimiter-wrapped selector and the dataset suffix.
                if k.endswith(ds) and f"|{sel}|" in k:
                    ref[ds] = float(v["acc_pct"])
        if len(ref) == len(DS_ORDER):
            out[arm] = ref
    return out


def _gate_F(rec):
    """Harness identity: the engine's pre-LLM arms must reproduce the published
    held-out numbers (Stage-1 §5.2 / §5.5) to +/- 0.5 points."""
    published = _published_baselines()
    out = {}
    for arm, ref in published.items():
        r = rec["arms"].get(arm)
        if not r or "error" in r:
            out[arm] = dict(checked=False, reason="arm did not run")
            continue
        d = {ds: float(r["per_benchmark"][ds]["acc_pct"] - ref[ds])
             for ds in ref}
        out[arm] = dict(published=ref,
                        measured={ds: r["per_benchmark"][ds]["acc_pct"]
                                  for ds in ref},
                        delta=d, worst_abs_delta=max(abs(v) for v in d.values()),
                        passed=bool(max(abs(v) for v in d.values()) <= 0.5))
        print(f"[G-F] {arm}: worst |delta| vs published = "
              f"{out[arm]['worst_abs_delta']:.3f} -> "
              f"{'PASS' if out[arm]['passed'] else 'FAIL'}")
    if out:
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
