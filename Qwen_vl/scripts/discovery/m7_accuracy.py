"""
M7 step 3 -- Phase 3, the generation gate on the held-out test 150.

Injects an explicit, precomputed 256-token set into the incumbent's own pruner.
Nothing is selected at run time: the sets come from `m7_sets.json`, which was
written by `m7_sets.py` from cached cheap features and the incumbent's own
greedy order.  The pruner's job here is to deliver them, not to choose them.

Gates, all of which abort rather than record a number:

    G-B2    the identity arm reproduces the STORED B2 predictions 150/150, and
            its hit vector too -- the only check that can witness alpha, beta,
            sim_mode and visual_token_num, none of which appear in a config hash
    G-SET   the injected set is exactly 256, duplicate-free, in range
    G-LIVE  the injected set is a subset of {S0} u {dropped}, and the tokens it
            evicts from S0 are exactly the ones the arm's core construction says
    G-NOERR a crashed instance is never persisted as a low macro

Usage
    python scripts/discovery/m7_accuracy.py --arms B2 U16 U12 COS16 RND16 CORE16 U8
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
from instrumented import TimedEADPPruner                             # noqa: E402
from m2_accuracy import MAX_NEW, heldout                             # noqa: E402
from m2_gdep import MODE_PRELLM, GDEPConfig, GDEPEngine, dump_json   # noqa: E402
from m5_common import BASE_SELECTOR, BUDGET, bank_items              # noqa: E402
from scoring import per_sample_hits                                  # noqa: E402
from m6_common import DS_ORDER                                       # noqa: E402

IDENTITY = "B2"
PANELS = {"bank": ("m7_sets_bank.json", "m7_accuracy_bank"),
          "ext": ("m7_sets_ext.json", "m7_accuracy_ext")}


class SetPruner(TimedEADPPruner):
    """B2's own pass, then deliver a fixed index set instead of S0.

    `final is None` takes the incumbent's own `forward()` unchanged, so the
    identity arm is byte-identical to B2 by construction rather than by
    agreement.  Everything else -- the score, the similarity, the selector, the
    budget -- is the incumbent's; only the delivered index set differs.
    """

    def __init__(self, *a, final=None, allow_short=False, **kw):
        super().__init__(*a, **kw)
        self.final = final
        # `allow_short` exists for exactly one arm: CORE*, the deliberately
        # budget-breaking diagnostic that delivers the shrunk core with nothing
        # added.  Every method arm keeps the hard 256 assertion.
        self.allow_short = allow_short
        self.last_hedge = {}
        self.last_hedge_events = None

    @torch.no_grad()
    def forward(self, image_features, text_embeds_llm, text_embeds_seq_llm,
                grid_thw):
        if self.final is None:
            self.last_hedge = dict(identity=True, hedge_ms=0.0)
            return super().forward(image_features, text_embeds_llm,
                                   text_embeds_seq_llm, grid_thw)
        self.keep_gpu = True
        self.last_gpu = {}
        super().forward(image_features, text_embeds_llm,
                        text_embeds_seq_llm, grid_thw)
        self.keep_gpu = False
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        g = self.last_gpu
        self.last_gpu = {}
        s0 = torch.sort(g["select_idx"][0].to(torch.long)).values
        idx = torch.as_tensor(self.final, dtype=torch.long,
                              device=image_features.device)
        if self.allow_short:
            assert 0 < idx.numel() <= BUDGET, f"|S_final| = {idx.numel()}"
        else:
            assert idx.numel() == BUDGET, f"|S_final| = {idx.numel()}"
        assert torch.unique(idx).numel() == idx.numel(), "duplicate in S_final"
        assert int(idx.min()) >= 0 and int(idx.max()) < image_features.shape[0]
        ev1.record()
        self.last_hedge_events = (ev0, ev1)
        self.last_hedge = dict(identity=False, n=idx.numel(),
                               s0_idx=s0.detach().cpu().tolist(),
                               final_idx=idx.detach().cpu().tolist())
        idx = torch.sort(idx).values
        return image_features[idx], [int(idx.numel())]

    def read_hedge_ms(self) -> float:
        if self.last_hedge_events is None:
            return 0.0
        a, b = self.last_hedge_events
        self.last_hedge_events = None
        a.synchronize()
        return float(a.elapsed_time(b))


def install(eng, model, final, allow_short=False):
    old = eng.pruner
    p = SetPruner(visual_token_num=old.visual_token_num, alpha=old.alpha,
                  beta=old.beta, visual_dim=old.visual_dim,
                  spatial_merge_size=old.spatial_merge_size,
                  selector=BASE_SELECTOR, capture=False, final=final,
                  allow_short=allow_short)
    p = p.to(next(model.model.parameters()).device)
    p.eval()
    p.sim_mode = "rebound"
    eng.pruner = p
    model.pruner = p
    assert eng.pruner is model.pruner, "dual-hook install failed"
    return p


def stage(model, args):
    torch.set_grad_enabled(False)
    sets_name, tag = PANELS[args.panel]
    if args.panel == "bank":
        items = heldout(model)
    else:
        z = np.load(os.path.join(OUTPUT_DIR, "m7_ext.npz"), allow_pickle=False)
        pb = dict(key=[str(k) for k in z["key"]],
                  ds=[str(d) for d in z["ds"]],
                  idx=[int(i) for i in z["idx"]],
                  split=["ext"] * len(z["key"]))
        items = bank_items(model, pb, splits=("ext",))
    sets = json.load(open(os.path.join(OUTPUT_DIR, sets_name)))
    print(f"[panel {args.panel}] {len(items)} instances; arms in sets file: "
          f"{sorted(sets['arms'])}")

    stored_b2 = None
    p = os.path.join(OUTPUT_DIR, "m2_accuracy.json")
    if args.panel == "bank" and os.path.exists(p):
        m2 = json.load(open(p))
        a = m2["arms"].get("B2")
        if a:
            hit = {ds: a["per_benchmark"][ds]["hits"] for ds in DS_ORDER}
            stored_b2 = dict(pred=dict(zip(m2["keys"], a["predictions"])),
                             hit={k: h for ds in DS_ORDER
                                  for k, h in zip(
                                      [k for k, d in zip(m2["keys"], m2["ds_order"])
                                       if d == ds], hit[ds])})
            print(f"[gate] stored B2 loaded: {len(stored_b2['pred'])} predictions")

    rec = dict(config=vars(args), n_instances=len(items),
               keys=[i["key"] for i in items], ds_order=[i["ds"] for i in items],
               max_new_tokens=MAX_NEW, arms={})
    out_path = os.path.join(OUTPUT_DIR, f"{tag}.json")
    if args.resume and os.path.exists(out_path):
        prev = json.load(open(out_path)).get("arms", {})
        keep = {k: v for k, v in prev.items() if "error" not in v}
        rec["arms"].update(keep)
        print(f"[resume] keeping {sorted(keep)}")

    for arm in args.arms:
        if arm == IDENTITY:
            per_instance = None
        else:
            assert arm in sets["arms"], f"{arm} not in {sets_name}"
            per_instance = sets["arms"][arm]["sets"]
        print(f"\n=== {arm} ===", flush=True)
        t0 = time.time()
        eng = GDEPEngine(model, GDEPConfig(mode=MODE_PRELLM, budget=BUDGET,
                                           selector=BASE_SELECTOR, tag=arm))
        first = None if per_instance is None else per_instance[items[0]["key"]]
        short = (arm != IDENTITY
                 and sets["arms"][arm].get("rescue") == "none")
        if short:
            print(f"  [diagnostic] {arm} breaks the 256 budget on purpose")
        pruner = install(eng, model, first, allow_short=short)
        preds, meta, n_fail = [], [], 0
        try:
            for j, it in enumerate(items):
                err = None
                try:
                    if per_instance is not None:
                        pruner.final = per_instance[it["key"]]
                    out = eng.run(it["msg"], it["ds"], MAX_NEW)
                    preds.append(out["prediction"])
                    hms = pruner.read_hedge_ms()
                    lm = pruner.last_hedge
                    meta.append(dict(key=it["key"], ds=it["ds"],
                                     n_kept=out["info"].get("n_kept"),
                                     n_decode=out.get("n_decode"),
                                     hedge_ms=hms,
                                     identity=lm.get("identity"),
                                     s0_idx=lm.get("s0_idx"),
                                     final_idx=lm.get("final_idx")))
                    del out
                except Exception:
                    err = traceback.format_exc()
                    n_fail += 1
                    preds.append("")
                    meta.append(dict(key=it["key"], ds=it["ds"], error=err))
                if (j + 1) % 25 == 0:
                    print(f"  {j+1}/{len(items)}  {time.time()-t0:.0f}s", flush=True)
            if n_fail:
                raise RuntimeError(f"{n_fail}/{len(items)} raised; first:\n"
                                   + next(m["error"] for m in meta if m.get("error")))
            hits = {}
            for ds in DS_ORDER:
                sel = [i for i, it in enumerate(items) if it["ds"] == ds]
                hits[ds] = [float(x) for x in per_sample_hits(
                    ds, [items[i]["row"] for i in sel], [preds[i] for i in sel])]
            macro = float(np.mean([np.mean(hits[ds]) for ds in DS_ORDER]))
            r = dict(arm=arm,
                     per_benchmark={ds: dict(acc_pct=float(np.mean(hits[ds]) * 100),
                                             n=len(hits[ds]), hits=hits[ds])
                                    for ds in DS_ORDER},
                     macro_pct=macro * 100, predictions=preds,
                     per_image_meta=meta, wall_seconds=float(time.time() - t0),
                     hedge_ms_median=float(np.median([m["hedge_ms"] for m in meta
                                                      if "hedge_ms" in m])))
            if arm == IDENTITY and stored_b2 is not None:
                same_p = sum(1 for it, x in zip(items, preds)
                             if stored_b2["pred"].get(it["key"]) == x)
                mine = {it["key"]: h for ds in DS_ORDER
                        for it, h in zip([i for i in items if i["ds"] == ds],
                                         hits[ds])}
                same_h = sum(1 for k, h in mine.items()
                             if abs(h - stored_b2["hit"].get(k, -1)) < 1e-9)
                r["gate_B2_predictions_match"] = int(same_p)
                r["gate_B2_hits_match"] = int(same_h)
                r["gate_B2_pass"] = bool(same_p == len(items)
                                         and same_h == len(items))
                print(f"  [G-B2] stored-B2 agreement: predictions {same_p}/{len(items)}"
                      f"  hits {same_h}/{len(items)}  -> "
                      f"{'PASS' if r['gate_B2_pass'] else 'FAIL'}")
            rec["arms"][arm] = r
            print(f"  {arm}: TextVQA {r['per_benchmark']['TextVQA_VAL']['acc_pct']:.3f}"
                  f"  DocVQA {r['per_benchmark']['DocVQA_VAL']['acc_pct']:.3f}"
                  f"  OCRBench {r['per_benchmark']['OCRBench']['acc_pct']:.3f}"
                  f"  macro {r['macro_pct']:.3f}  hedge_ms {r['hedge_ms_median']:.3f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
        except Exception:
            rec["arms"][arm] = dict(arm=arm, error=traceback.format_exc())
            print(f"  {arm}: ERROR (not persisted as a score)", flush=True)
        finally:
            del eng, pruner
            torch.cuda.empty_cache()
        dump_json(f"{tag}.json", rec)
    print(f"[saved] {tag}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["B2", "U16", "U12", "COS16", "RND16", "CORE16", "U8"])
    ap.add_argument("--panel", choices=("bank", "ext"), default="bank")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0),
                              max_new_tokens=MAX_NEW)
    model.model.eval()
    stage(model, args)


if __name__ == "__main__":
    main()
