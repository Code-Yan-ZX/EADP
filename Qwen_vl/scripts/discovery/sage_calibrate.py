"""
SAGE step 3 -- image-disjoint calibration (decisions 8-10).

For every (g, family, width): deploy the LIVE pruner on the 60 val images with
tau = -inf semantics (record the argmax edge, always exchange), generate y(Se*),
and sweep tau OFFLINE over the pre-registered grid plus the deciles of the val
predicted-gain distribution -- legitimate because eq. 5 deploys exactly ONE
candidate edge per instance, so every tau shares the same generations.

TTFT constraint (decision 9): the deployed arm's median wall TTFT must stay
<= 1.10 x the B2 reference measured on the same images, same convention
(sync -> [prepare + prefill] -> sync; the B2 reference skips decode, which the
TTFT window does not include).

Outputs
    sage_valdeploy_g{g}_{family}_w{w}.json   per-instance records
    sage_calib.json                          frozen (g*, family*, width*, tau*)
                                             for the set critic AND the unary
                                             control, with TTFT stats

Usage
    python scripts/discovery/sage_calibrate.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                        # noqa: E402
from common import OUTPUT_DIR, eadp_model_name                       # noqa: E402
from m2_gdep import MODE_PRELLM, GDEPConfig, GDEPEngine              # noqa: E402
from m2_accuracy import MAX_NEW                                      # noqa: E402
from m5_common import BUDGET                                         # noqa: E402
from m6_common import load_bank                                      # noqa: E402
from scoring import per_sample_hits                                  # noqa: E402
import sage_common as S                                              # noqa: E402
from sage_common import dump_json                                    # noqa: E402
from sage_live import load_critic, install_sage, run_one             # noqa: E402


def one_hit(ds, row, prediction):
    return float(per_sample_hits(ds, [row], [prediction])[0])


def sweep_tau(preds, hits_se, hits_s0, ds_of):
    """Official val macro for every tau of the pre-registered grid (the
    deciles of the predicted-gain distribution are added to the grid)."""
    grid = sorted(set(S.TAU_GRID) | set(np.quantile(preds, np.arange(0.1, 1.0, 0.1)).round(6).tolist()))
    out = []
    for tau in grid:
        hits = np.where(np.asarray(preds) > tau, hits_se, hits_s0)
        macro = float(np.mean([np.mean(hits[np.asarray(ds_of) == ds]) * 100
                               for ds in S.DS_ORDER]))
        out.append(dict(tau=float(tau), macro=macro,
                        exchange_rate=float((np.asarray(preds) > tau).mean())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b2-only", action="store_true",
                    help="only (re)measure the B2 TTFT reference")
    args = ap.parse_args()

    bank = load_bank()
    val_rows = [i for i in range(bank["n"]) if bank["split"][i] == "val"]
    labels = S.load_json(S.F_LABELS.format(split="val"))
    h0 = labels["s0_hits"]
    summary = S.load_json(S.F_CRITIC_SUMMARY)

    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0),
                              max_new_tokens=MAX_NEW)
    model.model.eval()
    torch.set_grad_enabled(False)
    eng = GDEPEngine(model, GDEPConfig(mode=MODE_PRELLM, budget=BUDGET,
                                       selector="block8", tag="SAGECAL"))
    dev = next(model.model.parameters()).device

    # items in benchmark order
    dataset_cache, items = {}, []
    for i in val_rows:
        ds, idx = bank["ds"][i], int(bank["idx"][i])
        if ds not in dataset_cache:
            dataset_cache[ds] = common.build_dataset(ds)
            model.set_dump_image(dataset_cache[ds].dump_image)
        dataset = dataset_cache[ds]
        row = dataset.data.iloc[idx]
        items.append(dict(key=bank["key"][i], ds=ds, row=row,
                          msg=common.build_message(model, dataset, ds, row)))
    assert all(it["key"] in h0 for it in items), "val labels incomplete"
    print(f"[val] {len(items)} images")

    # ---- B2 TTFT reference (prepare + prefill only: decode is not in TTFT) --
    ref_path = os.path.join(OUTPUT_DIR, "sage_val_ttft_b2.json")
    if os.path.exists(ref_path):
        b2_ttft = np.array(S.load_json("sage_val_ttft_b2.json")["ttft_ms"])
        print(f"[b2 ref] loaded, median {np.median(b2_ttft):.1f} ms")
    else:
        from m7_accuracy import install
        pruner = install(eng, model, None)
        b2_ttft = []
        for it in items:
            with S.TTFT() as t:
                prep = eng.prepare(it["msg"], it["ds"])
                eng.prefill(prep)
            b2_ttft.append(t.ms)
        dump_json("sage_val_ttft_b2.json", dict(ttft_ms=b2_ttft,
                  median=float(np.median(b2_ttft))))
        print(f"[b2 ref] median {np.median(b2_ttft):.1f} ms")
    b2_med = float(np.median(b2_ttft))

    if args.b2_only:
        return

    bound = dict(b2_median_ms=b2_med, multiplier=S.TTFT_MULT,
                 max_median_ms=b2_med * S.TTFT_MULT)
    print(f"[constraint] median TTFT <= {bound['max_median_ms']:.1f} ms")

    # unary width per g: best val RMSE from the training summary
    unary_w = {g: min(S.WIDTHS, key=lambda w: summary["families"][f"g{g}"][f"unary_w{w}"])
               for g in S.G_GRID}

    configs = []
    for g in S.G_GRID:
        for w in S.WIDTHS:
            configs.append((g, "set", w))
        configs.append((g, "unary", unary_w[g]))

    results = {}
    for g, family, w in configs:
        pt = (S.F_CRITIC.format(g=g, w=w) if family == "set"
              else f"sage_unary_g{g}_w{w}.pt")
        critic, mu, sd, _ = load_critic(pt, dev)
        pruner = install_sage(eng, model, critic, mu, sd, tau=-1e9, g=g,
                              family=family, record=True)
        recs, ttfts, sages = [], [], []
        t0 = time.time()
        for rank, it in enumerate(items):
            out = run_one(eng, it, MAX_NEW)
            if g == S.G_GRID[0] and family == "set" and w == S.WIDTHS[0] and rank == 0:
                # gate: the live S0 must be the bank's stored S0 (the M7 G-B2
                # discipline, applied to the calibration panel)
                i0 = next(i for i in val_rows if bank["key"][i] == it["key"])
                assert pruner.last_sage["s0_idx"] == \
                    sorted(np.asarray(bank["s0"][i0]).tolist()), \
                    "live S0 != bank S0 on the calibration panel"
            hit = one_hit(it["ds"], it["row"], out["prediction"])
            sg = pruner.read_sage_ms()
            assert pruner.last_sage["accepted"], "tau=-1e9 must always accept"
            recs.append(dict(key=it["key"], ds=it["ds"],
                             pred_gain=pruner.last_sage["pred_gain"],
                             hit_se=hit, h0=h0[it["key"]],
                             ttft_ms=out["ttft_ms"], sage_ms=sg))
            ttfts.append(out["ttft_ms"])
            sages.append(sg)
        preds = np.array([r["pred_gain"] for r in recs])
        hits_se = np.array([r["hit_se"] for r in recs])
        hits_s0 = np.array([r["h0"] for r in recs])
        ds_of = [r["ds"] for r in recs]
        taus = sweep_tau(preds, hits_se, hits_s0, ds_of)
        med = float(np.median(ttfts))
        ok = med <= bound["max_median_ms"]
        best = max((t for t in taus), key=lambda t: (t["macro"], -t["exchange_rate"], t["tau"]))
        tag = f"g{g}_{family}_w{w}"
        results[tag] = dict(g=g, family=family, width=w, ttft_median_ms=med,
                            ttft_ok=bool(ok), sage_ms_median=float(np.median(sages)),
                            taus=taus, best_tau=best, n=len(recs))
        dump_json(S.F_VAL_DEPLOY.format(g=g, family=family, w=w),
                  dict(config=results[tag], records=recs))
        print(f"[{tag}] macro@best {best['macro']:.3f} at tau {best['tau']:.3g} "
              f"(rate {best['exchange_rate']:.2f})  TTFT {med:.1f} ms "
              f"{'OK' if ok else 'VIOLATES'}  sage_ms {np.median(sages):.2f}  "
              f"({time.time()-t0:.0f}s)")

    # ---- freeze: best set config under the TTFT constraint ------------------
    ok_set = {k: r for k, r in results.items()
              if r["family"] == "set" and r["ttft_ok"]}
    ok_un = {k: r for k, r in results.items()
             if r["family"] == "unary" and r["ttft_ok"]}
    if not ok_set:
        raise RuntimeError("no (g, width) satisfies the TTFT constraint")
    pick = lambda d: max(d.values(), key=lambda r: (r["best_tau"]["macro"],
                                                    -r["best_tau"]["exchange_rate"],
                                                    r["best_tau"]["tau"]))
    chosen, chosen_u = pick(ok_set), pick(ok_un)
    calib = dict(
        bound=bound,
        chosen=dict(g=chosen["g"], family=chosen["family"], width=chosen["width"],
                    tau=chosen["best_tau"]["tau"],
                    val_macro=chosen["best_tau"]["macro"],
                    exchange_rate=chosen["best_tau"]["exchange_rate"],
                    ttft_median_ms=chosen["ttft_median_ms"]),
        unary=dict(g=chosen_u["g"], width=chosen_u["width"],
                   tau=chosen_u["best_tau"]["tau"],
                   val_macro=chosen_u["best_tau"]["macro"],
                   exchange_rate=chosen_u["best_tau"]["exchange_rate"],
                   ttft_median_ms=chosen_u["ttft_median_ms"]),
        results=results)
    dump_json(S.F_CALIB, calib)
    print(f"[FROZEN] set: g={calib['chosen']['g']} w={calib['chosen']['width']} "
          f"tau={calib['chosen']['tau']:.4g} val_macro={calib['chosen']['val_macro']:.3f} "
          f"rate={calib['chosen']['exchange_rate']:.2f}")


if __name__ == "__main__":
    main()
