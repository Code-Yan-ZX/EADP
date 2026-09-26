"""
M5 (Safe Removal) step 0 -- reconnaissance gates.

Before a single deletion is measured, three assumptions have to be checked,
because every downstream number rests on them:

    G-VIS   the m3_bank `vis` column (fp16) round-trips to the vision tower's
            bf16 output BIT-EXACTLY.  If it does, the whole teacher can run off
            the bank and no vision tower pass is needed.  If it does not, the
            teacher must recompute `vis` live for every instance.
    G-S0    the m3_bank `s0` column equals the LIVE pruner's select_idx for the
            same instance.  The bank was built under the same B2 configuration
            the M4 gate verified on the held-out 150; this re-checks it on the
            fit/val rows the teacher will actually use.
    G-COST  the wall cost of one teacher-forced NLL forward, so the teacher grid
            is sized against a measurement instead of a guess.

Writes `m5_recon.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                        # noqa: E402
from common import OUTPUT_DIR, eadp_model_name                       # noqa: E402
from m2_gdep import MODE_PRELLM, GDEPConfig, GDEPEngine, dump_json   # noqa: E402
from m5_common import BANK, BUDGET, bank_items, load_bank            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-vis", type=int, default=6)
    ap.add_argument("--n-s0", type=int, default=12)
    ap.add_argument("--tag", default="m5_recon")
    args = ap.parse_args()

    rep = dict(bank=BANK, n_vis_checked=0, n_s0_checked=0)
    bank = load_bank()
    keys = list(bank["key"])
    print(f"[bank] {len(keys)} instances  "
          f"{dict(zip(*np.unique(bank['split'], return_counts=True)))}")

    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0),
                              max_new_tokens=32)
    model.model.eval()
    torch.set_grad_enabled(False)

    # ---- the live B2 engine, exactly as the accuracy arms build it ---------
    cfg = GDEPConfig(mode=MODE_PRELLM, budget=BUDGET, selector="block8",
                     tag="M5-G")
    eng = GDEPEngine.from_checkpoint(cfg, model=model, max_new_tokens=32)
    # The prellm branch does not publish select_idx in `info`; the pruner's own
    # device capture is the live record of the same set.
    eng.pruner.keep_gpu = True
    model.pruner.keep_gpu = True

    items = bank_items(model, bank, splits=("fit", "val", "test"))
    by_key = {it["key"]: it for it in items}

    # ---- G-VIS / G-S0 ------------------------------------------------------
    rows = []
    # fit/val rows for S0, and a spread of instances for the vision check.
    order = [k for k in keys if bank["split"][keys.index(k)] in ("fit", "val")]
    vis_keys = [keys[i] for i in np.linspace(0, len(keys) - 1,
                                             args.n_vis).astype(int)]
    s0_keys = order[:args.n_s0]
    todo = sorted(set(vis_keys) | set(s0_keys))

    for k in todo:
        it = by_key[k]
        t0 = time.perf_counter()
        prep = eng.prepare(it["msg"], it["ds"])
        prep_ms = (time.perf_counter() - t0) * 1e3
        vis_live = prep["vis"]                     # bf16, the tower's own output
        i = keys.index(k)
        vis_bank = torch.from_numpy(bank["vis"][i]).to(torch.bfloat16).to(vis_live.device)

        rec = dict(key=k, ds=it["ds"], prep_ms=prep_ms,
                   vis_shape=list(vis_live.shape), dtype=str(vis_live.dtype))
        if k in vis_keys:
            exact = bool(torch.equal(vis_live, vis_bank))
            diff = (vis_live.float() - vis_bank.float()).abs()
            mad = float(diff.max())
            # How much of the difference is fp16-subnormal underflow rather
            # than a real disagreement: the bank stored fp16, whose smallest
            # subnormal is 5.96e-8, while bf16 goes far below that.
            small = float(vis_bank.float().abs().clamp_min(1e-30).min())
            rec.update(vis_bit_exact=exact, vis_max_abs_diff=mad,
                       vis_n_diff=int((diff > 0).sum()),
                       vis_n_total=int(diff.numel()),
                       vis_worst_rel=float(
                           (diff / vis_live.float().abs().clamp_min(1e-6)).max()),
                       bank_smallest_abs=small)
            rep["n_vis_checked"] += 1

        if k in s0_keys:
            _, info = eng.prefill(prep)
            live_s0 = np.sort(np.asarray(
                eng.pruner.last_gpu["select_idx"][0].detach().cpu(),
                dtype=np.int64))
            bank_s0 = np.sort(np.asarray(bank["s0"][i], dtype=np.int64))
            same = bool(np.array_equal(live_s0, bank_s0))
            rec.update(s0_identical=same,
                       s0_symdiff=int(len(set(live_s0.tolist())
                                          ^ set(bank_s0.tolist()))),
                       n_kept=int(info["n_kept"]))
            rep["n_s0_checked"] += 1
            eng.pruner.last_gpu = {}
            del info
        rows.append(rec)
        print(f"[recon] {k:22s} vis_exact={rec.get('vis_bit_exact')} "
              f"s0_same={rec.get('s0_identical')} prep={prep_ms:.0f}ms",
              flush=True)
        del prep, vis_bank
        torch.cuda.empty_cache()

    rep["rows"] = rows
    vis_rows = [r for r in rows if "vis_bit_exact" in r]
    s0_rows = [r for r in rows if "s0_identical" in r]
    rep["G_VIS"] = dict(checked=len(vis_rows),
                        n_exact=sum(int(r["vis_bit_exact"]) for r in vis_rows),
                        worst_abs_diff=float(max((r["vis_max_abs_diff"]
                                                  for r in vis_rows), default=-1)),
                        worst_rel=float(max((r["vis_worst_rel"]
                                             for r in vis_rows), default=-1)),
                        n_diff_frac=float(np.mean([r["vis_n_diff"]
                                                   / r["vis_n_total"]
                                                   for r in vis_rows])),
                        passed=bool(vis_rows and all(r["vis_bit_exact"]
                                                     for r in vis_rows)))
    rep["G_S0"] = dict(checked=len(s0_rows),
                       n_identical=sum(int(r["s0_identical"]) for r in s0_rows),
                       passed=bool(s0_rows and all(r["s0_identical"]
                                                   for r in s0_rows)))

    # ---- G-COST: one teacher-forced NLL forward ---------------------------
    from m5_teacher import TeacherHarness
    h = TeacherHarness(model)
    it = by_key[keys[0]]
    import s3a_common as C3
    p = h.prep(it["msg"], it["ds"], bank["vis"][0])
    ans = h.answer_ids(C3.golds_of(it["row"]))
    S0 = [int(t) for t in np.sort(bank["s0"][0])]
    h.nll(p, S0, ans)                                  # warm
    t0 = time.perf_counter()
    n = 10
    for _ in range(n):
        h.nll(p, S0, ans)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3 / n
    rep["G_COST"] = dict(key=it["key"], n_golds=len(ans), ms_per_nll=ms,
                         prompt_tokens=int(h.nll(p, S0, ans)[2]),
                         passed=True)
    print(f"[cost] {ms:.1f} ms per teacher-forced NLL "
          f"({len(ans)} golds, {rep['G_COST']['prompt_tokens']} prompt tokens)")

    dump_json(f"{args.tag}.json", rep)
    print(f"\n[G-VIS] {rep['G_VIS']}")
    print(f"[G-S0 ] {rep['G_S0']}")


if __name__ == "__main__":
    main()
