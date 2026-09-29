"""M12 gate 1: 100%-keep sparse-path correctness audit.

Checks, on DEV samples:
  C1  sparse_visual_forward @ keep-all reproduces the stock ``visual``
      forward bit-exactly: merged features V and all three DeepStack streams.
  C2  end-to-end: prefill with (keep_idx=all, V_sel, DS_sel from the sparse
      path) reproduces the stock prefill logits (all 32 forced-decode steps).
  C3  keep-all generation text == stock-path generation text.
  C4  at 50%: exact budget, no dups/OOB, ascending, DS lengths == K,
      layer_calls consistent, gate latency recorded.

Usage: python m12_correctness.py [--n 4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

DISC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "discovery")
sys.path.insert(0, DISC_DIR)
import common  # noqa: E402

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "m12")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "m12_correctness.json"))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    plan = json.load(open(os.path.join(common.QWEN_ROOT, "outputs", "e0",
                                       "e0_plan.json")))
    model = common.load_model(common.BASELINE_MODEL, max_new_tokens=64)
    from model.native_qwen3 import NativeEngine
    from model import retinagate as rg
    eng = NativeEngine(model)

    # stratified samples across the OCR panel
    samples = []
    for ds in ("TextVQA_VAL", "DocVQA_VAL", "OCRBench", "ChartQA_TEST"):
        rows = plan["datasets"][ds]["dev_rows"]
        step = max(1, len(rows) // args.n)
        for idx in rows[::step][:args.n]:
            dataset = common.build_dataset(ds)
            model.set_dump_image(dataset.dump_image)
            row = dataset.data.iloc[idx]
            samples.append((ds, common.build_message(model, dataset, ds, row)))

    res = dict(samples=[], c1_max_diff=[], c2_max_diff=[], c4=[])
    all_c1, all_c2, all_c3 = 0.0, 0.0, True
    for si, (ds, msg) in enumerate(samples):
        prep = eng.prepare(msg, ds)

        # stock path
        V, DS = eng.encode(prep)
        st_stock = eng.prefill(prep, V, DS, None, deepstack=True, pos="mrope3d",
                               full_lm_head=True)
        _, text_stock = eng.decode(st_stock, 32, ignore_eos=True)

        # sparse path at keep-all (n_vis IS the merged/group count: 1024)
        n_groups = prep["n_vis"]
        keep_all = torch.arange(n_groups, device=prep["pv"].device)
        V_sel, DS_sel = rg.sparse_visual_forward(
            eng.inner.visual, prep["pv"], prep["gthw"], keep_all)
        dV = (V_sel.float() - V.float()).abs().max().item()
        dDS = max((a.float() - b.float()).abs().max().item()
                  for a, b in zip(DS_sel, DS))
        all_c1 = max(all_c1, dV, dDS)

        st_sp = eng.prefill(prep, None, None, keep_all, deepstack=True,
                            pos="mrope3d", V_sel=V_sel, DS_sel=DS_sel,
                            full_lm_head=True)
        dl = (st_sp.logits.float() - st_stock.logits.float()).abs().max().item()
        # forced-decode logits
        st2 = eng.prefill(prep, None, None, keep_all, deepstack=True,
                          pos="mrope3d", V_sel=V_sel, DS_sel=DS_sel,
                          full_lm_head=False)
        st2s = eng.prefill(prep, V, DS, None, deepstack=True, pos="mrope3d",
                           full_lm_head=False)
        g_sp, t_sp = eng.decode(st2, 32, ignore_eos=True)
        g_st, t_st = eng.decode(st2s, 32, ignore_eos=True)
        l_ok = t_sp == t_st and g_sp == g_st
        all_c3 &= l_ok
        for a, b in zip(st2.decode_logits, st2s.decode_logits):
            dl = max(dl, (a.float() - b.float()).abs().max().item())
        all_c2 = max(all_c2, dl)

        # C4: 50% invariants
        K = n_groups // 2
        kg, _ = rg.select_groups("retinagate", K, prep, eng)
        tm = {}
        out = rg.rg_generate(eng, msg, ds, k=K, mode="retinagate",
                             max_new_tokens=8, ignore_eos=True, timings=tm)
        m = out["meta"]
        c4 = dict(K=K, exact=bool(m["n_groups_kept"] == K),
                  no_dup=bool(m["no_dup"]), in_range=bool(m["in_range"]),
                  ascending=bool(m["ascending"]),
                  ds_ok=bool(m.get("ds_lengths") == [K, K, K]),
                  # ignore_eos runs break after the last append, before the
                  # final layer call, so the expected count drops by one call
                  layer_ok=bool(m["layer_calls"]
                                == 36 * (1 + m["decode_steps"])
                                or m["layer_calls"]
                                == 36 * m["decode_steps"]),
                  gate_ms=tm.get("gate_ms"),
                  vision_ms=tm.get("vision_ms"),
                  vit_blocks_ms=tm.get("vit_blocks_ms"),
                  n_vis_kept=m["n_vis_kept"])
        res["samples"].append(dict(ds=ds, si=si, dV=dV, dDS=dDS, dlogits=dl,
                                   text_match=bool(l_ok), c4=c4))
        print(f"[{ds} #{si}] dV={dV:.3e} dDS={dDS:.3e} dlog={dl:.3e} "
              f"text={'OK' if l_ok else 'MISMATCH'} C4={c4}", flush=True)

    verdict = dict(
        c1_bit_exact=bool(all_c1 == 0.0), c1_max_diff=all_c1,
        c2_max_logit_diff=all_c2,
        c2_ok=bool(all_c2 <= 6.5e-2),  # N1/A2 ulp-level bar
        c3_text_all_match=bool(all_c3),
        passed=bool(all_c1 == 0.0 and all_c2 <= 6.5e-2 and all_c3),
    )
    res["verdict"] = verdict
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("[verdict]", verdict, flush=True)


if __name__ == "__main__":
    main()
