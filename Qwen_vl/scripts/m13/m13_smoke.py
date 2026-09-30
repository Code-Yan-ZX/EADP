"""M13 smoke: correctness gates for both PART A and PART B machinery, on one
sample, before any full run.  Exits non-zero on the first failed gate.

Gates:
  1. all-on custom vision forward bitwise == stock visual(pv, gthw);
  2. disabled-branch forward: main V bit-identical to stock V, sequence
     length unchanged; DS streams have the right shapes;
  3. L0-drop depth forward bitwise == M12 rg.sparse_visual_forward
     (V_sel and every DS_sel stream);
  4. merge partner: counts sum == #removed groups, rem rows == 4x that,
     every slot in range; merged sequence runs and V/DS shapes == K;
  5. full prefill+decode invariants for one L8-merge run (layer_calls,
     cache length, n_vis_kept == K);
  6. L0-drop K512 one-sample prediction equals the M12 variance@512 shard
     prediction when present (bit-identical path + deterministic selector).
"""

from __future__ import annotations

import json
import os
import sys

import torch

DISC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "discovery")
M13_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DISC_DIR)
sys.path.insert(0, M13_DIR)
import common  # noqa: E402
import m13_common as mc  # noqa: E402


def gate(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        sys.exit(1)


def main():
    plan = json.load(open(os.path.join(common.QWEN_ROOT, "outputs", "e0",
                                       "e0_plan.json")))
    model = common.load_model(common.BASELINE_MODEL, max_new_tokens=16)
    from model.native_qwen3 import NativeEngine
    from model import retinagate as rg
    eng = NativeEngine(model)
    mc.install_ds_skip_patch(eng)
    visual = eng.inner.visual

    ds = "TextVQA_VAL"
    idx = plan["datasets"][ds]["dev_rows"][0]
    dataset = common.build_dataset(ds)
    model.set_dump_image(dataset.dump_image)
    msg = common.build_message(model, dataset, ds, dataset.data.iloc[idx])
    prep = eng.prepare(msg, ds)
    pv, gthw = prep["pv"], prep["gthw"]
    K = 512

    # gate 1: all-on bitwise vs stock
    V_stock, DS_stock = eng.encode(prep)
    V1, DS1 = mc.visual_forward_ds(visual, pv, gthw, ds_on=(1, 1, 1))
    gate("all-on bitwise V", torch.equal(V_stock, V1))
    gate("all-on bitwise DS", all(torch.equal(a, b)
                                  for a, b in zip(DS_stock, DS1)))

    # gate 2: disabled branch
    V2, DS2 = mc.visual_forward_ds(visual, pv, gthw, ds_on=(1, 0, 1))
    gate("ds-off keeps main V bitwise", torch.equal(V_stock, V2))
    gate("ds-off skips merger", DS2[1] is None and DS2[0] is not None
         and DS2[2] is not None)
    streams = mc.make_ds_streams(eng, DS2, (1, 0, 1), 1024, pv.device)
    gate("flagged zero rows", streams[1].shape == (1024, eng.text.config.hidden_size)
         and getattr(streams[1], mc._DS_SKIP_ATTR, False))

    # gate 3: L0-drop == M12 rg sparse path
    keep_groups, _ = rg.select_groups("variance", K, prep, eng, seed=1)
    keep_groups = keep_groups.to(pv.device)
    V_m12, DS_m12 = rg.sparse_visual_forward(visual, pv, gthw, keep_groups)
    V_l0, DS_l0 = mc.depth_sparse_forward(visual, pv, gthw, L=0,
                                          keep_groups=keep_groups, mode="drop")
    gate("L0-drop bitwise V == M12", torch.equal(V_m12, V_l0))
    gate("L0-drop bitwise DS == M12", all(torch.equal(a, b)
                                          for a, b in zip(DS_m12, DS_l0)))

    # gate 4: merge partner
    partner = mc.build_merge_partner(gthw, visual.spatial_merge_size,
                                     keep_groups, pv.device)
    rem_patch, surv_slot, counts = partner
    n_rem_groups = 1024 - K
    gate("partner counts sum", int(counts.sum()) == n_rem_groups,
         f"counts.sum={int(counts.sum())} n_rem={n_rem_groups}")
    gate("partner rem rows", int(rem_patch.numel()) == 4 * n_rem_groups)
    gate("partner slot range", int(surv_slot.min()) >= 0
         and int(surv_slot.max()) < K)

    # gate 5: L8-merge prefill+decode invariants
    V_l8, DS_l8 = mc.depth_sparse_forward(visual, pv, gthw, L=8,
                                          keep_groups=keep_groups,
                                          mode="merge", partner=partner)
    gate("L8-merge V shape", V_l8.shape == (K, V_stock.shape[-1]), str(V_l8.shape))
    gate("L8-merge DS shapes", all(d.shape == (K, V_stock.shape[-1])
                                   for d in DS_l8))
    st = eng.prefill(prep, None, None, keep_groups, deepstack=True,
                     pos="mrope3d", V_sel=V_l8, DS_sel=DS_l8)
    gen_ids, text = eng.decode(st, 16, ignore_eos=True)
    meta = eng.invariants(prep, st, None, None, n_decode=len(gen_ids))
    # with ignore_eos=True and the cap hit, the LAST token needs no forward:
    # n_forwards = n_decode - 1, so cache_len = lk + n_decode - 1
    n_fwd = 1 + (len(gen_ids) - 1)   # prefill + decode forwards
    ok_inv = (meta["n_vis_kept"] == K
              and meta["layer_calls"] == meta["layer_calls_expected"] - 36
              and meta["cache_len"] == meta["lk"] + len(gen_ids) - 1
              and n_fwd * 36 == meta["layer_calls"])
    gate("L8-merge invariants", ok_inv, json.dumps(
        {k: meta[k] for k in ("n_vis_kept", "lk", "layer_calls",
                              "layer_calls_expected", "cache_len")}))
    print(f"[info] L8-merge sample decode: {text!r}", flush=True)

    # gate 6: L0-drop prediction equals M12 variance@512 shard (if present)
    m12_path = os.path.join(common.QWEN_ROOT, "outputs", "m12", "acc",
                            "variance", "K512", f"{ds}.json")
    if os.path.exists(m12_path):
        m12_rec = json.load(open(m12_path))["records"].get(str(idx))
        if m12_rec is not None:
            model.max_new_tokens = 2048
            model.generate_kwargs["max_new_tokens"] = 2048
            Vsel, DSsel = mc.depth_sparse_forward(visual, pv, gthw, L=0,
                                                  keep_groups=keep_groups,
                                                  mode="drop")
            stf = eng.prefill(prep, None, None, keep_groups, deepstack=True,
                              pos="mrope3d", V_sel=Vsel, DS_sel=DSsel)
            _, text_full = eng.decode(stf, 2048)
            gate("L0-drop prediction == M12 variance@512",
                 text_full == m12_rec["prediction"],
                 f"mine={text_full[:40]!r} m12={m12_rec['prediction'][:40]!r}")
    else:
        print("[skip] gate 6 (no M12 shard)")

    print("[smoke] all gates passed", flush=True)


if __name__ == "__main__":
    main()
