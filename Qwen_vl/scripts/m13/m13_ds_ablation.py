"""M13 PART A step 2: DeepStack branch ablation on the DEV OCR panel.

The main ViT always runs COMPLETE (all 4096 patches, all 27 blocks); only a
branch's DS merger compute + its LLM-side injection are skipped (genuinely,
see m13_common.visual_forward_ds / make_ds_streams).  Sequence length is
unchanged (K=1024) for every setting, so all differences are attributable to
the removed DeepStack contribution.

Gates:
  * all-on custom forward bitwise-equal to the stock visual pass;
  * native setting reproduces the E0 b0 predictions per sample (greedy
    decode is deterministic and the compute path is identical);
  * n_vis_kept == 1024 in every record.

Usage:
  python m13_ds_ablation.py --ds TextVQA_VAL,DocVQA_VAL,OCRBench --mode both
  python m13_ds_ablation.py --mode score
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

DISC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "discovery")
M13_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DISC_DIR)
sys.path.insert(0, M13_DIR)
import common  # noqa: E402
import m13_common as mc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=",".join(mc.OCR_PANEL))
    ap.add_argument("--mode", default="both", choices=["gen", "score", "both"])
    ap.add_argument("--settings", default=None,
                    help="comma list; default all DS_SETTINGS names")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ds_list = args.ds.split(",")
    names = args.settings.split(",") if args.settings else \
        [s[0] for s in mc.DS_SETTINGS]
    flags = {s[0]: s[1:] for s in mc.DS_SETTINGS}

    plan = json.load(open(os.path.join(common.QWEN_ROOT, "outputs", "e0",
                                       "e0_plan.json")))
    model = common.load_model(common.BASELINE_MODEL,
                              max_new_tokens=args.max_new_tokens)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)
    mc.install_ds_skip_patch(eng)
    visual = eng.inner.visual

    if args.mode in ("gen", "both"):
        for ds in ds_list:
            rows = plan["datasets"][ds]["dev_rows"]
            if args.limit:
                rows = rows[:args.limit]
            dataset = common.build_dataset(ds)
            model.set_dump_image(dataset.dump_image)
            shards = {n: mc.load_shard(mc.shard_path(n, 1024, ds)) for n in names}
            t0 = time.time()
            for n, idx in enumerate(rows):
                if all(str(idx) in sh["records"] for sh in shards.values()):
                    continue
                row = dataset.data.iloc[idx]
                msg = common.build_message(model, dataset, ds, row)
                wall0 = time.perf_counter()
                prep = eng.prepare(msg, ds)
                assert prep["n_vis"] == 1024, f"unexpected n_vis {prep['n_vis']}"
                for name in names:
                    if str(idx) in shards[name]["records"]:
                        continue
                    ds_on = flags[name]
                    t_v = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    t_v[0].record()
                    V, DSb = mc.visual_forward_ds(visual, prep["pv"],
                                                  prep["gthw"], ds_on=ds_on)
                    t_v[1].record()
                    torch.cuda.synchronize()
                    vision_ms = t_v[0].elapsed_time(t_v[1])
                    DS_sel = mc.make_ds_streams(eng, DSb, ds_on,
                                                int(prep["n_vis"]),
                                                prep["pv"].device)
                    t_p = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    t_p[0].record()
                    st = eng.prefill(prep, V, None, None, deepstack=True,
                                     pos="mrope3d", DS_sel=DS_sel)
                    t_p[1].record()
                    torch.cuda.synchronize()
                    llm_prefill_ms = t_p[0].elapsed_time(t_p[1])
                    ttft_ms = (time.perf_counter() - wall0) * 1e3
                    gen_ids, text = eng.decode(st, args.max_new_tokens)
                    meta = eng.invariants(prep, st, V, DSb, n_decode=len(gen_ids))
                    rec = dict(prediction=text,
                               truncated=len(gen_ids) >= args.max_new_tokens,
                               n_vis_kept=meta["n_vis_kept"],
                               layer_calls_ok=meta["layer_calls_ok"],
                               vision_ms=vision_ms,
                               llm_prefill_ms=llm_prefill_ms,
                               ttft_ms=ttft_ms)
                    shards[name]["records"][str(idx)] = rec
                    mc.save_shard(mc.shard_path(name, 1024, ds), shards[name])
                if n % 20 == 0:
                    el = time.time() - t0
                    print(f"[{ds}] {n}/{len(rows)} ({el/max(1, n+1):.2f}s/q)",
                          flush=True)
            for name in names:
                mc.save_shard(mc.shard_path(name, 1024, ds), shards[name])
                print(f"[done] {name} {ds}: {len(shards[name]['records'])} records",
                      flush=True)

    if args.mode in ("score", "both"):
        for name in names:
            for ds in ds_list:
                mc.score_shard(name, 1024, ds)

    # native vs e0-b0 identity check (per-sample prediction equality)
    if args.mode in ("score", "both") and "native" in names:
        for ds in ds_list:
            b0_path = os.path.join(common.QWEN_ROOT, "outputs", "e0", "acc",
                                   "b0", "K1024", f"{ds}.json")
            if not os.path.exists(b0_path):
                continue
            b0 = json.load(open(b0_path))["records"]
            mine = mc.load_shard(mc.shard_path("native", 1024, ds))["records"]
            shared = sorted(set(b0) & set(mine))
            mism = [i for i in shared
                    if b0[i]["prediction"] != mine[i]["prediction"]]
            print(f"[identity] native vs e0-b0 {ds}: {len(shared)} shared, "
                  f"{len(mism)} mismatched", flush=True)


if __name__ == "__main__":
    main()
