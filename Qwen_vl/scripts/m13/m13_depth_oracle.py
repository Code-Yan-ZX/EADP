"""M13 PART B: ViT compression-depth oracle -- WHEN to compress, not HOW.

Question: does pre-ViT (M12) irregular deletion fail because compression
happens too early?  Keep the selector FIXED and deterministic (M12's best
structured cheap selector: patch-variance), keep the keep-set IDENTICAL
across depths, and only move the compression point L:

    1024 groups -> dense blocks 0..L-1 (full contextualization)
              -> compress to K groups on native 2x2 merge-group atoms
                 (drop | scale-preserving spatial-nearest merge)
              -> blocks L..26 on the compressed sequence

L=0 + drop is the exact M12 pre-encoder sparse path (bit-identical to
rg.sparse_visual_forward; verified as a gate).  LLM side is M12's: prefill
consumes (keep_idx, V_sel, DS_sel), mRoPE3D positions subset from the
full-sequence get_rope_index, decode continues from prefill_max_pos.

Usage:
  python m13_depth_oracle.py --K 512 --ds TextVQA_VAL,DocVQA_VAL,OCRBench
  python m13_depth_oracle.py --mode score
  python m13_depth_oracle.py --smoke 3
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


def arm_id(L: int, mode: str, K: int) -> str:
    return f"L{L}_{mode}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--ds", default=",".join(mc.OCR_PANEL))
    ap.add_argument("--mode", default="both", choices=["gen", "score", "both"])
    ap.add_argument("--depths", default="0,4,8,12,16,20")
    ap.add_argument("--modes", default="drop,merge")
    ap.add_argument("--selector", default="variance",
                    help="M12 pre-encoder selector (fixed across depths)")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args()

    ds_list = args.ds.split(",")
    depths = [int(x) for x in args.depths.split(",")]
    modes = args.modes.split(",")
    K = args.K

    plan = json.load(open(os.path.join(common.QWEN_ROOT, "outputs", "e0",
                                       "e0_plan.json")))
    model = common.load_model(common.BASELINE_MODEL,
                              max_new_tokens=args.max_new_tokens)
    from model.native_qwen3 import NativeEngine
    from model import retinagate as rg
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
            arms = [(L, m) for L in depths for m in modes]
            shards = {arm_id(L, m, K): mc.load_shard(
                mc.shard_path(arm_id(L, m, K), K, ds)) for L, m in arms}
            t0 = time.time()
            for n, idx in enumerate(rows):
                if all(str(idx) in sh["records"] for sh in shards.values()):
                    continue
                row = dataset.data.iloc[idx]
                msg = common.build_message(model, dataset, ds, row)
                wall0 = time.perf_counter()
                prep = eng.prepare(msg, ds)
                assert prep["n_vis"] == 1024, f"unexpected n_vis {prep['n_vis']}"
                # FIXED keep-set: M12 selector on pixels, one per sample
                keep_groups, _ = rg.select_groups(args.selector, K, prep, eng,
                                                  seed=args.seed + 1000 * idx)
                keep_groups = keep_groups.to(prep["pv"].device)
                partner = mc.build_merge_partner(prep["gthw"],
                                                 visual.spatial_merge_size,
                                                 keep_groups, prep["pv"].device)
                for (L, m) in arms:
                    aid = arm_id(L, m, K)
                    if str(idx) in shards[aid]["records"]:
                        continue
                    t_v = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    t_v[0].record()
                    st_vit = {}
                    V_sel, DS_sel = mc.depth_sparse_forward(
                        visual, prep["pv"], prep["gthw"], L=L,
                        keep_groups=keep_groups, mode=m, partner=partner,
                        timings=st_vit)
                    t_v[1].record()
                    torch.cuda.synchronize()
                    vision_ms = t_v[0].elapsed_time(t_v[1])
                    t_p = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    t_p[0].record()
                    st = eng.prefill(prep, None, None, keep_groups,
                                     deepstack=True, pos="mrope3d",
                                     V_sel=V_sel, DS_sel=DS_sel)
                    t_p[1].record()
                    torch.cuda.synchronize()
                    llm_prefill_ms = t_p[0].elapsed_time(t_p[1])
                    ttft_ms = (time.perf_counter() - wall0) * 1e3
                    gen_ids, text = eng.decode(st, args.max_new_tokens)
                    meta = eng.invariants(prep, st, None, None,
                                          n_decode=len(gen_ids))
                    rec = dict(prediction=text,
                               truncated=len(gen_ids) >= args.max_new_tokens,
                               n_vis_kept=meta["n_vis_kept"],
                               layer_calls_ok=meta["layer_calls_ok"],
                               vision_ms=vision_ms,
                               llm_prefill_ms=llm_prefill_ms,
                               ttft_ms=ttft_ms,
                               vit={k: v for k, v in st_vit.items()})
                    shards[aid]["records"][str(idx)] = rec
                    mc.save_shard(mc.shard_path(aid, K, ds), shards[aid])
                if n % 20 == 0:
                    el = time.time() - t0
                    print(f"[{ds}] {n}/{len(rows)} ({el/max(1, n+1):.2f}s/q)",
                          flush=True)
            for (L, m) in arms:
                aid = arm_id(L, m, K)
                mc.save_shard(mc.shard_path(aid, K, ds), shards[aid])
                print(f"[done] {aid} {ds}: {len(shards[aid]['records'])} records",
                      flush=True)

    if args.mode in ("score", "both"):
        for L in depths:
            for m in modes:
                for ds in ds_list:
                    mc.score_shard(arm_id(L, m, K), K, ds)


if __name__ == "__main__":
    main()
