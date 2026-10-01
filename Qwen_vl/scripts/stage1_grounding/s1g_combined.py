"""Stage-1 grounding discovery — Stage-1 x Stage-2 combination runner.

Runs the FOUR-WAY ablation the protocol reserves for after a Stage-1
verdict (user directive §8):

  A  BASE            : official EADP (b1) + identity gather   (= amp BASE)
  B  BASE+AC         : official b1 keep + Anchor Completion U025 (= amp U025)
  C  <S1W>           : Stage-1 winner selector + identity gather (already run)
  D  <S1W>+AC        : Stage-1 winner keep + Anchor Completion U025  <- new

Selection is ONLINE (the Stage-1 keep set differs from the bank); the merge
then reuses amp_common.compute_assignment / merge_stream verbatim on that
keep set (frozen Stage-2 config: kind=uniform, lam=0.25, scope=both).

Usage:
  python s1g_combined.py --split confirm --arm D_G1 --lam 1.0 --mode both
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

import s1g_common as SC


@torch.no_grad()
def run_one_combined(eng, msg, ds, arm: str, keep: torch.Tensor,
                     max_new_tokens: int = 2048, timings: dict | None = None):
    """Selection (online, Stage-1 winner) + frozen Anchor Completion."""
    import amp_common as AC
    timings = timings if timings is not None else {}
    wall0 = time.perf_counter()
    prep = eng.prepare(msg, ds)
    timings["image_preprocess_ms"] = (time.perf_counter() - wall0) * 1e3
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    V, DS = eng.encode(prep)
    ev[1].record()
    torch.cuda.synchronize()
    timings["vision_ms"] = ev[0].elapsed_time(ev[1])

    n_vis = prep["n_vis"]
    K_ = SC.K
    if n_vis > K_:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dropped_idx, gid, sim = AC.compute_assignment(V, keep)
        y = AC.merge_stream(V, keep, dropped_idx, gid, "uniform", 0.25)
        ys = [AC.merge_stream(ds_, keep, dropped_idx, gid, "uniform", 0.25)
              for ds_ in DS]
        torch.cuda.synchronize()
        timings["merge_ms"] = (time.perf_counter() - t0) * 1e3
        V_sel, DS_sel = y, ys
    else:
        V_sel = DS_sel = None

    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    st = eng.prefill(prep, V, DS, keep, V_sel=V_sel, DS_sel=DS_sel)
    ev[1].record()
    torch.cuda.synchronize()
    timings["llm_prefill_ms"] = ev[0].elapsed_time(ev[1])
    timings["ttft_ms"] = (time.perf_counter() - wall0) * 1e3
    wall1 = time.perf_counter()
    gen_ids, text = eng.decode(st, max_new_tokens)
    timings["decode_wall_ms"] = (time.perf_counter() - wall1) * 1e3
    torch.cuda.synchronize()
    meta = eng.invariants(prep, st, V, DS, n_decode=len(gen_ids))
    return dict(text=text, gen_ids=gen_ids, keep_idx=st.keep_idx, meta=meta,
                timings=timings, state=st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "confirm"])
    ap.add_argument("--arm", required=True,
                    help="Stage-1 winner arm name; output arm = <ARM>+AC")
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--mode", default="both", choices=["gen", "score", "both"])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    out_arm = f"{args.arm}+AC"
    manifest = SC.load_manifest()

    if args.mode in ("gen", "both"):
        SC.ensure_selectors(args.lam)
        model = SC.common.load_model(SC.common.BASELINE_MODEL,
                                     max_new_tokens=args.max_new_tokens)
        from model.native_qwen3 import NativeEngine
        eng = NativeEngine(model)
        from model.e0_selectors import run_selector

        for ds in SC.DS_LIST:
            items = manifest["datasets"][ds][args.split]
            dataset = SC.common.build_dataset(ds)
            eng.vlm.set_dump_image(dataset.dump_image)
            path = SC.shard_path(args.split, out_arm, ds, args.tag)
            shard = SC.load_shard(path)
            for n, it in enumerate(items):
                key = str(it["idx"])
                if key in shard["records"]:
                    continue
                row = dataset.data.iloc[it["idx"]]
                msg = SC.common.build_message(eng.vlm, dataset, ds, row)
                prep = eng.prepare(msg, ds)
                V, DS = eng.encode(prep)
                tm, ts = eng.instruction_embeds(msg, ds)
                keep = None
                if prep["n_vis"] > SC.K:
                    ctx = dict(prep=prep, V=V, DS=DS, K=SC.K, engine=eng,
                               text_mean=tm, text_seq=ts, attn_list=None,
                               vz=None, seed=None)
                    keep = run_selector(SC.arm_selector_name(args.arm),
                                        SC.K, ctx)
                else:
                    keep = torch.arange(prep["n_vis"], device=V.device)
                timings = {}
                out = run_one_combined(eng, msg, ds, args.arm, keep,
                                       max_new_tokens=args.max_new_tokens,
                                       timings=timings)
                rec = dict(
                    prediction=out["text"],
                    truncated=len(out["gen_ids"]) >= args.max_new_tokens,
                    n_vis_kept=out["meta"]["n_vis_kept"],
                    ttft_ms=timings.get("ttft_ms"),
                    merge_ms=timings.get("merge_ms"),
                    selector_ms=timings.get("selector_ms"),
                    vision_ms=timings.get("vision_ms"),
                    llm_prefill_ms=timings.get("llm_prefill_ms"),
                    layer_calls_ok=out["meta"]["layer_calls_ok"],
                    degeneracy=SC.degeneracy(out["text"]))
                shard["records"][key] = rec
                shard.setdefault("meta", dict(
                    split=args.split, arm=out_arm, ds=ds, K=SC.K,
                    selector=SC.arm_selector_name(args.arm), lam=args.lam,
                    stage2="uniform+0.25+both", base_commit=manifest["base_commit"],
                    max_new_tokens=args.max_new_tokens))
                if len(shard["records"]) % 10 == 0 or \
                        len(shard["records"]) == len(items):
                    SC.save_shard(path, shard)
                if n % 10 == 0:
                    print(f"[{args.split}/{out_arm} {ds}] {n}/{len(items)}",
                          flush=True)
            SC.save_shard(path, shard)

    if args.mode in ("score", "both"):
        import amp_accuracy as AA
        orig = AA.shard_path
        AA.shard_path = lambda split, arm, ds, K=256: SC.shard_path(
            split, out_arm, ds, args.tag)

        class _Args:
            pass
        try:
            a = _Args()
            a.split, a.arm, a.K = args.split, out_arm, SC.K
            a.selector = SC.arm_selector_name(args.arm)
            AA.run_score(a, None)
        finally:
            AA.shard_path = orig


if __name__ == "__main__":
    main()
