"""Stage-1 x Stage-2 combination (2x2 cell D): NEW Stage-1 (frozen winner
C: v1_knn, k=16, beta=0.5) + Anchor Completion MAIN025 (uniform, lam=0.25,
MAIN feature stream only; DeepStack streams identity gather).

Mirrors AC.run_one (anchor_merge_pilot) except the keep set comes from the
ONLINE fused selector instead of the frozen b1 bank; the merge machinery
(compute_assignment / merge_stream / deterministic group_sum) is reused
verbatim.

Usage:
  python s1n_combo.py --split confirm --arms C_M025 --mode both
  python s1n_combo.py --split confirm --arms C_M025 --mode score
"""

from __future__ import annotations

import argparse
import time

import torch

import s1n_common as SC
import amp_common as AC
from amp_accuracy import run_score as amp_run_score  # noqa: F401

COMBO_ARMS: dict[str, dict] = {
    "C_M025": dict(stage1="C", kind="uniform", lam=0.25, scope="main"),
}


@torch.no_grad()
def run_one_combo(eng, msg, ds, cfg: dict, max_new_tokens: int = 2048,
                  timings: dict | None = None):
    from model.e0_selectors import run_selector
    timings = timings if timings is not None else {}
    wall0 = time.perf_counter()
    prep = eng.prepare(msg, ds)

    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    V, DS = eng.encode(prep)
    ev[1].record()
    torch.cuda.synchronize()
    timings["vision_ms"] = ev[0].elapsed_time(ev[1])

    text_mean, text_seq = eng.instruction_embeds(msg, ds)
    ctx = dict(prep=prep, V=V, DS=DS, K=SC.K, engine=eng,
               text_mean=text_mean, text_seq=text_seq,
               attn_list=None, vz=None, seed=None)
    t_sel0 = time.perf_counter()
    keep = run_selector(SC.arm_selector_name(cfg["stage1"]), SC.K, ctx)
    torch.cuda.synchronize()
    timings["selector_ms"] = (time.perf_counter() - t_sel0) * 1e3

    n_vis = prep["n_vis"]
    if n_vis <= SC.K:
        keep = torch.arange(n_vis, device=V.device)

    V_sel = DS_sel = None
    t0 = time.perf_counter()
    dropped_idx, gid, sim = AC.compute_assignment(V, keep)
    if dropped_idx.numel() > 0:
        if cfg["scope"] in ("both", "main"):
            V_sel = AC.merge_stream(V, keep, dropped_idx, gid, cfg["kind"],
                                    cfg["lam"])
        if cfg["scope"] in ("both", "ds"):
            DS_sel = [AC.merge_stream(ds_, keep, dropped_idx, gid,
                                      cfg["kind"], cfg["lam"]) for ds_ in DS]
    torch.cuda.synchronize()
    timings["merge_ms"] = (time.perf_counter() - t0) * 1e3

    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    st = eng.prefill(prep, V, DS, keep, V_sel=V_sel, DS_sel=DS_sel)
    ev[1].record()
    torch.cuda.synchronize()
    timings["llm_prefill_ms"] = ev[0].elapsed_time(ev[1])
    timings["ttft_ms"] = (time.perf_counter() - wall0) * 1e3

    gen_ids, text = eng.decode(st, max_new_tokens)
    meta = eng.invariants(prep, st, V, DS, n_decode=len(gen_ids))
    return dict(text=text, gen_ids=gen_ids, keep_idx=st.keep_idx, meta=meta,
                timings=timings, state=st)


def run_gen(args, eng, manifest):
    SC.ensure_selectors()
    for ds in SC.DS_LIST:
        items = manifest["datasets"][ds][args.split]
        dataset = SC.common.build_dataset(ds)
        eng.vlm.set_dump_image(dataset.dump_image)
        path = SC.shard_path(args.split, args.arm, ds)
        shard = SC.load_shard(path)
        cfg = COMBO_ARMS[args.arm]
        t0 = time.time()
        for n, it in enumerate(items):
            key = str(it["idx"])
            if key in shard["records"]:
                continue
            row = dataset.data.iloc[it["idx"]]
            msg = SC.common.build_message(eng.vlm, dataset, ds, row)
            timings = {}
            out = run_one_combo(eng, msg, ds, cfg,
                                max_new_tokens=args.max_new_tokens,
                                timings=timings)
            keep = out["keep_idx"]
            keep = keep.detach().cpu().tolist() if keep is not None else None
            shard["records"][key] = dict(
                prediction=out["text"],
                truncated=len(out["gen_ids"]) >= args.max_new_tokens,
                n_vis_kept=out["meta"]["n_vis_kept"],
                keep=keep,
                ttft_ms=timings.get("ttft_ms"),
                selector_ms=timings.get("selector_ms"),
                merge_ms=timings.get("merge_ms"),
                vision_ms=timings.get("vision_ms"),
                llm_prefill_ms=timings.get("llm_prefill_ms"),
                layer_calls_ok=out["meta"]["layer_calls_ok"],
                degeneracy=SC.degeneracy(out["text"]))
            shard.setdefault("meta", dict(
                split=args.split, arm=args.arm, ds=ds, K=SC.K,
                stage1_selector=SC.arm_selector_name(cfg["stage1"]),
                stage1_cfg=SC.ARMS[cfg["stage1"]], merge_cfg=cfg,
                base_commit=manifest["base_commit"],
                max_new_tokens=args.max_new_tokens))
            if len(shard["records"]) % 10 == 0 or \
                    len(shard["records"]) == len(items):
                SC.save_shard(path, shard)
            if n % 10 == 0:
                el = time.time() - t0
                print(f"[{args.split}/{args.arm} {ds}] {n}/{len(items)} "
                      f"({el/max(1, n+1):.2f}s/q)", flush=True)
        SC.save_shard(path, shard)
        print(f"[done] {args.split}/{args.arm} {ds}: "
              f"{len(shard['records'])} records", flush=True)


def run_score(args):
    import amp_accuracy as AA

    orig = AA.shard_path

    def patched(split, arm, ds, K=256):
        return SC.shard_path(split, arm, ds)

    AA.shard_path = patched
    try:
        class _Args:
            pass
        a = _Args()
        a.split, a.arm, a.K = args.split, args.arm, SC.K
        a.selector = f"s1n_{COMBO_ARMS[args.arm]['stage1']}+{args.arm}"
        amp_run_score(a, None)
    finally:
        AA.shard_path = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "confirm"])
    ap.add_argument("--arms", required=True)
    ap.add_argument("--mode", default="both", choices=["gen", "score", "both"])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    manifest = SC.load_manifest()
    if args.mode in ("gen", "both"):
        model = SC.common.load_model(SC.common.BASELINE_MODEL,
                                     max_new_tokens=args.max_new_tokens)
        from model.native_qwen3 import NativeEngine
        eng = NativeEngine(model)
        for arm in args.arms.split(","):
            args.arm = arm
            run_gen(args, eng, manifest)
    if args.mode in ("score", "both"):
        for arm in args.arms.split(","):
            args.arm = arm
            run_score(args)


if __name__ == "__main__":
    main()
