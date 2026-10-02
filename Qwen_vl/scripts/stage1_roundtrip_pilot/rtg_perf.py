"""RTG pilot — paired live efficiency (protocol §6).

30 DEV samples (10 per dataset), arms E_MAIN025 vs R_MAIN025, FULLY LIVE
(prepare -> encode -> instruction embeds -> RTG/EADP scoring -> official
facility -> completion -> prefill -> decode; no bank lookups).  15 warm-up
paired blocks, interleaved arm order with alternating direction, CUDA
synced; BEFORE each measurement the previous out/state/KV references are
released, cache emptied, synced and peak VRAM reset; only CPU scalars are
stored.  ignore_eos fixed 64-token decode with length assertion.

Usage: python rtg_perf.py
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import rtg_common as RC
import amp_common as AC


N_WARMUP = 15
FIXED_TOKENS = 64


def release(*objs):
    for _o in objs:
        del _o
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


@torch.no_grad()
def run_live(eng, msg, ds, scorer: str, lam: float, timings: dict,
             qid=None, max_new_tokens: int = FIXED_TOKENS):
    wall0 = time.perf_counter()
    prep = eng.prepare(msg, ds)
    timings["image_preprocess_ms"] = (time.perf_counter() - wall0) * 1e3
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    V, DS = eng.encode(prep)
    ev[1].record()
    torch.cuda.synchronize()
    timings["vision_ms"] = ev[0].elapsed_time(ev[1])

    K_ = RC.K
    n_vis = prep["n_vis"]
    V_sel = None
    keep = None
    if scorer == "full":
        keep = torch.arange(n_vis, device=V.device)   # keep-all reference
    elif n_vis <= K_:
        keep = torch.arange(n_vis, device=V.device)
    elif scorer == "eadp":
        t0 = time.perf_counter()
        text_mean, text_seq = eng.instruction_embeds(msg, ds)
        torch.cuda.synchronize()
        timings["text_embeds_ms"] = (time.perf_counter() - t0) * 1e3
        ctx = dict(prep=prep, V=V, DS=DS, K=K_, engine=eng,
                   text_mean=text_mean, text_seq=text_seq)
        t0 = time.perf_counter()
        keep = AC.official_facility_keep(ctx, K_)
        torch.cuda.synchronize()
        timings["eadp_stage1_facility_ms"] = \
            (time.perf_counter() - t0) * 1e3
    else:
        from model.pruner import _sim_visual_impl
        import instrumented
        sms = eng.inner.visual.spatial_merge_size
        split_sizes = (prep["gthw"].prod(-1) // (sms ** 2)).tolist()
        pruner = RC._get_pruner(V, eng)
        keep_parts, offset = [], 0
        for i, n in enumerate(split_sizes):
            token_num = min(K_, n)
            if token_num >= n:
                keep_parts.append(
                    torch.arange(offset, offset + n, device=V.device))
                offset += n
                continue
            feats = V[offset:offset + n].unsqueeze(0)
            t0 = time.perf_counter()
            text_mean, text_seq = eng.instruction_embeds(msg, ds)
            torch.cuda.synchronize()
            timings["text_embeds_ms"] = \
                timings.get("text_embeds_ms", 0.0) + \
                (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            sim01 = _sim_visual_impl(feats)
            imp, w = RC.rtg_importance(feats, text_mean, text_seq,
                                       int(prep["gthw"][i, 1]) // sms,
                                       int(prep["gthw"][i, 2]) // sms,
                                       scorer, ds=ds,
                                       qid=qid, img_idx=i)
            torch.cuda.synchronize()
            timings["stage1_ms"] = timings.get("stage1_ms", 0.0) + \
                (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            sel, _ = instrumented.SELECTORS["facility"](imp, sim01,
                                                        token_num)
            torch.cuda.synchronize()
            timings["facility_ms"] = timings.get("facility_ms", 0.0) + \
                (time.perf_counter() - t0) * 1e3
            keep_parts.append(offset + sel[0].sort().values.to(V.device))
            offset += n
        keep = torch.cat(keep_parts)

    if lam > 0.0 and n_vis > K_:
        t0 = time.perf_counter()
        dropped_idx, gid, _ = AC.compute_assignment(V, keep)
        y = AC.merge_stream(V, keep, dropped_idx, gid, "uniform", lam)
        V_sel = y
        torch.cuda.synchronize()
        timings["completion_ms"] = (time.perf_counter() - t0) * 1e3

    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    st = eng.prefill(prep, V, DS, keep, V_sel=V_sel, DS_sel=None)
    ev[1].record()
    torch.cuda.synchronize()
    timings["llm_prefill_ms"] = ev[0].elapsed_time(ev[1])
    timings["ttft_ms"] = (time.perf_counter() - wall0) * 1e3
    wall1 = time.perf_counter()
    gen_ids, _text = eng.decode(st, max_new_tokens, ignore_eos=True)
    torch.cuda.synchronize()
    timings["decode_fixed_ms"] = (time.perf_counter() - wall1) * 1e3
    timings["n_gen"] = len(gen_ids)
    assert timings["n_gen"] == max_new_tokens, "fixed-length decode violated"
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    st_ref = st
    release(st_ref, prep, V, DS, V_sel)
    return dict(timings=timings, peak_mb=peak)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default=os.path.join(RC.OUT_DIR, "rtg_perf.json"))
    args = ap.parse_args()
    arms = [("FULL", "full", 0.0), ("E_GATHER", "eadp", 0.0),
            ("E_MAIN025", "eadp", 0.25), ("R_MAIN025", "rtg", 0.25)]

    model = AC.common.load_model(AC.common.BASELINE_MODEL,
                                 max_new_tokens=FIXED_TOKENS)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)
    manifest = json.load(open(os.path.join(AC.OUT_DIR, "manifest.json")))

    items = []
    for ds in RC.DS_LIST:
        dataset = AC.common.build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        for it in manifest["datasets"][ds]["dev"][:10]:
            row = dataset.data.iloc[it["idx"]]
            msg = AC.common.build_message(model, dataset, ds, row)
            items.append(dict(ds=ds, idx=it["idx"], message=msg))

    for i in range(N_WARMUP):
        it = items[i % len(items)]
        for _name, scorer, lam in arms:
            out = run_live(eng, it["message"], it["ds"], scorer, lam, {},
                           qid=it["idx"])
            release(out)
    torch.cuda.synchronize()

    records = {a: [] for a, _s, _l in arms}
    vram = {a: [] for a, _s, _l in arms}
    for i, it in enumerate(items):
        order = arms if i % 2 == 0 else list(reversed(arms))
        for arm, scorer, lam in order:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timings = {}
            out = run_live(eng, it["message"], it["ds"], scorer, lam,
                           timings, qid=it["idx"])
            vram[arm].append(out["peak_mb"])
            records[arm].append(dict(ds=it["ds"], idx=it["idx"],
                                     **out["timings"]))
            release(out)

    def med(x):
        return float(np.median(x)) if len(x) else None

    summary = dict(
        arms=[a for a, _s, _l in arms], warmup=N_WARMUP,
        fixed_tokens=FIXED_TOKENS, live=True,
        ttft_boundary="prepare -> encode -> embeds/scoring/selection -> "
                      "completion -> prefill -> first token",
        stats={a: dict(
            ttft_med_ms=med([r["ttft_ms"] for r in records[a]]),
            vision_med_ms=med([r["vision_ms"] for r in records[a]]),
            text_embeds_med_ms=med([r.get("text_embeds_ms") or 0.0
                                    for r in records[a]]),
            eadp_stage1_facility_med_ms=med(
                [r.get("eadp_stage1_facility_ms") or 0.0
                 for r in records[a]]),
            stage1_med_ms=med([r.get("stage1_ms") or 0.0
                               for r in records[a]]),
            facility_med_ms=med([r.get("facility_ms") or 0.0
                                 for r in records[a]]),
            completion_med_ms=med([r.get("completion_ms") or 0.0
                                   for r in records[a]]),
            prefill_med_ms=med([r["llm_prefill_ms"] for r in records[a]]),
            decode_fixed_med_ms=med([r["decode_fixed_ms"]
                                     for r in records[a]]),
            total_gen_med_ms=med([r["ttft_ms"] + r["decode_fixed_ms"]
                                  for r in records[a]]),
            peak_vram_p50_mb=float(np.median(vram[a])),
        ) for a in records},
        blocks=records)
    e = summary["stats"]["E_MAIN025"]["ttft_med_ms"]
    r = summary["stats"]["R_MAIN025"]["ttft_med_ms"]
    summary["ttft_increment_pct"] = (r - e) / e * 100.0
    with open(args.out + ".tmp", "w") as f:
        json.dump(summary, f, indent=1)
    os.replace(args.out + ".tmp", args.out)
    for a in summary["stats"]:
        s = summary["stats"][a]
        print(f"[{a}] ttft={s['ttft_med_ms']:.1f}ms "
              f"embeds={s['text_embeds_med_ms']:.2f} "
              f"stage1={s['stage1_med_ms']:.2f} "
              f"facility={s['facility_med_ms']:.2f} "
              f"eadp_s1+fac={s['eadp_stage1_facility_med_ms']:.2f} "
              f"completion={s['completion_med_ms']:.2f} "
              f"decode64={s['decode_fixed_med_ms']:.0f}ms "
              f"vram={s['peak_vram_p50_mb']:.0f}MB")
    print(f"TTFT increment R vs E: {summary['ttft_increment_pct']:.2f}%")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
