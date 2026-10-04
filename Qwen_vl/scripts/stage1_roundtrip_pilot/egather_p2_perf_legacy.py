"""P2 — legacy online efficiency panel (2026-10-04 AnchorZip round).

Arms: FULL / E_GATHER (official EADP) / L_R_MAIN025 (RTG + facility +
Completion 0.25), ALL under the official-legacy engine config
(deepstack=False, pos='1d') — same scenario as the Table-4 accuracy runs.
Selection and Completion are fully ONLINE (no bank reads): prepare ->
encode -> instruction embeds -> EADP-or-RTG scoring -> official facility ->
completion -> prefill -> decode.

Per item x arm the run executes BOTH decode settings:
  * fixed:  ignore_eos=True, exactly 64 tokens (length asserted)
  * natural: ignore_eos=False, max 2048 tokens (actual output length kept)

Interleaved arm order with alternating direction, 15 warm-up paired
blocks, CUDA-synced boundaries; before each measurement the previous
state/KV references are released, cache emptied, peak VRAM reset; only
CPU scalars are stored.  Paired per-item raw values + medians/p90 +
paired differences are saved.

FLOPs: NOT recorded — a reliable per-range FLOP count needs operator-level
accounting that this round does not have; fields are left blank rather
than inferred from token-reduction ratios.

Usage: python egather_p2_perf_legacy.py --panel dev30
       python egather_p2_perf_legacy.py --panel table4 --out <json>
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
import rtg_perf as RP

LEGACY_FLAGS = dict(deepstack=False, pos="1d")
ARMS = [("FULL", "full", 0.0), ("E_GATHER", "eadp", 0.0),
        ("L_R_MAIN025", "rtg", 0.25)]
TABLE4_DS = ["TextVQA_VAL", "ChartQA_TEST", "AI2D_TEST", "OCRBench",
             "HallusionBench", "MME", "MMBench_DEV_EN_V11",
             "MMBench_DEV_CN_V11", "DocVQA_VAL", "InfoVQA_VAL"]


@torch.no_grad()
def run_live(eng, msg, ds, scorer: str, lam: float, timings: dict,
             qid=None, max_new_tokens: int = RP.FIXED_TOKENS,
             ignore_eos: bool = True):
    """Legacy variant of rtg_perf.run_live — only deltas: prefill legacy
    flags, natural-EOS mode, actual N/K recording."""
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
    st = eng.prefill(prep, V, DS, keep, V_sel=V_sel, DS_sel=None,
                     **LEGACY_FLAGS)
    ev[1].record()
    torch.cuda.synchronize()
    timings["llm_prefill_ms"] = ev[0].elapsed_time(ev[1])
    timings["ttft_ms"] = (time.perf_counter() - wall0) * 1e3
    wall1 = time.perf_counter()
    gen_ids, _text = eng.decode(st, max_new_tokens, ignore_eos=ignore_eos)
    torch.cuda.synchronize()
    timings["decode_ms"] = (time.perf_counter() - wall1) * 1e3
    timings["n_gen"] = len(gen_ids)
    timings["n_vis"] = int(n_vis)
    timings["n_kept"] = int(st.keep_idx.numel())
    if ignore_eos:
        assert timings["n_gen"] == max_new_tokens, \
            "fixed-length decode violated"
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    st_ref = st
    RP.release(st_ref, prep, V, DS, V_sel)
    return dict(timings=timings, peak_mb=peak)


def build_items(panel: str):
    if panel == "dev30":
        manifest = json.load(open(
            os.path.join(AC.OUT_DIR, "manifest.json")))
        out = []
        for ds in ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]:
            for it in manifest["datasets"][ds]["dev"][:10]:
                out.append((ds, it["idx"]))
        return out
    elif panel == "table4":
        out = []
        for ds in TABLE4_DS:
            for i in range(5):        # FIXED rule: first 5 rows per task
                out.append((ds, i))
        return out
    raise ValueError(panel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="dev30",
                    choices=["dev30", "table4"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or os.path.join(
        RC.OUT_DIR, "legacy_full", f"p2_perf_legacy_{args.panel}.json")

    model = AC.common.load_model(AC.common.BASELINE_MODEL,
                                 max_new_tokens=2048)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)

    spec = build_items(args.panel)
    datasets = {}
    for ds, _i in spec:
        if ds not in datasets:
            d = AC.common.build_dataset(ds)
            model.set_dump_image(d.dump_image)
            datasets[ds] = d

    def message_for(ds, idx):
        row = datasets[ds].data.iloc[idx]
        return AC.common.build_message(model, datasets[ds], ds, row)

    modes = [("fixed64", 64, True), ("natural", 2048, False)]
    runs = [(a, s, l, m) for a, s, l in ARMS for m in modes]

    for i in range(RP.N_WARMUP):
        ds, idx = spec[i % len(spec)]
        msg = message_for(ds, idx)
        for _a, scorer, lam, (mname, mnt, mignore) in runs:
            out = run_live(eng, msg, ds, scorer, lam, {}, qid=idx,
                           max_new_tokens=mnt, ignore_eos=mignore)
            RP.release(out)
    torch.cuda.synchronize()

    records = {a: {m[0]: [] for m in modes} for a, _s, _l in ARMS}
    for i, (ds, idx) in enumerate(spec):
        msg = message_for(ds, idx)
        order = runs if i % 2 == 0 else list(reversed(runs))
        for arm, scorer, lam, (mname, mnt, mignore) in order:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timings = {}
            out = run_live(eng, msg, ds, scorer, lam, timings, qid=idx,
                           max_new_tokens=mnt, ignore_eos=mignore)
            records[arm][mname].append(
                dict(ds=ds, idx=idx, **out["timings"],
                     peak_mb=out["peak_mb"]))
            RP.release(out)

    def stats(rs):
        if not rs:
            return {}
        f = np.array([r["ttft_ms"] + r["decode_ms"] for r in rs])
        return dict(
            n=len(rs),
            ttft_med_ms=float(np.median([r["ttft_ms"] for r in rs])),
            ttft_p90_ms=float(np.percentile([r["ttft_ms"] for r in rs],
                                            90)),
            total_med_ms=float(np.median(f)),
            total_p90_ms=float(np.percentile(f, 90)),
            vision_med_ms=float(np.median([r["vision_ms"] for r in rs])),
            text_embeds_med_ms=float(np.median(
                [r.get("text_embeds_ms") or 0.0 for r in rs])),
            eadp_stage1_facility_med_ms=float(np.median(
                [r.get("eadp_stage1_facility_ms") or 0.0 for r in rs])),
            stage1_med_ms=float(np.median(
                [r.get("stage1_ms") or 0.0 for r in rs])),
            facility_med_ms=float(np.median(
                [r.get("facility_ms") or 0.0 for r in rs])),
            completion_med_ms=float(np.median(
                [r.get("completion_ms") or 0.0 for r in rs])),
            prefill_med_ms=float(np.median(
                [r["llm_prefill_ms"] for r in rs])),
            decode_med_ms=float(np.median([r["decode_ms"] for r in rs])),
            n_gen_med=float(np.median([r["n_gen"] for r in rs])),
            n_vis_med=float(np.median([r["n_vis"] for r in rs])),
            n_kept_med=float(np.median([r["n_kept"] for r in rs])),
            peak_vram_p50_mb=float(np.median([r["peak_mb"] for r in rs])),
        )

    def paired(arm_a, arm_b, mode):
        ra = {(r["ds"], r["idx"]): r for r in records[arm_a][mode]}
        rb = {(r["ds"], r["idx"]): r for r in records[arm_b][mode]}
        keys = sorted(set(ra) & set(rb))
        return [dict(ds=k[0], idx=k[1],
                     d_ttft_ms=ra[k]["ttft_ms"] - rb[k]["ttft_ms"],
                     d_total_ms=(ra[k]["ttft_ms"] + ra[k]["decode_ms"])
                                - (rb[k]["ttft_ms"] + rb[k]["decode_ms"]),
                     n_gen_a=ra[k]["n_gen"], n_gen_b=rb[k]["n_gen"])
               for k in keys]

    summary = dict(
        arm="legacy_p2_perf", panel=args.panel, pipeline="official_legacy",
        legacy_flags=LEGACY_FLAGS, warmup=RP.N_WARMUP,
        fixed_tokens=64, live=True, n_items=len(spec),
        items=[dict(ds=d, idx=i) for d, i in spec],
        ttft_boundary="prepare -> encode -> embeds/scoring/selection -> "
                      "completion -> prefill -> first token",
        flops=dict(status="not_recorded",
                   reason="no reliable per-range FLOP counter in this "
                          "round; left blank instead of inferring from "
                          "token-reduction ratios"),
        stats={a: {m: stats(records[a][m]) for m in records[a]}
               for a in records},
        paired={f"{a}-vs-{b}": {m: paired(a, b, m)
                                for m in ("fixed64", "natural")}
                for a, b in [("L_R_MAIN025", "E_GATHER"),
                             ("E_GATHER", "FULL"),
                             ("L_R_MAIN025", "FULL")]},
        blocks=records)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=1)
    os.replace(tmp, out_path)
    for a in records:
        for m in records[a]:
            s = summary["stats"][a][m]
            if s:
                print(f"[{a}/{m}] ttft={s['ttft_med_ms']:.1f}ms "
                      f"total={s['total_med_ms']:.0f}ms "
                      f"prefill={s['prefill_med_ms']:.0f} "
                      f"decode={s['decode_med_ms']:.0f} "
                      f"n_gen={s['n_gen_med']:.0f} "
                      f"vram={s['peak_vram_p50_mb']:.0f}MB")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
