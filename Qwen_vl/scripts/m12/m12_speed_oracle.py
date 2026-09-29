"""M12 gate 2: speed oracle -- does a real pre-encoder sparse ViT path buy
real wall-clock on the A40, independent of selector quality?

Protocol (E0 perf-paired style, clean GPU required):
  * structured deterministic selection = uniform 2-D lattice (quality-agnostic);
  * keep ratios of the 1024 merged tokens: 100 / 75 / 50 / 37.5 / 25 %;
  * R-res arms at matched merged-token counts (side = 32*round(sqrt(K))/1
    quantized to a multiple of 32);
  * 15 warm-up + >= 30 measured reps per (arm, sample), samples stratified
    over the OCR-panel DEV rows (include small and large originals);
  * per-rep windows: preprocess (CPU wall), gate (sync'd wall), ViT stages
    (patch_embed / pos prep / blocks / deepstack+merger), total vision path,
    LLM prefill (CUDA event), TTFT (sync-to-sync wall), decode 32 steps
    (ignore_eos), peak memory;
  * report median / mean / std; theoretical ViT GFLOPs per arm.

Usage: python m12_speed_oracle.py [--blocks 30]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

DISC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "discovery")
sys.path.insert(0, DISC_DIR)
import common  # noqa: E402

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "m12")
WARMUP = 15

# (arm_id, kind, param): kind 'rg' -> sparse path with mode/keep;
# 'rres' -> fixed resolution side; 'b0' -> stock full path
RATIOS = [1.0, 0.75, 0.5, 0.375, 0.25]


def build_arms():
    arms = [("b0", "b0", 1024)]
    for r in RATIOS[1:]:
        arms.append((f"lat{int(round(r*100))}", "rg", ("uniform", int(round(1024 * r)))))
    # R-res matched: merged tokens (side/32)^2, side multiple of 32
    for r in RATIOS[1:]:
        k = int(round(1024 * r))
        side = int(32 * round((k ** 0.5)))
        arms.append((f"rres{side}", "rres", side))
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "m12_speed_oracle.json"))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    plan = json.load(open(os.path.join(common.QWEN_ROOT, "outputs", "e0",
                                       "e0_plan.json")))
    model = common.load_model(common.BASELINE_MODEL, max_new_tokens=64)
    from model.native_qwen3 import NativeEngine
    from model import retinagate as rg
    import vlmeval.vlm.qwen3_vl.model_fixed_res as mfr
    eng = NativeEngine(model)

    samples = []
    for ds in ("TextVQA_VAL", "DocVQA_VAL", "OCRBench", "ChartQA_TEST"):
        rows = plan["datasets"][ds]["dev_rows"]
        step = max(1, len(rows) // 3)
        for idx in rows[::step][:3]:
            dataset = common.build_dataset(ds)
            model.set_dump_image(dataset.dump_image)
            row = dataset.data.iloc[idx]
            samples.append((ds, common.build_message(model, dataset, ds, row)))

    arms = build_arms()
    vcfg = eng.inner.visual.config

    def run_arm(arm, msg, ds):
        aid, kind, param = arm
        t = {}
        if kind == "rres":
            mfr.QWEN3_FIXED_RESOLUTION = param
            model.min_pixels = param * param
            model.max_pixels = param * param
        try:
            if kind == "rg":
                out = rg.rg_generate(eng, msg, ds, k=param[1], mode=param[0],
                                     max_new_tokens=8, ignore_eos=True,
                                     timings=t, seed=args.seed)
            else:
                k = 1024
                out = eng.generate(msg, ds, K=k, selector="identity",
                                   max_new_tokens=8, ignore_eos=True, timings=t)
        finally:
            if kind == "rres":
                mfr.QWEN3_FIXED_RESOLUTION = 1024
                model.min_pixels = 1024 * 1024
                model.max_pixels = 1024 * 1024
        n_groups = out["meta"]["n_groups_kept"] if kind == "rg" \
            else out["meta"]["n_vis_kept"]
        n_patch = n_groups * 4
        rec = dict(ttft_ms=t.get("ttft_ms"),
                   preprocess_ms=t.get("image_preprocess_ms"),
                   vision_ms=t.get("vision_ms"),
                   gate_ms=t.get("gate_ms"),
                   vit_patch_embed_ms=t.get("vit_patch_embed_ms"),
                   vit_pos_prep_ms=t.get("vit_pos_prep_ms"),
                   vit_blocks_ms=t.get("vit_blocks_ms"),
                   vit_merger_ms=t.get("vit_merger_ms"),
                   selector_ms=t.get("selector_ms"),
                   llm_prefill_ms=t.get("llm_prefill_ms"),
                   n_vis_kept=n_groups,
                   vit_gflops=rg.vision_flopsgf(n_patch, vcfg))
        return rec

    # warm-up
    print(f"[oracle] warm-up {WARMUP} blocks x {len(arms)} arms", flush=True)
    for b in range(WARMUP):
        s = samples[b % len(samples)]
        for arm in arms:
            run_arm(arm, s[1], s[0])

    blocks = []
    for b in range(args.blocks):
        s = samples[b % len(samples)]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        runs = {}
        for arm in arms:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rec = run_arm(arm, s[1], s[0])
            rec["wall_ms"] = (time.perf_counter() - t0) * 1e3
            rec["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
            runs[arm[0]] = rec
        blocks.append(dict(sample=dict(ds=s[0], bi=b), runs=runs))
        print(f"[oracle] block {b+1}/{args.blocks} done", flush=True)

    # summary: median / mean / std per arm
    def agg(key):
        vals = [r[key] for r in runs.values() if r.get(key) is not None]
        return vals

    summary = {}
    for arm in arms:
        aid = arm[0]
        keys = blocks[0]["runs"][aid].keys()
        s = {}
        for key in keys:
            vals = [bl["runs"][aid][key] for bl in blocks
                    if bl["runs"][aid].get(key) is not None]
            if not vals:
                continue
            s[key] = dict(median=statistics.median(vals),
                          mean=sum(vals) / len(vals),
                          std=statistics.pstdev(vals), n=len(vals))
        summary[aid] = s

    out = dict(arms=[a[0] for a in arms], ratios=RATIOS, seed=args.seed,
               n_blocks=args.blocks, summary=summary, blocks=blocks)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[saved] {args.out}", flush=True)

    # console table
    print(f"{'arm':10s} {'K':>5s} {'vit_med':>8s} {'blocks':>8s} "
          f"{'prep_med':>9s} {'prefill':>8s} {'ttft':>8s} {'gflops':>8s}")
    for aid in summary:
        s = summary[aid]
        k = int(s.get("n_vis_kept", {}).get("median", 0))
        print(f"{aid:10s} {k:5d} "
              f"{s.get('vision_ms', {}).get('median', float('nan')):8.1f} "
              f"{s.get('vit_blocks_ms', {}).get('median', float('nan')):8.1f} "
              f"{s.get('preprocess_ms', {}).get('median', float('nan')):9.1f} "
              f"{s.get('llm_prefill_ms', {}).get('median', float('nan')):8.1f} "
              f"{s.get('ttft_ms', {}).get('median', float('nan')):8.1f} "
              f"{s.get('vit_gflops', {}).get('median', float('nan')):8.1f}")


if __name__ == "__main__":
    main()
