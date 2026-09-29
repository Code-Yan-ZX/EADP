"""E0 M5a: paired-block efficiency protocol (prereg §5.2, M2 amendment §7).

Same-process, same-sample paired measurement: 15 warm-up + >=100 measured
blocks; arm order shuffled per block; samples stratified over DEV datasets
including high- and low-resolution originals.  Windows per run: preprocess
(CPU wall), ViT (CUDA event), selection, LLM prefill (CUDA event), TTFT
(wall, sync-to-sync), decode 32 steps (ignore_eos), peak memory.  Paired
contrast: within-block difference of the two windows, block-bootstrap 95 % CI
of the mean difference, win counts.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import time

import torch
from PIL import Image

DISC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "discovery")
sys.path.insert(0, DISC_DIR)
import common  # noqa: E402

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "e0")

WARMUP_BLOCKS = 15
MEASURE_BLOCKS = 100
DECODE_STEPS = 32

# (arm_id, selector, deepstack, pos, side, K)
def build_arms(K=256, include_pace_all_k=True):
    arms = [("b0", "identity", True, "mrope3d", None, 1024)]
    for name in ("b1", "b2", "divprune", "cdpruner", "hiprune", "visionzip",
                 "fastv", "pdrop", "sparsevlm"):
        arms.append((name, name, True, "mrope3d", None, K))
    arms.append((f"rres512", "identity", True, "mrope3d", 512, 1024))
    arms.append((f"rres352", "identity", True, "mrope3d", 352, 1024))
    arms.append((f"rres384", "identity", True, "mrope3d", 384, 1024))
    arms.append((f"rres256", "identity", True, "mrope3d", 256, 1024))
    if include_pace_all_k:
        for k in (64, 128, 256):
            arms.append((f"pace", "pace", True, "mrope3d", None, k))
    return arms


def pick_samples(plan, n_per_ds=4):
    """Stratified DEV samples; include high/low original resolution."""
    from vlmeval.dataset import build_dataset as vlmeval_build
    samples = []
    for ds, info in plan["datasets"].items():
        dataset = vlmeval_build(ds)
        rows = info["dev_rows"]
        step = max(1, len(rows) // n_per_ds)
        picked = rows[::step][:n_per_ds]
        for idx in picked:
            row = dataset.data.iloc[idx]
            side = None
            ip = row.get("image_path", None)
            try:
                if isinstance(ip, str) and ip.strip():
                    with Image.open(ip) as im:
                        side = max(im.size)
                elif isinstance(row.get("image"), str) and row["image"].startswith("/9j"):
                    with Image.open(io.BytesIO(
                            __import__("base64").b64decode(row["image"]))) as im:
                        side = max(im.size)
            except Exception:
                side = None
            samples.append(dict(ds=ds, idx=int(idx), side=side))
    samples.sort(key=lambda s: (s["side"] or 0))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=MEASURE_BLOCKS)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "e0_perf_paired.json"))
    args = ap.parse_args()

    plan = json.load(open(os.path.join(OUT_DIR, "e0_plan.json")))
    model = common.load_model(common.BASELINE_MODEL, max_new_tokens=64)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)
    from model.e0_selectors import run_selector
    import vlmeval.vlm.qwen3_vl.model_fixed_res as mfr

    samples = pick_samples(plan)
    arms = build_arms()
    rng = random.Random(args.seed)

    # pre-build messages
    msgs = {}
    for s in samples:
        if s["ds"] not in msgs:
            dataset = common.build_dataset(s["ds"])
            model.set_dump_image(dataset.dump_image)
            msgs[s["ds"]] = dataset
        ds = msgs[s["ds"]]
        row = ds.data.iloc[s["idx"]]
        s["message"] = common.build_message(model, ds, s["ds"], row)

    def run_arm(arm, message, ds_name, timings):
        arm_id, selector, deepstack, pos, side, K = arm
        if side:
            mfr.QWEN3_FIXED_RESOLUTION = side
        try:
            out = eng.generate(message, ds_name, K=K, selector=selector,
                               deepstack=deepstack, pos=pos,
                               max_new_tokens=8, ignore_eos=True,
                               timings=timings)
        finally:
            if side:
                mfr.QWEN3_FIXED_RESOLUTION = 1024
        return out

    # warm-up
    print(f"[perf] warm-up {WARMUP_BLOCKS} blocks", flush=True)
    for b in range(WARMUP_BLOCKS):
        s = samples[b % len(samples)]
        for arm in arms:
            run_arm(arm, s["message"], s["ds"], {})

    blocks = []
    for b in range(args.blocks):
        s = samples[b % len(samples)]
        order = arms[:]
        rng.shuffle(order)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        block = dict(sample=dict(ds=s["ds"], idx=s["idx"], side=s["side"]),
                     runs={})
        for arm in order:
            timings = {}
            t0 = time.perf_counter()
            out = run_arm(arm, s["message"], s["ds"], timings)
            wall = (time.perf_counter() - t0) * 1e3
            # decode 32 steps ignore_eos
            torch.cuda.synchronize()
            td = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            td[0].record()
            gen_ids, _ = eng.decode(out["state"], DECODE_STEPS, ignore_eos=True)
            td[1].record()
            torch.cuda.synchronize()
            rec = dict(ttft_ms=timings.get("ttft_ms"),
                       preprocess_ms=timings.get("image_preprocess_ms"),
                       vision_ms=timings.get("vision_ms"),
                       selector_ms=timings.get("selector_ms"),
                       llm_prefill_ms=timings.get("llm_prefill_ms"),
                       decode32_ms=td[0].elapsed_time(td[1]),
                       wall_ms=wall,
                       peak_mem_mb=torch.cuda.max_memory_allocated() / 2**20,
                       n_vis_kept=out["meta"]["n_vis_kept"])
            block["runs"][arm[0] + (f"@{arm[5]}" if arm[0] == "pace" else "")] = rec
        blocks.append(block)
        print(f"[perf] block {b+1}/{args.blocks} done", flush=True)

    with open(args.out, "w") as f:
        json.dump(dict(arms=[a[0] for a in arms],
                       blocks=blocks, seed=args.seed), f, indent=1)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
