"""
M2 — performance harness. One engine, one code path, eight arms.

Pre-registration §6.3: batch size 1, >= 20 warm-up, >= 100 measured iterations,
a `torch.cuda.synchronize()` around every measured region, and **both** CUDA-event
and synchronised wall-clock timing. The prefill benchmark ends at the first
generated token; the decode benchmark generates exactly 32 tokens for every arm.

The 15-field breakdown (prereg §6.3) is produced per arm, plus the KV-cache split
between the early (L0-L4) and late (L5+) layers, the throughput, and the explicit
"did an extra forward / recomputation occur" flag.

Usage
    python scripts/discovery/m2_perf.py --iters 100 --warmup 20
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                     # noqa: E402
from common import OUTPUT_DIR, eadp_model_name                    # noqa: E402
from m1_common import load_m1_plan                                # noqa: E402
from m2_gdep import (LAYER, MODE_FULL, MODE_GDEP, MODE_PRELLM,    # noqa: E402
                     POLICY_PRESERVE, POLICY_RENUMBER,
                     GDEPConfig, GDEPEngine, N_VIS, dump_json)

# ---------------------------------------------------------------------------
# arm table.  `seed` for the candidates is the validation-only profiling choice
# (prereg §6.2): n=960 -> seed 2 (val head_recall@8 0.8167), n=240 -> seed 0
# (0.7854).  No held-out quantity was read to make that choice.
# ---------------------------------------------------------------------------
ARMS = [
    dict(tag="B0", mode=MODE_FULL, n_arm=None, seed=None, selector=None,
         policy=None, label="Full model, 1024 visual tokens, unpruned"),
    dict(tag="B1", mode=MODE_PRELLM, n_arm=None, seed=None, selector="facility",
         policy=None, label="Official EADP facility@256 (pre-LLM)"),
    dict(tag="B2", mode=MODE_PRELLM, n_arm=None, seed=None, selector="block8",
         policy=None, label="EADP block8@256 (pre-LLM)"),
    dict(tag="C1", mode=MODE_GDEP, n_arm=960, seed=2, selector="topk",
         policy=POLICY_PRESERVE, label="GDEP n=960 L4 LOCAL-MLP + TopK@256"),
    dict(tag="C0", mode=MODE_GDEP, n_arm=240, seed=0, selector="topk",
         policy=POLICY_PRESERVE, label="GDEP n=240 L4 LOCAL-MLP + TopK@256"),
    dict(tag="C2", mode=MODE_GDEP, n_arm=960, seed=2, selector="block8",
         policy=POLICY_PRESERVE, label="GDEP n=960 L4 LOCAL-MLP + block8@256"),
    dict(tag="C3", mode=MODE_GDEP, n_arm=960, seed=2, selector="facility",
         policy=POLICY_PRESERVE, label="GDEP n=960 L4 LOCAL-MLP + facility@256"),
    dict(tag="C1-R", mode=MODE_GDEP, n_arm=960, seed=2, selector="topk",
         policy=POLICY_RENUMBER, label="C1 under RENUMBER (declared control)"),
]

STAGE_KEYS = ["image_preprocess_ms", "vision_encoder_ms", "eadp_scoring_ms",
              "selector_ms", "L0_L4_ms", "scorer_ms", "token_compaction_ms",
              "llm_forward_ms"]


def build_engine(model, arm):
    cfg = GDEPConfig(mode=arm["mode"], budget=256, selector=arm["selector"] or "topk",
                     pos_policy=arm["policy"] or POLICY_PRESERVE,
                     n_arm=arm["n_arm"] or 240, seed=arm["seed"] or 0,
                     tag=arm["tag"])
    return GDEPEngine.from_checkpoint(cfg, model=model, max_new_tokens=32)


def bench_prefill(eng, msg, ds_name, iters, warmup, timings_stats):
    """One prefill benchmark; returns (cuda_event_ms, wall_ms, last info).

    The **whole per-request prefill cycle** is inside the timed region: image
    preprocessing, the vision tower, whatever the arm does to choose tokens, and
    the LLM layers. TTFT without the vision tower is not TTFT -- the tower is
    ~115 ms and every arm pays it -- so it is timed on every iteration rather
    than hoisted out.
    """
    ev, wall = [], []
    first_info = None
    for i in range(warmup + iters):
        t = {}
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        prep = eng.prepare(msg, ds_name, t)
        st, info = eng.prefill(prep, timings=t)
        torch.cuda.synchronize()
        w1 = time.perf_counter()
        if i >= warmup:
            wall.append((w1 - w0) * 1e3)
            ev.append(sum(t.get(k, 0.0) for k in STAGE_KEYS))
            for k in STAGE_KEYS:
                timings_stats.setdefault(k, []).append(float(t.get(k, 0.0)))
        if i == warmup:
            first_info = info
        del st, prep
    return ev, wall, first_info


def bench_decode(eng, msg, ds_name, n_tokens, iters, warmup):
    """Time exactly `n_tokens` decode steps per iteration.

    Each iteration re-runs the whole per-request front end (prepare + prefill)
    and discards its time: the decode cache must start from the same state on
    every iteration, and reusing one cache across iterations would let it grow
    unbounded.
    """
    out = []
    with torch.no_grad():
        for i in range(warmup + iters):
            prep = eng.prepare(msg, ds_name)
            st2, _ = eng.prefill(prep)
            del prep
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            w0 = time.perf_counter()
            t0.record()
            ids = eng.decode(st2, n_tokens, ignore_eos=True)
            t1.record()
            torch.cuda.synchronize()
            w1 = time.perf_counter()
            if i >= warmup:
                out.append(dict(cuda=t0.elapsed_time(t1), wall=(w1 - w0) * 1e3,
                                n=len(ids)))
            del st2
    return out


def stats_of(xs):
    xs = sorted(float(x) for x in xs)
    n = len(xs)
    return dict(mean=float(np.mean(xs)), median=float(np.median(xs)),
                p90=float(xs[min(n - 1, int(round(0.9 * (n - 1))))]),
                std=float(np.std(xs, ddof=1) if n > 1 else 0.0),
                min=xs[0], max=xs[-1], n=n)


def stage(model, args):
    """Run the timing grid against an already-loaded model."""
    _, plan, keys, rows_of = load_m1_plan()
    test_rows = list(rows_of["test"])
    # Fixed input for every arm: the first held-out instance, TextVQA_VAL.
    key0 = keys[test_rows[0]]
    ds_name, idx = key0.rsplit("_", 1)
    dataset = common.build_dataset(ds_name)
    print(f"[input] fixed instance {key0} ({ds_name}[{idx}])")
    torch.set_grad_enabled(False)
    model.set_dump_image(dataset.dump_image)
    msg = common.build_message(model, dataset, ds_name, dataset.data.iloc[int(idx)])

    base_mem = torch.cuda.memory_allocated() / (1024 ** 2)
    results = dict(config=vars(args), arms={}, input_key=key0, input_dataset=ds_name,
                   baseline_allocated_mb=base_mem, batch_size=1,
                   batch4=dict(run=False, reason=(
                       "the engine's compaction is per-sequence (each instance has "
                       "its own visual span and its own kept set); batch>1 was not "
                       "implemented and is therefore not reported, per prereg 6.3")))

    for arm in ARMS:
        if arm["tag"] not in args.arms:
            continue
        print(f"\n=== {arm['tag']}: {arm['label']} ===")
        rec = dict(arm)
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated() / (1024 ** 2)
            eng = build_engine(model, arm)

            # ---- prefill (whole per-request cycle inside the timed region) ----
            tstats = {}
            torch.cuda.reset_peak_memory_stats()
            ev, wall, info = bench_prefill(eng, msg, ds_name, args.iters,
                                           args.warmup, tstats)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

            rec["prefill"] = dict(
                event=stats_of(ev), wall=stats_of(wall),
                stages={k: stats_of(v) for k, v in tstats.items() if v},
                n_kept=info["n_kept"], context_len=info["context_len"],
                seq_full=info["seq_full"], n_text=info["n_text"],
                kv_seq_len=info["kv_seq_len"],
                kv_bytes_early=info["kv_bytes_early"],
                kv_bytes_late=info["kv_bytes_late"],
                kv_bytes_total=info["kv_bytes_early"] + info["kv_bytes_late"],
                kv_total_if_no_prune=_kv_reference_bytes(eng, info),
                pos_ids_contiguous=info.get("pos_ids_contiguous"),
                cache_policy=info["cache_policy"],
                pos_policy=info["pos_policy"],
                scorer_params=info["scorer_params"],
                cfg_hash=info["cfg_hash"], cfg_key=info["cfg_key"],
                layer_calls=info["layer_calls_prefill"],
                stage_timing_raw=info.get("stage_timing"),
            )
            rec["peak_allocated_mb"] = float(peak)
            rec["peak_delta_over_load_mb"] = float(peak - mem_before)

            # ---- decode, exactly 32 tokens ----
            torch.cuda.reset_peak_memory_stats()
            dec = bench_decode(eng, msg, ds_name, args.decode_tokens,
                               args.iters, args.warmup)
            rec["decode"] = dict(
                total_cuda=stats_of([d["cuda"] for d in dec]),
                total_wall=stats_of([d["wall"] for d in dec]),
                tpot_ms=float(np.mean([d["cuda"] for d in dec])
                              / args.decode_tokens),
                wall_tpot_ms=float(np.mean([d["wall"] for d in dec])
                                   / args.decode_tokens),
                n_tokens=args.decode_tokens,
                peak_allocated_mb=float(torch.cuda.max_memory_allocated()
                                        / (1024 ** 2)),
            )
            rec["throughput_rps"] = 1000.0 / rec["prefill"]["event"]["mean"]
            rec["extra_forward_or_recompute"] = _recompute_flag(info, eng)
            print(f"  prefill event mean {rec['prefill']['event']['mean']:.2f} ms "
                  f"(median {rec['prefill']['event']['median']:.2f}, "
                  f"p90 {rec['prefill']['event']['p90']:.2f})  "
                  f"wall mean {rec['prefill']['wall']['mean']:.2f} ms")
            print(f"  decode {args.decode_tokens}t: "
                  f"{rec['decode']['total_cuda']['mean']:.2f} ms  "
                  f"TPOT {rec['decode']['tpot_ms']:.3f} ms  "
                  f"peak {rec['peak_allocated_mb']:.0f} MB")
            for k, v in rec["prefill"]["stages"].items():
                print(f"    {k:24s} {v['mean']:8.3f} ms")
            del eng
        except Exception:
            traceback.print_exc()
            rec["error"] = traceback.format_exc()[-2000:]
        results["arms"][arm["tag"]] = rec
        dump_json(f"{args.tag}.json", results)

    _speedups(results)
    dump_json(f"{args.tag}.json", results)
    print(f"\n[done] {len(results['arms'])} arms -> {args.tag}.json")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--arms", nargs="+", default=[a["tag"] for a in ARMS])
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--tag", default="m2_perf")
    args = ap.parse_args()
    model = common.load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=32)
    model.model.eval()
    stage(model, args)


def _kv_reference_bytes(eng, info):
    """What the cache would cost if nothing had been pruned (same dtype/heads)."""
    cfg = eng.text.config
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    per_tok = 2 * cfg.num_key_value_heads * hd * eng.n_layers * 2   # K+V, fp16
    return int(per_tok * info["seq_full"])


def _recompute_flag(info, eng):
    """No arm may run a layer it does not need, nor run one twice."""
    if info["mode"] == MODE_GDEP:
        expect = (info["layer"] + 1) + (eng.n_layers - info["layer"] - 1)
    else:
        expect = eng.n_layers
    return bool(info["layer_calls_prefill"] != expect)


def _speedups(results):
    pre = results["arms"].get("B0", {}).get("prefill", {}).get("event", {}).get("median")
    for ref in ("B0", "B1", "B2"):
        r = results["arms"].get(ref, {}).get("prefill", {}).get("event", {}).get("median")
        if not r:
            continue
        for tag, rec in results["arms"].items():
            if "prefill" in rec:
                rec.setdefault("speedup_vs", {})[ref] = r / rec["prefill"]["event"]["median"]


if __name__ == "__main__":
    main()
