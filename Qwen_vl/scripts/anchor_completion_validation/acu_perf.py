"""Anchor Completion Validation — paired real-path efficiency (protocol §9).

30 questions (first 10 bank-covered rows per main-panel task), BASE vs
MAIN025 (full-method control NOT implementable, protocol §4.3 — never
faked).  Fixed 64-token generation with ignore_eos + actual-length assert;
interleaved block order; every measurement:

  1. releases the previous out/state/KV and ANY GPU tensor references,
  2. torch.cuda.synchronize() then reset_peak_memory_stats(),
  3. runs the FULL live path (prepare/encode/EADP scoring/facility/
     assignment/completion/prefill/decode — never a bank-lookup shortcut),
  4. stores CPU scalars only.

Usage: python acu_perf.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

import acu_common as AU
from acu_common import common as C

N_PER_DS = 10
WARMUP_PAIRS = 5
BLOCKS = 10
GEN_TOKENS = 64


def release(*objs):
    for _o in objs:
        del _o
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def measure_arm(eng, msg, ds, keep_fn, cfg):
    """One full live-path run; per-stage CUDA timings; CPU scalars only."""
    timings = {}
    wall0 = time.perf_counter()
    prep = eng.prepare(msg, ds)
    timings["image_preprocess_ms"] = (time.perf_counter() - wall0) * 1e3

    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    V, DS = eng.encode(prep)
    ev[1].record()
    torch.cuda.synchronize()
    timings["vision_ms"] = ev[0].elapsed_time(ev[1])

    # live EADP scoring + official facility (full path, no bank lookup)
    keep, stage_ms = keep_fn(eng, prep, V, msg, ds)
    timings.update(stage_ms)

    kind, lam = cfg["kind"], cfg.get("lam", 0.0)
    V_sel = DS_sel = None
    if kind != "base":
        t0 = time.perf_counter()
        dropped_idx, gid, sim = AU.compute_assignment(V, keep)
        V_sel = AU.merge_stream(V, keep, dropped_idx, gid, kind, lam,
                                cfg.get("tau", 0.1),
                                sim if kind == "sim" else None)
        torch.cuda.synchronize()
        timings["merge_ms"] = (time.perf_counter() - t0) * 1e3

    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    st = eng.prefill(prep, V, DS, keep, V_sel=V_sel, DS_sel=DS_sel)
    ev[1].record()
    torch.cuda.synchronize()
    timings["llm_prefill_ms"] = ev[0].elapsed_time(ev[1])
    timings["ttft_ms"] = (time.perf_counter() - wall0) * 1e3

    wall1 = time.perf_counter()
    gen_ids, _text = eng.decode(st, GEN_TOKENS, ignore_eos=True)
    torch.cuda.synchronize()
    timings["decode64_ms"] = (time.perf_counter() - wall1) * 1e3
    timings["total_ms"] = (time.perf_counter() - wall0) * 1e3
    timings["n_gen"] = len(gen_ids)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    out = dict(timings=timings, peak_mb=peak)
    release(st, prep, V, DS, V_sel)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(AU.OUT_DIR, "perf.json"))
    args = ap.parse_args()

    eng = AU.load_engine(max_new_tokens=GEN_TOKENS)
    from model.e0_selectors import _eadp_parts

    def keep_fn(eng_, prep, V, msg, ds):
        tm, ts = eng_.instruction_embeds(msg, ds)
        ctx = dict(prep=prep, V=V, K=AU.K, engine=eng_,
                   text_mean=tm, text_seq=ts)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        tms = {}
        ev[0].record()
        keep = _eadp_parts(AU.K, ctx, "facility")
        ev[1].record()
        torch.cuda.synchronize()
        tms["eadp_stage1_facility_ms"] = ev[0].elapsed_time(ev[1])
        return keep, tms

    records = []
    for ds in AU.DS_MAIN:
        bank = AU.load_bank(ds)
        keys = [k for k, v in bank.items() if v["n_vis"] > AU.K][:N_PER_DS]
        dataset = C.build_dataset(ds)
        eng.vlm.set_dump_image(dataset.dump_image)
        msgs = []
        for k in keys:
            row = dataset.data.iloc[int(k)]
            msgs.append(C.build_message(eng.vlm, dataset, ds, row))
        arms = [("BASE", AU.arm_cfg("BASE")),
                ("MAIN025", AU.arm_cfg("MAIN025"))]
        for w in range(WARMUP_PAIRS):
            for _name, cfg in arms:
                out = measure_arm(eng, msgs[w % len(msgs)], ds, keep_fn, cfg)
                release(out)
        for b in range(BLOCKS):
            for name, cfg in arms:
                msg = msgs[b % len(msgs)]
                out = measure_arm(eng, msg, ds, keep_fn, cfg)
                assert out["timings"]["n_gen"] == GEN_TOKENS, \
                    f"expected {GEN_TOKENS} tokens, " \
                    f"got {out['timings']['n_gen']}"
                records.append(dict(ds=ds, key=keys[b % len(keys)],
                                    arm=name, block=b, **out["timings"],
                                    peak_mb=out["peak_mb"]))
                release(out)
        print(f"[perf {ds}] done", flush=True)

    with open(args.out, "w") as f:
        json.dump(dict(records=records, code_commit=AU.repo_commit(),
                       gen_tokens=GEN_TOKENS, blocks=BLOCKS,
                       warmup_pairs=WARMUP_PAIRS), f, indent=1)

    print("\narm          TTFT  vision scor+fac  merge prefill decode64   VRAM")
    for name in ("BASE", "MAIN025"):
        sel = [r for r in records if r["arm"] == name]
        med = lambda k: statistics.median(r[k] for r in sel)  # noqa: E731
        merge_med = med("merge_ms") if any(r.get("merge_ms") for r in sel) \
            else 0.0
        print(f"{name:12s} {med('ttft_ms'):6.1f} {med('vision_ms'):6.1f} "
              f"{med('eadp_stage1_facility_ms'):8.1f} {merge_med:6.2f} "
              f"{med('llm_prefill_ms'):7.1f} {med('decode64_ms'):7.1f} "
              f"{med('peak_mb'):7.0f}")


if __name__ == "__main__":
    main()
