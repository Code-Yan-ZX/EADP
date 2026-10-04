"""Quick paired cost check (diagnostic, NOT part of the frozen protocol):

FULL (identity, all 1024 visual tokens)  vs
FAC256 (official facility hard prune K=256 = paper EADP arm = ACV "BASE") vs
MAIN025 (facility + Anchor Completion merge lam=0.25 = frozen method).

Question being answered: does the pruned path actually run faster / cheaper
than passing all tokens?  Same paired-block methodology as acu_perf.py
(protocol §9 idiom): same question measured in all arms, arm order shuffled
per block, full live path, fixed 64-token decode with ignore_eos,
per-stage CUDA timings + peak VRAM.

Usage: python acu_perf_quick_cost.py [--n-questions 10] [--gen-tokens 64]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time

import torch

import acu_common as AU
from acu_common import common as C

WARMUP_PAIRS = 2


def release(*objs):
    for _o in objs:
        del _o
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def measure_arm(eng, msg, ds, keep_fn, cfg):
    """One full live-path run; per-stage timings; CPU scalars only."""
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
    gen_ids, _text = eng.decode(st, cfg["gen_tokens"], ignore_eos=True)
    torch.cuda.synchronize()
    timings["decode_ms"] = (time.perf_counter() - wall1) * 1e3
    timings["total_ms"] = (time.perf_counter() - wall0) * 1e3
    timings["n_gen"] = len(gen_ids)
    timings["n_vis"] = int(V.shape[0])
    timings["n_keep"] = int(keep.numel())
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    out = dict(timings=timings, peak_mb=peak)
    release(st, prep, V, DS, V_sel)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=10)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--out",
                    default="perf_quick_cost.json")
    args = ap.parse_args()

    eng = AU.load_engine(max_new_tokens=args.gen_tokens)
    from model.e0_selectors import _eadp_parts

    def keep_fn_full(eng_, prep, V, msg, ds):
        return torch.arange(V.shape[0], device=V.device), \
            dict(selector_ms=0.0)

    def keep_fn_fac(eng_, prep, V, msg, ds):
        tm, ts = eng_.instruction_embeds(msg, ds)
        ctx = dict(prep=prep, V=V, K=AU.K, engine=eng_,
                   text_mean=tm, text_seq=ts)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        tms = {}
        ev[0].record()
        keep = _eadp_parts(AU.K, ctx, "facility")
        ev[1].record()
        torch.cuda.synchronize()
        tms["selector_ms"] = ev[0].elapsed_time(ev[1])
        return keep, tms

    arms = [("FULL", dict(kind="base", lam=0.0, selector="identity",
                          keep_fn=keep_fn_full)),
            ("FAC256", dict(kind="base", lam=0.0, selector="b1",
                            keep_fn=keep_fn_fac)),
            ("MAIN025", dict(kind="uniform", lam=0.25, selector="b1",
                             keep_fn=keep_fn_fac))]
    for a in arms:
        a[1]["gen_tokens"] = args.gen_tokens

    bank = AU.load_bank("TextVQA_VAL")
    keys = [k for k, v in bank.items() if v["n_vis"] > AU.K]
    keys = sorted(keys)[: args.n_questions * 3]
    step = max(1, len(keys) // args.n_questions)
    keys = keys[::step][: args.n_questions]
    print(f"[quick-cost] {len(keys)} questions, n_vis>256 all; "
          f"arms={[a[0] for a in arms]}", flush=True)

    dataset = C.build_dataset("TextVQA_VAL")
    eng.vlm.set_dump_image(dataset.dump_image)
    msgs = []
    for k in keys:
        row = dataset.data.iloc[int(k)]
        msgs.append(C.build_message(eng.vlm, dataset, "TextVQA_VAL", row))

    rng = random.Random(args.seed)

    def run_pair(msg):
        order = arms[:]
        rng.shuffle(order)
        recs = {}
        for name, cfg in order:
            out = measure_arm(eng, msg, "TextVQA_VAL", cfg["keep_fn"], cfg)
            assert out["timings"]["n_gen"] == args.gen_tokens, \
                f"{name}: expected {args.gen_tokens} tokens, " \
                f"got {out['timings']['n_gen']}"
            recs[name] = out["timings"] | {"peak_mb": out["peak_mb"]}
            release(out)
        return recs

    print(f"[quick-cost] warmup {WARMUP_PAIRS} pairs", flush=True)
    for w in range(WARMUP_PAIRS):
        run_pair(msgs[w % len(msgs)])

    records = []
    for b, msg in enumerate(msgs):
        for name, rec in run_pair(msg).items():
            records.append(dict(ds="TextVQA_VAL", key=keys[b], block=b,
                                arm=name, **rec))
        print(f"[quick-cost] question {b + 1}/{len(msgs)} done", flush=True)

    with open(args.out, "w") as f:
        json.dump(dict(records=records, code_commit=AU.repo_commit(),
                       gen_tokens=args.gen_tokens, n_questions=len(msgs),
                       seed=args.seed, note="diagnostic, not protocol"),
                  f, indent=1)

    print("\narm       TTFT  vision select merge prefill decode64 tok/s  VRAM")
    for name, _cfg in arms:
        sel = [r for r in records if r["arm"] == name]
        med = lambda k: statistics.median(r[k] for r in sel)  # noqa: E731
        merge_med = med("merge_ms") if any(r.get("merge_ms") for r in sel) \
            else 0.0
        tps = args.gen_tokens * 1000.0 / med("decode_ms")
        print(f"{name:9s} {med('ttft_ms'):6.1f} {med('vision_ms'):6.1f} "
              f"{med('selector_ms'):6.2f} {merge_med:5.2f} "
              f"{med('llm_prefill_ms'):7.1f} {med('decode_ms'):7.1f} "
              f"{tps:5.1f} {med('peak_mb'):6.0f}")

    print("\npaired medians (per question, arm minus FULL):")
    by = {}
    for r in records:
        by.setdefault((r["block"], r["key"]), {})[r["arm"]] = r
    for name in ("FAC256", "MAIN025"):
        d_ttft = [b[name]["ttft_ms"] - b["FULL"]["ttft_ms"] for b in by.values()]
        d_pre = [b[name]["llm_prefill_ms"] - b["FULL"]["llm_prefill_ms"]
                 for b in by.values()]
        d_dec = [b[name]["decode_ms"] - b["FULL"]["decode_ms"]
                 for b in by.values()]
        d_tot = [b[name]["total_ms"] - b["FULL"]["total_ms"] for b in by.values()]
        print(f"{name:9s} dTTFT {statistics.median(d_ttft):+7.1f}ms  "
              f"dprefill {statistics.median(d_pre):+7.1f}ms  "
              f"ddecode64 {statistics.median(d_dec):+7.1f}ms  "
              f"dtotal {statistics.median(d_tot):+7.1f}ms")


if __name__ == "__main__":
    main()
