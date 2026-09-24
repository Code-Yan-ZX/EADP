"""
M2 — implementation-defect amendment: the paired / interleaved performance
benchmark (prereg §6.3 timing discipline, amendment §6 protocol).

Why this exists instead of the sequential m2_perf grid: the first perf pass ran
each arm in one long block, minutes apart, so (a) any host-state drift between
blocks masquerades as an arm effect, and (b) the selector window for the pre-LLM
arms was double-counted (internal pruner timer + an outer CUDA bracket on the
same call), which is why that table's "model prefill median" exceeded its own
TTFT median for B1. This harness measures every arm inside every block, in a
randomised order, so contrasts are PAIRED within blocks and the timing windows
are defined once:

    TTFT (end-to-end)   sync -> [image preprocess (CPU) -> vision tower ->
                        scoring/selection -> LLM prefill -> first-token logits]
                        -> sync. Wall clock. This is time-to-first-token.
    model-only prefill  TTFT minus the image-preprocess window, i.e. the part
                        of the request that token pruning can act on at all
                        (vision tower + prune machinery + LLM layers). Reported
                        as the event-sum of the non-preprocess stages.
    selector            the arm's selection machinery alone: for B1/B2 the
                        pruner-internal windows (scoring stages and the
                        facility/block loop, each counted exactly once); for
                        the gdep arms the L0-L4 prefix, the online scorer,
                        the selection and the compaction, each event-bracketed
                        once.
    decode              fixed 32 steps with ignore_eos=True, CUDA-event timed,
                        same window for every arm (prereg §6.3).

The stage sum and the wall TTFT are saved per block so component accounting is
checkable post hoc (residual = launch overhead + norm + lm_head + sync jitter).

Usage
    python scripts/discovery/m2_perf_paired.py --blocks 120 --warmup-blocks 15
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                     # noqa: E402
from common import OUTPUT_DIR, eadp_model_name                    # noqa: E402
from m1_common import load_m1_plan                                # noqa: E402
from m2_gdep import (MODE_FULL, MODE_GDEP, MODE_PRELLM,            # noqa: E402
                     POLICY_PRESERVE, POLICY_RENUMBER, ENGINE_VERSION,
                     GDEPConfig, GDEPEngine, N_VIS, dump_json)

# Arm set fixed by the amendment (§6): the three baselines plus the corrected
# candidate and the one valid pre-fix candidate. C0/C2/C3 stay out of the
# first corrected phase.
ARMS = [
    dict(tag="B0", mode=MODE_FULL, n_arm=None, seed=None, selector=None,
         policy=None, label="Full model, unpruned"),
    dict(tag="B1", mode=MODE_PRELLM, n_arm=None, seed=None, selector="facility",
         policy=None, label="Official EADP facility@256 (pre-LLM)"),
    dict(tag="B2", mode=MODE_PRELLM, n_arm=None, seed=None, selector="block8",
         policy=None, label="EADP block8@256 (pre-LLM)"),
    dict(tag="C1-P", mode=MODE_GDEP, n_arm=960, seed=2, selector="topk",
         policy=POLICY_PRESERVE,
         label="GDEP n=960 topk@256, corrected PRESERVE (v2)"),
    dict(tag="C1-R", mode=MODE_GDEP, n_arm=960, seed=2, selector="topk",
         policy=POLICY_RENUMBER, label="GDEP n=960 topk@256, RENUMBER"),
]

# M3 arms.  Kept OUT of the default ARMS on purpose: the amendment's record is
# five arms over 120 blocks, and silently changing that set would change what
# `m2_perf_paired.json` means if the M2 benchmark were ever re-run.  Select
# these with `--tags MG OR`.
M3_ARMS = [
    # The pre-registered primary MissGuard arm, measured against B2 in the same
    # block: the within-block TTFT contrast MG - B2 is the only number that
    # includes every host-side cost of the audit.
    dict(tag="MG", mode=MODE_PRELLM, n_arm=None, seed=None, selector="block8",
         policy=None, student="m3_miss",
         miss=dict(source="learned", r=16, rule="lowimp"),
         label="MissGuard learned r=16 lowimp (pre-LLM)"),
    dict(tag="OR", mode=MODE_PRELLM, n_arm=None, seed=None, selector="block8",
         policy=None, miss=dict(source="teacher", r=16, rule="lowimp"),
         label="Oracle-Miss r=16 lowimp (teacher rescue)"),
]

STAGE_KEYS = ["image_preprocess_ms", "vision_encoder_ms", "eadp_scoring_ms",
              "selector_ms", "L0_L4_ms", "scorer_ms", "token_compaction_ms",
              "miss_ms", "llm_forward_ms"]
MODEL_ONLY_KEYS = [k for k in STAGE_KEYS if k != "image_preprocess_ms"]


def build_engine(model, arm):
    cfg = GDEPConfig(mode=arm["mode"], budget=256, selector=arm["selector"] or "topk",
                     pos_policy=arm["policy"] or POLICY_PRESERVE,
                     n_arm=arm["n_arm"] or 240, seed=arm["seed"] or 0,
                     tag=arm["tag"])
    eng = GDEPEngine.from_checkpoint(cfg, model=model, max_new_tokens=32)
    if arm.get("miss") is not None:                     # M3: audit-and-correct
        from m3_common import install_missguard
        miss = dict(arm["miss"])
        if miss.get("source") == "learned":
            import numpy as np
            ck = torch.load(os.path.join(OUTPUT_DIR, f"{arm['student']}.pt"),
                            map_location="cpu", weights_only=False)
            from m3_common import MissStudent, feature_index
            st = MissStudent(ck["d_hand"], ck["d_vis"])
            st.load_state_dict(ck["state_dict"])
            st.eval().to(next(model.model.parameters()).device)
            miss.update(student=st,
                        mu_hand=torch.from_numpy(np.asarray(ck["mu_hand"], np.float32)).to(st.proj_v.weight.device),
                        sd_hand=torch.from_numpy(np.asarray(ck["sd_hand"], np.float32)).to(st.proj_v.weight.device),
                        mu_vis=torch.from_numpy(np.asarray(ck["mu_vis"], np.float32)).to(st.proj_v.weight.device),
                        sd_vis=torch.from_numpy(np.asarray(ck["sd_vis"], np.float32)).to(st.proj_v.weight.device),
                        feature_idx=feature_index(ck["features"]).to(st.proj_v.weight.device),
                        per_instance_z=bool(ck["per_instance_z"]))
        install_missguard(eng, model, miss)
    return eng


def timed_request(eng, msg, ds_name, decode_tokens):
    """One full request cycle: returns (ttft_ms, stages, decode_ms, layer_calls)."""
    t = {}
    torch.cuda.synchronize()
    w0 = time.perf_counter()
    prep = eng.prepare(msg, ds_name, t)
    st, info = eng.prefill(prep, timings=t)
    torch.cuda.synchronize()
    w1 = time.perf_counter()                      # first token is in `st`
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    ids = eng.decode(st, decode_tokens, ignore_eos=True)
    e1.record()
    torch.cuda.synchronize()
    dec_ms = e0.elapsed_time(e1)
    layer_calls = int(eng.counts.get("layer_calls", 0))
    steps = int(eng.counts.get("decode_steps", 0))
    del st, prep
    return dict(ttft_ms=(w1 - w0) * 1e3,
                stages={k: float(t.get(k, 0.0)) for k in STAGE_KEYS},
                stage_sum_ms=float(sum(t.get(k, 0.0) for k in STAGE_KEYS)),
                model_only_ms=float(sum(t.get(k, 0.0) for k in MODEL_ONLY_KEYS)),
                decode_ms=float(dec_ms), n_decode=len(ids),
                layer_calls=layer_calls, decode_steps=steps,
                n_kept=info["n_kept"], context_len=info["context_len"],
                cfg_hash=info["cfg_hash"],
                no_recompute=bool(layer_calls == 36 * (1 + steps)))


def stats_of(xs):
    xs = sorted(float(x) for x in xs)
    n = len(xs)
    return dict(mean=float(np.mean(xs)), median=float(np.median(xs)),
                p10=float(xs[max(0, int(round(0.1 * (n - 1))))]),
                p90=float(xs[min(n - 1, int(round(0.9 * (n - 1))))]),
                std=float(np.std(xs, ddof=1) if n > 1 else 0.0),
                min=xs[0], max=xs[-1], n=n)


def paired_contrast(a_by_block, b_by_block, blocks, rng, nb=10000):
    """Within-block difference (a - b) with a block-bootstrap CI."""
    d = np.array([a_by_block[k] - b_by_block[k] for k in blocks])
    draws = np.array([rng.choice(d, size=len(d), replace=True).mean()
                      for _ in range(nb)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(median_diff_ms=float(np.median(d)), mean_diff_ms=float(d.mean()),
                ci_mean=[float(lo), float(hi)], n_blocks=int(len(d)),
                wins_a=int((d < 0).sum()), wins_b=int((d > 0).sum()))


def stage(model, args):
    _, plan, keys, rows_of = load_m1_plan()
    test_rows = list(rows_of["test"])
    key0 = keys[test_rows[0]]
    ds_name, idx = key0.rsplit("_", 1)
    dataset = common.build_dataset(ds_name)
    model.set_dump_image(dataset.dump_image)
    msg = common.build_message(model, dataset, ds_name, dataset.data.iloc[int(idx)])
    print(f"[paired] fixed input {key0}; arms {[a['tag'] for a in ARMS]}; "
          f"{args.blocks} blocks x {len(ARMS)} arms; warmup {args.warmup_blocks}")

    torch.set_grad_enabled(False)
    engines = {}
    for arm in ARMS:
        engines[arm["tag"]] = build_engine(model, arm)
        m = arm.get("miss") or {}
        if m.get("source") == "teacher":
            from m3_common import load_teacher
            engines[arm["tag"]].pruner.miss["teacher"] = load_teacher()[key0]

    rng = random.Random(args.seed)
    boot = np.random.default_rng(args.seed + 1)
    blocks = []                                # list of {tag: record}
    orders = []
    t_start = time.time()
    for b in range(args.warmup_blocks + args.blocks):
        tags = [a["tag"] for a in ARMS]
        rng.shuffle(tags)                     # randomised within-block order
        rec = {}
        for pos, tag in enumerate(tags):
            try:
                rec[tag] = timed_request(engines[tag], msg, ds_name,
                                         args.decode_tokens)
                rec[tag]["order_pos"] = pos
            except Exception:
                traceback.print_exc()
                rec[tag] = dict(error=traceback.format_exc()[-500:])
        torch.cuda.empty_cache()
        if b >= args.warmup_blocks:
            blocks.append(rec)
            orders.append(tags)
        if (b + 1) % 10 == 0:
            el = time.time() - t_start
            print(f"  block {b + 1}/{args.warmup_blocks + args.blocks} "
                  f"({el:.0f}s, {(b + 1) / el:.2f} blocks/s)", flush=True)

    tags = [a["tag"] for a in ARMS]
    ids = [i for i, blk in enumerate(blocks)
           if all("error" not in blk.get(t, {}) for t in tags)]
    print(f"[paired] {len(ids)}/{len(blocks)} complete blocks retained")
    per_arm = {t: dict(ttft=[blocks[i][t]["ttft_ms"] for i in ids],
                       decode=[blocks[i][t]["decode_ms"] for i in ids],
                       model_only=[blocks[i][t]["model_only_ms"] for i in ids],
                       stage_sum=[blocks[i][t]["stage_sum_ms"] for i in ids],
                       stages={k: [blocks[i][t]["stages"][k] for i in ids]
                               for k in STAGE_KEYS},
                       no_recompute=all(blocks[i][t]["no_recompute"] for i in ids),
                       cfg_hash=blocks[ids[0]][t]["cfg_hash"],
                       n_kept=blocks[ids[0]][t]["n_kept"])
                for t in tags if all(t in blocks[i] for i in ids)}

    out = dict(stage="M2 paired perf", engine_version=ENGINE_VERSION,
               config=vars(args), input_key=key0, batch_size=1,
               decode_tokens=args.decode_tokens,
               n_blocks=len(blocks), n_blocks_retained=len(ids),
               blocks_raw=blocks, orders=orders,
               summary={}, contrasts={}, accounting={}, drift={})
    for t, d in per_arm.items():
        out["summary"][t] = dict(
            ttft=stats_of(d["ttft"]), decode32=stats_of(d["decode"]),
            model_only=stats_of(d["model_only"]),
            stage_sum=stats_of(d["stage_sum"]),
            stages={k: stats_of(v) for k, v in d["stages"].items() if any(v)},
            no_recompute=d["no_recompute"], cfg_hash=d["cfg_hash"],
            n_kept=d["n_kept"])
        half = len(d["ttft"]) // 2
        out["drift"][t] = dict(first_half_median=float(np.median(d["ttft"][:half])),
                               second_half_median=float(np.median(d["ttft"][half:])),
                               note="same-arm drift across the run; the old grid's "
                                    "load-drift reading is tested against this")
        # component accounting: wall TTFT vs the sum of its stage windows
        resid = [b_ - s_ for b_, s_ in zip(d["ttft"], d["stage_sum"])]
        out["accounting"][t] = dict(
            ttft_minus_stage_sum_median_ms=float(np.median(resid)),
            note="positive residual = un-bracketed glue (launch overhead, norm, "
                 "lm_head, sync); the OLD table's negative-appearing gap was the "
                 "prellm selector double count, removed in v2")

    for a in tags:
        for b_ in ("B1", "B2", "B0"):
            if a == b_ or a not in per_arm or b_ not in per_arm:
                continue
            key = f"{a}_minus_{b_}"
            out["contrasts"][key] = dict(
                ttft=paired_contrast(
                    {i: blocks[i][a]["ttft_ms"] for i in ids},
                    {i: blocks[i][b_]["ttft_ms"] for i in ids}, ids, boot),
                decode32=paired_contrast(
                    {i: blocks[i][a]["decode_ms"] for i in ids},
                    {i: blocks[i][b_]["decode_ms"] for i in ids}, ids, boot))
    dump_json(args.tag + ".json", out)

    print(f"\n{'arm':6s} {'TTFT med':>9s} {'p10-p90':>14s} {'model-only':>10s} "
          f"{'selector':>9s} {'decode32':>9s} {'no-recomp':>9s}")
    for t in tags:
        s = out["summary"][t]
        med = lambda k: s["stages"].get(k, {"median": 0.0})["median"]
        sel = med("selector_ms") + med("L0_L4_ms") + med("scorer_ms") + med(
            "eadp_scoring_ms")
        print(f"{t:6s} {s['ttft']['median']:9.1f} "
              f"[{s['ttft']['p10']:.1f},{s['ttft']['p90']:.1f}] "
              f"{s['model_only']['median']:10.1f} {sel:9.1f} "
              f"{s['decode32']['median']:9.1f} {str(s['no_recompute']):>9s}")
    for k, v in out["contrasts"].items():
        c = v["ttft"]
        print(f"  {k:16s} TTFT paired median {c['median_diff_ms']:+7.2f} ms "
              f"(mean CI [{c['ci_mean'][0]:+.2f}, {c['ci_mean'][1]:+.2f}], "
              f"blocks {c['wins_a']}:{c['wins_b']} for B)  "
              f"decode {v['decode32']['median_diff_ms']:+.2f} ms")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=120)
    ap.add_argument("--warmup-blocks", type=int, default=15)
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--tag", default="m2_perf_paired")
    ap.add_argument("--tags", nargs="+", default=None,
                    help="restrict the arm set by tag (default: the whole ARMS)")
    args = ap.parse_args()
    if args.tags:
        global ARMS
        pool = ARMS + M3_ARMS
        ARMS = [a for a in pool if a["tag"] in args.tags]
        missing = set(args.tags) - {a["tag"] for a in pool}
        if missing:
            raise SystemExit(f"unknown arm tag(s): {sorted(missing)}")
    model = common.load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=32)
    model.model.eval()
    stage(model, args)


if __name__ == "__main__":
    main()
