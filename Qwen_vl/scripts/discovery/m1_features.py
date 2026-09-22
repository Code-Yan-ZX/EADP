"""
M1 step 1: cache the L4 visual-token hidden states for the 720 extension images.

Same forward, same hook, same construction as ``s2c1_features.py`` -- the only
thing that changes is the instance list. Per instance: one prompt-only forward
with the visual embeddings injected at the image-token positions, a forward hook
on decoder layer 4 capturing the 1024 visual rows. No gradient, no query, no
layer other than L4: M1 trains the L4 arm and nothing else.

    rows  0 .. 464 of the combined space  -> s2c1_feats_L4.npy       (untouched)
    rows 465 .. 1184                      -> m1_feats_L4_extra.npy   (written here)

Before writing anything the generator is checked against the published cache on a
sample of already-cached instances (prereg §3.3, gate G1). A generator that does
not reproduce the existing rows cannot be trusted to produce the new ones, so the
check runs first and a failure exits non-zero.

Usage
    python m1_features.py                 # check then generate (~3 min)
    python m1_features.py --check-only    # gate G1 only
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                                         # noqa: E402
from common import eadp_model_name                                    # noqa: E402
from s1_audit import OUT                                              # noqa: E402
from s2c1_features import PrefixCapture, load_instance_tensors        # noqa: E402
from m1_common import (DS_ORDER, EXTRA_FEATS, FEATS_ORIG, N_ORIG,     # noqa: E402
                       N_VIS, M1_PLAN, dump, load_m1_plan)

D_MODEL = 4096
LAYER = 4
CHECK_KEYS = 8          # already-cached instances the generator must reproduce


def dataset_objects():
    bank = {}
    for ds in DS_ORDER:
        bank[ds] = common.build_dataset(ds)
        print(f"[dataset] {ds}: {len(bank[ds].data)} rows")
    return bank


def forward_rows(model, datasets, ds, idx, layers, probes, image_token_id,
                 timings=None):
    """One prompt-only forward; returns the L4 visual slice (1024, 4096) fp16."""
    row = datasets[ds].data.iloc[idx]
    model.set_dump_image(datasets[ds].dump_image)
    msg = common.build_message(model, datasets[ds], ds, row)
    inputs = model._processor_inputs(model._build_messages(msg, dataset=ds))
    prompt, mask, s, e = load_instance_tensors(model, inputs, image_token_id)
    assert e - s == N_VIS, f"{ds}_{idx}: {e - s} visual tokens, expected {N_VIS}"
    for L in layers:
        probes[L].visual_slice = (s, e)
        probes[L].h = None
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
    t0.record()
    model.model.model(inputs_embeds=prompt, attention_mask=mask,
                      use_cache=False, return_dict=True)
    t1.record()
    torch.cuda.synchronize()
    if timings is not None:
        timings.append(t0.elapsed_time(t1))
    h = probes[LAYER].h
    if h is None:
        raise RuntimeError(f"layer {LAYER} hook never fired for {ds}_{idx}")
    return h, int(prompt.shape[1]), int(s), float(t0.elapsed_time(t1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--n-check", type=int, default=CHECK_KEYS)
    args = ap.parse_args()

    _, plan, keys, rows_of = load_m1_plan()
    ext = json.load(open(os.path.join(OUT, M1_PLAN)))["instances"]
    print(f"[plan] {len(keys)} combined rows; {len(ext)} extension instances")

    datasets = dataset_objects()
    model = common.load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=32)
    model.model.eval()
    torch.set_grad_enabled(False)
    image_token_id = model.model.config.image_token_id
    layers = model.model.model.language_model.layers

    probes = {LAYER: PrefixCapture(LAYER)}
    probes[LAYER].attach(layers[LAYER])

    # ---------------------------------------------------------------- gate G1 --
    # Re-derive already-cached rows through this generator. The forward is not
    # run under deterministic kernels in S2-C1 either, so the comparison is
    # reported as a tolerance rather than asserted to be exactly zero; what it
    # has to exclude is a *different* computation, and that shows up as a
    # mismatch orders of magnitude above the fp16 resolution.
    H0 = np.load(os.path.join(OUT, FEATS_ORIG), mmap_mode="r")
    rng = np.random.default_rng(20260924)
    check_rows = sorted(rng.choice(N_ORIG, size=args.n_check, replace=False))
    report = {"n_checked": len(check_rows), "rows": []}
    worst = 0.0
    for r in check_rows:
        k = keys[r]
        ds, idx = k.rsplit("_", 1)
        h, _, _, ms = forward_rows(model, datasets, ds, int(idx), [LAYER], probes,
                                   image_token_id)
        ref = np.asarray(H0[r], dtype=np.float32)
        cur = h.astype(np.float32)
        d = np.abs(cur - ref)
        worst = max(worst, float(d.max()))
        report["rows"].append(dict(key=k, max_abs_diff=float(d.max()),
                                   mean_abs_diff=float(d.mean()),
                                   frac_exact=float((cur == ref).mean()),
                                   ref_abs_mean=float(np.abs(ref).mean())))
        print(f"[gate G1] {k}: max|d|={d.max():.3e} mean|d|={d.mean():.3e} "
              f"exact={100.0*(cur == ref).mean():.2f}%")
    report["worst_max_abs_diff"] = worst
    report["ref_abs_mean"] = float(np.abs(H0[check_rows]).mean())
    report["frac_exact_min"] = float(min(r["frac_exact"] for r in report["rows"]))
    # The published cache is fp16, so a recomputation can differ in the last
    # stored bit if the reduction order moved. What the gate has to exclude is a
    # *different computation*: a wrong hook, a wrong layer, a wrong token slice.
    # That shows up at the scale of the data, not at the scale of an fp16 ulp.
    # Threshold: worst-case element error below 1 % of the mean |h|, which is
    # ~20x the fp16 quantisation step at that magnitude.
    report["threshold"] = 0.01 * report["ref_abs_mean"]
    report["passed"] = bool(worst < report["threshold"])
    dump("m1_feature_repro.json", report)
    if not report["passed"]:
        raise SystemExit(f"gate G1b FAILED: worst max|d| = {worst:.3e} "
                         f">= {report['threshold']:.3e}")
    print(f"[gate G1] PASS  worst max|d| = {worst:.3e} "
          f"(data scale {report['ref_abs_mean']:.3f})")
    if args.check_only:
        return

    # ------------------------------------------------------------- generate ---
    H = np.lib.format.open_memmap(os.path.join(OUT, EXTRA_FEATS), mode="w+",
                                  dtype=np.float16,
                                  shape=(len(ext), N_VIS, D_MODEL))
    meta, timings, cur_ds = [], [], None
    t_start = time.time()
    for j, p in enumerate(ext):
        ds = p["ds"]
        if ds != cur_ds:
            model.set_dump_image(datasets[ds].dump_image)
            cur_ds = ds
        h, seq_len, vis_start, ms = forward_rows(model, datasets, ds, p["idx"],
                                                 [LAYER], probes, image_token_id,
                                                 timings)
        H[j] = h
        meta.append(dict(key=p["key"], ds=ds, idx=int(p["idx"]),
                         extra_rank=int(p["extra_rank"]), seq_len=seq_len,
                         vis_start=vis_start, fwd_ms=ms))
        if (j + 1) % 60 == 0 or j == len(ext) - 1:
            print(f"  {j+1}/{len(ext)}  mean fwd {np.mean(timings):.0f} ms  "
                  f"({time.time() - t_start:.0f} s)", flush=True)
    H.flush()
    probes[LAYER].detach()
    dump("m1_features_extra.json",
         dict(layer=LAYER, d_model=D_MODEL, n_vis=N_VIS, instances=meta,
              fwd_ms_mean=float(np.mean(timings)),
              fwd_ms_median=float(np.median(timings)),
              wall_seconds=float(time.time() - t_start),
              source="s2c1_features.py forward, extension instance list",
              plan_file=M1_PLAN))
    print(f"[saved] {EXTRA_FEATS} ({len(ext)}, {N_VIS}, {D_MODEL}) fp16, "
          f"mean fwd {np.mean(timings):.0f} ms, {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
