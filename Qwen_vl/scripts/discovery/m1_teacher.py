"""
M1 step 2: compute the S2-B P1-G2 gradient teacher map for the 720 extension rows.

The objective is S2-B's, verbatim -- no answer, no ground truth, no continuation:

    k        = argmax(logits at the first answer position)
    J        = logits[k]
    score_i  = sum_d | dJ/dv_{i,d} * v_{i,d} |          (the G2 map, 1024 values)

One prompt forward plus one backward per instance. The existing 450 maps in
``s2b_gradient_scores.npz`` are read-only; the extension maps are written to
``m1_gradient_scores_extra.npz`` and concatenated at load time by key.

The published generator also ran the official EADP pruner on the same features to
record the facility-location selection. M1 does not need that selection -- it
trains the L4 arm and nothing else -- so the pruner is not attached here. That
makes this a *reduced* code path, which is exactly why gate G2 exists: eight
already-cached instances are re-scored through this path first, and the maps have
to match the published ones. If the pruner perturbed the forward, G2 fails.

Usage
    python m1_teacher.py                 # check then generate (~7 min)
    python m1_teacher.py --check-only    # gate G2 only
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
from m1_common import (DS_ORDER, EXTRA_TEACHER, EXTRA_TEACHER_META,   # noqa: E402
                       M1_PLAN, N_ORIG, TEACHER, dump, load_m1_plan)

CHECK_KEYS = 8
REL_TOL = 1e-4          # prereg 3.3 gate G2


def dataset_objects():
    return {ds: common.build_dataset(ds) for ds in DS_ORDER}


def p1g2(model, datasets, ds, idx, emb_layer, image_token_id, dev):
    """One forward + one backward; returns (g2, top1_token_id, ms, peak_mb)."""
    from vlmeval.vlm.qwen3_vl.model_fixed_res import unwrap_visual_output

    row = datasets[ds].data.iloc[idx]
    model.set_dump_image(datasets[ds].dump_image)
    msg = common.build_message(model, datasets[ds], ds, row)
    inputs = model._processor_inputs(model._build_messages(msg, dataset=ds))
    ids = inputs["input_ids"]

    with torch.no_grad():
        pv = inputs["pixel_values"].type(model.model.visual.dtype)
        vis0 = unwrap_visual_output(
            model.model.visual(pv, grid_thw=inputs["image_grid_thw"]))
        emb = emb_layer(ids)
        pos = (ids[0] == image_token_id).nonzero(as_tuple=True)[0]
        s, e = int(pos[0]), int(pos[-1]) + 1
        prefix, suffix = emb[:, :s], emb[:, e:]

    vt = vis0.detach().clone().requires_grad_(True)
    prompt = torch.cat([prefix, vt.unsqueeze(0), suffix], dim=1)
    L = prompt.shape[1]
    torch.cuda.reset_peak_memory_stats()
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
    t0.record()
    out = model.model(inputs_embeds=prompt,
                      attention_mask=torch.ones(1, L, dtype=torch.long, device=dev),
                      use_cache=False, return_dict=True)
    last = out.logits[0, -1].float()
    k = int(torch.argmax(last))
    J = last[k]
    t1.record()
    t0b, t1b = torch.cuda.Event(True), torch.cuda.Event(True)
    t0b.record()
    J.backward()
    t1b.record()
    torch.cuda.synchronize()

    g = vt.grad.detach()
    g2 = (g * vt.detach()).abs().sum(dim=-1).float().cpu().numpy()
    ms = dict(fwd_ms=t0.elapsed_time(t1), bwd_ms=t0b.elapsed_time(t1b),
              peak_mb=torch.cuda.max_memory_allocated() / 2**20,
              top1_token_id=k,
              top1_token=str(model.processor.tokenizer.decode([k])))
    del out, prompt, last, J, vt, g, vis0, emb, prefix, suffix, inputs
    torch.cuda.empty_cache()
    return g2, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--n-check", type=int, default=CHECK_KEYS)
    ap.add_argument("--repeats-g2", type=int, default=2,
                    help="re-runs per checked instance, to measure the backward's "
                         "own reproducibility floor alongside the comparison")
    args = ap.parse_args()

    _, plan, keys, rows_of = load_m1_plan()
    ext = json.load(open(os.path.join(OUT, M1_PLAN)))["instances"]
    print(f"[plan] {len(keys)} combined rows; {len(ext)} extension instances")

    datasets = dataset_objects()
    model = common.load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=2048)
    model.model.eval()
    # vlmeval.config disables autograd globally; the P1-G2 objective needs it and
    # only the injected visual embeddings are differentiable.
    torch.set_grad_enabled(True)
    assert torch.is_grad_enabled()
    for p in model.model.parameters():
        p.requires_grad_(False)
    image_token_id = model.model.config.image_token_id
    dev = next(model.model.parameters()).device
    emb_layer = model.model.get_input_embeddings()

    # ---------------------------------------------------------------- gate G2 --
    # AMENDED (see docs/scoring_search_m1.md, amendment 1). The pre-registration
    # fixed a relative tolerance of 1e-4 before the quantity's reproducibility
    # was measured. The P1-G2 score is a *backward* pass: it does not reproduce
    # even itself to 1e-4. So the gate is now relative to the code path's own
    # floor, measured in the same run:
    #
    #   (a) worst |published - recomputed| <= 2 x worst |recomputed - recomputed|
    #   (b) the Top-32 change against the published map <= the Top-32 change the
    #       identical code path makes against itself
    #
    # (b) is the one that matters: HEAD_RANK supervises exactly the teacher's
    # Top-32 and their rank order, so the criterion is whether *the target* moves
    # more than the generator's own noise moves it.
    G = np.load(os.path.join(OUT, TEACHER))
    ref_meta = json.load(open(os.path.join(OUT, "s2b_gradient_scores_meta.json")))
    ref_keys = sorted(G.files)
    rng = np.random.default_rng(20260924)
    check = [ref_keys[i] for i in sorted(rng.choice(len(ref_keys),
                                                    size=args.n_check,
                                                    replace=False))]
    report = {"n_checked": len(check), "rows": [], "repeats": args.repeats_g2,
              "prereg_tol": REL_TOL,
              "criterion": "amended: (a) published_rel <= 2*self_rel and "
                           "(b) top32_diff_vs_published <= top32_diff_self"}
    worst_pub = worst_self = 0.0
    for k in check:
        ds, idx = k.rsplit("_", 1)
        runs = [p1g2(model, datasets, ds, int(idx), emb_layer, image_token_id,
                     dev) for _ in range(args.repeats_g2)]
        maps = [r[0].astype(np.float64) for r in runs]
        ref = np.asarray(G[k], dtype=np.float64)
        den = float(np.abs(ref).max())
        self_abs = max(float(np.abs(maps[i] - maps[j]).max())
                       for i in range(len(maps)) for j in range(i + 1, len(maps)))
        pub_abs = max(float(np.abs(m - ref).max()) for m in maps)
        top = lambda a, b: len(set(np.argsort(-a)[:32]) ^ set(np.argsort(-b)[:32]))
        t_self = max(top(maps[i], maps[j]) for i in range(len(maps))
                     for j in range(i + 1, len(maps)))
        t_pub = max(top(m, ref) for m in maps)
        worst_pub, worst_self = max(worst_pub, pub_abs / den), \
            max(worst_self, self_abs / den)
        report["rows"].append(dict(
            key=k, ref_max=den, self_max_abs=self_abs, self_rel=self_abs / den,
            published_max_abs=pub_abs, published_rel=pub_abs / den,
            top32_diff_self=t_self, top32_diff_vs_published=t_pub,
            top1_token_id_expected=int(ref_meta[k]["top1_token_id"]),
            top1_token_id_got=[int(r[1]["top1_token_id"]) for r in runs],
            top1_match=bool(all(int(r[1]["top1_token_id"]) ==
                                int(ref_meta[k]["top1_token_id"])
                                for r in runs))))
        print(f"[gate G2] {k}: self rel={self_abs/den:.3e} "
              f"published rel={pub_abs/den:.3e} | top32 diff self={t_self} "
              f"vs published={t_pub} | top1_match="
              f"{report['rows'][-1]['top1_match']}", flush=True)
    report["worst_self_rel"] = worst_self
    report["worst_published_rel"] = worst_pub
    report["a_ok"] = bool(worst_pub <= 2.0 * worst_self + 1e-12)
    report["b_ok"] = bool(all(r["top32_diff_vs_published"] <= r["top32_diff_self"]
                              for r in report["rows"]))
    report["passed"] = bool(report["a_ok"] and report["b_ok"])
    dump("m1_teacher_repro.json", report)
    if not report["passed"]:
        raise SystemExit(
            f"gate G2 FAILED: worst self rel={worst_self:.3e}, worst published "
            f"rel={worst_pub:.3e}, a_ok={report['a_ok']}, b_ok={report['b_ok']}")
    print(f"[gate G2] PASS  worst self rel={worst_self:.3e}, "
          f"worst published rel={worst_pub:.3e} (<= 2x self), "
          f"Top-32 change within the generator's own noise")
    if args.check_only:
        return

    # ------------------------------------------------------------- generate ---
    scores, meta, timings = {}, {}, []
    t_start = time.time()
    for j, p in enumerate(ext):
        g2, ms = p1g2(model, datasets, p["ds"], p["idx"], emb_layer,
                      image_token_id, dev)
        scores[p["key"]] = g2
        meta[p["key"]] = dict(dataset=p["ds"], index=int(p["idx"]),
                              extra_rank=int(p["extra_rank"]),
                              n_img_tokens=1024, g2_sum=float(g2.sum()),
                              g2_max=float(g2.max()), **ms)
        timings.append((ms["fwd_ms"], ms["bwd_ms"], ms["peak_mb"]))
        if (j + 1) % 40 == 0 or j == len(ext) - 1:
            print(f"  {j+1}/{len(ext)}  fwd={np.mean([t[0] for t in timings]):.0f}ms "
                  f"bwd={np.mean([t[1] for t in timings]):.0f}ms  "
                  f"({time.time() - t_start:.0f} s)", flush=True)

    assert len(scores) == len(ext)
    np.savez_compressed(os.path.join(OUT, EXTRA_TEACHER), **scores)
    meta["__summary__"] = dict(
        n=len(scores), fwd_ms_mean=float(np.mean([t[0] for t in timings])),
        bwd_ms_mean=float(np.mean([t[1] for t in timings])),
        peak_mb_mean=float(np.mean([t[2] for t in timings])),
        wall_seconds=float(time.time() - t_start),
        objective="S2-B P1-G2: k=argmax(logits[-1]); J=logits[k]; "
                  "score_i=sum_d |dJ/dv_id * v_id|",
        source="s2b_gradient_scores.py objective, extension instance list, "
               "pruner not attached")
    json.dump(meta, open(os.path.join(OUT, EXTRA_TEACHER_META), "w"), indent=1)
    print(f"[saved] {EXTRA_TEACHER} ({len(scores)} maps), "
          f"fwd {np.mean([t[0] for t in timings]):.0f} ms, "
          f"bwd {np.mean([t[1] for t in timings]):.0f} ms, "
          f"{time.time() - t_start:.0f} s "
          f"({time.time() - t_start:.0f} s / 60 = "
          f"{(time.time() - t_start)/60:.1f} min)")


if __name__ == "__main__":
    main()
