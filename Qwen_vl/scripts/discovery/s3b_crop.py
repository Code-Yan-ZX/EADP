"""
S3-B step 4 (GPU): Part C, the crop probe -- is a bundle a *legible evidence
unit*?

For each rescue bundle the union bounding box of its cells is cut out of the
1024x1024 frame the model actually sees, re-centred to a square (the wrapper's
own expand2square), resized back to 1024x1024, and the SAME question is asked
of that crop with nothing pruned (all 1024 merged tokens of the crop). If the
bundle really is a piece of evidence, the crop should answer; if it is just a
handful of high-saliency pixels, it should not.

Three families of arms, matched for ZOOM so that only LOCATION differs:

    whole          the unpruned full frame (calibration: should reproduce the
                   unpruned accuracy)
    bundle         bbox of G = T_only[:k*]
    randwin{r}     bbox of the r-th randwin null (same k*, same teacher-rank
                   window, different members)                        x4
    matched{r}     a square of the same h x w as the bundle bbox, placed at a
                   random position inside the frame                  x4

Everything is scored with the official metric and the gold-answer NLL of the
crop's own forward.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                                    # noqa: E402
from s1_audit import OUT                                         # noqa: E402
import s3b_common as B                                           # noqa: E402
import s3b_gpu as G                                              # noqa: E402
from scoring import per_sample_hits                              # noqa: E402

N_NULL = 4
N_MATCHED = 4
SIDE = 1024


# ---------------------------------------------------------------------------
def prep_nocache(h, message, ds):
    """h.prep minus the disk cache: a crop's vision embeddings are used once,
    and caching 9 of them per bundle would write ~700 MB per 10 bundles."""
    from vlmeval.vlm.qwen3_vl.model_fixed_res import unwrap_visual_output
    messages = h.vlm._build_messages(message, dataset=ds)
    inputs = h.vlm._processor_inputs(messages)
    ids = inputs["input_ids"]
    pos = (ids[0] == h.image_token_id).nonzero(as_tuple=True)[0]
    s, e = int(pos[0]), int(pos[-1]) + 1
    assert e - s == B.N_VIS, f"{e - s} visual tokens"
    with torch.no_grad():
        pv = inputs["pixel_values"].type(h.model.visual.dtype)
        vis = unwrap_visual_output(
            h.model.visual(pv, grid_thw=inputs["image_grid_thw"])).detach()
        emb = h.emb(ids)
    return dict(vis=vis, prompt_prefix=emb[0, :s], suffix=emb[0, e:],
                n_prefix=s)


def canonical_frame(datasets, rec):
    """the 1024x1024 frame the vision tower receives (the wrapper's own
    scale-longest-side + expand2square + resize)."""
    return B.canonical_frame(datasets[rec["ds"]], rec["idx"], side=SIDE)


def bbox_px(tokens):
    r0, r1, c0, c1 = B.bbox(tokens)
    cell = SIDE // B.GRID_HW
    return r0 * cell, (r1 + 1) * cell, c0 * cell, (c1 + 1) * cell


def _square_up(sub):
    from vlmeval.vlm.qwen3_vl.model_fixed_res import expand2square
    return expand2square(sub, (125, 125, 125)).resize((SIDE, SIDE))


def crop_of(frame, tokens):
    r0, r1, c0, c1 = bbox_px(tokens)
    return _square_up(frame.crop((c0, r0, c1, r1)))


def random_square_of_size(frame, hh, ww, rng):
    """same box size as the bundle bbox, uniformly placed so the box stays
    inside the frame (no clipping, no stretch)."""
    W, H = frame.size
    ww, hh = min(ww, W), min(hh, H)
    c0 = int(rng.integers(0, W - ww + 1))
    r0 = int(rng.integers(0, H - hh + 1))
    return _square_up(frame.crop((c0, r0, c0 + ww, r0 + hh)))


def swap_image(msg, path):
    if isinstance(msg, dict):
        m = dict(msg)
        m["image"] = path
        return m
    out = []
    for it in msg:
        if isinstance(it, dict) and it.get("type") == "image":
            it = dict(it)
            it["value"] = path
        out.append(it)
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s3b_crop")
    ap.add_argument("--banks", default="L,G")
    args = ap.parse_args()

    cases = json.load(open(os.path.join(OUT, B.CASES_JSON)))
    ablate = json.load(open(os.path.join(OUT, "s3b_ablate.json")))
    recs = {b: {r["key"]: r for r in cases["banks"][b]} for b in ("G", "L")}
    want = set(args.banks.split(","))
    todo = sorted({(v["bank"], v["key"]) for v in ablate["runs"].values()
                   if v["bank"] in want})
    print(f"[crop] {len(todo)} bundles")

    cropdir = os.path.join(OUT, "s3b_crops")
    os.makedirs(cropdir, exist_ok=True)
    out_path = os.path.join(OUT, f"{args.tag}.json")
    results = dict(config=dict(n_null=N_NULL, n_matched=N_MATCHED, side=SIDE,
                               gen_cap=G.GEN_CAP), done=[], runs={})
    if os.path.exists(out_path):
        prev = json.load(open(out_path))
        if prev["config"] != results["config"]:
            raise RuntimeError("existing s3b_crop.json under a different config")
        results["done"], results["runs"] = prev["done"], prev["runs"]
        print(f"[resume] {len(results['done'])} done")
    todo = [t for t in todo if f"{t[0]}|{t[1]}" not in results["done"]]

    datasets = {ds: common.build_dataset(ds) for ds in B.DS_ALL}
    model, h = G.load_stack()
    t0 = time.time()
    for bank, key in todo:
        rec = recs[bank][key]
        ent_ab = ablate["runs"][f"{bank}|{key}"]
        kstar = ent_ab["kstar"]
        try:
            frame = canonical_frame(datasets, rec)
            ds_obj = datasets[rec["ds"]]
            model.set_dump_image(ds_obj.dump_image)
            row = ds_obj.data.iloc[rec["idx"]]
            msg0 = common.build_message(model, ds_obj, rec["ds"], row)
            ans = h.answer_ids(B.A.golds_of(row))
            ent = dict(bank=bank, key=key, ds=rec["ds"], kstar=kstar, arms={})

            def score(name, img, tokens, box_hw=None):
                path = os.path.join(cropdir, f"{bank}_{key}_{name}.png")
                img.save(path)
                prep = prep_nocache(h, swap_image(msg0, path), rec["ds"])
                keep = list(range(B.N_VIS))                 # nothing pruned
                L = h.nll(prep, keep, ans)[0]
                ids, text = h.generate(prep, keep, max_new=G.GEN_CAP)
                pred = h.vlm._post_process_response(text)
                hit = float(per_sample_hits(rec["ds"], [row], [pred])[0])
                if box_hw is not None:          # matched control: own box size
                    area = float(box_hw[0] * box_hw[1])
                elif tokens:
                    r0, r1, c0, c1 = bbox_px(tokens)
                    area = float((r1 - r0) * (c1 - c0))
                else:
                    area = float(SIDE * SIDE)
                ent["arms"][name] = dict(hit=hit, L=float(L),
                                         pred=pred[:200], bbox_area_px=area,
                                         n_tok=len(tokens) if tokens else B.N_VIS)
                print(f"   {name:14s} hit={hit:.2f} L={L:.3f} "
                      f"area={area:8.0f} pred={pred[:34]!r}")

            adds0, _ = B.prefix_arms(rec, kstar)
            score("whole", frame, None)
            score("bundle", crop_of(frame, adds0), adds0)
            for r in range(N_NULL):
                nm = f"randwin{r}"
                if nm in ent_ab["arms"]:
                    toks = ent_ab["arms"][nm]["adds"]
                    score(nm, crop_of(frame, toks), toks)
            r0, r1, c0, c1 = bbox_px(adds0)
            rng = B._rng(rec, "cropmatched")
            for r in range(N_MATCHED):
                score(f"matched{r}", random_square_of_size(frame, r1 - r0,
                                                           c1 - c0, rng), None)
            results["runs"][f"{bank}|{key}"] = ent
            results["done"].append(f"{bank}|{key}")
        except Exception:
            traceback.print_exc()
            print(f"[skip] {bank}|{key}")
            continue
        tmp = out_path + ".tmp"
        json.dump(results, open(tmp, "w"))
        os.replace(tmp, out_path)
        print(f"[{len(results['done'])}/{len(todo)}] {bank}|{key} "
              f"{time.time() - t0:.0f}s")
    G.require_complete(results, len(todo), os.path.basename(out_path))
    print(f"[saved] {out_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
