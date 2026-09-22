#!/usr/bin/env python3
"""
Part 3 -- where do EADP errors actually come from?

Taxonomy (per the discovery brief):

  A  answer-bearing region never scored highly   -> importance scoring lost it
  B  region scored highly but FL dropped it      -> selector lost it
  C  region survived into the selected set, yet the answer is still wrong
  D  the unpruned model also fails               -> not a pruning failure

Stages (run separately so a crash costs one arm, not the whole study):

  select    CPU.  Read the official full-split EADP-256 predictions, take the
                  mispredicted instances, freeze a deterministic sample list.
  battery   GPU.  Re-run a ladder of configurations on that frozen list
                  (--model baseline | eadp), dump per-instance predictions and
                  the intermediate relevance maps for the EADP-256 arm.
  probe     GPU.  Causal necessity probe: occlude each 8x8 block of the 32x32
                  token grid and record which blocks flip the answer. Gives the
                  ground-truth "answer region" needed to separate A from C.
  classify  CPU.  Merge everything into the A/B/C/D taxonomy and report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
from common import (  # noqa: E402
    BASELINE_MODEL,
    build_dataset,
    build_message,
    eadp_model_name,
    ensure_out_dir,
    load_model,
    sample_indices,
)
from instrumented import attach_pruner  # noqa: E402
from scoring import per_sample_hits  # noqa: E402
from vlmeval.smp import load as vlm_load  # noqa: E402
from vlmeval.vlm.qwen3_vl.model_fixed_res import unwrap_visual_output  # noqa: E402

EADP_DIR = os.path.join(common.QWEN_ROOT, "outputs", "eadp", "Qwen3-VL-8B-EADP-256-a0.5-b2.0")
OFFICIAL_PREDS = {
    "TextVQA_VAL": os.path.join(
        EADP_DIR, "T20260921_Ge1a08801", "Qwen3-VL-8B-EADP-256-a0.5-b2.0_TextVQA_VAL.xlsx"
    ),
    "DocVQA_VAL": os.path.join(
        EADP_DIR, "T20260922_Ge1a08801", "Qwen3-VL-8B-EADP-256-a0.5-b2.0_DocVQA_VAL.xlsx"
    ),
    "OCRBench": os.path.join(
        EADP_DIR, "T20260921_Ge1a08801", "Qwen3-VL-8B-EADP-256-a0.5-b2.0_OCRBench.xlsx"
    ),
}

GRID = 32          # 32x32 merged visual tokens for 1024x1024 input
BLOCK = 8          # occlusion block size -> 4x4 = 16 blocks
SOLVED = 0.999     # per-instance hit threshold treated as "fully correct"


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def decode(model, out):
    text = model.processor.tokenizer.batch_decode(
        out, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return model._post_process_response(text)


def eadp_forward(model, message, dataset_name, budget, selector, capture=False,
                 occlude_block=None, image_embeds_override=None):
    """One EADP generation. Returns prediction + bookkeeping."""
    pruner = model.pruner
    pruner.visual_token_num = budget
    pruner.selector_name = selector
    pruner.capture = capture

    instr = model._get_instruction_sequence_embedding(message, dataset=dataset_name)
    inputs = model._processor_inputs(model._build_messages(message, dataset=dataset_name))
    pv = inputs["pixel_values"].type(model.model.visual.dtype)
    gthw = inputs["image_grid_thw"]

    with torch.no_grad():
        if image_embeds_override is not None:
            image_embeds = image_embeds_override.clone()
        else:
            image_embeds = unwrap_visual_output(model.model.visual(pv, grid_thw=gthw))

    if occlude_block is not None:
        r, c = divmod(occlude_block, (GRID // BLOCK))
        lo_r, lo_c = r * BLOCK, c * BLOCK
        idx = [
            i * GRID + j
            for i in range(lo_r, lo_r + BLOCK)
            for j in range(lo_c, lo_c + BLOCK)
        ]
        image_embeds[idx] = image_embeds.mean(dim=0, keepdim=True)

    n_img = gthw.shape[0]
    text_embeds_llm = instr.mean(dim=1).expand(n_img, -1)
    text_embeds_seq_llm = instr.expand(n_img, -1, -1)

    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    with torch.no_grad():
        pruned_embeds, sizes = pruner(
            image_embeds, text_embeds_llm, text_embeds_seq_llm, gthw
        )
    e.record()
    torch.cuda.synchronize()

    maps = dict(pruner.last_capture) if capture else {}
    select_idx = maps.pop("select_idx", None)

    inputs_embeds, attention_mask = model._build_pruned_inputs(
        inputs["input_ids"], inputs["attention_mask"], pruned_embeds, sizes
    )
    with torch.no_grad():
        out = model.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=False,
            **model.generate_kwargs,
        )
    return {
        "prediction": decode(model, out),
        "n_kept": int(sum(sizes)),
        "prune_ms": s.elapsed_time(e),
        "select_ms": pruner.last_timing.get("facility_location", float("nan")),
        "maps": maps,
        "select_idx": select_idx,
        "image_embeds": image_embeds.detach().float().cpu() if capture else None,
    }


def baseline_generate(model, message, dataset_name):
    inputs = model._processor_inputs(model._build_messages(message, dataset=dataset_name))
    kw = {k: v for k, v in inputs.items() if v is not None}
    with torch.no_grad():
        out = model.model.generate(do_sample=False, **model.generate_kwargs, **kw)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return decode(model, gen)


def load_bank(datasets, per_dataset, offset):
    """Rebuild the exact sample list (dataset, index) used everywhere."""
    bank = {}
    for ds in datasets:
        dataset = build_dataset(ds)
        idx = sample_indices(len(dataset.data), per_dataset, offset=offset)
        bank[ds] = {"idx": idx, "rows": [dataset.data.iloc[i] for i in idx]}
    return bank


# ---------------------------------------------------------------------------
# stage: select
# ---------------------------------------------------------------------------
def stage_select(args):
    out = {}
    for ds, path in OFFICIAL_PREDS.items():
        if ds not in args.datasets:
            continue
        data = vlm_load(path)
        rows = [data.iloc[i] for i in range(len(data))]
        preds = [r["prediction"] for r in rows]
        hits = per_sample_hits(ds, rows, preds)
        err = np.where(hits < args.error_threshold)[0]
        print(f"{ds}: {len(err)}/{len(hits)} mispredicted (hit < {args.error_threshold})")
        pick = sample_indices(len(err), args.per_dataset, offset=0)
        chosen = [int(err[i]) for i in pick]
        out[ds] = {
            "n_total": int(len(hits)),
            "n_error": int(len(err)),
            "error_rate_pct": float(100 * len(err) / len(hits)),
            "indices": chosen,
            "official_hits": [float(hits[i]) for i in chosen],
            "official_preds": [str(preds[i]) for i in chosen],
        }
        print(f"  -> froze {len(chosen)} error instances")

    path = os.path.join(ensure_out_dir(), "fa_samples.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "datasets": out}, f, indent=2)
    print(f"[saved] {path}")


# ---------------------------------------------------------------------------
# stage: battery
# ---------------------------------------------------------------------------
def stage_battery(args):
    with open(args.sample_file, encoding="utf-8") as f:
        samples = json.load(f)["datasets"]

    if args.model == "baseline":
        model = load_model(BASELINE_MODEL, max_new_tokens=args.max_new_tokens)
        arms = [("baseline", None, 1024)]
    else:
        model = load_model(
            eadp_model_name(256, args.alpha, args.beta), max_new_tokens=args.max_new_tokens
        )
        attach_pruner(model, selector="facility", capture=False)
        arms = [
            ("eadp", "facility", 128),
            ("eadp", "facility", 256),
            ("eadp", "facility", 512),
            ("topk", "topk", 256),
            ("topk", "topk", 512),
        ]

    results = {}
    maps_dir = os.path.join(ensure_out_dir(), "fa_maps")
    os.makedirs(maps_dir, exist_ok=True)

    for ds, info in samples.items():
        dataset = build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        rows = [dataset.data.iloc[i] for i in info["indices"]]
        msgs = [build_message(model, dataset, ds, r) for r in rows]
        for label, selector, budget in arms:
            key = f"{label}{budget if selector else ''}"
            store = results.setdefault(key, {})
            preds = []
            for j, m in enumerate(tqdm(msgs, desc=f"{key} {ds}", leave=False)):
                capture = selector == "facility" and budget == 256 and args.model == "eadp"
                try:
                    if args.model == "baseline":
                        p = baseline_generate(model, m, ds)
                        rec = {"prediction": p, "n_kept": GRID * GRID}
                    else:
                        rec = eadp_forward(
                            model, m, ds, budget, selector, capture=capture
                        )
                        if capture:
                            # Persist only the small relevance maps. sim_matrix
                            # (N,N) and image_embeds (N,D) are ~18 MB per sample
                            # combined and are not used downstream; compressing
                            # them cost ~3.5 s per instance.
                            heavy = {"sim_matrix", "image_embeds"}
                            np.savez_compressed(
                                os.path.join(maps_dir, f"{ds}_{info['indices'][j]}_b256.npz"),
                                **{
                                    k: v.numpy()
                                    for k, v in rec["maps"].items()
                                    if isinstance(v, torch.Tensor) and k not in heavy
                                },
                                select_idx=rec["select_idx"].numpy(),
                            )
                    preds.append(rec["prediction"])
                    store.setdefault("records", {}).setdefault(ds, {})[
                        str(info["indices"][j])
                    ] = {
                        "prediction": rec["prediction"],
                        "n_kept": rec["n_kept"],
                        "prune_ms": rec.get("prune_ms"),
                        "select_ms": rec.get("select_ms"),
                    }
                except Exception:
                    traceback.print_exc()
                    preds.append("")

            hits = per_sample_hits(ds, rows, preds)
            store[ds] = {
                "hits": [float(h) for h in hits],
                "acc_pct": float(np.mean(hits) * 100),
                "n": int(len(hits)),
            }
            print(f"{key:12s} {ds:12s} acc={store[ds]['acc_pct']:7.3f}")

        path = os.path.join(ensure_out_dir(), f"fa_battery_{args.model}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[saved] {path}")

    common.free_model(model)


# ---------------------------------------------------------------------------
# stage: probe  (causal necessity)
# ---------------------------------------------------------------------------
def stage_probe(args):
    with open(args.sample_file, encoding="utf-8") as f:
        samples = json.load(f)["datasets"]

    model = load_model(
        eadp_model_name(256, args.alpha, args.beta), max_new_tokens=args.max_new_tokens
    )
    attach_pruner(model, selector="facility", capture=False)

    n_blocks = (GRID // BLOCK) ** 2
    out = {}
    for ds, info in samples.items():
        if ds not in args.datasets:
            continue
        dataset = build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        idxs = info["indices"][: args.per_dataset]
        rows = [dataset.data.iloc[i] for i in idxs]
        msgs = [build_message(model, dataset, ds, r) for r in rows]
        out[ds] = {}

        for j, m in enumerate(tqdm(msgs, desc=f"probe {ds}", leave=False)):
            gi = int(idxs[j])
            try:
                # 1) unpruned reference (budget 1024 -> the pruner no-ops)
                ref = eadp_forward(model, m, ds, GRID * GRID, "facility")
                # 2) causal necessity: occlude each block, see which flip the answer
                needed = []
                for b in range(n_blocks):
                    r = eadp_forward(model, m, ds, GRID * GRID, "facility", occlude_block=b)
                    if r["prediction"].strip() != ref["prediction"].strip():
                        needed.append(b)
                # 3) the actual EADP-256 selection we are auditing
                sel_run = eadp_forward(model, m, ds, 256, "facility", capture=True)
                sel = sel_run["select_idx"][0].numpy()
                imp = sel_run["maps"]["importance"][0].numpy()
                np.savez_compressed(
                    os.path.join(ensure_out_dir(), "fa_maps", f"probe_{ds}_{gi}.npz"),
                    select_idx=sel, importance=imp, needed_blocks=np.array(needed),
                )
                out[ds][str(gi)] = {
                    "ref_prediction": ref["prediction"],
                    "prediction_256": sel_run["prediction"],
                    "needed_blocks": needed,
                    "n_blocks": n_blocks,
                }
            except Exception:
                traceback.print_exc()

        path = os.path.join(ensure_out_dir(), "fa_probe.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {path}")

    common.free_model(model)


# ---------------------------------------------------------------------------
# stage: classify
# ---------------------------------------------------------------------------
def stage_classify(args):
    with open(args.sample_file, encoding="utf-8") as f:
        samples = json.load(f)["datasets"]
    with open(os.path.join(ensure_out_dir(), "fa_battery_baseline.json"), encoding="utf-8") as f:
        base = json.load(f)
    with open(os.path.join(ensure_out_dir(), "fa_battery_eadp.json"), encoding="utf-8") as f:
        eadp = json.load(f)

    probe_path = os.path.join(ensure_out_dir(), "fa_probe.json")
    probe = json.load(open(probe_path, encoding="utf-8")) if os.path.exists(probe_path) else {}

    rows_out = []
    counts = {}
    for ds, info in samples.items():
        idxs = [str(i) for i in info["indices"]]
        dataset = build_dataset(ds)

        def hits_of(store, key):
            return {i: h for i, h in zip(idxs, store[key][ds]["hits"])}

        h_base = hits_of(base, "baseline")
        h_256 = hits_of(eadp, "eadp256")
        h_128 = hits_of(eadp, "eadp128")
        h_512 = hits_of(eadp, "eadp512")
        h_topk256 = hits_of(eadp, "topk256")
        h_topk512 = hits_of(eadp, "topk512")

        for i in idxs:
            solved_base = h_base[i] >= SOLVED
            solved_256 = h_256[i] >= SOLVED
            solved_topk256 = h_topk256[i] >= SOLVED
            solved_512 = h_512[i] >= SOLVED

            if not solved_base:
                cls = "D_unpruned_also_fails"
            elif solved_256:
                cls = "recovered"          # official run said wrong, our re-run says right
            elif solved_topk256:
                cls = "B_selector_dropped"
            elif solved_512:
                cls = "A_or_C_rank_below_cut"
            else:
                cls = "A_low_importance"

            row = dataset.data.iloc[int(i)]
            rec = {
                "dataset": ds,
                "index": int(i),
                "question": str(row.get("question", ""))[:300],
                "answer": str(row.get("answer", ""))[:300],
                "pred_official": info["official_preds"][idxs.index(i)],
                "pred_baseline": base["baseline"]["records"][ds].get(i, {}).get("prediction"),
                "pred_eadp256": eadp["eadp256"]["records"][ds].get(i, {}).get("prediction"),
                "pred_topk256": eadp["topk256"]["records"][ds].get(i, {}).get("prediction"),
                "class": cls,
                "hit_baseline": h_base[i],
                "hit_eadp128": h_128[i],
                "hit_eadp256": h_256[i],
                "hit_eadp512": h_512[i],
                "hit_topk256": h_topk256[i],
                "hit_topk512": h_topk512[i],
                "official_hit": info["official_hits"][idxs.index(i)],
            }
            # ---- causal refinement using the necessity probe, when available --
            pr = probe.get(ds, {}).get(i)
            if pr:
                rec["probe_needed_blocks"] = pr["needed_blocks"]
                npz_path = os.path.join(ensure_out_dir(), "fa_maps", f"probe_{ds}_{i}.npz")
                if os.path.exists(npz_path) and pr["needed_blocks"]:
                    z = np.load(npz_path)
                    sel = set(z["select_idx"].tolist())
                    imp = z["importance"]
                    nb = set()
                    for b in pr["needed_blocks"]:
                        r, c = divmod(int(b), GRID // BLOCK)
                        for ii in range(r * BLOCK, r * BLOCK + BLOCK):
                            for jj in range(c * BLOCK, c * BLOCK + BLOCK):
                                nb.add(ii * GRID + jj)
                    topk = set(np.argsort(-imp)[:256].tolist())
                    order = np.argsort(np.argsort(-imp))  # rank 0 = most important
                    rec["probe_coverage_eadp256"] = len(nb & sel) / len(nb)
                    rec["probe_coverage_topk256"] = len(nb & topk) / len(nb)
                    rec["probe_necessary_mean_rank_pct"] = float(
                        np.mean([order[t] / len(imp) for t in nb])
                    )
                    rec["probe_n_necessary_tokens"] = len(nb)
                    # refine A vs C: evidence present in the kept set yet still wrong
                    if rec["probe_coverage_eadp256"] >= 0.5 and not solved_256:
                        rec["class_refined"] = "C_kept_but_wrong"
                    elif rec["probe_coverage_topk256"] >= 0.5 and not solved_topk256:
                        rec["class_refined"] = "A_unranked"
                    elif rec["probe_coverage_topk256"] >= 0.5 and solved_topk256:
                        rec["class_refined"] = "B_selector_dropped"
                    else:
                        rec["class_refined"] = "A_unranked"

            rows_out.append(rec)
            counts[(ds, cls)] = counts.get((ds, cls), 0) + 1

    summary = {}
    for (ds, cls), n in sorted(counts.items()):
        summary.setdefault(cls, {})[ds] = n

    refined = {}
    for r in rows_out:
        c = r.get("class_refined")
        if c:
            refined.setdefault(c, {})
            refined[c][r["dataset"]] = refined[c].get(r["dataset"], 0) + 1

    path = os.path.join(ensure_out_dir(), "fa_classification.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "summary_refined": refined, "rows": rows_out}, f, indent=2)

    print("\n=== A/B/C/D taxonomy (EADP-256 errors) ===")
    for cls in sorted(summary):
        tot = sum(summary[cls].values())
        print(f"{cls:28s} n={tot:4d}  {summary[cls]}")

    if refined:
        print("\n=== refined with causal necessity probe (exploratory subset) ===")
        for cls in sorted(refined):
            tot = sum(refined[cls].values())
            print(f"{cls:28s} n={tot:4d}  {refined[cls]}")
    print(f"\n[saved] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["select", "battery", "probe", "classify"])
    ap.add_argument("--datasets", nargs="+", default=["TextVQA_VAL", "DocVQA_VAL", "OCRBench"])
    ap.add_argument("--per-dataset", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--error-threshold", type=float, default=0.5)
    ap.add_argument("--model", default="eadp", choices=["baseline", "eadp"])
    ap.add_argument("--sample-file", default=os.path.join(ensure_out_dir(), "fa_samples.json"))
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    {"select": stage_select, "battery": stage_battery,
     "probe": stage_probe, "classify": stage_classify}[args.stage](args)


if __name__ == "__main__":
    main()
