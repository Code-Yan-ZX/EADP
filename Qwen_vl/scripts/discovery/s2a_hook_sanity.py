"""
S2-A step 2: gradient-hook sanity check.

Establishes that the tensor we will differentiate has its rows in exact
one-to-one correspondence with (a) EADP's token index, (b) the Stage-1 saved
``select_idx``, and (c) the 32x32 spatial grid that the 8x8 occlusion blocks
were defined on.

Checks per case
  H1  visual tensor shape == [1024, hidden_dim]
  H2  the prompt contains exactly 1024 image-token positions, contiguous
  H3  re-running the *official* pruner on this tensor reproduces the saved
      Stage-1 ``select_idx`` bit-for-bit  (this is the load-bearing check: it
      proves we are differentiating the same tensor EADP scores)
  H4  our differentiable prompt-embedding reconstruction equals the official
      ``_build_pruned_inputs`` at budget 1024
  H5  block b <-> token indices: dump the 64 indices of several blocks and
      confirm the row/col unpacking matches failure_analysis.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import GRID_HW, eadp_model_name

BLOCK = 8
N_BLOCKS = (GRID_HW // BLOCK) ** 2
OUT = common.OUTPUT_DIR
MAPS = os.path.join(OUT, "fa_maps")


def causal_cases():
    probe = json.load(open(os.path.join(OUT, "fa_probe.json"), encoding="utf-8"))
    cases = []
    for ds, per in probe.items():
        for k, v in per.items():
            if v["needed_blocks"]:
                cases.append((ds, int(k), v))
    return sorted(cases)


def main():
    from vlmeval.vlm.qwen3_vl.model_fixed_res import unwrap_visual_output

    cases = causal_cases()
    print(f"causal cases: {len(cases)}")

    model = common.load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=128)
    tok = model.processor.tokenizer
    image_token_id = model.model.config.image_token_id
    dev = next(model.model.parameters()).device

    by_ds = {}
    for ds, idx, meta in cases:
        by_ds.setdefault(ds, []).append((idx, meta))

    report = []
    for ds, items in by_ds.items():
        dataset = common.build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        for idx, meta in items:
            row = dataset.data.iloc[idx]
            msg = common.build_message(model, dataset, ds, row)
            messages = model._build_messages(msg, dataset=ds)
            inputs = model._processor_inputs(messages)

            with torch.no_grad():
                pv = inputs["pixel_values"].type(model.model.visual.dtype)
                vis = unwrap_visual_output(
                    model.model.visual(pv, grid_thw=inputs["image_grid_thw"])
                )

            # H1
            shape_ok = tuple(vis.shape) == (1024, model.pruner.visual_dim)

            # H2
            ids = inputs["input_ids"]
            pos = (ids[0] == image_token_id).nonzero(as_tuple=True)[0]
            n_img = int(pos.numel())
            contiguous = bool(n_img and (pos[-1] - pos[0] + 1).item() == n_img)

            # H3 -- official pruner must reproduce the saved selection
            saved = np.load(os.path.join(MAPS, f"probe_{ds}_{idx}.npz"))["select_idx"]
            with torch.no_grad():
                instr = model._get_instruction_sequence_embedding(msg, dataset=ds)
                te = instr.mean(dim=1)
                from model.pruner import (_entropy_filter_impl, _greed_select_impl,
                                          _local_aggregation_impl, _sim_cross,
                                          _sim_visual_impl, _spatial_smoothing_impl)
                g, l_all = _sim_cross(te, instr, vis.unsqueeze(0))
                fl, tev = _entropy_filter_impl(l_all, T=100.0, entropy_keep_ratio=0.2)
                ls = _local_aggregation_impl(fl, tev, M_temp=0.01, strategy="negative_entropy")
                ts = (0.5 * g + 0.5 * ls).mean(dim=-1)
                imp = (ts - ts.min(-1, keepdim=True).values + 1e-6) / (
                    ts.max(-1, keepdim=True).values - ts.min(-1, keepdim=True).values + 1e-6)
                imp = _spatial_smoothing_impl(imp, GRID_HW, GRID_HW, 3, 1.0) ** 2.0
                sim = _sim_visual_impl(vis.unsqueeze(0))
                sel2, _ = _greed_select_impl(imp, sim, 256)
            # NOTE: capture stores the *unsorted* greedy order (B, T); compare as sets
            sel2 = sel2[0].cpu().numpy()
            sel_match = bool(np.array_equal(np.sort(sel2), np.sort(saved)))

            # H4 -- differentiable reconstruction == official builder
            with torch.no_grad():
                emb = model.model.get_input_embeddings()(ids)
                s, e = int(pos[0]), int(pos[-1]) + 1
                mine = torch.cat([emb[:, :s], vis.unsqueeze(0), emb[:, e:]], dim=1)
                official, _ = model._build_pruned_inputs(
                    ids, inputs["attention_mask"], vis, [vis.shape[0]]
                )
            recon = float((mine - official).abs().max())

            # H5 -- block geometry
            blocks = {}
            for b in [int(meta["needed_blocks"][0]), 5, 15]:
                r, c = divmod(b, GRID_HW // BLOCK)
                toks = [i * GRID_HW + j
                        for i in range(r * BLOCK, r * BLOCK + BLOCK)
                        for j in range(c * BLOCK, c * BLOCK + BLOCK)]
                blocks[b] = (toks[0], toks[-1], len(toks))

            rec = dict(ds=ds, idx=idx, shape=tuple(vis.shape), shape_ok=shape_ok,
                       n_img_tokens=n_img, contiguous=contiguous, sel_match=sel_match,
                       recon_maxdiff=recon, blocks=blocks)
            report.append(rec)
            print(f"{ds:12s} {idx:5d}  H1={shape_ok} H2={n_img}contig={contiguous} "
                  f"H3={sel_match} H4={recon:.2e} H5={blocks}")

    ok = all(r["shape_ok"] and r["n_img_tokens"] == 1024 and r["contiguous"]
             and r["sel_match"] and r["recon_maxdiff"] < 1e-5 for r in report)
    print(f"\nALL HOOK CHECKS PASS: {ok}  ({len(report)} cases)")
    json.dump(report, open(os.path.join(OUT, "s2a_hook_sanity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
