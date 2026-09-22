"""
S2-B step 1: compute the P1 + gradient x input (G2) score for the frozen 450.

Frozen set = the exact Stage-1 Part 2 sample bank:
``sample_indices(len(dataset.data), 150, offset=0)`` per benchmark, verified
against the ``idx`` field stored in ``diag_selectors_b256.json``.

Objective (identical to S2-A P1, no answer, no GT, no continuation):
    k = argmax(logits at the first answer position)
    J = logits[k]
    score_i = sum_d | dJ/dv_i,d * v_i,d |
One prompt forward + one backward per instance.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import eadp_model_name, sample_indices
from instrumented import attach_pruner
from s1_audit import OUT


def frozen_bank(datasets, per_dataset=150, offset=0):
    """Rebuild the frozen sample list and cross-check against the saved run."""
    saved = json.load(open(os.path.join(OUT, "diag_selectors_b256.json")))
    bank = {}
    for ds in datasets:
        dataset = common.build_dataset(ds)
        idx = sample_indices(len(dataset.data), per_dataset, offset=offset)
        ref = None
        for k, v in saved["runs"].items():
            if k.endswith(f"|{ds}") and v["selector"] == "facility":
                ref = v["idx"]
        if ref is not None and list(idx) != list(ref):
            raise SystemExit(f"{ds}: rebuilt indices do not match the saved run")
        bank[ds] = dict(dataset=dataset, idx=idx,
                        rows=[dataset.data.iloc[i] for i in idx])
        print(f"[frozen] {ds}: {len(idx)} instances, indices verified against "
              f"diag_selectors_b256.json")
    return bank


def main():
    from vlmeval.vlm.qwen3_vl.model_fixed_res import unwrap_visual_output

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["TextVQA_VAL", "DocVQA_VAL", "OCRBench"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    bank = frozen_bank(args.datasets)
    model = common.load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=2048)
    model.model.eval()
    torch.set_grad_enabled(True)          # vlmeval.config disables autograd globally
    assert torch.is_grad_enabled()
    for p in model.model.parameters():
        p.requires_grad_(False)
    image_token_id = model.model.config.image_token_id
    dev = next(model.model.parameters()).device
    emb_layer = model.model.get_input_embeddings()

    # Official pruner, used only to record the EADP-256 selection on the same
    # features (verified bit-identical to the official implementation in Stage 1).
    pruner = attach_pruner(model, selector="facility", capture=True)
    pruner.visual_token_num = 256

    scores, meta, timings = {}, {}, []
    off_sel, off_imp = {}, {}
    for ds, b in bank.items():
        model.set_dump_image(b["dataset"].dump_image)
        idxs = b["idx"][: args.limit] if args.limit else b["idx"]
        for n, i in enumerate(idxs):
            row = b["rows"][n]
            msg = common.build_message(model, b["dataset"], ds, row)
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

                # official EADP-256 scoring + facility selection on the same features
                instr = model._get_instruction_sequence_embedding(msg, dataset=ds)
                gthw = inputs["image_grid_thw"]
                ni = gthw.shape[0]
                pruner(vis0, instr.mean(dim=1).expand(ni, -1),
                       instr.expand(ni, -1, -1), gthw)
                off_sel[f"{ds}_{i}"] = pruner.last_capture["select_idx"][0].numpy()
                off_imp[f"{ds}_{i}"] = pruner.last_capture["importance"][0].numpy()

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
            key = f"{ds}_{i}"
            scores[key] = g2
            meta[key] = dict(dataset=ds, index=int(i), n_img_tokens=int(pos.numel()),
                             top1_token_id=k,
                             top1_token=str(model.processor.tokenizer.decode([k])),
                             fwd_ms=t0.elapsed_time(t1), bwd_ms=t0b.elapsed_time(t1b),
                             peak_mb=torch.cuda.max_memory_allocated() / 2**20,
                             g2_sum=float(g2.sum()), g2_max=float(g2.max()))
            timings.append((meta[key]["fwd_ms"], meta[key]["bwd_ms"], meta[key]["peak_mb"]))
            del out, prompt, last, J, vt, g
            torch.cuda.empty_cache()
            if (n + 1) % 25 == 0 or n == len(idxs) - 1:
                print(f"  {ds} {n+1}/{len(idxs)}  fwd={np.mean([t[0] for t in timings]):.0f}ms "
                      f"bwd={np.mean([t[1] for t in timings]):.0f}ms")

    np.savez_compressed(os.path.join(OUT, "s2b_gradient_scores.npz"), **scores)
    # official selections live in their own archive (int, not float, and needed
    # only for the overlap / spatial-concentration diagnostics)
    np.savez_compressed(
        os.path.join(OUT, "s2b_official_selection.npz"),
        **{f"sel__{k}": v for k, v in off_sel.items()},
        **{f"imp__{k}": v for k, v in off_imp.items()},
    )
    json.dump(meta, open(os.path.join(OUT, "s2b_gradient_scores_meta.json"), "w"), indent=1)
    print(f"\n[saved] {len(scores)} scores -> {os.path.join(OUT, 's2b_gradient_scores.npz')}")
    print(f"  mean fwd {np.mean([t[0] for t in timings]):.1f} ms, "
          f"bwd {np.mean([t[1] for t in timings]):.1f} ms, "
          f"peak {np.mean([t[2] for t in timings]):.0f} MB")


if __name__ == "__main__":
    main()
