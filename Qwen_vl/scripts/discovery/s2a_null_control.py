"""
S2-A false-positive control: run the *identical* A1 objective but teacher-force a
mismatched answer (the next case's answer, cyclically). The prompt, the image and
the visual tokens are unchanged, so if the saliency were driven by image content
or by the prompt alone, it would still localize the case's necessary blocks.

If the mismatched-answer saliency drops to chance while the matched one does not,
the signal is genuinely answer-conditioned.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import eadp_model_name
from s1_audit import BLOCK, OUT
from scoring_search_s1 import evaluate

ANSWER_TOKEN_CAP = 64


def main():
    from vlmeval.vlm.qwen3_vl.model_fixed_res import unwrap_visual_output

    D = json.load(open(os.path.join(OUT, "s2a_gradient_viability.json")))
    recs = D["records"]
    keys = sorted(recs)
    mism = {k: recs[keys[(i + 1) % len(keys)]]["answer"] for i, k in enumerate(keys)}

    model = common.load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=128)
    model.model.eval()
    torch.set_grad_enabled(True)
    for p in model.model.parameters():
        p.requires_grad_(False)
    tok = model.processor.tokenizer
    image_token_id = model.model.config.image_token_id
    dev = next(model.model.parameters()).device
    emb_layer = model.model.get_input_embeddings()

    out = {}
    by_ds = {}
    for k in keys:
        by_ds.setdefault(recs[k]["ds"], []).append(k)

    for ds, ks in by_ds.items():
        dataset = common.build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        for k in ks:
            idx = recs[k]["idx"]
            row = dataset.data.iloc[idx]
            msg = common.build_message(model, dataset, ds, row)
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

            ans = tok(mism[k], add_special_tokens=False,
                      return_tensors="pt").input_ids[:, :ANSWER_TOKEN_CAP].to(dev)
            vt = vis0.detach().clone().requires_grad_(True)
            prompt = torch.cat([prefix, vt.unsqueeze(0), suffix], dim=1)
            with torch.no_grad():
                ae = emb_layer(ans)
            full = torch.cat([prompt, ae], dim=1)
            P = prompt.shape[1]
            outm = model.model(inputs_embeds=full,
                               attention_mask=torch.ones(1, full.shape[1],
                                                         dtype=torch.long, device=dev),
                               use_cache=False, return_dict=True)
            lg = outm.logits[0, P - 1:P - 1 + ans.shape[1]].float()
            J = -torch.nn.functional.cross_entropy(lg, ans[0], reduction="sum")
            J.backward()
            g = vt.grad.detach()
            v = vt.detach()
            out[k] = {"G1": g.norm(dim=-1).float().cpu().numpy(),
                      "G2": (g * v).abs().sum(dim=-1).float().cpu().numpy(),
                      "nll": float(-J.item()),
                      "mismatched_answer": mism[k][:40]}
            print(f"{k:20s} nll={out[k]['nll']:8.2f}  "
                  f"(matched nll={recs[k].get('A1_nll', float('nan')):6.2f})  "
                  f"wrong ans={mism[k][:26]!r}")
            del outm, full, prompt, J, lg, vt
            torch.cuda.empty_cache()

    np.savez(os.path.join(OUT, "s2a_null_control.npz"),
             **{f"{k}__{m}": out[k][m] for k in out for m in ("G1", "G2")})
    json.dump({k: {"nll": out[k]["nll"], "mismatched_answer": out[k]["mismatched_answer"]}
               for k in out}, open(os.path.join(OUT, "s2a_null_control.json"), "w"), indent=1)

    print("\n=== mismatched-answer control vs matched A1 ===")
    print(f"{'tier':10s}{'method':14s}{'matched':>10s}{'mismatched':>12s}"
          f"{'matchedAUROC':>14s}{'mismatchAUROC':>15s}")
    for tk, fn in [("loc64", lambda n: n == 64), ("loc256", lambda n: n <= 256),
                   ("all", lambda n: True)]:
        sel = [k for k in keys if fn(len(recs[k]["needed"]) * BLOCK * BLOCK)]
        for m in ("G1", "G2"):
            mr, au_m, au_x = [], [], []
            for k in sel:
                rec = dict(needed=np.array(recs[k]["needed"]))
                r1 = evaluate(rec, np.asarray(D["scores"][k][f"A1_{m}"], dtype=np.float64))
                r2 = evaluate(rec, out[k][m].astype(np.float64))
                mr.append((r1["mean_rank_pct"], r2["mean_rank_pct"]))
                au_m.append(r1["block_auroc"])
                au_x.append(r2["block_auroc"])
            a, b = np.mean([x[0] for x in mr]), np.mean([x[1] for x in mr])
            print(f"{tk:10s}{m:14s}{a:10.4f}{b:12.4f}{np.nanmean(au_m):14.3f}"
                  f"{np.nanmean(au_x):15.3f}")


if __name__ == "__main__":
    main()
