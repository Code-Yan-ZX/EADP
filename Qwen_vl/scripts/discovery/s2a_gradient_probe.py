"""
S2-A: causal-gradient scoring viability probe.

Question: can gradient saliency w.r.t. the 1024 visual token embeddings entering
the LLM identify the causally necessary regions that the Stage-1 occlusion probe
already established?

Objectives
  A1 (answer-conditioned, diagnostic upper bound) -- teacher-forcing on the
     correct answer the unpruned model itself produced at probe time:
        J = sum_t log p(a_t | image, question, a_<t)
  A2 (answer-free, deployment-oriented) -- single forward, logits at the first
     answer position only:
        P1  J = logit[argmax]
        P2  J = logit[top1] - logit[top2]

Saliency per visual token i (hidden dim d):
  G1 = ||g_i||_2                    G1L1 = ||g_i||_1
  G2 = sum_d |g_i,d * v_i,d|

Everything is scored with the S1 `evaluate()` so the metrics are identical in
definition to the Stage-1 / S1 numbers.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import eadp_model_name
from s1_audit import MAPS, OUT, tokens_of_blocks
from scoring_search_s1 import evaluate

ANSWER_TOKEN_CAP = 64


# ---------------------------------------------------------------------------
def causal_cases():
    probe = json.load(open(os.path.join(OUT, "fa_probe.json"), encoding="utf-8"))
    cases = []
    for ds, per in probe.items():
        for k, v in per.items():
            if v["needed_blocks"]:
                cases.append(dict(ds=ds, idx=int(k), needed=np.array(v["needed_blocks"]),
                                  answer=v["ref_prediction"], pred256=v["prediction_256"]))
    return sorted(cases, key=lambda c: (c["ds"], c["idx"]))


def load_s1_best(keys):
    """S1's best ablation (global branch only, no smoothing) from the branch maps."""
    out = {}
    for k in keys:
        b = np.load(os.path.join(MAPS, f"{k}_b256.npz"))
        out[k] = b["global_sim"].reshape(-1).astype(np.float64)
    return out


class Timer:
    def __init__(self):
        self.e = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))]

    def __enter__(self):
        self.e[-1][0].record()
        return self

    def __exit__(self, *a):
        self.e[-1][1].record()

    @property
    def ms(self):
        torch.cuda.synchronize()
        return self.e[-1][0].elapsed_time(self.e[-1][1])


def grad_scores(vt):
    """G1 / G2 / G1L1 from vt.grad and vt."""
    g = vt.grad.detach()
    v = vt.detach()
    return {
        "G1": g.norm(dim=-1).float().cpu().numpy(),
        "G2": (g * v).abs().sum(dim=-1).float().cpu().numpy(),
        "G1L1": g.abs().sum(dim=-1).float().cpu().numpy(),
    }


# ---------------------------------------------------------------------------
def main():
    from vlmeval.vlm.qwen3_vl.model_fixed_res import unwrap_visual_output

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N cases")
    args = ap.parse_args()

    cases = causal_cases()
    if args.limit:
        cases = cases[: args.limit]
    print(f"causal cases: {len(cases)}")
    model = common.load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=128)
    model.model.eval()
    # Importing `vlmeval.config` calls torch.set_grad_enabled(False) globally, so
    # every earlier Stage-1/Stage-2 script silently ran with autograd off. Turn it
    # back on; params stay frozen so only the input embeddings accumulate grad.
    torch.set_grad_enabled(True)
    assert torch.is_grad_enabled(), "autograd must be on for saliency"
    for p in model.model.parameters():
        p.requires_grad_(False)          # gradient flows through, but is not accumulated
    tok = model.processor.tokenizer
    image_token_id = model.model.config.image_token_id
    dev = next(model.model.parameters()).device
    emb_layer = model.model.get_input_embeddings()

    records, scores_all = {}, {}
    by_ds = {}
    for c in cases:
        by_ds.setdefault(c["ds"], []).append(c)

    for ds, items in by_ds.items():
        dataset = common.build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        for c in items:
            key = f"{ds}_{c['idx']}"
            row = dataset.data.iloc[c["idx"]]
            msg = common.build_message(model, dataset, ds, row)
            messages = model._build_messages(msg, dataset=ds)
            inputs = model._processor_inputs(messages)
            ids, am = inputs["input_ids"], inputs["attention_mask"]

            with torch.no_grad():
                instr = model._get_instruction_sequence_embedding(msg, dataset=ds)
                pv = inputs["pixel_values"].type(model.model.visual.dtype)
                vis0 = unwrap_visual_output(
                    model.model.visual(pv, grid_thw=inputs["image_grid_thw"]))
                emb = emb_layer(ids)
                pos = (ids[0] == image_token_id).nonzero(as_tuple=True)[0]
                s, e = int(pos[0]), int(pos[-1]) + 1
                prefix, suffix = emb[:, :s], emb[:, e:]

            ans_ids = tok(c["answer"], add_special_tokens=False,
                          return_tensors="pt").input_ids.to(dev)
            n_ans_raw = int(ans_ids.shape[1])
            # OCRBench_214/222 hit the generation cap at probe time and produced a
            # 2048-token degenerate repetition. Cap teacher forcing uniformly so a
            # single repeated digit cannot dominate the objective; a no-op for the
            # other 13 cases (2-18 tokens).
            ans_ids = ans_ids[:, :ANSWER_TOKEN_CAP]

            rec = dict(key=key, ds=ds, idx=c["idx"], needed=[int(b) for b in c["needed"]],
                       answer=c["answer"], pred256=c["pred256"],
                       n_answer_tokens_raw=n_ans_raw,
                       n_answer_tokens=int(ans_ids.shape[1]),
                       prompt_tail=tok.decode(ids[0, max(0, ids.shape[1] - 24):]),
                       n_prompt_tokens=int(ids.shape[1]),
                       block_map={int(b): int(len(tokens_of_blocks([b])))
                                  for b in c["needed"]})
            scores = {}

            def fresh():
                return vis0.detach().clone().requires_grad_(True)

            # ---------------- A1: answer-conditioned -----------------------
            if ans_ids.shape[1] > 0:
                vt = fresh()
                prompt = torch.cat([prefix, vt.unsqueeze(0), suffix], dim=1)
                with torch.no_grad():
                    ae = emb_layer(ans_ids)
                full = torch.cat([prompt, ae], dim=1)
                L, P = full.shape[1], prompt.shape[1]
                torch.cuda.reset_peak_memory_stats()
                with Timer() as t_fwd:
                    out = model.model(inputs_embeds=full,
                                      attention_mask=torch.ones(1, L, dtype=am.dtype, device=dev),
                                      use_cache=False, return_dict=True)
                lg = out.logits[0, P - 1:P - 1 + ans_ids.shape[1]].float()
                J = -torch.nn.functional.cross_entropy(lg, ans_ids[0], reduction="sum")
                with Timer() as t_bwd:
                    J.backward()
                gs = grad_scores(vt)
                rec["A1_nll"] = float(-J.item())
                rec["t_A1_fwd_ms"], rec["t_A1_bwd_ms"] = t_fwd.ms, t_bwd.ms
                rec["A1_peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
                del out, full, prompt, J, lg
                scores.update({f"A1_{k}": v for k, v in gs.items()})
                del vt

            # ---------------- A2: answer-free ------------------------------
            vt = fresh()
            prompt = torch.cat([prefix, vt.unsqueeze(0), suffix], dim=1)
            torch.cuda.reset_peak_memory_stats()
            with Timer() as t_fwd:
                out = model.model(inputs_embeds=prompt,
                                  attention_mask=torch.ones(1, prompt.shape[1],
                                                            dtype=am.dtype, device=dev),
                                  use_cache=False, return_dict=True)
            last = out.logits[0, -1].float()
            top2 = torch.topk(last, 2).indices
            rec["P_top1_token"] = tok.decode([int(top2[0])])
            rec["P_margin"] = float((last[top2[0]] - last[top2[1]]).detach())
            rec["t_A2_fwd_ms"] = t_fwd.ms

            for name, J in (("P1", last[top2[0]]), ("P2", last[top2[0]] - last[top2[1]])):
                vt.grad = None
                with Timer() as t_bwd:
                    J.backward(retain_graph=True)
                gs = grad_scores(vt)
                rec[f"t_{name}_bwd_ms"] = t_bwd.ms
                rec[f"{name}_peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
                scores.update({f"{name}_{k}": v for k, v in gs.items()})
            del out, prompt, last, vt
            torch.cuda.empty_cache()

            rec["t_unpruned_fwd_ms"] = rec["t_A2_fwd_ms"]
            records[key] = rec
            scores_all[key] = scores
            print(f"{key:22s} ans={c['answer'][:28]!r:32s} nAns={rec['n_answer_tokens']:3d} "
                  f"fwd={rec['t_A2_fwd_ms']:7.1f}ms bwd={rec.get('t_P1_bwd_ms', float('nan')):7.1f}ms "
                  f"peak={rec['P1_peak_mem_mb']:6.0f}MB")

    json.dump({"records": records,
               "scores": {k: {m: v.tolist() for m, v in s.items()} for k, s in scores_all.items()},
               "meta": {"cases": [f"{c['ds']}_{c['idx']}" for c in cases],
                        "n_cases": len(cases)}},
              open(os.path.join(OUT, "s2a_gradient_viability.json"), "w"))
    print(f"\n[saved] {os.path.join(OUT, 's2a_gradient_viability.json')}")


if __name__ == "__main__":
    main()
