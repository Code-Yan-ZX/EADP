"""
S2-C1 step 5: latency, reported the two ways the brief asks for.

  A. scoring-only -- prefix forward to L + tiny scorer + Top-K. This is what the
     selector costs on top of whatever comes next.
  B. naive end-to-end -- A, plus actually re-running the pruned model on the 256
     kept visual tokens. A real system that scores from a prefix forward and then
     re-runs the pruned model pays the prefix on top of the full pruned prefill;
     the prefix is NOT amortised, and quoting the scoring-only number as the
     method's cost would hide that.

The pruned forward is measured, not extrapolated: the unselected visual rows are
deleted from the prompt embeddings and the model is run on what is left, exactly
as the generation harness would after selection. Decode steps are excluded --
they are identical for every arm -- so these are prefill numbers.

The vision tower (~117 ms) is excluded throughout: every method and the
unpruned baseline pay it identically.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import eadp_model_name, time_callable
from s1_audit import OUT
from s2b_gradient_scores import frozen_bank
from s2c1_features import PrefixCapture, _AbortForward, load_instance_tensors

DS_ORDER = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]


def bench(fn, repeat=20, warmup=5):
    return time_callable(fn, repeat=repeat, warmup=warmup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c1")
    ap.add_argument("--layers", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=20)
    args = ap.parse_args()

    T = json.load(open(os.path.join(OUT, f"{args.tag}_train.json")))
    bank = frozen_bank(DS_ORDER)
    ds = DS_ORDER[0]
    idx = int(bank[ds]["idx"][2])

    model = common.load_model(eadp_model_name(args.budget, 0.5, 2.0), max_new_tokens=32)
    model.model.eval()
    torch.set_grad_enabled(False)
    image_token_id = model.model.config.image_token_id
    layers = model.model.model.language_model.layers

    dataset = bank[ds]["dataset"]
    model.set_dump_image(dataset.dump_image)
    msg = common.build_message(model, dataset, ds, dataset.data.iloc[idx])
    inputs = model._processor_inputs(model._build_messages(msg, dataset=ds))
    prompt, mask, s, e = load_instance_tensors(model, inputs, image_token_id)
    n_vis = e - s
    print(f"[instance] {ds}_{idx}: seq_len={prompt.shape[1]} visual={n_vis}")

    out = {"instance": f"{ds}_{idx}", "seq_len_full": int(prompt.shape[1]),
           "n_visual": int(n_vis), "budget": args.budget, "repeat": args.repeat}

    # ---- unpruned full prefill (the baseline every method is measured against)
    full_ms, full_sd = bench(lambda: model.model.model(
        inputs_embeds=prompt, attention_mask=mask, use_cache=False, return_dict=True),
        repeat=args.repeat)
    out["full_prefill_ms"] = full_ms
    out["full_prefill_ms_std"] = full_sd
    print(f"full unpruned prefill ({prompt.shape[1]} tokens): {full_ms:.1f} ms")

    # ---- prefix forward, early-exit hook per layer -------------------------
    prefix = {}
    for L in args.layers:
        probe = PrefixCapture(L, abort_after=True)
        probe.attach(layers[L])
        probe.visual_slice = (s, e)

        def run_prefix():
            try:
                model.model.model(inputs_embeds=prompt, attention_mask=mask,
                                  use_cache=False, return_dict=True)
            except _AbortForward:
                pass
        ms, sd = bench(run_prefix, repeat=args.repeat)
        probe.detach()
        prefix[L] = ms
        print(f"prefix through layer {L} ({L + 1}/{len(layers)} layers): {ms:.1f} ms")
    out["prefix_ms"] = {str(k): v for k, v in prefix.items()}

    # ---- pruned prefill at the budget --------------------------------------
    # Keep the teacher's Top-256 visual rows and drop the rest, which is what the
    # harness hands to the model after selection.
    G = np.load(os.path.join(OUT, "s2b_gradient_scores.npz"))
    keep = np.sort(np.argsort(-G[f"{ds}_{idx}"].astype(np.float64), kind="stable")[:args.budget])
    vis = prompt[0, s:e]
    pruned = torch.cat([prompt[:, :s], vis[keep].unsqueeze(0),
                        prompt[:, e:]], dim=1)
    pmask = torch.ones(1, pruned.shape[1], dtype=torch.long, device=pruned.device)
    pruned_ms, pruned_sd = bench(lambda: model.model.model(
        inputs_embeds=pruned, attention_mask=pmask, use_cache=False, return_dict=True),
        repeat=args.repeat)
    out["pruned_prefill_ms"] = pruned_ms
    out["pruned_prefill_ms_std"] = pruned_sd
    out["pruned_seq_len"] = int(pruned.shape[1])
    print(f"pruned prefill ({pruned.shape[1]} tokens): {pruned_ms:.1f} ms")

    # ---- scorer + Top-K -----------------------------------------------------
    h = torch.randn(1, n_vis, 4096, device="cuda", dtype=torch.float32)
    q1 = torch.randn(1, 4096, device="cuda")
    scorer = {}
    for arm, res in T["results"].items():
        L = res["layer"]
        rank = res["rank_dim"]
        if res["family"] == "LIN":
            w = torch.randn(1, 1, 4096, device="cuda") * 0.01
            fn = lambda w=w: (h * w).sum(-1)
        else:
            Wv = torch.randn(4096, rank, device="cuda") * 0.01
            Wq = torch.randn(4096, rank, device="cuda") * 0.01
            fn = lambda Wv=Wv, Wq=Wq: ((h @ Wv) * (q1 @ Wq)).sum(-1) * rank ** -0.5
        sc_ms, sc_sd = bench(fn, repeat=args.repeat)
        tk_ms, tk_sd = bench(lambda: torch.topk(fn(), args.budget, dim=-1), repeat=args.repeat)
        scorer[arm] = dict(layer=L, family=res["family"], n_params=res["n_params"],
                           scorer_ms=sc_ms, topk_ms=tk_ms)
        print(f"scorer {arm}: {sc_ms:.3f} ms, +TopK {tk_ms:.3f} ms")
    out["scorer"] = scorer

    # ---- A and B ------------------------------------------------------------
    best = max(scorer, key=lambda a: scorer[a]["n_params"]) if scorer else None
    for arm, sc in scorer.items():
        L = sc["layer"]
        scoring_only = prefix[L] + sc["scorer_ms"] + sc["topk_ms"]
        e2e = scoring_only + pruned_ms
        sc["scoring_only_ms"] = scoring_only
        sc["naive_end_to_end_ms"] = e2e
        sc["vs_teacher_507ms"] = scoring_only / 507.0
    out["best_arm"] = best
    json.dump(out, open(os.path.join(OUT, f"{args.tag}_latency.json"), "w"), indent=1)
    print(f"\n[saved] {os.path.join(OUT, f'{args.tag}_latency.json')}")


if __name__ == "__main__":
    main()
