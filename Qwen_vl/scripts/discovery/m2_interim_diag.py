"""
M2 interim read-only diagnostic — run while the pre-registered grid is still
executing, to decide whether the C0/C1 degeneration is an implementation
artefact or a property of the mechanism.

This script **changes nothing the grid uses**: it opens its own model copy, runs
no accuracy arm, writes only `m2_interim_diag.json`, and never touches
`m2_accuracy.json` / `m2_perf.json` / `m2_correctness.json`.

Three diagnostics
-----------------
D1  K = 1024 multi-step identity. Gate G-C compared *prefill logits* only. This
    compares the engine's own greedy decode loop against `model.model.generate`
    token-by-token for 8+ steps on the identity path, where both must agree.

D2  Budget-256 introspection. Position ids handed to layers 5..35, the number of
    position gaps, per-layer KV shapes before/after compaction, cache_position
    progression across decode steps, and the exact EOS set both paths use.

D3  Same-token-set graph control. The single most informative read on the
    failure: take the *identical* 256-token set that GDEP selected mid-layer and
    run it through the incumbent's pre-LLM arrangement (all 36 layers over 290
    positions, contiguous ids). If the pre-LLM arm answers normally while the
    mid-LLM arm degenerates, the graph is the cause, not the token set and not
    the scorer. This is a diagnostic, not a new arm: nothing here is reported as
    an accuracy measurement.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                       # noqa: E402
from common import OUTPUT_DIR, eadp_model_name                      # noqa: E402
from m2_gdep import (MODE_GDEP, POLICY_PRESERVE, GDEPConfig,        # noqa: E402
                     GDEPEngine, dump_json)

PERIOD = 32          # the prompt's non-visual tokens are ~34; see report


def repetition(text: str, n: int = 12):
    """Crude loop detector: how often the most common n-gram repeats."""
    if len(text) < n:
        return dict(n=n, top_count=0, frac=0.0)
    grams = {}
    for i in range(0, len(text) - n + 1, max(1, n // 2)):
        g = text[i:i + n]
        grams[g] = grams.get(g, 0) + 1
    best = max(grams.values()) if grams else 0
    return dict(n=n, top_count=int(best),
                frac=float(best / max(1, len(text) - n)))


def one_instance(model, vlm, key):
    ds, idx = key.rsplit("_", 1)
    dataset = common.build_dataset(ds)
    model.set_dump_image(dataset.dump_image)
    row = dataset.data.iloc[int(idx)]
    msg = common.build_message(model, dataset, ds, row)
    return ds, int(idx), msg


def engine_for(model, budget, n_arm, seed, policy=POLICY_PRESERVE, tag="diag"):
    cfg = GDEPConfig(mode=MODE_GDEP, budget=budget, selector="topk",
                     pos_policy=policy, n_arm=n_arm, seed=seed, tag=tag)
    return GDEPEngine.from_checkpoint(cfg, model=model, max_new_tokens=64)


# ---------------------------------------------------------------------------
def d1_identity(model, ds, msg, steps=16):
    """Engine decode vs HF generate on the K = 1024 identity path."""
    eng = engine_for(model, 1024, 960, 2, tag="diag-K1024")
    prep = eng.prepare(msg, ds)
    st, info = eng.prefill(prep)
    eng_ids = eng.decode(st, steps)
    n = prep["prompt"].shape[1]
    with torch.no_grad():
        hf = model.model.generate(inputs_embeds=prep["prompt"],
                                  attention_mask=prep["mask"],
                                  do_sample=False, max_new_tokens=steps)
    # generate() with inputs_embeds and no input_ids returns the GENERATED tokens
    # ONLY -- the prompt is not prepended, because HF cannot reconstruct it. An
    # earlier version of this probe sliced hf[0, n:] and discarded the entire
    # continuation, reporting a vacuous 0-step match; the continuation is hf[0].
    hf_ids = hf[0].tolist()
    k = min(len(eng_ids), len(hf_ids))
    first = int(st["logits"].argmax())
    out = dict(
        engine_tokens=eng_ids, hf_tokens=hf_ids, prompt_len=int(n),
        hf_returned_shape=list(hf.shape),
        n_compared=k,
        all_steps_match=bool(eng_ids[:k] == hf_ids[:k]),
        per_step_match=[bool(a == b) for a, b in zip(eng_ids[:k], hf_ids[:k])],
        prefill_argmax_equals_hf_first=bool(first == hf_ids[0]) if hf_ids else None,
        engine_text=model.processor.tokenizer.decode(
            eng_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
        hf_text=model.processor.tokenizer.decode(
            hf_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
        fast_path=info.get("identity_fast_path"),
        n_kept=info["n_kept"], context_len=info["context_len"],
        kv_seq_len=info["kv_seq_len"],
        eos_generation_config=model.model.generation_config.eos_token_id,
        eos_config=model.model.config.eos_token_id,
        eos_tokenizer=model.processor.tokenizer.eos_token_id,
    )
    print(f"[D1] K=1024 identity: engine {eng_ids} vs HF {hf_ids} -> "
          f"{'MATCH' if out['all_steps_match'] else 'MISMATCH'} "
          f"(compared {k} steps)")
    print(f"     engine text={out['engine_text']!r}")
    print(f"     hf     text={out['hf_text']!r}")
    del st, prep, eng
    torch.cuda.empty_cache()
    return out


def d2_introspection(model, ds, msg, budget=256, n_arm=960, seed=2):
    """What the surviving layers actually see."""
    eng = engine_for(model, budget, n_arm, seed, tag="diag-256")
    prep = eng.prepare(msg, ds)
    st, info = eng.prefill(prep)
    cache = st["cache"]
    pos1 = st["pos1"]
    gaps = int((pos1[1:] - pos1[:-1] != 1).sum())
    shapes_early = {i: list(cache.layers[i].keys.shape) for i in (0, 4)}
    shapes_late = {i: list(cache.layers[i].keys.shape) for i in (5, 10, 35)}
    _ = eng.decode(st, 3)
    shapes_after3 = {i: list(cache.layers[i].keys.shape) for i in (0, 5, 35)}
    out = dict(
        budget=budget, n_arm=n_arm, seed=seed,
        seq_full=info["seq_full"], n_text=info["n_text"],
        n_kept=info["n_kept"], context_len=info["context_len"],
        kv_seq_len_before=info["kv_seq_len"],
        position_ids_min=int(pos1.min()), position_ids_max=int(pos1.max()),
        position_id_gaps=gaps, pos_ids_contiguous=info["pos_ids_contiguous"],
        decode_positions_used=eng.last_decode_positions,
        kv_shapes_prefill_early_layers=shapes_early,
        kv_shapes_prefill_late_layers=shapes_late,
        kv_shapes_after_3_decode_steps=shapes_after3,
        kv_bytes_early=info["kv_bytes_early"], kv_bytes_late=info["kv_bytes_late"],
        cache_policy=info["cache_policy"], pos_policy=info["pos_policy"],
        select_idx_head=info["select_idx"][:8],
        select_idx_tail=info["select_idx"][-8:],
        select_idx_sorted=bool(info["select_idx"] == sorted(info["select_idx"])),
        keep_full_len=len(info["keep_full"]),
    )
    print(f"[D2] budget 256: context_len={info['context_len']} "
          f"pos[{out['position_ids_min']},{out['position_ids_max']}] gaps={gaps} "
          f"cache early={shapes_early[0]} late={shapes_late[5]} "
          f"after3={shapes_after3[0]}")
    print(f"     decode (cache_position, rope_position) = "
          f"{eng.last_decode_positions}")
    del st, prep, eng
    torch.cuda.empty_cache()
    return out


def d3_same_tokens_prellm(model, ds, msg, keept, budget=256):
    """Run an arbitrary token set through the pre-LLM arrangement.

    All 36 layers see only the kept 256 visual + all text, contiguous ids from
    0 -- exactly the incumbent's graph, but with the candidate's token set.
    """
    eng = engine_for(model, budget, 960, 2, tag="diag-pre")
    prep = eng.prepare(msg, ds)
    eng.pruner = None                                  # no official scoring here
    vis = prep["vis"]
    s, e = prep["vis_slice"]
    keep = torch.as_tensor(sorted(keept), device=eng.dev)
    pruned = vis.index_select(0, keep)
    prompt, mask = prep["prompt"], prep["mask"]
    hidden = torch.cat([prompt[:, :s], pruned.unsqueeze(0), prompt[:, e:]], dim=1)
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache(config=eng.text.config)
    pos1 = torch.arange(hidden.shape[1], device=eng.dev)
    n = hidden.shape[1]
    cp = torch.arange(n, device=eng.dev)
    pos3 = pos1.view(1, 1, -1).expand(3, 1, -1)
    am = torch.ones(1, n, dtype=torch.long, device=eng.dev)
    attn = eng._causal(hidden, am, cp, cache, pos3[0])
    with torch.no_grad():
        h = eng._run(hidden, attn, pos3, cache, cp, 0, eng.n_layers)
        h = eng.text.norm(h)
        logits = eng.model.lm_head(h[:, -1:, :])[:, -1, :]
    st = dict(cache=cache, logits=logits, pos1=pos1,
              n_ctx=torch.ones(1, n, dtype=torch.long, device=eng.dev),
              hidden_len=n)
    ids = eng.decode(st, 12)
    text = model.processor.tokenizer.decode(ids, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
    print(f"[D3] pre-LLM arm on the SAME {len(keept)} tokens: {text[:120]!r}")
    del st, prep, eng
    torch.cuda.empty_cache()
    return dict(n_tokens=len(keept), first12_ids=ids, text=text[:400],
                context_len=int(n), repetition=repetition(text))


def d3_midllm(model, ds, msg, budget=256, n_arm=960, seed=2):
    """The same instance through GDEP's mid-layer graph."""
    eng = engine_for(model, budget, n_arm, seed, tag="diag-mid")
    prep = eng.prepare(msg, ds)
    st, info = eng.prefill(prep)
    ids = eng.decode(st, 12)
    text = model.processor.tokenizer.decode(ids, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
    print(f"[D3] mid-LLM (GDEP) arm same instance: {text[:120]!r}")
    kept = info["select_idx"]
    del st, prep, eng
    torch.cuda.empty_cache()
    return kept, dict(first12_ids=ids, text=text[:400], n_kept=info["n_kept"],
                      repetition=repetition(text))


def d4_runaways(model, acc, arm_key, n=3, budget=256, n_arm=None, seed=None):
    """Re-derive `n` capped instances and re-run them through both graphs.

    For each: the mid-LLM parent picks the token set; the pre-LLM child runs the
    same set through the incumbent's arrangement. Both are compared on the text
    they produce and on the repetition metric.
    """
    if arm_key not in acc["arms"]:
        return []
    r = acc["arms"][arm_key]
    spec = dict(C0=(240, 0), C1=(960, 0), C2=(960, 0), C3=(960, 0))
    n_arm = n_arm or spec.get(r["arm"], (960, 0))[0]
    seed = seed if seed is not None else r["seed"]
    keys = acc["keys"]
    preds = r["predictions"]
    capped = [i for i, p in enumerate(preds) if len(p) > 2000][:n]
    out = []
    for i in capped:
        key = keys[i]
        ds, idx, msg = one_instance(model, model, key)
        kept, mid = d3_midllm(model, ds, msg, budget, n_arm, seed)
        pre = d3_same_tokens_prellm(model, ds, msg, kept, budget)
        out.append(dict(key=key, benchmark=ds, charset=idx,
                        mid_llm=mid, pre_llm=pre,
                        mid_llm_repetition=mid["repetition"],
                        pre_llm_repetition=pre["repetition"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default=None,
                    help="key like TextVQA_VAL_0; default first held-out")
    ap.add_argument("--runaways", type=int, default=3)
    ap.add_argument("--arm", default="C0|s0")
    ap.add_argument("--out", default="m2_interim_diag.json")
    args = ap.parse_args()

    t0 = time.time()
    model = common.load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=64)
    model.model.eval()
    torch.set_grad_enabled(False)
    print(f"[diag] model loaded in {time.time()-t0:.0f} s")

    acc_path = os.path.join(OUTPUT_DIR, "m2_accuracy.json")
    acc = json.load(open(acc_path)) if os.path.exists(acc_path) else {"arms": {}, "keys": []}

    if args.instance:
        key = args.instance
    else:
        key = acc["keys"][0] if acc.get("keys") else "TextVQA_VAL_0"
    ds, idx, msg = one_instance(model, model, key)
    print(f"[diag] instance {key}")

    rep = dict(instance=key, started=time.strftime("%F %T"),
               environment=dict(torch=torch.__version__,
                                gpu=torch.cuda.get_device_name(0)))
    rep["D1_k1024_multistep_identity"] = d1_identity(model, ds, msg)
    rep["D2_budget256_introspection"] = d2_introspection(model, ds, msg)
    rep["D3_same_token_set_graph_control"] = d4_runaways(
        model, acc, args.arm, n=args.runaways)
    rep["wall_seconds"] = float(time.time() - t0)
    rep["peak_allocated_mb"] = float(torch.cuda.max_memory_allocated() / 1024**2)
    dump_json(args.out, rep)
    print(f"[diag] done in {time.time()-t0:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()