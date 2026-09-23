"""
M2 interim diagnostic D5 — isolate the PRESERVE position-advance defect.

Read-only, changes nothing the grid uses, writes only m2_interim_d5.json.

D2 showed that in the PRESERVE branch every generated token is re-issued the
RoPE position `max(prefill_position_ids) + 1` because `pos1` is never advanced
inside the decode loop, while the RENUMBER branch advances correctly through
`cache_position`. Those are two different bugs-in-waiting: a stuck position for
PRESERVE, and none for RENUMBER.

D5 measures what that costs, on three instances, by running the *identical*
engine three ways:

    P   mid-LLM, PRESERVE, positions frozen at max+1      (what the grid ran)
    P'  mid-LLM, PRESERVE, positions advanced max+1,+2,.. (the intended policy)
    R   mid-LLM, RENUMBER (contiguous, advancing)         (the declared control)
    L   pre-LLM, same token set, contiguous               (incumbent's graph)

If P' and R answer while P degenerates, the degradation is an implementation
defect in the decode position policy, not the token set and not the method.
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
from m2_gdep import (MODE_GDEP, POLICY_PRESERVE, POLICY_RENUMBER,   # noqa: E402
                     GDEPConfig, GDEPEngine, dump_json)
from m2_interim_diag import repetition                              # noqa: E402


def build(model, budget=256, n_arm=960, seed=0, policy=POLICY_PRESERVE, tag="d5"):
    cfg = GDEPConfig(mode=MODE_GDEP, budget=budget, selector="topk",
                     pos_policy=policy, n_arm=n_arm, seed=seed, tag=tag)
    return GDEPEngine.from_checkpoint(cfg, model=model, max_new_tokens=64)


@torch.no_grad()
def decode_advanced(eng, state, steps):
    """The intended PRESERVE policy: every step takes one more position.

    This is a diagnostic re-implementation of the loop. It does not modify the
    engine the grid is using; it exists only to quantify what the defect cost.
    """
    import torch as T
    from transformers.cache_utils import DynamicCache  # noqa: F401

    cache = state["cache"]
    n_ctx = state["n_ctx"]
    pos1 = state["pos1"]
    logits = state["logits"]
    eos = eng.model.generation_config.eos_token_id
    eos = set(eos) if isinstance(eos, (list, tuple)) else {eos}
    embed = eng.model.get_input_embeddings()
    nxt_pos = int(pos1.max().item()) + 1
    used = []
    out = []
    cur = int(T.argmax(logits, dim=-1).item())
    for _ in range(steps):
        out.append(cur)
        if cur in eos:
            break
        cur_len = int(cache.get_seq_length(0))
        n_ctx = T.cat([n_ctx, n_ctx.new_ones(1, 1)], dim=1)
        cp = T.tensor([cur_len], device=eng.dev)
        used.append((cur_len, nxt_pos))
        pos3 = T.tensor([[[nxt_pos]]], device=eng.dev).expand(3, 1, 1)
        h = embed(T.tensor([[cur]], device=eng.dev))
        attn = eng._causal(h, n_ctx, cp, cache, pos3[0])
        h = eng._run(h, attn, pos3, cache, cp, 0, eng.n_layers)
        eng.counts["decode_steps"] = eng.counts.get("decode_steps", 0) + 1
        h = eng.text.norm(h)
        logits = eng.model.lm_head(h[:, -1:, :])[:, -1, :]
        cur = int(T.argmax(logits, dim=-1).max().item())
        nxt_pos += 1
    eng.last_decode_positions = used
    return out


def run_variant(model, ds, msg, variant, steps=48, budget=256, n_arm=960, seed=0):
    policy = POLICY_RENUMBER if variant == "R" else POLICY_PRESERVE
    eng = build(model, budget, n_arm, seed, policy, tag=f"d5-{variant}")
    prep = eng.prepare(msg, ds)
    st, info = eng.prefill(prep)
    if variant == "P":
        ids = eng.decode(st, steps)
        pos = eng.last_decode_positions
    elif variant == "P2":
        ids = decode_advanced(eng, st, steps)
        pos = eng.last_decode_positions
    elif variant == "R":
        ids = eng.decode(st, steps)
        pos = eng.last_decode_positions
    else:
        raise KeyError(variant)
    text = model.processor.tokenizer.decode(ids, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
    kept = info["select_idx"]
    del st, prep, eng
    torch.cuda.empty_cache()
    return dict(variant=variant, n_tokens=len(ids), first_ids=ids[:12],
                text=text[:300], chars=len(text),
                repetition=repetition(text),
                positions_first3=pos[:3],
                keep_head=kept[:5], keep_tail=kept[-5:]), kept


def run_prellm(model, ds, msg, kept, steps=48, budget=256):
    # POLICY_RENUMBER here is not a policy claim about the incumbent: the
    # pre-LLM prompt is contiguous, so RENUMBER==PRESERVE on the prefill side,
    # and it makes the decode loop take its advancing branch. Building this arm
    # with POLICY_PRESERVE (as the first D5/D3 run did) silently inherited the
    # same frozen-position defect and made the control degenerate too.
    eng = build(model, budget, 960, 0, POLICY_RENUMBER, tag="d5-L")
    prep = eng.prepare(msg, ds)
    vis = prep["vis"]
    s, e = prep["vis_slice"]
    keep = torch.as_tensor(sorted(kept), device=eng.dev)
    pruned = vis.index_select(0, keep)
    prompt, mask = prep["prompt"], prep["mask"]
    hidden = torch.cat([prompt[:, :s], pruned.unsqueeze(0), prompt[:, e:]], dim=1)
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache(config=eng.text.config)
    n = hidden.shape[1]
    pos1 = torch.arange(n, device=eng.dev)
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
    ids = eng.decode(st, steps)
    positions = eng_positions(eng)
    text = model.processor.tokenizer.decode(ids, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
    del st, prep, eng
    torch.cuda.empty_cache()
    return dict(variant="L", n_tokens=len(ids), first_ids=ids[:12],
                text=text[:300], chars=len(text), repetition=repetition(text),
                positions_first3=positions)


def eng_positions(eng):
    return getattr(eng, "last_decode_positions", [])[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--out", default="m2_interim_d5.json")
    args = ap.parse_args()

    acc_path = os.path.join(OUTPUT_DIR, "m2_accuracy.json")
    acc = json.load(open(acc_path))
    keys = acc["keys"]
    if args.instances:
        keys = args.instances
    else:
        preds = acc["arms"]["C0|s0"]["predictions"]
        capped = [i for i, p in enumerate(preds) if len(p) > 2000][:3]
        keys = [keys[i] for i in capped]
        if len(keys) < 3:
            keys += acc["keys"][: 3 - len(keys)]

    model = common.load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=64)
    model.model.eval()
    torch.set_grad_enabled(False)

    out = dict(started=time.strftime("%F %T"), instances=[], steps=args.steps,
               note=("read-only diagnostic; the pre-registered grid keeps "
                     "running untouched"))
    for key in keys:
        ds, idx = key.rsplit("_", 1)
        dataset = common.build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        row = dataset.data.iloc[int(idx)]
        msg = common.build_message(model, dataset, ds, row)
        print(f"\n=== {key} ===")
        rec = dict(key=key, ds=ds, idx=int(idx))
        p, kept = run_variant(model, ds, msg, "P", args.steps)
        p2, _ = run_variant(model, ds, msg, "P2", args.steps)
        r, _ = run_variant(model, ds, msg, "R", args.steps)
        l = run_prellm(model, ds, msg, kept, args.steps)
        rec.update(P=p, P2=p2, R=r, L=l)
        for v in (p, p2, r, l):
            print(f"  [{v['variant']:2s}] {v['n_tokens']:3d} tok  "
                  f"rep={v['repetition']['frac']:.3f}  pos={v['positions_first3']}  "
                  f"{v['text'][:70]!r}")
        out["instances"].append(rec)
        dump_json(args.out, out)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()