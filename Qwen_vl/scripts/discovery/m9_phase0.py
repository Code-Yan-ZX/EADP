"""
M9 Phase 0 -- architecture / feasibility audit on the live model.

Produces (printed + m9_phase0.json):
    P0-1  prompt layout: text prefix/suffix rows, question span, decoded
    P0-2  G-CAP: the captured-slice chain (q/k norm, rope, GQA, scaling, causal
          mask, o_proj head slicing) reconstructs the real layer output for the
          text rows -- an fp32 eager reference from the retained raw q/k/v must
          match the layer's own (bf16 SDPA) output to within bf16 tolerance
    P0-3  capture overhead: wall time of layers [0,8) with capture vs without,
          and peak extra memory
No boundary quality claim is made here (Phase 1's job).
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

import common                                                        # noqa: E402
from common import eadp_model_name, load_model                       # noqa: E402
from m2_gdep import MODE_FULL, GDEPConfig, GDEPEngine, dump_json     # noqa: E402
from m5_common import bank_items                                     # noqa: E402
from m9_capture import EarlyLayerCapture                             # noqa: E402
from m9_common import load_banks, instance_boundary, assemble, POOL  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import (         # noqa: E402
    apply_rotary_pos_emb,
)

N_CAP_LAYERS = 8
TOL = 5e-2          # relative, fp32 eager reference vs bf16 SDPA module


def eager_reference(retained):
    """Independent fp32 recomputation of a whole decoder layer from raw q/k/v
    (attention in eager fp32 instead of the module's bf16 SDPA; norms, MLP and
    residuals are the layer's own modules)."""
    layer = retained["layer"]
    attn = layer.self_attn
    q, k, v = (retained["pend"][n] for n in ("q_proj", "k_proj", "v_proj"))
    cos, sin = retained["pos_emb"]
    h_in = retained["layer_in"]
    S = q.shape[1]
    H, KV, D = 32, 8, 128
    grp = H // KV
    qn = attn.q_norm(q.view(1, S, H, D)).transpose(1, 2).float()
    kn = attn.k_norm(k.view(1, S, KV, D)).transpose(1, 2).float()
    vn = v.view(1, S, KV, D).transpose(1, 2).float()
    qr, kr = apply_rotary_pos_emb(qn, kn, cos.float(), sin.float())
    logits = torch.einsum("hqd,hsd->hqs",
                          qr[0], kr[0].repeat_interleave(grp, dim=0))
    logits = logits * attn.scaling
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=q.device), 1)
    logits = logits.masked_fill(mask[None], float("-inf"))
    probs = torch.softmax(logits, dim=-1)                    # (H,S,S)
    vhe = vn[0].repeat_interleave(grp, dim=0)                # (H,S,D)
    ctx = torch.einsum("hqs,hsd->hqd", probs, vhe)           # (H,S,D)
    w = attn.o_proj.weight.float()
    attn_out = (ctx.permute(1, 0, 2).reshape(S, H * D) @ w.t()).to(h_in.dtype)
    h1 = h_in + attn_out[None]
    h2 = h1 + layer.mlp(layer.post_attention_layernorm(h1))
    return probs, h2                                         # (H,S,S), (1,S,4096)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, default=None,
                    help="bank row to audit (default: first val row)")
    args = ap.parse_args()

    model = load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=8)
    model.model.eval()
    eng = GDEPEngine(model, GDEPConfig(mode=MODE_FULL, tag="M9P0"))
    banks = load_banks()
    raw = {k: banks[k] for k in ("key", "ds", "idx", "split")}
    row = args.row
    if row is None:
        row = int(np.flatnonzero(banks["split"] == "val")[0])
    items = bank_items(model, raw, splits=(str(banks["split"][row]),))
    it = next(x for x in items if x["key"] == str(banks["key"][row]))

    tok = model.processor.tokenizer
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    prep = eng.prepare(it["msg"], it["ds"])
    b = instance_boundary(banks, row)
    asm = assemble(prep, b["s_early"], im_end_id)

    tokz = prep["inputs"]["input_ids"][0]
    info = dict(
        row=row, key=str(banks["key"][row]), ds=it["ds"],
        seq_full=int(prep["prompt"].shape[1]),
        vis_slice=[int(asm["s"]), int(asm["e"])],
        text_prefix_rows=int(asm["s"]),
        n_text_suffix=int(prep["prompt"].shape[1] - asm["e"]),
        n_vis_early=int(b["s_early"].shape[0]),
        q_span_rows=list(asm["q_span"]),
        question_decoded=tok.decode(
            prep["inputs"]["input_ids"][0][int(asm["e"]) + 1:
                                           asm["im_end"]].tolist()),
        n_tail=16, n_reserve=int(b["reserve"].shape[0]),
        n_boundary=int(b["boundary"].shape[0]),
    )
    print(json.dumps({k: v for k, v in info.items()}, indent=1, default=str))

    # ---- run layers [0, N_CAP_LAYERS) with capture -----------------------
    hidden = asm["prompt"]
    dev = hidden.device
    cache = None   # phase 0: no cache needed; run layers manually
    bpos = torch.as_tensor(np.concatenate([
        asm["s"] + np.searchsorted(b["s_early"], b["boundary"])]),
        dtype=torch.long, device=dev)
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask
    cache = DynamicCache(config=eng.text.config)
    am = torch.ones(1, hidden.shape[1], dtype=torch.long, device=dev)
    cp = torch.arange(hidden.shape[1], device=dev)
    pos3 = cp.view(1, 1, -1).expand(3, 1, -1)
    attn_mask = create_causal_mask(config=eng.text.config, input_embeds=hidden,
                                   attention_mask=am, cache_position=cp,
                                   past_key_values=cache,
                                   position_ids=pos3[0])

    def run_plain(h):
        pos_emb = eng.text.rotary_emb(h, pos3)
        x = h
        for i in range(N_CAP_LAYERS):
            x = eng.layers[i](x, attention_mask=attn_mask,
                              position_ids=pos3[0],
                              past_key_values=cache, cache_position=cp,
                              position_embeddings=pos_emb)
        return x

    # overhead measurement: plain vs captured, 3 repeats each
    res = {}
    for name, use_cap in (("plain", False), ("captured", True)):
        torch.cuda.synchronize()
        ts = []
        for _ in range(3):
            cache = DynamicCache(config=eng.text.config)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if use_cap:
                with EarlyLayerCapture(eng.text, 0, N_CAP_LAYERS,
                                       asm["suffix_rows"], bpos) as cap:
                    run_plain(hidden)
                torch.cuda.synchronize()
            else:
                run_plain(hidden)
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        res[name] = ts
    info["timing_ms"] = res
    info["capture_overhead_ms"] = float(np.median(res["captured"])
                                        - np.median(res["plain"]))
    print(f"[timing] plain {np.median(res['plain']):.1f} ms, "
          f"captured {np.median(res['captured']):.1f} ms "
          f"({N_CAP_LAYERS} layers, S={hidden.shape[1]})")

    # ---- G-CAP: reconstruct layer-0 output from retained raw q/k/v -------
    with EarlyLayerCapture(eng.text, 0, 1, asm["suffix_rows"], bpos,
                           retain_layer=0) as cap:
        cache = DynamicCache(config=eng.text.config)
        h = hidden
        pos_emb = eng.text.rotary_emb(h, pos3)
        h = eng.layers[0](h, attention_mask=attn_mask, position_ids=pos3[0],
                          past_key_values=cache, cache_position=cp,
                          position_embeddings=pos_emb)
    real_out = h[0, asm["suffix_rows"], :].float()
    probs, ref_layer = eager_reference(cap.retained)
    ref_rows = ref_layer[0, asm["suffix_rows"], :].float()
    rel = ((real_out - ref_rows).norm(dim=-1)
           / real_out.norm(dim=-1).clamp_min(1e-6))
    info["gcap_rel_err_median"] = float(rel.median())
    info["gcap_rel_err_max"] = float(rel.max())
    info["gcap_pass"] = bool(rel.max() < TOL)
    print(f"[G-CAP] layer-0 text-row reconstruction: rel err median "
          f"{rel.median():.3e}, max {rel.max():.3e} -> "
          f"{'PASS' if info['gcap_pass'] else 'FAIL'}")

    # sanity: captured slices exist and have the right shape
    l0 = cap.layers[0]
    info["capture_shapes"] = {k: list(v.shape) for k, v in l0.items()}
    print("[captured]", info["capture_shapes"])

    dump_json("m9_phase0.json", dict(config=vars(args), info=info))


if __name__ == "__main__":
    main()
