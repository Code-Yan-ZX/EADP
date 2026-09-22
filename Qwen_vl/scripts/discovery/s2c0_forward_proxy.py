"""
S2-C0 step 1: forward-only visual-influence proxies, no backward anywhere.

Question: can a *forward-only* signal taken from the first few LLM layers
approximate the S2-B P1-G2 gradient teacher (507 ms/image)?

Query position: the first answer position = the last token of the prompt-only
forward, exactly as S2-A's P1 objective. The prompt is built the same way as
``s2b_gradient_scores.py`` (inputs_embeds, no input_ids, no image_grid_thw), so
the position encoding the proxy sees is identical to the one the teacher saw.

Per layer L in {1,2,4,8}, per visual token i, with attention a_{h,i} of the last
query onto visual key i at head h and value v_{h,i}:

    A  score_i = mean_h a_{h,i}                       (attention only; ~FastV)
    B  score_i = mean_h a_{h,i} * ||v_{h,i}||_2       (attention x value norm)
    C  score_i = || W_o [a_{1,i} v_{1,i} ; ...] ||_2  (exact output contribution)

Implementation note: the model runs SDPA, so the attention matrix is never
materialised by the model. A forward pre-hook on ``layers[L].self_attn``
recomputes q/k/v for that layer with the layer's own projections, norms and
rotary embedding and evaluates **only the last query row** against the visual
keys -- a [heads, 1024] tensor, never a seq x seq matrix. The hook only reads;
the model's own forward is untouched and stays on SDPA. ``--sanity`` verifies the
hand-computed row against the official eager path on a real case.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import eadp_model_name, sample_indices
from s1_audit import OUT
from s2b_gradient_scores import frozen_bank

LAYERS = [1, 2, 4, 8]
VARIANTS = ["A", "B", "C"]


# ---------------------------------------------------------------------------
# forward pre-hook: last-prompt-query attention onto the visual keys
# ---------------------------------------------------------------------------
class LastQueryAttention:
    """Captures attn( last prompt token -> visual tokens ) for one layer."""

    def __init__(self, layer_idx: int, abort_after: bool = False):
        self.layer_idx = layer_idx
        self.abort_after = abort_after      # used only for the prefix-latency probe
        self.visual_slice = None            # (start, end) set by the caller
        self.out = None                     # dict of the three score vectors
        self._handle = None

    # -- the recomputation ---------------------------------------------------
    def _compute(self, module, hidden_states, position_embeddings):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

        s, e = self.visual_slice
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, module.head_dim)

        q = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # only the last query position is needed; all keys are causally visible
        n_groups = module.num_key_value_groups
        q_all = q[0, :, -1, :]                                # [H, D]
        k_all = k[0].repeat_interleave(n_groups, dim=0)       # [H, S, D]  (GQA, real layout)
        v_all = v[0].repeat_interleave(n_groups, dim=0)

        # The model dtype is bf16; the proxy itself is evaluated in fp32 (only the
        # ranking matters), but a bf16 mirror of the same row is kept so the
        # sanity check can separate "my derivation is wrong" from "bf16 rounds".
        with torch.no_grad():
            mirror = torch.softmax(
                (torch.einsum("hd,hsd->hs", q_all, k_all) * module.scaling).float(), dim=-1
            ).to(q_all.dtype)[:, s:e]

        q_last, k_all_f, v_f = q_all.float(), k_all.float(), v_all.float()
        scores = torch.einsum("hd,hsd->hs", q_last, k_all_f) * module.scaling
        attn = torch.softmax(scores, dim=-1)[:, s:e]           # [H, n_vis]
        vis_v = v_f[:, s:e, :]                                 # [H, n_vis, D]

        a_norm = vis_v.norm(dim=-1)                            # [H, n_vis]
        A = attn.mean(dim=0)
        B = (attn * a_norm).mean(dim=0)

        # o_proj consumes the heads concatenated in head-major order (the official
        # path does attn_output.transpose(1, 2).reshape(*input_shape, -1)), so the
        # per-token contribution must be permuted before flattening -- a plain
        # reshape of the [H, n_vis, D] tensor would interleave heads and tokens.
        contrib = (attn.unsqueeze(-1) * vis_v).permute(1, 0, 2).reshape(attn.shape[-1], -1)
        C = (contrib @ module.o_proj.weight.float().t()).norm(dim=-1)

        return {"A": A, "B": B, "C": C}, attn, mirror

    def hook(self, module, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
        res, attn, mirror = self._compute(module, hs, pe)
        self.out = {k: v.detach().float().cpu().numpy() for k, v in res.items()}
        self.attn = attn.detach()
        self.attn_bf16 = mirror.detach()
        if self.abort_after:
            raise _AbortForward
        return None

    def attach(self, layer):
        self._handle = layer.self_attn.register_forward_pre_hook(self.hook, with_kwargs=True)

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class _AbortForward(Exception):
    pass


# ---------------------------------------------------------------------------
def build_instance(model, dataset, ds, row):
    """Prompt-only forward inputs, identical construction to s2b_gradient_scores."""
    msg = common.build_message(model, dataset, ds, row)
    inputs = model._processor_inputs(model._build_messages(msg, dataset=ds))
    return msg, inputs


def load_instance_tensors(model, inputs, image_token_id):
    from vlmeval.vlm.qwen3_vl.model_fixed_res import unwrap_visual_output

    ids = inputs["input_ids"]
    with torch.no_grad():
        pv = inputs["pixel_values"].type(model.model.visual.dtype)
        vis0 = unwrap_visual_output(model.model.visual(pv, grid_thw=inputs["image_grid_thw"]))
        emb = model.model.get_input_embeddings()(ids)
        pos = (ids[0] == image_token_id).nonzero(as_tuple=True)[0]
        s, e = int(pos[0]), int(pos[-1]) + 1
        prompt = torch.cat([emb[:, :s], vis0.unsqueeze(0), emb[:, e:]], dim=1)
        mask = torch.ones(1, prompt.shape[1], dtype=torch.long, device=prompt.device)
    return prompt, mask, s, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["TextVQA_VAL", "DocVQA_VAL", "OCRBench"])
    ap.add_argument("--pilot-every", type=int, default=3,
                    help="pilot = frozen_150[::N] -> 50 per benchmark (N=3, as in S2-B)")
    ap.add_argument("--layers", nargs="+", type=int, default=LAYERS)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap instances per dataset")
    ap.add_argument("--prefix-latency", action="store_true",
                    help="also measure an early-exit prefix forward per layer")
    args = ap.parse_args()

    from vlmeval.vlm.qwen3_vl.model_fixed_res import unwrap_visual_output  # noqa: F401

    bank = frozen_bank(args.datasets)
    causal = json.load(open(os.path.join(OUT, "s2a_gradient_viability.json")))["cases"]
    causal_by_ds = {}
    for k, c in causal.items():
        causal_by_ds.setdefault(c["ds"], set()).add(int(c["idx"]))

    model = common.load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=32)
    model.model.eval()
    torch.set_grad_enabled(False)
    image_token_id = model.model.config.image_token_id
    layers = model.model.model.language_model.layers

    probes = {L: LastQueryAttention(L) for L in args.layers}
    for L in args.layers:
        probes[L].attach(layers[L])

    scores, meta, timings = {}, {}, []
    for ds, b in bank.items():
        model.set_dump_image(b["dataset"].dump_image)
        pilot = set(b["idx"][:: args.pilot_every])
        want = sorted(pilot | causal_by_ds.get(ds, set()))
        if args.limit:
            want = want[: args.limit]
        rows = {int(i): b["dataset"].data.iloc[int(i)] for i in want}
        print(f"[{ds}] {len(want)} instances "
              f"({len(pilot & set(want))} pilot, {len(causal_by_ds.get(ds, set()) & set(want))} causal)")

        for j, i in enumerate(want):
            msg, inputs = build_instance(model, b["dataset"], ds, rows[i])
            prompt, mask, s, e = load_instance_tensors(model, inputs, image_token_id)
            for L in args.layers:
                probes[L].visual_slice = (s, e)
                probes[L].out = None

            t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
            t0.record()
            with torch.no_grad():
                model.model.model(inputs_embeds=prompt, attention_mask=mask,
                                  use_cache=False, return_dict=True)
            t1.record()
            torch.cuda.synchronize()
            ms = t0.elapsed_time(t1)
            timings.append(ms)

            key = f"{ds}_{i}"
            for L in args.layers:
                if probes[L].out is None:
                    raise RuntimeError(f"layer {L} hook never fired for {key}")
                for v in VARIANTS:
                    scores[f"{v}_L{L}__{key}"] = probes[L].out[v]
            meta[key] = dict(dataset=ds, index=int(i), in_pilot=bool(i in pilot),
                             in_causal=bool(i in causal_by_ds.get(ds, set())),
                             n_img_tokens=int(e - s), seq_len=int(prompt.shape[1]),
                             fwd_ms=float(ms))
            del prompt, mask, inputs
            torch.cuda.empty_cache()
            if (j + 1) % 20 == 0 or j == len(want) - 1:
                print(f"  {ds} {j+1}/{len(want)}  mean fwd {np.mean(timings):.0f} ms")

    np.savez_compressed(os.path.join(OUT, "s2c0_forward_proxy.npz"), **scores)
    print(f"\n[saved] {len(scores)} score vectors -> "
          f"{os.path.join(OUT, 's2c0_forward_proxy.npz')}")

    # ---------------- prefix forward latency --------------------------------
    prefix_ms = {}
    if args.prefix_latency:
        for L in args.layers:
            probe = LastQueryAttention(L, abort_after=True)
            probe.attach(layers[L])
            ds = args.datasets[0]
            b = bank[ds]
            i = int(b["idx"][2])
            msg, inputs = build_instance(model, b["dataset"], ds, b["dataset"].data.iloc[i])
            prompt, mask, s, e = load_instance_tensors(model, inputs, image_token_id)
            probe.visual_slice = (s, e)
            for _ in range(2):                                   # warmup
                try:
                    model.model.model(inputs_embeds=prompt, attention_mask=mask,
                                      use_cache=False, return_dict=True)
                except _AbortForward:
                    pass
            ts = []
            for _ in range(5):
                t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
                t0.record()
                try:
                    model.model.model(inputs_embeds=prompt, attention_mask=mask,
                                      use_cache=False, return_dict=True)
                except _AbortForward:
                    pass
                t1.record()
                torch.cuda.synchronize()
                ts.append(t0.elapsed_time(t1))
            probe.detach()
            prefix_ms[L] = dict(mean=float(np.mean(ts)), std=float(np.std(ts)),
                                n_layers_run=L + 1, n_layers_total=len(layers))
            print(f"[prefix] through layer {L}: {np.mean(ts):.1f} ms "
                  f"({L+1}/{len(layers)} layers)")

    json.dump({"meta": meta, "fwd_ms_mean": float(np.mean(timings)),
               "fwd_ms_median": float(np.median(timings)),
               "prefix_ms": prefix_ms,
               "layers": args.layers, "variants": VARIANTS,
               "pilot_every": args.pilot_every},
              open(os.path.join(OUT, "s2c0_forward_proxy_meta.json"), "w"), indent=1)
    print(f"[saved] meta; mean prompt forward {np.mean(timings):.1f} ms")


if __name__ == "__main__":
    main()
