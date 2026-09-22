"""
S2-C1 step 1: cache the cheap prefix-forward representations a learned scorer is
allowed to see.

Per instance, one prompt-only forward, built exactly as in
``s2b_gradient_scores.py`` / ``s2c0_forward_proxy.py``: ``inputs_embeds`` with
the visual embeddings injected at the image-token positions, no ``input_ids``,
no ``image_grid_thw``. Forward hooks on decoder layers L2 and L4 (0-based,
matching the S2-C0 layer convention) capture

    h_i  -- the hidden state of visual token i at the output of layer L
            -> (1024, 4096)
    q    -- the hidden state of the first answer position -- the last token of
            the prompt-only forward, the same query position S2-A P1 and S2-C0
            used -> (4096,)

Both are what a scorer sitting *after* layer L gets for free. Nothing here is a
gradient: no backward pass runs anywhere in this file.

The full 36-layer forward is run rather than early-exiting after L4. The two
hooks are cheap and a full forward is ~222 ms, so aborting the layer loop would
save ~160 ms/instance at the cost of a fragile exception path through the
stack. The prefix cost a real system would pay is measured separately
(``--latency``) with the same abort hook S2-C0 used.

Instance set: the frozen 450 (the Stage-1 sample bank, 150/benchmark) plus the
15 S2-A causal cases, which are disjoint from it. 465 instances, one forward
each.
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
from s1_audit import OUT
from s2b_gradient_scores import frozen_bank

LAYERS = [2, 4]                 # 0-based, same convention as S2-C0
D_MODEL = 4096
N_VIS = 1024
DS_ORDER = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]


# ---------------------------------------------------------------------------
# image-level split. test == frozen_150[::3] == the S2-C0 / S2-B pilot, so every
# baseline arm for the accuracy translation is already cached on it.
# ---------------------------------------------------------------------------
def frozen_split(n_total: int, per_ds: int = 150):
    idx = sample_indices(n_total, per_ds)
    test = idx[::3]
    pool = [i for i in idx if i not in set(test)]
    val = pool[::5]
    fit = [i for i in pool if i not in set(val)]
    return fit, val, test


def build_plan(datasets):
    """Ordered instance list + per-instance split assignment. No model needed."""
    plan, seen = [], set()
    for ds in DS_ORDER:
        fit, val, test = frozen_split(len(datasets[ds].data))
        split_of = {**{i: "fit" for i in fit}, **{i: "val" for i in val},
                    **{i: "test" for i in test}}
        for i in sorted(split_of):
            key = f"{ds}_{i}"
            plan.append(dict(key=key, ds=ds, idx=int(i), split=split_of[i],
                             causal=False))
            seen.add(key)
    cases = json.load(open(os.path.join(OUT, "s2a_gradient_viability.json")))["cases"]
    n_overlap = 0
    for k in sorted(cases):
        if k in seen:
            n_overlap += 1
            continue
        plan.append(dict(key=k, ds=cases[k]["ds"], idx=int(cases[k]["idx"]),
                         split="causal", causal=True))
        seen.add(k)
    assert n_overlap == 0, f"{n_overlap} causal cases overlap the frozen bank"
    return plan


# ---------------------------------------------------------------------------
class PrefixCapture:
    """Forward hook on a decoder layer: keep the visual slice and the last row."""

    def __init__(self, layer_idx: int, abort_after: bool = False):
        self.layer_idx = layer_idx
        self.abort_after = abort_after
        self.visual_slice = None
        self.h = None
        self.q = None
        self._handle = None

    def attach(self, layer):
        self._handle = layer.register_forward_hook(self.hook)

    def hook(self, module, args, output):
        hs = output[0] if isinstance(output, (tuple, list)) else output
        s, e = self.visual_slice
        self.h = hs[0, s:e, :].detach().to(torch.float16).cpu().numpy()
        self.q = hs[0, -1, :].detach().to(torch.float16).cpu().numpy()
        if self.abort_after:
            raise _AbortForward
        return None

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class _AbortForward(Exception):
    pass


def load_instance_tensors(model, inputs, image_token_id):
    """Visually-injected prompt embeddings. Same construction as S2-B / S2-C0."""
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
    ap.add_argument("--layers", nargs="+", type=int, default=LAYERS)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap instances")
    ap.add_argument("--latency", action="store_true",
                    help="also measure prefix-forward latency per layer")
    ap.add_argument("--tag", default="s2c1")
    args = ap.parse_args()

    bank = frozen_bank(DS_ORDER)                       # cross-checks frozen indices
    datasets = {ds: bank[ds]["dataset"] for ds in DS_ORDER}

    plan = build_plan(datasets)
    if args.limit:
        plan = plan[: args.limit]
    n_split = {s: sum(1 for p in plan if p["split"] == s)
               for s in ("fit", "val", "test", "causal")}
    print(f"[plan] {len(plan)} instances: {n_split}")

    model = common.load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=32)
    model.model.eval()
    torch.set_grad_enabled(False)
    image_token_id = model.model.config.image_token_id
    layers = model.model.model.language_model.layers

    probes = {L: PrefixCapture(L) for L in args.layers}
    for L in args.layers:
        probes[L].attach(layers[L])

    # memmap straight to disk: fp16 (1025, 4096) per instance per layer is
    # ~8 MB, so 465 instances x 2 layers is ~7.8 GB and never needs to sit in RAM.
    H = {L: np.lib.format.open_memmap(
        os.path.join(OUT, f"{args.tag}_feats_L{L}.npy"), mode="w+",
        dtype=np.float16, shape=(len(plan), N_VIS, D_MODEL)) for L in args.layers}
    Q = {L: np.lib.format.open_memmap(
        os.path.join(OUT, f"{args.tag}_query_L{L}.npy"), mode="w+",
        dtype=np.float16, shape=(len(plan), D_MODEL)) for L in args.layers}

    timings = []
    cur_ds = None
    for j, p in enumerate(plan):
        ds = p["ds"]
        if ds != cur_ds:                     # the causal group interleaves datasets
            model.set_dump_image(datasets[ds].dump_image)
            cur_ds = ds
        row = datasets[ds].data.iloc[p["idx"]]
        msg = common.build_message(model, datasets[ds], ds, row)
        inputs = model._processor_inputs(model._build_messages(msg, dataset=ds))
        prompt, mask, s, e = load_instance_tensors(model, inputs, image_token_id)
        assert e - s == N_VIS, f"{p['key']}: {e - s} visual tokens, expected {N_VIS}"

        for L in args.layers:
            probes[L].visual_slice = (s, e)
            probes[L].h = probes[L].q = None

        t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
        t0.record()
        model.model.model(inputs_embeds=prompt, attention_mask=mask,
                          use_cache=False, return_dict=True)
        t1.record()
        torch.cuda.synchronize()
        timings.append(t0.elapsed_time(t1))

        for L in args.layers:
            if probes[L].h is None:
                raise RuntimeError(f"layer {L} hook never fired for {p['key']}")
            H[L][j] = probes[L].h
            Q[L][j] = probes[L].q
        p["seq_len"] = int(prompt.shape[1])
        p["vis_start"] = int(s)
        del prompt, mask, inputs
        torch.cuda.empty_cache()
        if (j + 1) % 50 == 0 or j == len(plan) - 1:
            print(f"  {j+1}/{len(plan)}  mean fwd {np.mean(timings):.0f} ms")

    for L in args.layers:
        H[L].flush()
        Q[L].flush()

    latency = {}
    if args.latency:
        p = next(x for x in plan if x["split"] == "test")
        ds = p["ds"]
        row = datasets[ds].data.iloc[p["idx"]]
        msg = common.build_message(model, datasets[ds], ds, row)
        inputs = model._processor_inputs(model._build_messages(msg, dataset=ds))
        prompt, mask, s, e = load_instance_tensors(model, inputs, image_token_id)
        for L in args.layers:
            probe = PrefixCapture(L, abort_after=True)
            probe.attach(layers[L])
            probe.visual_slice = (s, e)
            for _ in range(2):
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
            latency[L] = dict(prefix_ms_mean=float(np.mean(ts)),
                              prefix_ms_std=float(np.std(ts)),
                              n_layers_run=L + 1, n_layers_total=len(layers))
            print(f"[latency] prefix through layer {L}: {np.mean(ts):.1f} ms "
                  f"({L+1}/{len(layers)} layers)")
        # unpruned full forward, for the naive end-to-end denominator
        probe = PrefixCapture(len(layers) - 1)
        probe.attach(layers[-1])
        probe.visual_slice = (s, e)
        ts = []
        for _ in range(5):
            t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
            t0.record()
            model.model.model(inputs_embeds=prompt, attention_mask=mask,
                              use_cache=False, return_dict=True)
            t1.record()
            torch.cuda.synchronize()
            ts.append(t0.elapsed_time(t1))
        probe.detach()
        latency["full_unpruned"] = dict(prefix_ms_mean=float(np.mean(ts)),
                                        prefix_ms_std=float(np.std(ts)),
                                        n_layers_run=len(layers),
                                        n_layers_total=len(layers))
        print(f"[latency] full unpruned LLM forward: {np.mean(ts):.1f} ms")

    json.dump({"plan": plan, "layers": args.layers, "d_model": D_MODEL,
               "n_vis": N_VIS, "split_counts": n_split,
               "fwd_ms_mean": float(np.mean(timings)),
               "fwd_ms_median": float(np.median(timings)),
               "latency": latency},
              open(os.path.join(OUT, f"{args.tag}_features.json"), "w"), indent=1)
    print(f"[saved] {args.tag}_features.json + feats/query memmaps "
          f"({len(plan)} instances); mean prompt forward {np.mean(timings):.0f} ms")


if __name__ == "__main__":
    main()
