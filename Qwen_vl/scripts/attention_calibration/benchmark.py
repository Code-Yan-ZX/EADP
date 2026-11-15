"""Single-development-page paired end-to-end TTFT/fixed-16 benchmark."""
from __future__ import annotations

from contextlib import nullcontext
import json
import time

import torch

from experiment_common import (OUT, K, MAX_TOKENS, SEED, NativeEngine, common,
    message, resolution, save_json, sha256)
from runtime import encode_vz
from binding_selectors import select_binding
from calibration import AttentionLayout, attention_calibration
from effective_mass import cluster_weights
from method import ARMS, specification
from calibration_run import verify_freeze


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def measure(eng, item, arm):
    rz = resolution(eng, 512) if arm == "rres512" else nullcontext()
    with rz:
        begin = sync_time()
        msg = message(item)
        prep = eng.prepare(msg, "DocVQA_VAL")
        t_prep = sync_time()
        if arm in ("full", "rres512") or arm.startswith("b2_"):
            V, DS = eng.encode(prep)
            scores = keys = None
        else:
            V, DS, scores, keys = encode_vz(eng, prep)
        t_vision = sync_time()
        keep = vs = ds = None
        layout = spec = None
        if arm.startswith("b2_"):
            text_mean, text_seq = eng.instruction_embeds(msg, "DocVQA_VAL")
            ctx = dict(prep=prep, V=V, DS=DS, K=K, engine=eng,
                       text_mean=text_mean, text_seq=text_seq, seed=SEED)
            keep = eng.select("b2", K, ctx)
            weights = torch.ones(K, device=V.device)
        elif arm not in ("full", "rres512"):
            selection = select_binding(V, DS, scores, keys, K, None, None,
                                       "baseline_fixed", SEED)
            keep, vs, ds, _ = selection
            if arm == "effective":
                _, weights = cluster_weights(keys, selection)
            elif arm in ("count", "restore_prefill", "restore_continuous"):
                tensors = selection[3]["tensors"]
                weights = torch.ones(K, device=V.device)
                weights[torch.searchsorted(keep, tensors["anchor_idx"])] = tensors["cluster_sizes"].float()
            else:
                weights = torch.ones(K, device=V.device)
        t_select = sync_time()
        if keep is not None:
            layout = AttentionLayout.from_engine(eng, prep, keep, key_weights=weights)
            spec = specification(eng, arm)
            scope = attention_calibration(eng.text, layout, spec)
        else:
            scope = nullcontext()
        with scope:
            t_setup = sync_time()
            st = eng.prefill(prep, V, DS, keep, V_sel=vs, DS_sel=ds)
            first = int(st.logits.argmax(dim=-1).item())
            ttft = sync_time()
            ids, _ = eng.decode(st, 16, ignore_eos=True)
            end = sync_time()
        if len(ids) != 16 or not ids or ids[0] != first:
            raise AssertionError("fixed-length timing generation failed")
        return dict(ttft_ms=(ttft-begin)*1000, total_16_ms=(end-begin)*1000,
            preprocess_ms=(t_prep-begin)*1000, vision_ms=(t_vision-t_prep)*1000,
            selector_ms=(t_select-t_vision)*1000, bias_setup_ms=(t_setup-t_select)*1000,
            prefill_and_first_choice_ms=(ttft-t_setup)*1000,
            decode_after_first_choice_ms=(end-ttft)*1000, generated_tokens=len(ids),
            n_vis=st.n_vis)


def main(eng=None):
    freeze = verify_freeze(OUT / "data/manifest.json")
    dev_path = OUT / "data/development_manifest.json"
    if sha256(dev_path) != freeze["development_manifest_sha256"]:
        raise RuntimeError("development manifest changed after freeze")
    item = json.loads(dev_path.read_text())["items"][0]
    if sha256(item["image_path"]) != item["image_sha256"]:
        raise RuntimeError("development image changed after data freeze")
    if eng is None:
        model = common.load_model(common.BASELINE_MODEL, max_new_tokens=MAX_TOKENS)
        eng = NativeEngine(model)
    if eng.text.config._attn_implementation != "sdpa":
        raise RuntimeError("timing backend differs from frozen SDPA")
    result = dict(scope="one outside-panel development page, batch 1, model load excluded",
        dev_item_id=item["item_id"], dev_image_sha256=item["image_sha256"],
        seed=SEED, warmups_per_arm=3, paired_blocks=10, fixed_decode_tokens=16,
        run_manifest_sha256=sha256(OUT / "run_manifest.json"),
        selection="fresh preprocess, vision, selector, bias, prefill and first choice in each timed call",
        arms=list(ARMS), blocks=[], complete=False)
    for arm in ARMS:
        for rep in range(3):
            measure(eng, item, arm)
        print(f"warmup complete: {arm}", flush=True)
    for block in range(10):
        offset = block % len(ARMS)
        order = list(ARMS[offset:]) + list(ARMS[:offset])
        times = {}
        for arm in order:
            times[arm] = measure(eng, item, arm)
        result["blocks"].append(dict(block=block, order=order, arms=times))
        save_json(OUT / "latency.json", result)
        print(f"timing block {block+1}/10 complete", flush=True)
    result["complete"] = True
    save_json(OUT / "latency.json", result)


if __name__ == "__main__":
    main()
