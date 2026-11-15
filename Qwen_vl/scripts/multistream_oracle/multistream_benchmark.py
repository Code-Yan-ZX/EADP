"""Paired single-development-page timing of the seven forward-only references.

No gold tokenization, gradient teacher, oracle ranking or teacher-selected set
is evaluated here. Each call pays for its own preprocessing, native vision and
applicable forward selector. DS-off disables decoder injections while keeping
the native base visual sequence and the same vision encoding computation.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
import time

import torch

from experiment_common import (HERE, OUT, K, MAX_TOKENS, SEED, NativeEngine,
    common, message, random_keep, resolution, save_json, sha256)


REFERENCE_ARMS=("full","full_no_ds","rres512","b2","b2_no_ds",
                "random_on","random_off")
DS_OFF_ARMS=frozenset(("full_no_ds","b2_no_ds","random_off"))
FIXED_TOKENS=16
WARMUPS=3
PAIRED_BLOCKS=10


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def measure(eng,item,arm):
    if arm not in REFERENCE_ARMS:
        raise ValueError("timing accepts only frozen forward-only reference arms")
    deepstack=arm not in DS_OFF_ARMS
    size=512 if arm=="rres512" else 1024
    scope=resolution(eng,size) if size==512 else nullcontext()
    with scope:
        begin=sync_time()
        msg=message(item)
        prep=eng.prepare(msg,"DocVQA_VAL")
        t_prep=sync_time()
        expected_input=K if arm=="rres512" else 1024
        if prep["n_vis"]!=expected_input:
            raise AssertionError("timing input visual-token count differs from protocol")
        V,DS=eng.encode(prep)
        t_vision=sync_time()
        keep=None
        random_seed=None
        if arm in ("b2","b2_no_ds"):
            text_mean,text_seq=eng.instruction_embeds(msg,"DocVQA_VAL")
            context=dict(prep=prep,V=V,DS=DS,K=K,engine=eng,
                         text_mean=text_mean,text_seq=text_seq,seed=SEED)
            keep=eng.select("b2",K,context)
        elif arm in ("random_on","random_off"):
            random_seed,keep=random_keep(item["item_id"],n=expected_input,k=K)
            keep=keep.to(V.device)
        t_select=sync_time()
        state=eng.prefill(prep,V,DS,keep,deepstack=deepstack)
        first=int(state.logits.argmax(dim=-1).item())
        first_time=sync_time()
        ids,_=eng.decode(state,FIXED_TOKENS,ignore_eos=True)
        end=sync_time()
    actual=state.keep_idx.numel()
    expected=1024 if arm in ("full","full_no_ds") else K
    if actual!=expected or state.ds_lengths!=([expected]*3 if deepstack else None):
        raise AssertionError("timing changed physical visual rows or DeepStack lengths")
    if len(ids)!=FIXED_TOKENS or ids[0]!=first:
        raise AssertionError("timing did not execute the fixed 16-token continuation")
    if int(state.cache.get_seq_length())!=state.lk+len(state.decode_positions):
        raise AssertionError("timing cache length is inconsistent")
    if len(state.decode_positions)!=FIXED_TOKENS-1:
        raise AssertionError("fixed-token decode used an unexpected number of model calls")
    if state.decode_positions[0]!=state.prefill_max_pos+1:
        raise AssertionError("timing decode position drift")
    return dict(ttft_ms=(first_time-begin)*1000,total_16_ms=(end-begin)*1000,
        preprocess_ms=(t_prep-begin)*1000,vision_ms=(t_vision-t_prep)*1000,
        selector_ms=(t_select-t_vision)*1000,
        prefill_and_first_choice_ms=(first_time-t_select)*1000,
        decode_after_first_choice_ms=(end-first_time)*1000,
        generated_tokens=len(ids),fixed_ignore_eos=True,deepstack=deepstack,
        actual_visual_count=actual,native_state_n_vis=state.n_vis,
        ds_lengths=state.ds_lengths,random_seed=random_seed,
        first_token_id=first,prefill_length=state.lk,
        cache_length=int(state.cache.get_seq_length()))


def summarize(blocks):
    result={}
    for arm in REFERENCE_ARMS:
        values={key:[block["arms"][arm][key] for block in blocks]
                for key in ("ttft_ms","total_16_ms","preprocess_ms","vision_ms",
                            "selector_ms","prefill_and_first_choice_ms",
                            "decode_after_first_choice_ms")}
        result[arm]={key:dict(median=float(torch.tensor(samples,dtype=torch.float64).median())
                             if len(samples)%2 else float(torch.tensor(samples,dtype=torch.float64).quantile(.5)),
                             minimum=min(samples),maximum=max(samples),samples=samples)
                     for key,samples in values.items()}
    return result


def main(eng=None):
    # Import lazily: this file can be CPU-inspected while the runner is built.
    from multistream_run import verify_freeze

    frozen=verify_freeze(OUT/"data/manifest.json")
    dev_path=OUT/"data/development_manifest.json"
    if sha256(dev_path)!=frozen["development_manifest_sha256"]:
        raise RuntimeError("development manifest differs from run freeze")
    manifest=json.loads(dev_path.read_text())
    if manifest.get("role")!="development" or len(manifest["items"])!=1:
        raise ValueError("timing requires the one frozen independent development page")
    item=manifest["items"][0]
    if sha256(item["image_path"])!=item["image_sha256"]:
        raise RuntimeError("development image changed after data freeze")
    if eng is None:
        model=common.load_model(common.BASELINE_MODEL,max_new_tokens=MAX_TOKENS)
        eng=NativeEngine(model)
    if eng.text.config._attn_implementation!="sdpa":
        raise RuntimeError("timing backend differs from frozen SDPA")
    if eng.model.training:
        raise RuntimeError("timing requires evaluation mode")
    result=dict(scope="one outside-panel development page, batch 1, model load excluded",
        dev_item_id=item["item_id"],dev_image_sha256=item["image_sha256"],
        run_manifest_sha256=sha256(OUT/"run_manifest.json"),
        benchmark_source_sha256=sha256(HERE/"multistream_benchmark.py"),
        seed=SEED,warmups_per_arm=WARMUPS,paired_blocks=PAIRED_BLOCKS,
        fixed_decode_tokens=FIXED_TOKENS,arms=list(REFERENCE_ARMS),blocks=[],complete=False,
        device=str(next(eng.model.parameters()).device),dtype=str(next(eng.model.parameters()).dtype),
        gpu_name=torch.cuda.get_device_name(),attention_backend=eng.text.config._attn_implementation,
        selection="fresh preprocess, vision, applicable selector, native prefill and first choice in every timed call",
        limitations=["Single-page timing does not establish benchmark-wide efficiency.",
                     "Teacher-selected sets and all gold/backward costs are excluded from deployable timing claims.",
                     "DS-off keeps visual tokens; native_state_n_vis may misleadingly report zero, so physical count is separate."])
    for arm in REFERENCE_ARMS:
        for _ in range(WARMUPS):
            measure(eng,item,arm)
        print("reference warm-up complete: "+arm,flush=True)
    for block in range(PAIRED_BLOCKS):
        offset=block%len(REFERENCE_ARMS)
        order=list(REFERENCE_ARMS[offset:])+list(REFERENCE_ARMS[:offset])
        times={arm:measure(eng,item,arm) for arm in order}
        result["blocks"].append(dict(block=block,order=order,arms=times))
        save_json(OUT/"latency.json",result)
        print(f"reference timing block {block+1}/{PAIRED_BLOCKS} complete",flush=True)
    result["summary"]=summarize(result["blocks"])
    result["complete"]=True
    save_json(OUT/"latency.json",result)


if __name__=="__main__":
    main()
