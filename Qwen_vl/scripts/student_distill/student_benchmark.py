"""Fresh end-to-end forward-only timings on the frozen development document."""
from __future__ import annotations
from contextlib import nullcontext
import json
from statistics import median
import time
import torch
from student_eval import (OUT, K, MAX_TOKENS, NativeEngine, common, resolution,
    message, random_keep, select_student, load_students, verify_freeze, verify_runtime_model, sha256, save_json,
    TEACHERS, DEV_MANIFEST, FREEZE, RANDOM_SEED)

ARMS = ('full', 'rres512', 'b2', 'random') + tuple(f'{x}__seed_11' for x in TEACHERS)
WARMUPS = 3
BLOCKS = 10
FIXED_TOKENS = 16


def clock():
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def measure(eng, image_path, question, item_id, arm, students):
    if arm not in ARMS: raise ValueError('benchmark only accepts forward-only arms')
    scope = resolution(eng, 512) if arm == 'rres512' else nullcontext()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    with scope:
        begin = clock()
        msg = message(image_path, question)
        prep = eng.prepare(msg, 'DocVQA_VAL')
        prepped = clock()
        expected_input = K if arm == 'rres512' else 1024
        if prep['n_vis'] != expected_input: raise ValueError('unexpected visual token count')
        V, DS = eng.encode(prep)
        vision = clock()
        if arm in ('full', 'rres512'):
            keep = None
        elif arm == 'b2':
            tm, ts = eng.instruction_embeds(msg, 'DocVQA_VAL')
            keep = eng.select('b2', K, dict(prep=prep, V=V, DS=DS, K=K, engine=eng,
                                           text_mean=tm, text_seq=ts, seed=RANDOM_SEED))
        elif arm == 'random':
            _, keep = random_keep(item_id)
        else:
            q, _ = eng.instruction_embeds(msg, 'DocVQA_VAL')
            keep, _ = select_student(students[arm], V, DS, q)
        selected = clock()
        state = eng.prefill(prep, V, DS, keep, deepstack=True)
        first = int(state.logits.argmax(dim=-1).item())
        ttft = clock()
        ids, _ = eng.decode(state, FIXED_TOKENS, ignore_eos=True)
        end = clock()
    actual = 1024 if arm == 'full' else K
    if state.keep_idx.numel() != actual or state.ds_lengths != [actual]*3:
        raise AssertionError('physical K or DS length mismatch')
    if len(ids) != FIXED_TOKENS or first != ids[0] or len(state.decode_positions) != FIXED_TOKENS-1:
        raise AssertionError('fixed continuation bookkeeping mismatch')
    if state.decode_positions[0] != state.prefill_max_pos+1:
        raise AssertionError('decode 3D position drift')
    if int(state.cache.get_seq_length()) != state.lk + len(state.decode_positions):
        raise AssertionError('cache length mismatch')
    return dict(ttft_ms=1000*(ttft-begin), total_16_ms=1000*(end-begin),
        preprocess_ms=1000*(prepped-begin), vision_ms=1000*(vision-prepped),
        question_and_selector_ms=1000*(selected-vision),
        prefill_and_choice_ms=1000*(ttft-selected), decode_ms=1000*(end-ttft),
        baseline_allocated_bytes=baseline_allocated, baseline_reserved_bytes=baseline_reserved,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        generated_tokens=len(ids), actual_visual_count=actual, ds_lengths=state.ds_lengths,
        prefill_length=state.lk, cache_length=int(state.cache.get_seq_length()),
        first_token_id=first, fixed_ignore_eos=True)


def summarize(blocks):
    keys = ('ttft_ms','total_16_ms','preprocess_ms','vision_ms',
            'question_and_selector_ms','prefill_and_choice_ms','decode_ms',
            'baseline_allocated_bytes','baseline_reserved_bytes','peak_allocated_bytes','peak_reserved_bytes')
    return {arm:{key:dict(median=median(values),minimum=min(values),maximum=max(values),samples=values)
                 for key in keys for values in [[b['arms'][arm][key] for b in blocks]]} for arm in ARMS}


def main():
    frozen = verify_freeze()
    dev = json.loads(DEV_MANIFEST.read_text())
    if dev['role'] != 'dev' or len(dev['items']) != 1: raise ValueError('expected one dev page')
    item = dev['items'][0]
    if sha256(item['image_path']) != item['image_sha256']: raise RuntimeError('dev image changed')
    output = OUT / 'latency.json'
    config = dict(eval_freeze_sha256=sha256(FREEZE), dev_manifest_sha256=sha256(DEV_MANIFEST),
        dev_item_id=item['item_id'],dev_image_sha256=item['image_sha256'],arms=list(ARMS),
        warmups_per_arm=WARMUPS,blocks=BLOCKS,fixed_decode_tokens=FIXED_TOKENS,
        scope='one development page; batch 1; load excluded; all arms fresh preprocessing and vision',
        limitations=['One page does not establish a benchmark-wide latency distribution.',
                     'All 12 students are resident in GPU memory in every arm; selected scorer executes in student arms only.',
                     'Offline gold teacher and student training are excluded and must be separately reported.'])
    if output.exists():
        result = json.loads(output.read_text())
        if result['config'] != config: raise RuntimeError('timing resume configuration mismatch')
    else: result = dict(config=config, blocks=[], complete=False)
    eng = NativeEngine(common.load_model(common.BASELINE_MODEL, max_new_tokens=MAX_TOKENS))
    runtime_model = verify_runtime_model(eng)
    models = load_students(frozen, next(eng.model.parameters()).device)
    result['environment'] = dict(gpu_name=torch.cuda.get_device_name(),
        device=str(next(eng.model.parameters()).device),dtype=str(next(eng.model.parameters()).dtype),
        attention_backend=eng.text.config._attn_implementation,
        allocated_with_all_students_bytes=torch.cuda.memory_allocated(),
        reserved_with_all_students_bytes=torch.cuda.memory_reserved())
    result['environment']['runtime_model'] = runtime_model
    for arm in ARMS:
        for _ in range(WARMUPS): measure(eng,item['image_path'],item['question'],item['item_id'],arm,models)
        print('timing warmup complete:',arm,flush=True)
    for block in range(len(result['blocks']), BLOCKS):
        offset = block % len(ARMS)
        order = list(ARMS[offset:]) + list(ARMS[:offset])
        times = {arm:measure(eng,item['image_path'],item['question'],item['item_id'],arm,models) for arm in order}
        result['blocks'].append(dict(block=block, order=order, arms=times))
        save_json(output,result)
        print(f'timing block {block+1}/{BLOCKS}',flush=True)
    result['summary'] = summarize(result['blocks']); result['complete'] = True
    save_json(output,result)

if __name__ == '__main__': main()
