"""DocVQA full EADP K256 DS-on run with light performance hooks.

Each completed record is flushed before its prior-run comparisons are checked.
The resume protocol accepts only a contiguous, newline-terminated prefix. A
damaged tail needs manual preservation and repair before resuming.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback
from pathlib import Path

R = Path('/media/disk2/YZX/research/EADP_v2')
Q = R / 'reproduction/20261028/qwen'
OUT = Path(__file__).resolve().parent
OLD = Q / 'gpu_formula_facility_panel_run02/run_plan.json'
MANIFEST = Q / 'doc_full_two_routes/input_manifest.jsonl'
PAIRED = Q / 'doc_full_k256_ds_pair/execution'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def rows(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def save(path, obj):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def completed_prefix(path, manifest):
    if not path.exists():
        return 0
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        raise ValueError('incomplete JSONL tail; preserve it and repair explicitly')
    lines = data.splitlines()
    if len(lines) > len(manifest):
        raise ValueError('more records than manifest')
    for i, line in enumerate(lines):
        v = json.loads(line)
        if v['position'] != i or str(v['index']) != str(manifest[i]['index']):
            raise ValueError('non-contiguous or mismatched checkpoint at %d' % i)
        if v.get('prior_selection_match') is not True or v.get('injected') != 1:
            raise ValueError('failed selection or injection at %d; preserve and investigate' % i)
        if i < 16 and not (v.get('prior_output_ids_match') is True and v.get('prior_response_match') is True):
            raise ValueError('failed first-16 output check at %d; preserve and investigate' % i)
    return len(lines)


def main(resume):
    old = json.loads(OLD.read_text())
    manifest = rows(MANIFEST)
    assert len(manifest) == 5349 and [r['position'] for r in manifest] == list(range(5349))
    predictions = rows(PAIRED / 'ds_on.ordered.predictions.jsonl')
    observations = rows(PAIRED / 'ds_on.ordered.observations.jsonl')
    assert len(predictions) == len(observations) == len(manifest)
    for i, (m, p, o) in enumerate(zip(manifest, predictions, observations)):
        assert p['position'] == o['position'] == i and str(m['index']) == str(p['index']) == str(o['index'])
    output = OUT / 'observations.jsonl'
    start = completed_prefix(output, manifest)
    if start and not resume:
        raise ValueError('existing checkpoint; use --resume after inspecting previous attempt')
    if start == len(manifest):
        raise ValueError('all rows already present; audit rather than rerun')
    identity = dict(code_sha256=sha(__file__), manifest_sha256=sha(MANIFEST),
                    old_plan_sha256=sha(OLD), prior_prediction_sha256=sha(PAIRED / 'ds_on.ordered.predictions.jsonl'),
                    prior_observation_sha256=sha(PAIRED / 'ds_on.ordered.observations.jsonl'))
    if (OUT / 'inputs.json').exists():
        assert json.loads((OUT / 'inputs.json').read_text()) == identity, 'inputs or code changed since initial attempt'
    else:
        assert not start
        save(OUT / 'inputs.json', identity)
    for path, digest in old['source_sha256'].items():
        assert sha(path) == digest, path
    for path, digest in old['implementation_sha256'].items():
        assert sha(path) == digest, path
    sys.path.insert(0, str(OLD.parent))
    from namespace_guards import check_model_stats
    from diagnose import stat
    check_model_stats(old, stat, 'container')
    images = {r['image_path']: r['image_sha256'] for r in manifest}
    assert len(images) == 1286
    for path, digest in images.items():
        assert sha(path) == digest, path
    sys.path[:0] = [old['author_root'], str(Path(old['author_root']) / 'VLMEvalKit')]
    import torch
    import model.pruner as native
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatCDPruner, Qwen3VLChatEADP
    assert {n: importlib.metadata.version(n) for n in old['expected_versions']} == old['expected_versions']
    torch.set_num_threads(4)
    assert torch.cuda.get_device_name(0) == 'NVIDIA A40'
    load_start = time.perf_counter()
    w = Qwen3VLChatEADP(**old['recipe'])
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    assert w.visual_token_num == w.pruner.visual_token_num == 256
    assert w.model.config._attn_implementation == w.model.visual.config._attn_implementation == 'flash_attention_2'
    assert w.model.dtype == torch.bfloat16 and {str(p.device) for p in w.model.parameters()} == {'cuda:0'}
    assert w.post_process is False
    w.set_dump_image(lambda row: [row['image_path']])
    method = Qwen3VLChatCDPruner.generate_inner_transformers_fixed
    lang = w.model.model.language_model
    runtime = dict(attempt_start_position=start, recipe=old['recipe'], torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0), dtype=str(w.model.dtype), attention='flash_attention_2',
        code_inputs=identity, pid=os.getpid(), model_instances=1, model_load_seconds=load_seconds,
        wall_scope='synchronized image read, processor, visual tower, selector, generation, decode; excludes model load and validation',
        prefill_scope='CUDA event first language_model forward including DeepStack gather/injection; excludes visual tower and selector',
        memory_scope='PyTorch peak allocated/reserved whole inference including model; excludes driver and other allocators',
        warmup='one natural-output question per attempt, not in measured output', gt_read=False)
    save(OUT / ('runtime_%d.json' % os.getpid()), runtime)

    def run(row):
        msg = w._build_vqa_prompt(dict(row), 'DocVQA_VAL')
        full = w._build_messages(msg, dataset='DocVQA_VAL')
        assert hashlib.sha256(json.dumps(full, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == row['message_sha256']
        state = {'calls': 0, 'injected': 0}
        handles = []
        original_select = native._greed_select_impl
        original_processor, original_generate = w._processor_inputs, w.model.generate

        def choose(*a, **k):
            result = original_select(*a, **k)
            state['selected'] = result[0]
            return result

        def processor(*a, **k):
            x = original_processor(*a, **k)
            pos = (x['input_ids'][0] == w.model.config.image_token_id).nonzero(as_tuple=True)[0]
            assert pos.numel() == 1024
            first_pos, last_pos = int(pos[0].item()), int(pos[-1].item())
            assert last_pos - first_pos == 1023
            state['start'] = first_pos
            return x

        def visual(module, args, result):
            state['deepstack'] = result[1]

        def generate(*a, **k):
            result = original_generate(*a, **k)
            state['output'] = result
            return result

        first = torch.cuda.Event(enable_timing=True)
        last = torch.cuda.Event(enable_timing=True)

        def inject(module, args, kwargs):
            state['calls'] += 1
            if state['calls'] != 1:
                return None
            first.record()
            assert kwargs.get('deepstack_visual_embeds') is None and kwargs.get('visual_pos_masks') is None
            ids = state['selected'][0].sort().values
            x = kwargs['inputs_embeds']
            mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
            mask[:, state['start']:state['start'] + 256] = True
            kwargs = dict(kwargs)
            kwargs['visual_pos_masks'] = mask
            kwargs['deepstack_visual_embeds'] = [v.index_select(0, ids) for v in state['deepstack']]
            assert len(kwargs['deepstack_visual_embeds']) == 3
            state['injected'] += 1
            return args, kwargs

        def post(module, args, kwargs, result):
            if state['calls'] == 1:
                last.record()

        try:
            native._greed_select_impl = choose
            w._processor_inputs, w.model.generate = processor, generate
            handles.append(w.model.visual.register_forward_hook(visual))
            handles.append(lang.register_forward_pre_hook(inject, with_kwargs=True))
            handles.append(lang.register_forward_hook(post, with_kwargs=True))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            with torch.no_grad():
                response = method(w, msg, dataset='DocVQA_VAL')
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
            prefill = first.elapsed_time(last)
            ordered = state['selected'][0].detach().cpu().tolist()
            ids = state['output'][0].detach().cpu().tolist()
            assert len(ordered) == 256 and len(set(ordered)) == 256 and ids and state['injected'] == 1
            assert min(ordered) >= 0 and max(ordered) < 1024
            eos = w.model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else list(eos or [])
            return dict(position=row['position'], index=row['index'], seconds=elapsed, prefill_gpu_ms=prefill,
                        peak_allocated_bytes=allocated, peak_reserved_bytes=reserved, output_tokens=len(ids),
                        output_ids=ids, response=response, ordered_ids=ordered, prefill_calls=state['calls'],
                        injected=state['injected'], eos_present=any(x in eos for x in ids),
                        at_max_tokens=len(ids) == old['recipe']['max_new_tokens'])
        finally:
            native._greed_select_impl = original_select
            w._processor_inputs, w.model.generate = original_processor, original_generate
            for h in handles:
                h.remove()

    warm = run(manifest[start])
    warm['prior_response_match'] = warm['response'] == predictions[start]['response']
    warm['prior_output_ids_match'] = warm['output_ids'] == observations[start]['output_ids']
    warm['prior_selection_match'] = warm['ordered_ids'] == observations[start]['ordered_ids']
    save(OUT / ('warmup_%d.json' % os.getpid()), warm)
    assert warm['prior_selection_match'] and warm['prior_response_match'] and warm['prior_output_ids_match'], 'warmup differs from prior experiment'
    print('warmup complete; resuming at', start, flush=True)
    with output.open('a', encoding='utf-8') as f:
        for i in range(start, len(manifest)):
            try:
                r = run(manifest[i])
                r['prior_response_match'] = r['response'] == predictions[i]['response']
                r['prior_output_ids_match'] = r['output_ids'] == observations[i]['output_ids']
                r['prior_selection_match'] = r['ordered_ids'] == observations[i]['ordered_ids']
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
                if not r['prior_selection_match']:
                    raise RuntimeError('selection differs from prior experiment at position %d' % i)
                if i < 16 and not (r['prior_response_match'] and r['prior_output_ids_match']):
                    raise RuntimeError('first-16 output differs from prior experiment at position %d' % i)
                if i == 15:
                    print('first 16 selection and output checks passed', flush=True)
                if (i + 1) % 50 == 0:
                    print('completed', i + 1, '/', len(manifest), flush=True)
            except BaseException as exc:
                save(OUT / ('failure_%d.json' % os.getpid()), dict(position=i, index=manifest[i]['index'],
                    error=repr(exc), traceback=traceback.format_exc(),
                    recorded_lines=sum(1 for _ in output.open(encoding='utf-8'))))
                raise
    save(OUT / 'complete.json', dict(n=len(manifest), observations_sha256=sha(output), inputs=identity))
    print('complete', len(manifest), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    main(args.resume)
