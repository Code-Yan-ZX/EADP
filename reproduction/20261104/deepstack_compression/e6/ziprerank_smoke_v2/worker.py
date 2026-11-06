"""Bounded diagnostic of the unmodified author's QI-EI generation path.

One frozen DEV image, no labels/metrics. Failures are recorded per arm; this
worker is not an evaluation or proof of general correctness.
"""
import argparse
import hashlib
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/media/disk2/YZX/research/EADP_v2')
BASE = ROOT / 'reproduction/20261104/deepstack_compression'
HERE = BASE / 'e6/ziprerank_smoke_v2'
ZIP = BASE / 'e6/external_sources/ziprerank_f367b09'
AUTHOR = ROOT / '_reproduction/author_eadp/Qwen_vl/VLMEvalKit'
MODEL = Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
sys.path[:0] = [str(AUTHOR), str(ZIP)]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    with open(path, 'x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def weight_stats(weights):
    result = []
    for item in weights['files']:
        path = Path(item['path'])
        stat = path.stat()
        if stat.st_size != item['bytes'] or item['sha256'] != item['expected_lfs_sha256']:
            raise ValueError('weight identity/size mismatch')
        result.append({'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                       'inode': stat.st_ino, 'device': stat.st_dev})
    if len(result) != 4:
        raise ValueError('unexpected weight count')
    return result


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(__import__('torch').uint8).cpu().numpy().tobytes()).hexdigest()


def cache_summary(cache, *, fingerprint_prefix=None):
    """Read all layer shapes and optional prefix content fingerprints."""
    if cache is None:
        return []
    result = []
    for layer in cache.layers:
        k, v = layer.keys, layer.values
        if k is None or v is None:
            result.append({'key_shape': None if k is None else list(k.shape),
                           'value_shape': None if v is None else list(v.shape)})
            if fingerprint_prefix is not None:
                raise ValueError('uninitialized cache layer')
            continue
        record = {'key_shape': list(k.shape), 'value_shape': list(v.shape)}
        if fingerprint_prefix is not None:
            if k.shape[-2] < fingerprint_prefix or v.shape[-2] < fingerprint_prefix:
                raise ValueError('short cache prefix')
            record['key_prefix_sha256'] = tensor_sha(k[..., :fingerprint_prefix, :])
            record['value_prefix_sha256'] = tensor_sha(v[..., :fingerprint_prefix, :])
        result.append(record)
    return result


def diagnostic_checks(arm, layer_count):
    """Fail closed on missing calls, incorrect mRoPE or inconsistent per-layer KV."""
    checks = {}
    trace = arm['language_model_calls']
    logits = arm['logits_trace']
    prefill = arm['prefill_checks']
    expected_prefill = arm['original_sequence_length'] - (768 if arm['name'] == 'keep_256' else 0)
    checks['generated_33_forced_tokens'] = arm.get('status') == 'completed' and arm.get('forced_tokens_match') is True and arm.get('generated_token_count') == 33
    checks['one_prefill_plus_32_decode_calls'] = len(trace) == len(logits) == 33
    checks['all_logits_finite'] = len(logits) == 33 and all(x['finite'] for x in logits)
    checks['prefill_inputs_and_three_ds_match'] = (
        prefill.get('positions_match') is True and prefill.get('image_main_matches') is True
        and prefill.get('text_embeddings_match') is True and prefill.get('ds_matches') == [True] * 3
        and prefill.get('ds_count') == prefill.get('expected_ds_count') == 3
        and prefill.get('selected_unique') is True and prefill.get('selected_in_range') is True
        and prefill.get('selected_count') == (256 if arm['name'] == 'keep_256' else 1024))
    checks['decode_positions'] = len(trace) == 33 and all(
        x.get('position_ids') == [[[arm['original_max_position'] + i]]] * 3
        for i, x in enumerate(trace) if i > 0)
    checks['all_layer_cache_lengths'] = len(logits) == 33 and all(
        len(row['all_layer_cache_after']) == layer_count and all(
            isinstance(z.get('key_shape'), list) and isinstance(z.get('value_shape'), list)
            and z['key_shape'][-2] == expected_prefill + i and z['value_shape'][-2] == expected_prefill + i
            for z in row['all_layer_cache_after']) for i, row in enumerate(logits))
    fingerprints = arm['cache_fingerprints']
    checks['all_layer_prefix_unchanged'] = (
        len(fingerprints) == 2 and [x['call_index'] for x in fingerprints] == [0, 32]
        and len(fingerprints[0]['layers']) == len(fingerprints[1]['layers']) == layer_count
        and all(a.get('key_prefix_sha256') == b.get('key_prefix_sha256')
                and a.get('value_prefix_sha256') == b.get('value_prefix_sha256')
                for a, b in zip(fingerprints[0]['layers'], fingerprints[1]['layers'])))
    if arm['name'] == 'keep_all':
        compared = arm['keep_all_vs_disabled_logit_checks']
        checks['keep_all_logits_match_disabled'] = len(compared) == 33 and all(
            x.get('comparable') is True and x.get('allclose_atol_0.01_rtol_0.01') is True
            and x.get('argmax_equal') is True for x in compared)
    return checks


def preflight(plan):
    for path, expected in plan['sha256'].items():
        if sha(path) != expected:
            raise ValueError('source mismatch: ' + path)
    rows = [json.loads(x) for x in (BASE / 'e1/input_preparation_002/inputs.jsonl').read_text().splitlines()]
    anchor = plan['anchor']
    matches = [r for r in rows if r['task'] == anchor['task'] and str(r['qid']) == anchor['qid']]
    if len(matches) != 1:
        raise ValueError('anchor uniqueness')
    row = matches[0]
    if row['message_sha256'] != anchor['message_sha256'] or sha(row['image_path']) != anchor['image_sha256']:
        raise ValueError('anchor fingerprint mismatch')
    if [x['type'] for x in row['message']] != ['image', 'text']:
        raise ValueError('message order mismatch')
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((HERE / 'plan001.json').read_text())
    row = preflight(plan)
    import torch
    import transformers
    from transformers import AutoProcessor, LogitsProcessor
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
    from models.qwen3vl_with_qi_early import Qwen3VLWithQIEarly
    if Path(inspect.getfile(Qwen3VLWithQIEarly)).resolve() != ZIP / 'models/qwen3vl_with_qi_early.py':
        raise ValueError('unexpected author class')
    weights = json.loads((ROOT / 'reproduction/20261028/qwen/qwen_weight_identity.json').read_text())
    stats = weight_stats(weights)
    if args.preflight_only:
        if torch.cuda.is_initialized():
            raise ValueError('CPU preflight initialized CUDA')
        for item in weights['files']:
            if sha(item['path']) != item['expected_lfs_sha256']:
                raise ValueError('weight SHA mismatch')
        if weight_stats(weights) != stats:
            raise ValueError('weight changed during hashing')
        save(args.out, {'status': 'passed', 'cuda_initialized': False, 'model_loaded': False,
                        'GT_opened': False, 'source_plan_sha256': sha(HERE / 'plan001.json'),
                        'torch': torch.__version__, 'transformers': transformers.__version__,
                        'weight_sha256_verified': 4, 'weight_stats': stats})
        return
    previous = json.loads((HERE / 'preflight001.json').read_text())
    if stats != previous['weight_stats']:
        raise ValueError('weight stats changed after full SHA preflight')
    if torch.cuda.get_device_name(0) != 'NVIDIA A40':
        raise ValueError('unexpected GPU')
    args.out.mkdir(exist_ok=False)
    save(args.out / 'started.json', {'plan_sha256': sha(HERE / 'plan001.json'), 'task': row['task'],
                                    'qid': row['qid'], 'GT_opened': False, 'scoring': False})
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = Qwen3VLWithQIEarly.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.bfloat16,
                                               attn_implementation='flash_attention_2').to('cuda').eval()
    wrapper = object.__new__(Qwen3VLChatFixedRes)
    wrapper.system_prompt = None
    wrapper.min_pixels = wrapper.max_pixels = 1048576
    wrapper.total_pixels = None
    wrapper.processor = processor
    wrapper.model = SimpleNamespace(device='cuda', dtype=torch.bfloat16)
    inputs = wrapper._processor_inputs(wrapper._build_messages(row['message'], dataset=row['task']))
    ids = inputs['input_ids']
    original_positions, rope_deltas = model.model.get_rope_index(ids, inputs['image_grid_thw'], None, inputs['attention_mask'])
    seq_len = ids.shape[1]
    image_positions = torch.where(ids[0] == model.config.image_token_id)[0]
    if len(image_positions) != 1024:
        raise ValueError('expected exactly 1024 image tokens')
    forced_id = processor.tokenizer.encode(' the', add_special_tokens=False)[0]

    class ForceToken(LogitsProcessor):
        def __call__(self, input_ids, scores):
            forced = torch.full_like(scores, -float('inf'))
            forced[:, forced_id] = 0
            return forced

    original_text_embeds = model.get_input_embeddings()(ids).detach()
    summaries = []
    baseline_logits = []
    for name, enabled, ratio in [('disabled', False, 1.0), ('keep_all', True, 1.0), ('keep_256', True, .25)]:
        model.model.rope_deltas = None
        model.set_qi_early_enabled(enabled)
        model.set_qi_early_keep_ratio(ratio)
        trace = []
        selected = []
        logit_checks = []
        logits_trace = []
        vision_capture = []
        cache_fingerprints = []
        prefill_checks = {}
        save(args.out / (name + '_started.json'), {'arm': name, 'enabled': enabled, 'ratio': ratio})
        def vision_hook(_module, _args, output):
            vision_capture.append((output[0].detach(), [x.detach() for x in output[1]]))
        def pre_lm(_module, _args, kwargs):
            pos = kwargs.get('position_ids')
            emb = kwargs.get('inputs_embeds')
            mask = kwargs.get('visual_pos_masks')
            ds = kwargs.get('deepstack_visual_embeds')
            cache = kwargs.get('past_key_values')
            if not trace:
                if (enabled and len(selected) != 1) or len(vision_capture) != 1 or emb is None or ds is None or mask is None:
                    raise ValueError('missing selection/vision/prefill capture')
                picked = selected[0] if enabled else list(range(1024))
                idx = torch.tensor(picked, dtype=torch.long, device=ids.device)
                keep = torch.ones(seq_len, dtype=torch.bool, device=ids.device)
                keep[image_positions] = False
                keep[image_positions[idx]] = True
                expected_pos = original_positions[:, :, keep]
                nonvision = ~mask[0].bool()
                original_nonvision = ids[0] != model.config.image_token_id
                visual, original_ds = vision_capture[0]
                prefill_checks.update(
                    positions_match=pos is not None and bool(torch.equal(pos, expected_pos)),
                    image_main_matches=bool(torch.equal(emb[0, mask[0].bool()], visual[idx].to(emb.dtype))),
                    text_embeddings_match=bool(torch.equal(emb[0, nonvision], original_text_embeds[0, original_nonvision])),
                    ds_matches=[bool(torch.equal(a, b[idx].to(a.dtype))) for a, b in zip(ds, original_ds)],
                    ds_count=len(ds), expected_ds_count=len(original_ds),
                    selected_count=len(picked), selected_unique=len(set(picked)) == len(picked),
                    selected_in_range=all(0 <= x < 1024 for x in picked))
            trace.append({'input_length': None if emb is None else int(emb.shape[1]),
                          'position_ids': None if pos is None else pos.detach().cpu().tolist(),
                          'visual_count': None if mask is None else int(mask.sum().item()),
                          'ds_rows': None if ds is None else [int(x.shape[0]) for x in ds],
                          'cache_length_before': None if cache is None else int(cache.get_seq_length()),
                          'all_layer_cache_before': cache_summary(cache),
                          'cache_position': None if kwargs.get('cache_position') is None else kwargs['cache_position'].detach().cpu().tolist()})
        def pruner_hook(_module, _args, output):
            selected.extend(x.detach().cpu().tolist() for x in output[1])
        def model_hook(_module, _args, output):
            value = output.logits[:, -1, :].detach().float().cpu()
            i = len(logits_trace)
            logits_trace.append({'finite': bool(torch.isfinite(value).all()), 'argmax': value.argmax(-1).tolist(),
                                 'rope_deltas': None if output.rope_deltas is None else output.rope_deltas.detach().cpu().tolist(),
                                 'all_layer_cache_after': cache_summary(output.past_key_values)})
            if i in (0, 32) and output.past_key_values is not None:
                expected_prefix = seq_len if not enabled or ratio == 1.0 else seq_len - 768
                cache_fingerprints.append({'call_index': i, 'prefix_length': int(expected_prefix),
                                           'layers': cache_summary(output.past_key_values, fingerprint_prefix=expected_prefix)})
            if name == 'disabled':
                baseline_logits.append(value)
            elif name == 'keep_all':
                i = len(logit_checks)
                if i >= len(baseline_logits) or value.shape != baseline_logits[i].shape:
                    logit_checks.append({'comparable': False})
                else:
                    reference = baseline_logits[i]
                    logit_checks.append({'comparable': True,
                        'allclose_atol_0.01_rtol_0.01': bool(torch.allclose(value, reference, atol=.01, rtol=.01)),
                        'max_abs_difference': float((value - reference).abs().max().item()),
                        'argmax_equal': bool(torch.equal(value.argmax(-1), reference.argmax(-1)))})
        handles = [model.model.language_model.register_forward_pre_hook(pre_lm, with_kwargs=True),
                   model.qi_early_pruner.register_forward_hook(pruner_hook), model.register_forward_hook(model_hook),
                   model.model.visual.register_forward_hook(vision_hook)]
        arm = {'name': name, 'enabled': enabled, 'keep_ratio': ratio}
        try:
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, use_cache=True,
                    max_new_tokens=33, min_new_tokens=33, logits_processor=[ForceToken()],
                    return_dict_in_generate=True)
            torch.cuda.synchronize()
            tokens = generated.sequences[0, seq_len:].tolist()
            save(args.out / (name + '_generated.json'), {'arm': name, 'tokens': tokens, 'forced_token_id': forced_id})
            arm.update(status='completed', generated_token_count=len(tokens), forced_tokens_match=tokens == [forced_id] * 33,
                       cache_length_final=int(generated.past_key_values.get_seq_length()),
                       final_rope_deltas=None if model.model.rope_deltas is None else model.model.rope_deltas.detach().cpu().tolist())
        except Exception as exc:
            arm.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            save(args.out / (name + '_exception.json'), {'arm': name, 'error_type': type(exc).__name__, 'error': str(exc),
                 'traceback': traceback.format_exc()})
        finally:
            for handle in handles:
                handle.remove()
        arm['selected_indices'] = selected
        arm['prefill_checks'] = prefill_checks
        arm['logits_trace'] = logits_trace
        arm['cache_fingerprints'] = cache_fingerprints
        arm['keep_all_vs_disabled_logit_checks'] = logit_checks
        arm['language_model_calls'] = trace
        arm['original_sequence_length'] = int(seq_len)
        arm['original_rope_deltas'] = rope_deltas.detach().cpu().tolist()
        arm['original_max_position'] = int(original_positions.max().item())
        if selected and trace and trace[0]['position_ids'] is not None:
            idx = torch.tensor(selected[0], dtype=torch.long, device=ids.device)
            keep = torch.ones(seq_len, dtype=torch.bool, device=ids.device)
            keep[image_positions] = False
            keep[image_positions[idx]] = True
            expected = original_positions[:, :, keep].cpu().tolist()
            arm['prefill_positions_match_selected_original'] = trace[0]['position_ids'] == expected
        arm['diagnostic_checks'] = diagnostic_checks(arm, model.config.text_config.num_hidden_layers)
        summaries.append(arm)
        save(args.out / (name + '.json'), arm)
    if weight_stats(weights) != stats:
        raise ValueError('weight stats changed during GPU run')
    save(args.out / 'worker_complete.json', {'status': 'diagnostic_completed', 'arms': [x['name'] for x in summaries],
         'arm_statuses': {x['name']: x['status'] for x in summaries}, 'no_GT': True,
         'weight_stats_unchanged': True,
         'warning': 'completion means attempts recorded, not that author paths passed'})


if __name__ == '__main__':
    main()
