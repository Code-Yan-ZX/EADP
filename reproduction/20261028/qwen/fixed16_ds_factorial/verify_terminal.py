"""CPU-only terminal audit of the fixed16 native DeepStack interventions.

This independently reads the actual arrays and prior accepted endpoints. It does
not load a model, run the selector, open labels, tokenize, decode, or score.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import common as c

ARMS = ('plain', 'eadp1024', 'plain_ds_off', 'eadp1024_ds_on')
FIELDS = ('inputs_embeds', 'position_ids', 'attention_mask', 'cache_position',
          'visual_pos_masks', 'deepstack_0', 'deepstack_1', 'deepstack_2')
COMMON = ('visual_features', 'visual_deepstack_0', 'visual_deepstack_1', 'visual_deepstack_2',
          'processor_input_ids', 'processor_attention_mask', 'processor_pixel_values',
          'processor_image_grid_thw', 'prefill_inputs_embeds', 'prefill_position_ids',
          'prefill_attention_mask', 'prefill_cache_position', 'rope_deltas')


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def read_archive(record, path):
    assert Path(record['archive']) == path and c.sha(path) == record['archive_sha256']
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    assert set(data) == set(record['arrays'])
    for name, array in data.items():
        meta = record['arrays'][name]
        assert list(array.shape) == meta['shape']
        assert hashlib.sha256(array.tobytes()).hexdigest() == meta['sha256']
        if meta['dtype'] == 'torch.bfloat16':
            assert array.dtype == np.uint16
            assert np.isfinite((array.astype(np.uint32) << 16).view(np.float32)).all()
        else:
            dtypes = {'torch.bool': np.bool_, 'torch.int64': np.int64,
                      'torch.float16': np.float16, 'torch.float32': np.float32,
                      'torch.float64': np.float64, 'torch.int32': np.int32}
            assert meta['dtype'] in dtypes and array.dtype == dtypes[meta['dtype']], name
            if array.dtype.kind == 'f':
                assert np.isfinite(array).all()
    return data


def check_prefill(data, arm, image_token_id):
    """Check captured original, proposed and final hook arguments bytewise."""
    for prefix in ('prefill_original_', 'prefill_proposed_', 'prefill_'):
        layers = {key for key in data if re.fullmatch(re.escape(prefix) + r'deepstack_\d+', key)}
        expected = {prefix + 'deepstack_' + str(i) for i in range(3)}
        assert not layers or layers == expected
    assert {key for key in data if re.fullmatch(r'visual_deepstack_\d+', key)} == {
        'visual_deepstack_' + str(i) for i in range(3)}
    for field in FIELDS:
        original = data.get('prefill_original_' + field)
        proposed = data.get('prefill_proposed_' + field)
        final = data.get('prefill_' + field)
        assert (proposed is None) == (final is None), field
        if proposed is not None:
            assert same(proposed, final), field
        changed = (arm == 'plain_ds_off' and field in FIELDS[4:]) or (
            arm == 'eadp1024_ds_on' and field in FIELDS[4:])
        if not changed:
            assert (original is None) == (final is None), field
            if original is not None:
                assert same(original, final), field
        elif arm == 'plain_ds_off':
            assert original is not None and final is None, field
        else:
            assert original is None and final is not None, field
    if arm in ('plain', 'plain_ds_off'):
        mask = data['prefill_original_visual_pos_masks']
        assert mask.dtype == np.bool_ and np.array_equal(mask, data['processor_input_ids'] == image_token_id)
        for i in range(3):
            layer = data['prefill_original_deepstack_' + str(i)]
            assert same(layer, data['visual_deepstack_' + str(i)]) and layer.shape[0] == int(mask.sum())
    if arm == 'eadp1024_ds_on':
        mask = data['prefill_visual_pos_masks']
        assert mask.dtype == np.bool_ and same(mask, data['injection_image_mask'])
        assert mask.shape == data['processor_input_ids'].shape
        assert np.array_equal(mask, data['processor_input_ids'] == image_token_id)
        for i in range(3):
            layer = data['prefill_deepstack_' + str(i)]
            assert same(layer, data['visual_deepstack_' + str(i)])
            assert layer.shape[0] == int(mask.sum())


def check_continuation(record, data):
    n = record['continuation_steps']
    assert n == record['llm_calls'] - 1 == len(record['continuation_presence'])
    expected = set()
    for i, presence in enumerate(record['continuation_presence']):
        assert set(presence) == {'visual_mask_present', 'deepstack_present'}
        assert type(presence['visual_mask_present']) is bool and type(presence['deepstack_present']) is bool
        for field in ('position_ids', 'cache_position', 'rope_deltas'):
            key = 'continuation_%04d_%s' % (i, field)
            assert key in data, key
            expected.add(key)
    assert {k for k in data if k.startswith('continuation_')} == expected


def common_prefix_positions(base, variant, base_record, variant_record):
    """Only compare decode steps with the same preceding generated tokens."""
    x = base['output_suffix_ids'].reshape(-1)
    y = variant['output_suffix_ids'].reshape(-1)
    prefix = 0
    for left, right in zip(x, y):
        if left != right:
            break
        prefix += 1
    steps = min(prefix, base_record['continuation_steps'], variant_record['continuation_steps'])
    differing = {field: [] for field in ('position_ids', 'cache_position', 'rope_deltas')}
    for i in range(steps):
        for field in differing:
            name = 'continuation_%04d_%s' % (i, field)
            if not same(base[name], variant[name]):
                differing[field].append(i)
    return prefix, steps, differing


def main():
    plan = c.read(c.PLAN)
    c.check(plan)
    state = c.read(c.HERE / 'state.json')
    assert state['status'] == 'native_exited_validation_pending' and state['native_exit_code'] == 0
    assert state['original_container_exited'] and state['GPU_pid_absent']
    assert state['plan_sha256'] == c.sha(c.PLAN)
    sys.path.insert(0, str(c.OLD))
    from run_panel import identity, gpu_pids
    original = state['container_identity']['host_init']
    current = identity(original['pid'])
    assert current is None or (current['start_ticks'], current['boot_id']) != (original['start_ticks'], original['boot_id'])
    assert original['pid'] not in gpu_pids()
    out = c.HERE / 'execution'
    summary = c.read(out / 'summary.json')
    endpoints = c.read(out / 'endpoints_verified.json')
    assert summary['success'] and summary['n_questions'] == 16 and summary['n_native_generations'] == 64
    assert summary['endpoint_anchors'] == summary['toggles'] == 32 and not summary['GT_read'] and not summary['scoring_started']
    assert endpoints['n_endpoints'] == 32 and endpoints['all_byte_anchors_passed']
    assert summary['plan_sha256'] == endpoints['plan_sha256'] == c.sha(c.PLAN)
    old_plan = c.read(c.OLD / 'run_plan.json')
    model_path = Path(old_plan['recipe']['model_path'])
    model_config = c.read(model_path / 'config.json')
    image_token_id = model_config['image_token_id']
    vocab_size = model_config['text_config']['vocab_size']
    eos_ids = c.read(model_path / 'generation_config.json')['eos_token_id']
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
    results = []
    arrays_checked = 0
    for i, item in enumerate(plan['panel']):
        records, arrays, previous = {}, {}, {}
        for arm in ARMS:
            record = c.read(out / ('%02d_%s.json' % (i, arm)))
            assert record['index'] == item['index'] and record['position'] == item['position'] and record['arm'] == arm
            assert record['status'] == 'raw_preserved' and record['native_failure'] is None and not record['observation_errors']
            assert isinstance(record['response'], str) and not record['GT_or_score_used']
            assert record['vision_calls'] == record['processor_calls'] == record['generate_calls'] == 1
            assert record['llm_calls'] >= 1
            assert record['intervention_applied'] == (arm in ('plain_ds_off', 'eadp1024_ds_on'))
            assert record['intervention_attempts'] == (1 if arm in ('plain_ds_off', 'eadp1024_ds_on') else 0)
            assert record['intervention_error'] is None
            assert record['generate_non_tensor_kwargs'] == dict(do_sample=False, max_new_tokens=2048, top_p=.001,
                top_k=1, temperature=.01, repetition_penalty=1.)
            data = read_archive(record, out / ('%02d_%s.npz' % (i, arm)))
            check_prefill(data, arm, image_token_id)
            check_continuation(record, data)
            for key, expected in item['processor_expected'].items():
                assert record['arrays']['processor_' + key] == expected
            assert same(data['generate_attention_mask'], data['processor_attention_mask'])
            ids, suffix, inputs = data['output_ids'], data['output_suffix_ids'], data['processor_input_ids']
            assert ids.ndim == suffix.ndim == inputs.ndim == 2 and ids.shape[0] == suffix.shape[0] == inputs.shape[0] == 1
            assert suffix.dtype == np.int64 and 0 < suffix.shape[1] <= 2048
            assert np.all((suffix >= 0) & (suffix < vocab_size))
            if arm.startswith('plain'):
                assert same(ids[:, :inputs.shape[1]], inputs) and same(ids[:, inputs.shape[1]:], suffix)
                for key in ('input_ids', 'pixel_values', 'image_grid_thw'):
                    assert same(data['generate_' + key], data['processor_' + key])
            else:
                assert same(ids, suffix) and same(data['generate_inputs_embeds'], data['prefill_inputs_embeds'])
            records[arm], arrays[arm] = record, data
            arrays_checked += len(data)
        for arm in ('plain', 'eadp1024'):
            record = c.read(c.PREV / 'execution' / ('%02d_%s.json' % (i, arm)))
            data = read_archive(record, c.PREV / 'execution' / ('%02d_%s.npz' % (i, arm)))
            previous[arm] = (record, data)
            keys = list(COMMON) + ['output_ids', 'output_suffix_ids']
            if arm == 'plain':
                keys += ['prefill_visual_pos_masks', 'prefill_deepstack_0', 'prefill_deepstack_1', 'prefill_deepstack_2']
            assert all(same(arrays[arm][key], data[key]) for key in keys)
            assert records[arm]['response'] == record['response']
            assert records[arm]['generate_kwargs_keys'] == record['generate_kwargs_keys']
            assert records[arm]['generate_non_tensor_kwargs'] == record['generate_non_tensor_kwargs']
        changes = {}
        for arm, base in (('plain_ds_off', 'plain'), ('eadp1024_ds_on', 'eadp1024')):
            a, b = arrays[base], arrays[arm]
            assert all(same(a[key], b[key]) for key in COMMON)
            assert records[arm]['generate_kwargs_keys'] == records[base]['generate_kwargs_keys']
            assert records[arm]['generate_non_tensor_kwargs'] == records[base]['generate_non_tensor_kwargs']
            for field in FIELDS:
                key = 'prefill_' + field
                original = b.get('prefill_original_' + field)
                assert (key in a) == (original is not None), field
                if original is not None:
                    assert same(a[key], original), field
            if arm == 'eadp1024_ds_on':
                assert same(b['prefill_visual_pos_masks'], arrays['plain']['prefill_visual_pos_masks'])
            prefix, steps, differing = common_prefix_positions(a, b, records[base], records[arm])
            changes[arm] = dict(output_ids_equal=same(a['output_suffix_ids'], b['output_suffix_ids']),
                response_equal=records[base]['response'] == records[arm]['response'],
                common_output_prefix_tokens=prefix, same_prefix_continuation_steps_checked=steps,
                same_prefix_continuation_differing_steps=differing)
        endings = {arm: dict(output_tokens=int(arrays[arm]['output_suffix_ids'].shape[1]),
                             EOS_present=bool(np.isin(arrays[arm]['output_suffix_ids'], eos_ids).any())) for arm in ARMS}
        results.append(dict(position=item['position'], index=item['index'], within_path=changes, endings=endings))
    c.check(plan)
    target = out / 'independent_verification.json'
    assert not target.exists()
    c.save(target, dict(success=True, n_questions=16, n_native_generations=64, endpoint_anchors=32,
        toggles=32, arrays_checked=arrays_checked, results=results, plan_sha256=c.sha(c.PLAN),
        CPU_only=True, model_loaded=False, GT_read=False, scoring_started=False, benchmark_result=False,
        suffix_slicing_verified=True, response_independently_redecoded=False,
        continuation_comparison='position/cache/rope only on equal generated prefixes; later branches not compared',
        native_container_process_exited=True, controller_exit_requires_root_pty_proof=True,
        limits=plan['limits']))
    print(json.dumps(dict(success=True, n_questions=16, n_native_generations=64, arrays_checked=arrays_checked)))


if __name__ == '__main__':
    main()
