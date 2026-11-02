"""Independent no-GPU audit of two full DocVQA DeepStack toggle generations.

The recorded high-dimensional tensor hashes are captured during generation;
this verifier compares their metadata but cannot reconstruct unsaved tensors.
It never reads ground truth, calls a scorer, or loads model weights.
"""
import hashlib
import json
import os
import struct
from pathlib import Path
import sys


FIELDS = ('inputs_embeds', 'position_ids', 'attention_mask', 'cache_position',
          'visual_pos_masks', 'deepstack_0', 'deepstack_1', 'deepstack_2')
ARMS = ('plain', 'eadp1024', 'plain_ds_off', 'eadp1024_ds_on')


def ids_sha(ids):
    assert all(type(value) is int for value in ids)
    return hashlib.sha256(struct.pack('<' + 'q' * len(ids), *ids)).hexdigest()


def mask_sha(mask):
    assert all(type(value) is bool for value in mask)
    return hashlib.sha256(bytes(mask)).hexdigest()


def check_metadata(meta):
    assert set(meta) == {'dtype', 'shape', 'sha256'}
    assert isinstance(meta['dtype'], str) and isinstance(meta['shape'], list)
    assert all(type(x) is int and x >= 0 for x in meta['shape'])
    assert isinstance(meta['sha256'], str) and len(meta['sha256']) == 64
    int(meta['sha256'], 16)


def check_observation(item, prediction, obs, arm, image_token_id, vocab_size):
    """Verify recorded input IDs/masks and single-factor prefill boundaries."""
    assert arm in ARMS and obs['arm'] == arm
    assert obs['position'] == prediction['position'] == item['position']
    assert obs['index'] == prediction['index'] == item['index']
    assert prediction['native_failure'] is None and isinstance(prediction['prediction'], str)
    assert obs['native_failure'] is None and not obs['observer_errors']
    assert obs['actual_processor'] == item['processor_expected']
    ids = obs['input_ids']
    assert ids and ids_sha(ids) == obs['actual_processor']['input_ids']['sha256']
    assert obs['actual_processor']['input_ids']['dtype'] == 'torch.int64'
    assert obs['actual_processor']['input_ids']['shape'] == [1, len(ids)]
    actual_mask = [value == image_token_id for value in ids]
    mask = obs['image_mask']
    assert mask == actual_mask and sum(mask) == obs['image_mask_true_count'] == 1024
    assert mask_sha(mask) == obs['image_mask_sha256']
    assert obs['image_mask_shape'] == [1, len(ids)]
    originals, proposed, final = (obs[name] for name in ('prefill_original', 'prefill_proposed', 'prefill_final'))
    for entry in (originals, proposed, final):
        assert set(entry) == set(FIELDS)
        for value in entry.values():
            if value is not None:
                check_metadata(value)
    assert proposed == final
    for field in FIELDS[:4]:
        assert originals[field] is not None and originals[field] == final[field], field
    for field, dtype, shape in (
        ('inputs_embeds', 'torch.bfloat16', [1, len(ids), 4096]),
        ('position_ids', 'torch.int64', [3, 1, len(ids)]),
        ('attention_mask', 'torch.int64', [1, len(ids)]),
        ('cache_position', 'torch.int64', [len(ids)]),
    ):
        assert originals[field]['dtype'] == dtype and originals[field]['shape'] == shape, field
    assert originals['attention_mask'] == obs['actual_processor']['attention_mask']
    mask_meta = dict(dtype='torch.bool', shape=[1, len(ids)], sha256=mask_sha(mask))
    visual = obs['visual_deepstack']
    assert isinstance(visual, list) and len(visual) == 3
    check_metadata(obs['vision_features'])
    assert obs['vision_features']['dtype'] == 'torch.bfloat16'
    assert obs['vision_features']['shape'] == [1024, 4096]
    if arm.startswith('plain'):
        assert obs['pruner_calls'] == 0 and obs['pruner_split_sizes'] is None
        assert obs['pruner_input_features'] is None and obs['pruner_output_features'] is None
    else:
        assert obs['pruner_calls'] == 1 and obs['pruner_split_sizes'] == [1024]
        assert obs['pruner_input_features'] == obs['pruner_output_features'] == obs['vision_features']
    for layer in visual:
        check_metadata(layer)
        assert layer['dtype'] == 'torch.bfloat16' and layer['shape'] == [1024, 4096]
    if arm.startswith('plain'):
        assert originals['visual_pos_masks'] == mask_meta
        for i in range(3):
            assert originals['deepstack_' + str(i)] == visual[i]
    else:
        assert originals['visual_pos_masks'] is None
        for i in range(3):
            assert originals['deepstack_' + str(i)] is None
    if arm in ('plain', 'eadp1024_ds_on'):
        assert final['visual_pos_masks'] == mask_meta
        for i in range(3):
            assert final['deepstack_' + str(i)] == visual[i]
    else:
        assert final['visual_pos_masks'] is None
        for i in range(3):
            assert final['deepstack_' + str(i)] is None
    toggle = arm in ('plain_ds_off', 'eadp1024_ds_on')
    assert obs['intervention_attempts'] == int(toggle) and obs['intervention_applied'] is toggle
    assert obs['intervention_error'] is None
    for key in ('vision_calls', 'processor_calls', 'generate_calls'):
        assert obs[key] == 1
    assert obs['llm_calls'] >= 1 and obs['continuation_steps'] == obs['llm_calls'] - 1
    assert obs['generate_non_tensor_kwargs'] == dict(do_sample=False, max_new_tokens=2048, top_p=.001,
        top_k=1, temperature=.01, repetition_penalty=1.)
    keys = set(obs['generate_kwargs_keys'])
    scalar_keys = set(obs['generate_non_tensor_kwargs'])
    expected = {'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'} if arm.startswith('plain') else {'inputs_embeds', 'attention_mask'}
    assert keys == scalar_keys | expected
    tensor_kwargs = obs['generate_tensor_kwargs']
    assert set(tensor_kwargs) == expected
    if arm.startswith('plain'):
        assert tensor_kwargs == obs['actual_processor']
    else:
        assert tensor_kwargs['inputs_embeds'] == originals['inputs_embeds']
        assert tensor_kwargs['attention_mask'] == obs['actual_processor']['attention_mask']
    check_metadata(obs['rope_deltas'])
    suffix = obs['output_suffix_ids']
    assert 0 < len(suffix) <= 2048 and all(type(v) is int and 0 <= v < vocab_size for v in suffix)
    assert obs['continuation_steps'] == len(suffix) - 1
    full_output = obs['output_ids']
    assert full_output == (ids + suffix if arm.startswith('plain') else suffix)
    assert obs['native_redecode'] == prediction['prediction']
    continuation = obs['continuation']
    assert len(continuation) == obs['continuation_steps']
    for step in continuation:
        assert set(step) == {'position_ids', 'cache_position', 'rope_deltas',
                             'visual_pos_masks_present', 'deepstack_visual_embeds_present'}
        for key, shape in (('position_ids', [3, 1, 1]), ('cache_position', [1]), ('rope_deltas', [1, 1])):
            assert step[key] is not None
            check_metadata(step[key])
            assert step[key]['dtype'] == 'torch.int64' and step[key]['shape'] == shape
        for key in ('visual_pos_masks_present', 'deepstack_visual_embeds_present'):
            assert step[key] is False


def check_prior_anchor(c, index, arm, prediction, obs):
    """Bind a new record to the accepted fixed16 actual output and metadata."""
    prior = c.read(c.PANEL / 'execution' / ('%02d_%s.json' % (index, arm)))
    archive = c.PANEL / 'execution' / ('%02d_%s.npz' % (index, arm))
    assert Path(prior['archive']) == archive and c.sha(archive) == prior['archive_sha256']
    arrays = prior['arrays']
    required = {'output_ids', 'output_suffix_ids', 'visual_features', 'rope_deltas'}
    required.update('visual_deepstack_' + str(i) for i in range(3))
    required.update('processor_' + key for key in obs['actual_processor'])
    assert prior['response'] == prediction['prediction']
    assert prior['generate_kwargs_keys'] == obs['generate_kwargs_keys']
    assert prior['generate_non_tensor_kwargs'] == obs['generate_non_tensor_kwargs']
    ids = obs['output_suffix_ids']
    assert arrays['output_suffix_ids'] == dict(dtype='torch.int64', shape=[1, len(ids)], sha256=ids_sha(ids))
    full_ids = obs['output_ids']
    assert arrays['output_ids'] == dict(dtype='torch.int64', shape=[1, len(full_ids)], sha256=ids_sha(full_ids))
    for key, value in obs['actual_processor'].items():
        assert arrays['processor_' + key] == value
    assert arrays['visual_features'] == obs['vision_features']
    for i, value in enumerate(obs['visual_deepstack']):
        assert arrays['visual_deepstack_' + str(i)] == value
    for label, prefix in (('prefill_original', 'prefill_original_'),
                          ('prefill_proposed', 'prefill_proposed_'), ('prefill_final', 'prefill_')):
        for key, value in obs[label].items():
            assert arrays.get(prefix + key) == value, (index, arm, prefix + key)
            if value is not None:
                required.add(prefix + key)
    assert arrays.get('rope_deltas') == obs['rope_deltas']
    assert prior['continuation_steps'] == obs['continuation_steps'] == len(prior['continuation_presence'])
    for i, (step, presence) in enumerate(zip(obs['continuation'], prior['continuation_presence'])):
        assert step['visual_pos_masks_present'] == presence['visual_mask_present']
        assert step['deepstack_visual_embeds_present'] == presence['deepstack_present']
        for key in ('position_ids', 'cache_position', 'rope_deltas'):
            old_key = 'continuation_%04d_%s' % (i, key)
            assert arrays.get(old_key) == step[key], (index, arm, old_key)
            required.add(old_key)
    # These previous fixed16 arrays really exist on disk. Recompute every
    # required old array's hash instead of relying only on its JSON metadata.
    # The new full-run high-dimensional tensors are intentionally not saved.
    import numpy as np
    with np.load(archive, allow_pickle=False) as stored:
        for key in required:
            value = stored[key]
            record = arrays[key]
            assert list(value.shape) == record['shape']
            assert hashlib.sha256(value.tobytes()).hexdigest() == record['sha256']
            dtype = record['dtype']
            expected = {'torch.bfloat16': np.uint16, 'torch.int64': np.int64,
                        'torch.int32': np.int32, 'torch.bool': np.bool_,
                        'torch.float32': np.float32, 'torch.float16': np.float16}
            assert dtype in expected and value.dtype == expected[dtype], (index, arm, key)


def check_runtime(c, runtime, plan_sha):
    old = c.read(c.OLD / 'run_plan.json')
    expected_recipe = dict(old['recipe'])
    expected_recipe['visual_token_num'] = 1024
    assert runtime['actual_recipe'] == expected_recipe
    assert runtime['model_dtype'] == 'torch.bfloat16' and runtime['backend'] == 'flash_attention_2'
    assert runtime['post_process'] is False and runtime['GPU'] == 'NVIDIA A40'
    assert runtime['model_instances'] == 1 and runtime['plan_sha256'] == plan_sha
    assert runtime['GT_read'] is False and runtime['scoring_started'] is False
    assert set(runtime['actual_sources']) == {
        'vlmeval.vlm.qwen3_vl.model_fixed_res', 'vlmeval.vlm.qwen3_vl.model',
        'model.pruner', 'transformers.models.qwen3_vl.modeling_qwen3_vl'}
    for path in runtime['actual_sources'].values():
        assert c.sha(path) == old['source_sha256'][path], path


def main():
    import common as c
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not any(Path('/dev', name).exists() for name in ('nvidia0', 'nvidiactl', 'nvidia-uvm'))
    plan = c.read(c.PLAN)
    c.check(plan, 'container')
    state = c.read(c.HERE / 'state.json')
    assert state['status'] == 'native_exited_validation_pending' and state['native_exit_code'] == 0
    assert state['original_container_exited'] and state['GPU_pid_absent']
    assert state['plan_sha256'] == c.sha(c.PLAN)
    out = c.HERE / 'execution'
    target = out / 'generation_verification.json'
    assert not target.exists()
    runtime_path = out / 'runtime.json'
    check_runtime(c, c.read(runtime_path), c.sha(c.PLAN))
    done = c.read(out / 'native_complete.json')
    endpoints = c.read(out / 'endpoints_verified.json')
    assert done['success'] and done['n_questions'] == 5349 and done['n_generations'] == 10698
    assert done['endpoint_anchors'] == endpoints['n_endpoints'] == 32 and endpoints['success']
    assert done['plan_sha256'] == endpoints['plan_sha256'] == c.sha(c.PLAN)
    assert done['endpoints_verified_sha256'] == c.sha(out / 'endpoints_verified.json')
    assert done['all_generation_before_scoring'] and not done['GT_read'] and not done['scoring_started']
    assert endpoints['old_full_suffix_matches'] == endpoints['old_processor_matches'] == endpoints['old_prefill_metadata_matches'] == 32
    prior_proof = c.read(c.PANEL / 'root_terminal_process_proof.json')
    prior_ver = c.read(c.PANEL / 'execution/independent_verification.json')
    assert prior_proof['root_owned_exec_actual_exit_code'] == 0 and prior_proof['all_original_processes_exited']
    assert prior_proof['independent_verification_sha256'] == c.sha(c.PANEL / 'execution/independent_verification.json')
    assert prior_ver['success'] and prior_ver['n_native_generations'] == 64
    base_proof = c.read(c.BASE / 'root_terminal_process_proof.json')
    base_ver = c.read(c.BASE / 'execution/generation_verification.json')
    base_done = c.read(c.BASE / 'execution/native_complete.json')
    assert base_proof['root_owned_exec_actual_exit_code'] == 0 and base_proof['all_original_processes_exited']
    assert base_proof['generation_verification_sha256'] == c.sha(c.BASE / 'execution/generation_verification.json')
    assert base_ver['success'] and base_ver['n_questions'] == 5349 and base_ver['n_generations'] == 10698
    assert base_ver['native_complete_sha256'] == c.sha(c.BASE / 'execution/native_complete.json')
    assert base_done['predictions'] == base_ver['prediction_sha256'] and base_done['observations'] == base_ver['observation_sha256']
    items = c.rows(c.BASE / 'input_manifest.jsonl')
    assert len(items) == 5349 and [r['position'] for r in items] == list(range(5349))
    panel = c.read(c.PANEL / 'plan.json')['panel']
    assert len(panel) == 16 and len({r['position'] for r in panel}) == 16
    panel_positions = {entry['position']: i for i, entry in enumerate(panel)}
    old_plan = c.read(c.OLD / 'run_plan.json')
    model_path = Path(old_plan['recipe']['model_path'])
    config = c.read(model_path / 'config.json')
    image_token_id = config['image_token_id']
    vocab_size = config['text_config']['vocab_size']
    endpoint_obs = {}
    for arm in ('plain', 'eadp1024'):
        base_pred_path = c.BASE / 'execution' / (arm + '.predictions.jsonl')
        base_obs_path = c.BASE / 'execution' / (arm + '.observations.jsonl')
        assert c.sha(base_pred_path) == base_done['predictions'][arm]
        assert c.sha(base_obs_path) == base_done['observations'][arm]
        base_predictions, base_observations = c.rows(base_pred_path), c.rows(base_obs_path)
        assert len(base_predictions) == len(base_observations) == 5349
        pred_path = out / ('endpoint_' + arm + '.predictions.jsonl')
        obs_path = out / ('endpoint_' + arm + '.observations.jsonl')
        assert c.sha(pred_path) == endpoints['prediction_sha256'][arm]
        assert c.sha(obs_path) == endpoints['observation_sha256'][arm]
        predictions, observations = c.rows(pred_path), c.rows(obs_path)
        assert len(predictions) == len(observations) == 16
        for i, (panel_item, prediction, obs) in enumerate(zip(panel, predictions, observations)):
            pos = panel_item['position']; item = items[pos]
            assert item['index'] == panel_item['index']
            check_observation(item, prediction, obs, arm, image_token_id, vocab_size)
            check_prior_anchor(c, i, arm, prediction, obs)
            assert prediction['prediction'] == base_predictions[pos]['prediction']
            assert obs['output_suffix_ids'] == base_observations[pos]['output_suffix_ids']
            assert obs['actual_processor'] == base_observations[pos]['actual_processor']
        endpoint_obs[arm] = observations
    observed = {}
    for arm in ('plain_ds_off', 'eadp1024_ds_on'):
        pred_path = out / (arm + '.predictions.jsonl')
        obs_path = out / (arm + '.observations.jsonl')
        assert c.sha(pred_path) == done['prediction_sha256'][arm]
        assert c.sha(obs_path) == done['observation_sha256'][arm]
        predictions, observations = c.rows(pred_path), c.rows(obs_path)
        assert len(predictions) == len(observations) == 5349
        checked_prior = 0
        for pos, (item, prediction, obs) in enumerate(zip(items, predictions, observations)):
            check_observation(item, prediction, obs, arm, image_token_id, vocab_size)
            if pos in panel_positions:
                check_prior_anchor(c, panel_positions[pos], arm, prediction, obs)
                checked_prior += 1
        assert checked_prior == 16
        observed[arm] = (predictions, observations)
    # Only after the complete structural and prior-anchor audit, load the
    # processor/tokenizer on CPU. No model weights, CUDA or GT are accessed.
    import torch
    from transformers import AutoProcessor
    assert not torch.cuda.is_initialized()
    tokenizer = AutoProcessor.from_pretrained(str(model_path), local_files_only=True).tokenizer
    for arm, (predictions, observations) in observed.items():
        decoded = tokenizer.batch_decode([r['output_suffix_ids'] for r in observations],
                                        skip_special_tokens=True, clean_up_tokenization_spaces=False)
        assert decoded == [r['prediction'] for r in predictions], arm
    assert not torch.cuda.is_initialized()
    for path, digest in plan['file_guards'].items():
        assert c.sha(path) == digest, path
    c.save(target, dict(success=True, n_questions=5349, n_generations=10698, endpoint_anchors=32,
        fixed16_variant_anchors=32, processor_identity_checks=10698, prefill_intervention_checks=10698,
        independent_token_redecodes=10698, plan_sha256=c.sha(c.PLAN),
        runtime_sha256=c.sha(runtime_path),
        native_complete_sha256=c.sha(out / 'native_complete.json'),
        endpoints_verified_sha256=c.sha(out / 'endpoints_verified.json'),
        prediction_sha256=done['prediction_sha256'], observation_sha256=done['observation_sha256'],
        CPU_only=True, model_loaded=False, CUDA_initialized=False, GT_read=False, scoring_started=False,
        native_container_exit_verified_by_controller=True, root_controller_actual_exit_proof_required=True,
        high_dimensional_arrays_independently_reconstructed=False,
        limits=['New full-run pixel, feature, DeepStack, embedding and position tensors were not saved: their hashes were captured during generation and compared as metadata, not independently reconstructed.',
                'The historical full baseline lacks per-question internal tensors; fixed16 anchors apply only to those 16 samples.',
                'Changing DeepStack can naturally change later KV contents and generated tokens.',
                'No ground truth or scoring is used in this generation audit.']))
    print(json.dumps(dict(success=True, n_questions=5349, n_generations=10698, independent_token_redecodes=10698)))


if __name__ == '__main__':
    main()

