"""Independent NumPy audit of native32; no model, selector or accuracy score."""
import hashlib
import os
from pathlib import Path
import sys
import common as c


def bits(a,b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def array_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def check_native_arrays(arrays,record,np):
    ids = arrays['actual_LLM_input_ids']
    assert ids.dtype == np.int64 and ids.ndim == 2 and ids.shape[0] == 1
    length = ids.shape[1]
    shapes = dict(projected_fp16=((1,576,4096),np.float16),image_embeds_fp32=((1,576,768),np.float32),
        text_global_fp32=((1,768),np.float32),text_dense_fp32=((1,77,768),np.float32),
        global_fp32=((1,576,1),np.float32),dense_fp32=((1,576,1,77),np.float32),
        visual_sim_fp32=((1,576,576),np.float32),importance_fp32=((1,576),np.float32),
        internal_ids=((1,32),np.int64),quotas=((1,),np.int64),gain_trace_fp32=((32,1,576),np.float32),
        final_mask=((1,576),np.bool_),actual_input_fp16=((1,3,336,336),np.float16),
        prepared_embeddings_fp16=((1,length-1+32,4096),np.float16))
    assert set(arrays) == set(shapes) | {'actual_LLM_input_ids'}
    assert set(record['tensors']) == set(arrays)
    for name,value in arrays.items():
        if name in shapes:
            shape,dtype = shapes[name]
            assert value.shape == shape and value.dtype == dtype, name
        meta = record['tensors'][name]
        assert list(value.shape) == meta['shape'] and str(value.dtype) == meta['dtype']
        assert array_sha(value) == meta['sha256'] and np.isfinite(value).all()
    ordered = arrays['internal_ids'][0]; mask = arrays['final_mask']; gains = arrays['gain_trace_fp32']
    assert arrays['quotas'].tolist() == [32] and len(set(ordered.tolist())) == 32
    assert np.all((ordered >= 0) & (ordered < 576))
    expected_mask = np.zeros((1,576),dtype=np.bool_); expected_mask[0,ordered] = True
    assert bits(mask,expected_mask)
    assert gains[:,0,:].argmax(axis=-1).tolist() == ordered.tolist()
    for step in range(32):
        assert np.all(gains[step,0,ordered[:step]] == np.float32(-1e9))
    spatial = np.flatnonzero(mask[0]); assert spatial.tolist() == sorted(ordered.tolist())
    selected = dict(greedy_quotas=[32],effective_mask_indices=[spatial.tolist()],
        greedy_effective_prefixes=[ordered.tolist()],returned_retained_tokens=32)
    assert selected == record['selection']
    markers = np.flatnonzero(ids[0] == -200); assert len(markers) == 1
    marker = int(markers[0]); embedded = arrays['prepared_embeddings_fp16']
    visual = arrays['projected_fp16'][0,spatial,:]
    assert bits(embedded[0,marker:marker+32,:],visual)
    nonvisual = np.concatenate([embedded[0,:marker,:],embedded[0,marker+32:,:]],axis=0)
    assert nonvisual.shape == (length-1,4096)
    contract = dict(marker_index=marker,n_visual=32,visual_fp16_sha256=array_sha(visual),
        nonvisual_fp16_sha256=array_sha(nonvisual),prepared_embeddings_shape=list(embedded.shape))
    assert contract == record['embedded_input_contract']
    assert array_sha(ids) == record['actual_LLM_input_ids_sha256']
    return nonvisual,visual


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules
    state = c.read(c.HERE/'state.json')
    assert state['status'] == 'native_exited' and state['native_exit_code'] == 0
    assert not (c.HERE/'failure.json').exists() and not (c.HERE/'worker_failure.json').exists()
    from run import identity,gpu_pids
    assert identity(state['native_worker']['pid']) != state['native_worker']
    assert state['native_worker']['pid'] not in gpu_pids()
    plan = c.read(c.PLAN); c.check(plan)
    complete = c.read(c.OUT/'native_complete.json')
    assert complete['success'] is True and complete['n_generations'] == 32
    assert complete['source_generations'] == complete['variant_generations'] == 16
    assert complete['all16_source_gate_passed'] is True and complete['plan_sha256'] == c.sha(c.PLAN)
    assert complete['GT_or_accuracy_used'] is False and complete['benchmark_result'] is False
    records = complete['records']; assert len(records) == 32
    assert [r['arm'] for r in records] == ['source']*16 + ['beta2']*16
    assert [r['position'] for r in records[:16]] == [r['position'] for r in records[16:]] == list(range(16))
    assert all(r['all_source_anchors_passed'] is True for r in records[:16])
    gate = c.read(c.OUT/'all16_source_anchor_gate.json')
    assert gate['passed'] is True and gate['n_sources'] == 16 and gate['all_sources_before_variant'] is True
    assert len(gate['source_records']) == 16
    for i,entry in enumerate(gate['source_records']):
        assert Path(entry['path']) == c.OUT/('source_%02d.json' % i)
        assert c.sha(entry['path']) == entry['sha256']
    own = c.read(c.OUT/'process_identity.json')
    assert {k:own[k] for k in ['pid','start_ticks','boot_id']} == state['native_worker']
    assert own['plan_sha256'] == c.sha(c.PLAN) and own['allocation_note_sha256'] == state['allocation_note_sha256']
    fixture = c.read(c.HERE/'source_anchor_fixture.json')
    assert fixture['historical_output_IDs_available'] is False
    assert fixture['historical_input_IDs_available'] is False and fixture['historical_prepared_embeddings_available'] is False
    anchors = fixture['rows']; assert len(anchors) == 16
    generation = c.read(Path(plan['model_path'])/'generation_config.json')
    config = c.read(Path(plan['model_path'])/'config.json')
    assert generation['bos_token_id'] == config['bos_token_id'] == 1
    assert generation['eos_token_id'] == config['eos_token_id'] == 2
    loaded = c.read(c.OUT/'loaded_runtime.json')
    assert loaded['bos_token_id'] == 1 and loaded['eos_token_id'] == 2
    assert loaded['plan_sha256'] == c.sha(c.PLAN) and loaded['worker_sha256'] == c.sha(c.HERE/'worker.py')
    assert loaded['initial_alpha'] == loaded['initial_beta'] == 1. and loaded['K'] == 32
    assert loaded['backend'] == plan['backend'] == 'sdpa' and loaded['model_dtype'] == plan['dtype'] == 'torch.float16'
    assert loaded['torch'] == '2.1.2+cu118' and loaded['transformers'] == '4.37.2'
    assert loaded['component_dtypes'] == plan['expected_component_dtypes']
    observer = loaded['gain_observer_proof']; expected = plan['gain_observer_proof']
    assert {key:observer[key] for key in expected} == expected
    assert observer['actual_code_sha256'] == observer['expected_code_sha256'] and observer['inverse_restores_original'] is True
    import numpy as np
    changes = []; artifacts = []; finite_arrays = 0
    for index,(panel,anchor) in enumerate(zip(plan['panel'],anchors)):
        assert anchor['position'] == index and anchor['question_id'] == panel['question_id']
        assert anchor['prompt'] == panel['prompt'] and 'expected_source_output_token_ids' not in anchor
        pair = []
        for arm in ['source','beta2']:
            path = c.OUT/('%s_%02d.json' % (arm,index)); record = c.read(path)
            assert record == records[(0 if arm == 'source' else 16)+index]
            assert record['index'] == record['position'] == index and record['question_id'] == panel['question_id'] and record['image'] == panel['image']
            assert record['actual_alpha_beta_K'] == [1.,1. if arm == 'source' else 2.,32]
            assert record['calls'] == dict(encode=1,cross=1,greedy=1,full_prepare=1)
            assert record['checks_status'] == 'all_passed' and record['post_generation_tensor_reads_only'] is True
            assert record['all_source_anchors_passed'] is (arm == 'source')
            assert not (c.OUT/('%s_%02d.failed_checks.json' % (arm,index))).exists()
            raw = record['raw_attempt']; assert Path(raw['path']) == c.OUT/('%s_%02d.attempt.json' % (arm,index))
            assert c.sha(raw['path']) == raw['sha256']; attempted = c.read(raw['path'])
            assert attempted['arrays'] == record['arrays'] and attempted['output_token_ids'] == record['output_token_ids'] and attempted['text'] == record['text']
            assert attempted['actual_alpha_beta_K'] == record['actual_alpha_beta_K'] and attempted['calls'] == record['calls']
            assert attempted['actual_retained_tokens'] == attempted['gain_reference_count'] == 32
            assert attempted['post_generation_tensor_reads_only'] is True and attempted['all_checks_completed'] is False
            ids = record['output_token_ids']; assert len(ids) == 1 and ids[0][0] == 1 and 1 < len(ids[0]) <= 129
            assert all(type(value) is int and 0 <= value < config['vocab_size'] for value in ids[0])
            assert record['returned_tokens'] == len(ids[0]) and record['initial_BOS_tokens'] == 1
            assert record['new_tokens'] == len(ids[0])-1 <= 128 and record['generation_horizon'] == 128
            assert record['EOS_present'] == (2 in ids[0])
            entry = record['arrays']; assert Path(entry['path']) == c.OUT/('%s_%02d.npz' % (arm,index))
            assert c.sha(entry['path']) == entry['sha256'] and Path(entry['path']).stat().st_size == entry['bytes']
            with np.load(entry['path'],allow_pickle=False) as cache:
                arrays = {key:cache[key] for key in cache.files}
            assert set(arrays) == set(plan['saved_arrays'])
            nonvisual,visual = check_native_arrays(arrays,record,np); finite_arrays += len(arrays)
            assert record['full_LLM_prompt_sha256'] == hashlib.sha256(panel['prompt'].encode()).hexdigest()
            guidance = panel['prompt'].replace('\nAnswer the question using a single word or phrase.','')
            assert record['guidance_sha256'] == hashlib.sha256(guidance.encode()).hexdigest()
            if arm == 'source':
                assert record['text'] == anchor['expected_source_text']
                with np.load(panel['cache']['path'],allow_pickle=False) as old:
                    mapping = {'projected_fp16':'projected_fp16','actual_input_fp16':'input_fp16',
                        'global_fp32':'global_fp32','dense_fp32':'dense_fp32','visual_sim_fp32':'visual_sim_fp32',
                        'importance_fp32':'native_importance','internal_ids':'native_selected_ids','final_mask':'native_final_mask',
                        'gain_trace_fp32':'alpha1_beta1__source__gain_trace'}
                    assert all(bits(arrays[name],old[old_name]) for name,old_name in mapping.items())
            pair.append((record,arrays,nonvisual,visual)); artifacts.extend([c.describe(path),c.describe(Path(entry['path'])),c.describe(Path(raw['path']))])
        (source,sa,sn,sv),(variant,va,vn,vv) = pair
        unchanged = ['projected_fp16','image_embeds_fp32','text_global_fp32','text_dense_fp32','global_fp32','dense_fp32',
                     'visual_sim_fp32','actual_input_fp16','actual_LLM_input_ids']
        assert all(bits(sa[name],va[name]) for name in unchanged) and bits(sn,vn)
        source_ids,variant_ids = sa['internal_ids'][0],va['internal_ids'][0]
        changes.append(dict(index=index,position=index,question_id=panel['question_id'],
            quotas_changed=not bits(sa['quotas'],va['quotas']),final_masks_changed=not bits(sa['final_mask'],va['final_mask']),
            effective_prefixes_changed=not bits(source_ids,variant_ids),actual_visual_FP16_changed=not bits(sv,vv),
            added_visual_ids=np.flatnonzero(va['final_mask'][0]&~sa['final_mask'][0]).tolist(),
            removed_visual_ids=np.flatnonzero(sa['final_mask'][0]&~va['final_mask'][0]).tolist(),
            importance_changed=not bits(sa['importance_fp32'],va['importance_fp32']),gain_trace_changed=not bits(sa['gain_trace_fp32'],va['gain_trace_fp32']),
            output_token_IDs_changed=source['output_token_ids'] != variant['output_token_ids'],raw_text_changed=source['text'] != variant['text'],
            source_returned_tokens=source['returned_tokens'],variant_returned_tokens=variant['returned_tokens'],
            source_new_tokens=source['new_tokens'],variant_new_tokens=variant['new_tokens'],
            source_EOS=source['EOS_present'],variant_EOS=variant['EOS_present']))
    c.check(plan); assert 'torch' not in sys.modules
    c.save(c.OUT/'independent_verification.json',dict(success=True,n_generations=32,source_anchors_verified=16,variant_inputs_verified=16,
        finite_arrays_checked=finite_arrays,full_LLM_IDs_and_nonvisual_embeddings_unchanged=True,
        real_projected_image_embeds_text_global_dense_visualsim_pixels_equal=True,
        source_native_cache_and_historical_text_anchors=True,historical_output_ID_anchor_claimed=False,
        source_gain_anchor_scope='Old same-source AST-traced helper,not historical native internal gain capture.',
        changes=changes,plan_sha256=c.sha(c.PLAN),native_complete_sha256=c.sha(c.OUT/'native_complete.json'),
        artifacts=artifacts,model_loaded=False,CPU_only=True,GT_or_accuracy_used=False,new_benchmark_row=False,
        independent_token_redecode_performed=False,text_decoding_scope='Worker native tokenizer; CPU verifies IDs/source text and BOS/EOS/horizon,without separate tokenizer decode.',
        limits=plan['limits'],root_actual_controller_exit_required=True))
    print(__import__('json').dumps(dict(success=True,n_generations=32,source_anchors=16,variant_inputs=16,benchmark_result=False)))


if __name__ == '__main__':
    main()
