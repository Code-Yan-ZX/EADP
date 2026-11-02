"""Independent CPU terminal review; no original verifier/scorer/model imports."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE / 'execution'
ROOT_SHA = '77dca7df064b943cd61b309943800827c9dd1622362e7a2c126a69968c86a068'
PLAN_SHA = '34b815cc26db836ad6bc9f7ea69eb08956680d350a86b8e7a697c904545a84c3'
REPORT_SHA = '637a1adf82a3757dbe8b14f6e927dc5c5d4b4a608facf0ebce14cbe638f8ef7a'
ARMS = ['source', 'eq6_quantile_membership_only']


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def stat(p):
    p = Path(p)
    s = p.stat()
    return dict(resolved=str(p.resolve()), bytes=s.st_size, device=s.st_dev,
                inode=s.st_ino, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def spatial_order(mask):
    # Independent explicit coordinate traversal of the native spatial branch:
    # first base crop raster, then the48x48 raster made from the four local crops.
    ids = [token for token in range(576) if mask[0, token]]
    for row in range(48):
        for column in range(48):
            crop = 1 + 2 * (row // 24) + column // 24
            token = 24 * (row % 24) + column % 24
            if mask[crop, token]:
                ids.append(crop * 576 + token)
    return np.asarray(ids, dtype=np.int64)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules
    assert not (HERE / 'failure.json').exists()
    assert sha(HERE / 'root_terminal_process_proof.json') == ROOT_SHA
    assert sha(HERE / 'plan.json') == PLAN_SHA
    assert sha(RUN / 'independent_verification.json') == REPORT_SHA
    root, plan, claim, state, note = [read(HERE / n) for n in
        ('root_terminal_process_proof.json', 'plan.json', 'launch.claim.json', 'state.json', 'root_allocation_note.json')]
    prior, release, native, runtime = [read(RUN / n) for n in
        ('independent_verification.json', 'release.json', 'native_complete.json', 'loaded_runtime.json')]
    assert root['root_owned_exec_handle'] == 89061 and root['root_owned_exec_actual_exit_code'] == 0 and root['all_original_processes_exited']
    assert root['plan_sha256'] == claim['plan_sha256'] == note['plan_sha256'] == release['plan_sha256'] == prior['plan_sha256'] == runtime['plan_sha256'] == PLAN_SHA
    assert root['allocation_note_sha256'] == claim['note_sha256'] == sha(HERE / 'root_allocation_note.json')
    assert root['controller_state_sha256'] == sha(HERE / 'state.json')
    assert root['launch_process_snapshot_sha256'] == sha(HERE / 'root_launch_process_snapshot.json')
    assert root['release_sha256'] == state['release_sha256'] == sha(RUN / 'release.json')
    assert root['independent_verification_sha256'] == release['independent_verification_sha256'] == REPORT_SHA
    assert root['controller'] == state['controller'] == claim['controller'] == release['controller']
    assert root['native_worker'] == state['native_worker'] == release['native_worker']
    assert state['status'] == 'complete' and state['native_exit_code'] == release['native_exit_code'] == 0
    assert release['actual_native_exited'] and release['success'] and prior['success'] and native['success']
    assert release['n_generations'] == prior['n_generations'] == native['n_generations'] == 2
    assert prior['native_complete_sha256'] == sha(RUN / 'native_complete.json')
    assert not prior['GT_or_accuracy_used'] and not prior['benchmark_result'] and not native['GT_or_accuracy_used']
    for path, digest in plan['source_sha256'].items():
        assert sha(path) == digest
    for entry in plan['frozen_evidence']:
        assert sha(entry['path']) == entry['sha256']
    for path, expected in plan['model_stat_guards'].items():
        assert stat(path) == {k:v for k,v in expected.items() if k != 'prior_verified_sha256'}
    model = Path(plan['model_path'])
    generation_path, config_path = model / 'generation_config.json', model / 'config.json'
    for path in (generation_path, config_path):
        assert sha(path) == plan['model_stat_guards'][str(path)]['prior_verified_sha256']
    generation, config = read(generation_path), read(config_path)
    assert generation['bos_token_id'] == config['bos_token_id'] == 1
    assert generation['eos_token_id'] == config['eos_token_id'] == 2
    assert config['mm_patch_merge_type'] == 'spatial_unpad' and config['image_aspect_ratio'] == 'anyres'
    assert plan['parameters'] == dict(K=128,alpha=.5,beta=2.,max_new_tokens=128,temperature=0.,top_p=None,num_beams=1,conv_mode='vicuna_v1',use_cache=True)
    assert runtime['backend'] == 'sdpa' and runtime['model_dtype'] == 'torch.float16' and runtime['same_GPU_cached_features_object'] and not runtime['vision_or_projector_forward_allowed']
    assert runtime['worker_sha256'] == sha(HERE / 'probe.py')
    assert runtime['component_dtypes'] == plan['cache_capture_runtime']['dtype']
    # Source is inspected, never imported. _unpad removal makes the local raster
    # branch apply; its permute(0,2,1,3,4) matches our explicit coordinate walk.
    arch_path = Path(runtime['native_prepare_file'])
    arch = arch_path.read_text()
    assert arch_path.as_posix() in plan['source_sha256']
    assert "mm_patch_merge_type.replace('_unpad', '')" in arch
    assert 'image_feature.permute(0, 2, 1, 3, 4)' in arch
    contract = plan['cache_contract']
    cache_path = Path(contract['cache_path'])
    assert sha(cache_path) == contract['cache_sha256']
    with np.load(cache_path, allow_pickle=False) as cache:
        features = cache['projected_fp16']
        pixels = cache['actual_input_fp16']
        masks = {a:cache[f'K128_alpha05_beta2__{a}__final_mask'] for a in ARMS}
        quotas = {a:cache[f'K128_alpha05_beta2__{a}__quotas'] for a in ARMS}
        internal = {a:cache[f'K128_alpha05_beta2__{a}__internal_ids'] for a in ARMS}
    assert features.shape == (5,576,4096) and features.dtype == np.float16 and np.isfinite(features).all()
    assert pixels.shape == (1,5,3,336,336) and pixels.dtype == np.float16 and np.isfinite(pixels).all()
    assert array_sha(features) == contract['projected_fp16_sha256']
    assert array_sha(pixels) == contract['actual_input_fp16_sha256']
    records = native['records']
    assert [x['arm'] for x in records] == ARMS and records == prior['records']
    embedded, orders, per_arm = {}, {}, []
    artifacts = [HERE / n for n in ('plan.json','root_allocation_note.json','launch.claim.json','state.json','root_terminal_process_proof.json','root_launch_process_snapshot.json','probe.py','cache_contract.py')]
    artifacts += [RUN / n for n in ('native_complete.json','independent_verification.json','release.json','loaded_runtime.json')]
    artifacts += [cache_path, generation_path, config_path, Path(__file__).resolve()]
    for arm, record in zip(ARMS, records):
        assert read(RUN / (arm + '.json')) == record
        assert record['calls'] == dict(encode=1, full_prepare=1) and record['CLIP_forward_calls'] == record['projector_forward_calls'] == 0
        assert record['observations_after_generate'] and record['actual_retained_tokens'] == 637
        assert record['marker_index'] == 35 and record['input_token_count'] == plan['input_reference']['n_tokens'] == 248
        assert record['input_ids_sha256'] == plan['input_reference']['token_ids_sha256']
        mask, quota, ids = masks[arm], quotas[arm], internal[arm]
        assert mask.dtype == np.bool_ and mask.shape == (5,576) and quota.tolist() == [146,162,106,130,93]
        assert np.array_equal(mask.sum(axis=1),quota) and array_sha(mask) == contract['arms'][arm]['mask_sha256']
        for crop in range(5):
            prefix = ids[crop,:int(quota[crop])]
            assert len(set(prefix.tolist())) == quota[crop] and np.array_equal(np.sort(prefix),np.flatnonzero(mask[crop]))
        order = spatial_order(mask)
        orders[arm] = order
        assert len(order) == 637 and array_sha(order) == contract['arms'][arm]['visual_order_sha256']
        expected = features.reshape(2880,4096)[order]
        assert array_sha(expected) == contract['arms'][arm]['selected_visual_fp16_sha256']
        path = Path(record['prepared_embeddings_path'])
        assert path == RUN / (arm + '.inputs_embeds.npy') and sha(path) == record['prepared_embeddings_sha256']
        embedded[arm] = np.load(path,allow_pickle=False)
        x = embedded[arm]
        assert x.shape == (1,884,4096) and x.dtype == np.float16 and np.isfinite(x).all()
        assert np.array_equal(x[0,35:672].view(np.uint16),expected.view(np.uint16))
        assert record['shared_features_after_generation_sha256'] == array_sha(features)
        tokens = record['output_token_ids']
        assert len(tokens) == 1 and len(tokens[0]) == 129 and tokens[0][0] == 1 and 2 not in tokens[0]
        assert all(type(t) is int and 0 <= t < config['vocab_size'] for t in tokens[0])
        artifacts += [path,RUN / (arm+'.json')]
        per_arm.append(dict(arm=arm,embeddings_shape=list(x.shape),embeddings_dtype=str(x.dtype),visual_rows_exact_FP16_bits=637,
            visual_selected_sha256=array_sha(expected),returned_tokens=129,initial_BOS=1,new_token_count=128,EOS_present=False))
    source, variant = [embedded[a] for a in ARMS]
    assert np.array_equal(source[:,:35].view(np.uint16),variant[:,:35].view(np.uint16))
    assert np.array_equal(source[:,672:].view(np.uint16),variant[:,672:].view(np.uint16))
    assert np.argwhere(masks[ARMS[0]] != masks[ARMS[1]]).tolist() == [[3,115],[3,330]]
    shifted = np.flatnonzero(orders[ARMS[0]] != orders[ARMS[1]]).tolist()
    assert shifted == list(range(465,548))
    assert records[0]['output_token_ids'] == records[1]['output_token_ids'] and records[0]['text'] == records[1]['text']
    gpu = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    gpu_pids = [int(x.strip()) for x in gpu.splitlines() if x.strip()]
    processes = []
    for role in ('controller','native_worker'):
        original = root[role]
        exists = Path('/proc',str(original['pid'])).exists()
        assert not exists and original['pid'] not in gpu_pids
        processes.append(dict(role=role,original=original,proc_absent=True,GPU_pid_absent=True))
    artifact_rows = [dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in dict.fromkeys(artifacts)]
    assert 'torch' not in sys.modules and not any(n.startswith('llava') for n in sys.modules)
    report = dict(success=True,checked_utc=datetime.now(timezone.utc).isoformat(),reviewer='/root/evidence_provenance',
        scope='Independent completed-case CPU bit review; original verify not invoked, no GT/old answers/scorer/model/GPU generation.',
        plan_sha256=PLAN_SHA,root_terminal_process_proof_sha256=ROOT_SHA,original_independent_report_sha256=REPORT_SHA,
        source_file_SHA_checks=len(plan['source_sha256']),frozen_evidence_SHA_checks=len(plan['frozen_evidence']),model_stat_checks=len(plan['model_stat_guards']),
        native_source_spatial_order_independently_rebuilt=True,per_arm=per_arm,
        nonvisual_FP16_embeddings_bit_identical=True,nonvisual_prefix_rows=35,nonvisual_suffix_rows=212,
        same_crop_quotas=[146,162,106,130,93],changed_mask=dict(crop=3,removed=330,added=115),
        changed_visual_sequence_rows=len(shifted),shifted_visual_sequence_positions=shifted,
        actual_input_embedding_rows_different=int(np.any(source.view(np.uint16)!=variant.view(np.uint16),axis=2).sum()),
        output_129_token_IDs_identical=True,recorded_texts_identical=True,
        generation_config=dict(path=str(generation_path),sha256=sha(generation_path),bos_token_id=1,eos_token_id=2),
        horizon=dict(max_new_tokens=128,returned_length_including_BOS=129,initial_BOS_tokens=1,new_tokens=128,
            EOS2_present_either_arm=False,observed_through_configured_max_horizon=True,natural_EOS_completion_claimed=False),
        source_beta2_anchor_limit='Native feature/extraction anchor was beta1/K128. The beta2 source mask is a source-helper replay over those exact native cached FP16 projected features, not a separate end-to-end beta2 extraction anchor. This generation is a declared cached-mask counterfactual.',
        output_token_redecode_repeated_in_this_peer=False,previous_terminal_decoder_checks=prior['output_token_decoding_checks'],
        root_owned_actual_exec_exit_code=0,actual_process_checks=processes,current_GPU_pids=gpu_pids,
        no_GT_or_old_answers_read=True,new_benchmark_row=False,accuracy_scoring=False,model_loaded=False,original_verifier_invoked=False,
        limits=['Equal discrete outputs through128new tokens does not rule out logit differences, later divergence or other examples.',
            'No EOS appeared. This is equality up to the configured horizon, not a naturally terminated answer.',
            'One previously discovered case is neither an unbiased sample nor an estimate of full5000 accuracy or the paper gap.',
            'No further model run, full benchmark variant or parameter selection follows automatically.'],artifacts=artifact_rows)
    with (HERE / 'peer_terminal_review.json').open('x') as f:
        json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(success=True,visual_rows_each=637,nonvisual_bit_identical=True,shifted_rows=len(shifted),returned_lengths=[129,129],EOS_present=False,report_sha256=sha(HERE/'peer_terminal_review.json'))))


if __name__ == '__main__':
    main()
