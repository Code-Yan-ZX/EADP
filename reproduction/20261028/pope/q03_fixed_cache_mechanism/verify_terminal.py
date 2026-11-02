"""Independent NumPy terminal audit; no selector, model, generation or scoring."""
import hashlib
import json
import os
from pathlib import Path
import sys
import common as c


def exact(a,b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def source_checks(source,original,np):
    prefix = 'alpha05_beta1__source__'
    spatial = np.flatnonzero(original[prefix+'final_mask'][0])
    sequence = original['projected_fp16'][0,spatial,:]
    return dict(importance=exact(source['source__importance'],original[prefix+'importance']),
        ordered32=exact(source['source__ordered_ids'],original[prefix+'ids']),
        full32_gain_trace=exact(source['source__gains'],original[prefix+'gain_trace']),
        finalmask=exact(source['source__final_mask'],original[prefix+'final_mask']),
        derived_spatial_FP16_sequence=exact(source['source__final_projected_FP16_sequence'],sequence))


def check_arm(arrays,arm,features,np):
    count = 15 if arm == 'source' else 23
    shapes = dict(importance=((1,576),np.float32), ordered_ids=((1,32),np.int64),
        gains=((32,1,576),np.float32), quotas=((1,),np.int64), final_mask=((1,576),np.bool_),
        spatial_ids=((32,),np.int64), final_projected_FP16_sequence=((32,4096),np.float16),
        filtered_local_sim=((1,576,1,count),np.float32), selected_entropy=((1,1,count),np.float32),
        local_sim=((1,576,1),np.float32))
    for name,(shape,dtype) in shapes.items():
        value = arrays[arm+'__'+name]
        assert value.shape == shape and value.dtype == dtype and np.isfinite(value).all(), (arm,name)
    ids = arrays[arm+'__ordered_ids']; gains = arrays[arm+'__gains']
    mask = arrays[arm+'__final_mask']; spatial = arrays[arm+'__spatial_ids']
    assert len(set(ids[0].tolist())) == 32 and np.all((ids >= 0) & (ids < 576))
    assert arrays[arm+'__quotas'].tolist() == [32]
    assert np.array_equal(gains.argmax(axis=-1).reshape(1,32),ids)
    expected_mask = np.zeros((1,576),dtype=np.bool_); expected_mask[0,ids[0]] = True
    assert exact(mask,expected_mask) and int(mask.sum()) == 32
    assert exact(spatial,np.sort(ids[0])) and exact(spatial,np.flatnonzero(mask[0]))
    assert exact(features[0,spatial,:],arrays[arm+'__final_projected_FP16_sequence'])
    for step in range(32):
        assert np.all(gains[step,0,ids[0,:step]] == np.float32(-1e9))


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan = c.read(c.PLAN); c.check(plan)
    state = c.read(c.HERE/'state.json')
    assert state['status'] == 'worker_exited_validation_pending' and state['worker_exit_code'] == 0
    assert not (c.HERE/'failure.json').exists() and not (c.HERE/'worker_failure.json').exists()
    from run import gpu_pids
    assert c.identity(state['worker']['pid']) != state['worker'] and state['worker']['pid'] not in gpu_pids()
    summary = c.read(c.EXEC/'summary.json')
    assert summary['status'] in ['complete','complete_with_explicit_unmeasured'] and summary['n_questions'] == 16
    assert summary['plan_sha256'] == c.sha(c.PLAN)
    assert summary['per_question_sha256'] == c.sha(c.EXEC/'per_question.jsonl')
    assert summary['runtime_sha256'] == c.sha(c.EXEC/'runtime.json')
    assert all(summary[k] is False for k in ['model_loaded','FA2_imported','GT_read','answers_read',
        'accuracy_computed','LLM_generated','score_direction_selection','automatic_retry'])
    rows = [json.loads(line) for line in (c.EXEC/'per_question.jsonl').read_text().splitlines()]
    assert len(rows) == 16 and [r['position'] for r in rows] == plan['positions']
    own = c.read(c.EXEC/'identity.json')
    assert own['worker'] == state['worker'] and own['parent_controller'] == state['controller']
    assert own['plan_sha256'] == c.sha(c.PLAN) and own['allocation_note_sha256'] == state['allocation_note_sha256']
    runtime = c.read(c.EXEC/'runtime.json'); expected = plan['expected_runtime']
    assert runtime['torch'] == expected['torch_version'] == '2.1.2+cu118'
    assert runtime['cuda_build'] == expected['torch_cuda_build'] == '11.8'
    assert Path(runtime['torch_module']).resolve() == Path(expected['torch_module']).resolve()
    assert runtime['device'] == 'cuda:0' and runtime['GPU'] == 'NVIDIA A40' and runtime['CUDA_initialized']
    assert runtime['default_dtype'] == 'torch.float32' and runtime['autocast'] is False
    assert runtime['matmul_TF32'] is False and runtime['cudnn_TF32'] is True
    assert runtime['cudnn_benchmark'] == expected['cudnn_benchmark']
    assert runtime['cudnn_deterministic'] == expected['cudnn_deterministic']
    assert runtime['model_loaded'] is False and runtime['FA2_imported'] is False and runtime['forbidden_imports'] == []
    assert runtime['graph_cut'] == plan['graph_cut'] and runtime['anchor_boundary'] == plan['anchor_boundary']
    import numpy as np
    measured = ordered_changes = set_changes = sequence_changes = finite_arrays = 0
    details = []
    fields = ['importance','ordered_ids','gains','quotas','final_mask','spatial_ids',
              'final_projected_FP16_sequence','filtered_local_sim','selected_entropy','local_sim']
    for row,item in zip(rows,plan['records']):
        assert row['index'] == item['index'] and row['image'] == item['image'] and row['question_id'] == item['question_id']
        path = Path(row['arrays']['path'])
        assert path == c.EXEC/('image_%02d.npz' % item['index'])
        assert c.sha(path) == row['arrays']['sha256'] and path.stat().st_size == row['arrays']['bytes']
        with np.load(path,allow_pickle=False) as output,np.load(item['cache']['path'],allow_pickle=False) as original:
            for name in output.files:
                value = output[name]; meta = row['tensor_metadata'][name]
                assert list(value.shape) == meta['shape'] and hashlib.sha256(value.tobytes()).hexdigest() == meta['sha256']
                dtype = {'torch.float32':np.dtype('float32'),'torch.int64':np.dtype('int64'),
                         'torch.float16':np.dtype('float16'),'torch.bool':np.dtype('bool')}[meta['dtype']]
                assert value.dtype == dtype and meta['compute_device'] == 'cuda:0' and np.isfinite(value).all()
                finite_arrays += 1
            checks = source_checks(output,original,np)
            assert checks == row['source_anchor_checks']
            check_arm(output,'source',original['projected_fp16'],np)
            accepted = all(checks.values()); assert row['variant_measured'] is accepted
            arms = ['source','q03'] if accepted else ['source']
            assert set(output.files) == set(row['tensor_metadata']) == {arm+'__'+key for arm in arms for key in fields}
            for name,old in [('importance','importance'),('gains','gain_trace')]:
                a,b = output['source__'+name],original['alpha05_beta1__source__'+old]
                assert row['source_differences'][name]['bitwise_equal'] == exact(a,b)
                assert row['source_differences'][name]['max_abs_delta'] == float(np.abs(a-b).max())
            if accepted:
                assert row['status'] == 'measured'; measured += 1
                check_arm(output,'q03',original['projected_fp16'],np)
                comparison = row['source_vs_q03']
                for key in ['local_sim','importance','gains']:
                    a,b = output['source__'+key],output['q03__'+key]
                    assert comparison[key]['bitwise_equal'] == exact(a,b)
                    assert comparison[key]['max_abs_delta'] == float(np.abs(a-b).max())
                same = {key:exact(output['source__'+name],output['q03__'+name]) for key,name in [
                    ('ordered_ids_equal','ordered_ids'),('finalmask_equal','final_mask'),
                    ('final_spatial_ids_equal','spatial_ids'),
                    ('derived_final_FP16_sequence_equal','final_projected_FP16_sequence')]}
                assert all(comparison[key] == value for key,value in same.items())
                left,right = output['source__ordered_ids'][0],output['q03__ordered_ids'][0]
                first = next((i for i,(a,b) in enumerate(zip(left,right)) if a != b),None)
                stop = 32 if first is None else first+1
                a,b = output['source__gains'][:stop],output['q03__gains'][:stop]
                valid = (a > -1e8) & (b > -1e8)
                assert comparison['first_ordered_divergence'] == first and comparison['common_history_steps'] == stop
                assert comparison['common_history_max_abs_gain_delta_excluding_selected_sentinels'] == float(np.abs(a-b)[valid].max())
                for arm in ['source','q03']:
                    top2 = np.sort(output[arm+'__gains'],axis=-1)[...,-2:]
                    assert comparison[arm+'_min_argmax_margin'] == float((top2[...,1]-top2[...,0]).min())
                ordered_changes += not same['ordered_ids_equal']; set_changes += not same['finalmask_equal']
                sequence_changes += not same['derived_final_FP16_sequence_equal']
            else:
                assert row['status'] == 'source_anchor_failed_unmeasured' and row['source_vs_q03'] is None
                assert not any(name.startswith('q03__') for name in output.files)
            details.append(dict(position=row['position'],image=row['image'],question_id=row['question_id'],
                source_anchor_checks=checks,variant_measured=accepted,arrays=row['arrays']))
    assert summary['measured_questions'] == measured and summary['unmeasured_questions'] == 16-measured
    assert summary['status'] == ('complete' if measured == 16 else 'complete_with_explicit_unmeasured')
    assert not c.forbidden_imports() and 'torch' not in sys.modules
    c.check(plan)
    c.save(c.EXEC/'independent_verification.json',dict(success=True,n_questions=16,measured_questions=measured,
        unmeasured_questions=16-measured,all_variants_measured=measured==16,ordered_ids_changed=ordered_changes,
        final_spatial_sets_changed=set_changes,derived_final_FP16_sequences_changed=sequence_changes,
        finite_arrays_checked=finite_arrays,plan_sha256=c.sha(c.PLAN),summary_sha256=c.sha(c.EXEC/'summary.json'),
        per_question_sha256=c.sha(c.EXEC/'per_question.jsonl'),runtime_sha256=c.sha(c.EXEC/'runtime.json'),
        worker_original_exited=True,worker_GPU_pid_absent=True,CPU_only=True,torch_imported=False,
        model_loaded=False,GT_read=False,answers_read=False,scorer_invoked=False,records=details,
        limitations='Audit success includes explicit source-anchor failures as unmeasured. Target .5/1 source is a saved real-GPU helper baseline; original native encode anchor belongs to1/1. FP16 final sequences are lossless cache derivations, not actual LLM prepare captures. No CPU selector, answer, accuracy, paper argv or reproduction claim.'),exclusive=True)


if __name__ == '__main__':
    main()
