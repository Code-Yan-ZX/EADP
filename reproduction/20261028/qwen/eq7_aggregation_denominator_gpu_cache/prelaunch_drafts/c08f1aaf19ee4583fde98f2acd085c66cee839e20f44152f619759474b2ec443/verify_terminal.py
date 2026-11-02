"""Independent NumPy terminal audit. Never reruns pruning, a model or scoring."""
import hashlib
import json
import os
from pathlib import Path
import sys
import common as c


def source_checks(source,original,np):
    return dict(importance=np.array_equal(source['source__importance'],original['native_importance']),
        ordered256=np.array_equal(source['source__ordered_ids'],original['native_ordered_ids']),
        full256_gain_trace=np.array_equal(source['source__gains'],original['source__gain_trace']),
        final_spatial_BF16_sequence=np.array_equal(source['source__final_BF16_bits'],original['native_pruned_BF16_bits']))


def check_arm(arrays,arm,features,np):
    ids=arrays[arm+'__ordered_ids'];gains=arrays[arm+'__gains'];spatial=arrays[arm+'__spatial_ids']
    assert ids.dtype==np.int64 and ids.shape==(1,256) and len(set(ids[0].tolist()))==256
    assert np.all((ids>=0)&(ids<1024)) and gains.dtype==np.float32 and gains.shape==(256,1,1024)
    assert np.array_equal(gains.argmax(axis=-1).reshape(1,256),ids)
    assert np.array_equal(np.sort(ids[0]),spatial)
    assert np.array_equal(features[spatial],arrays[arm+'__final_BF16_bits'])
    assert arrays[arm+'__importance'].dtype==np.float32 and arrays[arm+'__importance'].shape==(1,1024)
    for step in range(256):assert np.all(gains[step,0,ids[0,:step]]==np.float32(-1e9))


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan=c.read(c.PLAN);c.check(plan)
    state=c.read(c.HERE/'state.json')
    assert state['status']=='worker_exited_validation_pending' and state['worker_exit_code']==0
    assert not (c.HERE/'failure.json').exists() and not (c.HERE/'worker_failure.json').exists()
    from run import gpu_pids
    assert c.identity(state['worker']['pid'])!=state['worker'] and state['worker']['pid'] not in gpu_pids()
    summary=c.read(c.EXEC/'summary.json')
    assert summary['status'] in ['complete','complete_with_explicit_unmeasured'] and summary['n_questions']==16
    assert summary['plan_sha256']==c.sha(c.PLAN)
    assert summary['per_question_sha256']==c.sha(c.EXEC/'per_question.jsonl') and summary['runtime_sha256']==c.sha(c.EXEC/'runtime.json')
    assert all(summary[k] is False for k in ['model_loaded','FA2_imported','GT_read','answers_read','accuracy_computed','LLM_generated','score_direction_selection','automatic_retry'])
    rows=[json.loads(line) for line in (c.EXEC/'per_question.jsonl').read_text().splitlines()]
    assert len(rows)==16 and [r['position'] for r in rows]==plan['positions']
    own=c.read(c.EXEC/'identity.json')
    assert own['worker']==state['worker'] and own['parent_controller']==state['controller']
    assert own['plan_sha256']==c.sha(c.PLAN) and own['allocation_note_sha256']==state['allocation_note_sha256']
    runtime=c.read(c.EXEC/'runtime.json');assert runtime['torch']=='2.8.0+cu128' and runtime['cuda_build']=='12.8'
    assert runtime['device']=='cuda:0' and runtime['GPU']=='NVIDIA A40' and runtime['CUDA_initialized']
    assert runtime['default_dtype']=='torch.float32' and not runtime['autocast'] and not runtime['matmul_TF32']
    assert runtime['model_loaded'] is False and runtime['FA2_imported'] is False and runtime['forbidden_imports']==[]
    import numpy as np
    measured=ordered_changes=set_changes=0;details=[]
    for row,item in zip(rows,plan['records']):
        assert row['index']==item['index'] and c.sha(row['arrays']['path'])==row['arrays']['sha256']
        assert Path(row['arrays']['path']).resolve().parent==c.EXEC.resolve()
        with np.load(row['arrays']['path'],allow_pickle=False) as output,np.load(item['cache']['path'],allow_pickle=False) as original:
            for name in output.files:
                array=output[name];meta=row['tensor_metadata'][name]
                assert list(array.shape)==meta['shape'] and hashlib.sha256(array.tobytes()).hexdigest()==meta['sha256']
                expected_dtype={'torch.float32':np.dtype('float32'),'torch.int64':np.dtype('int64'),'torch.bfloat16':np.dtype('uint16')}[meta['dtype']]
                assert array.dtype==expected_dtype
                finite_value=(array.astype(np.uint32)<<16).view(np.float32) if meta['dtype']=='torch.bfloat16' else array
                assert meta['compute_device']=='cuda:0' and np.isfinite(finite_value).all()
            checks=source_checks(output,original,np);assert checks==row['source_anchor_checks']
            check_arm(output,'source',original['visual_features_BF16_bits'],np)
            accepted=all(checks.values());assert row['variant_measured'] is accepted
            fields=['importance','ordered_ids','gains','spatial_ids','final_BF16_bits','logits','weights','local_sim','entropy']
            expected_keys={arm+'__'+key for arm in (['source','eq7_exact_denominator'] if accepted else ['source']) for key in fields}
            assert set(output.files)==set(row['tensor_metadata'])==expected_keys
            if accepted:
                assert row['status']=='measured';measured+=1
                check_arm(output,'eq7_exact_denominator',original['visual_features_BF16_bits'],np)
                assert np.array_equal(output['source__entropy'],output['eq7_exact_denominator__entropy'])
                order=np.array_equal(output['source__ordered_ids'],output['eq7_exact_denominator__ordered_ids'])
                same=np.array_equal(output['source__spatial_ids'],output['eq7_exact_denominator__spatial_ids'])
                assert order==row['source_vs_eq7']['ordered_ids_equal'] and same==row['source_vs_eq7']['final_spatial_ids_equal']
                left=output['source__ordered_ids'][0];right=output['eq7_exact_denominator__ordered_ids'][0]
                first=next((i for i,(a,b) in enumerate(zip(left,right)) if a!=b),None)
                stop=256 if first is None else first+1
                source_gains=output['source__gains'][:stop];variant_gains=output['eq7_exact_denominator__gains'][:stop]
                valid=(source_gains>-1e8)&(variant_gains>-1e8)
                assert row['source_vs_eq7']['first_ordered_divergence']==first and row['source_vs_eq7']['common_history_steps']==stop
                assert row['source_vs_eq7']['common_history_max_abs_gain_delta_excluding_selected_sentinels']==float(np.abs(source_gains-variant_gains)[valid].max())
                ordered_changes+=not order;set_changes+=not same
            else:
                assert row['status']=='source_anchor_failed_unmeasured' and row['source_vs_eq7'] is None
                assert not any(name.startswith('eq7_') for name in output.files)
            details.append(dict(position=row['position'],source_anchor_checks=checks,variant_measured=accepted,arrays=row['arrays']))
    assert summary['measured_questions']==measured and summary['unmeasured_questions']==16-measured
    assert not c.forbidden_imports() and 'torch' not in sys.modules
    c.check(plan)
    c.save(c.EXEC/'independent_verification.json',dict(success=True,n_questions=16,measured_questions=measured,
        unmeasured_questions=16-measured,all_variants_measured=measured==16,ordered_ids_changed=ordered_changes,
        final_spatial_sets_changed=set_changes,plan_sha256=c.sha(c.PLAN),summary_sha256=c.sha(c.EXEC/'summary.json'),
        per_question_sha256=c.sha(c.EXEC/'per_question.jsonl'),runtime_sha256=c.sha(c.EXEC/'runtime.json'),
        worker_original_exited=True,worker_GPU_pid_absent=True,CPU_only=True,torch_imported=False,
        model_loaded=False,GT_read=False,answers_read=False,scorer_invoked=False,records=details,
        limitations='Audit success includes explicit source-anchor failures as unmeasured. No CPU approximation, answer, accuracy, LLM or paper-reproduction claim.'),exclusive=True)


if __name__=='__main__':main()
