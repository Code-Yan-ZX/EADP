"""Independent CPU-only audit of actual 32 arrays, outcomes and native exit."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import common as c

def meta(a,dtype):
    return dict(dtype=dtype,shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())

def main():
    p=c.read(c.PLAN);c.plan_checks(p);state=c.read(c.HERE/'state.json')
    assert state['status']=='container_exited_validation_pending' and state['exit_code']==0
    assert state['plan_sha256']==c.sha(c.PLAN)
    sys.path.insert(0,str(c.PREV));from run_panel import identity,gpu_pids
    original=state['container_identity']['host_init'];now=identity(original['pid'])
    assert now is None or now['start_ticks']!=original['start_ticks']
    assert original['pid'] not in gpu_pids()
    out=c.HERE/'execution';summary=c.read(out/'summary.json')
    assert summary['success'] and summary['n_questions']==16 and summary['n_native_generations']==32
    assert not summary['benchmark_result'] and not summary['GT_read'] and not summary['scoring_started']
    assert summary['run_plan_sha256']==c.sha(c.PLAN)
    old=[json.loads(s) for s in Path(p['source_panel_records']).read_text().splitlines()]
    checked=[]
    for i,(item,oldrec) in enumerate(zip(p['panel'],old)):
        results=[];arrays=[]
        for arm in ('plain','eadp1024'):
            r=c.read(out/f'{i:02d}_{arm}.json');path=out/f'{i:02d}_{arm}.npz'
            assert r['position']==item['position'] and r['index']==item['index'] and r['arm']==arm
            assert r['status']=='raw_preserved' and r['native_failure'] is None and not r['observation_errors']
            assert c.sha(path)==r['archive_sha256'] and Path(r['archive'])==path
            with np.load(path,allow_pickle=False) as archive:
                data={k:archive[k] for k in archive.files}
            assert set(data)==set(r['arrays'])
            for key,a in data.items():
                expected=r['arrays'][key];assert meta(a,expected['dtype'])==expected
                if expected['dtype']=='torch.bfloat16':
                    assert a.dtype==np.uint16 and np.isfinite((a.astype(np.uint32)<<16).view(np.float32)).all()
                elif a.dtype.kind=='f':assert np.isfinite(a).all()
            from observer import validate_capture
            validate_capture(arm,r['counts'],r['observation_errors'],r['native_failure'],r['response'],r['arrays'],r['pruner_splits'],{key:True for key in r['generate_kwargs_keys']})
            for key,expected in item['processor_expected'].items():assert r['arrays']['processor_'+key]==expected
            expected_scalars=dict(do_sample=False,max_new_tokens=2048,top_p=.001,top_k=1,temperature=.01,repetition_penalty=1.)
            assert r['generate_non_tensor_kwargs']==expected_scalars
            expected_keys=set(expected_scalars)|({'input_ids','attention_mask','pixel_values','image_grid_thw'} if arm=='plain' else {'inputs_embeds','attention_mask'})
            assert set(r['generate_kwargs_keys'])==expected_keys
            assert np.array_equal(data['generate_attention_mask'],data['processor_attention_mask'])
            if arm=='plain':
                for key in ('input_ids','pixel_values','image_grid_thw'):assert np.array_equal(data['generate_'+key],data['processor_'+key])
            else:assert np.array_equal(data['generate_inputs_embeds'],data['prefill_inputs_embeds'])
            ids=data['output_ids'];suffix=data['output_suffix_ids'];inputs=data['processor_input_ids']
            assert ids.ndim==inputs.ndim==suffix.ndim==2 and ids.shape[0]==inputs.shape[0]==suffix.shape[0]==1
            assert np.array_equal(suffix,ids[:,inputs.shape[1]:] if arm=='plain' else ids)
            if arm=='plain':assert np.array_equal(ids[:,:inputs.shape[1]],inputs)
            results.append(r);arrays.append(data)
        a,b=arrays
        assert a['visual_features'].shape==(1024,4096) and b['pruner_input'].shape==(1024,4096)
        for data in arrays:
            assert data['prefill_inputs_embeds'].shape[:2]==data['processor_input_ids'].shape
            assert data['prefill_position_ids'].shape[-1]==data['processor_input_ids'].shape[-1]
        assert np.array_equal(b['pruner_input'],b['pruner_output']) and np.array_equal(b['pruner_input'],b['visual_features'])
        with np.load(oldrec['cache']['path'],allow_pickle=False) as prev:old_feature=prev['visual_features_BF16_bits']
        comp={'visual_features':np.array_equal(a['visual_features'],b['visual_features']),
              'prefill_inputs_embeds':np.array_equal(a['prefill_inputs_embeds'],b['prefill_inputs_embeds']),
              'prefill_position_ids':np.array_equal(a['prefill_position_ids'],b['prefill_position_ids']),
              'prefill_attention_mask':np.array_equal(a['prefill_attention_mask'],b['prefill_attention_mask']),
              'old_feature_anchor':np.array_equal(a['visual_features'],old_feature),
              'eadp_no_prune_feature_identity':True,
              'output_suffix_ids_equal':np.array_equal(a['output_suffix_ids'],b['output_suffix_ids']),
              'response_equal':results[0]['response']==results[1]['response']}
        comp={key:bool(value) for key,value in comp.items()}
        declared=summary['results'][i]
        assert declared['position']==item['position'] and declared['index']==item['index']
        assert declared['comparisons']==comp
        for j,tag in enumerate(('plain','eadp')):
            assert declared[tag+'_deepstack']==results[j]['prefill_deepstack_present']
            assert declared[tag+'_visual_mask']==results[j]['prefill_visual_mask_present']
        checked.append(dict(position=item['position'],comparisons=comp))
    c.plan_checks(p)
    target=out/'independent_verification.json';assert not target.exists()
    c.save(target,dict(success=True,n_questions=16,n_native_generations=32,plan_sha256=c.sha(c.PLAN),
        results=checked,CPU_only=True,model_loaded=False,GT_read=False,scoring_started=False,benchmark_result=False,
        output_suffix_slicing_verified=True,response_native_decoding_independently_rechecked=False,
        native_container_process_exited=True,controller_exit_requires_root_pty_proof=True,limits=p['limits']))
    print(json.dumps(dict(success=True,n_native_generations=32)))
if __name__=='__main__':main()
