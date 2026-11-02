"""One allocated GPU, cached pruning only. No model or answer-generation code."""
import json
import os
from pathlib import Path
import sys
import common as c


def main(note_path):
    plan=c.read(c.PLAN);c.check(plan)
    from run import gpu_pids
    note=c.read(note_path);claim=c.read(c.HERE/'launch.claim.json')
    assert c.sha(note_path)==claim['allocation_note_sha256'] and c.sha(c.PLAN)==claim['plan_sha256']
    assert c.identity(os.getppid())==claim['controller']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    pids=gpu_pids();c.validate_allocation(note,c.sha(c.PLAN),pids,{pid:c.identity(pid) for pid in pids})
    c.EXEC.mkdir(exist_ok=False)
    own=c.identity(os.getpid())
    c.save(c.EXEC/'identity.json',dict(worker=own,parent_controller=claim['controller'],argv=sys.argv,
        cgroup=Path('/proc/self/cgroup').read_text(),plan_sha256=c.sha(c.PLAN),allocation_note_sha256=c.sha(note_path)),exclusive=True)
    import numpy as np
    import torch
    assert not torch.cuda.is_initialized() and not c.forbidden_imports()
    expected=plan['expected_runtime']
    assert torch.__version__==expected['torch_version'] and torch.version.cuda==expected['torch_cuda_build']
    assert Path(torch.__file__).resolve()==Path(expected['torch_module']).resolve()
    assert str(torch.get_default_dtype())=='torch.float32' and not torch.is_autocast_enabled()
    assert not torch.backends.cuda.matmul.allow_tf32
    assert torch.backends.cudnn.allow_tf32==expected['cudnn_TF32']
    assert torch.backends.cudnn.benchmark==expected['cudnn_benchmark'] and torch.backends.cudnn.deterministic==expected['cudnn_deterministic']
    torch.set_num_threads(4)
    # This is the first CUDA access, after the exact root allocation and parent
    # process/plan/note identities have passed all host guards.
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA A40'
    c.save(c.EXEC/'runtime.json',dict(torch=torch.__version__,torch_module=torch.__file__,cuda_build=torch.version.cuda,
        device='cuda:0',GPU=torch.cuda.get_device_name(0),default_dtype=str(torch.get_default_dtype()),
        autocast=torch.is_autocast_enabled(),matmul_TF32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_TF32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,CUDA_initialized=torch.cuda.is_initialized(),
        threads=torch.get_num_threads(),model_loaded=False,FA2_imported=False,forbidden_imports=c.forbidden_imports(),
        graph_cut=plan['graph_cut']),exclusive=True)
    helper=c.helper();functions,compute,changed,observed,_=helper.nodes()
    records=[]
    def array(value):
        actual=value.detach().cpu().contiguous()
        return actual.view(torch.uint16).numpy() if actual.dtype==torch.bfloat16 else actual.numpy()
    def exact(a,b):
        return a.dtype==b.dtype and a.shape==b.shape and bool(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)))
    def difference(a,b):
        return dict(bitwise_equal=exact(a,b),max_abs_delta=float((a.float()-b.float()).abs().max()))
    with torch.inference_mode(),(c.EXEC/'per_question.jsonl').open('x') as output:
        for number,item in enumerate(plan['records']):
            assert c.sha(item['cache']['path'])==item['cache']['sha256']
            cache=np.load(item['cache']['path'],allow_pickle=False)
            def load(name):
                data=cache[name];meta=item['tensors'][name]
                import hashlib
                assert hashlib.sha256(data.tobytes()).hexdigest()==meta['sha256']
                value=torch.from_numpy(data.copy())
                if meta['dtype']=='torch.bfloat16':value=value.view(torch.bfloat16)
                assert str(value.dtype)==meta['dtype'] and list(value.shape)==meta['shape']
                return value.to('cuda:0')
            features=load('visual_features_BF16_bits');text=load('text_sequence_BF16_bits');global_text=load('text_global_mean_BF16_bits')
            global_sim=load('global_sim');dense=load('dense_sim');sim=load('visual_sim')
            native_importance=load('native_importance');native_ids=load('native_ordered_ids');native_gains=load('source__gain_trace');native_final=load('native_pruned_BF16_bits')
            values={};stored={}
            def measure(name,aggregation):
                namespace,plain=helper.compiled(torch,functions,compute,aggregation,observed,global_sim,dense)
                from types import SimpleNamespace
                importance=namespace['compute_importance'](SimpleNamespace(alpha=.5,beta=2.),features.unsqueeze(0),global_text,text,32,32)
                ids,quota=namespace['_greed_select_impl'](importance,sim,256)
                plain_ids,plain_quota=plain(importance,sim,256)
                assert torch.equal(ids,plain_ids) and torch.equal(quota,plain_quota) and quota.tolist()==[256]
                gains=torch.stack(namespace['gain_trace'])
                spatial=ids[0].sort().values;final=features[spatial]
                value=dict(importance=importance,ordered_ids=ids,gains=gains,spatial_ids=spatial,final_BF16_bits=final,**namespace['aggregation_capture'])
                assert all(bool(torch.isfinite(v).all()) for v in value.values())
                assert len(set(ids[0].tolist()))==256
                values[name]=value
                stored.update({name+'__'+key:array(v) for key,v in value.items()})
                return value
            source=measure('source',functions['_local_aggregation_impl'])
            checks=dict(importance=exact(source['importance'],native_importance),
                ordered256=exact(source['ordered_ids'],native_ids),full256_gain_trace=exact(source['gains'],native_gains),
                final_spatial_BF16_sequence=exact(source['final_BF16_bits'],native_final))
            anchored=all(checks.values());comparison=None
            if anchored:
                eq7=measure('eq7_exact_denominator',changed)
                comparison={key:difference(source[key],eq7[key]) for key in ['weights','local_sim','importance','gains']}
                comparison.update(ordered_ids_equal=torch.equal(source['ordered_ids'],eq7['ordered_ids']),
                    final_spatial_ids_equal=torch.equal(source['spatial_ids'],eq7['spatial_ids']),
                    final_BF16_sequence_equal=exact(source['final_BF16_bits'],eq7['final_BF16_bits']))
                first=next((i for i,(a,b) in enumerate(zip(source['ordered_ids'][0].tolist(),eq7['ordered_ids'][0].tolist())) if a!=b),None)
                stop=256 if first is None else first+1
                left=source['gains'][:stop];right=eq7['gains'][:stop]
                valid=(left>-1e8)&(right>-1e8)
                comparison.update(first_ordered_divergence=first,common_history_steps=stop,
                    common_history_max_abs_gain_delta_excluding_selected_sentinels=float((left-right).abs()[valid].max()),
                    whole_trace_gain_delta_scope='Descriptive only after divergent history; -1e9 selected-mask locations can differ',
                    source_min_argmax_margin=float((source['gains'].topk(2,dim=-1).values[...,0]-source['gains'].topk(2,dim=-1).values[...,1]).min()),
                    Eq7_min_argmax_margin=float((eq7['gains'].topk(2,dim=-1).values[...,0]-eq7['gains'].topk(2,dim=-1).values[...,1]).min()))
            destination=c.EXEC/f'question_{number:02d}.npz'
            np.savez_compressed(destination,**stored)
            record=dict(position=item['position'],index=item['index'],status='measured' if anchored else 'source_anchor_failed_unmeasured',
                source_anchor_checks={k:bool(v) for k,v in checks.items()},variant_measured=anchored,
                source_differences=dict(importance=difference(source['importance'],native_importance),
                    gains=difference(source['gains'],native_gains)),source_vs_eq7=comparison,
                arrays=dict(path=str(destination),sha256=c.sha(destination),bytes=destination.stat().st_size),
                tensor_metadata={name:dict(dtype=str(value.dtype),shape=list(value.shape),
                    compute_device=str(value.device),sha256=__import__('hashlib').sha256(array(value).tobytes()).hexdigest())
                    for arm,values_ in values.items() for key,value in values_.items() for name in [arm+'__'+key]})
            output.write(json.dumps(record)+'\n');output.flush();records.append(record)
            print(json.dumps(dict(completed=len(records),position=item['position'],status=record['status'])),flush=True)
    for entry in plan['references']+plan['implementations']+[r['cache'] for r in plan['records']]:assert c.sha(entry['path'])==entry['sha256']
    assert c.sha(note_path)==claim['allocation_note_sha256']
    assert not c.forbidden_imports()
    measured=sum(r['variant_measured'] for r in records)
    c.save(c.EXEC/'summary.json',dict(status='complete' if measured==16 else 'complete_with_explicit_unmeasured',n_questions=16,
        measured_questions=measured,unmeasured_questions=16-measured,plan_sha256=c.sha(c.PLAN),
        per_question_sha256=c.sha(c.EXEC/'per_question.jsonl'),runtime_sha256=c.sha(c.EXEC/'runtime.json'),
        model_loaded=False,FA2_imported=False,GT_read=False,answers_read=False,accuracy_computed=False,LLM_generated=False,
        score_direction_selection=False,automatic_retry=False),exclusive=True)


if __name__=='__main__':
    assert len(sys.argv)==2
    try:main(Path(sys.argv[1]).resolve())
    except BaseException as error:
        c.save(c.HERE/'worker_failure.json',dict(status='terminal_failure_no_retry',error=repr(error),automatic_retry=False))
        raise
