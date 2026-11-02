"""One allocated GPU for fixed-cache pruning only; no model or generation."""
import json
import os
from pathlib import Path
import sys
import common as c


def main(note_path):
    plan = c.read(c.PLAN); c.check(plan)
    from run import gpu_pids
    note = c.read(note_path); claim = c.read(c.HERE/'launch.claim.json')
    assert c.sha(note_path) == claim['allocation_note_sha256'] and c.sha(c.PLAN) == claim['plan_sha256']
    assert c.identity(os.getppid()) == claim['controller'] and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    pids = gpu_pids(); c.validate_allocation(note, c.sha(c.PLAN), pids, {pid:c.identity(pid) for pid in pids})
    c.EXEC.mkdir(exist_ok=False)
    c.save(c.EXEC/'identity.json', dict(worker=c.identity(os.getpid()), parent_controller=claim['controller'],
        argv=sys.argv, cgroup=Path('/proc/self/cgroup').read_text(), plan_sha256=c.sha(c.PLAN),
        allocation_note_sha256=c.sha(note_path)), exclusive=True)
    import numpy as np
    import torch
    import helpers
    assert not torch.cuda.is_initialized() and not c.forbidden_imports()
    expected = plan['expected_runtime']
    assert torch.__version__ == expected['torch_version'] == '2.1.2+cu118' and torch.version.cuda == expected['torch_cuda_build'] == '11.8'
    assert Path(torch.__file__).resolve() == Path(expected['torch_module']).resolve()
    assert str(torch.get_default_dtype()) == 'torch.float32' and not torch.is_autocast_enabled()
    assert not torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    assert torch.backends.cudnn.benchmark == expected['cudnn_benchmark'] and torch.backends.cudnn.deterministic == expected['cudnn_deterministic']
    torch.set_num_threads(4)
    # First CUDA query occurs only after allocation/parent/source/cache guards.
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_name(0) == 'NVIDIA A40'
    c.save(c.EXEC/'runtime.json', dict(torch=torch.__version__, torch_module=torch.__file__, cuda_build=torch.version.cuda,
        device='cuda:0', GPU=torch.cuda.get_device_name(0), default_dtype=str(torch.get_default_dtype()),
        autocast=torch.is_autocast_enabled(), matmul_TF32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_TF32=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic, CUDA_initialized=torch.cuda.is_initialized(),
        threads=torch.get_num_threads(), model_loaded=False, FA2_imported=False, forbidden_imports=c.forbidden_imports(),
        graph_cut=plan['graph_cut'], anchor_boundary=plan['anchor_boundary']), exclusive=True)
    def array(value): return value.detach().cpu().contiguous().numpy()
    def exact(a,b):
        return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))
    def difference(a,b): return dict(bitwise_equal=exact(a,b), max_abs_delta=float((a.float()-b.float()).abs().max()))
    records = []
    with torch.inference_mode(), (c.EXEC/'per_question.jsonl').open('x') as output:
        for item in plan['records']:
            assert c.sha(item['cache']['path']) == item['cache']['sha256']
            cache = np.load(item['cache']['path'], allow_pickle=False)
            def load(name, dtype, shape):
                data = cache[name]
                assert data.dtype == np.dtype(dtype) and list(data.shape) == shape and np.isfinite(data).all()
                return torch.from_numpy(data.copy()).to('cuda:0')
            projected = load('projected_fp16', 'float16', [1,576,4096])
            sim = load('visual_sim_fp32', 'float32', [1,576,576])
            global_sim = load('global_fp32', 'float32', [1,576,1])
            dense = load('dense_fp32', 'float32', [1,576,1,77])
            prefix = 'alpha05_beta1__source__'
            saved_importance = load(prefix+'importance', 'float32', [1,576])
            saved_ids = load(prefix+'ids', 'int64', [1,32])
            saved_gains = load(prefix+'gain_trace', 'float32', [32,1,576])
            saved_mask = load(prefix+'final_mask', 'bool', [1,576])
            saved_spatial = saved_mask[0].nonzero(as_tuple=False).flatten()
            saved_sequence = projected[0,saved_spatial]
            values = {}; stored = {}
            def measure(arm,q):
                result = helpers.measure(torch,global_sim,dense,sim,projected,q=q)
                assert all(bool(torch.isfinite(v).all()) for v in result.values())
                assert result['selected_entropy'].shape == (1,1,max(1,int(77*q)))
                assert result['filtered_local_sim'].shape == (1,576,1,max(1,int(77*q)))
                assert len(set(result['ordered_ids'][0].tolist())) == 32 and int(result['final_mask'].sum()) == 32
                values[arm] = result
                stored.update({arm+'__'+key:array(value) for key,value in result.items()})
                return result
            source = measure('source',.2)
            checks = dict(importance=exact(source['importance'],saved_importance),
                ordered32=exact(source['ordered_ids'],saved_ids), full32_gain_trace=exact(source['gains'],saved_gains),
                finalmask=exact(source['final_mask'],saved_mask),
                derived_spatial_FP16_sequence=exact(source['final_projected_FP16_sequence'],saved_sequence))
            anchored = all(checks.values()); comparison = None
            if anchored:
                variant = measure('q03',.3)
                comparison = {key:difference(source[key],variant[key]) for key in ['local_sim','importance','gains']}
                comparison.update(ordered_ids_equal=exact(source['ordered_ids'],variant['ordered_ids']),
                    finalmask_equal=exact(source['final_mask'],variant['final_mask']),
                    final_spatial_ids_equal=exact(source['spatial_ids'],variant['spatial_ids']),
                    derived_final_FP16_sequence_equal=exact(source['final_projected_FP16_sequence'],variant['final_projected_FP16_sequence']))
                first = next((i for i,(a,b) in enumerate(zip(source['ordered_ids'][0].tolist(),variant['ordered_ids'][0].tolist())) if a != b),None)
                stop = 32 if first is None else first+1
                left,right = source['gains'][:stop],variant['gains'][:stop]
                valid = (left > -1e8) & (right > -1e8)
                comparison.update(first_ordered_divergence=first, common_history_steps=stop,
                    common_history_max_abs_gain_delta_excluding_selected_sentinels=float((left-right).abs()[valid].max()),
                    whole_trace_gain_delta_scope='Descriptive after divergent histories; selected -1e9 sentinel positions can differ, not numerical perturbation',
                    source_min_argmax_margin=float((source['gains'].topk(2,dim=-1).values[...,0]-source['gains'].topk(2,dim=-1).values[...,1]).min()),
                    q03_min_argmax_margin=float((variant['gains'].topk(2,dim=-1).values[...,0]-variant['gains'].topk(2,dim=-1).values[...,1]).min()))
            destination = c.EXEC / ('image_%02d.npz' % item['index'])
            np.savez_compressed(destination, **stored)
            record = dict(position=item['position'], index=item['index'], image=item['image'], question_id=item['question_id'],
                status='measured' if anchored else 'source_anchor_failed_unmeasured',
                source_anchor_checks={k:bool(v) for k,v in checks.items()}, variant_measured=anchored,
                source_differences=dict(importance=difference(source['importance'],saved_importance),gains=difference(source['gains'],saved_gains)),
                source_vs_q03=comparison, arrays=dict(path=str(destination),sha256=c.sha(destination),bytes=destination.stat().st_size),
                tensor_metadata={arm+'__'+key:dict(dtype=str(value.dtype),shape=list(value.shape),compute_device=str(value.device),
                    sha256=__import__('hashlib').sha256(array(value).tobytes()).hexdigest())
                    for arm,result in values.items() for key,value in result.items()})
            output.write(json.dumps(record)+'\n'); output.flush(); records.append(record)
            print(json.dumps(dict(completed=len(records),index=item['index'],status=record['status'])),flush=True)
    for entry in plan['references']+plan['implementations']+[r['cache'] for r in plan['records']]: assert c.sha(entry['path']) == entry['sha256']
    assert c.sha(note_path) == claim['allocation_note_sha256'] and not c.forbidden_imports()
    measured = sum(r['variant_measured'] for r in records)
    c.save(c.EXEC/'summary.json', dict(status='complete' if measured == 16 else 'complete_with_explicit_unmeasured',
        n_questions=16, measured_questions=measured, unmeasured_questions=16-measured, plan_sha256=c.sha(c.PLAN),
        per_question_sha256=c.sha(c.EXEC/'per_question.jsonl'), runtime_sha256=c.sha(c.EXEC/'runtime.json'),
        model_loaded=False,FA2_imported=False,GT_read=False,answers_read=False,accuracy_computed=False,LLM_generated=False,
        score_direction_selection=False,automatic_retry=False),exclusive=True)


if __name__ == '__main__':
    assert len(sys.argv) == 2
    try: main(Path(sys.argv[1]).resolve())
    except BaseException as error:
        c.save(c.HERE/'worker_failure.json',dict(status='terminal_failure_no_retry',error=repr(error),automatic_retry=False)); raise
