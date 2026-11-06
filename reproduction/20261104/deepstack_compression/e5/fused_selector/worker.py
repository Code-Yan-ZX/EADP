"""Bounded GPU numeric-input probe; never modifies the production model."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT=Path('/media/disk2/YZX/research/EADP_v2');BASE=ROOT/'reproduction/20261104/deepstack_compression';HERE=BASE/'e5/fused_selector';AUTHOR=ROOT/'_reproduction/author_eadp/Qwen_vl'
sys.path.insert(0,str(AUTHOR))
os.environ['TRITON_CACHE_DIR']=str(HERE/'triton_cache')


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def save(p,obj):
    with open(p,'x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


def append(p,obj):
    with open(p,'a') as f:f.write(json.dumps(obj)+'\n');f.flush();os.fsync(f.fileno())


def preflight(plan):
    for p,h in plan['sha256'].items():
        if sha(p)!=h:raise ValueError('source mismatch: '+p)
    result=json.loads((BASE/'e5/selector_fused_cpu/result001.json').read_text())
    path=BASE/'e5/selector_fused_cpu/numeric_inputs001.pt'
    if (result['status']!='completed' or result['cpu_vs_gpu_same_order'] is not True or
            result['artifact_sha256']!=sha(path)):
        raise ValueError('numeric input provenance or identity')
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plan=json.loads((HERE/'plan001.json').read_text());path=preflight(plan)
    import torch,triton
    from kernel import fused_greedy,validate
    from model.pruner import _greed_select_impl
    data=torch.load(path,map_location='cpu',weights_only=True)
    sim=data['similarity'];importance=data['importance'];expected=data['gpu_accepted_ordered']
    if sim.shape!=(1,1024,1024) or importance.shape!=(1,1024) or expected.shape!=(256,):raise ValueError('numeric shape')
    validate(sim[0],importance[0],256)
    if (not torch.isfinite(sim).all() or not torch.isfinite(importance).all() or
            sim.min().item()<0 or sim.max().item()>1 or importance.min().item()<0 or
            expected.unique().numel()!=256):raise ValueError('numeric assumptions')
    if a.preflight_only:
        if torch.cuda.is_initialized():raise ValueError('CUDA initialized')
        save(a.out,{'status':'passed','source_plan_sha256':sha(HERE/'plan001.json'),'cuda_initialized':False,
                    'model_loaded':False,'GT_opened':False,'input_sha256_verified':sha(path),
                    'torch':torch.__version__,'triton':triton.__version__});return
    if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('GPU mismatch')
    a.out.mkdir(exist_ok=False);append(a.out/'progress.jsonl',{'event':'worker_started'})
    save(a.out/'started.json',{'plan_sha256':sha(HERE/'plan001.json'),'GT_opened':False,'model_loaded':False,
                              'device':torch.cuda.get_device_name(0),'numeric_input_sha256':sha(path)})
    sim=sim.to('cuda');importance=importance.to('cuda');expected=expected.to('cuda')
    def original():return _greed_select_impl(importance,sim,256)[0][0]
    def fused():return fused_greedy(sim[0],importance[0],256)[0]
    with torch.inference_mode():
        torch.manual_seed(20261104)
        near=torch.ones((8,8),device='cuda',dtype=torch.float32)*.5
        near[2,0]=.50000012;near[3,0]=.50000006
        toy=[('all_tied',torch.ones((8,8),device='cuda'),torch.ones(8,device='cuda'),4),
             ('zero_gain',torch.zeros((8,8),device='cuda'),torch.zeros(8,device='cuda'),4),
             ('near_tied',near,torch.ones(8,device='cuda'),4),
             ('seeded_random',torch.rand((32,32),device='cuda'),torch.rand(32,device='cuda'),8)]
        toy_results=[]
        for label,s,w,k in toy:
            author=_greed_select_impl(w.unsqueeze(0),s.unsqueeze(0),k)[0][0]
            candidate=fused_greedy(s,w,k)[0]
            equal=bool(torch.equal(author,candidate))
            toy_results.append({'case':label,'author':author.tolist(),'fused':candidate.tolist(),'equal':equal})
        save(a.out/'synthetic_identity.json',toy_results)
        append(a.out/'progress.jsonl',{'event':'synthetic_identity_completed'})
        if not all(x['equal'] for x in toy_results):return 2
        torch.cuda.synchronize();original_warm=original();torch.cuda.synchronize()
        append(a.out/'progress.jsonl',{'event':'original_warmed'})
        fused_warm=fused();torch.cuda.synchronize();append(a.out/'progress.jsonl',{'event':'fused_compiled_and_warmed'})
        preliminary={'original_equals_accepted':bool(torch.equal(original_warm,expected)),
                     'fused_equals_original':bool(torch.equal(fused_warm,original_warm)),
                     'original_indices':original_warm.tolist(),'fused_indices':fused_warm.tolist()}
        save(a.out/'identity.json',preliminary)
        if not all(preliminary[x] for x in ('original_equals_accepted','fused_equals_original')):
            append(a.out/'progress.jsonl',{'event':'identity_failed'});return 2
        for repeat in range(5):
            for name,fn in ([('original',original),('fused',fused)] if repeat%2==0 else [('fused',fused),('original',original)]):
                torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                t0=time.perf_counter();start.record();selected=fn();end.record();end.synchronize();wall=(time.perf_counter()-t0)*1000
                same=bool(torch.equal(selected,expected))
                append(a.out/'measurements.jsonl',{'repeat':repeat,'implementation':name,'cuda_event_ms':start.elapsed_time(end),
                       'host_wall_ms':wall,'equals_accepted_order':same,'selected_indices':selected.tolist()})
                append(a.out/'progress.jsonl',{'event':'measured','implementation':name,'repeat':repeat,'equals_accepted_order':same})
                if not same:raise ValueError('selected order changed during measurement')
    if any(sha(p)!=h for p,h in plan['sha256'].items()):raise ValueError('source changed')
    save(a.out/'worker_complete.json',{'status':'completed','measurements':10,'all_same_order':True,
         'limitations':'single numeric input; timing includes launches, allocator and warmup context; not model end-to-end'})
    return 0


if __name__=='__main__':sys.exit(main())
