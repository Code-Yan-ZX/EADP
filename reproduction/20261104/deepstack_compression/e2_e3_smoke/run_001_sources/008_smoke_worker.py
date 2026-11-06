"""Locked diagnostic draft; no GT, scoring, or performance interpretation.

Only an independently reviewed launcher may enable this worker; missing external
approval always fails closed. No real-model run has yet validated this draft.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
E4=HERE.parent/'e4'
CONTRACT=HERE.parent/'e2_e3_contract'
sys.path[:0]=[str(HERE),str(E4),str(CONTRACT)]
from identity_worker import PLAN,WEIGHTS,check_e1,sha
from smoke_contract import select_smoke

ARMS=(('E2','000'),('E2','111'),('E2','100'),('E2','010'),('E2','001'),('E3','PERM'),('E3','MEAN'))

def selected(proof):
    rows=select_smoke(proof)
    return [next(r for r in rows if r['task']==task) for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')]

def run_diagnostic(a):
    proof=check_e1(a.e1_acceptance,a.e1_started,a.e1_terminal)
    panel=selected(proof); plan=json.loads(PLAN.read_text());recipe=plan['recipe']
    expected={'visual_token_num':256,'alpha':0.5,'beta':2.0,'min_pixels':1048576,
              'max_pixels':1048576,'max_new_tokens':2048,'use_custom_prompt':True,'use_vllm':False}
    if any(recipe.get(k)!=v for k,v in expected.items()):raise ValueError('recipe mismatch')
    for group in ('source_sha256','implementation_sha256'):
        for path,digest in plan[group].items():
            if sha(path)!=digest:raise ValueError('author source changed')
    weights=json.loads(WEIGHTS.read_text())['files'];stats={}
    if len(weights)!=4:raise ValueError('expected four model shards')
    for item in weights:
        path=Path(item['path']);s=path.stat()
        if s.st_size!=item['bytes'] or sha(path)!=item['expected_lfs_sha256']:raise ValueError('weight mismatch')
        stats[str(path)]=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    out=a.out.resolve()
    if not out.is_dir() or any((out/x).exists() for x in ('frozen.json','records.jsonl','generated.jsonl','arrays')):
        raise ValueError('invalid or reused output')
    # The container preflight executes the exact import chain used for a real
    # run, but does not instantiate a model or touch CUDA.
    sys.path[:0]=[str(HERE),str(E4),str(CONTRACT),plan['author_root'],str(Path(plan['author_root'])/'VLMEvalKit')]
    import inspect
    import torch
    import model_adapter
    from local_imports import load_local
    intervention_runner=load_local('intervention_runner');cache_fingerprints=load_local('cache_fingerprints')
    from model import pruner as native
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP,Qwen3VLChatCDPruner
    imports={'model_adapter':Path(model_adapter.__file__).resolve(),'intervention_runner':Path(intervention_runner.__file__).resolve(),
             'cache_fingerprints':Path(cache_fingerprints.__file__).resolve(),'native_pruner':Path(native.__file__).resolve(),
             'author_wrapper':Path(inspect.getsourcefile(Qwen3VLChatEADP)).resolve()}
    wanted={'model_adapter':CONTRACT/'model_adapter.py','intervention_runner':HERE/'intervention_runner.py',
            'cache_fingerprints':HERE/'cache_fingerprints.py','native_pruner':Path(plan['author_root'])/'model/pruner.py',
            'author_wrapper':Path(plan['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py'}
    if imports!={k:v.resolve() for k,v in wanted.items()}:raise ImportError('unexpected model import paths')
    versions={n:importlib.metadata.version(n) for n in plan['expected_versions']}
    if versions!=plan['expected_versions']:raise ValueError('version mismatch')
    if a.preflight_only:
        if torch.cuda.is_initialized():raise ValueError('preflight initialized CUDA')
        print(json.dumps({'status':'preflight_passed','model_loaded':False,'predictions':False,'cuda_initialized':False,
                          'imports':{k:str(v) for k,v in imports.items()},'versions':versions}),flush=True)
        return
    sources=[HERE/'smoke_worker.py',HERE/'intervention_runner.py',HERE/'cache_fingerprints.py',CONTRACT/'model_adapter.py',CONTRACT/'interventions.py']
    sources += [E4/x for x in ('smoke_contract.py','budget_adapter.py','decode_observer.py','off_adapter.py',
                                'skip_deepstack.py','identity_records.py','identity_worker.py')]
    frozen={'e1_acceptance_sha256':sha(a.e1_acceptance),'plan_sha256':sha(PLAN),
            'weights_identity_sha256':sha(WEIGHTS),'weight_stats_before':stats,
            'source_sha256':{str(x):sha(x) for x in sources},
            'selected':[{k:r[k] for k in ('task','qid','group_id','message_sha256','image_sha256')} for r in panel],
            'arms':[list(x) for x in ARMS],'forced_reference_arms':['D0','D1'],'forced_text':' the','forced_decode_steps':32,
            'GT_opened':False,'scoring_performed':False,'timing_claims':False}
    (out/'arrays').mkdir()
    with (out/'frozen.json').open('x') as f:json.dump(frozen,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    run_arm=intervention_runner.run_arm
    if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
    wrapper=Qwen3VLChatEADP(**recipe)
    if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('backend mismatch')
    scalars={k:recipe[k] for k in ('top_p','top_k','temperature','repetition_penalty')};scalars.update(do_sample=False,max_new_tokens=2048)
    def generate(w,row):
        with torch.inference_mode():return Qwen3VLChatCDPruner.generate_inner_transformers_fixed(w,row['message'],dataset=row['task'])
    token=wrapper.processor.tokenizer.encode(' the',add_special_tokens=False)
    if not token:raise ValueError('forced token missing')
    def one(row,family,arm,forced):
        name=hashlib.sha256(json.dumps([row['task'],str(row['qid']),family,arm,forced],separators=(',',':')).encode()).hexdigest()+'.npz'
        def journal(record):
            record.update(family=family,arm=arm)
            with (out/'generated.jsonl').open('a') as f:
                f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
        def archive(tensors):
            import numpy as np
            path=out/'arrays'/name
            payload={key:value.detach().contiguous().cpu().view(torch.uint8).numpy() for key,value in tensors.items()}
            with path.open('xb') as f:np.savez(f,**payload);f.flush();os.fsync(f.fileno())
            return {'kind':'full_arrays','path':str(path),'sha256':sha(path)}
        record=run_arm(torch,wrapper,native,row,family,arm,generate,journal,archive,
                       forced_token=int(token[0]) if forced else None,capture_prefill_logits=not forced)
        expected=dict(scalars)
        if forced:expected['max_new_tokens']=33
        if record['generation_scalars']!=expected:raise ValueError('generation recipe mismatch')
        with (out/'records.jsonl').open('a') as f:
            f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
    for item in panel:
        row=dict(item,split='DEV')
        for family,arm in ARMS:
            one(row,family,arm,False);one(row,family,arm,True)
        for arm in ('D0','D1'):one(row,'REF',arm,True)
        print('completed diagnostic question',row['task'],row['qid'],flush=True)
    for path,before in stats.items():
        s=Path(path).stat()
        if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weights changed')
    if any(sha(p)!=h for p,h in frozen['source_sha256'].items()):raise ValueError('source changed')
    with (out/'worker_complete.json').open('x') as f:
        json.dump({'questions':3,'natural_arms':21,'forced_arms':27},f);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
    p=argparse.ArgumentParser()
    for name in ('e1-acceptance','e1-started','e1-terminal','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args()
    if not a.execute:raise ValueError('explicit execution required')
    from local_imports import load_local
    load_local('gate').check_approval(preflight=a.preflight_only)
    run_diagnostic(a)

if __name__=='__main__':main()
