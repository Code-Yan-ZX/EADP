"""Prepared E4-A 32-question smoke worker; launcher remains locked.

No labels or scores. Natural output diagnostics only, not a performance run.
Caller must establish real controller/container termination separately.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from identity_worker import HERE, PLAN, WEIGHTS, check_e1, sha
from smoke_contract import select_smoke


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    p.add_argument('--e1-acceptance',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--e1-acceptance-started',type=Path,required=True);p.add_argument('--e1-acceptance-terminal',type=Path,required=True)
    a=p.parse_args()
    if not a.execute:raise ValueError('explicit execute required')
    from smoke_launch import check_approval
    check_approval()
    proof=check_e1(a.e1_acceptance,a.e1_acceptance_started,a.e1_acceptance_terminal);panel=select_smoke(proof)
    plan=json.loads(PLAN.read_text());recipe=plan['recipe']
    expected={'visual_token_num':256,'alpha':0.5,'beta':2.0,'min_pixels':1048576,'max_pixels':1048576,
              'max_new_tokens':2048,'use_custom_prompt':True,'use_vllm':False}
    if any(recipe.get(k)!=v for k,v in expected.items()):raise ValueError('unexpected recipe')
    for group in ('source_sha256','implementation_sha256'):
        for path,digest in plan[group].items():
            if sha(path)!=digest:raise ValueError('source changed: '+path)
    weights=json.loads(WEIGHTS.read_text())['files'];stats={}
    if len(weights)!=4:raise ValueError('expected four weight shards')
    for item in weights:
        path=Path(item['path']);s=path.stat()
        if s.st_size!=item['bytes'] or sha(path)!=item['expected_lfs_sha256']:raise ValueError('weight identity mismatch')
        stats[str(path)]=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    out=a.out.resolve()
    if not out.is_dir() or any((out/x).exists() for x in ('frozen.json','records.jsonl','forced_records.jsonl','generated.jsonl','arrays')):
        raise ValueError('invalid or reused output')
    (out/'arrays').mkdir()
    sources=[HERE/x for x in ('smoke_worker.py','smoke_contract.py','budget_runner.py','budget_adapter.py','decode_observer.py',
                               'off_adapter.py','skip_deepstack.py','identity_records.py','identity_worker.py')]
    frozen={'stage':'smoke','n_questions':32,'e1_acceptance_sha256':sha(a.e1_acceptance),'plan_sha256':sha(PLAN),
            'weights_identity_sha256':sha(WEIGHTS),'weight_stats_before':stats,'code_sha256':{str(x):sha(x) for x in sources},
            'selected':[{k:r[k] for k in ('task','qid','group_id','message_sha256','image_sha256')} for r in panel],
            'budgets':[128,256,512,1024],'ds_on':[False,True],
            'forced_anchor':{k:panel[0][k] for k in ('task','qid','group_id')},'forced_text':' the','forced_decode_steps':32,
            'gt_read':False,'scoring':False,'timing_claims':False}
    with (out/'frozen.json').open('x') as f:json.dump(frozen,f,indent=2,ensure_ascii=False);f.write('\n')
    sys.path[:0]=[str(HERE),plan['author_root'],str(Path(plan['author_root'])/'VLMEvalKit')]
    import torch
    from model import pruner as native
    from budget_runner import run_arm
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP,Qwen3VLChatCDPruner
    if {n:importlib.metadata.version(n) for n in plan['expected_versions']}!=plan['expected_versions']:
        raise ValueError('runtime versions changed')
    if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
    wrapper=Qwen3VLChatEADP(**recipe)
    if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('unexpected backend')
    expected_scalars={key:recipe[key] for key in ('top_p','top_k','temperature','repetition_penalty')}
    expected_scalars.update(do_sample=False,max_new_tokens=2048)
    def generate(w,row):
        with torch.inference_mode():
            return Qwen3VLChatCDPruner.generate_inner_transformers_fixed(w,row['message'],dataset=row['task'])
    def on_generated(record):
        with (out/'generated.jsonl').open('a') as f:
            f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
    def one(row,budget,on,token=None):
        wrapper.visual_token_num=budget;wrapper.pruner.visual_token_num=budget
        forced=token is not None
        name=hashlib.sha256(json.dumps([row['task'],str(row['qid']),budget,on,forced],separators=(',',':')).encode()).hexdigest()+'.npz'
        def archive(tensors):
            import numpy as np
            path=out/'arrays'/name
            payload={key:value.detach().contiguous().cpu().view(torch.uint8).numpy() for key,value in tensors.items()}
            with path.open('xb') as f:
                np.savez(f,**payload);f.flush();os.fsync(f.fileno())
            return {'kind':'full_arrays','path':str(path),'sha256':sha(path)}
        anchor_natural=not forced and row['task']==panel[0]['task'] and str(row['qid'])==str(panel[0]['qid'])
        record=run_arm(torch,wrapper,native,row,budget,on,generate,on_generated,archive,
                       forced_token=token,capture_prefill_logits=anchor_natural)
        expected=dict(expected_scalars)
        if forced:expected['max_new_tokens']=33
        if record['generation_scalars']!=expected:raise ValueError('actual generation recipe mismatch')
        with (out/('forced_records.jsonl' if forced else 'records.jsonl')).open('a') as f:
            f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
    for item in panel:
        row=dict(item,split='DEV')
        for budget in (128,256,512,1024):
            for on in (False,True):one(row,budget,on)
        print('completed smoke question',row['task'],row['qid'],flush=True)
    encoded=wrapper.processor.tokenizer.encode(' the',add_special_tokens=False)
    if not encoded:raise ValueError('forced prefix token unavailable')
    anchor=dict(panel[0],split='DEV')
    for budget in (128,256,512,1024):
        for on in (False,True):one(anchor,budget,on,int(encoded[0]))
    for path,before in stats.items():
        s=Path(path).stat()
        if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weight identity changed')
    if any(sha(p)!=h for p,h in frozen['code_sha256'].items()):raise ValueError('worker source changed')
    with (out/'worker_complete.json').open('x') as f:
        json.dump({'n_questions':32,'n_natural_generations':256,'n_forced_generations':8,'time':time.time()},f);f.write('\n')

if __name__=='__main__':main()
