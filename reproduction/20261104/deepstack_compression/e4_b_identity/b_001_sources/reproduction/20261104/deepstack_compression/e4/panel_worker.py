"""Prepared full DEV budget panel. No labels, scoring, or GPU timing.

Hard-locked pending independent full-panel review. Every K256 arm is freshly
generated; the 32-question smoke identity proof is not extrapolated.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from identity_worker import HERE,PLAN,WEIGHTS,sha
from panel_gate import validated_dev,validate_smoke_gate


def parse():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    for name in ('e1-acceptance','e1-started','e1-terminal','smoke-dir','smoke-proof','smoke-started','smoke-terminal','out'):
        p.add_argument('--'+name,type=Path,required=True)
    return p.parse_args()


def main():
    a=parse()
    if not a.execute:raise ValueError('explicit execute required')
    from panel_launch import check_panel_approval
    check_panel_approval()
    proof,panel=validated_dev(a.e1_acceptance,a.e1_started,a.e1_terminal)
    validate_smoke_gate(a.e1_acceptance,a.smoke_dir,a.smoke_proof,a.smoke_started,a.smoke_terminal)
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
    if not out.is_dir() or any((out/x).exists() for x in ('frozen.json','records.jsonl','generated.jsonl','arrays')):
        raise ValueError('invalid or reused output')
    sources=[HERE/x for x in ('panel_worker.py','panel_launch.py','panel_gate.py','smoke_contract.py','budget_runner.py','decode_observer.py',
        'budget_adapter.py','off_adapter.py','skip_deepstack.py','identity_records.py','identity_worker.py','verify_k256.py')]
    frozen={'stage':'full','n_questions':384,'e1_acceptance_sha256':sha(a.e1_acceptance),'smoke_proof_sha256':sha(a.smoke_proof),
        'plan_sha256':sha(PLAN),'weights_identity_sha256':sha(WEIGHTS),'weight_stats_before':stats,
        'code_sha256':{str(x):sha(x) for x in sources},'inputs_sha256':proof['files']['inputs']['sha256'],
        'selected':[{k:r[k] for k in ('task','qid','group_id','message_sha256','image_sha256')} for r in panel],
        'budgets':[128,256,512,1024],'ds_on':[False,True],'k256_reuse':False,'gt_read':False,'scoring':False,
        'timing_claims':False,'tensor_retention':'summary_only_runtime_checked_no_tensor_reconstruction'}
    with (out/'frozen.json').open('x') as f:json.dump(frozen,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
    sys.path[:0]=[str(HERE),plan['author_root'],str(Path(plan['author_root'])/'VLMEvalKit')]
    import torch
    from model import pruner as native
    from budget_runner import run_arm
    from identity_records import tensor_meta
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP,Qwen3VLChatCDPruner
    if {n:importlib.metadata.version(n) for n in plan['expected_versions']}!=plan['expected_versions']:
        raise ValueError('runtime versions changed')
    if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
    wrapper=Qwen3VLChatEADP(**recipe)
    if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('unexpected backend')
    scalars={key:recipe[key] for key in ('top_p','top_k','temperature','repetition_penalty')}
    scalars.update(do_sample=False,max_new_tokens=2048)
    def generate(w,row):
        with torch.inference_mode():return Qwen3VLChatCDPruner.generate_inner_transformers_fixed(w,row['message'],dataset=row['task'])
    def journal(record):
        with (out/'generated.jsonl').open('a') as f:
            f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
    for item in panel:
        row=dict(item,split='DEV')
        for budget in (128,256,512,1024):
            wrapper.visual_token_num=budget;wrapper.pruner.visual_token_num=budget
            for on in (False,True):
                def archive(tensors):
                    return {'kind':'summary_only','tensors':{key:tensor_meta(value) for key,value in tensors.items()}}
                record=run_arm(torch,wrapper,native,row,budget,on,generate,journal,archive)
                if record['generation_scalars']!=scalars:raise ValueError('generation recipe mismatch')
                with (out/'records.jsonl').open('a') as f:
                    f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
        print('completed panel question',row['task'],row['qid'],flush=True)
    for path,before in stats.items():
        s=Path(path).stat()
        if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weight identity changed')
    if any(sha(p)!=h for p,h in frozen['code_sha256'].items()):raise ValueError('panel source changed')
    with (out/'worker_complete.json').open('x') as f:
        json.dump({'n_questions':384,'n_natural_generations':3072,'time':time.time()},f);f.write('\n')

if __name__=='__main__':main()
