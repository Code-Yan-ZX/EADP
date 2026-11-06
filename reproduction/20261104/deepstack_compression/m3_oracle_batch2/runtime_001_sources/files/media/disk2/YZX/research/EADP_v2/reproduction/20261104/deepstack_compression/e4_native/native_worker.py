"""Three DEV anchors: actual stock native vs EADP K1024 DS-on; no GT."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from native_contract import HERE,E4,select_anchors,compare_pair
from native_gate import check_gates,REVIEW,REVIEW_TERMINAL
from native_capture import run_native,CacheFingerprints
sys.path.insert(0,str(E4))
from identity_worker import PLAN,WEIGHTS,sha

def arguments():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    for name in ('e1-acceptance','e1-started','e1-terminal','smoke-dir','smoke-proof','smoke-started','smoke-terminal','out'):
        p.add_argument('--'+name,type=Path,required=True)
    return p.parse_args()

def main():
    a=arguments()
    if not a.execute:raise ValueError('explicit execute required')
    from native_launch import check_approval
    check_approval();proof=check_gates(a);anchors=select_anchors(proof);plan=json.loads(PLAN.read_text())
    for group in ('source_sha256','implementation_sha256'):
        for path,digest in plan[group].items():
            if sha(path)!=digest:raise ValueError('source changed: '+path)
    recipe=dict(plan['recipe']);recipe['visual_token_num']=1024
    expected={'alpha':0.5,'beta':2.0,'min_pixels':1048576,'max_pixels':1048576,'max_new_tokens':2048,
              'use_custom_prompt':True,'use_vllm':False}
    if any(recipe.get(k)!=v for k,v in expected.items()):raise ValueError('unexpected author recipe')
    weights=json.loads(WEIGHTS.read_text())['files'];stats={}
    if len(weights)!=4:raise ValueError('expected four weight shards')
    for item in weights:
        path=Path(item['path']);s=path.stat()
        if s.st_size!=item['bytes'] or sha(path)!=item['expected_lfs_sha256']:raise ValueError('weight mismatch')
        stats[str(path)]=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    out=a.out.resolve()
    if not out.is_dir() or any((out/x).exists() for x in ('frozen.json','generated.jsonl','records.jsonl','arrays')):
        raise ValueError('invalid or reused output')
    (out/'arrays').mkdir()
    sources=[HERE/x for x in ('native_worker.py','native_launch.py','native_gate.py','native_contract.py','native_capture.py','native_verify.py')]
    sources += [E4/x for x in ('budget_runner.py','budget_adapter.py','decode_observer.py','off_adapter.py','identity_records.py')]
    frozen={'stage':'stock_native_k1024','n_questions':3,'anchors':[{k:r[k] for k in ('task','qid','group_id','message_sha256','image_sha256')} for r in anchors],
            'paths':['stock_native','budget1024'],'modes':['natural','forced33'],'e1_acceptance_sha256':sha(a.e1_acceptance),
            'smoke_proof_sha256':sha(a.smoke_proof),'smoke_review_sha256':sha(REVIEW),'smoke_review_terminal_sha256':sha(REVIEW_TERMINAL),
            'plan_sha256':sha(PLAN),'weight_identity_sha256':sha(WEIGHTS),'weight_stats_before':stats,
            'code_sha256':{str(x):sha(x) for x in sources},'gt_read':False,'scoring':False,'timing_claims':False}
    with (out/'frozen.json').open('x') as f:json.dump(frozen,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
    sys.path[:0]=[str(E4),plan['author_root'],str(Path(plan['author_root'])/'VLMEvalKit')]
    import torch
    from model import pruner as native_pruner
    from budget_runner import run_arm
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP,Qwen3VLChatFixedRes,Qwen3VLChatCDPruner
    if {n:importlib.metadata.version(n) for n in plan['expected_versions']}!=plan['expected_versions']:
        raise ValueError('runtime versions differ')
    if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
    w=Qwen3VLChatEADP(**recipe)
    if w.model.dtype!=torch.bfloat16 or w.model.config._attn_implementation!='flash_attention_2':raise ValueError('unexpected backend')
    encoded=w.processor.tokenizer.encode(' the',add_special_tokens=False)
    if not encoded:raise ValueError('forced token unavailable')
    fixed=int(encoded[0]);scalars={k:recipe[k] for k in ('top_p','top_k','temperature','repetition_penalty')}
    scalars.update(do_sample=False,max_new_tokens=2048)
    def on_generated(record):
        with (out/'generated.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
    def native_generate(model,row):
        with torch.inference_mode():return Qwen3VLChatFixedRes.generate_inner_transformers_fixed(model,row['message'],dataset=row['task'])
    def budget_generate(model,row):
        with torch.inference_mode():return Qwen3VLChatCDPruner.generate_inner_transformers_fixed(model,row['message'],dataset=row['task'])
    for item in anchors:
        row=dict(item,split='DEV')
        for forced in (False,True):
            pair=[]
            for path in ('stock_native','budget1024'):
                token=fixed if forced else None
                key=hashlib.sha256(json.dumps([row['task'],str(row['qid']),path,forced],separators=(',',':')).encode()).hexdigest()
                def archive(tensors):
                    import numpy as np
                    dst=out/'arrays'/(key+'.npz')
                    data={k:v.detach().contiguous().cpu().view(torch.uint8).numpy() for k,v in tensors.items()}
                    with dst.open('xb') as f:np.savez(f,**data);f.flush();os.fsync(f.fileno())
                    return {'kind':'full_arrays','path':str(dst),'sha256':sha(dst)}
                if path=='stock_native':
                    rec=run_native(torch,w,row,native_generate,on_generated,archive,forced_token=token)
                else:
                    captured={};old=w._processor_inputs;had='_processor_inputs' in w.__dict__;prior=w.__dict__.get('_processor_inputs')
                    def processor(*args,**kwargs):
                        inputs=old(*args,**kwargs)
                        captured.update({k:inputs[k].detach().clone() for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')})
                        return inputs
                    w._processor_inputs=processor
                    def budget_archive(tensors):return archive(dict(tensors,**{'processor_'+k:v for k,v in captured.items()}))
                    def budget_generated(record):on_generated(dict(record,path='budget1024'))
                    try:
                        with CacheFingerprints(w,forced) as kv:
                            rec=run_arm(torch,w,native_pruner,row,1024,True,budget_generate,budget_generated,budget_archive,
                                        forced_token=token,capture_prefill_logits=True)
                        rec['cache_fingerprints']=kv.fingerprints
                    finally:
                        if had:w._processor_inputs=prior
                        else:delattr(w,'_processor_inputs')
                expected_scalars=dict(scalars)
                if forced:expected_scalars['max_new_tokens']=33
                if rec['generation_scalars']!=expected_scalars:raise ValueError('actual generation recipe mismatch')
                with (out/'records.jsonl').open('a') as f:f.write(json.dumps(rec,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
                pair.append(rec)
            compare_pair(*pair)
            with (out/'comparisons.jsonl').open('a') as f:
                f.write(json.dumps({'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'forced':forced,'status':'passed'})+'\n');f.flush();os.fsync(f.fileno())
            print('completed native identity pair',row['task'],'forced',forced,flush=True)
    for path,before in stats.items():
        s=Path(path).stat()
        if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weight identity changed')
    if any(sha(p)!=h for p,h in frozen['code_sha256'].items()):raise ValueError('source changed during run')
    with (out/'worker_complete.json').open('x') as f:
        json.dump({'n_questions':3,'n_generations':12,'n_pairs':6,'time':time.time()},f);f.write('\n')

if __name__=='__main__':main()
