"""One bounded real-model synthetic-target backward; no labels or generation."""
import argparse
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;E4=ROOT/'e4'
sys.path.insert(1,str(E4))
from identity_worker import PLAN,WEIGHTS
from local_imports import local,teacher
contract=local('contract');sha=contract.sha

def event(out,name,**extra):
    with (out/'events.jsonl').open('a') as f:
        f.write(json.dumps({'event':name,'unix':time.time(),**extra},separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--preflight-only',action='store_true');p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    if not a.execute:raise ValueError('explicit execute required')
    gate=local('gate');approval=gate.check_approval(preflight_only=a.preflight_only);out=a.out.resolve()
    receipt=out/'host_upstream.json';launch=json.loads((out/'launch.json').read_text())
    if launch.get('host_upstream')!={'path':str(receipt),'sha256':sha(receipt)} or json.loads(receipt.read_text())!=approval['upstream']:
        raise ValueError('host native inspection receipt mismatch')
    plan=json.loads((HERE/'plan.json').read_text());native_dir=ROOT/'e4_native/native_001';records=native_dir/'records.jsonl'
    row,ref=contract.anchor(ROOT/'m3_oracle/sample_001.json',records,plan['task'],plan['qid'],plan['group_id'])
    e1=json.loads((ROOT/'e1/e1full001_completion.json').read_text());inputs_path=Path(e1['files']['inputs']['path'])
    if sha(inputs_path)!=e1['files']['inputs']['sha256']:raise ValueError('E1 inputs changed')
    with inputs_path.open() as f:
        rows=[r for line in f if (r:=json.loads(line)).get('task')==plan['task'] and str(r.get('qid'))==plan['qid']]
    if len(rows)!=1 or any(rows[0].get(k)!=row.get(k) for k in ('group_id','image_sha256','message_sha256')):
        raise ValueError('frozen E1 input mismatch')
    row=dict(rows[0]);
    if sha(row['image_path'])!=row['image_sha256']:raise ValueError('input image changed')
    with records.open() as f:
        forced=[r for line in f if (r:=json.loads(line)).get('task')==plan['task'] and str(r.get('qid'))==plan['qid'] and
                r.get('group_id')==plan['group_id'] and r.get('path')=='stock_native' and r.get('forced') is True]
    if len(forced)!=1:raise ValueError('actual native forced decode reference missing')
    author=json.loads(PLAN.read_text())
    for group in ('source_sha256','implementation_sha256'):
        for path,digest in author[group].items():
            if sha(path)!=digest:raise ValueError('author source changed: '+path)
    weights=json.loads(WEIGHTS.read_text())['files'];stats={}
    if len(weights)!=4:raise ValueError('invalid weight inventory')
    for item in weights:
        path=Path(item['path']);s=path.stat()
        if s.st_size!=item['bytes'] or sha(path)!=item['expected_lfs_sha256']:raise ValueError('model payload mismatch')
        stats[str(path)]=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    if not out.is_dir() or any((out/x).exists() for x in ('frozen.json','result.json','events.jsonl','arrays.npz')):
        raise ValueError('invalid or reused output directory')
    sys.path[:0]=[str(HERE),str(E4),author['author_root'],str(Path(author['author_root'])/'VLMEvalKit')]
    import torch
    from transformers import AutoTokenizer
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP
    model_source=(Path(author['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve()
    if Path(inspect.getfile(Qwen3VLChatEADP)).resolve()!=model_source:raise ValueError('author wrapper path mismatch')
    versions={k:importlib.metadata.version(k) for k in author['expected_versions']}
    if versions!=author['expected_versions']:raise ValueError('runtime version mismatch')
    teacher_mod,saliency=teacher();core=local('core')
    tokenizer=AutoTokenizer.from_pretrained(author['recipe']['model_path'],local_files_only=True,trust_remote_code=False)
    contract.check_targets(plan['target_token_ids'],len(tokenizer),tokenizer.all_special_ids)
    if a.preflight_only:
        if torch.cuda.is_initialized():raise ValueError('CUDA initialized during preflight')
        print(json.dumps({'status':'preflight_passed','cuda_initialized':False,'model_loaded':False,'predictions':False,
                          'GT_opened':False,'target_ids':plan['target_token_ids'],'wrapper':str(model_source),'versions':versions}),flush=True)
        return
    if torch.is_inference_mode_enabled():raise ValueError('inference_mode would disable autograd')
    if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
    frozen={'stage':'synthetic_target_gradient_smoke','plan_sha256':sha(HERE/'plan.json'),'approval':approval,
        'sample_sha256':sha(ROOT/'m3_oracle/sample_001.json'),'anchor_record_sha256':sha(records),
        'anchor_array_sha256':ref['arrays']['sha256'],'weights_identity_sha256':sha(WEIGHTS),'weight_stats_before':stats,
        'task':plan['task'],'qid':plan['qid'],'group_id':plan['group_id'],'target_ids':plan['target_token_ids'],
        'gt_read':False,'scoring':False,'generation':False}
    with (out/'frozen.json').open('x') as f:json.dump(frozen,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    event(out,'worker_started');started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    try:
        recipe=dict(author['recipe']);recipe['visual_token_num']=1024
        t=time.monotonic();w=Qwen3VLChatEADP(**recipe);torch.cuda.synchronize();load_seconds=time.monotonic()-t
        if w.model.dtype!=torch.bfloat16 or w.model.config._attn_implementation!='flash_attention_2':raise ValueError('backend mismatch')
        memory_after_load={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
                           'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
        event(out,'model_loaded',load_seconds=load_seconds,**memory_after_load);torch.cuda.reset_peak_memory_stats()
        def archive(tensors):
            import numpy as np
            path=out/'arrays.npz';data={k:x.detach().contiguous().cpu().view(torch.uint8).numpy() for k,x in tensors.items()}
            with path.open('xb') as f:np.savez(f,**data);f.flush();os.fsync(f.fileno())
            return {'path':str(path),'sha256':sha(path),'keys':sorted(data)}
        result=core.run(torch,w,row,ref,forced[0],teacher_mod,saliency,plan,lambda name:event(out,name),archive)
        torch.cuda.synchronize();result.update(model_load_seconds=load_seconds,memory_after_load=memory_after_load,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            total_seconds=time.monotonic()-started,versions=versions)
        for path,before in stats.items():
            s=Path(path).stat()
            if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weights changed during smoke')
        with (out/'result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        event(out,'worker_complete')
    except BaseException as exc:
        report={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'seconds':time.monotonic()-started,
                'oom':isinstance(exc,torch.cuda.OutOfMemoryError),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        with (out/'failure.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        event(out,'worker_failed',error_type=type(exc).__name__,oom=report['oom']);raise

if __name__=='__main__':main()
