"""Three fixed DEV C-DROP/C-SKIP natural identities and one forced 32-step pair.

No accuracy labels, scoring, or timing; execution requires a separately
reviewed B approval and a fully accepted finished E4-A panel.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;E4=ROOT/'e4';NATIVE=ROOT/'e4_native'
sys.path[:0]=[str(E4),str(NATIVE)]
from identity_worker import PLAN,WEIGHTS,sha,read_jsonl
from native_contract import select_anchors
from native_gate import check_gates
from b_imports import local

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--preflight-only',action='store_true')
    for name in ('e1-acceptance','e1-started','e1-terminal','smoke-dir','smoke-proof','smoke-started','smoke-terminal','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if not a.execute:raise ValueError('explicit execution required')
    gate=local('b_gate');approval=gate.check_approval(preflight_only=a.preflight_only)
    approved=json.loads(gate.APPROVAL.read_text())['panel']
    for field in ('e1_acceptance','e1_started','e1_terminal'):
        if Path(approved[field]).resolve()!=getattr(a,field).resolve():raise ValueError('B upstream E1 mismatch')
    out=a.out.resolve();receipt=out/'host_panel_gate.json';launch=json.loads((out/'launch.json').read_text())
    if launch.get('host_panel_gate')!={'path':str(receipt),'sha256':sha(receipt)}:raise ValueError('B host panel receipt unbound')
    if json.loads(receipt.read_text())!={'panel_proof_sha256':approval['panel_proof_sha256'],
        'panel_review_sha256':approval['panel_review_sha256'],'panel_gate_source_sha256':sha(ROOT/'e4_analysis/score_e4.py'),
        'host_live_docker_inspection':True}:raise ValueError('B host panel receipt invalid')
    e1=check_gates(a);anchors=select_anchors(e1)
    refs=read_jsonl(a.smoke_dir/'records.jsonl');reference={}
    for r in refs:
        if r.get('budget')==256 and r.get('forced') is False:
            key=(r['task'],str(r['qid']),r['ds_on'])
            if key in reference:raise ValueError('duplicate accepted smoke reference')
            reference[key]=r
    for row in anchors:
        for on in (False,True):
            if (row['task'],str(row['qid']),on) not in reference:raise ValueError('missing accepted K256 reference')
    plan=json.loads(PLAN.read_text());recipe=dict(plan['recipe']);recipe['visual_token_num']=256
    for group in ('source_sha256','implementation_sha256'):
        for name,digest in plan[group].items():
            if sha(name)!=digest:raise ValueError('source changed: '+name)
    weights=json.loads(WEIGHTS.read_text())['files'];stats={}
    if len(weights)!=4:raise ValueError('wrong weight inventory')
    for item in weights:
        path=Path(item['path']);s=path.stat()
        if s.st_size!=item['bytes'] or sha(path)!=item['expected_lfs_sha256']:raise ValueError('weight mismatch')
        stats[str(path)]=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    out=a.out.resolve()
    if not out.is_dir() or any((out/x).exists() for x in ('frozen.json','generated.jsonl','records.jsonl','arrays')):
        raise ValueError('invalid/reused run directory')
    sys.path[:0]=[str(HERE),str(E4),plan['author_root'],str(Path(plan['author_root'])/'VLMEvalKit')]
    import inspect
    import torch
    run_arm=local('arm').run_arm;contract=local('pair_contract')
    validate_pair,validate_forced_pair=contract.validate_pair,contract.validate_forced_pair
    from model import pruner
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP,Qwen3VLChatCDPruner
    versions={n:importlib.metadata.version(n) for n in plan['expected_versions']}
    expected_pruner=(Path(plan['author_root'])/'model/pruner.py').resolve()
    expected_wrapper=(Path(plan['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve()
    if (versions!=plan['expected_versions'] or Path(pruner.__file__).resolve()!=expected_pruner or
        Path(inspect.getfile(Qwen3VLChatEADP)).resolve()!=expected_wrapper or
        Path(inspect.getfile(Qwen3VLChatCDPruner)).resolve()!=expected_wrapper or
        Path(inspect.getfile(run_arm)).resolve()!=(HERE/'arm.py').resolve()):raise ValueError('runtime imports or versions mismatch')
    if a.preflight_only:
        print(json.dumps({'status':'preflight_passed','model_loaded':False,'predictions':False,
            'cuda_initialized':torch.cuda.is_initialized(),'pruner':str(expected_pruner),'wrapper':str(expected_wrapper),
            'arm':str((HERE/'arm.py').resolve()),'versions':versions}),flush=True)
        if torch.cuda.is_initialized():raise ValueError('preflight unexpectedly initialized CUDA')
        return
    (out/'arrays').mkdir()
    sources=[HERE/x for x in ('b_worker.py','b_launch.py','b_verify.py','b_gate.py','b_imports.py','budget_plan.json','arm.py','off_capture.py','pair_contract.py','verify_records.py')]
    sources += [E4/x for x in ('off_adapter.py','skip_deepstack.py','identity_records.py','decode_observer.py')]
    sources += [NATIVE/x for x in ('native_capture.py','native_contract.py','native_gate.py')]
    frozen={'stage':'skip_drop_identity','anchors':[{k:r[k] for k in ('task','qid','group_id','message_sha256','image_sha256')} for r in anchors],
        'forced_anchor_index':0,'modes':['C-DROP','C-SKIP'],'natural_pairs':3,'forced_pairs':1,
        'e1_acceptance_sha256':sha(a.e1_acceptance),'smoke_proof_sha256':sha(a.smoke_proof),'plan_sha256':sha(PLAN),
        'weights_identity_sha256':sha(WEIGHTS),'weight_stats_before':stats,'code_sha256':{str(x):sha(x) for x in sources},
        'gt_read':False,'scoring':False,'timing_claims':False}
    with (out/'frozen.json').open('x') as f:json.dump(frozen,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    if torch.cuda.get_device_name(0)!='NVIDIA A40':
        raise ValueError('runtime mismatch')
    w=Qwen3VLChatEADP(**recipe)
    if w.model.dtype!=torch.bfloat16 or w.model.config._attn_implementation!='flash_attention_2':raise ValueError('backend mismatch')
    tokens=w.processor.tokenizer.encode(' the',add_special_tokens=False)
    if not tokens:raise ValueError('forced token unavailable')
    token=int(tokens[0])
    def append(name,record):
        with (out/name).open('a') as f:f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
    def generate(model,row):
        with torch.inference_mode():return Qwen3VLChatCDPruner.generate_inner_transformers_fixed(model,row['message'],dataset=row['task'])
    for index,row in enumerate(anchors):
        natural=None
        for forced in ((False,True) if index==0 else (False,)):
            pair=[]
            for mode in ('C-DROP','C-SKIP'):
                key=hashlib.sha256(json.dumps([row['task'],str(row['qid']),mode,forced],separators=(',',':')).encode()).hexdigest()
                def archive(tensors):
                    import numpy as np
                    dst=out/'arrays'/(key+'.npz');data={k:v.detach().contiguous().cpu().view(torch.uint8).numpy() for k,v in tensors.items()}
                    with dst.open('xb') as f:np.savez(f,**data);f.flush();os.fsync(f.fileno())
                    return {'kind':'full_arrays','path':str(dst),'sha256':sha(dst)}
                rec=run_arm(torch,w,pruner,row,mode,generate,lambda r:append('generated.jsonl',r),archive,token if forced else None)
                append('records.jsonl',rec);pair.append(rec)
            if forced:result=validate_forced_pair(*pair,*natural,token)
            else:
                natural=pair;result=validate_pair(*pair,reference[(row['task'],str(row['qid']),True)],reference[(row['task'],str(row['qid']),False)])
            append('comparisons.jsonl',dict(result,forced=forced))
            print('completed B diagnostic',row['task'],'forced',forced,flush=True)
    for path,before in stats.items():
        s=Path(path).stat()
        if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weight identity changed')
    if any(sha(p)!=h for p,h in frozen['code_sha256'].items()):raise ValueError('source changed during run')
    with (out/'worker_complete.json').open('x') as f:json.dump({'natural_pairs':3,'forced_pairs':1,'generations':8,'time':time.time()},f);f.write('\n');f.flush();os.fsync(f.fileno())

if __name__=='__main__':main()
