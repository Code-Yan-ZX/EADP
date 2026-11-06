"""Future E4 C-DROP/C-SKIP 16-DEV identity worker. Never runs without explicit execute.

Requires accepted, finished full E1 DEV generation. This diagnostic records
natural outputs, not timings or scores. Supervise the real container externally;
worker_complete alone is not proof of process/container exit.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
ROOT=Path('/media/disk2/YZX/research/EADP_v2')
PLAN=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
WEIGHTS=ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json'
DATA=PROJECT/'data_manifest'
sys.path.insert(0,str(PROJECT/'e1_analysis'))
from completion import sha,read_jsonl,validate_terminal

def check_e1(path,acceptance_started=None,acceptance_terminal=None):
    proof=json.loads(path.read_text())
    completion=PROJECT/'e1_analysis/completion.py'
    if (proof.get('status')!='passed' or proof.get('stage')!='full' or proof.get('natural_questions')!=384 or
        proof.get('natural_records')!=1536 or proof.get('tokenizer_decode_checked')!=1536 or
        proof.get('GT_opened') is not False or proof.get('scoring_performed') is not False or
        proof.get('verifier_sha256')!=sha(completion)):
        raise ValueError('requires a completed, independently accepted E1 DEV run')
    files=proof.get('files',{})
    required=('frozen','worker_complete','journal','panel','terminal','container_exit','launch','inputs','input_proof','plan')
    if any(k not in files for k in required):raise ValueError('E1 acceptance lacks required artifacts')
    for item in files.values():
        if sha(item['path'])!=item['sha256']:raise ValueError('accepted E1 artifact changed')
    if files['plan']['sha256']!=sha(PLAN):raise ValueError('E4 recipe differs from accepted E1')
    terminal=json.loads(Path(files['terminal']['path']).read_text())
    exited=json.loads(Path(files['container_exit']['path']).read_text())
    launch=json.loads(Path(files['launch']['path']).read_text())
    validate_terminal(terminal,exited,launch,Path(files['journal']['path']).parent)
    if acceptance_started is None or acceptance_terminal is None:
        raise ValueError('E1 acceptance verifier supervisor exit evidence required')
    from score import validate_acceptance_supervisor
    validate_acceptance_supervisor(path,acceptance_started,acceptance_terminal)
    return proof

def select_panel(proof):
    files=proof['files'];panel=read_jsonl(files['panel']['path']);inputs=read_jsonl(files['inputs']['path'])
    dev=[r for r in panel if r.get('split')=='DEV']
    if len(dev)!=384 or len(inputs)!=384:raise ValueError('expected frozen full DEV input')
    ref={(r['task'],str(r['qid'])):r for r in dev}
    if len(ref)!=384:raise ValueError('duplicate DEV qid')
    if {(r['task'],str(r['qid'])) for r in inputs}!=set(ref) or len({(r['task'],str(r['qid'])) for r in inputs})!=384:
        raise ValueError('DEV input inventory mismatch')
    for row in inputs:
        key=(row['task'],str(row['qid']))
        if key not in ref or row['group_id']!=ref[key]['group_id']:raise ValueError('input/panel mismatch')
        msg_hash=hashlib.sha256(json.dumps(row['message'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        if msg_hash!=row['message_sha256'] or sha(row['image_path'])!=row['image_sha256']:
            raise ValueError('frozen input changed')
    exclusion=json.loads((DATA/'anchor_exclusion/summary.json').read_text())
    verification=json.loads((DATA/'anchor_exclusion/verification.json').read_text())
    if (verification.get('status')!='pass' or verification.get('summary_sha256')!=sha(DATA/'anchor_exclusion/summary.json') or
        verification.get('verifier_sha256')!=sha(DATA/'anchor_exclusion/verify.py') or
        exclusion.get('builder_sha256')!=sha(DATA/'anchor_exclusion/build.py') or
        exclusion.get('panel_manifest_sha256')!=sha(files['panel']['path']) or
        exclusion.get('full_manifest_sha256')!=sha(DATA/'full_manifest.jsonl') or
        exclusion['outputs_sha256']['anchor_groups.jsonl']!=sha(DATA/'anchor_exclusion/anchor_groups.jsonl')):
        raise ValueError('anchor exclusion proof failed')
    old={r['group_id'] for r in read_jsonl(DATA/'anchor_exclusion/anchor_groups.jsonl')}
    if len(old)!=16:raise ValueError('original anchor group inventory mismatch')
    chosen=sorted((r for r in inputs if r['task']=='DocVQA_VAL' and r['group_id'] not in old),
        key=lambda r:(hashlib.sha256(('20261104|'+r['group_id']).encode()).hexdigest(),r['group_id']))[:16]
    if len(chosen)!=16 or len({x['group_id'] for x in chosen})!=16:raise ValueError('16 unique DEV groups required')
    return chosen

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    p.add_argument('--e1-acceptance',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--e1-acceptance-started',type=Path,required=True);p.add_argument('--e1-acceptance-terminal',type=Path,required=True)
    a=p.parse_args()
    if not a.execute:raise ValueError('identity worker disabled without explicit --execute')
    proof=check_e1(a.e1_acceptance,a.e1_acceptance_started,a.e1_acceptance_terminal);panel=select_panel(proof)
    plan=json.loads(PLAN.read_text())
    expected={'visual_token_num':256,'alpha':0.5,'beta':2.0,'min_pixels':1048576,'max_pixels':1048576,
              'max_new_tokens':2048,'use_custom_prompt':True,'use_vllm':False}
    if any(plan['recipe'].get(k)!=v for k,v in expected.items()):raise ValueError('unexpected author recipe')
    for group in ('source_sha256','implementation_sha256'):
        for name,digest in plan[group].items():
            if sha(name)!=digest:raise ValueError('frozen author source changed: '+name)
    weights=json.loads(WEIGHTS.read_text())['files']
    if len(weights)!=4:raise ValueError('expected four model shards')
    stats={}
    for item in weights:
        path=Path(item['path']);s=path.stat()
        if s.st_size!=item['bytes'] or sha(path)!=item['expected_lfs_sha256']:raise ValueError('model weights changed')
        stats[str(path)]=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True)
    if gpu.stdout.strip():raise ValueError('GPU already has compute processes')
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    code=[HERE/'identity_worker.py',HERE/'identity_records.py',HERE/'off_adapter.py',HERE/'skip_deepstack.py']
    frozen={'e1_acceptance_sha256':sha(a.e1_acceptance),'author_plan_sha256':sha(PLAN),'weights_identity_sha256':sha(WEIGHTS),
            'code_sha256':{str(x):sha(x) for x in code},'weight_stats_before':stats,'selected':[{k:r[k] for k in ('task','qid','group_id')} for r in panel],
            'n_questions':16,'modes':['C-DROP','C-SKIP'],'scoring':False,'gt_read':False,'performance_measurement':False}
    (out/'frozen.json').write_text(json.dumps(frozen,indent=2,ensure_ascii=False)+'\n')
    sys.path[:0]=[str(HERE),plan['author_root'],str(Path(plan['author_root'])/'VLMEvalKit')]
    import torch
    from off_adapter import E4OffAdapter
    from identity_records import make_record,compare_pair
    from model import pruner as native
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP,Qwen3VLChatCDPruner
    if {n:importlib.metadata.version(n) for n in plan['expected_versions']}!=plan['expected_versions']:
        raise ValueError('runtime version mismatch')
    if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
    recipe=dict(plan['recipe']);recipe['visual_token_num']=256
    w=Qwen3VLChatEADP(**recipe)
    if w.model.dtype!=torch.bfloat16 or w.model.config._attn_implementation!='flash_attention_2':raise ValueError('backend mismatch')
    for row in panel:
        pair=[]
        for mode in ('C-DROP','C-SKIP'):
            seen={};original=w.model.generate
            def capture(*args,**kwargs):
                eos=kwargs.get('eos_token_id',w.model.generation_config.eos_token_id)
                seen['eos_ids']=[eos] if isinstance(eos,int) else list(eos or [])
                result=original(*args,**kwargs);seen['output_ids']=result.detach().cpu().tolist()[0]
                return result
            def first_logits(_module,_args,output):
                if 'logits' not in seen:seen['logits']=output[:,-1,:].detach().clone()
            w.model.generate=capture;handle=w.model.lm_head.register_forward_hook(first_logits)
            try:
                with E4OffAdapter(torch,w,native,mode) as adapter:
                    with torch.inference_mode():
                        answer=Qwen3VLChatCDPruner.generate_inner_transformers_fixed(w,row['message'],dataset=row['task'])
                    record=make_record(adapter,row,seen['output_ids'],answer,seen['logits'],seen['eos_ids'])
            finally:
                handle.remove();w.model.generate=original
            target=out/(str(row['qid'])+'_'+mode+'.json')
            with target.open('x') as f:json.dump(record,f,ensure_ascii=False);f.write('\n')
            pair.append(record)
        comparison=compare_pair(*pair)
        with (out/'comparisons.jsonl').open('a') as f:f.write(json.dumps(comparison,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
        print('completed diagnostic pair',row['qid'],flush=True)
    for path,before in stats.items():
        s=Path(path).stat()
        if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weight identity changed during run')
    for path,digest in frozen['code_sha256'].items():
        if sha(path)!=digest:raise ValueError('diagnostic source changed during run')
    (out/'worker_complete.json').write_text(json.dumps({'n_questions':16,'n_natural_generations':32,'n_pairs':16,'time':time.time()})+'\n')

if __name__=='__main__':main()
