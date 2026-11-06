"""Independently check archived CPU decode traces without running a model."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

BASE=Path(__file__).resolve().parents[2]
D=BASE/'e6/pixelprune_cpu_decode'
OUT=BASE/'e6/pixelprune_cpu_review/decode_review001.json'
COMMIT='8d347eb1bdec92c5991057e75d424ecc1ecfe53e'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())

def main():
    checkout=BASE/'e6/external_sources/pixelprune_8d347eb'
    head=subprocess.run(['git','-C',str(checkout),'rev-parse','HEAD'],check=True,capture_output=True,text=True,timeout=10).stdout.strip()
    dirty=subprocess.run(['git','-C',str(checkout),'status','--short'],check=True,capture_output=True,text=True,timeout=10).stdout
    plan=read(D/'plan001.json'); result=read(D/'result001.json'); started=read(D/'monitor001/started.json'); terminal=read(D/'monitor001/terminal.json')
    if head!=COMMIT or dirty or plan['source_commit']!=COMMIT or len(plan['sources'])!=23: raise ValueError('source checkout')
    if any(sha(p)!=h for p,h in plan['sources'].items()) or result['plan_sha256']!=sha(D/'plan001.json'): raise ValueError('source or plan sha')
    if ((Path(started['cwd'])/started['command'][-1]).resolve()!=(D/'probe.py').resolve() or len(started['command'])!=2
        or started['worker_pid']!=started['process_group'] or terminal['worker_pid']!=started['worker_pid']
        or terminal['reason']!='completed' or terminal['worker_returncode']!=0 or terminal['remaining_process_group']!=[]): raise ValueError('supervisor')
    if (result['cuda_initialized'] is not False or result['selector_kept_indices']!=[[0]]
        or result['original_merged_tokens']!=4 or result['original_max_position']!=6 or result['original_rope_delta']!=[[-2]]): raise ValueError('input identity')
    original=[[[0,1,2,2,2,2,4,5,6]],[[0,1,2,2,3,3,4,5,6]],[[0,1,2,3,2,3,4,5,6]]]
    kept=[[[0,1,2,4,5,6]]]*3
    rows=result['records']
    if [x['arm'] for x in rows]!=['unpatched','patched_explicit_disabled','patched_enabled_threshold_zero']: raise ValueError('arms')
    for n,row in enumerate(rows):
        is_pruned=n==2; first=6 if is_pruned else 9; images=1 if is_pruned else 4
        inputs=row['language_model_inputs']; caches=row['layer_cache_shapes']; ids=row['output_ids']
        if row['status']!='completed' or len(inputs)!=33 or len(caches)!=33 or len(ids)!=1 or ids[0][9:]!=[7]*33: raise ValueError('forced or call count')
        if (inputs[0]['positions']!=(kept if is_pruned else original) or inputs[0]['visual_count']!=images
            or inputs[0]['ds_rows']!=[images]*3): raise ValueError('prefill position / DS')
        for i in range(33):
            if i and (inputs[i]['positions']!=[[[6+i]]]*3 or inputs[i]['visual_count'] is not None or inputs[i]['ds_rows'] is not None): raise ValueError('decode position / DS')
            if len(caches[i])!=3 or any(layer[k]!=[1,2,first+i,8] for layer in caches[i] for k in ('key','value')): raise ValueError('per-layer KV length')
        if row['cache_length']!=first+32: raise ValueError('final KV length')
    if rows[0]['language_model_inputs']!=rows[1]['language_model_inputs'] or rows[0]['layer_cache_shapes']!=rows[1]['layer_cache_shapes']: raise ValueError('disabled trace differs')
    paths=[D/'plan001.json',D/'probe.py',D/'result001.json',D/'monitor001/started.json',D/'monitor001/terminal.json']
    out={'status':'passed','scope':'read-only independent recheck of saved traces; no model or selector rerun',
         'source_commit':head,'source_files_verified':23,'evidence_sha256':{str(x):sha(x) for x in paths},
         'findings':{'calls_per_arm':33,'forced_token_id':7,'forced_output_tokens':33,'selector_direct_call_kept_indices':[[0]],
                     'prefill_visual_rows':[4,4,1],'prefill_three_ds_rows':[[4,4,4],[4,4,4],[1,1,1]],
                     'prefill_cache_lengths':[9,9,6],'last_cache_lengths':[41,41,38],
                     'original_max_position':6,'original_rope_delta':-2,'decode_positions':'7 through 38 in each of 3 axes'},
         'limitations':['tiny random CPU eager model and synthetic identical patches only',
                        'selector [[0]] is a separate direct call; generation traces did not independently archive actual selector IDs',
                        'cache tensor shapes only; KV contents and logits were not archived',
                        'no real model, GPU/FA2 compatibility, accuracy or timing conclusion'],
         'reviewer_sha256':sha(__file__)}
    with OUT.open('x') as f: json.dump(out,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())

if __name__=='__main__': main()
