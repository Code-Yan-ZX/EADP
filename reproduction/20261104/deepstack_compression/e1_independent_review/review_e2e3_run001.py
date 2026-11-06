"""Independent readback of E2/E3 real-model smoke; no dataset labels or scoring."""
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer

BASE = Path(__file__).resolve().parent.parent
SMOKE = BASE/'e2_e3_smoke'
RUN = SMOKE/'run_001'
FORMAL = SMOKE/'run_001_verification.json'
ARMS = (('E2','000'),('E2','111'),('E2','100'),('E2','010'),('E2','001'),('E3','PERM'),('E3','MEAN'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def lines(path):
    data=Path(path).read_bytes()
    if data and not data.endswith(b'\n'):raise ValueError('partial journal')
    return [json.loads(x) for x in data.splitlines()]

def req(condition, message):
    if not condition:raise ValueError(message)

def metadata_matches(raw, meta):
    return hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()==meta['sha256']

def main():
    formal=json.loads(FORMAL.read_text())
    req(formal['status']=='passed' and formal['GT_opened'] is False and formal['scoring_performed'] is False, 'formal proof')
    req(formal['natural_verified']==21 and formal['forced_verified']==27 and formal['forced_kv_reference_pairs']==6, 'formal inventory')
    for p,h in {**formal['files_sha256'],**formal['array_sha256']}.items():req(sha(p)==h,'formal file changed')
    for terminal in (SMOKE/'monitor_accept001/terminal.json', SMOKE/'monitor_001/terminal.json'):
        d=json.loads(terminal.read_text());req(d['reason']=='completed' and d['worker_returncode']==0 and d['remaining_process_group']==[],'supervisor exit')
    ex=json.loads((RUN/'exit.json').read_text());req(ex['returncode']==0 and ex['container_stopped_verified'] is True and ex['container_exit_code']==0 and ex['gpu_pids_after']==[],'launcher exit')
    inspected=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=10).stdout)[0]
    req(not inspected['State']['Running'] and inspected['State']['Pid']==0 and inspected['State']['ExitCode']==0,'container exit')
    frozen=json.loads((RUN/'frozen.json').read_text());records=lines(RUN/'records.jsonl');journal=lines(RUN/'generated.jsonl')
    expected=[]
    for item in frozen['selected']:
        expected += [(item['task'],str(item['qid']),a,b,f) for a,b in ARMS for f in (False,True)]
        expected += [(item['task'],str(item['qid']),'REF',b,True) for b in ('D0','D1')]
    req(len(expected)==48 and len(records)==48 and len(journal)==48,'record count')
    req([(r['task'],str(r['qid']),r['family'],r['arm'],r['forced']) for r in records]==expected,'record order')
    e1=json.loads((BASE/'e1/e1full001_completion.json').read_text())
    refs={(r['task'],str(r['qid']),r['arm']):r for r in lines(e1['files']['journal']['path']) if not r['forced']}
    plan=BASE.parents[2]/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
    tokenizer=AutoTokenizer.from_pretrained(json.loads(plan.read_text())['recipe']['model_path'],local_files_only=True,trust_remote_code=False)
    paired={};checks=0
    for r,g in zip(records,journal):
        for key in ('task','qid','group_id','budget','ds_on','output_ids','eos_ids','generation_scalars','forced','family','arm'):
            req(r[key]==g[key],'generation journal mismatch');checks+=1
        req(tokenizer.decode(r['output_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['answer'],'token decode');checks+=1
        start=r['image_start']; length=r['original_length'];spatial=r['spatial_ids']
        req(len(spatial)==256 and spatial==sorted(set(r['ordered_ids'])) and len(set(spatial))==256,'selected spatial IDs')
        retained=list(range(start))+[start+i for i in spatial]+list(range(start+1024,length))
        req(r['retained_ids']==retained,'retained IDs')
        full=np.asarray(r['full_position_ids'],dtype=np.int64).reshape(3,length)
        req(r['prefill_position_ids']==full[:,retained][:,None,:].tolist(),'prefill P3 positions')
        ref=refs[(r['task'],str(r['qid']),'P3-D1')]
        for a,b in (('processor','processor'),('main_visual','visual_sha256'),('ordered_ids','ordered_ids'),('spatial_ids','spatial_ids'),('retained_ids','retained_ids'),('prefill_position_ids','prefill_position_ids')):
            req(r[a]==ref[b],'E1 identity '+a);checks+=1
        if not r['forced'] and r['family']=='E2' and r['arm'] in ('000','111'):
            e=refs[(r['task'],str(r['qid']),'P3-D0' if r['arm']=='000' else 'P3-D1')]
            for key in ('output_ids','answer','eos_ids','eos','truncated'):req(r[key]==e[key],'E1 endpoint '+key);checks+=1
            req(r['prefill_logits']==e['logits'][0],'E1 prefill logits');checks+=1
        req(len(r['decode_trace'])==len(r['output_ids']),'decode inventory')
        for step,z in enumerate(r['decode_trace']):
            req(z['layer_cache_lengths_after']==[len(retained)+step]*36,'KV lengths');checks+=36
        with np.load(r['arrays']['path'],allow_pickle=False) as arrays:
            req(metadata_matches(arrays['visual'],r['main_visual']),'main visual hash')
            req(metadata_matches(arrays['visual'][spatial],r['pruned_visual']),'selected visual hash')
            req(metadata_matches(arrays['prefill'],r['prefill_embeds']),'prefill hash')
            req(np.array_equal(arrays['prefill'][0,start:start+256],arrays['visual'][spatial]),'prefill visual rows')
            for i in range(3):
                before=arrays['selected_before'+str(i)];full_ds=arrays['ds'+str(i)]
                req(metadata_matches(full_ds,r['full_deepstack'][i]) and np.array_equal(before,full_ds[spatial]),'DS selected rows')
                off=r['family']=='REF' and r['arm']=='D0'
                if off:
                    req('injected'+str(i) not in arrays and r['injected_deepstack'] is None,'reference off');continue
                actual=arrays['injected'+str(i)];req(metadata_matches(actual,r['injected_deepstack'][i]),'injected hash')
                if r['family']=='E2':want=before if r['arm'][i]=='1' else np.zeros_like(before)
                elif r['family']=='REF':want=before
                elif r['arm']=='PERM':
                    h=hashlib.sha256(('20261104|'+r['image_sha256']+'|perm').encode()).digest();rng=random.Random(int.from_bytes(h[:8],'big'));perm=list(range(256))
                    for j in range(255,0,-1):
                        k=rng.randrange(j);perm[j],perm[k]=perm[k],perm[j]
                    req(perm==r['intervention_metadata']['permutation'] and all(j!=v for j,v in enumerate(perm)),'permutation')
                    want=before[perm]
                elif r['arm']=='MEAN':
                    dtype={'torch.bfloat16':torch.bfloat16,'torch.float16':torch.float16,'torch.float32':torch.float32}[r['full_deepstack'][i]['dtype']]
                    z=torch.from_numpy(before.copy()).view(dtype).reshape(256,-1)
                    want=z.mean(0,keepdim=True).expand_as(z).contiguous().view(torch.uint8).numpy().reshape(actual.shape)
                else:raise ValueError('unknown intervention')
                req(np.array_equal(actual,want),'injected stream mismatch');checks+=1
        if r['forced'] and (r['family']=='REF' or (r['family']=='E2' and r['arm'] in ('000','111'))):
            fp=r['cache_fingerprints'];req(len(fp)==33 and all(len(x)==36 for x in fp),'cache fingerprint inventory')
            for layer in (y for x in fp for y in x):
                req(set(layer)=={'key','value'} and all(len(layer[k]['sha256'])==64 for k in layer),'cache metadata')
            paired[(r['task'],str(r['qid']),r['family'],r['arm'])]=fp
    for item in frozen['selected']:
        prefix=(item['task'],str(item['qid']))
        for arm,ref in (('000','D0'),('111','D1')):
            req(paired[(*prefix,'E2',arm)]==paired[(*prefix,'REF',ref)],'forced KV content identity');checks+=1
    proof={'status':'passed','independent_review':True,'n_records':len(records),'GT_opened':False,'scoring_performed':False,
           'formal_sha256':sha(FORMAL),'accept_terminal_sha256':sha(SMOKE/'monitor_accept001/terminal.json'),
           'run_terminal_sha256':sha(SMOKE/'monitor_001/terminal.json'),'reviewer_sha256':sha(__file__),
           'n_checks':checks,'n_kv_reference_pairs':6,'kv_steps':33,'finished_unix':time.time(),
           'limitations':['Three DEV examples only; full arrays are diagnostic, KV content stored as fingerprints.']}
    out=Path(__file__).resolve().parent/'e2e3_run001_review.json'
    with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
    print(json.dumps({'status':'passed','records':len(records),'checks':checks}))

if __name__=='__main__':main()
