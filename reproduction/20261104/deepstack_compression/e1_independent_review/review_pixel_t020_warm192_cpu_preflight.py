"""Independent CPU-only preflight review; no GPU or new results."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

B=Path(__file__).resolve().parents[1];D=B/'e6/pixel_wavefront_t020_warm192';I=B/'e1_independent_review'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def supervised(mon,script):
    s=read(mon/'started.json');t=read(mon/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
    if (len(cmd)<2 or (cwd/cmd[1]).resolve()!=script.resolve() or s.get('worker_pid')!=t.get('worker_pid')
        or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[]):
        raise ValueError('unfinished supervisor '+str(mon))

def rank(seed,purpose,row):
    data=[seed,purpose,row['task'],str(row['qid']),row['group_id'],row['image_sha256']]
    return hashlib.sha256(json.dumps(data,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.resolve()!=(I/'pixel_t020_warm192_cpu_preflight_review.json').resolve():raise ValueError('output path')
    plan=read(D/'preflight_plan001.json');source=plan['source_sha256'];cpu=read(D/'cpu_proof001.json');design=read(D/'design001.json');rule=read(D/'rule001.json')
    if any(sha(k)!=v for k,v in source.items()) or plan.get('GPU_authorized') is not False or plan.get('GT_opened') is not False:raise ValueError('source or scope')
    supervised(D/'monitor_preflight_freeze001',D/'freeze_preflight.py');supervised(D/'monitor_cpu001',D/'cpu_check.py');supervised(D/'monitor_design001',D/'freeze_design.py');supervised(D/'monitor_worker_smoke001',D/'worker.py')
    if cpu.get('status')!='passed' or cpu.get('GPU_used') is not False or cpu.get('GT_opened') is not False or cpu.get('source_sha256_before')!=cpu.get('source_sha256_after') or any(source.get(k)!=v for k,v in cpu['source_sha256_after'].items()):raise ValueError('CPU proof')
    smoke=read(D/'monitor_worker_smoke001/started.json');smokeout=read(D/'monitor_worker_smoke001/stdout.log')
    if '--no-gpu-dependency-smoke' not in smoke['command'] or smokeout.get('actual_inputs')!=193 or smokeout.get('cuda_initialized') is not False or smokeout.get('model_loaded') is not False:raise ValueError('worker smoke')
    extra=B/'e6/pixel_wavefront_t020_warm192_extra_checks';host=read(extra/'host_entry002.json')
    supervised(extra/'monitor_host_entry002',extra/'host_entry_smoke002.py')
    if (host.get('status')!='passed' or host.get('GPU_queried') is not False or host.get('model_loaded') is not False
        or host.get('GT_opened') is not False or host.get('preflight_plan_sha256')!=sha(D/'preflight_plan001.json')
        or any(sha(k)!=v for k,v in host['source_sha256'].items())):raise ValueError('real host entry smoke')
    # Reconstruct E4 measured cohort and accepted E1 identities without labels.
    sys.path.insert(0,str(B/'e4_c_latency'));import input_preflight
    e1=B/'e1';rows,refs=input_preflight.load_accepted_e1_references(e1/'e1full001_completion.json',e1/'monitorverifye1full001/started.json',e1/'monitorverifye1full001/terminal.json')
    bygroup={(x['task'],x['group_id']):x for x in rows};cohort=read(B/'e4_c_latency_full/full_001/frozen.json')['schedule']['cohort']
    tasks=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST');chosen=[]
    for task in tasks:
        groups=cohort[task]['measured'];warm=cohort[task]['warmup']
        if len(groups)!=64 or len(set(groups))!=64 or set(groups)&set(warm):raise ValueError('cohort')
        for group in groups:
            row=bygroup[(task,group)];ref=refs[(task,str(row['qid']))]
            if row.get('split')!='DEV' or any(str(ref[k])!=str(row[k]) for k in ('task','qid','group_id','image_sha256','message_sha256')):raise ValueError('E1 identity')
            chosen.append({**{k:row[k] for k in ('task','qid','group_id','image_path','image_sha256','message_sha256')},'qid':str(row['qid']),'image_start':ref['image_start'],'processor':ref['processor']})
    if len(chosen)!=192 or len({x['group_id'] for x in chosen})!=192 or Counter(x['task'] for x in chosen)!=Counter({t:64 for t in tasks}):raise ValueError('192 inventory')
    blocks=[]
    for block in range(4):
        items=[]
        for task in tasks:
            ordered=sorted((x for x in chosen if x['task']==task),key=lambda x:(rank(rule['seed'],'partition',x),str(x['qid']),x['group_id']))
            items.extend(ordered[16*block:16*(block+1)])
        blocks.append(sorted(items,key=lambda x:(rank(rule['seed'],'block-'+str(block),x),x['task'],str(x['qid']),x['group_id'])))
    if blocks!=design['blocks']:raise ValueError('independent block reconstruction')
    warm_group=cohort[tasks[0]]['warmup'][0];row=bygroup[(tasks[0],warm_group)];ref=refs[(row['task'],str(row['qid']))]
    warm={**{k:row[k] for k in ('task','qid','group_id','image_path','image_sha256','message_sha256')},'qid':str(row['qid']),'image_start':ref['image_start'],'processor':ref['processor']}
    if warm_group in {x['group_id'] for x in chosen}:raise ValueError('warmup overlap')
    expected=[]
    for block,arms in enumerate((('author','wavefront'),('wavefront','author'),('wavefront','author'),('author','wavefront'))):
        for arm in arms:expected.append({'order':len(expected),'block':block,'arm':arm,'warmup':warm,'measured':blocks[block]})
    if expected!=design['processes'] or len(expected)!=8 or sum(len(x['measured']) for x in expected)!=384:raise ValueError('process pairing')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-v'],cwd=D,env=env,capture_output=True,text=True,timeout=120)
    if result.returncode or 'Ran 19 tests' not in result.stderr:raise ValueError('independent synthetic tests')
    if any(sha(k)!=v for k,v in source.items()):raise ValueError('source changed during review')
    output={'status':'passed','independent_review':True,'preflight_only':True,'GPU_authorized':False,'GT_opened':False,
            'existing_DEV_predictions_opened':True,'errors':[],'source_sha256':source,'plan_sha256':sha(D/'preflight_plan001.json'),
            'reviewer_sha256':sha(__file__),'questions':192,'input_identities':193,'processes':8,'measured':384,'warmups':8,
            'worker_smoke_started_sha256':sha(D/'monitor_worker_smoke001/started.json'),'worker_smoke_terminal_sha256':sha(D/'monitor_worker_smoke001/terminal.json'),
            'host_smoke_sha256':sha(extra/'host_entry002.json'),'host_smoke_started_sha256':sha(extra/'monitor_host_entry002/started.json'),
            'host_smoke_terminal_sha256':sha(extra/'monitor_host_entry002/terminal.json'),
            'limits':['only noGPU preflight allowed, not model/GPU or quality','E1 mixed DEV journal parsed for identity',
                      'four prior t020 smoke generated files and previous journals opened only for source SHA',
                      'GPU launcher dependency smoke is independently inspected but not in current preflight plan source']}
    with a.out.open('x') as f:json.dump(output,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
