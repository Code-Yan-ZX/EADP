"""Independent no-GT review after the complete timing run and formal verifier exit.

Reads the accepted completed mixed DEV-prediction journal for endpoint identity;
never reports or aggregates durations. Observer windows are validated, not used
for selection or exclusion. All earlier intermediate data remain untouched.
"""
import argparse,hashlib,json,os,subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
B=Path(__file__).resolve().parent.parent;I=B/'e1_independent_review';D=B/'e4_c_multibudget_worker_v3';A=B/'e4_c_multibudget_worker_accept_v3';E=B/'e4_c_multibudget_environment';O=D/'run_001'
KS=(128,256,512,1024);TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST');QUOTAS=(11,11,10);SEED='20261105|e4c-four-k-pilot-v1'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def jsonl(path):
 with Path(path).open() as f:return [json.loads(x) for x in f if x.strip()]
def term(folder,script,out=None):
 a=read(folder/'started.json');b=read(folder/'terminal.json');cmd=a['command'];cwd=Path(a['cwd'])
 if (b.get('reason')!='completed' or b.get('worker_returncode')!=0 or b.get('remaining_process_group')!=[] or type(a.get('worker_pid')) is not int or a['worker_pid']<=0 or a['worker_pid']!=b.get('worker_pid') or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()))):raise ValueError('supervisor not complete: '+str(folder))
 return a,b
def rank(*parts):return hashlib.sha256('|'.join(map(str,(SEED,*parts))).encode()).hexdigest()
def rebuild(cohort,identities):
 by={(x['task'],x['group_id']):x for x in identities}
 if len(by)!=62:raise ValueError('identity groups not unique')
 chosen={};measured=[];warm=[]
 for task,q in zip(TASKS,QUOTAS):
  src=cohort[task];m=src['measured'];w=src['warmup']
  if len(set(m))!=64 or len(m)!=64 or len(set(w))!=10 or len(w)!=10 or set(m)&set(w):raise ValueError('bad parent cohort')
  selected=sorted(m,key=lambda g:rank('pilot',task,g))[:q];chosen[task]={'measured':selected,'warmup':list(w)}
  measured += [by[(task,g)] for g in selected];warm += [by[(task,g)] for g in w]
 jobs=[]
 for phase,items,repeats in (('warmup',warm,1),('measure',measured,5)):
  for rep in range(repeats):
   for row in sorted(items,key=lambda r:rank('image',phase,rep,r['task'],r['group_id'])):
    arms=sorted([(k,p) for k in KS for p in ('natural','fixed32')],key=lambda z:rank('arm',phase,rep,row['group_id'],z[0],z[1]))
    for k,protocol in arms:jobs.append({'phase':phase,'replicate':rep,'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'budget':k,'protocol':protocol,'order':len(jobs)})
 if len(jobs)!=1520:raise ValueError('schedule size')
 return chosen,jobs

def check_job(job,triplet,panel,fixed,decode,namespace):
 if [x.get('event') for x in triplet]!=['job_started','raw_generation','job_completed'] or any(x.get('job')!=job for x in triplet):raise ValueError('event identity')
 raw=triplet[1]['raw'];r=triplet[2]['record']
 for name in ('task','qid','group_id','budget','protocol','output_ids','generation_scalars','min_new_tokens','eos_ids'):
  if raw.get(name)!=r.get(name):raise ValueError('raw mismatch '+name)
 for name in ('task','qid','group_id','budget','protocol'):
  if str(r.get(name))!=str(job[name]):raise ValueError('record/job mismatch '+name)
 if r.get('selector_calls')!=(0 if job['budget']==1024 else 1):raise ValueError('selection path')
 fields=('output_ids','answer','generation_scalars','eos_ids')
 reference=panel if job['protocol']=='natural' else fixed
 if any(r.get(x)!=reference.get(x) for x in fields):raise ValueError('reference output/config mismatch')
 ids=r.get('output_ids');eos=r.get('eos_ids')
 if not isinstance(ids,list) or not ids or any(type(x) is not int for x in ids) or not isinstance(eos,list) or not eos or any(x in eos for x in ids[:-1]):raise ValueError('invalid tokens')
 if job['protocol']=='natural':
  if r.get('min_new_tokens') is not None or any(r.get(k)!=panel.get(k) for k in ('eos','truncated')) or len(ids)>2048 or not (ids[-1] in eos or len(ids)==2048):raise ValueError('natural termination')
 elif job['protocol']=='fixed32':
  if len(ids)!=32 or r.get('min_new_tokens')!=32 or fixed.get('min_new_tokens')!=32 or r['generation_scalars'].get('max_new_tokens')!=32:raise ValueError('fixed32 termination')
 else:raise ValueError('unknown protocol')
 if decode(ids)!=r['answer']:raise ValueError('decode mismatch')
 b=r['boundaries'];a=b['t0_before_image_read_ns'];x=b['first_token_selected_ns'];z=b['generation_complete_ns']
 if any(type(v) is not int or v<0 for v in (a,x,z)) or not a<x<=z or raw.get('generation_complete_ns')!=z or b.get('first_criteria_length')!=1 or b.get('criteria_calls')!=len(ids) or b.get('clock_implementation')!='clock_gettime(CLOCK_MONOTONIC)' or b.get('explicit_sync_boundaries')!=['before_t0','first_token','generate_complete'] or b.get('time_namespace')!=namespace:raise ValueError('timing boundary')
 return (a,x,z)
def validate_observer(samples,health,manual,namespace,start_utc,end_utc):
 def utc(s):
  value=datetime.fromisoformat(s.replace('Z','+00:00'))
  if value.tzinfo is None:raise ValueError('UTC lacks timezone')
  return value
 if len(samples)<3 or len(samples)%2!=1:raise ValueError('incomplete observation')
 end=samples[-1]
 if end.get('event')!='watch_end' or end.get('reason')!='stop_marker' or end.get('time_namespace')!=namespace or end.get('samples')!=(len(samples)-1)//2:raise ValueError('watcher end')
 prior=-1
 for i in range((len(samples)-1)//2):
  a,b=samples[i*2:i*2+2];x=a.get('query_start_ns');y=a.get('query_end_ns');z=b.get('first_fsync_end_ns')
  if a.get('event')!='snapshot' or b.get('event')!='logging_complete' or a.get('index')!=i or b.get('index')!=i or a.get('time_namespace')!=namespace or b.get('time_namespace')!=namespace or any(type(v) is not int for v in (x,y,z)) or not prior<x<=y<=z or b.get('query_start_ns')!=x or a.get('returncode')!=0 or not isinstance(a.get('stdout'),str):raise ValueError('watcher pair')
  for value in (a['query_start_utc'],a['query_end_utc'],b['first_fsync_end_utc']):utc(value)
  prior=z
 if type(end.get('end_ns')) is not int or end['end_ns']<prior or utc(samples[0]['query_start_utc'])>utc(start_utc) or utc(end['end_utc'])<utc(end_utc):raise ValueError('watcher incomplete coverage')
 if not health:raise ValueError('observer health missing')
 for h in health:
  v=h.get('health_snapshot_window_monotonic_ns')
  if h.get('time_namespace')!=namespace or not isinstance(v,list) or len(v)!=2 or any(type(x) is not int for x in v) or v[0]>v[1]:raise ValueError('observer health window')
 prior=-1
 for v in manual:
  a=v.get('start_monotonic_ns');b=v.get('end_monotonic_ns')
  if v.get('event')!='manual_health_check' or v.get('time_namespace')!=namespace or type(a) is not int or type(b) is not int or a<prior or a>b or not isinstance(v.get('details'),dict):raise ValueError('manual health window')
  utc(v['start_utc']);utc(v['end_utc']);prior=b
 return {'snapshots':(len(samples)-1)//2,'observer_health_records':len(health),'manual_health_records':len(manual),'manual_missing_is_unknown':not bool(manual)}

def main(out):
 # These three completed supervisors and the formal proof are checked before
 # opening any prediction-bearing output from this run.
 rs,rt=term(D/'monitor_run001',D/'launch.py');term(A/'monitor_snapshot001',A/'snapshot.py');term(A/'monitor_verify001',A/'verify.py',A/'verification001.json')
 formal=read(A/'verification001.json');vp=read(A/'verification_plan001.json');plan=read(D/'runtime_plan001.json')
 if (formal.get('status')!='passed' or (formal.get('jobs'),formal.get('warmups'),formal.get('measured'))!=(1520,240,1280) or formal.get('GT_opened') is not False or formal.get('scoring_performed') is not False or formal.get('errors')!=[] or formal.get('verification_plan_sha256')!=sha(A/'verification_plan001.json') or formal.get('verifier_sha256')!=sha(A/'verify.py')):raise ValueError('formal acceptance')
 for p,digest in vp['source_sha256'].items():
  if sha(p)!=digest:raise ValueError('verifier source changed')
 paths={'run_started_sha256':D/'monitor_run001/started.json','run_terminal_sha256':D/'monitor_run001/terminal.json','launch_sha256':O/'launch.json','exit_sha256':O/'exit.json','worker_complete_sha256':O/'worker_complete.json','health_sha256':D/'monitor_run001/health.jsonl','snapshot_manifest_sha256':D/'run_001_sources/manifest.json','snapshot_started_sha256':A/'monitor_snapshot001/started.json','snapshot_terminal_sha256':A/'monitor_snapshot001/terminal.json'}
 if any(formal.get(k)!=sha(p) for k,p in paths.items()):raise ValueError('formal source/exit changed')
 snap=read(paths['snapshot_manifest_sha256']);sources=plan['source_sha256']
 if snap.get('status')!='passed' or snap.get('runtime_plan_sha256')!=sha(D/'runtime_plan001.json') or snap.get('snapshot_script_sha256')!=sha(A/'snapshot.py') or set(snap['sources'])!=set(sources):raise ValueError('snapshot inventory')
 for p,h in sources.items():
  item=snap['sources'][p];cp=Path(item['snapshot']).resolve()
  if item.get('source')!=p or item.get('sha256')!=h or not cp.is_relative_to((D/'run_001_sources').resolve()) or sha(cp)!=h:raise ValueError('snapshot bytes')
 launch=read(O/'launch.json');ex=read(O/'exit.json');cid=(ex.get('container_identity') or {}).get('container_id');last=ex.get('last_inspect') or {}
 if (not cid or (launch.get('controller') or {}).get('pid')!=rs['worker_pid'] or ex.get('controller')!=launch.get('controller') or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('container_stopped_verified') is not True or ex.get('client_exited_while_container_running') is not False or ex.get('gpu_pids_after')!=[] or last.get('running') is not False or last.get('pid')!=0 or last.get('id')!=cid):raise ValueError('launcher exit')
 doc=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=15).stdout)[0];state=doc['State']
 if (doc['Id']!=cid or state['Running'] is not False or state['Pid']!=0 or state['ExitCode']!=0 or state['OOMKilled'] is not False or doc['Image']!=last['image'] or doc['HostConfig']['NetworkMode']!='none' or doc['Config']['Cmd']!=[str(D/'worker.py'),'--out',str(O)]):raise ValueError('actual Docker state/command')
 # Require completed watcher/health evidence before reading this run's journal.
 wp=read(E/'plan001.json');ws,wt=term(E/'monitor_watch001',E/'watch.py')
 if sha(E/'plan001.json')!='3dd79ba0d438946d73b2c3565f72804083b22738a88eec2928e6835256a85a32' or ws['command']!=wp['command'][wp['command'].index('--')+1:] or Path(ws['cwd']).resolve()!=B or not (E/'stop001').is_file():raise ValueError('observer command/stop')
 observed=validate_observer(jsonl(E/'samples001.jsonl'),jsonl(E/'monitor_watch001/health.jsonl'),jsonl(E/'manual_checks001.jsonl'),launch['time_namespace'],rs['created_utc'],rt['utc'])
 # Existing DEV endpoints have their own independent, completed acceptance.
 panelroot=B/'e4_panel_v2';pf=read(panelroot/'panel_003_verification.json');pr=read(I/'e4panel003_review.json')
 term(panelroot/'monitor_verify_panel003',panelroot/'panel_verify.py',panelroot/'panel_003_verification.json');term(I/'monitor_e4panel003',I/'review_e4panel.py',I/'e4panel003_review.json')
 if pf.get('status')!='passed' or pf.get('n_questions')!=384 or pf.get('all_arms_validated')!=3072 or pr.get('status')!='passed' or pr.get('n_verified')!=384 or pr.get('arms_verified')!=3072 or pr.get('formal_proof_sha256')!=sha(panelroot/'panel_003_verification.json') or pr.get('reviewer_sha256')!=sha(I/'review_e4panel.py') or pf.get('GT_opened') is not False or pr.get('GT_opened') is not False:raise ValueError('E4 reference admission')
 panel_path=panelroot/'panel_003/records.jsonl'
 if pf['files_sha256'].get(str(panel_path.resolve()))!=sha(panel_path):raise ValueError('E4 reference bytes')
 fixedroot=B/'e4_c_multibudget_references_v3';fa=B/'e4_c_multibudget_references_accept_v3';fixedfile=fixedroot/'run_001/diagnostic/records.jsonl';ff=read(fa/'verification001.json');fr=read(I/'e4c_multibudget_references_v3_result_review.json')
 term(fa/'monitor_verify001',fa/'verify.py',fa/'verification001.json');term(I/'monitor_e4c_multibudget_references_v3_result',I/'review_e4c_multibudget_references_v3_result.py',I/'e4c_multibudget_references_v3_result_review.json');term(fixedroot/'monitor_run001',fixedroot/'launch.py')
 if (ff.get('status')!='passed' or (ff.get('inputs'),ff.get('generations'))!=(62,248) or ff.get('GT_opened') is not False or ff.get('scoring_performed') is not False or ff.get('records_sha256')!=sha(fixedfile) or fr.get('status')!='passed' or fr.get('independent_review') is not True or fr.get('formal_sha256')!=sha(fa/'verification001.json') or fr.get('records_sha256')!=sha(fixedfile) or fr.get('reviewer_sha256')!=sha(I/'review_e4c_multibudget_references_v3_result.py')):raise ValueError('fixed32 reference admission')
 sample=read(B/'e4_c_multibudget_expansion/pilot_plan001.json');source=read(B/'e4_c_latency_full/full_001/frozen.json')['schedule']['cohort'];cohort,jobs=rebuild(source,sample['input_identities'])
 if cohort!=sample['cohort'] or jobs!=sample['jobs'] or read(O/'frozen.json')['schedule']!=jobs:raise ValueError('schedule mismatch')
 anchors={(x['task'],str(x['qid'])):x for x in sample['input_identities']};panels={};fixed={}
 for p in jsonl(panel_path):
  key=(p['task'],str(p['qid']))
  if key in anchors and p.get('ds_on') is True and p.get('forced') is False and p.get('budget') in KS:
   k=(*key,p['budget'])
   if k in panels or any(p.get(x)!=anchors[key].get(x) for x in ('group_id','image_sha256','message_sha256')):raise ValueError('natural reference mismatch')
   panels[k]=p
 for p in jsonl(fixedfile):
  key=(p['task'],str(p['qid']));k=(*key,p['budget'])
  if key not in anchors or k in fixed or any(p.get(x)!=anchors[key].get(x) for x in ('group_id','image_sha256','message_sha256')) or p.get('protocol')!='fixed32':raise ValueError('fixed reference mismatch')
  fixed[k]=p
 if len(panels)!=248 or len(fixed)!=248:raise ValueError('reference inventory')
 # Journal is opened only after real exits, formal proof and reference gates.
 journal=O/'journal.jsonl'
 if formal.get('journal_sha256')!=sha(journal):raise ValueError('accepted journal changed')
 events=jsonl(journal)
 if len(events)!=4560:raise ValueError('not all jobs generated')
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',local_files_only=True,trust_remote_code=False)
 decode=lambda ids:tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
 model_health=jsonl(D/'monitor_run001/health.jsonl');windows=[]
 for h in model_health:
  v=h.get('health_snapshot_window_monotonic_ns')
  if h.get('time_namespace')!=launch['time_namespace'] or not isinstance(v,list) or len(v)!=2 or any(type(x) is not int for x in v) or v[0]>v[1]:raise ValueError('model health window')
  windows.append(v)
 if not windows or windows!=formal.get('health_windows') or len(formal.get('annotations',[]))!=1520:raise ValueError('model health formal binding')
 counts=Counter()
 for i,job in enumerate(jobs):
  key=(job['task'],str(job['qid']),job['budget']);ticks=check_job(job,events[3*i:3*i+3],panels[key],fixed[key],decode,launch['time_namespace']);ann=formal['annotations'][i]
  overlaps=lambda a,b:[j for j,(u,v) in enumerate(windows) if u<=b and a<=v]
  if ann.get('order')!=i or any(ann.get(k)!=job[k] for k in ('phase','task','budget','protocol')) or ann.get('ttft_health_overlap_indices')!=overlaps(ticks[0],ticks[1]) or ann.get('full_job_health_overlap_indices')!=overlaps(ticks[0],ticks[2]):raise ValueError('formal health annotation')
  counts[job['phase']]+=1
 if counts!=Counter({'warmup':240,'measure':1280}):raise ValueError('phase denominator')
 proof={'status':'passed','independent_review':True,'jobs':1520,'warmups':240,'measured':1280,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'errors':[],'formal_sha256':sha(A/'verification001.json'),'formal_started_sha256':sha(A/'monitor_verify001/started.json'),'formal_terminal_sha256':sha(A/'monitor_verify001/terminal.json'),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'journal_sha256':sha(journal),'health_sha256':sha(D/'monitor_run001/health.jsonl'),'snapshot_manifest_sha256':sha(paths['snapshot_manifest_sha256']),'reviewer_sha256':sha(__file__),'observer_samples_sha256':sha(E/'samples001.jsonl'),'observer_health_sha256':sha(E/'monitor_watch001/health.jsonl'),'observer_started_sha256':sha(E/'monitor_watch001/started.json'),'observer_terminal_sha256':sha(E/'monitor_watch001/terminal.json'),'manual_checks_sha256':sha(E/'manual_checks001.jsonl'),'observer':observed,'limits':['All jobs retained; no timing values summarized here.','Existing completed DEV mixed predictions parsed for endpoint matching; no GT opened.','Observer second fsync, manual log fsync and residual host activity remain unknown; UTC coverage depends on wall-clock stability.','Synchronized completion excludes token copy/fsync and text decode.']}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
 if a.out.resolve()!=I/'e4c_multibudget_worker_v3_result_review.json' or a.out.exists():raise ValueError('fixed output required')
 main(a.out)
