"""Independent completed-pilot statistics rebuild; never opens ground truth."""
import argparse,hashlib,json,os,sys
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent.parent;I=B/'e1_independent_review';A=B/'e4_c_multibudget_analysis';R=B/'e4_c_multibudget_worker_v3';E=B/'e4_c_multibudget_environment';F=B/'e4_c_multibudget_worker_accept_v3';SEED=20261105;N=10000;KS=(128,256,512,1024);TASKS={'DocVQA_VAL':11,'TextVQA_VAL':11,'AI2D_TEST':10}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def jsonl(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def term(directory,script,out=None):
 s=read(directory/'started.json');t=read(directory/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or [(cwd/x).resolve() for x in c if Path(x).name==script.name]!=[script.resolve()] or (out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervisor incomplete '+str(directory))
 return s,t
def aggregate(jobs,events):
 if len(jobs)!=1520 or len(events)!=4560:raise ValueError('incomplete journal')
 groups=defaultdict(list);outcome=defaultdict(Counter);phase=Counter()
 for i,j in enumerate(jobs):
  e=events[3*i:3*i+3]
  if [x.get('event') for x in e]!=['job_started','raw_generation','job_completed'] or any(x.get('job')!=j for x in e):raise ValueError('event identity')
  raw=e[1]['raw'];r=e[2]['record'];b=r['boundaries'];a=b['t0_before_image_read_ns'];x=b['first_token_selected_ns'];z=b['generation_complete_ns']
  if not (type(a) is type(x) is type(z) is int and 0<=a<x<=z and raw.get('generation_complete_ns')==z):raise ValueError('time bounds')
  phase[j['phase']]+=1
  if j['phase']=='measure':
   key=(j['task'],j['protocol'],j['group_id'],str(j['qid']),j['budget']);groups[key].append((j['replicate'],(x-a)/1e6,(z-a)/1e6));outcome[(j['task'],j['protocol'],j['budget'])].update({'eos':int(bool(r['eos'])),'truncated':int(bool(r['truncated'])),'tokens':len(r['output_ids'])})
 if phase!=Counter({'warmup':240,'measure':1280}) or len(groups)!=256 or any(len(x)!=5 or len(set(r for r,_,_ in x))!=5 for x in groups.values()):raise ValueError('denominator')
 out={}
 for task,n in TASKS.items():
  out[task]={};proto_id=[]
  for protocol in ('natural','fixed32'):
   ids=sorted({(g,q) for t,p,g,q,k in groups if t==task and p==protocol});proto_id.append(ids)
   if len(ids)!=n or len({g for g,q in ids})!=n:raise ValueError('group denominator')
   rng=np.random.default_rng(SEED);draw=rng.integers(0,n,size=(N,n));values={}
   for k in KS:
    values[k]=np.array([[float(np.median([x[1] for x in groups[(task,protocol,g,q,k)]])),float(np.median([x[2] for x in groups[(task,protocol,g,q,k)]]))] for g,q in ids])
   rows={}
   for k in KS:
    d={}
    for col,name in enumerate(('ttft_ms','generation_complete_ms')):
     value=values[k][:,col];base=values[256][:,col]
     if np.any(value<=0) or np.any(base<=0) or not np.isfinite(value).all():raise ValueError('duration')
     def dist(x):return {'median':float(np.median(x)),'min':float(np.min(x)),'max':float(np.max(x)),'bootstrap_ci95':[float(v) for v in np.quantile(np.median(x[draw],axis=1),[.025,.975])]}
     d[name]={'time':dist(value),'paired_difference_vs_k256_ms':dist(value-base),'paired_ratio_vs_k256':dist(value/base)}
    d['counts']={'images':n,'runs':n*5,**dict(outcome[(task,protocol,k)])};rows[str(k)]=d
   out[task][protocol]=rows
  if proto_id[0]!=proto_id[1]:raise ValueError('unpaired cohorts')
 return out

def overlaps(windows,a,b):return [i for i,(x,y) in enumerate(windows) if x<=b and a<=y]
def validate_details(jobs,events,result,formal,watch,watchhealth,manual,namespace):
 if len(result['jobs_detail'])!=1520 or len(formal['annotations'])!=1520:raise ValueError('missing job detail')
 wins={'model_health':formal['health_windows'],'query':[[watch[i]['query_start_ns'],watch[i]['query_end_ns']] for i in range(0,len(watch)-1,2)],'query_first_fsync':[[watch[i]['query_start_ns'],watch[i+1]['first_fsync_end_ns']] for i in range(0,len(watch)-1,2)],'watcher_health':[x['health_snapshot_window_monotonic_ns'] for x in watchhealth],'manual_health':[[x['start_monotonic_ns'],x['end_monotonic_ns']] for x in manual]}
 if result.get('observer_windows')!=wins:raise ValueError('window inventory mismatch')
 summary={}
 for i,j in enumerate(jobs):
  rec=events[3*i+2]['record'];b=rec['boundaries'];a=b['t0_before_image_read_ns'];x=b['first_token_selected_ns'];z=b['generation_complete_ns'];item=result['jobs_detail'][i]
  ids={k:j[k] for k in ('order','phase','task','qid','group_id','budget','protocol','replicate')};ids.update(t0_before_image_read_ns=a,first_token_selected_ns=x,generation_complete_ns=z,token_count=len(rec['output_ids']),eos=rec['eos'],truncated=rec['truncated'])
  if any(item.get(k)!=val for k,val in ids.items()) or b['time_namespace']!=namespace:raise ValueError('job detail mismatch')
  expected={k:{'ttft':overlaps(v,a,x),'generation':overlaps(v,a,z)} for k,v in wins.items()}
  if item.get('overlap_indices')!=expected:raise ValueError('overlap mismatch')
  if j['phase']=='measure':
   key='/'.join((j['task'],j['protocol'],str(j['budget'])));s=summary.setdefault(key,{'measured_jobs':0,'overlap_counts':{k:{'ttft':0,'generation':0} for k in wins}});s['measured_jobs']+=1
   for kind in wins:
    for boundary in ('ttft','generation'):s['overlap_counts'][kind][boundary]+=bool(expected[kind][boundary])
 if result.get('health_overlap')!=summary:raise ValueError('aggregate health windows mismatch')
 first=min(x['t0_before_image_read_ns'] for x in result['jobs_detail']);last=max(x['generation_complete_ns'] for x in result['jobs_detail']);hours=[];period=3600*10**9;slack=300*10**9;n=1;health=wins['model_health']+wins['watcher_health']
 while first+n*period<=last:
  target=first+n*period;mi=[i for i,(a,b) in enumerate(wins['manual_health']) if a<=target+slack and b>=target-slack];hi=[i for i,(a,b) in enumerate(health) if a<=target+slack and b>=target-slack]
  hours.append({'hour':n,'manual_indices':mi,'supervisor_health_indices':hi,'manual_status':'observed' if mi else 'unknown','supervisor_health_status':'observed' if hi else 'unknown','status':'some_health_observed' if mi or hi else 'unknown'});n+=1
 if result.get('hourly_health_coverage')!=hours or result.get('manual_check_evidence')!=('recorded' if manual else 'unknown: empty manual log'):raise ValueError('hourly coverage mismatch')
 return {'jobs':len(jobs),'hourly_entries':len(hours),'manual_hour_unknown':sum(x['manual_status']=='unknown' for x in hours),'supervisor_hour_unknown':sum(x['supervisor_health_status']=='unknown' for x in hours)}

def main(out):
 if np.__version__!='2.2.6':raise ValueError('NumPy differs from frozen plan')
 stat=A/'analysis004.json';term(A/'monitor_analysis004',A/'report.py',stat)
 term(A/'monitor_cpu004',A/'freeze.py');plan=read(A/'analysis_code_plan004.json');cpu=read(A/'cpu_proof004.json')
 if plan.get('status')!='frozen' or cpu.get('status')!='passed' or cpu.get('source_sha256_before')!=cpu.get('source_sha256_after') or any(sha(p)!=v for p,v in plan['source_sha256'].items()):raise ValueError('statistics source proof')
 review=I/'e4c_multibudget_worker_v3_result_review.json';term(I/'monitor_e4c_multibudget_worker_v3_result',I/'review_e4c_multibudget_worker_v3_result.py',review);term(F/'monitor_verify001',F/'verify.py',F/'verification001.json');term(R/'monitor_run001',R/'launch.py');term(E/'monitor_watch001',E/'watch.py')
 rv=read(review);formal=read(F/'verification001.json');j=R/'run_001/journal.jsonl'
 proof_fields={'formal_sha256':F/'verification001.json','formal_started_sha256':F/'monitor_verify001/started.json','formal_terminal_sha256':F/'monitor_verify001/terminal.json','run_started_sha256':R/'monitor_run001/started.json','run_terminal_sha256':R/'monitor_run001/terminal.json','journal_sha256':j,'health_sha256':R/'monitor_run001/health.jsonl','snapshot_manifest_sha256':R/'run_001_sources/manifest.json','reviewer_sha256':I/'review_e4c_multibudget_worker_v3_result.py','observer_samples_sha256':E/'samples001.jsonl','observer_health_sha256':E/'monitor_watch001/health.jsonl','observer_started_sha256':E/'monitor_watch001/started.json','observer_terminal_sha256':E/'monitor_watch001/terminal.json','manual_checks_sha256':E/'manual_checks001.jsonl'}
 if rv.get('status')!='passed' or rv.get('independent_review') is not True or (rv.get('jobs'),rv.get('warmups'),rv.get('measured'))!=(1520,240,1280) or rv.get('GT_opened') is not False or rv.get('scoring_performed') is not False or rv.get('errors')!=[] or formal.get('status')!='passed' or any(rv.get(k)!=sha(p) for k,p in proof_fields.items()):raise ValueError('completed independent result proof')
 # The above completed gates are checked before opening completed pilot timings.
 result=read(stat);events=jsonl(j);jobs=read(B/'e4_c_multibudget_expansion/pilot_plan001.json')['jobs']
 if result.get('status')!='completed' or (result.get('jobs'),result.get('warmups'),result.get('measured'),result.get('images'))!=(1520,240,1280,32) or result.get('GT_opened') is not False or result.get('scoring_performed') is not False or result.get('predictions_opened') is not True or result['admission'].get('independent_result_sha256')!=sha(review) or result['admission'].get('journal_sha256')!=sha(j) or result['admission'].get('analysis_code_plan_sha256')!=sha(A/'analysis_code_plan004.json'):raise ValueError('analysis identity')
 expected=aggregate(jobs,events)
 if result.get('tasks')!=expected:raise ValueError('independent paired statistics mismatch')
 watch=jsonl(E/'samples001.jsonl');health=jsonl(E/'monitor_watch001/health.jsonl');manual=jsonl(E/'manual_checks001.jsonl');namespace=read(R/'run_001/launch.json')['time_namespace'];summary=validate_details(jobs,events,result,formal,watch,health,manual,namespace)
 proof={'status':'passed','independent_review':True,'jobs':1520,'warmups':240,'measured':1280,'images':32,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'errors':[],'analysis_sha256':sha(stat),'analysis_started_sha256':sha(A/'monitor_analysis004/started.json'),'analysis_terminal_sha256':sha(A/'monitor_analysis004/terminal.json'),'analysis_plan_sha256':sha(A/'analysis_code_plan004.json'),'formal_sha256':sha(F/'verification001.json'),'independent_result_sha256':sha(review),'journal_sha256':sha(j),'observer_samples_sha256':sha(E/'samples001.jsonl'),'observer_health_sha256':sha(E/'monitor_watch001/health.jsonl'),'manual_checks_sha256':sha(E/'manual_checks001.jsonl'),'reviewer_sha256':sha(__file__),'numpy_version':np.__version__,'recomputed_cells':24,'bootstrap_samples_per_contrast':N,'background_validation':summary,'limitations':['Posthoc 32-image pilot, with task denominators 11/11/10.','No samples removed for possible interference; ps is sparse and process CPU is lifetime average.','Observer second fsync, manual-log fsync, stop marker and residual background activity unknown.','TTFT and synchronized generation completion stop before token copy, fsync and text decoding; not end-to-end latency.']}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
 if a.out.resolve()!=I/'e4c_multibudget_analysis004_review.json' or a.out.exists():raise ValueError('fixed output required')
 main(a.out)
