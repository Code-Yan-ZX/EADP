"""Independent post-run review: 48 EADP timing jobs, no GT or statistics."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parents[1];D=B/'e4_c_budget_ttft_pilot_v3';A=B/'e4_c_budget_ttft_accept_v3';I=B/'e1_independent_review';OUT=I/'e4c_budget_ttft_v3_result_review.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(start,end,script=None,out=None):
 s,t=read(start),read(end)
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or s.get('worker_pid')!=s.get('process_group'):raise ValueError('supervisor did not complete '+str(end))
 if script:
  c=s['command'];cwd=Path(s['cwd'])
  if len(c)<2 or (cwd/c[1]).resolve()!=script.resolve():raise ValueError('supervisor command mismatch')
  if out and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve()):raise ValueError('supervisor output mismatch')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def run():
 formal=A/'verification001.json';fs=A/'monitor_verify001/started.json';ft=A/'monitor_verify001/terminal.json';rs=D/'monitor_run001/started.json';rt=D/'monitor_run001/terminal.json'
 term(fs,ft,A/'verify.py',formal);term(rs,rt,D/'launch.py')
 f=read(formal)
 if f.get('status')!='passed' or [f.get(k) for k in ('jobs','warmups','measured')]!=[48,12,36] or f.get('errors')!=[] or f.get('GT_opened') is not False or f.get('scoring_performed') is not False:raise ValueError('formal acceptance invalid')
 vp=A/'verification_plan002.json';v=read(vp)
 if sha(vp)!=f['verification_plan_sha256'] or v.get('time_boundary')!='t0 < first selected token <= completion':raise ValueError('formal plan identity/boundary')
 for path,d in v['source_sha256'].items():
  if sha(path)!=d:raise ValueError('formal source changed '+path)
 if f['verifier_sha256']!=sha(A/'verify.py'):raise ValueError('verifier changed')
 run=D/'run_001'; mon=D/'monitor_run001'; snap=D/'run_001_sources/manifest.json'; runtime=D/'runtime_plan001.json';health=mon/'health.jsonl';journal=run/'journal.jsonl'
 files={'plan_sha256':runtime,'run_started_sha256':rs,'run_terminal_sha256':rt,'launch_sha256':run/'launch.json','exit_sha256':run/'exit.json','worker_complete_sha256':run/'worker_complete.json','journal_sha256':journal,'health_sha256':health,'snapshot_manifest_sha256':snap,'snapshot_started_sha256':A/'monitor_snapshot001/started.json','snapshot_terminal_sha256':A/'monitor_snapshot001/terminal.json'}
 for k,p in files.items():
  if f.get(k)!=sha(p):raise ValueError('formal file mismatch '+k)
 term(A/'monitor_snapshot001/started.json',A/'monitor_snapshot001/terminal.json',A/'snapshot.py')
 sm=read(snap);plan=read(runtime);root=snap.parent.resolve()
 if sm.get('status')!='passed' or sm.get('runtime_plan_sha256')!=sha(runtime) or set(sm['sources'])!=set(plan['source_sha256']) or len(sm['sources'])!=185:raise ValueError('snapshot inventory')
 for path,d in plan['source_sha256'].items():
  item=sm['sources'][path];copy=Path(item['snapshot']).resolve()
  if item.get('source')!=path or item.get('sha256')!=d or not copy.is_relative_to(root) or sha(copy)!=d:raise ValueError('snapshot bytes '+path)
 launch,ex,complete=map(read,(run/'launch.json',run/'exit.json',run/'worker_complete.json'))
 if complete.get('status')!='passed' or [complete.get(k) for k in ('jobs','warmups','measured')]!=[48,12,36] or complete.get('source_unchanged') is not True or complete.get('weights_stat_unchanged') is not True or complete.get('journal_sha256')!=sha(journal):raise ValueError('worker completion')
 if ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('client_exited_while_container_running') is not False or ex.get('container_stopped_verified') is not True or ex.get('gpu_pids_after')!=[]:raise ValueError('run exit')
 cid=ex['container_identity']['container_id'];ins=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=10,check=True).stdout)[0];st=ins['State']
 if ins['Id']!=cid or st['Running'] or st['Pid']!=0 or st['ExitCode']!=0 or st['OOMKilled'] or ins['Image']!=ex['last_inspect']['image']:raise ValueError('real container exit')
 # Accepted E4 source performs its own prior formal/independent/input binding; no GT.
 ref=load('_independent_budget_reference',B/'e4_c_budget_pilot_v2/reference.py').load_references();anchors=ref['rows'];seed='eadp-budget-ttft-20261105-001'
 def key(*parts):return hashlib.sha256('|'.join(map(str,(seed,*parts))).encode()).hexdigest()
 expected=[]
 for phase,reps in (('warmup',1),('measure',3)):
  for rep in range(reps):
   for row in sorted(anchors,key=lambda x:key('anchor',phase,rep,x['task'],x['group_id'])):
    for budget in sorted((128,256,512,1024),key=lambda x:key('budget',phase,rep,row['group_id'],x)):
     expected.append(dict(task=row['task'],qid=str(row['qid']),group_id=row['group_id'],phase=phase,replicate=rep,budget=budget,order=len(expected)))
 planned=read(D/'pilot_plan001.json')['jobs']
 if len(expected)!=48 or planned!=expected:raise ValueError('independently reconstructed schedule')
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',local_files_only=True,trust_remote_code=False)
 events=[json.loads(x) for x in journal.read_text().splitlines() if x.strip()];samples=[json.loads(x) for x in health.read_text().splitlines() if x.strip()];windows=[]
 if len(events)!=144 or not samples:raise ValueError('journal or health missing')
 for x in samples:
  w=x.get('health_snapshot_window_monotonic_ns')
  if not isinstance(w,list) or len(w)!=2 or any(type(y) is not int for y in w) or w[0]>w[1] or x.get('time_namespace')!=launch['time_namespace']:raise ValueError('health clock')
  windows.append(w)
 annotations=[]
 for n,job in enumerate(expected):
  triplet=events[n*3:n*3+3]
  if [x.get('event') for x in triplet]!=['started','raw_generation','completed'] or any(x.get('job')!=job for x in triplet):raise ValueError('job event identity')
  raw,res=triplet[1]['payload'],triplet[2]['payload'];accepted=ref['records'][(job['task'],job['qid'],job['budget'])]
  for k in ('task','qid','group_id','budget','output_ids','generation_scalars','eos_ids'):
   if raw.get(k)!=res.get(k):raise ValueError('raw mismatch '+k)
  for k in ('output_ids','answer','generation_scalars','eos_ids','eos','truncated'):
   if res.get(k)!=accepted.get(k):raise ValueError('accepted same-arm mismatch '+k)
  ids=res['output_ids'];b=res['boundaries'];t=[b.get(k) for k in ('t0_before_image_read_ns','first_token_selected_ns','generation_complete_ns')]
  if (not ids or any(type(x) is not int or x<0 for x in t) or not t[0]<t[1]<=t[2] or b.get('first_criteria_length')!=1 or b.get('criteria_calls')!=len(ids) or b.get('logits_ready_ns') is not None or b.get('clock_implementation')!='clock_gettime(CLOCK_MONOTONIC)' or b.get('explicit_sync_boundaries')!=['before_t0','first_token','generation_end'] or b.get('time_namespace')!=launch['time_namespace'] or res.get('natural') is not True or res.get('selector_calls')!=(0 if job['budget']==1024 else 1) or tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)!=res['answer']):raise ValueError('identity or timing boundary')
  def overlaps(interval):return [i for i,w in enumerate(windows) if interval[0]<=w[1] and w[0]<=interval[1]]
  annotations.append({'order':n,'phase':job['phase'],'task':job['task'],'budget':job['budget'],'ttft_health_overlap_indices':overlaps(t[:2]),'full_job_health_overlap_indices':overlaps((t[0],t[2]))})
 if windows!=f['health_windows'] or annotations!=f['annotations']:raise ValueError('health overlap annotations')
 return {'status':'passed','independent_review':True,'jobs':48,'warmups':12,'measured':36,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'errors':[], 'formal_sha256':sha(formal),'formal_started_sha256':sha(fs),'formal_terminal_sha256':sha(ft),'run_started_sha256':sha(rs),'run_terminal_sha256':sha(rt),'reviewer_sha256':sha(__file__),'snapshot_manifest_sha256':sha(snap),'health_sha256':sha(health),'verification_plan_sha256':sha(vp),'health_window_count':len(windows),'limitations':['Three previously examined DEV anchors, each arm measured three times; exploratory and not a population estimate.','Completion boundary includes raw output journal fsync; only first-token timing is intended.','Direct health overlap only; unknown background and residual monitoring overhead cannot be ruled out.','Source snapshot was made after the run; source before/after hashes do not exclude transient changes.']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=OUT:raise ValueError('unplanned output')
 report=run()
 with a.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
