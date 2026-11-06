"""Post-run no-GT validation of 8 containers and all 392 requests."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
E=Path(__file__).resolve().parent;D=E.parent/'pixel_wavefront_t020_warm192';B=D.parents[1];ROOT=B.parents[2]
sys.path.insert(0,str(D))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def load(name,path,source):
 p=Path(path).resolve()
 if name in sys.modules or sha(p)!=source[str(p)]:raise ValueError('formal module source')
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m
 try:sp.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if len(c)<2 or (cwd/c[1]).resolve()!=script.resolve() or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or (out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervisor terminal')
 return s,t
def observe_windows():
 health=[]
 for v in lines(D/'monitor_run001/health.jsonl')+lines(D/'monitor_observer001/health.jsonl'):
  w=v.get('health_snapshot_window_monotonic_ns');ns=v.get('time_namespace')
  if not isinstance(w,list) or len(w)!=2 or not ns:raise ValueError('supervisor health window')
  health.append({'start_ns':w[0],'end_ns':w[1],'time_namespace':ns})
 obs=lines(D/'observations001.jsonl');snap=[x for x in obs if x.get('event')=='snapshot'];done=[x for x in obs if x.get('event')=='logging_complete'];ends=[x for x in obs if x.get('event')=='observer_end']
 if len(ends)!=1 or ends[0].get('reason')!='stop' or len(snap)!=len(done) or ends[0].get('samples')!=len(snap):raise ValueError('observer coverage')
 queries=[]
 for i,(a,b) in enumerate(zip(snap,done)):
  if a.get('index')!=i or b.get('index')!=i or a.get('time_namespace')!=b.get('time_namespace'):raise ValueError('observer sequence')
  for item in (a['ps'],a['docker']):queries.append({'start_ns':item['query_start_ns'],'end_ns':item['query_end_ns'],'time_namespace':a['time_namespace']})
  queries.append({'start_ns':a['docker']['query_end_ns'],'end_ns':b['first_fsync_end_ns'],'time_namespace':a['time_namespace']})
 manual_path=D/'manual_health_windows001.jsonl'
 if not manual_path.is_file():raise ValueError('manual window inventory missing')
 manual=lines(manual_path);return health,queries,manual if manual else None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve()
 if out!=(D/'verification001.json').resolve():raise ValueError('formal output')
 formal=read(D/'formal_plan001.json');gpu=read(D/'gpu_plan001.json');source=gpu['source_sha256']
 if formal.get('gpu_plan_sha256')!=sha(D/'gpu_plan001.json') or formal.get('GT_opened') is not False or any(sha(k)!=v for k,v in source.items()):raise ValueError('frozen source')
 gate=load('_warm192_formal_gate',D/'gate.py',source);approval=gate.admit_gpu();started,ended=terminal(D/'monitor_run001',E/'gpu_launch_v2.py');terminal(D/'monitor_observer001',D/'observer.py');terminal(D/'monitor_snapshot001',D/'snapshot.py',D/'run_001_sources')
 snapshot=read(D/'run_001_sources/manifest.json')
 if snapshot.get('status')!='passed' or snapshot.get('gpu_plan_sha256')!=sha(D/'gpu_plan001.json') or set(snapshot['sources'])!=set(source) or any(v['sha256']!=source[k] or sha(v['copy'])!=v['sha256'] for k,v in snapshot['sources'].items()):raise ValueError('source snapshot')
 barrier=load('_warm192_formal_barrier',B/'e6/pixel_wavefront_t020_timing_runtime_v4/worker.py',source);verify=load('_warm192_formal_contract',D/'verify_contract.py',source);docker=load('_warm192_formal_docker',D/'docker_contract.py',source);window=load('_warm192_formal_window',B/'e6/pixel_wavefront_t020_timing_accept_v5/window_overlap.py',source);author_loader=load('_warm192_formal_author',B/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py',source);runtime=author_loader.load_author(source);author=read(ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json');health,observer,manual=observe_windows()
 cjournal=lines(D/'run_001/controller.jsonl');jobs=approval['jobs']
 if len(cjournal)!=18 or cjournal[0].get('event')!='controller_started' or cjournal[-1].get('event')!='all_completed' or any(cjournal[1+2*i].get('event')!='job_started' or cjournal[2+2*i].get('event')!='job_completed' or cjournal[1+2*i].get('order')!=i or cjournal[2+2*i].get('order')!=i for i in range(8)):raise ValueError('controller inventory')
 final=read(D/'run_001/final.json')
 if final!={'status':'completed','processes':8,'warmups':8,'measured':384,'post_source_and_weight_check':True}:raise ValueError('run final')
 guard=read(D/'observer_guard001.json')
 if guard.get('GPU_queried') is not False or guard.get('required_remaining_seconds')!=gpu['outer_timeout_seconds']+gpu['grace_seconds'] or guard['deadline_lower_monotonic_ns']<guard['checked_monotonic_ns']+guard['required_remaining_seconds']*1000000000:raise ValueError('observer lifetime guard')
 hashes={str((D/'observer_guard001.json').resolve()):sha(D/'observer_guard001.json')};records=[];ids=set()
 for j in jobs:
  d=D/'run_001'/('job_%03d'%j['order']);launch=read(d/'launch.json');exit=read(d/'exit.json');cid=(exit.get('container_identity') or {}).get('container_id')
  if not cid or cid in ids or exit.get('returncode')!=0 or exit.get('container_exit_code')!=0 or exit.get('container_stopped_verified') is not True or exit.get('gpu_pids_after')!=[] or exit.get('timed_out') is not False or exit.get('interrupted') is not None or exit.get('failure_type') is not None:raise ValueError('container exit')
  ids.add(cid);obj=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=15,check=True).stdout)[0];docker.validate(j,launch,exit,obj,author,runtime.container_command,D/'worker.py');release=read(d/'release.json')
  barrier.validate_release(j,launch,release,cid[:12],d/'launch.json',approval['plan_sha256'])
  if obj.get('Config',{}).get('Hostname')!=cid[:12] or release.get('host_init')!=(exit.get('container_identity') or {}).get('host_init'):raise ValueError('release/container identity')
  complete=read(d/'worker_complete.json');model=read(d/'model_load.json');events=lines(d/'journal.jsonl')
  if complete.get('status')!='completed' or complete.get('order')!=j['order'] or complete.get('arm')!=j['arm'] or complete.get('warmup')!=1 or complete.get('measured')!=48 or complete.get('post_source_and_weight_check') is not True or model.get('job')!=j or model.get('container_id')!=cid or model.get('patch_before_model_load') is not True or model.get('model_load_outside_TTFT') is not True or len(events)!=98:raise ValueError('worker completion')
  for i,identity in enumerate([j['warmup']]+j['measured']):
   start,end=events[2*i:2*i+2];phase='warmup' if i==0 else 'measure';raw=d/('raw_%03d.json'%i)
   if start.get('event')!='started' or end.get('event')!='completed' or start.get('index')!=i or end.get('index')!=i or end.get('raw_path')!=str(raw) or end.get('raw_sha256')!=sha(raw) or start.get('monotonic_ns',0)>=end.get('monotonic_ns',0):raise ValueError('per-request journal')
   record=end['record'];bounds=verify.check(record,read(raw),identity,j,phase,i)
   if not model['load_end_monotonic_ns']<start['monotonic_ns']<=bounds['start_ns']<bounds['first_ns']<=bounds['completion_ns']<=end['monotonic_ns']:raise ValueError('load/request ordering')
   annotation={'supervisor_health':window.classify(bounds['start_ns'],bounds['first_ns'],bounds['time_namespace'],health),'observer':window.classify(bounds['start_ns'],bounds['first_ns'],bounds['time_namespace'],observer),'manual':window.classify(bounds['start_ns'],bounds['first_ns'],bounds['time_namespace'],manual)}
   records.append({'process_order':j['order'],'block':j['block'],'phase':phase,'record':record,'monitoring':annotation});hashes[str(raw.resolve())]=sha(raw)
  for name in ('launch.json','release.json','container_identity.json','exit.json','journal.jsonl','worker_complete.json','model_load.json'):p=d/name;hashes[str(p.resolve())]=sha(p)
 if sum(x['phase']=='measure' for x in records)!=384 or sum(x['phase']=='warmup' for x in records)!=8:raise ValueError('full denominator')
 if any(sha(k)!=v for k,v in source.items()):raise ValueError('source drift after verification')
 value={'status':'passed','processes':8,'warmups':8,'measured':384,'records':records,'files_sha256':hashes,'gpu_plan_sha256':sha(D/'gpu_plan001.json'),'formal_plan_sha256':sha(D/'formal_plan001.json'),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'snapshot_manifest_sha256':sha(D/'run_001_sources/manifest.json'),'health_sha256':sha(D/'monitor_run001/health.jsonl'),'observer_sha256':sha(D/'observations001.jsonl'),'manual_sha256':sha(D/'manual_health_windows001.jsonl'),'verifier_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False,'existing_DEV_predictions_opened':True,'errors':[],'limitations':['post-hoc cohort and single-device observations','one-token max differs from previous 2048-token natural generation','GPU KV/selection observation and cache-position scalar synchronization are part of measured TTFT','source and weight endpoint checks cannot exclude transient replacement','manual window coverage empty means unknown, observer second fsync and residual interference unknown','processor and selected-index live association relies on runtime checks; no full-generation or full-KV equality claim']}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
