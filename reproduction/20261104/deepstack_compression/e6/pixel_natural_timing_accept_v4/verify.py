"""Formal evidence verification candidate for v4; no GT, scores or GPU.

Imported as a pure verifier for CPU tests. The CLI remains locked until a
pre-run verification plan and a complete launch/monitor validation are frozen.
"""
import json,sys,hashlib,struct,os,subprocess,time,importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];PILOT=BASE/'e6/pixel_natural_timing_pilot_v4'
spec=importlib.util.spec_from_file_location('_pixel_accept_contract_v4',PILOT/'pilot_contract.py');pilot_contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot_contract)
spec=importlib.util.spec_from_file_location('_pixel_accept_profile_v4',Path(__file__).resolve().parent/'container_profile.py');profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)

EVENTS=['model_loading','model_ready','started','raw_generation','completed','started','raw_generation','completed','job_complete']

def verify_job(job,rows,ref):
 if [x.get('event') for x in rows]!=EVENTS or any(x.get('job')!=job for x in rows):raise ValueError('job lifecycle/identity')
 setup=rows[1].get('setup_wall_ns')
 if type(setup) is not int or setup<0:raise ValueError('missing model setup boundary')
 outcomes=[]
 for i,phase in ((2,'warmup'),(5,'measured')):
  segment=rows[i:i+3]
  if any(x.get('phase')!=phase for x in segment):raise ValueError('phase mismatch')
  raw=segment[1].get('ids');v=segment[2].get('result',{});memory=segment[2].get('memory',{})
  if not isinstance(raw,list) or len(raw)!=1 or not isinstance(raw[0],list):raise ValueError('raw token shape')
  tokens=raw[0]
  prompt_len=ref['processor']['input_ids']['shape'][1] if job['arm'].startswith('pixel_') else 0
  if len(tokens)<prompt_len or tokens[prompt_len:]!=v.get('tokens') or v.get('tokens')!=ref['tokens']:raise ValueError('raw or accepted output mismatch')
  if job['arm'].startswith('pixel_'):
   meta=ref['processor']['input_ids']
   if meta.get('dtype')!='torch.int64' or sys.byteorder!='little' or hashlib.sha256(struct.pack('<'+'q'*prompt_len,*tokens[:prompt_len])).hexdigest()!=meta.get('sha256'):raise ValueError('raw prompt ID identity mismatch')
  if v.get('recipe')!=pilot_contract.RECIPE or v.get('eos_ids')!=ref['eos_ids'] or v.get('processor_verified') is not True:raise ValueError('recipe/EOS/processor gate')
  times=v.get('boundaries',{});start=times.get('t0_before_image_read_ns');first=times.get('first_token_selected_ns');end=times.get('generation_complete_ns')
  if any(type(x) is not int for x in (start,first,end)) or not start<=first<=end or not times.get('time_namespace') or not times.get('clock_implementation'):raise ValueError('invalid timing boundaries')
  if times.get('explicit_sync_boundaries')!=['before_t0','first_token','generation_end'] or type(times.get('criteria_calls')) is not int or times['criteria_calls']<1:raise ValueError('synchronization or criterion record')
  outcome=pilot_contract.validate_generation(v['tokens'],v['eos_ids'],times.get('first_criteria_length'),prompt_len,job['arm'].startswith('pixel_'))
  if any(v.get(k)!=outcome[k] for k in ('eos','truncated','output_tokens')):raise ValueError('termination metadata mismatch')
  k=v.get('dynamic_k');enabled=v.get('count_observer_enabled')
  if phase=='warmup' and (enabled is not False or k is not None):raise ValueError('warmup observer unexpectedly enabled')
  if phase=='measured' and (enabled is not True or k!=ref['selected_count']):raise ValueError('actual K mismatch')
  before=memory.get('baseline',{});after=memory.get('end',{})
  for field in ('allocated','reserved'):
   b=before.get(field+'_bytes');a=after.get(field+'_bytes');peak=memory.get('max_'+field+'_bytes')
   if any(type(x) is not int or x<0 for x in (a,b,peak)) or peak<max(a,b):raise ValueError('invalid PyTorch allocator accounting')
  if 'not NVML' not in memory.get('scope',''):raise ValueError('memory scope missing')
  outcomes.append({'phase':phase,'task':job['task'],'qid':job['qid'],'arm':job['arm'],'output_tokens':len(v['tokens']),
                   'dynamic_k':k,'t0_ns':start,'first_token_ns':first,'generation_end_ns':end,'time_namespace':times['time_namespace'],'clock_implementation':times['clock_implementation']})
 return outcomes

def annotate_health(records,snapshots,started,terminal):
 if (terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[] or started.get('worker_pid')!=terminal.get('worker_pid')):raise ValueError('pilot monitor failure')
 if len(snapshots)<2 or snapshots[-1].get('event')!='terminal_snapshot':raise ValueError('health journal incomplete')
 windows=[]
 for item in snapshots:
  pair=item.get('health_snapshot_window_monotonic_ns');namespace=item.get('time_namespace')
  if not isinstance(pair,list) or len(pair)!=2 or any(type(x) is not int or x<0 for x in pair) or pair[0]>pair[1] or not isinstance(namespace,str) or not namespace:raise ValueError('invalid health window')
  windows.append(pair)
 if any(b[0]<a[1] for a,b in zip(windows,windows[1:])):raise ValueError('reordered health windows')
 max_gap=int((started['snapshot_seconds']+2*started['poll_seconds']+30)*1e9)
 if any(b[0]-a[1]>max_gap for a,b in zip(windows,windows[1:])):raise ValueError('health sampling gap')
 out=[]
 for r in records:
  if any(x['time_namespace']!=r['time_namespace'] for x in snapshots):raise ValueError('monitor and sample time namespaces differ')
  if r.get('clock_implementation')!='clock_gettime(CLOCK_MONOTONIC)':raise ValueError('incompatible measurement clock')
  start=r['t0_ns'];first=r['first_token_ns'];end=r['generation_end_ns']
  if windows[0][0]>start or windows[-1][1]<end:raise ValueError('health snapshots do not bracket job')
  out.append({**r,'ttft_health_window_indices':[i for i,(a,b) in enumerate(windows) if a<=first and start<=b],
                'generation_health_window_indices':[i for i,(a,b) in enumerate(windows) if a<=end and start<=b]})
 return out

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def accept():
 here=Path(__file__).resolve().parent;vplan=read(here/'verification_plan001.json')
 for path,digest in vplan['sha256'].items():
  if sha(path)!=digest:raise ValueError('verification source/evidence changed: '+path)
 if vplan['pilot_plan_sha256']!=sha(PILOT/'plan001.json'):raise ValueError('pilot plan mismatch')
 # Reuse source-pinned pilot dependency imports only after our frozen plan gate.
 spec=importlib.util.spec_from_file_location('_pixel_accept_bootstrap_v4',PILOT/'bootstrap.py');boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.pin_all()
 pre=boot.pin('preflight',PILOT/'preflight.py');adm=boot.pin('pilot_admission_v4',PILOT/'admission.py');gpu=boot.pin('gpu_launch_v4',PILOT/'gpu_launch.py');gw=boot.pin('gpu_worker_candidate_v4',PILOT/'gpu_worker_candidate.py');refmod=boot.pin('reference_candidate',PILOT/'reference_candidate.py')
 approved=adm.require();plan=pre.read(PILOT/'plan001.json');pilot_contract.validate_schedule(plan['jobs'],plan['anchors']);refs=refmod.load_refs(plan['anchors'])
 env=BASE/'e6/pixel_natural_timing_v4_environment';bg=read(env/'background_summary002.json');bs=read(env/'monitor_summary002/started.json');bt=read(env/'monitor_summary002/terminal.json')
 if bg.get('status')!='completed' or bg.get('all_queries_exit0') is not True or type(bg.get('snapshot_count')) is not int or bg['snapshot_count']<1 or bt.get('reason')!='completed' or bt.get('worker_returncode')!=0 or bt.get('remaining_process_group')!=[] or bs.get('worker_pid')!=bt.get('worker_pid'):raise ValueError('background summary evidence failed')
 bgcmd=bs.get('command',[]);bgcwd=Path(bs.get('cwd','/'));bgresolved=[str((bgcwd/x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()) for x in bgcmd]
 if bgresolved.count(str((env/'summarize002.py').resolve()))!=1:raise ValueError('background summary supervisor command')
 for path,digest in bg.get('source_sha256',{}).items():
  if sha(path)!=digest:raise ValueError('background summary input changed')
 monitor=PILOT/'monitor_pilot001';start=read(monitor/'started.json');terminal=read(monitor/'terminal.json');gpu.validate_supervisor(start,start['worker_pid'],start['process_group'],start['cwd'])
 if terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[] or start['worker_pid']!=terminal.get('worker_pid'):raise ValueError('GPU supervisor did not exit cleanly')
 out=PILOT/'run_pilot001';events=lines(out/'controller.jsonl');expected=['controller_started']+sum((['job_started','job_exited'] for _ in plan['jobs']),[])+['controller_complete']
 if [x.get('event') for x in events]!=expected or events[-1].get('jobs')!=9:raise ValueError('controller event sequence incomplete')
 for i,job in enumerate(plan['jobs']):
  if events[1+2*i].get('job')!=job or events[2+2*i].get('job')!=job:raise ValueError('controller job sequence mismatch')
 root=BASE.parents[2];author=root/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
 spec=importlib.util.spec_from_file_location('_pixel_accept_runtime',author.parent/'run_panel.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime);author_plan=read(author)
 samples=[];evidence=[PILOT/'plan001.json',PILOT/'independent_prelaunch_review.json',monitor/'started.json',monitor/'terminal.json',monitor/'health.jsonl',out/'controller.jsonl'];identities=set()
 for job in plan['jobs']:
  d=out/('job_%03d'%job['order']);lp=d/'launch.json';ep=d/'exit.json';rp=d/'release.json';jp=d/'observations/journal.jsonl';launch=read(lp);ex=read(ep);release=read(rp)
  if launch.get('plan_sha256')!=approved['plan_sha256'] or launch.get('controller',{}).get('pid')!=start['worker_pid'] or ex.get('controller')!=launch.get('controller'):raise ValueError('GPU launcher/controller identity')
  if (ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('gpu_pids_after')!=[] or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('failure_type') is not None or ex.get('client_exited_while_container_running') is not False):raise ValueError('GPU container exit failed')
  identity=ex.get('container_identity') or {};host=identity.get('host_init') or {};pid=host.get('pid');ticks=host.get('start_ticks');unique=(pid,ticks,host.get('boot_id'))
  if type(pid) is not int or type(ticks) is not int or pid<=0 or ticks<=0 or not unique[2] or unique in identities:raise ValueError('worker process identity reused/missing')
  identities.add(unique);live=runtime.identity(pid)
  if live is not None and live.get('start_ticks')==ticks and live.get('boot_id')==unique[2]:raise ValueError('worker process still alive')
  cid=identity.get('container_id');obj=json.loads(subprocess.check_output(['docker','inspect',cid],text=True,timeout=15))[0]
  profile.validate(d,launch,ex,obj,author_plan,runtime.container_command,job,PILOT)
  gw.validate_release(job,launch,release,obj['Config']['Hostname'],sha(lp),approved['plan_sha256'])
  if release.get('container_id')!=cid or release.get('host_init')!=host:raise ValueError('released and observed process identity mismatch')
  if (d/'worker_failure.jsonl').exists():raise ValueError('worker failure artifact present')
  rec=verify_job(job,lines(jp),refs[(job['arm'],job['task'],str(job['qid']))]);samples.extend(rec)
  evidence.extend([lp,ep,rp,jp,d/'container_identity.json',d/'docker.log'])
 if len(samples)!=18 or len(identities)!=9:raise ValueError('sample/worker count')
 annotated=annotate_health(samples,lines(monitor/'health.jsonl'),start,terminal)
 return {'status':'passed','stage':'formal evidence validation; no quality scoring','jobs':9,'generations':18,'measured':9,'GT_opened':False,'scoring_performed':False,'errors':[],
         'plan_sha256':approved['plan_sha256'],'background_summary_sha256':sha(env/'background_summary002.json'),'background_snapshot_count':bg['snapshot_count'],'annotations':annotated,'evidence_sha256':{str(p.resolve()):sha(p) for p in evidence},
         'limits':['three fixed DEV images; observational pilot, not population speed claim',
                   'Pixel K observer and EADP light adapter have nonzero included overhead',
                   'PyTorch allocator figures cover entire generation and are not NVML or whole-device peak',
                   'ps watcher has no recorded time namespace and sampled fsync lies outside its query window; do not assign per-sample overlap',
                   'background summary says per-job monotonic boundaries were unavailable; job boundaries exist but watcher namespace is not bound, so comparison remains unsupported',
                   'ps/fsync and unrelated host Java load have unquantified residual effects; no samples removed',
                   'EADP actual processor identity is checked in worker; full input tensors are not archived for independent rehash']}

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();begin=time.time()
 try:r=accept()
 except Exception as exc:r={'status':'failed','errors':[{'type':type(exc).__name__,'message':str(exc)}],'GT_opened':False,'scoring_performed':False}
 r['started_unix']=begin;r['finished_unix']=time.time()
 with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if r['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
