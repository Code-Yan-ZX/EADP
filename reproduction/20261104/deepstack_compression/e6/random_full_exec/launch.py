"""Supervised exclusive-GPU controller; all routes require independent review."""
import argparse,importlib.util,json,os,shutil,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2];LAT=B/'e4_c_latency';P=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
def load(name,path):
 path=Path(path).resolve();old=sys.modules.get(name)
 if old is not None:
  if not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=path:raise ValueError('foreign module '+name)
  return old
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 gate=load('_random_full_host_gate',D/'gate.py');approval=gate.admit(a.preflight_only);plan=gate.plan()
 out=D/('run_preflight001' if a.preflight_only else 'run_001');monitor=D/('monitor_preflight001' if a.preflight_only else 'monitor_run001')
 deadline=time.monotonic()+30
 while True:
  try:started=gate.read(monitor/'started.json');break
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()>=deadline:raise ValueError('supervisor identity unavailable')
   time.sleep(.1)
 cmd=started['command'];cwd=Path(started['cwd']);outer=plan['preflight_outer_seconds'] if a.preflight_only else plan['run_outer_seconds'];inner=plan['preflight_inner_seconds'] if a.preflight_only else plan['run_inner_seconds'];grace=plan['grace_seconds']
 if (started.get('worker_pid')!=os.getpid() or started.get('process_group')!=os.getpgrp() or os.getpid()!=os.getpgrp() or started.get('timeout_seconds')!=outer or started.get('grace_seconds')!=grace or [(cwd/x).resolve() for x in cmd if Path(x).name==Path(__file__).name]!=[Path(__file__).resolve()] or ('--preflight-only' in cmd)!=a.preflight_only):raise ValueError('supervisor mismatch')
 if a.preflight_only:
  if started.get('stale_seconds') is not None:raise ValueError('preflight stale monitor')
 elif started.get('stale_seconds')!=plan['stale_seconds'] or started.get('progress_paths')!=[str(out/'progress.jsonl')]:raise ValueError('GPU heartbeat monitor')
 if shutil.disk_usage(D).free<32*1024**3:raise ValueError('insufficient disk')
 sys.path[:0]=[str(LAT)]
 for name in ('identity_gate','real_worker','timing_launch'):
  old=sys.modules.get(name)
  if old is not None and (not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=(LAT/(name+'.py')).resolve()):raise ValueError('foreign module '+name)
 from timing_launch import run_bounded,validate_budget_window
 initial=gate.read(B/'initial_plan.json');budget={'budget_source':str(B/'initial_plan.json'),'budget_source_sha256':plan['budget_source_sha256'],'budget_start_unix':initial['budget_start_unix'],'budget_end_unix':initial['budget_end_unix']}
 validate_budget_window(budget,time.time(),inner,outer,grace)
 if not a.preflight_only and time.time()+outer+plan['acceptance_reserve_seconds']>initial['budget_end_unix']:raise ValueError('insufficient time reserved for acceptance')
 runtime=load('_random_full_container_runtime',P.parent/'run_panel.py')
 if not a.preflight_only and runtime.gpu_pids()!=[]:raise ValueError('GPU not idle')
 out.mkdir(exist_ok=False);name='eadp-random-full-'+('cpu-' if a.preflight_only else 'gpu-')+str(os.getpid());worker=D/'worker.py'
 command=[str(worker),'--out',str(out)]+(['--preflight-only'] if a.preflight_only else []);argv=runtime.container_command(gate.read(P),command,not a.preflight_only,name)
 if argv.count('--rm')!=1:raise ValueError('unexpected retention');
 argv.remove('--rm')
 if a.preflight_only and ('--gpus' in argv or '--device' in argv):raise ValueError('GPU exposed')
 with (out/'launch.json').open('x') as f:json.dump({'approval_bundle':approval,'controller':runtime.identity(os.getpid()),'command':argv,'container_name':name,'run_out':str(out),'worker_sha256':gate.sha(worker),'launcher_sha256':gate.sha(__file__),'timeout_seconds':inner},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return run_bounded(argv,name,inner,None if a.preflight_only else runtime.gpu_pids,runtime.identity,out,require_gpu_idle=not a.preflight_only)
if __name__=='__main__':raise SystemExit(main())
