"""Draft bounded controller, locked until complete independent admission."""
import argparse,hashlib,importlib.util,json,os,shutil,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2];L=B/'e4_c_latency';P=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
def load(name,path):
 path=Path(path).resolve();old=sys.modules.get(name)
 if old is not None:
  if not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=path:raise ValueError('foreign preloaded module '+name)
  return old
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
 try:spec.loader.exec_module(mod)
 except BaseException:del sys.modules[name];raise
 return mod
def main():
 a=argparse.ArgumentParser();a.add_argument('--preflight-only',action='store_true');args=a.parse_args()
 if hashlib.sha256((D/'smoke_gate.py').read_bytes()).hexdigest()!='90d5dbfeb4c9315958233bdc5651e928bae3e1ae21dec8a491fa6338bc4181e7':raise ValueError('model launcher not frozen')
 gate=load('_random_model_host_gate',D/'smoke_gate.py');approval=gate.admit(args.preflight_only);plan=gate.plan()
 out=D/('model_preflight001' if args.preflight_only else 'model_run001');monitor=D/('monitor_model_preflight001' if args.preflight_only else 'monitor_model_run001');started=gate.read(monitor/'started.json');cmd=started['command'];cwd=Path(started['cwd']);outer=2100 if args.preflight_only else 7500;inner=1800 if args.preflight_only else 7200;grace=90
 if (started.get('worker_pid')!=os.getpid() or started.get('process_group')!=os.getpgrp() or os.getpid()!=os.getpgrp() or started.get('timeout_seconds')!=outer or started.get('grace_seconds')!=grace or [(cwd/x).resolve() for x in cmd if Path(x).name==Path(__file__).name]!=[Path(__file__).resolve()] or ('--preflight-only' in cmd)!=args.preflight_only):raise ValueError('supervisor mismatch')
 if args.preflight_only:
  if started.get('stale_seconds') is not None:raise ValueError('CPU preflight stale monitor mismatch')
 elif started.get('stale_seconds')!=1200 or started.get('progress_paths')!=[str(out/'progress.jsonl')]:raise ValueError('GPU heartbeat supervision mismatch')
 if shutil.disk_usage(D).free<32*1024**3:raise ValueError('insufficient disk')
 sys.path[:0]=[str(L)]
 for name in ('identity_gate','real_worker','timing_launch'):
  old=sys.modules.get(name)
  if old is not None and (not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=(L/(name+'.py')).resolve()):raise ValueError('foreign module '+name)
 from timing_launch import run_bounded,validate_budget_window
 initial=gate.read(B/'initial_plan.json');budget={'budget_source':str(B/'initial_plan.json'),'budget_source_sha256':plan['budget_source_sha256'],'budget_start_unix':initial['budget_start_unix'],'budget_end_unix':initial['budget_end_unix']};validate_budget_window(budget,time.time(),inner,outer,grace)
 runtime=load('_random_model_container_runtime',P.parent/'run_panel.py')
 if not args.preflight_only and runtime.gpu_pids()!=[]:raise ValueError('GPU not idle')
 out.mkdir(exist_ok=False);name='eadp-random-model-'+('cpu-' if args.preflight_only else 'gpu-')+str(os.getpid());worker=D/'model_worker.py';command=[str(worker),'--out',str(out)]+(['--preflight-only'] if args.preflight_only else []);argv=runtime.container_command(gate.read(P),command,not args.preflight_only,name)
 if argv.count('--rm')!=1:raise ValueError('unexpected docker retention')
 argv.remove('--rm')
 if args.preflight_only and ('--gpus' in argv or '--device' in argv):raise ValueError('GPU exposed in CPU preflight')
 with (out/'launch.json').open('x') as f:json.dump({'approval_bundle':approval,'controller':runtime.identity(os.getpid()),'command':argv,'container_name':name,'run_out':str(out),'worker_sha256':gate.sha(worker),'launcher_sha256':gate.sha(__file__),'timeout_seconds':inner},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return run_bounded(argv,name,inner,None if args.preflight_only else runtime.gpu_pids,runtime.identity,out,require_gpu_idle=not args.preflight_only)
if __name__=='__main__':raise SystemExit(main())
