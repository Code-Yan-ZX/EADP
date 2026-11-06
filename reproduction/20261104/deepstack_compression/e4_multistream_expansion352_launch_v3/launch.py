"""GPU-only host launcher: exact previously-preflighted v2 worker."""
import argparse,hashlib,importlib.util,json,os,shutil,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;V=B/'e4_multistream_expansion352_gpu_v2'
GATE_SHA='d01ebd2d7a1a7f58350a254d5043b38e0709f51bd00969bccdcd240bef8c20d8'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-id');ap.add_argument('--no-gpu-dependency-smoke',action='store_true');a=ap.parse_args()
 if a.no_gpu_dependency_smoke:
  if a.run_id is not None or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU smoke args')
  path=D/'gpu_gate.py'
  if hashlib.sha256(path.read_bytes()).hexdigest()!=GATE_SHA:raise ValueError('GPU gate source mismatch')
  spec=importlib.util.spec_from_file_location('_multistream_v3_smoke_gate',path);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g);src=g.read(V/'cpu_check001.json')['source_sha256_after']
  if any(g.sha(k)!=v for k,v in src.items()):raise ValueError('v2 source drift')
  inspection=g.inspect_preflight(src)
  for name in ('identity_gate','real_worker','timing_launch'):
   if name in sys.modules:raise ValueError('preloaded '+name)
  sys.path.insert(0,str(B/'e4_c_latency'));import identity_gate,real_worker,timing_launch
  for module,p in ((identity_gate,B/'e4_c_latency/identity_gate.py'),(real_worker,B/'e4_c_latency/real_worker.py'),(timing_launch,B/'e4_c_latency/timing_launch.py')):
   if Path(module.__file__).resolve()!=p.resolve() or g.sha(p)!=src[str(p.resolve())]:raise ValueError('dependency mismatch')
  print(json.dumps({'status':'passed','GPU_used':False,'model_loaded':False,'actual_preflight_inspection':inspection,'imports':['diagnose','run_panel','identity_gate','real_worker','timing_launch']}));return 0
 if a.run_id!='001':raise ValueError('run id')
 path=D/'gpu_gate.py'
 if hashlib.sha256(path.read_bytes()).hexdigest()!=GATE_SHA:raise ValueError('GPU gate source mismatch')
 s=importlib.util.spec_from_file_location('_multistream_v3_host_gate',path);gate=importlib.util.module_from_spec(s);s.loader.exec_module(gate);guard=gate.admit();approval=guard['v2_approval'];plan=gate.read(V/'execution_plan001.json');budget=gate.read(plan['budget_source']);started=gate.read(D/'monitor_run001/started.json');cmd=started['command'];cwd=Path(started['cwd']);out=V/'run001'
 if (gate.sha(plan['budget_source'])!=plan['budget_source_sha256'] or budget['budget_start_unix']!=plan['budget_start_unix'] or budget['budget_end_unix']!=plan['budget_end_unix'] or cwd.resolve()!=B or started.get('worker_pid')!=os.getpid() or started.get('process_group')!=os.getpgrp() or started.get('timeout_seconds')!=plan['gpu_outer_seconds'] or started.get('grace_seconds')!=plan['grace_seconds'] or started.get('stale_seconds')!=plan['gpu_stale_seconds'] or started.get('progress_paths')!=[str((out/'journal.jsonl').resolve())] or len(cmd)<2 or (cwd/cmd[1]).resolve()!=Path(__file__) or cmd[-2:]!=['--run-id','001'] or time.time()+plan['gpu_outer_seconds']>=budget['budget_end_unix'] or shutil.disk_usage(D).free<plan['minimum_disk_bytes']):raise ValueError('supervisor, budget or disk')
 author=sys.modules.get('_multistream_v3_preflight_author')
 if author is None:raise ValueError('missing inspected author module')
 for name in ('identity_gate','real_worker','timing_launch'):
  if name in sys.modules:raise ValueError('preloaded '+name)
 sys.path.insert(0,str(B/'e4_c_latency'));import identity_gate,real_worker,timing_launch
 for m,p in ((identity_gate,B/'e4_c_latency/identity_gate.py'),(real_worker,B/'e4_c_latency/real_worker.py'),(timing_launch,B/'e4_c_latency/timing_launch.py')):
  if Path(m.__file__).resolve()!=p.resolve() or gate.sha(p)!=approval['source_sha256'][str(p.resolve())]:raise ValueError('launcher dependency')
 if author.gpu_pids()!=[]:raise ValueError('GPU occupied')
 planpath=B.parents[2]/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';author_plan=gate.read(planpath);out.mkdir(exist_ok=False);name='eadp-multistream352v3-gpu-'+str(os.getpid());worker=V/'worker.py';command=author.container_command(author_plan,[str(worker),'--out',str(out)],True,name)
 if command.count('--rm')!=1:raise ValueError('container flags')
 command.remove('--rm');launch={'command':command,'name':name,'out':str(out),'controller':author.identity(os.getpid()),'approval_bundle':approval,'v3_guard':guard,'source_sha256':approval['source_sha256'],'worker_sha256':gate.sha(worker),'launcher_sha256':gate.sha(__file__),'gpu_requested':True,'timeout_seconds':plan['gpu_inner_seconds']}
 with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return timing_launch.run_bounded(command,name,plan['gpu_inner_seconds'],author.gpu_pids,author.identity,out,require_gpu_idle=True)
if __name__=='__main__':raise SystemExit(main())
