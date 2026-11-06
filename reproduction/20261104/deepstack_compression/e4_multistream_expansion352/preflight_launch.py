"""Bounded same-container CPU preflight; this entry cannot request a GPU."""
import argparse,hashlib,importlib.util,json,os,shutil,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
ADMISSION_SHA='483095eee79d9ad74312e5b70fb3cf6ec3c3f3525e97351c886721927f7e35e3'
def load(name,path,digest):
 p=Path(path).resolve()
 if hashlib.sha256(p.read_bytes()).hexdigest()!=digest or name in sys.modules:raise ValueError('source or module cache')
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def main():
 a=argparse.ArgumentParser();a.add_argument('--run-id',required=True);args=a.parse_args()
 if args.run_id!='001':raise ValueError('unexpected run id')
 gate=load('_expansion352_host_admission',D/'admission.py',ADMISSION_SHA);approval=gate.admit(True)
 started=gate.read(D/'monitor_preflight001/started.json');cmd=started['command'];cwd=Path(started['cwd']);budget=gate.read(B/'initial_plan.json')
 if (cwd.resolve()!=B or started.get('worker_pid')!=os.getpid() or started.get('process_group')!=os.getpgrp() or started.get('timeout_seconds')!=2700 or started.get('grace_seconds')!=90 or len(cmd)<2 or (cwd/cmd[1]).resolve()!=Path(__file__) or cmd[-2:]!=['--run-id','001'] or time.time()+2700>=budget['budget_end_unix'] or shutil.disk_usage(D).free<3*1024**3):raise ValueError('supervisor, budget or disk')
 source=approval['source_sha256'];root=B.parents[2];path=root/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02';diagnose=path/'diagnose.py';run_panel=path/'run_panel.py'
 if 'diagnose' in sys.modules:raise ValueError('diagnose cached')
 load('diagnose',diagnose,source[str(diagnose.resolve())]);author=load('_expansion352_author_runtime',run_panel,source[str(run_panel.resolve())])
 for name in ('identity_gate','real_worker','timing_launch'):
  if name in sys.modules:raise ValueError('preloaded '+name)
 sys.path.insert(0,str(B/'e4_c_latency'));import identity_gate,real_worker,timing_launch
 for m,p in ((identity_gate,B/'e4_c_latency/identity_gate.py'),(real_worker,B/'e4_c_latency/real_worker.py'),(timing_launch,B/'e4_c_latency/timing_launch.py')):
  if Path(m.__file__).resolve()!=p.resolve() or gate.sha(p)!=source[str(p.resolve())]:raise ValueError('launcher dependency')
 plan=gate.read(path/'run_plan.json');out=D/'preflight001';out.mkdir(exist_ok=False);name='eadp-multistream352-cpu-001-'+str(os.getpid());worker=D/'pilot_worker.py';command=author.container_command(plan,[str(worker),'--out',str(out),'--preflight-only'],False,name)
 if command.count('--rm')!=1 or '--gpus' in command or '--device' in command:raise ValueError('CPU container flags')
 command.remove('--rm');launch={'command':command,'name':name,'out':str(out),'controller':author.identity(os.getpid()),'approval_bundle':approval,'source_sha256':source,'worker_sha256':gate.sha(worker),'launcher_sha256':gate.sha(__file__),'gpu_requested':False,'timeout_seconds':2400}
 with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return timing_launch.run_bounded(command,name,2400,None,author.identity,out,require_gpu_idle=False)
if __name__=='__main__':raise SystemExit(main())
