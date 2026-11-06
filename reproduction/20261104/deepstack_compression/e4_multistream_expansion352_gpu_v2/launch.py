"""Reviewed bounded preflight / GPU pilot; GPU requires its own final proof."""
import argparse,hashlib,importlib.util,json,os,shutil,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
ADMISSION_SHA='1e6bb7d1ccee1e18340f7d25110509ae2637683097af4ae7c050e013bbd70ab6'
def load(name,path,digest):
 p=Path(path).resolve()
 if hashlib.sha256(p.read_bytes()).hexdigest()!=digest or name in sys.modules:raise ValueError('source or module cache')
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);ap.add_argument('--preflight-only',action='store_true');a=ap.parse_args()
 if a.run_id!='001':raise ValueError('unexpected run id')
 gpu=not a.preflight_only;gate=load('_expansion352v2_host_admission',D/'admission.py',ADMISSION_SHA);approval=gate.admit(a.preflight_only,host=True);plan=gate.read(D/'execution_plan001.json')
 run='run001' if gpu else 'preflight001';out=D/run;monitor=D/('monitor_'+run);started=gate.read(monitor/'started.json');cmd=started['command'];cwd=Path(started['cwd']);budget=gate.read(plan['budget_source']);outer=plan['gpu_outer_seconds' if gpu else 'preflight_outer_seconds'];inner=plan['gpu_inner_seconds' if gpu else 'preflight_inner_seconds']
 if (gate.sha(plan['budget_source'])!=plan['budget_source_sha256'] or budget['budget_start_unix']!=plan['budget_start_unix'] or budget['budget_end_unix']!=plan['budget_end_unix'] or cwd.resolve()!=B or started.get('worker_pid')!=os.getpid() or started.get('process_group')!=os.getpgrp() or started.get('timeout_seconds')!=outer or started.get('grace_seconds')!=plan['grace_seconds'] or len(cmd)<2 or (cwd/cmd[1]).resolve()!=Path(__file__) or cmd.count('--run-id')!=1 or cmd[cmd.index('--run-id')+1]!='001' or ('--preflight-only' in cmd)!=a.preflight_only or (gpu and (started.get('stale_seconds')!=plan['gpu_stale_seconds'] or started.get('progress_paths')!=[str((out/'journal.jsonl').resolve())])) or time.time()+outer>=budget['budget_end_unix'] or shutil.disk_usage(D).free<plan['minimum_disk_bytes']):raise ValueError('supervisor, budget or disk')
 source=approval['source_sha256'];root=B.parents[2];path=root/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02';diagnose=path/'diagnose.py';run_panel=path/'run_panel.py'
 if 'diagnose' in sys.modules:raise ValueError('diagnose cached')
 load('diagnose',diagnose,source[str(diagnose.resolve())]);author=load('_expansion352v2_author_runtime',run_panel,source[str(run_panel.resolve())])
 for name in ('identity_gate','real_worker','timing_launch'):
  if name in sys.modules:raise ValueError('preloaded '+name)
 sys.path.insert(0,str(B/'e4_c_latency'));import identity_gate,real_worker,timing_launch
 for m,p in ((identity_gate,B/'e4_c_latency/identity_gate.py'),(real_worker,B/'e4_c_latency/real_worker.py'),(timing_launch,B/'e4_c_latency/timing_launch.py')):
  if Path(m.__file__).resolve()!=p.resolve() or gate.sha(p)!=source[str(p.resolve())]:raise ValueError('launcher dependency')
 if gpu and author.gpu_pids()!=[]:raise ValueError('GPU occupied')
 author_plan=gate.read(path/'run_plan.json');out.mkdir(exist_ok=False);name='eadp-multistream352v2-'+('gpu' if gpu else 'cpu')+'-'+str(os.getpid());worker=D/'worker.py';command=author.container_command(author_plan,[str(worker),'--out',str(out)]+([] if gpu else ['--preflight-only']),gpu,name)
 if command.count('--rm')!=1 or (not gpu and ('--gpus' in command or '--device' in command)):raise ValueError('container flags')
 command.remove('--rm');launch={'command':command,'name':name,'out':str(out),'controller':author.identity(os.getpid()),'approval_bundle':approval,'source_sha256':source,'worker_sha256':gate.sha(worker),'launcher_sha256':gate.sha(__file__),'gpu_requested':gpu,'timeout_seconds':inner}
 with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return timing_launch.run_bounded(command,name,inner,author.gpu_pids if gpu else None,author.identity,out,require_gpu_idle=gpu)
if __name__=='__main__':raise SystemExit(main())
