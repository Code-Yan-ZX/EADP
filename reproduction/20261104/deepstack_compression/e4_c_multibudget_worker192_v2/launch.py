"""Bounded retained no-GPU preflight launcher; GPU path remains hard locked."""
import argparse,hashlib,importlib.util,json,os,shutil,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;ROOT=B.parents[2]
CPU_GATE_SHA='8bc5e5072dd99ac06e6c79622b2cacb19dff54afd604f56c7ebe045d170349f4';GPU_GATE_SHA='ecfe5d7ef2c09932fe59e2836f0a8c9c1b34b44268d6329d1ddccea43fa63d3f'
def load(name,path,digest):
 path=Path(path).resolve()
 if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('launcher dependency changed before import')
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def started(path,now=time.monotonic,sleep=time.sleep):
 end=now()+30
 while now()<end:
  try:return json.loads(Path(path).read_text())
  except (FileNotFoundError,json.JSONDecodeError):sleep(.1)
 raise ValueError('supervisor started record missing')
def check_supervisor(s):
 c=s['command'];cwd=Path(s['cwd'])
 if s.get('worker_pid')!=os.getpid() or s.get('process_group')!=os.getpgrp() or os.getpid()!=os.getpgrp() or s.get('timeout_seconds')!=900 or s.get('stale_seconds') is not None or s.get('grace_seconds')!=90 or s.get('snapshot_seconds')!=3600 or s.get('poll_seconds')!=5 or [(cwd/x).resolve() for x in c if Path(x).name=='launch.py']!=[(D/'launch.py').resolve()] or c.count('--preflight-only')!=1 or c.count('--run-id')!=1 or c[c.index('--run-id')+1]!='001':raise ValueError('CPU launcher supervision mismatch')
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 if not a.preflight_only:
  lock=load('_timing192_launcher_gpu_lock',D/'gate.py',GPU_GATE_SHA);lock.authorize_gpu();raise RuntimeError('unreachable without newly reviewed GPU gate')
 if a.run_id!='001':raise ValueError('fixed no-GPU preflight id')
 cg=load('_timing192_launcher_cpu_gate',D/'cpu_gate.py',CPU_GATE_SHA);approval=cg.admit();plan=cg.plan();monitor=D/'monitor_preflight001';check_supervisor(started(monitor/'started.json'))
 if shutil.disk_usage(D).free<plan['minimum_free_bytes']:raise ValueError('insufficient free disk')
 boot=load('_timing192_launcher_boot',B/'e6/pixel_natural_timing_pilot_v4/bootstrap.py','ab81e2f48f995403afc21ab6ab4875c8b4ae4e9bcca72be063ab8fc42a53fdbc');boot.pin_all()
 from timing_launch import run_bounded,validate_budget_window
 initial=cg.read(B/'initial_plan.json')
 if cg.sha(B/'initial_plan.json')!=plan['budget_source_sha256']:raise ValueError('budget source changed')
 validate_budget_window({'budget_source':str(B/'initial_plan.json'),'budget_source_sha256':plan['budget_source_sha256'],'budget_start_unix':initial['budget_start_unix'],'budget_end_unix':initial['budget_end_unix']},time.time(),780,900,90)
 author=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';runtime=load('_timing192_launcher_runtime',author.parent/'run_panel.py','aeee2a27c4394fc683245dce6fe2629f2fa1d9948ab66bb68edbdd46a9c65645')
 out=D/'run_preflight001';out.mkdir(exist_ok=False);name='eadp-timing192-cpu-'+str(os.getpid());worker=D/'worker.py';argv=runtime.container_command(cg.read(author),[str(worker),'--out',str(out),'--preflight-only'],False,name)
 if argv.count('--rm')!=1:raise ValueError('unexpected container retention')
 argv.remove('--rm')
 with (out/'launch.json').open('x') as f:json.dump({'stage':'192 timing worker no-GPU preflight','command':argv,'container_name':name,'controller':runtime.identity(os.getpid()),'run_out':str(out),'approval_bundle':approval,'worker_sha256':cg.sha(worker),'launcher_sha256':cg.sha(__file__),'cpu_plan_sha256':cg.sha(D/'cpu_plan001.json'),'timeout_seconds':780},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return run_bounded(argv,name,780,None,runtime.identity,out,require_gpu_idle=False)
if __name__=='__main__':raise SystemExit(main())
