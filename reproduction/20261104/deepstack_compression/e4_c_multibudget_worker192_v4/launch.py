"""Bounded retained Docker launcher; absent GPU proofs fail closed."""
import argparse,hashlib,importlib.util,json,os,shutil,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;ROOT=B.parents[2]
CPU_GATE_SHA='fd15237c70608b78b60e6eac5b35a89cbc8e9a9ec497cf51000ab11cb05b31f1';GPU_GATE_SHA='cd5f5f45cbe341644b96dd1f4fb284ac111387da8f68b01bd219214b6fc38f59'
def load(name,path,digest):
 path=Path(path).resolve()
 if hashlib.sha256(path.read_bytes()).hexdigest()!=digest or name in sys.modules:raise ValueError('launcher dependency changed/preloaded before import')
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
 try:spec.loader.exec_module(module)
 except BaseException:del sys.modules[name];raise
 return module
def started(path,now=time.monotonic,sleep=time.sleep):
 end=now()+30
 while now()<end:
  try:return json.loads(Path(path).read_text())
  except (FileNotFoundError,json.JSONDecodeError):sleep(.1)
 raise ValueError('supervisor started record missing')
def check_supervisor(s):
 c=s['command'];cwd=Path(s['cwd'])
 if s.get('worker_pid')!=os.getpid() or s.get('process_group')!=os.getpgrp() or os.getpid()!=os.getpgrp() or s.get('timeout_seconds')!=900 or s.get('stale_seconds') is not None or s.get('grace_seconds')!=90 or s.get('snapshot_seconds')!=3600 or s.get('poll_seconds')!=5 or [(cwd/x).resolve() for x in c if Path(x).name=='launch.py']!=[(D/'launch.py').resolve()] or c.count('--preflight-only')!=1 or c.count('--run-id')!=1 or c[c.index('--run-id')+1]!='001':raise ValueError('CPU launcher supervision mismatch')
def gpu_main(runid):
 if runid!='001':raise ValueError('fixed GPU run ID')
 admission=load('_timing192_launcher_gpu_gate',D/'gpu_gate.py',GPU_GATE_SHA);approval=admission.admit(host=True);plan=admission.plan()
 mon=D/'monitor_run001';st=started(mon/'started.json');cmd=st['command'];cwd=Path(st['cwd'])
 parent=[x.decode(errors='replace') for x in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().split(b'\0') if x]
 if st.get('worker_pid')!=os.getpid() or st.get('process_group')!=os.getpgrp() or os.getpid()!=os.getpgrp() or st.get('timeout_seconds')!=plan['outer_timeout_seconds'] or st.get('stale_seconds')!=plan['stale_seconds'] or st.get('progress_paths')!=[str(D/'run_001/journal.jsonl')] or st.get('grace_seconds')!=90 or st.get('snapshot_seconds')!=3600 or st.get('poll_seconds')!=5 or [(cwd/x).resolve() for x in cmd if Path(x).name=='launch.py']!=[(D/'launch.py').resolve()] or cmd.count('--run-id')!=1 or cmd[cmd.index('--run-id')+1]!='001' or '--preflight-only' in cmd or not any(Path(x).resolve()==(B/'e4_c_latency/timing_supervise.py').resolve() for x in parent):raise ValueError('GPU supervisor identity mismatch')
 if shutil.disk_usage(D).free<plan['minimum_free_bytes']:raise ValueError('insufficient free disk')
 boot=load('_timing192_gpu_launcher_boot',B/'e6/pixel_natural_timing_pilot_v4/bootstrap.py','ab81e2f48f995403afc21ab6ab4875c8b4ae4e9bcca72be063ab8fc42a53fdbc');boot.pin_all()
 from timing_launch import run_bounded,validate_budget_window
 initial=admission.read(B/'initial_plan.json')
 if admission.sha(B/'initial_plan.json')!=plan['budget_source_sha256']:raise ValueError('budget source changed')
 inner=plan['inner_timeout_seconds'];outer=plan['outer_timeout_seconds']
 validate_budget_window({'budget_source':str(B/'initial_plan.json'),'budget_source_sha256':plan['budget_source_sha256'],'budget_start_unix':initial['budget_start_unix'],'budget_end_unix':initial['budget_end_unix']},time.time(),inner,outer,90)
 if time.time()+outer+plan['acceptance_reserve_seconds']>initial['budget_end_unix']:raise ValueError('deadline lacks acceptance reserve')
 author=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';runtime=load('_timing192_gpu_launcher_runtime',author.parent/'run_panel.py','aeee2a27c4394fc683245dce6fe2629f2fa1d9948ab66bb68edbdd46a9c65645')
 if runtime.gpu_pids()!=[]:raise ValueError('GPU occupied')
 out=D/'run_001';out.mkdir(exist_ok=False);name='eadp-timing192-gpu-001-'+str(os.getpid());worker=D/'worker.py';argv=runtime.container_command(admission.read(author),[str(worker),'--out',str(out)],True,name)
 if argv.count('--rm')!=1:raise ValueError('unexpected container retention')
 argv.remove('--rm')
 with (out/'launch.json').open('x') as f:json.dump({'stage':'192 image timing','command':argv,'container_name':name,'controller':runtime.identity(os.getpid()),'run_out':str(out),'approval_bundle':approval,'worker_sha256':admission.sha(worker),'launcher_sha256':admission.sha(__file__),'gpu_plan_sha256':admission.sha(D/'gpu_plan001.json'),'time_namespace':os.readlink('/proc/self/ns/time'),'timeout_seconds':inner},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return run_bounded(argv,name,inner,runtime.gpu_pids,runtime.identity,out,require_gpu_idle=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 if not a.preflight_only:
  return gpu_main(a.run_id)
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
