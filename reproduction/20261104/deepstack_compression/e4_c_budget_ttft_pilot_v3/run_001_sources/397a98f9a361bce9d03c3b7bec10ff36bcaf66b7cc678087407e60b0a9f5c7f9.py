"""Supervised, retained Docker no-GPU preflight. No GPU execution path."""
import importlib.util,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=BASE.parents[2]
spec=importlib.util.spec_from_file_location('_budget_host_preflight',HERE/'preflight.py');preflight=importlib.util.module_from_spec(spec);spec.loader.exec_module(preflight)
boot=preflight.load('_budget_launch_bootstrap',BASE/'e6/pixel_natural_timing_pilot_v4/bootstrap.py');boot.pin_all()
from timing_launch import run_bounded,validate_budget_window

def main():
 plan=json.loads((HERE/'preflight_plan001.json').read_text());preflight.check_sources(plan)
 mon=HERE/'monitor_preflight001';started=json.loads((mon/'started.json').read_text())
 if started.get('worker_pid')!=os.getpid() or started.get('timeout_seconds')!=900 or started.get('grace_seconds')!=90:raise ValueError('supervisor mismatch')
 cmd=started.get('command',[]);cwd=Path(started.get('cwd',''))
 if str(HERE/'preflight_launch.py') not in [str((cwd/x).resolve()) if not Path(x).is_absolute() else x for x in cmd]:raise ValueError('supervisor argv mismatch')
 initial=json.loads((BASE/'initial_plan.json').read_text())
 validate_budget_window({'budget_source':str(BASE/'initial_plan.json'),'budget_source_sha256':plan['budget_source_sha256'],
  'budget_start_unix':initial['budget_start_unix'],'budget_end_unix':initial['budget_end_unix']},time.time(),780,900,90)
 author_plan=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
 spec=importlib.util.spec_from_file_location('budget_preflight_runtime',author_plan.parent/'run_panel.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
 out=HERE/'run_preflight001';out.mkdir(exist_ok=False);name='eadp-budget-preflight-'+str(os.getpid())
 argv=runtime.container_command(json.loads(author_plan.read_text()),[str(HERE/'preflight.py'),'--out',str(HERE/'preflight001.json')],False,name)
 if argv.count('--rm')!=1:raise ValueError('unexpected container retention')
 argv.remove('--rm')
 with (out/'launch.json').open('x') as f:
  json.dump({'stage':'four-budget no-GPU processor and weight preflight','plan_sha256':preflight.load('budget_reference',HERE/'reference.py').sha(HERE/'preflight_plan001.json'),
             'command':argv,'controller':runtime.identity(os.getpid()),'container_name':name},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return run_bounded(argv,name,780,None,runtime.identity,out,require_gpu_idle=False)
if __name__=='__main__':raise SystemExit(main())
