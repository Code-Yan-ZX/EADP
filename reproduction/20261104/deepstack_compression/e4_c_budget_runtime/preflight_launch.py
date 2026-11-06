"""Bounded retained Docker for model-before-load preflight; never uses GPU."""
import importlib.util,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=BASE.parents[2]
spec=importlib.util.spec_from_file_location('_budget_runtime_launcher_gate',HERE/'gate.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
def main():
 plan=gate.validate_runtime_sources();prior=gate.validate_prior()
 mon=HERE/'monitor_preflight001';start=json.loads((mon/'started.json').read_text())
 if start.get('worker_pid')!=os.getpid() or start.get('timeout_seconds')!=900 or start.get('grace_seconds')!=90:raise ValueError('supervisor mismatch')
 cwd=Path(start.get('cwd',''));command=start.get('command',[])
 if str(HERE/'preflight_launch.py') not in [str((cwd/x).resolve()) if not Path(x).is_absolute() else x for x in command]:raise ValueError('supervisor command mismatch')
 bootpath=BASE/'e6/pixel_natural_timing_pilot_v4/bootstrap.py';spec=importlib.util.spec_from_file_location('_budget_runtime_boot',bootpath);boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.pin_all()
 from timing_launch import run_bounded,validate_budget_window
 initial=json.loads((BASE/'initial_plan.json').read_text());validate_budget_window({'budget_source':str(BASE/'initial_plan.json'),'budget_source_sha256':plan['budget_source_sha256'],'budget_start_unix':initial['budget_start_unix'],'budget_end_unix':initial['budget_end_unix']},time.time(),780,900,90)
 author=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
 spec=importlib.util.spec_from_file_location('_budget_runtime_docker',author.parent/'run_panel.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
 out=HERE/'run_preflight001';out.mkdir(exist_ok=False);name='eadp-budget-worker-preflight-'+str(os.getpid())
 argv=runtime.container_command(json.loads(author.read_text()),[str(HERE/'worker.py'),'--preflight-only','--out',str(out)],False,name)
 if argv.count('--rm')!=1:raise ValueError('unexpected retention');
 argv.remove('--rm')
 with (out/'launch.json').open('x') as f:
  json.dump({'stage':'model-before-load CPU worker preflight','command':argv,'controller':runtime.identity(os.getpid()),'container_name':name,
             'runtime_plan_sha256':gate.sha(HERE/'runtime_plan001.json'),'prior_proof_sha256':prior['prior_proof_sha256']},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return run_bounded(argv,name,780,None,runtime.identity,out,require_gpu_idle=False)
if __name__=='__main__':raise SystemExit(main())
