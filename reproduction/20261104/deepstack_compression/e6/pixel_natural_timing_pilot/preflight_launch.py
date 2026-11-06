"""Bounded no-GPU preflight only; never launches a GPU worker."""
import importlib.util,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[2]
sys.path[:0]=[str(HERE),str(BASE/'e4_c_latency')]
import preflight
from timing_launch import run_bounded,validate_budget_window

def main():
 plan=preflight.read(HERE/'plan001.json');preflight.verify_source_plan(plan)
 mon=HERE/'monitor_preflight001';started=preflight.read(mon/'started.json')
 if (started.get('worker_pid')!=os.getpid() or started.get('timeout_seconds')!=900 or started.get('grace_seconds')!=90):raise ValueError('external supervisor mismatch')
 cmd=started.get('command',[]);cwd=Path(started.get('cwd',''))
 resolved=[str((cwd/x).resolve()) if not Path(x).is_absolute() else x for x in cmd]
 if str(HERE/'preflight_launch.py') not in resolved:raise ValueError('external supervisor command mismatch')
 initial=preflight.read(BASE/'initial_plan.json')
 budget={'budget_source':str(BASE/'initial_plan.json'),'budget_source_sha256':plan['budget_source_sha256'],
         'budget_start_unix':initial['budget_start_unix'],'budget_end_unix':initial['budget_end_unix']}
 validate_budget_window(budget,time.time(),780,900,90)
 author_plan=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
 spec=importlib.util.spec_from_file_location('pixel_preflight_runtime',author_plan.parent/'run_panel.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
 out=HERE/'run_preflight001';out.mkdir(exist_ok=False);name='eadp-pixel-naturaltime-preflight-'+str(os.getpid())
 argv=runtime.container_command(preflight.read(author_plan),[str(HERE/'preflight.py'),'--out',str(HERE/'preflight001.json')],False,name)
 if argv.count('--rm')!=1:raise ValueError('unexpected container retention')
 argv.remove('--rm')
 with (out/'launch.json').open('x') as f:
  json.dump({'stage':'no GPU processor/weight preflight','plan_sha256':preflight.sha(HERE/'plan001.json'),'command':argv,
             'controller':runtime.identity(os.getpid()),'container_name':name},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return run_bounded(argv,name,780,None,runtime.identity,out,require_gpu_idle=False)
if __name__=='__main__':raise SystemExit(main())
