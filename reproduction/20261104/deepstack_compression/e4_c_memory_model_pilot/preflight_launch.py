"""Reviewed bounded no-GPU same-image preflight with retained Docker evidence."""
import argparse,importlib.util,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;ROOT=B.parents[2];AP=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';sys.path[:0]=[str(D),str(B/'e4_c_latency')]
_imports=sys.modules.get('imports')
if _imports is not None and (not getattr(_imports,'__file__',None) or Path(_imports.__file__).resolve()!=D/'imports.py'):raise ValueError('foreign imports module')
from imports import checked_import,check_existing,latency_bindings
check_existing(latency_bindings())
gate=checked_import('gate',D/'gate.py');timing_launch=checked_import('timing_launch',B/'e4_c_latency/timing_launch.py')
check_existing(latency_bindings())
if Path(gate.__file__).resolve()!=D/'gate.py' or Path(timing_launch.__file__).resolve()!=B/'e4_c_latency/timing_launch.py':raise ValueError('preflight module collision')
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if out!=(D/'run_preflight001').resolve():raise ValueError('preflight output path')
 approval=gate.review(preflight_only=True,host=True);s=gate.read(D/'monitor_preflight001/started.json');cmd=s['command'];cwd=Path(s['cwd'])
 if s['worker_pid']!=os.getpid() or s['timeout_seconds']!=1500 or s['grace_seconds']!=90 or [(cwd/x).resolve() for x in cmd if Path(x).name=='preflight_launch.py']!=[Path(__file__).resolve()] or cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out:raise ValueError('preflight supervisor mismatch')
 plan=gate.verify_plan();initial=gate.read(B/'initial_plan.json');budget={'budget_source':str(B/'initial_plan.json'),'budget_source_sha256':plan['budget_source_sha256'],'budget_start_unix':initial['budget_start_unix'],'budget_end_unix':initial['budget_end_unix']};timing_launch.validate_budget_window(budget,time.time(),1200,1500,90)
 check_existing({'diagnose':AP.parent/'diagnose.py'})
 spec=importlib.util.spec_from_file_location('eadp_modelmemory_author_runtime',AP.parent/'run_panel.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
 check_existing({'diagnose':AP.parent/'diagnose.py'})
 out.mkdir(exist_ok=False);name='eadp-modelmemory-preflight-'+str(os.getpid());args=[str(D/'worker.py'),'--preflight-only','--out',str(out)];argv=runtime.container_command(gate.read(AP),args,False,name)
 if argv.count('--rm')!=1 or '--gpus' in argv or '--device' in argv:raise ValueError('preflight devices/retention')
 argv.remove('--rm')
 with (out/'launch.json').open('x') as f:json.dump({'approval':approval,'controller':runtime.identity(os.getpid()),'command':argv,'name':name},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return timing_launch.run_bounded(argv,name,1200,None,runtime.identity,out,require_gpu_idle=False)
if __name__=='__main__':raise SystemExit(main())
