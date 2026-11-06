"""Additional same-container, no-device regression of controller/sampler imports."""
import importlib.util,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;ROOT=B.parents[2];AP=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';sys.path[:0]=[str(D),str(B/'e4_c_latency')]
import gate,timing_launch
if Path(gate.__file__).resolve()!=D/'gate.py' or Path(timing_launch.__file__).resolve()!=B/'e4_c_latency/timing_launch.py':raise ValueError('import origin')
def main():
 plan=gate.verify_plan();gate.terminal(D/'monitor_cpu001',D/'cpu_check.py',D/'cpu001.json');gate.terminal(D/'monitor_preflight001',D/'preflight_launch.py')
 s=gate.read(D/'monitor_container_cpu001/started.json');cmd=s['command'];cwd=Path(s['cwd'])
 if s['worker_pid']!=os.getpid() or s['timeout_seconds']!=300 or s['grace_seconds']!=90 or [(cwd/x).resolve() for x in cmd if Path(x).name==Path(__file__).name]!=[Path(__file__).resolve()]:raise ValueError('supervision mismatch')
 initial=gate.read(B/'initial_plan.json');timing_launch.validate_budget_window({'budget_source':str(B/'initial_plan.json'),'budget_source_sha256':plan['budget_source_sha256'],'budget_start_unix':initial['budget_start_unix'],'budget_end_unix':initial['budget_end_unix']},time.time(),180,300,90)
 sp=importlib.util.spec_from_file_location('eadp_memory_cpu_regression_runtime',AP.parent/'run_panel.py');runtime=importlib.util.module_from_spec(sp);sp.loader.exec_module(runtime)
 out=D/'container_cpu001';out.mkdir(exist_ok=False);name='eadp-memory-cpureg-'+str(os.getpid());argv=runtime.container_command(gate.read(AP),[str(D/'test_handshake.py'),'-v'],False,name)
 if argv.count('--rm')!=1:raise ValueError('retention mismatch')
 argv.remove('--rm')
 with (out/'launch.json').open('x') as f:json.dump({'plan_sha256':gate.sha(D/'plan001.json'),'probe_sha256':gate.sha(__file__),'controller':runtime.identity(os.getpid()),'command':argv,'container_name':name},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return timing_launch.run_bounded(argv,name,180,None,runtime.identity,out,require_gpu_idle=False)
if __name__=='__main__':raise SystemExit(main())
