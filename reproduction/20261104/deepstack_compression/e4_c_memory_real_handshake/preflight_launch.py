"""No-GPU container preflight, bounded and retained for inspect."""
import importlib.util,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;ROOT=B.parents[2];AP=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
sys.path[:0]=[str(D),str(B/'e4_c_latency')]
import gate,timing_launch
if Path(gate.__file__).resolve()!=D/'gate.py' or Path(timing_launch.__file__).resolve()!=B/'e4_c_latency/timing_launch.py':raise ValueError('preflight import collision')
def main():
 plan=gate.verify_plan();gate.terminal(D/'monitor_cpu001',D/'cpu_check.py',D/'cpu001.json');cpu=gate.read(D/'cpu001.json')
 if cpu.get('status')!='passed' or cpu.get('plan_sha256')!=gate.sha(D/'plan001.json'):raise ValueError('CPU proof mismatch')
 s=gate.read(D/'monitor_preflight001/started.json');cmd=s['command'];cwd=Path(s['cwd'])
 if s['worker_pid']!=os.getpid() or s['timeout_seconds']!=300 or s['grace_seconds']!=90 or [(cwd/x).resolve() for x in cmd if Path(x).name=='preflight_launch.py']!=[Path(__file__).resolve()]:raise ValueError('supervision mismatch')
 initial=gate.read(B/'initial_plan.json');budget={'budget_source':str(B/'initial_plan.json'),'budget_source_sha256':plan['budget_source_sha256'],'budget_start_unix':initial['budget_start_unix'],'budget_end_unix':initial['budget_end_unix']};timing_launch.validate_budget_window(budget,time.time(),180,300,90)
 spec=importlib.util.spec_from_file_location('eadp_memory_handshake_author_runtime',AP.parent/'run_panel.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
 out=D/'run_preflight001';out.mkdir(exist_ok=False);name='eadp-memory-realpreflight-'+str(os.getpid());args=[str(D/'worker.py'),'--preflight-only','--out',str(D/'preflight001.json')];argv=runtime.container_command(gate.read(AP),args,False,name)
 if argv.count('--rm')!=1:raise ValueError('retention mismatch')
 argv.remove('--rm')
 with (out/'launch.json').open('x') as f:json.dump({'plan_sha256':gate.sha(D/'plan001.json'),'controller':runtime.identity(os.getpid()),'command':argv,'container_name':name},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return timing_launch.run_bounded(argv,name,180,None,runtime.identity,out,require_gpu_idle=False)
if __name__=='__main__':raise SystemExit(main())
