"""Bounded, reviewer-gated same-image no-GPU import/weights preflight only."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2];P=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';L=B/'e4_c_latency'
if hashlib.sha256((D/'runtime_gate.py').read_bytes()).hexdigest()!='083e0a69b4296c7f02c82d31c96d567d2f4026fabd2be7a40c644c4ff43d31b4':raise ValueError('runtime gate bytes changed')
sys.path[:0]=[str(D),str(L)]
if 'runtime_gate' in sys.modules and Path(getattr(sys.modules['runtime_gate'],'__file__','')).resolve()!=D/'runtime_gate.py':raise ValueError('foreign runtime gate')
import runtime_gate as gate
if Path(gate.__file__).resolve()!=D/'runtime_gate.py':raise ValueError('wrong runtime gate')
def check_existing(names):
 for n,p in names.items():
  m=sys.modules.get(n)
  if m is not None and (not getattr(m,'__file__',None) or Path(m.__file__).resolve()!=p.resolve()):raise ValueError('foreign cached module '+n)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve()
 if out!=(D/'run_preflight001').resolve():raise ValueError('wrong output')
 approved=gate.require_review();s=gate.read(D/'monitor_preflight001/started.json');cmd=s['command'];cwd=Path(s['cwd'])
 if s['worker_pid']!=os.getpid() or s['timeout_seconds']!=1500 or s['grace_seconds']!=90 or [(cwd/x).resolve() for x in cmd if Path(x).name=='preflight_launch.py']!=[Path(__file__).resolve()] or cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out:raise ValueError('supervisor mismatch')
 names={n:L/(n+'.py') for n in ('identity_gate','real_worker','timing_launch')};check_existing(names)
 from timing_launch import run_bounded,validate_budget_window
 check_existing(names)
 initial=gate.read(B/'initial_plan.json');budget={'budget_source':str(B/'initial_plan.json'),'budget_source_sha256':gate.verify_plan()['budget_source_sha256'],'budget_start_unix':initial['budget_start_unix'],'budget_end_unix':initial['budget_end_unix']};validate_budget_window(budget,time.time(),1200,1500,90)
 check_existing({'diagnose':P.parent/'diagnose.py'});spec=importlib.util.spec_from_file_location('eadp_random_preflight_runtime',P.parent/'run_panel.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime);check_existing({'diagnose':P.parent/'diagnose.py'})
 out.mkdir(exist_ok=False);name='eadp-random-preflight-'+str(os.getpid());argv=runtime.container_command(gate.read(P),[str(D/'preflight_worker.py'),'--out',str(out/'worker_preflight.json')],False,name)
 if argv.count('--rm')!=1 or '--gpus' in argv or '--device' in argv:raise ValueError('unsafe preflight command')
 argv.remove('--rm')
 with (out/'launch.json').open('x') as f:json.dump({'approval':approved,'controller':runtime.identity(os.getpid()),'command':argv,'name':name},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return run_bounded(argv,name,1200,None,runtime.identity,out,require_gpu_idle=False)
if __name__=='__main__':raise SystemExit(main())
