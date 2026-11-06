"""Retained bounded CPU preflight / separately reviewed GPU diagnostic."""
import argparse,importlib.util,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2];AP=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';sys.path.insert(0,str(B/'e4_c_latency'))
import timing_launch
if Path(timing_launch.__file__).resolve()!=B/'e4_c_latency/timing_launch.py':raise ValueError('foreign timing launcher')
from worker import check_plan,sha,save
if Path(sys.modules['worker'].__file__).resolve()!=D/'worker.py':raise ValueError('foreign worker')
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=('preflight','smoke'),required=True);a=p.parse_args();plan=check_plan();gpu=a.stage=='smoke'
 if gpu:
  import gate
  if Path(gate.__file__).resolve()!=D/'gate.py':raise ValueError('foreign gate')
  gate.require_final()
 budget0=json.loads((B/'initial_plan.json').read_text());budget={'budget_source':str(B/'initial_plan.json'),'budget_source_sha256':plan['budget_source_sha256'],'budget_start_unix':budget0['budget_start_unix'],'budget_end_unix':budget0['budget_end_unix']};inner,outer,grace=(1800,1920,90) if gpu else (600,720,90);timing_launch.validate_budget_window(budget,time.time(),inner,outer,grace)
 mon=D/('monitor_'+a.stage+'001');s=json.loads((mon/'started.json').read_text());cmd=s['command'];cwd=Path(s['cwd'])
 if s['worker_pid']!=os.getpid() or s['timeout_seconds']!=outer or s['grace_seconds']!=grace or [(cwd/x).resolve() for x in cmd if Path(x).name=='launch.py']!=[Path(__file__).resolve()] or cmd[-2:]!=['--stage',a.stage]:raise ValueError('supervisor mismatch')
 if gpu and (s.get('stale_seconds')!=600 or s.get('progress_paths')!=[str(D/'run_smoke001/observations/progress.jsonl')]):raise ValueError('progress supervision mismatch')
 spec=importlib.util.spec_from_file_location('eadp_wavefront_container_runtime',AP.parent/'run_panel.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
 if Path(runtime.__file__).resolve()!=AP.parent/'run_panel.py':raise ValueError('runtime source')
 if gpu and runtime.gpu_pids():raise ValueError('GPU occupied')
 out=D/('run_'+a.stage+'001');out.mkdir(exist_ok=False);name='eadp-wavefront-'+a.stage+'-'+str(os.getpid());args=[str(D/'worker.py')]+([] if gpu else ['--preflight-only'])+['--out',str(out/'observations' if gpu else D/'preflight001.json')];c=runtime.container_command(json.loads(AP.read_text()),args,gpu,name)
 if c.count('--rm')!=1 or (not gpu and ('--device' in c or '--gpus' in c)):raise ValueError('devices');
 c.remove('--rm');save(out/'launch.json',{'command':c,'name':name,'controller':runtime.identity(os.getpid()),'plan_sha256':sha(D/'plan001.json'),'stage':a.stage});return timing_launch.run_bounded(c,name,inner,runtime.gpu_pids if gpu else None,runtime.identity,out,require_gpu_idle=gpu)
if __name__=='__main__':sys.exit(main())
