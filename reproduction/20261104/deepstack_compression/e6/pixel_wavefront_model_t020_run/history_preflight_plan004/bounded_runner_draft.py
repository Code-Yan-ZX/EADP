"""Draft retained-container runner for one model identity arm, not executable.

Source/approval/budget/parent-process verification remains to be implemented
in a separately frozen admission plan. All public paths call a hard GPU lock.
"""
import importlib.util,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2];P=B/'e6/pixel_wavefront_model_t020';AUTHOR=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
def load(name,path):
 path=Path(path).resolve();cached=sys.modules.get(name)
 if cached is not None:
  if not getattr(cached,'__file__',None) or Path(cached.__file__).resolve()!=path:raise ValueError('foreign module '+name)
  return cached
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m

def run_arm(job,out):
 # Must reject before docker, NVML, any model import, or creating outputs.
 gate=load('_wavefront_t020_bound_gpu_gate',D/'gpu_admission.py');approval=gate.require_gpu()
 upstream=load('_wavefront_t020_bound_upstream',D/'upstream_identity.py');upstream.verify()
 plan=json.loads((D/'execution_plan001.json').read_text())
 if job not in plan['jobs'] or job['arm'] not in ('author','wavefront'):raise ValueError('arm not in immutable schedule')
 timing=load('_wavefront_t020_timing_launch',B/'e4_c_latency/timing_launch.py')
 runtime=load('_wavefront_t020_author_runtime',AUTHOR.parent/'run_panel.py')
 if runtime.gpu_pids():raise ValueError('GPU occupied before arm')
 out=Path(out).resolve();root=(D/'run_001').resolve()
 if out.parent!=root or out.name!=f"arm_{job['order']:03d}":raise ValueError('unplanned arm output')
 out.mkdir(exist_ok=False);name='eadp-wavefront-t020-identity-'+str(job['order'])+'-'+str(os.getpid())
 cmd=draft_command(job,out,name)
 with (out/'launch.json').open('x') as f:
  json.dump({'command':cmd,'name':name,'job':job,'approval':approval},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 code=timing.run_bounded(cmd,name,plan['per_arm_timeout_seconds'],runtime.gpu_pids,runtime.identity,out,require_gpu_idle=True)
 result=json.loads((out/'exit.json').read_text());record=out/'observations'/'record.json'
 status='passed' if code==0 and record.is_file() and json.loads(record.read_text()).get('status')=='completed' else 'failed'
 return {'container_id':(result.get('container_identity') or {}).get('container_id'),'returncode':result.get('returncode'),
         'container_exit_code':result.get('container_exit_code'),'container_stopped_verified':result.get('container_stopped_verified'),
         'timed_out':result.get('timed_out'),'interrupted':result.get('interrupted'),'result_status':status}

def draft_command(job,out,name):
 """Pure command construction helper; does not authorize or spawn a container."""
 if job.get('arm') not in ('author','wavefront') or type(job.get('order')) is not int:raise ValueError('invalid arm')
 runtime=load('_wavefront_t020_author_runtime',AUTHOR.parent/'run_panel.py')
 cmd=runtime.container_command(json.loads(AUTHOR.read_text()),[str(D/'worker_entry.py'),'--task',job['task'],'--qid',str(job['qid']),'--arm',job['arm'],'--out',str(Path(out)/'observations')],True,name)
 if cmd.count('--rm')!=1:raise ValueError('unexpected retained container command')
 cmd.remove('--rm')
 return cmd

if __name__=='__main__':raise SystemExit('GPU locked: real six-arm launcher and independent review not implemented')
