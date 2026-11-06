"""Prospective multi-container launcher; unconditional GPU lock in CLI."""
import hashlib,importlib.util,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2];R=B/'e6/pixel_wavefront_t020_timing_runtime_v3';C=B/'e6/pixel_wavefront_t020_model_timing_v2';V4=B/'e6/pixel_natural_timing_pilot_v4'
def load(name,path):
 p=Path(path).resolve();m=sys.modules.get(name)
 if m is not None:
  if not getattr(m,'__file__',None) or Path(m.__file__).resolve()!=p:raise ValueError('foreign '+name)
  return m
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m

def execute_after_admission(plan_sha,expected_worker_sha,design_sha,expected_jobs,out,budget_end,runtime,supervisor_check,inner_seconds=1200,cleanup_seconds=120):
 """Unsafe: caller must verify independent GPU admission and deadlines first."""
 raw=(C/'design001.json').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=design_sha:raise ValueError('design identity changed')
 design=json.loads(raw);jobs=design['jobs']
 if jobs!=expected_jobs:raise ValueError('approved exact schedule differs')
 controller=load('_pixel_t020_gpu_run_controller',R/'controller.py');host=load('_pixel_t020_gpu_run_host',R/'host_runner.py');worker=load('_pixel_t020_gpu_run_worker',R/'worker.py');check=load('_pixel_t020_gpu_run_job_check',B/'e6/pixel_wavefront_t020_timing_controller_draft/job_contract.py');reference=load('_pixel_t020_gpu_run_reference',C/'reference.py');bounded=load('_pixel_t020_gpu_run_bounded',V4/'gpu_bounded.py')
 refs,_=reference.load();author=json.loads((ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json').read_text())
 def run_job(job,jobdir):
  name='eadp-pixel-t020-timing-%03d-%d'%(job['order'],os.getpid())
  command=runtime.container_command(author,[str(D/'gpu_worker.py'),'--job-index',str(job['order'])],True,name)
  if command.count('--rm')!=1:raise ValueError('unexpected Docker command')
  command.remove('--rm')
  return host.bounded_one(job,jobdir,command,name,runtime.identity(os.getpid()),plan_sha,D/'gpu_worker.py',expected_worker_sha,bounded.run_bounded,runtime.gpu_pids,runtime.identity,supervisor_check,inner_seconds)
 def verify_job(job,jobdir,exit):
  record=json.loads((jobdir/'record.json').read_text());ref=refs[(job['task'],str(job['qid']),job['arm'])]
  if record.get('job')!=job or check.check(job,record['record'],ref,jobdir/'raw.json',jobdir)!=record['metrics']:raise ValueError('worker record mismatch')
 return controller.run_all(jobs,out,run_job,verify_job,supervisor_check,budget_end,inner_seconds,cleanup_seconds,time.time)

def main():raise RuntimeError('GPU locked: no independently verified complete launch plan')
if __name__=='__main__':main()
