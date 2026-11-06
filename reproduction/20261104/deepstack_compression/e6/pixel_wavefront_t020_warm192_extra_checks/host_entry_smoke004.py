"""Actual GPU-launcher main import path, stopped before any GPU query.

Fake independent admission and supervisor solely to exercise the production
launcher module's real dependency loaders in exact order. No model/GPU launch.
"""
import hashlib,importlib.util,json,os,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
D=Path(__file__).resolve().parent;B=D.parents[1];W=B/'e6/pixel_wavefront_t020_warm192';out=D/'host_entry004.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class StopBeforeGpu(Exception):pass
def main():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('noGPU only')
 plan=json.loads((W/'preflight_plan001.json').read_text());source=plan['source_sha256'];path=D/'gpu_launch_v3.py';source[str(path.resolve())]=sha(path);source[str((D/'observer_guard_v3.py').resolve())]=sha(D/'observer_guard_v3.py')
 if sha(path)!=source[str(path.resolve())] or out.exists():raise ValueError('source/output')
 spec=importlib.util.spec_from_file_location('_warm192_smoke_actual_launch',path);launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch);original=launch.load;paths=[]
 fake_plan={'outer_timeout_seconds':1,'grace_seconds':1,'budget_end_unix':time.time()+600,'minimum_disk_bytes':0};approval={'plan':fake_plan,'source_sha256':source,'jobs':[],'plan_sha256':'fake'}
 class FakeGate:
  def admit_gpu(self):return approval
  def read(self,path):
   if Path(path).name=='started.json':return {}
   return json.loads(Path(path).read_text())
 def intercepted(name,path,digest):
  if name=='_warm192_gpu_gate':return FakeGate()
  m=original(name,path,digest);paths.append(str(Path(m.__file__).resolve()))
  if name=='_warm192_observer_check':m.live=lambda:{'worker_pid':1}
  if name=='_warm192_observer_guard_v3':
   m.proc_stat=lambda pid:{'parent_pid':2}
   m.verify=lambda *a,**k:{'fake':True}
  if name=='_warm192_author_loader':
   actual=m.load_author
   def author(src):
    runtime=actual(src);paths.append(str(Path(runtime.__file__).resolve()))
    def stop():raise StopBeforeGpu()
    runtime.gpu_pids=stop;return runtime
   m.load_author=author
  return m
 with tempfile.TemporaryDirectory() as td:
  temp=Path(td)
  for name in ('controller.py','observer_check.py'): (temp/name).symlink_to(W/name)
  with patch.object(launch,'D',temp),patch.object(launch,'load',side_effect=intercepted),patch.object(launch,'validate_supervisor',return_value=None),patch.object(sys,'argv',['launch.py','--run-id','001']):
   try:launch.main()
   except StopBeforeGpu:pass
   else:raise ValueError('did not reach pre-GPU boundary')
 expected=[W/'observer_check.py',D/'observer_guard_v3.py',W/'controller.py',B/'e6/pixel_wavefront_t020_timing_runtime_v4/host_runner.py',B/'e6/pixel_wavefront_t020_timing_final_v9/gpu_bounded.py',B/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py',B.parents[2]/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_panel.py']
 if paths!=[str(x.resolve()) for x in expected] or (W/'run_001').exists():raise ValueError('unexpected dependency path or output')
 result={'status':'passed','scope':'actual host launcher main with fake gate/supervisor; stopped immediately before GPU query','source_sha256':{str(p.resolve()):sha(p) for p in [Path(__file__),path,*expected]},'preflight_plan_sha256':sha(W/'preflight_plan001.json'),'module_paths':paths,'GPU_queried':False,'model_loaded':False,'GT_opened':False,'existing_DEV_predictions_opened':False}
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
