"""Use actual author command and dispatch entry with fake device/Docker callbacks."""
import hashlib,importlib.util,json,os,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
D=Path(__file__).resolve().parent;B=D.parents[1];E=B/'e6/rres_k256_execution_v2';ROOT=B.parents[2];P=B/'e6/rres_k256_pilot_v2'
sys.path[:0]=[str(E),str(B/'e4_c_latency')]
import launch,gpu_dispatch,timing_launch
source=json.loads((P/'preflight_plan001.json').read_text())['source_sha256'];loaderpath=B/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py';spec=importlib.util.spec_from_file_location('_rres_host_smoke_loader',loaderpath);loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader);runtime=loader.load_author(source);author=json.loads((ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json').read_text())
class FakeRuntime:
 approval={'synthetic':True}
 def __init__(self,pids):self.pids=pids;self.calls=0
 def gpu_pids(self):self.calls+=1;return self.pids
 def identity(self,pid):return runtime.identity(pid)
 def container_command(self,*args):return runtime.container_command(*args)
class FakeTiming:
 def __init__(self):self.calls=0
 def run_bounded(self,command,name,timeout,gpu_pids,identity,out,require_gpu_idle):
  self.calls+=1;assert callable(gpu_pids) and gpu_pids()==[] and timeout==3600 and require_gpu_idle and '--rm' not in command and str(E/'worker.py') in command;return 0
with tempfile.TemporaryDirectory() as td:
 good=FakeRuntime([]);timing=FakeTiming();out=Path(td)/'good';assert launch.dispatch(good,timing,author,out,E/'worker.py')==0 and good.calls==2 and timing.calls==1 and json.loads((out/'launch.json').read_text())['gpu_pids_before']==[]
 busy=FakeRuntime([42]);timing=FakeTiming();out=Path(td)/'busy'
 try:launch.dispatch(busy,timing,author,out,E/'worker.py')
 except ValueError:pass
 else:raise ValueError('busy device was accepted')
 assert not out.exists() and timing.calls==0
base={'timed_out':False,'interrupted':None,'failure_type':None,'client_exited_while_container_running':False,'container_stopped_verified':True,'container_identity':{'container_id':'c','host_init':{'pid':1,'start_ticks':1,'boot_id':'b'}},'container_exit_code':0,'returncode':0}
assert timing_launch.exit_code({**base,'gpu_pids_after':[]})==0 and timing_launch.exit_code({**base,'gpu_pids_after':None})==1 and timing_launch.exit_code({**base,'gpu_pids_after':[42]})==1
with (D/'host_smoke002.json').open('x') as f:json.dump({'status':'passed','actual_launch_module':str(Path(launch.__file__).resolve()),'actual_dispatch_module':str(Path(gpu_dispatch.__file__).resolve()),'actual_author_module':str(Path(runtime.__file__).resolve()),'actual_timing_exit_code_module':str(Path(timing_launch.__file__).resolve()),'actual_author_container_command_constructed':True,'empty_callback_passes':True,'busy_before_creates_nothing':True,'missing_or_busy_after_fails':True,'fake_gpu_callbacks':True,'docker_started':False,'GPU_queried':False,'model_loaded':False,'GT_opened':False},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
