"""Real controller.main under real supervisor with all subprocesses blocked.

Fake runtime and fake bounded callback never spawn Docker, NVML or GPU work.
"""
import importlib.util,json,subprocess,sys,types
from pathlib import Path
import controller
HERE=Path(__file__).resolve().parent;SMOKE=HERE/'host_smoke002';ROOT=HERE.parents[1]
def blocked(*a,**k):raise AssertionError('all subprocess creation is blocked')
subprocess.Popen=blocked
try:subprocess.Popen(['not-executed'])
except AssertionError:pass
else:raise RuntimeError('subprocess blocker inactive')
source=json.loads((ROOT/'e6/rres_eadp_identity3_v2/preflight_plan001.json').read_text())['source_sha256'];loader_path=ROOT/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py'
spec=importlib.util.spec_from_file_location('_identity_smoke_real_loader',loader_path);loader=importlib.util.module_from_spec(spec);sys.modules[spec.name]=loader;spec.loader.exec_module(loader)
runtime=loader.load_author(source);runtime.gpu_pids=lambda:[];runtime.identity=lambda pid:{'pid':pid,'synthetic':True};timing=types.ModuleType('timing_launch')
def bounded(command,name,timeout,pids,identity,out,require_gpu_idle):
 envs=[command[i+1] for i,x in enumerate(command[:-1]) if x=='--env']
 if subprocess.Popen is not blocked or timeout!=1600 or not require_gpu_idle or pids()!=[] or command.count('--device')!=4 or command.count('--mount')!=12 or 'CUDA_VISIBLE_DEVICES=0' not in envs:raise AssertionError('fake boundary')
 (Path(out)/'observations/worker_complete.json').write_text('{"status":"passed"}\n');return 0
timing.run_bounded=bounded;sys.modules['timing_launch']=timing
guard=types.SimpleNamespace(LATENCY='fake',check=lambda *a,**k:None)
class Gate:
 def admit(self):return {'synthetic':True,'GPU_authorized':False}
 def read(self,p):
  p=Path(p)
  if p.name=='gpu_plan001.json':
   return {'source_sha256':source}
  return json.loads(p.read_text())
gate=Gate()
def fake_load(name,path,sha):
 if name=='_identity_gpu_host_gate':return gate
 if name=='_identity_gpu_guard':return guard
 if name=='_identity_gpu_author_loader':return types.SimpleNamespace(load_author=lambda source:runtime)
 raise AssertionError('unexpected dynamic load')
controller.load=fake_load;controller.HERE=SMOKE
if controller.load is not fake_load or subprocess.Popen is not blocked:raise RuntimeError('two independent interceptors required')
result=controller.main()
if result!=0:raise RuntimeError('synthetic controller failed')
print('passed actual controller.main under supervisor; two subprocess/device interceptors active; no Docker or GPU')
