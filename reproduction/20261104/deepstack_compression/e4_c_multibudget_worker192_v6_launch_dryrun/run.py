"""Exercise the exact v6 launcher main at real cwd with safe external fakes.

Real bootstrap imports and real Docker command construction run; proof gate,
GPU enumeration, parent supervisor and Docker execution are simulated. No
model, GPU, Docker, target records or timing values are opened.
"""
import hashlib,importlib.abc,importlib.util,json,os,subprocess,sys,tempfile
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;LAUNCH=B/'e4_c_multibudget_worker192_v6/launch.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def child(temp):
 import unittest.mock as mock
 temp=Path(temp);work=temp;(work/'gpu_gate.py').symlink_to(B/'e4_c_multibudget_worker192_v6/gpu_gate.py');(work/'worker.py').symlink_to(B/'e4_c_multibudget_worker192_v6/worker.py')
 out=work/'monitor_run001';out.mkdir();plan={'outer_timeout_seconds':28800,'inner_timeout_seconds':28680,'stale_seconds':1800,'acceptance_reserve_seconds':3600,'minimum_free_bytes':1,'budget_source_sha256':sha(B/'initial_plan.json')}
 (work/'gpu_plan001.json').write_text(json.dumps(plan))
 cmd=[sys.executable,str(work/'launch.py'),'--run-id','001'];(out/'started.json').write_text(json.dumps({'worker_pid':os.getpid(),'process_group':os.getpgrp(),'command':cmd,'cwd':str(B),'timeout_seconds':28800,'stale_seconds':1800,'progress_paths':[str(work/'run_001/journal.jsonl')],'grace_seconds':90,'snapshot_seconds':3600,'poll_seconds':5}))
 real_spec=importlib.util.spec_from_file_location;real_read=Path.read_bytes;called=[]
 class GateLoader(importlib.abc.Loader):
  def create_module(self,spec):return None
  def exec_module(self,m):
   m.admit=lambda host=False: {'status':'synthetic','host_checked':host}
   m.plan=lambda:plan;m.read=lambda p:json.loads(Path(p).read_text());m.sha=sha
 class BootWrapper(importlib.abc.Loader):
  def __init__(self,actual):self.actual=actual
  def create_module(self,spec):return None
  def exec_module(self,m):
   self.actual.exec_module(m);real=m.pin_all
   def pinned():
    real();import timing_launch
    def bounded(argv,name,timeout,gpu_pids,identity,out,require_gpu_idle):
     if timeout!=28680 or require_gpu_idle is not True or '--rm' in argv or gpu_pids()!=[] or str(work/'worker.py') not in argv:raise AssertionError('actual launcher contract')
     called.append({'command':argv,'name':name,'timeout':timeout,'require_gpu_idle':require_gpu_idle});return 0
    timing_launch.run_bounded=bounded
   m.pin_all=pinned
 class RuntimeWrapper(importlib.abc.Loader):
  def __init__(self,actual):self.actual=actual
  def create_module(self,spec):return None
  def exec_module(self,m):self.actual.exec_module(m);m.gpu_pids=lambda:[]
 def spec(name,location,*a,**kw):
  s=real_spec(name,location,*a,**kw)
  if name=='_timing192_launcher_gpu_gate':s.loader=GateLoader()
  elif name=='_timing192_gpu_launcher_boot':s.loader=BootWrapper(s.loader)
  elif name=='_timing192_gpu_launcher_runtime':s.loader=RuntimeWrapper(s.loader)
  return s
 def readbytes(path):
  if str(path)==f'/proc/{os.getppid()}/cmdline':return (sys.executable+'\0'+str(B/'e4_c_latency/timing_supervise.py')+'\0').encode()
  return real_read(path)
 source=LAUNCH.read_bytes();before=hashlib.sha256(source).hexdigest();(work/'launch.py').write_bytes(source)
 with mock.patch.object(importlib.util,'spec_from_file_location',side_effect=spec),mock.patch.object(Path,'read_bytes',readbytes),mock.patch.object(sys,'argv',cmd[1:]):
  try:exec(compile(source,str(LAUNCH),'exec'),{'__file__':str(work/'launch.py'),'__name__':'__main__'})
  except SystemExit as e:
   if e.code!=0:raise
 if len(called)!=1 or sha(work/'launch.py')!=before:raise AssertionError('launcher not reached/completed')
 launch=json.loads((work/'run_001/launch.json').read_text())
 if launch['worker_sha256']!=sha(B/'e4_c_multibudget_worker192_v6/worker.py') or launch['timeout_seconds']!=28680 or launch['approval_bundle']!={'status':'synthetic','host_checked':True} or launch['command']!=called[0]['command']:raise AssertionError('launch receipt')
 return {'status':'passed','launcher_sha256':before,'cwd':str(Path.cwd()),'gpu_used':False,'docker_executed':False,'model_loaded':False,'GT_opened':False,'target_predictions_opened':False,'bootstrap_real':True,'container_command_real':True,'limit':'approval, supervisor parent, GPU PID query and Docker execution are synthetic; does not replace real no-GPU worker preflight'}
def main():
 if len(sys.argv)==3 and sys.argv[1]=='--child':print(json.dumps(child(sys.argv[2])));return
 if len(sys.argv)!=1:raise ValueError('invalid arguments')
 with tempfile.TemporaryDirectory(dir=B,prefix='.v6_launch_dryrun_') as temp:
  r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child',temp],cwd=B,start_new_session=True,capture_output=True,text=True,timeout=120,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
  if r.returncode:raise ValueError('launch dry-run failed: '+r.stderr[-4000:])
  result=json.loads(r.stdout.strip().splitlines()[-1]);result['dryrun_source_sha256']=sha(__file__)
  with (D/'proof001.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  print(json.dumps(result))
if __name__=='__main__':main()
