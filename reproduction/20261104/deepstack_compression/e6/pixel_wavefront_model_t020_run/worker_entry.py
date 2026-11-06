"""Draft PID1 model worker. GPU admission is currently hard locked."""
import argparse,importlib.util,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent

def load(name,path):
 path=Path(path).resolve();old=sys.modules.get(name)
 if old is not None:
  if not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=path:raise ValueError('foreign cached module '+name)
  return old
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
 try:spec.loader.exec_module(module)
 except BaseException:del sys.modules[name];raise
 return module

def check_identity(item,own_cgroup,pid,boot_id,host_argv):
 cid=item.get('container_id');host=item.get('host_init') or {}
 if (pid!=1 or not isinstance(cid,str) or len(cid)!=64 or any(x not in '0123456789abcdef' for x in cid)
     or cid not in own_cgroup or cid not in item.get('host_cgroup','') or type(host.get('pid')) is not int or host['pid']<=0
     or type(host.get('start_ticks')) is not int or host['start_ticks']<=0 or host.get('boot_id')!=boot_id or item.get('host_argv')!=host_argv):
  raise ValueError('container identity mismatch')
 return cid

def await_identity(path,timeout=30):
 deadline=time.monotonic()+timeout
 while True:
  try:
   with Path(path).open() as f:return json.load(f)
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()>=deadline:raise TimeoutError('container identity not published')
   time.sleep(.1)

def main():
 gate=load('_wavefront_t020_worker_gpu_gate',D/'gpu_admission.py');approval=gate.require_gpu()
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--qid',required=True);p.add_argument('--arm',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 plan=json.loads((D/'execution_plan001.json').read_text());out=a.out.resolve();candidates=[j for j in plan['jobs'] if (j['task'],str(j['qid']),j['arm'])==(a.task,a.qid,a.arm)]
 if len(candidates)!=1:raise ValueError('unplanned model arm')
 job=candidates[0];expected=(D/'run_001'/f"arm_{job['order']:03d}"/'observations').resolve()
 if out!=expected:raise ValueError('unplanned output directory')
 launch=json.loads((out.parent/'launch.json').read_text());cmd=launch.get('command',[])
 worker=str(D/'worker_entry.py')
 if launch.get('job')!=job or launch.get('approval')!=approval or cmd.count(worker)!=1 or '--entrypoint' not in cmd:raise ValueError('launch/approval identity')
 host_argv=[cmd[cmd.index('--entrypoint')+1]]+cmd[cmd.index(worker):]
 if sys.argv!=cmd[cmd.index(worker):]:raise ValueError('worker argv mismatch')
 identity=await_identity(out.parent/'container_identity.json');check_identity(identity,Path('/proc/self/cgroup').read_text(),os.getpid(),Path('/proc/sys/kernel/random/boot_id').read_text().strip(),host_argv)
 out.mkdir(exist_ok=False)
 single=load('_wavefront_t020_worker_single_arm',D/'single_arm.py');single._draft_unadmitted_implementation(a.task,a.qid,a.arm,out)
if __name__=='__main__':main()
