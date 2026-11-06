"""Real no-device bridge/import preflight; no model creation or GPU call."""
import argparse,hashlib,importlib.util,json,os,sys,socket,time
from pathlib import Path
D=Path(__file__).resolve().parent
GATE_SHA='8f41a01902b1ad5fa8cf0852e9131407132561cac9673b20edcf06ca06ebab1f'
def check_gate(path):
 p=Path(path).resolve()
 if hashlib.sha256(p.read_bytes()).hexdigest()!=GATE_SHA:raise ValueError('preflight gate source mismatch')
 return p
def load(name,path):
 p=Path(path).resolve();old=sys.modules.get(name)
 if old is not None:
  if not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=p:raise ValueError('foreign module '+name)
  return old
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m

def verify_identity(approval,launch,identity,hostname,self_cgroup,expected_image,out):
 cid=identity.get('container_id');host=identity.get('host_init') or {};argv=identity.get('host_argv',[]);command=launch.get('command',[])
 if (launch.get('approval')!=approval or not isinstance(cid,str) or len(cid)!=64 or any(c not in '0123456789abcdef' for c in cid)
     or hostname!=cid[:12] or cid not in identity.get('host_cgroup','') or cid not in self_cgroup or identity.get('image')!=expected_image
     or type(host.get('pid')) is not int or host['pid']<=0 or type(host.get('start_ticks')) is not int or host['start_ticks']<=0 or not host.get('boot_id')
     or argv.count(str(D/'preflight_entry.py'))!=1 or argv[-2:]!=['--out',str(out)] or command.count(str(D/'preflight_entry.py'))!=1
     or '--gpus' in command or '--device' in command or '--rm' in command):raise ValueError('not the approved noGPU Docker worker')
 return cid

def wait_identity(path,timeout=60):
 start=time.monotonic()
 while True:
  try:
   with Path(path).open() as f:return json.load(f)
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()-start>=timeout:raise TimeoutError('container identity absent or incomplete')
   time.sleep(.2)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if out!=(D/'run_preflight001/worker_preflight.json').resolve() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('unplanned output or device')
 gate=load('_pixel_t020_final_preflight_gate',check_gate(D/'preflight_gate.py'));approval=gate.preflight()
 launch=json.loads((D/'run_preflight001/launch.json').read_text());identity=wait_identity(D/'run_preflight001/container_identity.json')
 prior=json.loads((D.parent/'pixel_wavefront_t020_timing_run_v2/run_preflight001/exit.json').read_text())
 cid=verify_identity(approval,launch,identity,socket.gethostname(),Path('/proc/self/cgroup').read_text(),prior['container_identity']['image'],out)
 launch_mod=load('_pixel_t020_final_gpu_launch',D/'launch.py');worker_mod=load('_pixel_t020_final_gpu_worker',D/'worker.py')
 gpu_gate=load('_pixel_t020_final_gpu_gate',D/'gpu_gate.py')
 try:gpu_gate.admit()
 except ValueError:pass
 else:raise ValueError('GPU gate unexpectedly admitted')
 bridge=load('_pixel_t020_runtime_bridge',D.parent/'pixel_wavefront_t020_timing_runtime_v4/bridge.py');loaded=bridge.prepare_cpu();prepared=loaded['model'].prepare_cpu()
 if len(prepared['references'])!=6 or prepared['torch'].cuda.is_initialized():raise ValueError('preflight initialized CUDA or missed reference')
 try:gate.gpu()
 except RuntimeError:pass
 else:raise ValueError('GPU not locked')
 value={'status':'passed','approval':approval,'references':6,'container_id':cid,'gpu_worker_path':str(Path(worker_mod.__file__).resolve()),'gpu_launcher_path':str(Path(launch_mod.__file__).resolve()),'container_identity_sha256':hashlib.sha256((D/'run_preflight001/container_identity.json').read_bytes()).hexdigest(),'module_paths':{k:str(Path(v.__file__).resolve()) for k,v in loaded.items()},'model_module_paths':prepared['module_paths'],'GPU_authorized':False,'model_loaded':False,'cuda_initialized':False,'GT_opened':False,'existing_DEV_predictions_opened':True}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
