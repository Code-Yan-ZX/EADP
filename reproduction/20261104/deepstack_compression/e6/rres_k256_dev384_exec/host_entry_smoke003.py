"""Exercise real launcher main and supervisor, stop before any GPU query."""
import hashlib,importlib.util,json,os,sys
from pathlib import Path
from unittest.mock import patch
E=Path(__file__).resolve().parent;B=E.parents[1];F=B/'e6/rres_k256_dev384';T=E/'host_smoke_case003'
class StopBeforeGpu(Exception):pass
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only smoke')
 path=E/'launch.py';sp=importlib.util.spec_from_file_location('_rres384_actual_launcher_smoke',path);launch=importlib.util.module_from_spec(sp);sp.loader.exec_module(launch);real_load=launch.load;paths=[];monitor_reads=[]
 source=json.loads((F/'preflight_plan001.json').read_text())['source_sha256'];source[str((E/'gpu_dispatch.py').resolve())]=sha(E/'gpu_dispatch.py')
 class Gate:
  def admit_gpu(self):return {'synthetic':True}
  def read(self,path):
   path=Path(path)
   if path.name=='gpu_plan001.json':return {'source_sha256':source}
   if path.name=='started.json':monitor_reads.append(str(path.resolve()))
   return json.loads(path.read_text())
 def load(name,path,digest):
  if name=='_rres_launch_gate':return Gate()
  module=real_load(name,path,digest);paths.append(str(Path(module.__file__).resolve()))
  if name=='_rres384_gpu_dispatch':
   def stop_dispatch(*args,**kwargs):raise StopBeforeGpu()
   module.dispatch=stop_dispatch
  if name=='_rres_gpu_author_loader':
   original=module.load_author
   def author(src):
    runtime=original(src);paths.append(str(Path(runtime.__file__).resolve()))
    def stop():raise StopBeforeGpu()
    runtime.gpu_pids=stop;return runtime
   module.load_author=author
  return module
 with patch.object(launch,'F',T),patch.object(launch,'load',side_effect=load),patch('subprocess.Popen',side_effect=AssertionError('subprocess prohibited during smoke')):
  try:launch.main()
  except StopBeforeGpu:pass
  else:raise ValueError('did not reach GPU boundary')
 if monitor_reads!=[str((T/'monitor_run001/started.json').resolve())] or (T/'run001').exists():raise ValueError('incorrect host monitor or GPU path')
 value={'status':'passed','scope':'actual launcher main and real supervisor with synthetic admission; stops before GPU query','actual_monitor_path':monitor_reads[0],'module_paths':paths,'GPU_queried':False,'model_loaded':False,'GT_opened':False,'launcher_sha256':sha(path),'script_sha256':sha(__file__)}
 with (E/'host_entry_smoke003.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
