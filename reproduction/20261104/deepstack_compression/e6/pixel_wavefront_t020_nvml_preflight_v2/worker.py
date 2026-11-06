"""Actual same-image noGPU subprocess imports for all three runtime entries."""
import argparse,hashlib,importlib.util,json,os,socket,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];E=B/'e6/pixel_wavefront_t020_nvml_execution_candidate_v2';GATE_SHA='f2c5089a3ab215949864d1f2cb188252200f3d3582c17d7c746e56504d448087'
def load(name,path,digest):
 p=Path(path).resolve()
 if name in sys.modules or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('preloaded or changed module')
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m
 try:s.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if out!=(D/'run_preflight001/worker_preflight.json').resolve() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('noGPU only')
 gate=load('_pixel_nvml_cpu_gate',D/'gate.py',GATE_SHA);approval=gate.admit();source=gate.read(D/'plan001.json')['source_sha256'];start=time.monotonic();identity_path=D/'run_preflight001/container_identity.json'
 while True:
  try:identity=gate.read(identity_path);break
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()-start>60:raise TimeoutError('container identity')
   time.sleep(.2)
 launch=gate.read(D/'run_preflight001/launch.json');cid=identity.get('container_id','')
 if len(cid)!=64 or socket.gethostname()!=cid[:12] or cid not in Path('/proc/self/cgroup').read_text() or launch.get('approval')!=approval or '--gpus' in launch['command'] or '--device' in launch['command']:raise ValueError('not approved noGPU container')
 results={}
 for name,file in [('worker','worker.py'),('controller','controller_entry.py'),('sampler','sampler_entry.py')]:
  path=(E/file).resolve()
  if gate.sha(path)!=source[str(path)]:raise ValueError('runtime source changed')
  r=subprocess.run([sys.executable,str(path),'--no-gpu-dependency-smoke'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120,check=True)
  value=json.loads(r.stdout)
  if value.get('status')!='passed' or value.get('model_loaded') is not False:raise ValueError('runtime smoke failed')
  results[name]=value
 prior=B/'e6/pixel_wavefront_t020_timing_final_v9/gpu_gate.py';weightgate=load('_pixel_nvml_weight_gate',prior,source[str(prior.resolve())]);weights=weightgate.weight_preflight()
 if len(weights['expected'])!=4 or results['worker'].get('references')!=9 or results['worker'].get('cuda_initialized') is not False or results['controller'].get('gpu_queried') is not False or results['sampler'].get('gpu_queried') is not False:raise ValueError('unexpected device, references or weights')
 try:gate.gpu()
 except RuntimeError:pass
 else:raise ValueError('GPU gate unlocked')
 value={'status':'passed','approval':approval,'container_id':cid,'actual_entry_smokes':results,'weight_stat_preflight':weights,'cuda_initialized':False,'model_loaded':False,'GPU_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'prediction_scope':'accepted same-arm reference identities only','limits':['host process dependencies tested in noGPU container, no GPU or NVML query','weight endpoint stat cannot exclude transient replacement','full executable admission and processor/worker execution require separate review']}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
