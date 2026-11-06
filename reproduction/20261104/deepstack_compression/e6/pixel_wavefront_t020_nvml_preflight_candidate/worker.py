"""Actual same-image noGPU import preflight; intentionally no GPU entry."""
import argparse,hashlib,importlib.util,json,os,socket,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1]
def load(name,path,digest):
 path=Path(path).resolve()
 if name in sys.modules or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('preloaded or changed module')
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve()
 if out!=(D/'run_preflight001/worker_preflight.json').resolve() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('noGPU only')
 gate=load('_pixel_nvml_cpu_gate',D/'gate.py',GATE_SHA);approval=gate.admit();src=gate.read(D/'plan001.json')['source_sha256']
 start=time.monotonic();identity_path=D/'run_preflight001/container_identity.json'
 while True:
  try:identity=gate.read(identity_path);break
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()-start>60:raise TimeoutError('container identity')
   time.sleep(.2)
 launch=gate.read(D/'run_preflight001/launch.json');cid=identity.get('container_id','')
 if len(cid)!=64 or socket.gethostname()!=cid[:12] or cid not in Path('/proc/self/cgroup').read_text() or launch.get('approval')!=approval or '--gpus' in launch['command'] or '--device' in launch['command']:raise ValueError('not approved noGPU container')
 refpath=B/'e6/pixel_wavefront_t020_nvml_reference/reference.py';refsmod=load('_pixel_nvml_9refs',refpath,src[str(refpath.resolve())]);refs,prior=refsmod.load_refs()
 modpath=B/'e6/pixel_wavefront_t020_model_timing_v2/model_candidate.py';model=load('_pixel_nvml_model_dependencies',modpath,src[str(modpath.resolve())]);prepared=model.prepare_cpu()
 candidate=B/'e6/pixel_wavefront_t020_nvml_model_candidate/model_run.py';load('_pixel_nvml_model_run',candidate,src[str(candidate.resolve())])
 torch=prepared['torch']
 if len(refs)!=9 or len(prepared['references'])!=6 or torch.cuda.is_initialized():raise ValueError('reference count or CUDA initialization')
 try:gate.gpu()
 except RuntimeError:pass
 else:raise ValueError('GPU gate unlocked')
 value={'status':'passed','approval':approval,'container_id':cid,'nine_reference_count':len(refs),'six_t020_reference_count':len(prepared['references']),'prior':prior,'module_paths':prepared['module_paths'],'cuda_initialized':False,'model_loaded':False,'GPU_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'prediction_scope':'accepted same-arm reference identity only'}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
GATE_SHA='b9300e39d4936abf337ec59c885664deb810723ee9cf4cd29034fd77ac008dc1'
if __name__=='__main__':main()
