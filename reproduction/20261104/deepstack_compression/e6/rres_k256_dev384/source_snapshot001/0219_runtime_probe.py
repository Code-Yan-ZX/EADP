"""Import actual worker dependencies in the reviewed no-device container."""
import argparse,hashlib,importlib.util,json,os,socket,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];F=B/'e6/rres_k256_dev384';P=B/'e6/rres_k256_pilot_v2';ROOT=B.parents[2];GATE_SHA='40d96e042d5d2006e18dd5cb68a49572a20041e72eee1ede16531c1b5e90895c'
def load(name,path,digest):
 path=Path(path).resolve()
 if name in sys.modules or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('foreign/preloaded module')
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();run=D/'runtime_preflight001';out=a.out.resolve()
 if out!=run/'worker_preflight.json' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('only noGPU')
 gate=load('_rres384_runtime_gate',D/'runtime_admit.py',GATE_SHA);approval=gate.admit();plan=gate.read(D/'runtime_preflight_plan001.json');deadline=time.monotonic()+60
 while True:
  try:identity=gate.read(run/'container_identity.json');launch=gate.read(run/'launch.json');break
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()>deadline:raise TimeoutError('container identity publication')
   time.sleep(.2)
 cid=identity['container_id']
 if len(cid)!=64 or socket.gethostname()!=cid[:12] or cid not in Path('/proc/self/cgroup').read_text() or launch.get('approval')!=approval or '--device' in launch['command'] or '--gpus' in launch['command']:raise ValueError('noGPU container identity')
 source=plan['source_sha256'];worker=load('_rres384_worker_import',D/'worker.py',source[str(D/'worker.py')]);sys.path[:0]=[str(F),str(P),str(B/'e4_c_latency'),str(B/'e4'),str(ROOT/'_reproduction/author_eadp/Qwen_vl'),str(ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit')]
 import torch,arm,capture,resolution,panel_core,decode_observer,input_preflight,vlmeval.vlm.qwen3_vl.model_fixed_res as fixed
 expected={arm:P/'arm.py',capture:P/'capture.py',resolution:P/'resolution.py',panel_core:F/'panel_core.py',decode_observer:B/'e4/decode_observer.py',input_preflight:B/'e4_c_latency/input_preflight.py',fixed:ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py'}
 paths={}
 for module,path in expected.items():
  actual=Path(module.__file__).resolve()
  if actual!=path.resolve() or gate.sha(actual)!=source[str(actual)]:raise ValueError('runtime module identity')
  paths[module.__name__]=str(actual)
 from transformers import AutoConfig
 from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
 if torch.cuda.is_initialized() or fixed.QWEN3_FIXED_RESOLUTION!=1024:raise ValueError('unexpected CUDA or resolution')
 if any(gate.sha(k)!=v for k,v in source.items()):raise ValueError('runtime source changed')
 result={'status':'passed','scope':'same-image noGPU worker imports; no model or generation','worker_sha256':gate.sha(D/'worker.py'),'module_paths':paths,'container_id':cid,'approval':approval,'model_loaded':False,'cuda_initialized':False,'GPU_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'scoring_performed':False,'errors':[]}
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
