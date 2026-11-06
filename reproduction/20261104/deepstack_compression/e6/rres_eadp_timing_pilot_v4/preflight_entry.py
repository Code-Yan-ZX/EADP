"""Same-image import/reference probe only: no model loading, CUDA or GPU path."""
import argparse,inspect,json,os,socket,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PROJECT=ROOT.parents[2]
def read(path):return json.loads(Path(path).read_text())
def identity(out):
 deadline=time.monotonic()+60
 while True:
  try:i=read(out/'container_identity.json');launch=read(out/'launch.json');break
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()>deadline:raise TimeoutError('container identity')
   time.sleep(.2)
 cid=i.get('container_id');command=launch.get('command',[])
 if not isinstance(cid,str) or len(cid)!=64 or socket.gethostname()!=cid[:12] or cid not in Path('/proc/self/cgroup').read_text() or '--gpus' in command or '--device' in command or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('not approved no-device container')
 return cid
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out.resolve()
 if out!=(HERE/'preflight001').resolve() or not out.is_dir():raise ValueError('unplanned output')
 import preflight_gate
 approval=preflight_gate.preflight();cid=identity(out)
 if read(out/'launch.json').get('approval')!=approval:raise ValueError('preflight approval mismatch')
 import reference_inputs
 refs=reference_inputs.load()
 author=PROJECT/'_reproduction/author_eadp/Qwen_vl';sys.path[:0]=[str(author),str(author/'VLMEvalKit'),str(ROOT/'e6/rres_k256_pilot_v2')]
 import torch,vlmeval.vlm.qwen3_vl.model_fixed_res as fixed,resolution
 import qwen_request,process_core,process_worker,generation_capture,paired_runner,schedule
 if qwen_request.runner is not paired_runner or process_core.runner is not paired_runner or Path(paired_runner.__file__).resolve()!=(HERE/'paired_runner.py').resolve():raise ValueError('incorrect timing runner')
 import module_load
 adapter_path=ROOT/'e4_c_budget_pilot_v2/adapter.py';adapter=module_load.load('_paired_timing_adapter',adapter_path)
 from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
 paths={'fixed':Path(inspect.getfile(fixed.Qwen3VLChatFixedRes)).resolve(),'eadp':Path(inspect.getfile(fixed.Qwen3VLChatEADP)).resolve(),'resolution':Path(resolution.__file__).resolve(),'adapter':Path(inspect.getfile(adapter.BudgetLightAdapter)).resolve(),'qwen_model':Path(inspect.getfile(Qwen3VLForConditionalGeneration)).resolve(),'paired_runner':Path(paired_runner.__file__).resolve(),'qwen_request':Path(qwen_request.__file__).resolve(),'process_core':Path(process_core.__file__).resolve()}
 if paths['fixed']!=(author/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve() or paths['eadp']!=paths['fixed'] or paths['resolution']!=(ROOT/'e6/rres_k256_pilot_v2/resolution.py').resolve() or paths['adapter']!=adapter_path.resolve():raise ValueError('actual runtime import path')
 if torch.cuda.is_initialized() or fixed.QWEN3_FIXED_RESOLUTION!=1024:raise ValueError('unexpected CUDA or resolution')
 result={'status':'passed','scope':'imports and accepted references only; no model or generation','container_id':cid,'source_paths':{k:str(v) for k,v in paths.items()},'source_sha256':{k:reference_inputs.sha(v) for k,v in paths.items()},'inputs':len(refs['rows']),'same_arm_references':len(refs['references']),'GT_opened':False,'existing_predictions_opened':True,'model_loaded':False,'cuda_initialized':False,'GPU_authorized':False,'errors':[]}
 with (out/'worker_preflight.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
