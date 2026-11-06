"""Run execution-module import and GPU-lock checks in retained no-device container."""
import argparse,importlib.util,inspect,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D))
cached=sys.modules.get('preflight_admission')
if cached is not None and (not getattr(cached,'__file__',None) or Path(cached.__file__).resolve()!=D/'preflight_admission.py'):raise ValueError('foreign cached admission')
import preflight_admission
if Path(preflight_admission.__file__).resolve()!=D/'preflight_admission.py':raise ValueError('foreign admission')
def load_exact(name,path):
 cached=sys.modules.get(name)
 if cached is not None:
  if not getattr(cached,'__file__',None) or Path(cached.__file__).resolve()!=path.resolve():raise ValueError('foreign cached module '+name)
  return cached
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if out!=(D/'run_preflight001/worker_preflight.json').resolve() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('invalid no-device preflight')
 approval=preflight_admission.require_preflight()
 import torch
 if torch.cuda.is_initialized():raise ValueError('CUDA preinitialized')
 names=('gpu_admission','worker_entry','single_arm','controller','bounded_runner_draft','schedule_contract','upstream_identity','execution_upstream','snapshot_sources','weight_guard','supervision_contract','imports_guard','upstream_admission')
 modules={name:load_exact(name,D/(name+'.py')) for name in names}
 prior=modules['execution_upstream'].verify()
 # Reproduce the worker import order, including the actual accepted upstream
 # gate and its own identically named source_inventory dependency.
 accepted=modules['upstream_admission'].load_prior();accepted.require_preflight()
 L=D.parents[1]/'e4_c_latency';sys.path.insert(0,str(L));guard=modules['imports_guard'];guard.check(guard.LATENCY)
 import timing_launch
 guard.check(guard.LATENCY,require=True)
 B=D.parents[1];ROOT=B.parents[2];pixel=B/'e6/external_sources/pixelprune_8d347eb';author=ROOT/'_reproduction/author_eadp/Qwen_vl';sys.path[:0]=[str(pixel),str(author),str(author/'VLMEvalKit')]
 expected={'pixelprune':pixel/'pixelprune/__init__.py','pixelprune.methods.pred_2d':pixel/'pixelprune/methods/pred_2d.py','pixelprune.patches.qwen3_vl_hf':pixel/'pixelprune/patches/qwen3_vl_hf.py','vlmeval.vlm.qwen3_vl.model_fixed_res':author/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py','transformers.models.qwen3_vl.modeling_qwen3_vl':ROOT/'_reproduction/envs/qwen_author/lib/python3.10/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py'}
 guard.check(expected)
 import pixelprune
 from pixelprune.methods.pred_2d import Pred2DSelector
 from pixelprune.patches import qwen3_vl_hf as patch
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
 from transformers import Qwen3VLForConditionalGeneration
 guard.check(expected,require=True)
 if (Path(inspect.getfile(Pred2DSelector)).resolve()!=expected['pixelprune.methods.pred_2d'] or Path(inspect.getfile(Qwen3VLChatFixedRes)).resolve()!=expected['vlmeval.vlm.qwen3_vl.model_fixed_res'] or Path(inspect.getfile(Qwen3VLForConditionalGeneration)).resolve()!=expected['transformers.models.qwen3_vl.modeling_qwen3_vl']):raise ValueError('model class source mismatch')
 try:modules['gpu_admission'].require_gpu()
 except RuntimeError as e:
  if 'GPU locked' not in str(e):raise
 else:raise ValueError('GPU gate not locked')
 if torch.cuda.is_initialized():raise ValueError('CUDA initialized')
 result={'status':'passed','approval':approval,'gpu_authorized':False,'cuda_initialized':False,'model_loaded':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'module_paths':{**{x:str(D/(x+'.py')) for x in names},**{x:str(p) for x,p in expected.items()}},'upstream_review_sha256':prior['review_sha256']}
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
