"""Prepared t=.2 model continuation; no CLI or GPU authority is provided.

A future reviewer must bind this module and every transitive source, enforce
all gates in the worker/controller and perform a real no-GPU preflight first.
"""
import hashlib,importlib.util,inspect,os,sys
from pathlib import Path
from types import SimpleNamespace
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2];V4=B/'e6/pixel_natural_timing_pilot_v4';PIXEL=B/'e6/external_sources/pixelprune_8d347eb';AUTHOR=ROOT/'_reproduction/author_eadp/Qwen_vl';MODEL=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
def load(name,path):
 path=Path(path).resolve();old=sys.modules.get(name)
 if old is not None:
  if not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=path:raise ValueError('foreign source '+name)
  return old
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m

def prepare_cpu():
 """Actual import/source and six-reference check only; never loads a model."""
 ref=load('_pixel_t020_timing_reference',D/'reference.py');references,identity=ref.load()
 base=load('_pixel_t020_timing_base_worker',V4/'worker.py');continuation=load('_pixel_t020_timing_worker',D/'worker_candidate.py');candidate=load('_pixel_t020_timing_wavefront',B/'e6/pixel_wavefront_cpu/candidate.py')
 import torch
 if torch.cuda.is_initialized():raise ValueError('CUDA initialized before preflight')
 sys.path[:0]=[str(AUTHOR),str(AUTHOR/'VLMEvalKit'),str(PIXEL)]
 expected={'pixelprune':PIXEL/'pixelprune/__init__.py','pixelprune.methods.pred_2d':PIXEL/'pixelprune/methods/pred_2d.py','pixelprune.patches.qwen3_vl_hf':PIXEL/'pixelprune/patches/qwen3_vl_hf.py','vlmeval.vlm.qwen3_vl.model_fixed_res':AUTHOR/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py','transformers.models.qwen3_vl.modeling_qwen3_vl':ROOT/'_reproduction/envs/qwen_author/lib/python3.10/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py'}
 for name,path in expected.items():
  old=sys.modules.get(name)
  if old is not None and (not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=path.resolve()):raise ValueError('foreign cached module '+name)
 from pixelprune.methods.pred_2d import Pred2DSelector
 from pixelprune.patches import qwen3_vl_hf as pixel
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
 from transformers import Qwen3VLForConditionalGeneration
 for name,path in expected.items():
  if name not in sys.modules or Path(sys.modules[name].__file__).resolve()!=path.resolve():raise ValueError('wrong actual source '+name)
 if Path(inspect.getfile(Pred2DSelector)).resolve()!=expected['pixelprune.methods.pred_2d'].resolve() or Path(inspect.getfile(Qwen3VLChatFixedRes)).resolve()!=expected['vlmeval.vlm.qwen3_vl.model_fixed_res'].resolve() or Path(inspect.getfile(Qwen3VLForConditionalGeneration)).resolve()!=expected['transformers.models.qwen3_vl.modeling_qwen3_vl'].resolve() or torch.cuda.is_initialized():raise ValueError('model source or CUDA state')
 return {'references':references,'identity':identity,'base':base,'continuation':continuation,'candidate':candidate,'torch':torch,'selector':Pred2DSelector,'pixel':pixel,'wrapper_class':Qwen3VLChatFixedRes,'model_class':Qwen3VLForConditionalGeneration,'module_paths':{k:str(v) for k,v in expected.items()}}

def unsafe_continuation(prepared,job,row,e1ref,emit_raw):
 """Implementation only. Calling requires a separately reviewed real gate."""
 import torch,transformers
 from input_preflight import validate_processor
 from transformers import AutoProcessor
 from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
 from pixelprune import apply_pixelprune
 key=(job['task'],str(job['qid']),job['arm']);ref=prepared['references'][key]
 if any(str(row.get(k))!=str(job[k]) for k in ('task','qid','group_id','image_sha256','message_sha256')) or e1ref['processor']!=ref['processor']:raise ValueError('actual row identity')
 h=hashlib.sha256()
 with Path(row['image_path']).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 if h.hexdigest()!=job['image_sha256']:raise ValueError('image bytes changed')
 if transformers.__version__!='4.57.1' or torch.cuda.get_device_name(0)!='NVIDIA A40' or getattr(hf.Qwen3VLVisionModel,'__patch_selector_patched__',False):raise ValueError('runtime identity')
 os.environ.update(PIXELPRUNE_ENABLED='true',PIXELPRUNE_METHOD='pred_2d',PIXELPRUNE_METRIC='max',PIXELPRUNE_THRESHOLD='0.2',PIXELPRUNE_ANCHORED='true',PIXELPRUNE_VIT_LAYER='0',PIXELPRUNE_VERBOSE='false');os.environ.pop('PIXELPRUNE_LOG_FILE',None)
 apply_pixelprune(model='qwen3_vl');patch=prepared['pixel']
 if hf.Qwen3VLVisionModel.forward is not patch._vision_forward or hf.Qwen3VLForConditionalGeneration.forward is not patch._cg_forward:raise ValueError('wrong pixel patch')
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False);model=prepared['model_class'].from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to('cuda').eval()
 if model.dtype!=torch.bfloat16 or model.config._attn_implementation!='flash_attention_2':raise ValueError('wrong actual backend')
 wrapper=object.__new__(prepared['wrapper_class']);wrapper.system_prompt=None;wrapper.min_pixels=wrapper.max_pixels=1048576;wrapper.total_pixels=None;wrapper.fps=None;wrapper.processor=processor;wrapper.model=model;wrapper.verbose=False;wrapper.post_process=False;wrapper.generate_kwargs={k:v for k,v in prepared['base'].RECIPE.items() if k not in ('do_sample','num_beams','use_cache')}
 observed={}
 def validate_actual(inputs):
  value=validate_processor(torch,inputs,ref['processor'],model.config.image_token_id,e1ref['image_start']);observed.update(value)
 result=prepared['continuation'].execute(prepared['base'],prepared['selector'],prepared['candidate'].select,job['arm'],wrapper,patch,torch,row,torch.cuda.synchronize,ref,emit_raw,validate_actual)
 times=result['boundaries'];result.update(job,processor=observed['processor'],output_ids=result['tokens'],generation_scalars=result['recipe'],start_monotonic_ns=times['t0_before_image_read_ns'],first_token_monotonic_ns=times['first_token_selected_ns'],completion_monotonic_ns=times['generation_complete_ns'],start_unix_ns=times['t0_unix_ns'],completion_unix_ns=times['completion_unix_ns'],first_token_synchronized=True,gpu_synchronized_before_start=True)
 return result
if __name__=='__main__':raise SystemExit('GPU locked: controller and independent review not implemented')
