"""Draft single-process model identity continuation. No GPU entry is admitted.

The future controller and reviewer must independently bind this module, the
accepted no-GPU preflight, six scheduled containers and all input/weight hashes.
This module is deliberately not executable; it has not run on a real model.
"""
import importlib.util,inspect,json,os,sys
from pathlib import Path
from types import SimpleNamespace
D=Path(__file__).resolve().parent;B=D.parents[1];P=B/'e6/pixel_wavefront_model_t020';S=B/'e6/pixelprune_t020_smoke';I=B/'e1_independent_review';PIXEL=B/'e6/external_sources/pixelprune_8d347eb';ROOT=B.parents[2]
def load(name,path):
 path=Path(path).resolve();old=sys.modules.get(name)
 if old is not None:
  if not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=path:raise ValueError('foreign cached module '+name)
  return old
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def choose_row(rows,anchor):
 matches=[x for x in rows if x['task']==anchor['task'] and str(x['qid'])==str(anchor['qid'])]
 if len(matches)!=1 or any(matches[0][k]!=anchor[k] for k in ('group_id','image_sha256','message_sha256')):raise ValueError('actual image/input changed')
 return matches[0]

def run_one(*_args,**_kwargs):
 raise RuntimeError('GPU locked: execution plan, controller and independent review not implemented')
def _draft_unadmitted_implementation(task,qid,arm,out):
 """Draft body; future reviewed PID1 controller must replace the hard lock."""
 if arm not in ('author','wavefront'):raise ValueError('unplanned arm')
 # Bind the separately reviewed no-GPU result before any model/processor use.
 # The public entry remains locked; this check is not GPU admission.
 # The reviewed real no-GPU result is bound here by the future execution gate.
 gate=load('_wavefront_t020_execution_gpu_gate',D/'gpu_admission.py');gate.require_gpu()
 upstream=load('_wavefront_t020_execution_upstream',D/'upstream_identity.py');upstream.verify()
 admission=load('_wavefront_frozen_preflight_gate',P/'admission.py');approval=admission.require_preflight();plan=admission.read(P/'preflight_plan002.json');preflight=admission.read(P/'run_preflight001/worker_preflight.json')
 if preflight.get('status')!='passed' or preflight.get('approval')!=approval or preflight.get('cuda_initialized') is not False or preflight.get('model_loaded') is not False:raise ValueError('accepted CPU preflight missing')
 reference=load('_wavefront_frozen_reference',P/'reference_contract.py');refs=reference.reference(I/'pixel_t020_result_review.json',I/'review_pixel_t020_result.py',I/'monitor_pixel_t020_result/started.json',I/'monitor_pixel_t020_result/terminal.json',S/'run_smoke001/observations')
 if refs['images']!=plan['images']:raise ValueError('reference changed')
 matches=[x for x in refs['images'] if x['anchor']['task']==task and str(x['anchor']['qid'])==str(qid)]
 if len(matches)!=1:raise ValueError('anchor not frozen')
 entry=matches[0];anchor=entry['anchor'];expected=entry['expected'];input_path=B/'e4_c_latency/input_preflight.py';cached=sys.modules.get('input_preflight')
 if cached is not None and (not getattr(cached,'__file__',None) or Path(cached.__file__).resolve()!=input_path):raise ValueError('foreign cached input preflight')
 old=load('_wavefront_accepted_smoke',S/'worker.py');oldplan=admission.read(S/'plan001.json');rows,e1refs=old.preflight(oldplan)
 if 'input_preflight' not in sys.modules or Path(sys.modules['input_preflight'].__file__).resolve()!=input_path:raise ValueError('wrong actual input preflight')
 row=choose_row(rows,anchor)
 if old.sha(row['image_path'])!=anchor['image_sha256']:raise ValueError('actual image bytes changed')
 ref=e1refs[(task,str(qid))]
 expected_paths={'pixelprune':PIXEL/'pixelprune/__init__.py','pixelprune.methods.pred_2d':PIXEL/'pixelprune/methods/pred_2d.py','pixelprune.patches.qwen3_vl_hf':PIXEL/'pixelprune/patches/qwen3_vl_hf.py','vlmeval.vlm.qwen3_vl.model_fixed_res':ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py','transformers.models.qwen3_vl.modeling_qwen3_vl':ROOT/'_reproduction/envs/qwen_author/lib/python3.10/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py'}
 for name,path in expected_paths.items():
  cached=sys.modules.get(name)
  if cached is not None and (not getattr(cached,'__file__',None) or Path(cached.__file__).resolve()!=path.resolve()):raise ValueError('foreign cached module '+name)
 import torch
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
 from pixelprune import apply_pixelprune
 from pixelprune.methods.pred_2d import Pred2DSelector
 from pixelprune.patches import qwen3_vl_hf as patch
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
 for name,path in expected_paths.items():
  if name not in sys.modules or Path(sys.modules[name].__file__).resolve()!=path.resolve():raise ValueError('wrong actual module '+name)
 if Path(inspect.getfile(Qwen3VLChatFixedRes)).resolve()!=expected_paths['vlmeval.vlm.qwen3_vl.model_fixed_res'].resolve() or Path(inspect.getfile(Pred2DSelector)).resolve()!=expected_paths['pixelprune.methods.pred_2d'].resolve() or Path(inspect.getfile(Qwen3VLForConditionalGeneration)).resolve()!=expected_paths['transformers.models.qwen3_vl.modeling_qwen3_vl'].resolve():raise ValueError('wrong actual class source')
 if torch.cuda.is_initialized() or getattr(hf.Qwen3VLVisionModel,'__patch_selector_patched__',False):raise ValueError('unexpected preexisting CUDA/patch')
 os.environ.update(PIXELPRUNE_ENABLED='true',PIXELPRUNE_METHOD='pred_2d',PIXELPRUNE_METRIC='max',PIXELPRUNE_THRESHOLD='0.2',PIXELPRUNE_ANCHORED='true',PIXELPRUNE_VIT_LAYER='0',PIXELPRUNE_VERBOSE='false');os.environ.pop('PIXELPRUNE_LOG_FILE',None)
 apply_pixelprune(model='qwen3_vl')
 if (hf.Qwen3VLVisionModel.forward is not patch._vision_forward or hf.Qwen3VLModel.get_image_features is not patch._model_get_image_features or hf.Qwen3VLModel.forward is not patch._model_forward or hf.Qwen3VLForConditionalGeneration.forward is not patch._cg_forward):raise ValueError('author patch dispatch')
 modelpath=old.MODEL;processor=AutoProcessor.from_pretrained(modelpath,local_files_only=True,trust_remote_code=False)
 wrapper=object.__new__(Qwen3VLChatFixedRes);wrapper.system_prompt=None;wrapper.min_pixels=wrapper.max_pixels=1048576;wrapper.total_pixels=None;wrapper.processor=processor;wrapper.model=SimpleNamespace(device='cuda',dtype=torch.bfloat16)
 model=Qwen3VLForConditionalGeneration.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to('cuda').eval()
 if model.dtype!=torch.bfloat16 or model.config._attn_implementation!='flash_attention_2':raise ValueError('model recipe mismatch')
 inputs=wrapper._processor_inputs(wrapper._build_messages(row['message'],dataset=task));from input_preflight import validate_processor
 valid=validate_processor(torch,inputs,ref['processor'],model.config.image_token_id,ref['image_start'])
 if valid['processor']!=expected['processor']:raise ValueError('actual processor mismatch')
 worker=load('_wavefront_frozen_identity_worker',P/'identity_worker.py');candidate=load('_wavefront_selector',B/'e6/pixel_wavefront_cpu/candidate.py')
 result=worker.run_arm(torch,model,inputs,Pred2DSelector,candidate.select,arm,anchor,expected,expected['generation_kwargs'],Path(out))
 return result
if __name__=='__main__':raise SystemExit('GPU locked: draft single-arm implementation has no reviewed execution gate')
