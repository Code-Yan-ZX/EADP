"""Candidate real processor preflight helpers; CLI and GPU entry stay locked.

The eventual launcher must bind source hashes, approvals and real Docker identity.
Importing this module does not import torch, load models, read labels or use GPU.
"""
import hashlib,importlib.util,inspect,json,os,sys
from pathlib import Path
from types import SimpleNamespace
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2];S=B/'e6/pixelprune_t020_smoke';PIXEL=B/'e6/external_sources/pixelprune_8d347eb';AUTHOR=ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit';MODEL=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def load(name,path):
 path=Path(path).resolve();existing=sys.modules.get(name)
 if existing is not None:
  if not getattr(existing,'__file__',None) or Path(existing.__file__).resolve()!=path:raise ValueError('foreign cached module '+name)
  return existing
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def reject_foreign(name,path):
 m=sys.modules.get(name)
 if m is not None and (not getattr(m,'__file__',None) or Path(m.__file__).resolve()!=Path(path).resolve()):raise ValueError('foreign cached module '+name)
def preflight(plan):
 """Actual patch/import/three-processor/four-payload check; no model or CUDA."""
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('preflight requires hidden GPUs')
 inventory=plan.get('source_sha256',{})
 if not inventory or any(sha(path)!=digest for path,digest in inventory.items()):raise ValueError('source inventory mismatch')
 refpath=D/'reference_contract.py';oldplan=json.loads((S/'plan001.json').read_text());required=[D/'runtime_support.py',refpath,D/'identity_worker.py',D/'identity_contract.py',D/'archive_contract.py',S/'worker.py',S/'plan001.json',B/'e6/pixel_wavefront_cpu/candidate.py']
 if any(inventory.get(str(p.resolve()))!=sha(p) for p in required) or any(inventory.get(path)!=digest for path,digest in oldplan['sha256'].items()):raise ValueError('incomplete transitive source inventory')
 refs=load('_wavefront_identity_reference',refpath).reference(B/'e1_independent_review/pixel_t020_result_review.json',B/'e1_independent_review/review_pixel_t020_result.py',B/'e1_independent_review/monitor_pixel_t020_result/started.json',B/'e1_independent_review/monitor_pixel_t020_result/terminal.json',S/'run_smoke001/observations')
 input_path=B/'e4_c_latency/input_preflight.py';reject_foreign('input_preflight',input_path)
 old=load('_wavefront_t020_accepted_worker',S/'worker.py');chosen,e1refs=old.preflight(oldplan)
 if Path(sys.modules['input_preflight'].__file__).resolve()!=input_path:raise ValueError('wrong input preflight')
 rows={(x['task'],str(x['qid'])):x for x in chosen};
 if len(rows)!=4:raise ValueError('accepted smoke rows incomplete')
 for name,path in (('pixelprune',PIXEL/'pixelprune/__init__.py'),('pixelprune.methods.pred_2d',PIXEL/'pixelprune/methods/pred_2d.py'),('pixelprune.patches.qwen3_vl_hf',PIXEL/'pixelprune/patches/qwen3_vl_hf.py'),('vlmeval.vlm.qwen3_vl.model_fixed_res',AUTHOR/'vlmeval/vlm/qwen3_vl/model_fixed_res.py')):reject_foreign(name,path)
 import torch
 from transformers import AutoProcessor,AutoConfig
 from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
 from pixelprune import apply_pixelprune
 from pixelprune.methods import pred_2d
 from pixelprune.patches import qwen3_vl_hf as patch
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
 import pixelprune
 if torch.cuda.is_initialized() or Path(pixelprune.__file__).resolve()!=PIXEL/'pixelprune/__init__.py' or Path(inspect.getfile(apply_pixelprune)).resolve()!=PIXEL/'pixelprune/__init__.py' or Path(pred_2d.__file__).resolve()!=PIXEL/'pixelprune/methods/pred_2d.py' or Path(patch.__file__).resolve()!=PIXEL/'pixelprune/patches/qwen3_vl_hf.py' or Path(inspect.getfile(Qwen3VLChatFixedRes)).resolve()!=AUTHOR/'vlmeval/vlm/qwen3_vl/model_fixed_res.py':raise ValueError('runtime module source/CUDA mismatch')
 if getattr(hf.Qwen3VLVisionModel,'__patch_selector_patched__',False):raise ValueError('preexisting patch')
 apply_pixelprune(model='qwen3_vl')
 if (hf.Qwen3VLVisionModel.forward is not patch._vision_forward or hf.Qwen3VLModel.get_image_features is not patch._model_get_image_features or hf.Qwen3VLModel.forward is not patch._model_forward or hf.Qwen3VLForConditionalGeneration.forward is not patch._cg_forward):raise ValueError('author patch dispatch mismatch')
 candidate=load('_wavefront_candidate',B/'e6/pixel_wavefront_cpu/candidate.py')
 if not callable(candidate.select):raise ValueError('wavefront candidate missing')
 weights=json.loads((ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json').read_text());stats=old.weight_stats(weights)
 for w in weights['files']:
  if sha(w['path'])!=w['expected_lfs_sha256']:raise ValueError('model shard mismatch')
 if len(weights['files'])!=4 or old.weight_stats(weights)!=stats:raise ValueError('weights changed')
 from input_preflight import validate_processor
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False);config=AutoConfig.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
 stub=object.__new__(Qwen3VLChatFixedRes);stub.system_prompt=None;stub.min_pixels=stub.max_pixels=1048576;stub.total_pixels=None;stub.processor=processor;stub.model=SimpleNamespace(device='cpu',dtype=torch.bfloat16)
 verified=[]
 for entry in refs['images']:
  a=entry['anchor'];key=(a['task'],str(a['qid']));row=rows[key];e1=e1refs[key]
  if any(row[k]!=a[k] for k in ('group_id','image_sha256','message_sha256')) or sha(row['image_path'])!=a['image_sha256']:raise ValueError('anchor identity')
  actual=stub._processor_inputs(stub._build_messages(row['message'],dataset=row['task']));identity=validate_processor(torch,actual,e1['processor'],config.image_token_id,e1['image_start'])
  if identity['processor']!=entry['expected']['processor']:raise ValueError('same-arm processor mismatch')
  verified.append({'task':a['task'],'qid':a['qid'],'group_id':a['group_id'],'processor':identity['processor']})
 if torch.cuda.is_initialized() or len(verified)!=3:raise ValueError('unexpected CUDA or missing anchors')
 if any(sha(path)!=digest for path,digest in inventory.items()):raise ValueError('source inventory changed during preflight')
 return {'status':'passed','images':verified,'weights_verified':4,'weight_stats':stats,'source_sha256':inventory,'existing_DEV_predictions_opened':True,'GT_opened':False,'cuda_initialized':False,'model_loaded':False,'gpu_authorized':False,'author_patch_before_model':True}
if __name__=='__main__':raise SystemExit('locked: source-bound review, launcher and real no-GPU admission are not implemented')
