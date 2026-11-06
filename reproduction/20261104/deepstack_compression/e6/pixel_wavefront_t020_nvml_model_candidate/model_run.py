"""Pixel model-memory continuation. No CLI, no GPU authority, no sampler.

The caller must first admit the exact process, record a pre-CUDA NVML sample,
and complete the reviewed container/host PID handshake. No timing claim.
"""
import hashlib,importlib.util,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];C=B/'e6/pixel_wavefront_t020_model_timing_v2'
def load(name,path):
 p=Path(path).resolve()
 if name in sys.modules:raise ValueError('cached candidate module')
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def execute(prepared,job,row,e1ref,ref,record,emit_raw):
 """Run once and return live model and identity; caller must clean up."""
 torch=prepared['torch'];base=prepared['base'];selector=prepared['selector'];arm=job['arm']
 if torch.cuda.is_initialized() or arm not in ('t0_author','t020_author','t020_wavefront'):raise ValueError('before-CUDA or arm contract')
 if any(str(row.get(k))!=str(job[k]) for k in ('task','qid','group_id','image_sha256','message_sha256')) or e1ref['processor']!=ref['processor'] or sha(row['image_path'])!=job['image_sha256']:raise ValueError('input/reference identity')
 # Do not import GPU model modules until the caller has released the PID1 barrier.
 import transformers
 from transformers import AutoProcessor
 from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
 from pixelprune import apply_pixelprune
 from input_preflight import validate_processor
 if transformers.__version__!='4.57.1' or getattr(hf.Qwen3VLVisionModel,'__patch_selector_patched__',False):raise ValueError('model runtime identity')
 threshold=0.0 if arm=='t0_author' else 0.2
 os.environ.update(PIXELPRUNE_ENABLED='true',PIXELPRUNE_METHOD='pred_2d',PIXELPRUNE_METRIC='max',PIXELPRUNE_THRESHOLD=str(threshold),PIXELPRUNE_ANCHORED='true',PIXELPRUNE_VIT_LAYER='0',PIXELPRUNE_VERBOSE='false');os.environ.pop('PIXELPRUNE_LOG_FILE',None)
 apply_pixelprune(model='qwen3_vl');patch=prepared['pixel']
 if hf.Qwen3VLVisionModel.forward is not patch._vision_forward or hf.Qwen3VLForConditionalGeneration.forward is not patch._cg_forward:raise ValueError('wrong applied patch')
 mark=lambda name,**kw:record({'event':name,'monotonic_ns':time.monotonic_ns(),**kw})
 mark('model_load_start');processor=AutoProcessor.from_pretrained(prepared['model_path'],local_files_only=True,trust_remote_code=False)
 model=prepared['model_class'].from_pretrained(prepared['model_path'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to('cuda').eval();torch.cuda.synchronize()
 if model.dtype!=torch.bfloat16 or model.config._attn_implementation!='flash_attention_2':raise ValueError('wrong model backend')
 mark('model_load_end',allocated_peak_bytes=int(torch.cuda.max_memory_allocated()),reserved_peak_bytes=int(torch.cuda.max_memory_reserved()))
 wrapper=object.__new__(prepared['wrapper_class']);wrapper.system_prompt=None;wrapper.min_pixels=wrapper.max_pixels=1048576;wrapper.total_pixels=None;wrapper.fps=None;wrapper.processor=processor;wrapper.model=model;wrapper.verbose=False;wrapper.post_process=False;wrapper.generate_kwargs={k:v for k,v in base.RECIPE.items() if k not in ('do_sample','num_beams','use_cache')}
 actual={}
 def validate_actual(inputs):actual.update(validate_processor(torch,inputs,ref['processor'],model.config.image_token_id,e1ref['image_start']))
 torch.cuda.reset_peak_memory_stats();mark('request_start')
 if arm=='t0_author':
  trace=load('_pixel_t0_nvml_trace',D/'t0_trace.py');continuation=prepared['continuation']
  with trace.observe(selector,ref['selected_indices']) as selected:
   with continuation.timed_measure(base):result=base.execute_pixel(wrapper,patch,torch,row,torch.cuda.synchronize,ref,emit_raw,observe_count=True,validate_input=validate_actual)
  selected_ids=selected[0][1].cpu().tolist()
 else:
  result=prepared['continuation'].execute(base,selector,prepared['candidate'].select,arm[5:],wrapper,patch,torch,row,torch.cuda.synchronize,ref,emit_raw,validate_actual);selected_ids=result['selected_indices']
 torch.cuda.synchronize();mark('request_end',allocated_peak_bytes=int(torch.cuda.max_memory_allocated()),reserved_peak_bytes=int(torch.cuda.max_memory_reserved()))
 if selected_ids!=ref['selected_indices'] or result['tokens']!=ref['tokens'] or result['eos_ids']!=ref['eos_ids'] or actual['processor']!=ref['processor']:raise ValueError('same-arm identity mismatch')
 # The existing generation wrapper uses stopping criteria, synchronization and an
 # on-device selector copy; all affect memory. Its elapsed values are not retained.
 value={'status':'passed','job':job,'processor':actual['processor'],'selected_indices':selected_ids,'output_ids':result['tokens'],'eos_ids':result['eos_ids'],'generation_kwargs':result['recipe'],'GT_opened':False,'scoring_performed':False,'limits':['capture and synchronization affect memory','full hidden tensors not archived','no timing claim']}
 return model,wrapper,value
if __name__=='__main__':raise SystemExit('GPU locked: no reviewed admission, no sampler or executable launcher')
