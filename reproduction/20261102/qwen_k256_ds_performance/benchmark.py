"""Paired Qwen3 EADP K256 performance; natural EOS and fixed 32-token modes.
Uses original author generate and selector; light hooks forward real DeepStack rows.
No per-layer clones, array serialization, scoring, or GPU sync inside timed calls.
"""
import hashlib, json, os, sys, time
from pathlib import Path
R=Path('/media/disk2/YZX/research/EADP_v2')
Q=R/'reproduction/20261028/qwen'
OUT=R/'reproduction/20261102/qwen_k256_ds_performance'
old=json.loads((Q/'gpu_formula_facility_panel_run02/run_plan.json').read_text())
panel=json.loads((Q/'fixed16_k256_ds_toggle/plan.json').read_text())['panel']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''): h.update(chunk)
 return h.hexdigest()
def save(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
sys.path[:0]=[old['author_root'],str(Path(old['author_root'])/'VLMEvalKit')]
import torch, numpy as np
from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatCDPruner,Qwen3VLChatEADP
import model.pruner as native
import importlib.metadata
assert {n:importlib.metadata.version(n) for n in old['expected_versions']}==old['expected_versions']
for p,d in old['source_sha256'].items(): assert sha(p)==d,p
for row in panel: assert sha(row['image_path'])==row['image_sha256']
torch.set_num_threads(4)
assert torch.cuda.get_device_name(0)=='NVIDIA A40'
w=Qwen3VLChatEADP(**old['recipe'])
assert w.visual_token_num==w.pruner.visual_token_num==256
assert w.model.config._attn_implementation==w.model.visual.config._attn_implementation=='flash_attention_2'
assert w.model.dtype==torch.bfloat16
assert {str(p.device) for p in w.model.parameters()}=={'cuda:0'}
lang=w.model.model.language_model
method=Qwen3VLChatCDPruner.generate_inner_transformers_fixed
runtime=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),dtype=str(w.model.dtype), attention='flash_attention_2',recipe=old['recipe'],warmup='one sample per arm and output mode', wall_scope='processor (including file read), author visual encoder, pruner, LLM generation and tokenizer decode; excludes model load and validation', peak_scope='PyTorch CUDA peak allocated bytes from reset immediately before each call, includes resident model, not driver/other processes or allocations outside PyTorch', prefill_scope='CUDA event interval around first language_model forward only; timed call has lightweight hooks in both arms; does not include vision or selection', fixed_tokens='min_new_tokens=max_new_tokens=32; alters generation EOS only for equal-length speed control', author_source_sha256=old['source_sha256'],model_weight_stat=old['model_stat_guards'],panel_sha256=sha(Q/'fixed16_k256_ds_toggle/plan.json'),code_sha256=sha(__file__),pid=os.getpid(),GT_read=False)
save('runtime.json',runtime)
rows=[]

def run(row,arm,mode,warm=False):
 state={'calls':0}; handles=[]
 # Observing selection adds only a Python function call and reference; it uses
 # the real selection result and never changes the returned tensor.
 original_select=native._greed_select_impl
 def choose(*a,**k):
  result=original_select(*a,**k);state['selected']=result[0];return result
 original_processor=w._processor_inputs; original_generate=w.model.generate
 def processor(*a,**k):
  x=original_processor(*a,**k)
  # Both arms pay the identical one-off CPU sync, avoid data hashing in timing.
  ids=x['input_ids'][0];pos=(ids==w.model.config.image_token_id).nonzero(as_tuple=True)[0]
  assert pos.numel()==1024
  state['start']=int(pos[0].item());return x
 def visual(module,args,result):
  if arm=='on': state['deepstack']=result[1]
 def generate(*a,**k):
  if mode=='fixed32':k['max_new_tokens']=32;k['min_new_tokens']=32
  result=original_generate(*a,**k);state['output']=result;return result
 start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
 def inject(module,args,kwargs):
  state['calls']+=1
  if state['calls']!=1:return None
  start_event.record()
  if arm=='on':
   assert kwargs.get('deepstack_visual_embeds') is None and kwargs.get('visual_pos_masks') is None
   ordered=state['selected'][0];ids=ordered.sort().values
   x=kwargs['inputs_embeds']; mask=torch.zeros(x.shape[:2],dtype=torch.bool,device=x.device)
   mask[:,state['start']:state['start']+256]=True
   kwargs=dict(kwargs);kwargs['visual_pos_masks']=mask
   kwargs['deepstack_visual_embeds']=[v.index_select(0,ids) for v in state['deepstack']]
   return args,kwargs
 def post(module,args,kwargs,result):
  if state['calls']==1:end_event.record()
 try:
  native._greed_select_impl=choose;w._processor_inputs=processor;w.model.generate=generate
  handles.append(w.model.visual.register_forward_hook(visual))
  handles.append(lang.register_forward_pre_hook(inject,with_kwargs=True))
  handles.append(lang.register_forward_hook(post,with_kwargs=True))
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
  with torch.no_grad(): response=method(w,row['message'],dataset='DocVQA_VAL')
  torch.cuda.synchronize();elapsed=time.perf_counter()-begin
  peak=torch.cuda.max_memory_allocated();prefill_ms=start_event.elapsed_time(end_event)
  # All validation and transfers are outside the measured interval.
  ordered=state['selected'][0].detach().cpu().numpy();assert len(ordered)==256
  expected=np.load(Q/'fixed16_k256_ds_toggle/execution'/f'{panel.index(row):02d}_ds_off.npz')['ordered_ids'][0]
  assert np.array_equal(ordered,expected),('selection mismatch',row['index'],arm,mode)
  ids=state['output'][0].detach().cpu().tolist();assert ids
  if mode=='fixed32': assert len(ids)==32
  if mode=='natural':
   prior=json.loads((Q/'fixed16_k256_ds_toggle/execution'/f'{panel.index(row):02d}_ds_{arm}.json').read_text())
   assert response==prior['response'],('response mismatch',row['index'],arm,response,prior['response'])
  return dict(index=row['index'],position=row['position'],arm=arm,mode=mode,warmup=warm,seconds=elapsed,prefill_gpu_ms=prefill_ms,peak_allocated_bytes=peak,output_tokens=len(ids),output_ids=ids,response=response,ordered_selection_sha256=hashlib.sha256(ordered.tobytes()).hexdigest(),prefill_calls=state['calls'])
 finally:
  native._greed_select_impl=original_select;w._processor_inputs=original_processor;w.model.generate=original_generate
  for h in handles:h.remove()

for mode in ('natural','fixed32'):
 for arm in ('off','on'):
  z=run(panel[0],arm,mode,True);rows.append(z); print('warmup',mode,arm,flush=True)
 for i,row in enumerate(panel):
  for arm in (('off','on') if i%2==0 else ('on','off')):
   z=run(row,arm,mode);rows.append(z)
   with (OUT/'observations.jsonl').open('a') as f:f.write(json.dumps(z,ensure_ascii=False)+'\n')
   print(mode,i,arm,round(z['seconds'],3),z['output_tokens'],flush=True)
  pair=rows[-2:];assert pair[0]['ordered_selection_sha256']==pair[1]['ordered_selection_sha256']
save('complete.json',dict(status='complete',n_measured=64,n_warmup=4, rows=len(rows),code_sha256=sha(__file__),observations_sha256=sha(OUT/'observations.jsonl')))
