"""Instrumented identity arm using the unmodified lightweight adapter itself.

This diagnostic is not a latency implementation; all hooks/hashes are absent
from the main timing path. The caller owns evidence gates and GPU scheduling.
"""
from contextlib import ExitStack
import hashlib,importlib.util
from pathlib import Path
from budget_capture import BudgetCapture
E4=Path(__file__).resolve().parent.parent/'e4'

def tensor_meta(tensor):
 import torch
 if tensor is None:return None
 x=tensor.detach().contiguous().cpu()
 return {'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}

def load_decode():
 spec=importlib.util.spec_from_file_location('eadp_timing_decode_observer',E4/'decode_observer.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class CacheFingerprints:
 def __init__(self,w,all_steps):self.w=w;self.all_steps=all_steps;self.fingerprints=[]
 def __enter__(self):
  def observe(_m,_args,_kw,out):
   if self.fingerprints and not self.all_steps:return
   cache=out.past_key_values
   if cache is None or len(cache.layers)!=36:raise ValueError('expected actual 36-layer cache')
   self.fingerprints.append([{'key':tensor_meta(layer.keys),'value':tensor_meta(layer.values)} for layer in cache.layers])
  self.handle=self.w.model.model.language_model.register_forward_hook(observe,with_kwargs=True);return self
 def __exit__(self,*_args):self.handle.remove()

def run_arm(torch,w,pruner,row,budget,generate,on_generated,archive_arrays,forced_token=None,accepted_panel=None):
 if row.get('split')!='DEV' or budget not in (128,256,512,1024):raise ValueError('invalid diagnostic scope')
 if forced_token is not None:raise ValueError('fixed32 does not force tokens')
 forced=False;seen={};decode=load_decode()
 with ExitStack() as stack:
  adapter=stack.enter_context(BudgetCapture(torch,w,pruner,budget,row['image_start']));s=adapter.state
  original=w.model.generate
  had_generate='generate' in vars(w.model);stored_generate=vars(w.model).get('generate')
  def capture(*args,**kw):
   if seen:raise ValueError('multiple generations')
   if kw.get('max_new_tokens')!=2048:raise ValueError('wrong natural generation limit')
   if forced:
    from transformers import LogitsProcessor,LogitsProcessorList
    class ForceToken(LogitsProcessor):
     def __call__(self,_ids,scores):scores.fill_(-float('inf'));scores[:,forced_token]=0;return scores
    kw.update(max_new_tokens=33,min_new_tokens=33,eos_token_id=[],logits_processor=LogitsProcessorList([ForceToken()]))
   if kw.get('min_new_tokens') is not None:raise ValueError('unexpected original recipe')
   kw.update(min_new_tokens=32,max_new_tokens=32)
   names=('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty')
   seen['generation_scalars']={key:kw.get(key) for key in names};seen['min_new_tokens']=kw.get('min_new_tokens')
   eos=kw.get('eos_token_id',w.model.generation_config.eos_token_id);seen['eos_ids']=[eos] if isinstance(eos,int) else list(eos or [])
   from fixed_common import verify_kwargs
   verify_kwargs(kw,seen['eos_ids'],accepted_panel)
   result=original(*args,**kw);seen['output_ids']=result.detach().cpu().tolist()[0]
   on_generated({'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'budget':budget,'ds_on':True,'forced':forced,
                 'output_ids':seen['output_ids'],'eos_ids':seen['eos_ids'],'generation_scalars':seen['generation_scalars'],'min_new_tokens':seen['min_new_tokens'],'protocol':'fixed32'})
   return result
  def restore():
   if had_generate:w.model.generate=stored_generate
   else:delattr(w.model,'generate')
  w.model.generate=capture;stack.callback(restore)
  with decode.DecodeObserver(w) as trace,CacheFingerprints(w,True) as kv:answer=generate(w,row)
  ids=seen.get('output_ids');eos=seen.get('eos_ids');maximum=seen.get('generation_scalars',{}).get('max_new_tokens')
  if not ids or not isinstance(answer,str) or maximum!=32 or len(ids)!=32 or any(x in eos for x in ids[:-1]):raise ValueError('generation output invalid')
  if forced and (ids!=[forced_token]*33 or eos!=[]):raise ValueError('forced prefix invalid')
  if not torch.isfinite(s['logits']).all():raise ValueError('nonfinite prefill logits')
  if s['block_calls']!=[1]*len(w.model.visual.blocks) or s['main_merger_calls']!=1 or s['ds_calls']!=[1]*3 or s['ds_none']!=[False]*3:raise ValueError('visual branch execution mismatch')
  inp=s['inputs'];start=row['image_start'];ordered=s['ordered'];spatial=ordered.sort().values
  if ordered.numel()!=budget:raise ValueError('selected budget mismatch')
  full_pos,delta=w.model.model.get_rope_index(inp['input_ids'],inp['image_grid_thw'],None,attention_mask=inp['attention_mask'])
  retained=torch.cat((torch.arange(start,device=spatial.device),start+spatial,torch.arange(start+1024,inp['input_ids'].shape[1],device=spatial.device)))
  decode.validate_trace(trace.trace,ids,len(retained),int(s['adjusted_delta'][0,0]))
  if s['prefill_visual_pos_masks'] is None or s['injected_ds'] is None or len(s['injected_ds'])!=3:raise ValueError('ON injection missing')
  tensors={'visual':s['visual'],'pruned':s['pruned'],'prefill':s['prefill_inputs_embeds'],'prefill_logits':s['logits'],
           **{'processor_'+k:inp[k] for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}}
  if s['deepstack'] is not None:tensors.update({'ds'+str(i):x for i,x in enumerate(s['deepstack'])})
  if s['injected_ds'] is not None:tensors.update({'injected'+str(i):x for i,x in enumerate(s['injected_ds'])})
  arrays=archive_arrays(tensors)
  return {'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'image_sha256':row['image_sha256'],'message_sha256':row['message_sha256'],
   'budget':budget,'ds_on':True,'selection_mode':'keep_all' if budget==1024 else 'author_greedy','greedy_calls':adapter.adapter.state['greedy_calls'],'forced':forced,'forced_token':forced_token,'processor':{k:tensor_meta(inp[k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},
   'main_visual':tensor_meta(s['visual']),'pruned_visual':tensor_meta(s['pruned']),'image_start':start,'original_length':inp['input_ids'].shape[1],
   'full_deepstack':None if s['deepstack'] is None else [tensor_meta(x) for x in s['deepstack']],
   'ordered_ids':ordered.detach().cpu().tolist(),'spatial_ids':spatial.detach().cpu().tolist(),'retained_ids':retained.detach().cpu().tolist(),
   'full_position_ids':full_pos.detach().cpu().tolist(),'full_delta':delta.detach().cpu().tolist(),
   'prefill_position_ids':s['prefill_position_ids'].detach().cpu().tolist(),'adjusted_delta':s['adjusted_delta'].detach().cpu().tolist(),
   'prefill_embeds':tensor_meta(s['prefill_inputs_embeds']),'prefill_mask':tensor_meta(s['prefill_attention_mask']),
   'injected_mask':None if s['prefill_visual_pos_masks'] is None else s['prefill_visual_pos_masks'].detach().cpu().tolist(),
   'injected_deepstack':None if s['injected_ds'] is None else [tensor_meta(x) for x in s['injected_ds']],
   'prefill_logits':tensor_meta(s['logits']),'output_ids':ids,'answer':answer,'eos_ids':eos,'eos':ids[-1] in eos,
   'truncated':not forced and ids[-1] not in eos and len(ids)>=2048,'protocol':'fixed32','max_new_tokens':maximum,'generation_scalars':seen['generation_scalars'],'min_new_tokens':seen['min_new_tokens'],
   'decode_trace':trace.trace,'cache_fingerprints':kv.fingerprints,'block_calls':s['block_calls'],'main_merger_calls':s['main_merger_calls'],
   'ds_calls':s['ds_calls'],'ds_none':s['ds_none'],'arrays':arrays}
