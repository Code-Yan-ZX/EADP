"""Actual Qwen3 local StackTok diagnostic arm; no standalone GPU entry."""
from contextlib import ExitStack
import torch
import re
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];sys.path[:0]=[str(BASE/'e6/stacktok_audit/prototype'),str(BASE/'e4_native'),str(BASE/'e4')]
from model_adapter import LocalStackTokAdapter
from stock_capture import CacheFingerprints
from decode_observer import DecodeObserver
from identity_records import tensor_meta

def validate_recipe(record,recipe,forced,expected_eos):
 names=('top_p','top_k','temperature','repetition_penalty')
 expected={key:recipe[key] for key in names};expected.update(do_sample=False,max_new_tokens=33 if forced else 2048)
 if record.get('generation_scalars')!=expected:raise ValueError('generation recipe differs from frozen author recipe')
 if record.get('eos_ids')!=([] if forced else expected_eos) or not expected_eos:raise ValueError('generation EOS differs from native model')
 return True

def validate_fingerprints(records,forced,compact):
 expected_steps=33 if forced else 1
 if len(records)!=expected_steps:raise ValueError('cache fingerprint step count')
 for step,layers in enumerate(records):
  if len(layers)!=36:raise ValueError('cache fingerprint layer count')
  for layer in layers:
   if set(layer)!={'key','value'}:raise ValueError('cache fingerprint fields')
   for item in layer.values():
    shape=item.get('shape',[])
    if item.get('dtype')!='torch.bfloat16' or len(shape)!=4 or shape[0]!=1 or shape[-2]!=compact+step or min(shape)<=0 or not re.fullmatch('[0-9a-f]{64}',item.get('sha256','')):raise ValueError('cache fingerprint metadata')
 return True

def validate_trace(trace,ids,full_length,selected,delta):
 if len(trace)!=len(ids) or not trace:raise ValueError('forward count differs from generated tokens')
 removed=1024-selected;compact=full_length-removed;layers=36;first=trace[0]
 if first['cache_position']!=list(range(compact)) or first['rope_delta']!=[[delta]] or first['layer_cache_lengths'] not in ([],[0]*layers) or first['layer_cache_lengths_after']!=[compact]*layers:raise ValueError('prefill cache mismatch')
 for i,x in enumerate(trace[1:],start=1):
  physical=compact+i-1;nominal=full_length+i-1
  if x['cache_position']!=[physical] or x['position_ids']!=[[[nominal+delta]]]*3 or x['rope_delta']!=[[delta]] or x['layer_cache_lengths']!=[physical]*layers or x['layer_cache_lengths_after']!=[physical+1]*layers:raise ValueError('actual decode cache/position mismatch')
 return True

def run_local(torch,w,row,selector,k,generate,on_generated,archive,forced_token=None):
 forced=forced_token is not None;seen={};logits={}
 with ExitStack() as stack:
  adapter=stack.enter_context(LocalStackTokAdapter(w,row,selector,k))
  def actual_prefill(_m,_args,kw):
   if 'actual_prefill' in seen:return
   names=('inputs_embeds','attention_mask','position_ids','visual_pos_masks','deepstack_visual_embeds')
   def copy(x):return [copy(y) for y in x] if isinstance(x,(tuple,list)) else x.detach().clone()
   seen['actual_prefill']={name:copy(kw[name]) for name in names}
  h=w.model.model.language_model.register_forward_pre_hook(actual_prefill,with_kwargs=True);stack.callback(h.remove)
  def head(_m,_args,_kw,out):
   if not logits:logits['prefill']=out[:,-1,:].detach().clone()
  h=w.model.lm_head.register_forward_hook(head,with_kwargs=True);stack.callback(h.remove)
  original=w.model.generate;had='generate' in w.model.__dict__;raw=w.model.__dict__.get('generate')
  def restore():
   if had:w.model.generate=raw
   else:delattr(w.model,'generate')
  def capture(*args,**kw):
   if seen or 'input_ids' not in kw or kw.get('inputs_embeds') is not None:raise ValueError('native ID generation required')
   if forced:
    from transformers import LogitsProcessor,LogitsProcessorList
    class ForceToken(LogitsProcessor):
     def __call__(self,_ids,scores):scores.fill_(-float('inf'));scores[:,forced_token]=0;return scores
    kw.update(max_new_tokens=33,min_new_tokens=33,eos_token_id=[],logits_processor=LogitsProcessorList([ForceToken()]))
   names=('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty');seen['generation_scalars']={name:kw.get(name) for name in names};eos=kw.get('eos_token_id',w.model.generation_config.eos_token_id);seen['eos_ids']=[eos] if isinstance(eos,int) else list(eos or [])
   result=original(*args,**kw);allids=result.detach().cpu().tolist()[0];prompt=kw['input_ids'].detach().cpu().tolist()[0]
   if allids[:len(prompt)]!=prompt:raise ValueError('prompt changed')
   seen['output_ids']=allids[len(prompt):]
   on_generated({'path':'local_keepall' if k==1024 else 'local_stacktok','task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'forced':forced,'output_ids':seen['output_ids'],'eos_ids':seen['eos_ids'],'generation_scalars':seen['generation_scalars']})
   return result
  w.model.generate=capture;stack.callback(restore)
  with DecodeObserver(w) as trace,CacheFingerprints(w,forced) as kv:answer=generate(w,row)
  state=adapter.state;ids=seen['output_ids'];eos=seen['eos_ids'];maxlen=seen['generation_scalars']['max_new_tokens'];inp=state['inputs'];pf=state['prefill'];visual=state['visual'];delta=state['original_delta'];full=inp['input_ids'].shape[1]
  actual=seen['actual_prefill']
  for key in ('inputs_embeds','attention_mask','position_ids','visual_pos_masks'):
   if not torch.equal(actual[key],pf[key]):raise ValueError('actual prefill differs from selected reconstruction: '+key)
  if any(not torch.equal(x,y) for x,y in zip(actual['deepstack_visual_embeds'],pf['deepstack_visual_embeds'])):raise ValueError('actual DS injection mismatch')
  if not ids or len(ids)>maxlen or not isinstance(answer,str) or 'prefill' not in logits or not torch.isfinite(logits['prefill']).all():raise ValueError('invalid generation or logits')
  if forced and (ids!=[forced_token]*33 or eos!=[]):raise ValueError('forced prefix mismatch')
  if not forced and ids[-1] not in eos and len(ids)!=maxlen:raise ValueError('natural output did not terminate')
  validate_trace(trace.trace,ids,full,k,int(delta[0,0]));validate_fingerprints(kv.fingerprints,forced,full-1024+k)
  tensors={'premerger':visual['premerger'],'averaged_premerger':visual['averaged_premerger'],'visual':visual['postmerger'],'prefill':pf['inputs_embeds'],'prefill_logits':logits['prefill'],'question_ids':state['question']['input_ids'],'question_embeddings':state['question']['embeddings'],'prefill_mask':actual['attention_mask'],'prefill_positions':actual['position_ids'],'visual_mask':actual['visual_pos_masks'],'retained_indices':state['retained'],**{'ds'+str(i):x for i,x in enumerate(visual['deepstack'])},**{'injected'+str(i):x for i,x in enumerate(pf['deepstack_visual_embeds'])},**{'processor_'+q:inp[q] for q in ('input_ids','attention_mask','pixel_values','image_grid_thw')}}
  arr=archive(tensors);sel=state['selected'];q=state['question'];return {'path':'local_keepall' if k==1024 else 'local_stacktok','task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'image_sha256':row['image_sha256'],'message_sha256':row['message_sha256'],'forced':forced,'budget':k,'question_text':q['text'],'question_ids':tensor_meta(q['input_ids']),'question_embeddings':tensor_meta(q['embeddings']),'prefill_positions':tensor_meta(actual['position_ids']),'visual_mask':tensor_meta(actual['visual_pos_masks']),'retained_indices':tensor_meta(state['retained']),'processor':{n:tensor_meta(inp[n]) for n in ('input_ids','attention_mask','pixel_values','image_grid_thw')},'premerger':tensor_meta(visual['premerger']),'averaged_premerger':tensor_meta(visual['averaged_premerger']),'main_visual':tensor_meta(visual['postmerger']),'full_deepstack':[tensor_meta(x) for x in visual['deepstack']],'prefill_embeds':tensor_meta(pf['inputs_embeds']),'prefill_mask':tensor_meta(pf['attention_mask']),'prefill_position_ids':pf['position_ids'].detach().cpu().tolist(),'rope_delta':delta.detach().cpu().tolist(),'injected_mask':pf['visual_pos_masks'].detach().cpu().tolist(),'injected_deepstack':[tensor_meta(x) for x in pf['deepstack_visual_embeds']],'prefill_logits':tensor_meta(logits['prefill']),'spatial_ids':sel['spatial_ids'],'greedy_ids':sel['greedy_ids'],'selector_repeat_exact':True,'output_ids':ids,'answer':answer,'eos_ids':eos,'eos':ids[-1] in eos,'truncated':not forced and ids[-1] not in eos and len(ids)==maxlen,'max_new_tokens':maxlen,'generation_scalars':seen['generation_scalars'],'decode_trace':trace.trace,'cache_fingerprints':kv.fingerprints,'arrays':arr}

def compare_keepall(native,local):
 if native.get('path')!='stock_native' or local.get('path')!='local_keepall' or local.get('budget')!=1024:raise ValueError('wrong paths')
 for key in ('task','qid','group_id','image_sha256','message_sha256','forced','processor','main_visual','full_deepstack','prefill_embeds','prefill_mask','prefill_position_ids','rope_delta','injected_mask','injected_deepstack','prefill_logits','output_ids','answer','eos_ids','eos','truncated','max_new_tokens','generation_scalars','decode_trace','cache_fingerprints'):
  if native.get(key)!=local.get(key):raise ValueError('native keep-all mismatch: '+key)
 if local['spatial_ids']!=list(range(1024)):raise ValueError('keep-all indices')
 return True
