"""Candidate model-identity instrumentation; CLI deliberately locked.

No model is loaded on import. A future supervised worker must call run_after_admission
only after source, container, budget and independent prelaunch gates are complete.
Identity instrumentation is not a timing path.
"""
import hashlib,json,os,time,traceback
from contextlib import ExitStack
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def append(path,obj):
 with Path(path).open('a') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def write(path,obj):
 with Path(path).open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def tensor_meta(torch,t):
 x=t.detach().contiguous().cpu();return {'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}
def save_tensors(torch,path,values):
 import numpy as np
 arrays={name:t.detach().contiguous().cpu().view(torch.uint8).numpy() for name,t in values.items()}
 with Path(path).open('xb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
 return {'archive_path':str(Path(path).resolve()),'archive_sha256':sha(path)}
def run_arm(torch,model,inputs,selector_class,wavefront,arm,anchor,expected,kwargs,out):
 """Trace one actual natural generation; preserves raw output before checks."""
 if arm not in ('author','wavefront') or kwargs!=expected['generation_kwargs'] or kwargs.get('do_sample') is not False or kwargs.get('max_new_tokens')!=2048:raise ValueError('identity arm contract')
 actual_processor={k:tensor_meta(torch,inputs[k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
 if actual_processor!=expected['processor']:raise ValueError('actual processor input differs')
 eos=model.generation_config.eos_token_id;eos=[eos] if type(eos) is int else list(eos or [])
 if eos!=expected['eos_ids'] or not eos:raise ValueError('actual EOS differs')
 full_positions,full_delta=model.model.get_rope_index(inputs['input_ids'],inputs['image_grid_thw'],None,inputs['attention_mask'])
 original=selector_class._select_2d_loco_anchored;seen=[];prefill={};cache=[];logits={};trace=[];out=Path(out)
 def choose(self,tokens,h,w,device):
  if self.method!='max' or self.threshold!=.45 or not self.anchored:raise ValueError('actual selector configuration')
  result=original(self,tokens,h,w,device) if arm=='author' else wavefront(tokens,h,w,self.threshold)
  seen.append(result.detach().cpu().tolist());append(out/'events.jsonl',{'event':'selector_returned','arm':arm,'task':anchor['task'],'qid':anchor['qid'],'indices':seen[-1]});return result
 def pre(_module,_args,kw):
  position=kw.get('position_ids');cache_position=kw.get('cache_position')
  if position is None or cache_position is None or model.model.rope_deltas is None:raise ValueError('missing decode position/cache/delta')
  trace.append({'position_ids':position.detach().cpu().tolist(),'cache_position':cache_position.detach().cpu().tolist(),'rope_delta':model.model.rope_deltas.detach().cpu().tolist()})
  if prefill:return
  for name in ('inputs_embeds','attention_mask','position_ids','visual_pos_masks'):
   v=kw.get(name)
   if v is None:raise ValueError('missing actual prefill '+name)
   prefill[name]=v.detach().clone()
  ds=kw.get('deepstack_visual_embeds')
  if not isinstance(ds,(tuple,list)) or len(ds)!=3:raise ValueError('missing actual DS injections')
  prefill['ds']=[x.detach().clone() for x in ds]
  if model.model.rope_deltas is None:raise ValueError('missing rope delta')
  prefill['delta']=model.model.rope_deltas.detach().clone()
 def post(_module,_args,_kw,result):
  past=result.past_key_values
  if past is None or len(past.layers)!=36:raise ValueError('missing 36-layer actual cache')
  cache.append([{'key':tensor_meta(torch,x.keys),'value':tensor_meta(torch,x.values)} for x in past.layers])
 def head(_module,_args,_kw,value):
  if not logits:logits['prefill']=value[:,-1,:].detach().clone()
 start=inputs['input_ids'].shape[1];prefix=inputs['input_ids'][0].detach().cpu().tolist()
 append(out/'events.jsonl',{'event':'generation_started','arm':arm,'task':anchor['task'],'qid':anchor['qid'],'monotonic_ns':time.monotonic_ns()})
 with ExitStack() as stack:
  selector_class._select_2d_loco_anchored=choose;stack.callback(setattr,selector_class,'_select_2d_loco_anchored',original)
  for module,func,prehook in ((model.model.language_model,pre,True),(model.model.language_model,post,False),(model.lm_head,head,False)):
   h=module.register_forward_pre_hook(func,with_kwargs=True) if prehook else module.register_forward_hook(func,with_kwargs=True);stack.callback(h.remove)
  model.model.rope_deltas=None
  with torch.inference_mode():full=model.generate(**inputs,**kwargs)
  raw=full[0].detach().cpu().tolist();write(out/'returned.json',{'input_length':start,'full_ids':raw,'task':anchor['task'],'qid':anchor['qid'],'arm':arm});append(out/'events.jsonl',{'event':'generation_returned','arm':arm,'task':anchor['task'],'qid':anchor['qid'],'monotonic_ns':time.monotonic_ns()})
 if selector_class._select_2d_loco_anchored is not original:raise ValueError('selector hook did not restore')
 if len(seen)!=1:raise ValueError('actual selector call count')
 where=torch.nonzero(inputs['input_ids'][0]==model.config.image_token_id).flatten()
 if len(where)!=1024 or not torch.equal(where,torch.arange(where[0],where[0]+1024,device=where.device)):raise ValueError('image grid identity')
 begin=int(where[0]);keep=torch.tensor(list(range(begin))+[begin+i for i in seen[0]]+list(range(begin+1024,start)),device=where.device)
 if not torch.equal(prefill['position_ids'],full_positions.index_select(2,keep)) or prefill['delta'].detach().cpu().tolist()!=full_delta.detach().cpu().tolist():raise ValueError('actual P3 position/delta differs')
 mask=prefill['visual_pos_masks'];expected_mask=torch.zeros((1,len(keep)),device=mask.device,dtype=torch.bool);expected_mask[:,begin:begin+len(seen[0])]=True
 if not torch.equal(mask,expected_mask) or any(x.shape[0]!=len(seen[0]) for x in prefill['ds']):raise ValueError('actual visual/DS mask differs')
 if raw[:start]!=prefix or len(seen)!=1 or seen[0]!=expected['selected_indices'] or raw[start:]!=expected['tokens'] or not prefill or not logits or len(cache)!=len(raw[start:]) or len(trace)!=len(raw[start:]):raise ValueError('actual selector/output/cache identity mismatch')
 tensors={'prefill_embeddings':prefill['inputs_embeds'],'prefill_mask':prefill['attention_mask'],'prefill_positions':prefill['position_ids'],'visual_mask':prefill['visual_pos_masks'],'prefill_logits':logits['prefill'],**{'injected_ds'+str(i):x for i,x in enumerate(prefill['ds'])}}
 archive=save_tensors(torch,out/'actual_prefill.npz',tensors)
 record={'status':'completed','path':arm,'task':anchor['task'],'qid':anchor['qid'],'group_id':anchor['group_id'],'message_sha256':anchor['message_sha256'],'image_sha256':anchor['image_sha256'],'selected_indices':seen[0],'output_ids':raw[start:],'eos_ids':eos,'generation_scalars':kwargs,'processor':actual_processor,'original_sequence_length':start,'original_max_position':int(full_positions.max()),'prefill_mask':prefill['attention_mask'].detach().cpu().tolist(),'prefill_position_ids':prefill['position_ids'].detach().cpu().tolist(),'visual_mask':prefill['visual_pos_masks'].detach().cpu().tolist(),'main_visual':tensor_meta(torch,prefill['inputs_embeds'][prefill['visual_pos_masks']]),'injected_deepstack':[tensor_meta(torch,x) for x in prefill['ds']],'prefill_embeddings':tensor_meta(torch,prefill['inputs_embeds']),'prefill_logits':tensor_meta(torch,logits['prefill']),'prefill_tensors':{k:tensor_meta(torch,v) for k,v in tensors.items()},'rope_delta':prefill['delta'].detach().cpu().tolist(),'decode_trace':trace,'cache_fingerprints':cache,'GT_opened':False,'scoring_performed':False,**archive};write(out/'record.json',record);return record
if __name__=='__main__':raise SystemExit('GPU locked: controller, frozen plan and independent preflight not yet implemented')
