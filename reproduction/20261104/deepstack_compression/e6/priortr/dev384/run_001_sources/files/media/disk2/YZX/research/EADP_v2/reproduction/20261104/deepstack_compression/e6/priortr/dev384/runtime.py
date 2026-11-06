"""Prepared PriorTR actual per-arm core; no model loading or executable entrypoint.

Gating, container, input/weight identity, exclusive GPU and post-run formal
verification are the caller's responsibility. Do not run unreviewed.
"""
import hashlib,json,os,time
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()

def invoker(*,torch,model,processor,VTRConfig,Trace,prepare,arrays_dir):
 """Construct a 32-sample runner using already-loaded fixed author model."""
 import numpy as np
 arrays_dir=Path(arrays_dir);swapped=[False]
 def met(t):
  c=t.detach().contiguous().cpu()
  return {'dtype':str(c.dtype),'shape':list(c.shape),'sha256':hashlib.sha256(c.view(torch.uint8).numpy().tobytes()).hexdigest()}
 def invoke(row,arm,on_generated,save_full_arrays):
  if arm not in ('stock_native','author_default','priortr_k256'):raise ValueError('unknown arm')
  if arm=='stock_native' and swapped[0]:raise ValueError('stock model already patched')
  if arm!='stock_native' and not swapped[0]:model._replace_text_model();swapped[0]=True
  cfg={'enabled':arm!='stock_native','strategy':'priortr','prune_layer':3}
  if arm=='author_default':cfg['keep_ratio']=0.1111
  if arm=='priortr_k256':cfg['keep_tokens']=256
  model.vtr_config=VTRConfig(**cfg);model.vtr_strategy=model._create_strategy('priortr')
  inputs=prepare(processor,row['message']).to('cuda').to(torch.bfloat16);original_length=inputs['input_ids'].shape[1]
  processor_meta={k:met(inputs[k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
  kwargs={'do_sample':False,'max_new_tokens':2048,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0,'use_cache':True}
  scalars={k:kwargs[k] for k in ('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty')}
  began=time.monotonic()
  with torch.inference_mode(),Trace(model,forced=False) as trace:
   generated=model.generate(**inputs,**kwargs);ids=generated[0,original_length:].detach().cpu().tolist()
   # This event is fsynced by the caller before any trace postprocessing.
   on_generated({'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'arm':arm,'output_ids':ids,'generation_scalars':scalars,'forced':False})
  elapsed=time.monotonic()-began
  for k in ('input_ids','attention_mask','pixel_values','image_grid_thw'):trace.save('processor_'+k,inputs[k])
  tensor_metadata={k:met(v) for k,v in trace.arrays.items()}
  if save_full_arrays:
   token=hashlib.sha256(json.dumps([row['task'],str(row['qid']),arm],separators=(',',':')).encode()).hexdigest()
   path=arrays_dir/(token+'.npz')
   with path.open('xb') as f:np.savez(f,**{k:v.view(torch.uint8).numpy() for k,v in trace.arrays.items()});f.flush();os.fsync(f.fileno())
   archive={'kind':'full_arrays','path':str(path),'sha256':sha(path),'tensors':tensor_metadata}
  else:archive={'kind':'summary_only','tensors':tensor_metadata}
  eos=model.generation_config.eos_token_id;eos_ids=[eos] if isinstance(eos,int) else list(eos or [])
  return {'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'image_sha256':row['image_sha256'],'message_sha256':row['message_sha256'],'arm':arm,'output_ids':ids,'generation_scalars':scalars,'forced':False,'processor':processor_meta,'eos_ids':eos_ids,'eos':bool(ids and ids[-1] in eos_ids),'truncated':bool(len(ids)==2048 and ids[-1] not in eos_ids),'answer':processor.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),'backend':'sdpa','precision':'bfloat16','arrays':archive,'trace':{'calls':trace.calls,'layers':trace.layers,'injections':trace.injections,'prunes':trace.prunes},'rope_delta':None if model.model.rope_deltas is None else model.model.rope_deltas.detach().cpu().tolist(),'rope_offset':model._rope_position_offset,'instrumented_wall_seconds':elapsed,'use_cache':kwargs['use_cache'],'small_diagnostics':{k:v.tolist() for k,v in trace.arrays.items() if k in ('selected_ids_0','selection_scores_0','prune_before_position_0','prune_after_position_0','prune_after_mask_0')}}
 return invoke
