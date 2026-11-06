"""Stock-model three-anchor identity arm. Instrumented; never use as timing."""
import hashlib,json,os
from contextlib import ExitStack
from pathlib import Path
from capture import Capture,validate
from resolution import fixed_resolution

def meta(t):
 import torch
 x=t.detach().contiguous().cpu()
 return {'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}

def persist(path,value):
 with Path(path).open('x') as f:json.dump(value,f);f.write('\n');f.flush();os.fsync(f.fileno())

def run_arm(torch,wrapper,module,row,ref,raw_path,decode):
 """Each record is returned only after raw full sequence is durable and checks pass."""
 observed={};original_processor=wrapper._processor_inputs;original_generate=wrapper.model.generate
 hook_counts=lambda:[(id(m),len(m._forward_pre_hooks),len(m._forward_hooks)) for m in wrapper.model.modules()]
 before_hooks=hook_counts()
 had_proc='_processor_inputs' in wrapper.__dict__;old_proc=wrapper.__dict__.get('_processor_inputs');had_gen='generate' in wrapper.model.__dict__;old_gen=wrapper.model.__dict__.get('generate')
 def restore(obj,name,had,old):
  if had:setattr(obj,name,old)
  else:delattr(obj,name)
 def processor(*args,**kwargs):
  if 'inputs' in observed:raise ValueError('processor called more than once')
  with fixed_resolution(module,512):inputs=original_processor(*args,**kwargs)
  if any(meta(inputs[k])!=ref['low_processor'][k] for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')):raise ValueError('processor differs from accepted noGPU reference')
  observed['inputs']=inputs;return inputs
 def generate(*args,**kwargs):
  if 'raw' in observed or kwargs.get('do_sample') is not False or kwargs.get('max_new_tokens')!=2048:raise ValueError('generation control or duplicate')
  inputs=observed['inputs'];positions,delta=wrapper.model.model.get_rope_index(inputs['input_ids'],image_grid_thw=inputs['image_grid_thw'],attention_mask=inputs['attention_mask'])
  if meta(positions)!=ref['position_ids'] or delta.tolist()!=ref['rope_delta']:raise ValueError('processor mRoPE reference mismatch')
  observed['positions']=positions;observed['delta']=delta.tolist()
  result=original_generate(*args,**kwargs)
  # Preserve the entire returned sequence before interpreting its shape or decoding.
  raw={'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'full_sequence':result.detach().cpu().tolist(),'input_length':int(inputs['input_ids'].shape[1])}
  persist(raw_path,raw);observed['raw']=raw
  return result
 with ExitStack() as stack:
  wrapper._processor_inputs=processor;stack.callback(restore,wrapper,'_processor_inputs',had_proc,old_proc)
  wrapper.model.generate=generate;stack.callback(restore,wrapper.model,'generate',had_gen,old_gen)
  capture=stack.enter_context(Capture(torch,wrapper));trace=stack.enter_context(decode.DecodeObserver(wrapper))
  answer=wrapper.generate_inner(row['message'],dataset=row['task'])
  raw=observed.get('raw');inputs=observed['inputs'];full=raw['full_sequence'];length=raw['input_length']
  if len(full)!=1 or full[0][:length]!=inputs['input_ids'][0].detach().cpu().tolist():raise ValueError('raw sequence prefix')
  ids=full[0][length:];validate(torch,capture.state,inputs,ref['image_start'],observed['positions'],observed['delta']);decode.validate_trace(trace.trace,ids,length,int(observed['delta'][0][0]))
  eos=wrapper.model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos or [])
  if not ids or len(ids)>2048 or (ids[-1] not in eos and len(ids)!=2048):raise ValueError('generation end')
  s=capture.state
  record={'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'image_sha256':row['image_sha256'],'message_sha256':row['message_sha256'],'resolution':512,'visual_tokens':256,'processor':{k:meta(inputs[k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},'main_visual':meta(s['visual']),'deepstack':[meta(x) for x in s['deepstack']],'prefill_inputs_embeds':meta(s['prefill_inputs_embeds']),'prefill_attention_mask':meta(s['prefill_attention_mask']),'prefill_position_ids':meta(s['prefill_position_ids']),'visual_mask':meta(s['prefill_visual_pos_masks']),'injected_ds':[meta(x) for x in s['injected_ds']],'delta':observed['delta'],'prefill_logits':meta(s['logits']),'block_calls':s['block_calls'],'merger_calls':s['merger_calls'],'ds_calls':s['ds_calls'],'output_ids':ids,'answer':answer,'eos_ids':eos,'eos':ids[-1] in eos,'truncated':ids[-1] not in eos,'decode_trace':trace.trace,'timing_claims':False}
 if hook_counts()!=before_hooks or wrapper._processor_inputs!=original_processor or wrapper.model.generate!=original_generate or module.QWEN3_FIXED_RESOLUTION!=1024:raise ValueError('hook, method or resolution not restored')
 record['hooks_and_methods_restored']=True
 return record
