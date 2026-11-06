"""Candidate model run continuation; callable only by a future reviewed PID1 worker.

No CLI, no admission, and no CUDA initialization are provided here.
"""
from contextlib import ExitStack
import sys
from pathlib import Path
_HERE=Path(__file__).resolve().parent
_imports=sys.modules.get('imports')
if _imports is not None and (not getattr(_imports,'__file__',None) or Path(_imports.__file__).resolve()!=_HERE/'imports.py'):raise ValueError('foreign imports module')
from imports import checked_import,D,B
run=checked_import('observe',D/'observe.py').run

def raw_capture(original,record,identity,recipe,eos_ids,actual_default_eos):
 seen={}
 def capture(*args,**kw):
  if seen:raise ValueError('duplicate generation')
  names=('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty');scalars={k:kw.get(k) for k in names}
  expected={k:recipe[k] for k in ('top_p','top_k','temperature','repetition_penalty')};expected.update(do_sample=False,max_new_tokens=2048)
  eos=kw.get('eos_token_id',actual_default_eos);eos=[eos] if isinstance(eos,int) else list(eos or [])
  if scalars!=expected or eos!=eos_ids:raise ValueError('actual generation recipe differs')
  seen.update(generation_scalars=scalars,eos_ids=eos)
  result=original(*args,**kw)
  tokens=result.detach().cpu().tolist()[0]
  record({'event':'generate_returned',**identity,'output_ids':tokens,'generation_scalars':scalars,'eos_ids':eos})
  seen['output_ids']=tokens
  return result
 return capture,seen

def real_model(torch,row,ref,recipe,pruner,wrapper_class,cd_class,record,monotonic_ns):
 LightAdapter=checked_import('light_adapter',B/'e4_c_latency/light_adapter.py').LightAdapter
 model_holder={};state={}
 def load():
  w=wrapper_class(**recipe)
  if w.model.dtype!=torch.bfloat16 or w.model.config._attn_implementation!='flash_attention_2':raise ValueError('unexpected model backend')
  model_holder['wrapper']=w;return w
 def generate(w):
  expected_eos=ref['eos_ids'];original=w.model.generate;had='generate' in w.model.__dict__;raw=w.model.__dict__.get('generate')
  def restore():
   if had:w.model.generate=raw
   else:delattr(w.model,'generate')
  cap,seen=raw_capture(original,record,{'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id']},recipe,expected_eos,w.model.generation_config.eos_token_id)
  with ExitStack() as stack:
   w.model.generate=cap;stack.callback(restore);adapter=stack.enter_context(LightAdapter(torch,w,pruner,'C-ON',row['image_start']))
   with torch.inference_mode():answer=cd_class.generate_inner_transformers_fixed(w,row['message'],dataset=row['task'])
   if not adapter.state['prefill_seen'] or 'output_ids' not in seen:raise ValueError('incomplete natural generation')
   state.update(inputs=adapter.state['inputs'],ordered=adapter.state['ordered'],seen=seen,answer=answer)
  return state
 wrapper,response=run(torch,load,generate,record,monotonic_ns)
 # Caller records request_end before invoking expensive identity hashing.
 return wrapper,response

def validate_result(response,reference,tensor_metadata):
 """Call only after request_end; hashes affect later memory, not request peak."""
 if any(response['seen'].get(k)!=reference[k] for k in ('output_ids','generation_scalars','eos_ids')) or response.get('answer')!=reference.get('answer'):raise ValueError('accepted anchor output/recipe differs')
 if response['ordered'].detach().cpu().tolist()!=reference['ordered_ids']:raise ValueError('accepted anchor selection differs')
 actual={key:tensor_metadata(response['inputs'][key]) for key in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
 if actual!=reference['processor']:raise ValueError('accepted anchor processor differs')
 return {'status':'passed','scope':'same selected indices, processor and exact output; actual P3/DS behavior relies on separately accepted identical light adapter source'}
