"""Unlaunched draft for two independent author K256 generations.

No CLI, model loader, allocation, file writer, or scoring. An eventual reviewed
controller must supply the real wrapper/torch, historical-anchor verifier and
exception-safe durable raw/observation writers. Never treat this module as a
ready-to-run experiment; terminal provenance and archival are not implemented.
"""
from intervention import propose
from selection_observer import SelectionObserver

def snapshot(torch,value):
 if isinstance(value,torch.Tensor):return value.detach().clone()
 if isinstance(value,(tuple,list)):return type(value)(snapshot(torch,x) for x in value)
 if isinstance(value,dict):return {k:snapshot(torch,v) for k,v in value.items()}
 return value

def run_arm(torch,wrapper,native_module,method,item,arm,baseline,preserve_raw):
 """Invoke original author method once; preserve response before assertions.

 preserve_raw is a required durable callback (item, arm, response, failure).
 The caller owns a nonconcurrent wrapper and must separately persist every
 observation, output IDs and failure before applying anchor/completion gates.
 """
 if arm not in ('ds_off','ds_on') or (arm=='ds_on' and baseline is None):raise ValueError('invalid pair order')
 language=wrapper.model.model.language_model
 if language._forward_pre_hooks or torch.nn.modules.module._global_forward_pre_hooks:raise RuntimeError('unexpected existing prehooks')
 rec=dict(errors=[],vision_calls=0,pruner_calls=0,processor_calls=0,generate_calls=0,llm_calls=0,continuation=[],intervention_attempts=0,intervention_applied=False,intervention_error=None)
 def safe(key,fn):
  try:rec[key]=fn()
  except Exception as error:rec['errors'].append(dict(event=key,error=repr(error)))
 def visual(module,args,result):
  rec['vision_calls']+=1;safe('vision',lambda:snapshot(torch,result))
 def pruner(module,args,result):
  rec['pruner_calls']+=1;safe('pruner_input',lambda:snapshot(torch,args[0]));safe('pruner_output',lambda:snapshot(torch,result[0]));safe('split_sizes',lambda:list(result[1]))
 def inject(module,args,kwargs):
  rec['llm_calls']+=1
  if rec['llm_calls']!=1:return None
  safe('original_prefill',lambda:snapshot(torch,kwargs));safe('rope_deltas',lambda:snapshot(torch,wrapper.model.model.rope_deltas))
  changed=kwargs
  if arm=='ds_on':
   rec['intervention_attempts']+=1
   try:
    if rec['errors'] or selection.calls!=1 or len(selection.results)!=1 or selection.errors or rec['vision_calls']!=1 or rec['pruner_calls']!=1 or rec['processor_calls']!=1 or rec['generate_calls']!=1:
     raise ValueError('native capture counts or selection observation invalid')
    greedy=selection.results[0]
    if greedy['quotas'].detach().cpu().tolist()!=[256]:raise ValueError('unexpected native selection quota')
    vision=rec['vision'];ids=greedy['ordered_ids'][0]
    current=dict(ordered_ids=ids,spatial_ids=ids.sort().values,visual_features=vision[0],deepstack=vision[1],
      pruner_input=rec['pruner_input'],pruner_output=rec['pruner_output'],split_sizes=rec['split_sizes'],
      processor_input_ids=rec['processor_inputs']['input_ids'],processor_mask=rec['processor_inputs']['attention_mask'],processor_pixel_values=rec['processor_inputs']['pixel_values'],processor_image_grid_thw=rec['processor_inputs']['image_grid_thw'],
      original_token_embeddings=rec['original_token_embeddings'],generation_kwargs=rec['generation_kwargs'],
      rope_deltas=rec['rope_deltas'],image_token_id=wrapper.model.config.image_token_id)
    changed,rec['intervention_applied'],rec['intervention_error']=propose(torch,baseline,current,kwargs)
   except Exception as error:rec['intervention_error']='observer prerequisite: '+repr(error)
  safe('proposed_prefill',lambda:snapshot(torch,changed))
  if changed is not kwargs:return args,changed
 def final_observe(module,args,kwargs):
  if rec['llm_calls']==1:safe('final_prefill',lambda:snapshot(torch,kwargs))
  else:
   try:rec['continuation'].append(dict(position_ids=snapshot(torch,kwargs.get('position_ids')),cache_position=snapshot(torch,kwargs.get('cache_position')),rope_deltas=snapshot(torch,wrapper.model.model.rope_deltas),visual_mask_present=kwargs.get('visual_pos_masks') is not None,deepstack_present=kwargs.get('deepstack_visual_embeds') is not None))
   except Exception as error:rec['errors'].append(dict(event='continuation',error=repr(error)))
 handles=[]
 original_processor=wrapper._processor_inputs;original_generate=wrapper.model.generate
 def processor(*args,**kwargs):
  value=original_processor(*args,**kwargs);rec['processor_calls']+=1;safe('processor_inputs',lambda:snapshot(torch,value))
  def source_tokens():
   with torch.no_grad():return snapshot(torch,wrapper.model.get_input_embeddings()(value['input_ids']))
  safe('original_token_embeddings',source_tokens);return value
 def generate(*args,**kwargs):
  rec['generate_calls']+=1;safe('generation_kwargs',lambda:snapshot(torch,kwargs))
  result=original_generate(*args,**kwargs);safe('output_ids',lambda:snapshot(torch,result));return result
 response=None;failure=None;selection=SelectionObserver(native_module)
 try:
  handles.append(wrapper.model.visual.register_forward_hook(visual))
  handles.append(wrapper.pruner.register_forward_hook(pruner))
  handles.append(language.register_forward_pre_hook(inject,with_kwargs=True))
  handles.append(language.register_forward_pre_hook(final_observe,with_kwargs=True))
  wrapper._processor_inputs=processor;wrapper.model.generate=generate
  with selection:response=method(wrapper,item['message'],dataset='DocVQA_VAL')
 except BaseException as error:failure=repr(error)
 finally:
  wrapper._processor_inputs=original_processor;wrapper.model.generate=original_generate
  for h in handles:h.remove()
 # Even if capture/hook fails, keep the actual result before validation.
 preserve_raw(item,arm,response,failure)
 rec['selection_calls']=selection.calls;rec['selection_errors']=selection.errors;rec['selection_results']=selection.results
 rec['response']=response;rec['native_failure']=failure
 return rec

def baseline_capture(wrapper,rec):
 """Build baseline from actual observer results, only after artifact retention."""
 if rec['native_failure'] or rec['errors'] or rec['selection_errors']:raise ValueError('baseline observation incomplete')
 if any(rec[k]!=1 for k in ('vision_calls','pruner_calls','processor_calls','generate_calls','selection_calls')):raise ValueError('unexpected native call count')
 if rec['llm_calls']<1 or len(rec['continuation'])!=rec['llm_calls']-1:raise ValueError('incomplete continuation')
 selection=rec['selection_results'][0]
 if selection['quotas'].detach().cpu().tolist()!=[256]:raise ValueError('unexpected native selection quota')
 ids=selection['ordered_ids'][0];vision=rec['vision']
 return dict(ordered_ids=ids,spatial_ids=ids.sort().values,visual_features=vision[0],deepstack=vision[1],
  pruner_input=rec['pruner_input'],pruner_output=rec['pruner_output'],split_sizes=rec['split_sizes'],
  processor_input_ids=rec['processor_inputs']['input_ids'],processor_mask=rec['processor_inputs']['attention_mask'],processor_pixel_values=rec['processor_inputs']['pixel_values'],processor_image_grid_thw=rec['processor_inputs']['image_grid_thw'],
  original_token_embeddings=rec['original_token_embeddings'],generation_kwargs=rec['generation_kwargs'],
  original_prefill=rec['original_prefill'],rope_deltas=rec['rope_deltas'],image_token_id=wrapper.model.config.image_token_id)
