"""Instrument the actual timing request for three-anchor identity only.

Diagnostic hooks, cloning and KV inspection perturb runtime; no timings from
this module may be used as a performance result. No CLI/GPU authority.
"""
from contextlib import ExitStack
import hashlib

def meta(torch,t):
 x=t.detach().contiguous().cpu()
 return {'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}
def finish(*,torch,arm,row,ref,inputs,prompt,output,response,state,trace,selected,decode):
 """Return all comparison outcomes; malformed observations fail closed."""
 from process_core import processor_evidence,file_sha
 if arm not in ('rres','eadp') or len(output)==0:raise ValueError('identity scope/output')
 actual_proc=processor_evidence(torch,inputs)
 visual=meta(torch,state['visual']);deep=[meta(torch,x) for x in state['deepstack']];injected=[meta(torch,x) for x in state['injected_ds']]
 position=state['prefill_position_ids'].detach().cpu().tolist();delta=state['delta'].detach().cpu().tolist();mask=state['prefill_visual_pos_masks'].detach().cpu().tolist()
 prefill_embeds=meta(torch,state['prefill_inputs_embeds']);prefill_attention=meta(torch,state['prefill_attention_mask'])
 logits=meta(torch,state['logits'])
 execution={'block_calls':state['block_calls'],'merger_calls':state['merger_calls'],'ds_calls':state['ds_calls']}
 if arm=='rres':
  expected_position=ref['prefill_position_ids'];actual_position=meta(torch,state['prefill_position_ids']);expected_delta=ref['delta'];expected_visual=256;compact=len(prompt)
  expected_mask=[[row['image_start']<=i<row['image_start']+256 for i in range(len(prompt))]]
  checks={'position':actual_position==expected_position,'delta':delta==expected_delta,'visual_mask':mask==expected_mask,'three_stream_injection':injected==ref['injected_ds'],'visual':visual==ref['main_visual'],'three_streams':deep==ref['deepstack'],'prefill_embeds':prefill_embeds==ref['prefill_inputs_embeds'],'prefill_attention':prefill_attention==ref['prefill_attention_mask'],'prefill_logits':logits==ref['prefill_logits']}
 else:
  expected_visual=1024;compact=len(prompt)-1024+256
  checks={'position':position==ref['prefill_position_ids'],'delta':delta==ref['adjusted_delta'],'visual_mask':mask==ref['injected_mask'],'three_stream_injection':injected==ref['injected_deepstack'],'visual':visual==ref['main_visual'],'three_streams':deep==ref['full_deepstack'],'ordered_selection':selected==ref['ordered_ids'] and len(selected)==256,'prefill_embeds':prefill_embeds==ref['prefill_embeds'],'prefill_attention':prefill_attention==ref['prefill_mask']}
 image_sha=file_sha(row['image_path'])
 checks.update(processor=actual_proc==ref['processor'],image=image_sha==row['image_sha256'],natural_tokens=output==ref['output_ids'],response=response==ref['answer'],visual_size=visual['shape'][0]==expected_visual,stream_count=len(deep)==len(injected)==3)
 checks['visual_execution_counts']=(execution['block_calls']==ref['block_calls'] and execution['merger_calls']==ref['merger_calls' if arm=='rres' else 'main_merger_calls'] and execution['ds_calls']==ref['ds_calls'])
 try:decode.validate_trace(trace,output,compact,int(delta[0][0]));checks['all_36_layer_kv_positions']=True
 except (ValueError,IndexError,KeyError,TypeError):checks['all_36_layer_kv_positions']=False
 return {'status':'passed' if all(checks.values()) else 'failed','task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'arm':arm,'checks':checks,'image_sha256':image_sha,'processor':actual_proc,'visual':visual,'deepstack':deep,'injected_deepstack':injected,'visual_execution_counts':execution,'prefill_inputs_embeds':prefill_embeds,'prefill_attention_mask':prefill_attention,'prefill_logits':logits,'prefill_logits_reference_compared':arm=='rres','prefill_position_ids':position,'prefill_position_meta':meta(torch,state['prefill_position_ids']),'prefill_visual_mask':mask,'delta':delta,'ordered_ids':selected,'input_length':len(prompt),'output_ids':output,'response':response,'decode_trace':trace,'timing_claims':False,'high_dimensional_tensors_archived':False,'errors':[k for k,v in checks.items() if not v]}

def run_one(*,torch,wrapper,module,pruner,row,ref,arm,qwen_request,adapter_type,criteria_type,fixed_resolution,author_generate,capture_module,decode,persist):
 """Use the same request path; only this diagnostic installs extra hooks."""
 collected={};wrapper.model.model.rope_deltas=None
 class ObservedAdapter:
  def __init__(self,*args):self.args=args;self.stack=ExitStack()
  def __enter__(self):
   try:
    actual=self.stack.enter_context(adapter_type(*self.args));collected['adapter']=actual
    collected['capture']=self.stack.enter_context(capture_module.Capture(torch,wrapper));collected['trace']=self.stack.enter_context(decode.DecodeObserver(wrapper))
    return actual
   except BaseException:self.stack.close();raise
  def __exit__(self,*args):return self.stack.__exit__(*args)
 with ExitStack() as stack:
  if arm=='rres':
   collected['capture']=stack.enter_context(capture_module.Capture(torch,wrapper));collected['trace']=stack.enter_context(decode.DecodeObserver(wrapper))
  raw=[]
  def save(t):persist(t);raw.append(t.detach().cpu().tolist()[0])
  result=qwen_request.request(arm=arm,torch=torch,wrapper=wrapper,module=module,pruner=pruner,row=row,accepted=ref,adapter_type=ObservedAdapter if arm=='eadp' else adapter_type,criteria_type=criteria_type,fixed_resolution=fixed_resolution,author_generate=author_generate,persist=save)
  inputs=result.pop('processor_inputs');prompt=inputs['input_ids'].detach().cpu().tolist()[0]
  output=qwen_request.runner.normalize(arm,raw[0],prompt);selected=collected['adapter'].state['ordered'].detach().cpu().tolist() if arm=='eadp' else None
  outcome=finish(torch=torch,arm=arm,row=row,ref=ref,inputs=inputs,prompt=prompt,output=output,response=result['response'],state=collected['capture'].state,trace=collected['trace'].trace,selected=selected,decode=decode)
  times=result['times'];expected_first=len(prompt)+1 if arm=='rres' else 1
  outcome['checks']['first_criteria_length']=times['first_criteria_length']==expected_first
  outcome['checks']['criteria_calls']=times['criteria_calls']==len(output)
  outcome['errors']=[k for k,v in outcome['checks'].items() if not v];outcome['status']='passed' if not outcome['errors'] else 'failed'
  outcome['generation_observation']=result['generation'];outcome['first_criteria_length']=times['first_criteria_length'];outcome['criteria_calls']=times['criteria_calls'];outcome['timing_claims']=False
  return outcome
