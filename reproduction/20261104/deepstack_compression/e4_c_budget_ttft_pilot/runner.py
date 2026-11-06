"""Single natural TTFT job; no CLI, no admission, never a GPU authority."""
def execute(torch,wrapper,native,row,budget,accepted,adapter_type,measure,criteria_type,author_generate,on_raw):
 """Include image read/preprocess; fsync raw output after first-token boundary.

 `measure` must synchronize at pre-T0, after first chosen token, and at end.
 Raw journal persistence happens before the end boundary, so only TTFT, not
 this end-to-end duration, is a clean generation timing measurement.
 """
 if row.get('split')!='DEV' or budget not in (128,256,512,1024) or accepted.get('budget')!=budget or accepted.get('ds_on') is not True:raise ValueError('invalid accepted sample')
 for key in ('task','group_id','message_sha256','image_sha256'):
  if row.get(key)!=accepted.get(key):raise ValueError('sample identity mismatch')
 if str(row['qid'])!=str(accepted['qid']) or type(row.get('image_start')) is not int:raise ValueError('sample qid/start mismatch')
 wrapper.visual_token_num=budget;wrapper.pruner.visual_token_num=budget
 original=wrapper.model.generate;seen={}
 expected=accepted['generation_scalars'];names=('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty')
 if set(expected)!=set(names) or expected.get('max_new_tokens')!=2048 or expected.get('do_sample') is not False:raise ValueError('accepted recipe invalid')
 def invoke(boundary):
  def generate(*args,**kwargs):
   if seen:raise ValueError('multiple generate calls')
   if {k:kwargs.get(k) for k in names}!=expected or kwargs.get('min_new_tokens') is not None:raise ValueError('actual natural recipe mismatch before generation')
   eos=kwargs.get('eos_token_id',wrapper.model.generation_config.eos_token_id);eos=[eos] if isinstance(eos,int) else list(eos or [])
   if eos!=accepted['eos_ids']:raise ValueError('actual EOS mismatch before generation')
   if kwargs.get('stopping_criteria') is not None:raise ValueError('unexpected caller stopping criteria')
   kwargs['stopping_criteria']=criteria_type([boundary])
   result=original(*args,**kwargs)
   ids=result.detach().cpu().tolist()[0]
   # This is deliberately after first-token timing. It is inside the final
   # end boundary and therefore that end-to-end field includes journal cost.
   on_raw({'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'budget':budget,'output_ids':ids,'generation_scalars':expected,'eos_ids':eos})
   seen.update(ids=ids,eos=eos)
   return result
  wrapper.model.generate=generate
  try:return author_generate(wrapper,row)
  finally:wrapper.model.generate=original
 with adapter_type(torch,wrapper,native,row['image_start'],budget) as adapter:
  answer,times=measure(invoke,torch.cuda.synchronize)
  if not adapter.state.get('prefill_seen') or adapter.state.get('greedy_calls')!=(0 if budget==1024 else 1):raise ValueError('actual method path incomplete')
 ids=seen.get('ids');eos=seen.get('eos')
 if not ids or times.get('first_criteria_length')!=1 or times.get('criteria_calls')!=len(ids):raise ValueError('first selected token or decode count invalid')
 if ids!=accepted['output_ids'] or eos!=accepted['eos_ids'] or answer!=accepted['answer'] or (ids[-1] in eos)!=accepted['eos'] or (ids[-1] not in eos and len(ids)>=2048)!=accepted['truncated']:raise ValueError('same-arm accepted natural output mismatch')
 return {'budget':budget,'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'output_ids':ids,'answer':answer,
         'eos_ids':eos,'eos':ids[-1] in eos,'truncated':accepted['truncated'],'generation_scalars':expected,'boundaries':times,
         'selector_calls':adapter.state['greedy_calls'],'natural':True,'TTFT_scope':'before image read to selected first token with boundary synchronization',
         'completion_time_scope':'includes post-generation raw output fsync; not clean end-to-end latency'}
