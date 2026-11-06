"""Unadmitted four-budget natural/fixed32 timing primitive; no GPU authority."""
import os,time
class FirstToken:
 def __init__(self,synchronize,clock):self.sync=synchronize;self.clock=clock;self.first_ns=None;self.first_length=None;self.calls=0
 def __call__(self,input_ids,scores,**_kwargs):
  self.calls+=1
  if self.first_ns is None:self.sync();self.first_ns=self.clock();self.first_length=int(input_ids.shape[1])
  return False

def execute(torch,wrapper,native,row,budget,protocol,accepted,adapter_type,criteria_type,author_generate,on_raw,validate_generate,clock=time.perf_counter_ns):
 """Image read through synchronized generate-return, before decode/journaling.

 T0 is synchronized and before author message/image input. First token is
 recorded after token append and sync. Completion is recorded immediately
 after generate returns and a final sync, before copying tokens or fsync.
 """
 if protocol not in ('natural','fixed32') or budget not in (128,256,512,1024) or row.get('split')!='DEV' or type(row.get('image_start')) is not int:raise ValueError('invalid timing protocol/scope')
 if accepted.get('budget')!=budget or accepted.get('ds_on') is not True or str(accepted.get('qid'))!=str(row.get('qid')):raise ValueError('accepted endpoint mismatch')
 for key in ('task','group_id','message_sha256','image_sha256'):
  if row.get(key)!=accepted.get(key):raise ValueError('input identity mismatch')
 validate_generate(wrapper.model);wrapper.visual_token_num=budget;wrapper.pruner.visual_token_num=budget
 names=('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty');expected=dict(accepted['generation_scalars'])
 if set(expected)!=set(names) or expected['do_sample'] is not False or expected['max_new_tokens']!=2048:raise ValueError('bad accepted generation recipe')
 if protocol=='fixed32':expected['max_new_tokens']=32
 original=wrapper.model.generate;had='generate' in vars(wrapper.model);stored=vars(wrapper.model).get('generate');seen={}
 with adapter_type(torch,wrapper,native,row['image_start'],budget) as adapter:
  torch.cuda.synchronize();t0=clock();boundary=FirstToken(torch.cuda.synchronize,clock)
  def capture(*args,**kwargs):
   if seen:raise ValueError('multiple model generations')
   if protocol=='fixed32':
    if kwargs.get('max_new_tokens')!=2048 or kwargs.get('min_new_tokens') is not None:raise ValueError('unexpected original fixed32 recipe before override')
    kwargs.update(min_new_tokens=32,max_new_tokens=32)
   if {name:kwargs.get(name) for name in names}!=expected or kwargs.get('min_new_tokens')!=(32 if protocol=='fixed32' else None):raise ValueError('actual recipe mismatch before generate')
   eos=kwargs.get('eos_token_id',wrapper.model.generation_config.eos_token_id);eos=[eos] if isinstance(eos,int) else list(eos or [])
   if eos!=accepted['eos_ids']:raise ValueError('actual EOS mismatch before generate')
   if kwargs.get('stopping_criteria') is not None or kwargs.get('logits_processor') is not None:raise ValueError('unexpected generation controls')
   kwargs['stopping_criteria']=criteria_type([boundary])
   result=original(*args,**kwargs)
   # No token materialization, log write, answer decoding, or bookkeeping may
   # precede this completion boundary. The preceding sync is intentional.
   try:
    torch.cuda.synchronize();finished=clock()
   except Exception as error:
    # A completed generate must remain observable even when its timing
    # boundary fails. This record is a failure, not a usable measurement.
    ids=result.detach().cpu().tolist()[0]
    on_raw({'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'budget':budget,'protocol':protocol,
            'output_ids':ids,'eos_ids':eos,'generation_scalars':expected,'min_new_tokens':32 if protocol=='fixed32' else None,
            'generation_complete_ns':None,'boundary_error':type(error).__name__+': '+str(error)})
    raise
   ids=result.detach().cpu().tolist()[0]
   on_raw({'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'budget':budget,'protocol':protocol,
           'output_ids':ids,'eos_ids':eos,'generation_scalars':expected,'min_new_tokens':32 if protocol=='fixed32' else None,
           'generation_complete_ns':finished})
   seen.update(ids=ids,eos=eos,finished=finished)
   return result
  wrapper.model.generate=capture
  try:answer=author_generate(wrapper,row)
  finally:
   if had:wrapper.model.generate=stored
   else:delattr(wrapper.model,'generate')
  if not adapter.state.get('prefill_seen') or adapter.state.get('greedy_calls')!=(0 if budget==1024 else 1):raise ValueError('actual selector/P3 method path incomplete')
 ids=seen.get('ids');eos=seen.get('eos');finished=seen.get('finished')
 if not ids or boundary.first_ns is None or not t0<boundary.first_ns<=finished or boundary.first_length!=1 or boundary.calls!=len(ids):raise ValueError('actual token/time boundary invalid')
 if protocol=='fixed32' and (len(ids)!=32 or any(x in eos for x in ids[:-1])):raise ValueError('fixed32 length or early EOS mismatch')
 if protocol=='natural':
  if ids!=accepted['output_ids'] or answer!=accepted['answer'] or (ids[-1] in eos)!=accepted['eos'] or (ids[-1] not in eos and len(ids)>=2048)!=accepted['truncated']:raise ValueError('natural same-arm output mismatch')
 return {'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'budget':budget,'protocol':protocol,'output_ids':ids,'answer':answer,
         'eos_ids':eos,'eos':ids[-1] in eos,'truncated':protocol=='natural' and ids[-1] not in eos and len(ids)>=2048,
         'generation_scalars':expected,'min_new_tokens':32 if protocol=='fixed32' else None,'selector_calls':adapter.state['greedy_calls'],
         'boundaries':{'t0_before_image_read_ns':t0,'first_token_selected_ns':boundary.first_ns,'generation_complete_ns':finished,
          'first_criteria_length':boundary.first_length,'criteria_calls':boundary.calls,'clock_implementation':time.get_clock_info('perf_counter').implementation,
          'time_namespace':os.readlink('/proc/self/ns/time'),'explicit_sync_boundaries':['before_t0','first_token','generate_complete']},
         'completion_scope':'image read through synchronized model.generate return; excludes raw journal fsync and answer text decode'}
