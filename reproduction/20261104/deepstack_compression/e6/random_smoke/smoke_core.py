"""Single random K256 arm core, no model loader or GPU authority.

Caller must independently verify approved E1 references and supply inspected
source-pinned budget_runner and RandomAdapter before invoking this function.
"""
from contextlib import nullcontext
import re
SEEDS=(20261104,20261105,20261106)
def run_arm(torch,wrapper,native,sample,seed,e1,selection,budget_runner,random_adapter,generate,on_generated,archive_arrays,validate_identity,cache_factory,forced=False):
 if sample.get('split')!='DEV' or type(seed)!=int or seed not in SEEDS:raise ValueError('random arm scope')
 for key in ('task','group_id','message_sha256'):
  if e1.get(key)!=sample.get(key):raise ValueError('E1 input mismatch')
 if str(e1.get('qid'))!=str(sample.get('qid')) or e1.get('arm')!='P3-D1' or e1.get('forced') is not False:raise ValueError('E1 baseline mismatch')
 names=('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty');expected=e1.get('generation_scalars');eos=e1.get('eos_ids');greedy=e1.get('ordered_ids')
 if not isinstance(expected,dict) or set(expected)!=set(names) or expected.get('do_sample') is not False or expected.get('max_new_tokens')!=2048 or not isinstance(eos,list) or not eos or not isinstance(greedy,list) or len(greedy)!=256:raise ValueError('E1 controls/greedy missing')
 selected=selection(sample['image_sha256'],seed)
 if len(selected['spatial_ids'])!=256 or selected['spatial_ids']!=sorted(set(selected['spatial_ids'])):raise ValueError('bad random selection')
 def factory(t,w,p,budget,ds_on):
  if budget!=256 or ds_on is not True:raise ValueError('random adapter only supports K256 DS111')
  return random_adapter(t,w,p,sample['image_sha256'],seed,greedy)
 def guard_generate(w,row):
  original=w.model.generate
  def checked(*args,**kwargs):
   actual={name:kwargs.get(name) for name in names};actual_eos=kwargs.get('eos_token_id',w.model.generation_config.eos_token_id);actual_eos=[actual_eos] if isinstance(actual_eos,int) else list(actual_eos or [])
   if actual!=expected or actual_eos!=eos or kwargs.get('min_new_tokens') is not None or kwargs.get('logits_processor') is not None or kwargs.get('stopping_criteria') is not None:raise ValueError('actual generation controls differ from E1')
   return original(*args,**kwargs)
  had='generate' in vars(w.model);stored=vars(w.model).get('generate');w.model.generate=checked
  try:return generate(w,row)
  finally:
   if had:w.model.generate=stored
   else:delattr(w.model,'generate')
 def journal(raw):on_generated({**raw,'seed':seed,'protocol':'forced33' if forced else 'natural'})
 if not hasattr(budget_runner,'BudgetAdapter') or not hasattr(budget_runner,'run_arm'):raise ValueError('bad diagnostics runner')
 prior=budget_runner.BudgetAdapter;budget_runner.BudgetAdapter=factory
 try:
  # Forced diagnostics save actual cache fingerprints; natural generation
  # keeps actual position and 36-layer lengths without quadratic hash I/O.
  with cache_factory(wrapper) if forced else nullcontext(None) as kv:
   result=budget_runner.run_arm(torch,wrapper,native,sample,256,True,guard_generate,journal,archive_arrays,forced_token=100 if forced else None,capture_prefill_logits=True)
  if forced:
   fingerprints=kv.fingerprints
   if len(fingerprints)!=33 or any(len(step)!=36 for step in fingerprints):raise ValueError('forced KV fingerprint inventory')
   length=len(result['retained_ids'])
   for i,step in enumerate(fingerprints):
    for layer in step:
     if set(layer)!={'key','value'}:raise ValueError('forced KV pair')
     for value in layer.values():
      if (not isinstance(value,dict) or set(value)!={'dtype','shape','sha256'} or value['dtype']!='torch.bfloat16' or value['shape']!=[1,8,length+i,128] or not isinstance(value['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',value['sha256'])):raise ValueError('forced KV fingerprint invalid')
   result['cache_fingerprints']=fingerprints
 finally:budget_runner.BudgetAdapter=prior
 validate_identity(result,sample,e1)
 if result.get('selection_mode')!='e6_random_fixed_seed' or result.get('ordered_ids')!=selected['spatial_ids'] or result.get('spatial_ids')!=selected['spatial_ids']:raise ValueError('actual random selection differs from frozen choice')
 ids=result.get('output_ids',[])
 if forced:
  if ids!=[100]*33 or result.get('eos_ids')!=[] or len(result.get('decode_trace',[]))!=33:raise ValueError('forced33 incomplete')
 elif not ids or (ids[-1] not in eos and len(ids)!=2048):raise ValueError('natural completion invalid')
 return {**result,'seed':seed,'derived_seed64':selected['derived_seed64'],'protocol':'forced33' if forced else 'natural','extra_author_greedy':True,'timing_eligible':False}
