"""One token of a fresh request on an already-loaded model; not a GPU entry.

Identical selection/KV observation runs in both arms. Cache-position scalar
copy synchronizes inside TTFT; processor and index byte checks occur afterward.
"""
import contextlib
from pathlib import Path
RECIPE={'do_sample':False,'max_new_tokens':1,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0,'num_beams':1,'use_cache':True}
@contextlib.contextmanager
def selection(selector,candidate,arm):
 if arm not in ('author','wavefront'):raise ValueError('arm')
 old=selector._select_2d_loco_anchored;seen=[]
 def call(self,tokens,height,width,device):
  if self.method!='max' or self.threshold!=.2 or self.anchored is not True:raise ValueError('selector dispatch')
  value=old(self,tokens,height,width,device) if arm=='author' else candidate(tokens,height,width,self.threshold)
  seen.append((value,value.detach().clone()));return value
 selector._select_2d_loco_anchored=call
 try:
  yield seen
  if len(seen)!=1 or not seen[0][0].equal(seen[0][1]):raise ValueError('selection call count or mutation')
 finally:selector._select_2d_loco_anchored=old

def cache_before(kwargs):
 cache=kwargs.get('past_key_values');position=kwargs.get('cache_position')
 if position is None or position.numel()<1 or int(position.reshape(-1)[0].item())!=0:raise ValueError('prefill cache position')
 if cache is None:lengths=[0]*36;kind='none'
 else:
  if not callable(getattr(cache,'get_seq_length',None)):raise ValueError('unsupported cache')
  lengths=[int(cache.get_seq_length(i)) for i in range(36)];kind=type(cache).__name__
 if any(x!=0 for x in lengths):raise ValueError('nonempty initial KV cache')
 return {'initial_kv_lengths':lengths,'initial_cache_kind':kind,'initial_cache_position':0,'cache_position_check_synchronizes':True}

def execute(wrapper,patch,torch,row,reference,selector,candidate,arm,base,timer,validate,emit_raw):
 """Run exactly one prompt; caller handles verified process, model and output."""
 if not callable(validate) or not callable(emit_raw):raise ValueError('missing durable output or input validator')
 model=wrapper.model;inner=model.model
 if not hasattr(inner,'rope_deltas'):raise ValueError('missing Qwen rope state')
 inner.rope_deltas=None;captured={};pre=[];selector_original=selector._select_2d_loco_anchored
 def cache_hook(module,args,kwargs):
  if pre:raise ValueError('more than one prefill for one-token generation')
  pre.append(cache_before(kwargs))
 handle=inner.register_forward_pre_hook(cache_hook,with_kwargs=True)
 try:
  from transformers.generation.stopping_criteria import StoppingCriteriaList
  def processor(original):
   def call(*args,**kw):
    result=original(*args,**kw)
    if 'inputs' in captured:raise ValueError('multiple processor calls')
    captured['inputs']=result;captured['prompt_length']=int(result['input_ids'].shape[1]);return result
   return call
  def invoke(boundary):
   def generation(original):
    def call(*args,**kw):
     if 'full' in captured or kw.get('past_key_values') is not None or kw.get('cache_position') is not None:raise ValueError('unexpected cached generation')
     scalar={k:kw.get(k) for k in RECIPE if k not in ('num_beams','use_cache')}
     if scalar!={k:v for k,v in RECIPE.items() if k not in ('num_beams','use_cache')}:raise ValueError('generation recipe')
     for k in ('num_beams','use_cache'):
      if k in kw and kw[k]!=RECIPE[k]:raise ValueError('generation mode')
      kw[k]=RECIPE[k]
     if kw.get('stopping_criteria') is not None:raise ValueError('unexpected stopping criteria')
     kw['stopping_criteria']=StoppingCriteriaList([boundary]);eos=kw.get('eos_token_id',model.generation_config.eos_token_id);captured['eos_ids']=[eos] if type(eos) is int else list(eos or [])
     result=original(*args,**kw);captured['full']=result.detach().cpu().tolist();emit_raw(captured['full']);return result
    return call
   with base.replaced(model,'generate',generation):return wrapper.generate_inner_transformers_fixed(row['message'],dataset=row['task'])
  with selection(selector,candidate,arm) as chosen:
   with base.count_calls(patch,'compute_merged_keep_indices','pixel',1) as counts:
    with base.replaced(wrapper,'_processor_inputs',processor):
     with torch.inference_mode():answer,boundaries=timer.measure(invoke,torch.cuda.synchronize)
  if len(pre)!=1 or len(chosen)!=1 or len(counts)!=1 or len(captured.get('full',[]))!=1:raise ValueError('request observation incomplete')
  ids=chosen[0][1].cpu().tolist();n=captured['prompt_length'];full=captured['full'][0]
  if len(ids)!=counts[0] or ids!=sorted(set(ids)) or not 1<=len(ids)<=1024 or any(type(i) is not int or not 0<=i<1024 for i in ids):raise ValueError('selected index invalid')
  if len(full)!=n+1 or captured['inputs']['input_ids'].detach().cpu().tolist()[0]!=full[:n] or boundaries['first_criteria_length']!=n+1:raise ValueError('raw prompt or first token')
  if not captured['eos_ids'] or any(type(x) is not int for x in captured['eos_ids']):raise ValueError('EOS ids')
  actual=validate(torch,captured['inputs'],reference['processor'],model.config.image_token_id,reference['image_start'])
  if actual['processor']!=reference['processor']:raise ValueError('processor mismatch')
  result={'status':'completed','arm':arm,'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'processor':actual['processor'],'selected_indices':ids,'selected_count':len(ids),'first_token_id':full[-1],'eos_ids':captured['eos_ids'],'first_token_is_eos':full[-1] in captured['eos_ids'],'input_length':n,'recipe':dict(RECIPE),'boundaries':boundaries,'cache_check':pre[0],'rope_state_reset_before_request':True,'response':answer}
 finally:
  handle.remove()
  if handle.id in inner._forward_pre_hooks:raise ValueError('prefill hook not removed')
 if selector._select_2d_loco_anchored is not selector_original:raise ValueError('selector not restored')
 result['hook_cleanup_verified']=True
 return result
