"""One real Qwen request. This module has no CLI and grants no GPU access."""
from contextlib import ExitStack,nullcontext
import generation_capture,paired_runner as runner
from pathlib import Path
if Path(runner.__file__).resolve()!=Path(__file__).with_name('paired_runner.py').resolve() or Path(runner.measure.__code__.co_filename).resolve()!=Path(__file__).with_name('paired_runner.py').resolve() or Path(runner.normalize.__code__.co_filename).resolve()!=Path(__file__).with_name('paired_runner.py').resolve():raise ValueError('foreign paired runner')

RECIPE={'do_sample':False,'max_new_tokens':2048,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0}

def request(*,arm,torch,wrapper,module,pruner,row,accepted,adapter_type,criteria_type,fixed_resolution,author_generate,persist):
 """Caller must first independently admit the model, input and output references.

 persist(raw) must durably save the *whole* raw tensor after timing boundaries,
 including on errors; this function itself must never be used as a GPU gate.
 """
 if arm not in ('eadp','rres') or not callable(persist):raise ValueError('invalid arm or persistence')
 for key in ('task','group_id','image_sha256','message_sha256'):
  if row.get(key)!=accepted.get(key):raise ValueError('accepted input mismatch: '+key)
 if str(row.get('qid'))!=str(accepted.get('qid')):raise ValueError('accepted qid mismatch')
 if accepted.get('generation_scalars')!=RECIPE or accepted.get('eos_ids')!=[151645,151643]:raise ValueError('accepted generation recipe')
 if wrapper.model.model.rope_deltas is not None:raise ValueError('cross-request rope state must be reset by caller')
 model=wrapper.model;original_generate=model.generate;original_processor=wrapper._processor_inputs
 if 'generate' in vars(model) or '_processor_inputs' in vars(wrapper):raise ValueError('unexpected preexisting instance override')
 state={'processor_calls':0,'generate_calls':0};sync=torch.cuda.synchronize
 def processor(*args,**kwargs):
  state['processor_calls']+=1
  if state['processor_calls']!=1:raise ValueError('unexpected processor count')
  with fixed_resolution(module,512):inputs=original_processor(*args,**kwargs)
  state['inputs']=inputs
  return inputs
 def invoke(boundary):
  def generate(*args,**kwargs):
   state['generate_calls']+=1
   if state['generate_calls']!=1:raise ValueError('unexpected generation count')
   if any(k not in kwargs or type(kwargs[k]) is not type(v) or kwargs[k]!=v for k,v in RECIPE.items()):raise ValueError('actual runtime kwargs mismatch')
   eos=kwargs.get('eos_token_id',model.generation_config.eos_token_id);eos=[eos] if type(eos) is int else list(eos or [])
   if eos!=accepted['eos_ids'] or kwargs.get('stopping_criteria') is not None or kwargs.get('logits_processor') is not None:raise ValueError('unexpected runtime generation controls')
   kwargs['stopping_criteria']=criteria_type([boundary])
   with generation_capture.capture(model,kwargs) as observed:
    observed['explicit_positional_args']=generation_capture.describe(args)
    state['generation_observation']=observed
    raw=original_generate(*args,**kwargs)
    boundary.generation_returned(raw)
   return raw
  model.generate=generate
  try:return author_generate(wrapper,row)
  finally:del model.generate
 context=adapter_type(torch,wrapper,pruner,row['image_start'],256) if arm=='eadp' else nullcontext()
 with ExitStack() as stack:
  if arm=='rres':
   wrapper._processor_inputs=processor;stack.callback(delattr,wrapper,'_processor_inputs')
  adapter=stack.enter_context(context)
  response,times=runner.measure(invoke,sync,persist=persist)
  if arm=='eadp':
   if not adapter.state.get('prefill_seen') or adapter.state.get('greedy_calls')!=1:raise ValueError('EADP path missing')
   state['inputs']=adapter.state['inputs']
  elif state['processor_calls']!=1:raise ValueError('R-res processor missing')
 if model.generate!=original_generate or wrapper._processor_inputs!=original_processor or module.QWEN3_FIXED_RESOLUTION!=1024:raise ValueError('request hook restoration')
 raw=state['generation_observation'];effective=raw['effective_generation_config']
 if any(effective.get(k)!=v for k,v in RECIPE.items()):raise ValueError('actual effective generation config mismatch')
 effective_eos=effective.get('eos_token_id');effective_eos=[effective_eos] if type(effective_eos) is int else list(effective_eos or [])
 if effective_eos!=accepted['eos_ids']:raise ValueError('actual effective EOS mismatch')
 return {'response':response,'times':times,'generation':raw,'processor_inputs':state['inputs'],'output_match_pending':True}
