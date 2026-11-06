"""Last-boundary fixed32 controls; no model loading or GPU authority."""
NAMES=('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty')

class CacheFingerprints:
 """Capture actual returned KV per layer, including prefill and every decode."""
 def __init__(self,wrapper):self.wrapper=wrapper;self.fingerprints=[]
 def __enter__(self):
  def observe(_module,_args,_kwargs,out):
   from identity_records import tensor_meta
   cache=out.past_key_values
   if cache is None or len(cache.layers)!=36:raise ValueError('expected actual 36-layer cache')
   self.fingerprints.append([{'key':tensor_meta(layer.keys),'value':tensor_meta(layer.values)} for layer in cache.layers])
  self.handle=self.wrapper.model.model.language_model.register_forward_hook(observe,with_kwargs=True)
  return self
 def __exit__(self,*_args):self.handle.remove()
def verify_kwargs(kwargs,eos,accepted_panel):
 if not isinstance(accepted_panel,dict):raise ValueError('missing accepted E4 same-arm recipe')
 original=accepted_panel.get('generation_scalars')
 if not isinstance(original,dict) or set(original)!=set(NAMES) or original['do_sample'] is not False or original['max_new_tokens']!=2048:raise ValueError('invalid accepted recipe')
 expected=dict(original,max_new_tokens=32)
 if {name:kwargs.get(name) for name in NAMES}!=expected or kwargs.get('min_new_tokens')!=32:raise ValueError('fixed32 actual recipe mismatch before model.generate')
 if not isinstance(eos,list) or not eos or eos!=accepted_panel.get('eos_ids'):raise ValueError('fixed32 actual EOS mismatch before model.generate')
 if kwargs.get('stopping_criteria') is not None or kwargs.get('logits_processor') is not None:raise ValueError('fixed32 unexpected generation control')
 return True
