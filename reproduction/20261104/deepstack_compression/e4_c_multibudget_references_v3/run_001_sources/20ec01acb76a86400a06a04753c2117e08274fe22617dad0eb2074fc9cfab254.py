"""Capture only prefill KV metadata for bulk fixed32 reference generation.

Thirty-one subsequent decode forwards are captured by DecodeObserver as
positions and all-layer lengths, not KV content/fingerprints. The separate
12-pair fixed32 identity diagnostic captures fingerprints at every step.
"""
class FirstCacheFingerprint:
 def __init__(self,wrapper):self.wrapper=wrapper;self.fingerprints=[]
 def __enter__(self):
  def observe(_module,_args,_kwargs,out):
   if self.fingerprints:return
   from identity_records import tensor_meta
   cache=out.past_key_values
   if cache is None or len(cache.layers)!=36:raise ValueError('expected actual 36-layer cache')
   self.fingerprints.append([{'key':tensor_meta(layer.keys),'value':tensor_meta(layer.values)} for layer in cache.layers])
  self.handle=self.wrapper.model.model.language_model.register_forward_hook(observe,with_kwargs=True);return self
 def __exit__(self,*_args):self.handle.remove()
