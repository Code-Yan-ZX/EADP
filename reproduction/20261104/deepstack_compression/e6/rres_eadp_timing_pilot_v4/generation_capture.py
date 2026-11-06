"""In-memory capture of the kwargs and HF's actually prepared config.

Tensor values are described, not copied: this is not an input archive.  Invoke
the real generate inside capture and persist the record only after timing ends.
"""
from contextlib import contextmanager
import math

def describe(value):
 if value is None or type(value) in (str,bool,int): return value
 if type(value) is float:
  if not math.isfinite(value): raise ValueError('nonfinite generation argument')
  return value
 if isinstance(value,(list,tuple)): return [describe(v) for v in value]
 if isinstance(value,dict):
  if any(type(k) is not str for k in value): raise ValueError('non-string key')
  return {k:describe(v) for k,v in value.items()}
 if hasattr(value,'shape') and hasattr(value,'dtype'):
  return {'kind':'tensor','shape':list(value.shape),'dtype':str(value.dtype),'device':str(value.device)}
 if hasattr(value,'to_dict'): return {'kind':'configuration','class':type(value).__module__+'.'+type(value).__qualname__,'value':describe(value.to_dict())}
 return {'kind':'object','class':type(value).__module__+'.'+type(value).__qualname__}

@contextmanager
def capture(model, kwargs):
 """Capture the real HF config preparation exactly once; restore on any exit."""
 if '_prepare_generation_config' in model.__dict__: raise ValueError('instance config preparation override')
 original=model._prepare_generation_config
 result={'explicit_kwargs':describe(kwargs),'effective_generation_config':None,'prepare_calls':0}
 def wrapped(*args,**kw):
  result['prepare_calls']+=1
  if result['prepare_calls'] != 1: raise RuntimeError('unexpected generation config preparation count')
  config,remaining=original(*args,**kw)
  result['effective_generation_config']=describe(config.to_dict())
  return config,remaining
 model._prepare_generation_config=wrapped
 try:
  yield result
  if result['prepare_calls']!=1 or result['effective_generation_config'] is None: raise RuntimeError('generation config was not prepared')
 finally:
  del model._prepare_generation_config
