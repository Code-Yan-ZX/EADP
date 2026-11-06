"""Capture actual generate overrides and resolved scalar generation settings."""
KEYS=('do_sample','max_new_tokens','min_new_tokens','top_p','top_k','temperature','repetition_penalty',
      'eos_token_id','pad_token_id','forced_eos_token_id','forced_bos_token_id','use_cache','num_beams')
def controls(config,kwargs,forced):
 result={k:kwargs[k] if k in kwargs else getattr(config,k,None) for k in KEYS}
 for k,v in result.items():
  if not (v is None or type(v) in (int,float,bool,str) or isinstance(v,list) and all(type(x) is int for x in v)):
   raise ValueError('unsupported actual generate setting '+k)
 if forced and (result['min_new_tokens']!=33 or result['eos_token_id']!=[] or not kwargs.get('logits_processor')):
  raise ValueError('forced prefix controls missing')
 if not forced and kwargs.get('logits_processor'):
  raise ValueError('unexpected natural custom logits processor')
 return result
