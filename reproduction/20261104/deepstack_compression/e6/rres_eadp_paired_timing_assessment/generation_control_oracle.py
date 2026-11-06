"""Independent strict generation controls for paired performance auditing."""
EXPECTED={'do_sample':False,'max_new_tokens':2048,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0,'num_beams':1,'num_return_sequences':1,'use_cache':True}
def validate(observation,expected_eos):
 if observation.get('prepare_calls')!=1:raise ValueError('HF configuration count')
 cfg=observation['effective_generation_config'];kw=observation['explicit_kwargs']
 for key,value in EXPECTED.items():
  if type(cfg.get(key)) is not type(value) or cfg.get(key)!=value:raise ValueError('effective '+key)
  if key in ('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty') and (type(kw.get(key)) is not type(value) or kw.get(key)!=value):raise ValueError('explicit '+key)
 eos=cfg.get('eos_token_id');eos=[eos] if type(eos) is int else eos
 if eos!=expected_eos:raise ValueError('EOS')
 return True
