"""Independently recompute diagnostic outcomes from saved records and refs."""
import hashlib,struct
RECIPE={'do_sample':False,'max_new_tokens':2048,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0}
def check(record,raw,ref,decode):
 arm=record['arm'];out=record['output_ids'];n=record['input_length'];full=raw['full_sequence']
 if arm not in ('rres','eadp') or len(full)!=1 or not out or any(type(x) is not int for x in full[0]):raise ValueError('invalid raw identity')
 if (full[0][n:] if arm=='rres' else full[0])!=out:raise ValueError('raw continuation')
 for k in ('task','qid','group_id','image_sha256'):
  if str(record[k])!=str(ref[k]):raise ValueError('input identity '+k)
 if record['processor']!=ref['processor'] or out!=ref['output_ids'] or record['response']!=ref['answer']:raise ValueError('processor/output identity')
 if record['visual']!=ref['main_visual'] or record['deepstack']!=ref['deepstack' if arm=='rres' else 'full_deepstack'] or record['injected_deepstack']!=ref['injected_ds' if arm=='rres' else 'injected_deepstack']:raise ValueError('visual/DS')
 if record['prefill_inputs_embeds']!=ref['prefill_inputs_embeds' if arm=='rres' else 'prefill_embeds'] or record['prefill_attention_mask']!=ref['prefill_attention_mask' if arm=='rres' else 'prefill_mask']:raise ValueError('prefill')
 if record['prefill_logits_reference_compared'] is not (arm=='rres') or (arm=='rres' and record['prefill_logits']!=ref['prefill_logits']):raise ValueError('logits evidence')
 counts=record['visual_execution_counts']
 if counts['block_calls']!=ref['block_calls'] or counts['merger_calls']!=ref['merger_calls' if arm=='rres' else 'main_merger_calls'] or counts['ds_calls']!=ref['ds_calls']:raise ValueError('visual execution counts')
 pos=record['prefill_position_ids'];delta=record['delta'];mask=record['prefill_visual_mask'];length=n if arm=='rres' else n-768
 if len(pos)!=3 or any(len(axis)!=1 or len(axis[0])!=length for axis in pos):raise ValueError('position shape')
 flat=[x for axis in pos for batch in axis for x in batch]
 if any(type(x) is not int for x in flat):raise ValueError('position type')
 posmeta={'dtype':'torch.int64','shape':[3,1,length],'sha256':hashlib.sha256(struct.pack('<'+'q'*len(flat),*flat)).hexdigest()}
 if posmeta!=record['prefill_position_meta']:raise ValueError('position bytes')
 if arm=='rres':
  expected_mask=[[ref.get('image_start',0)<=i<ref.get('image_start',0)+256 for i in range(n)]]
  # Full-panel R-res refs need image_start supplied by separately bound input reference.
  if posmeta!=ref['prefill_position_ids'] or delta!=ref['delta'] or mask!=expected_mask:raise ValueError('rres positions/mask')
 else:
  if pos!=ref['prefill_position_ids'] or delta!=ref['adjusted_delta'] or mask!=ref['injected_mask'] or record['ordered_ids']!=ref['ordered_ids']:raise ValueError('eadp positions/mask/selection')
 decode.validate_trace(record['decode_trace'],out,length,int(delta[0][0]))
 if record['first_criteria_length']!=(n+1 if arm=='rres' else 1) or record['criteria_calls']!=len(out):raise ValueError('actual token boundary')
 obs=record['generation_observation'];kw=obs['explicit_kwargs'];config=obs['effective_generation_config']
 if obs['prepare_calls']!=1 or any(kw.get(k)!=v or config.get(k)!=v for k,v in RECIPE.items()):raise ValueError('actual runtime generation config')
 eos=config['eos_token_id'];eos=[eos] if type(eos) is int else list(eos or [])
 if eos!=ref['eos_ids'] or record['checks'].get('hooks_and_methods_restored') is not True or record['status']!='passed' or record['errors'] or not all(record['checks'].values()) or record['timing_claims'] is not False:raise ValueError('EOS/hooks/status')
 return True
