"""CPU-only comparison of separately captured fixed32 diagnostic metadata.

This does not capture model internals.  A future independently reviewed worker
must establish that the recorded metadata describes actual model inputs.
"""
import re
HEX=re.compile(r'^[0-9a-f]{64}$')
def fingerprint(x):
 if not isinstance(x,dict) or set(x)!={'dtype','shape','sha256'} or not isinstance(x['dtype'],str) or not isinstance(x['shape'],list) or any(type(d)!=int or d<0 for d in x['shape']) or not HEX.fullmatch(x['sha256']):raise ValueError('invalid actual tensor fingerprint')

def validate(record):
 k=record['budget'];ids=record['ordered_ids'];spatial=record['spatial_ids'];n=record['compact_prefill_length'];original=record['original_prefill_length']
 if k not in (128,256,512,1024) or len(ids)!=k or len(set(ids))!=k or any(type(x)!=int or not 0<=x<1024 for x in ids) or spatial!=sorted(ids):raise ValueError('invalid budget selection')
 if k==1024 and record['greedy_calls']!=0 or k!=1024 and record['greedy_calls']!=1:raise ValueError('wrong author selection path')
 if type(n)!=int or type(original)!=int or n!=original-1024+k or n<1:raise ValueError('invalid compact length')
 if record['protocol']!='fixed32' or record['generation_scalars']!={'do_sample':False,'min_new_tokens':32,'max_new_tokens':32,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0} or not record['eos_ids'] or len(record['output_ids'])!=32 or any(not isinstance(x,int) for x in record['output_ids']) or any(x in record['eos_ids'] for x in record['output_ids'][:-1]):raise ValueError('fixed32 generation mismatch')
 for key in ('input_ids','attention_mask','pixel_values','image_grid_thw','prefill_embeds','prefill_mask','prefill_position_ids','ds0','ds1','ds2'):
  fingerprint(record['tensors'][key])
 trace=record['decode_trace']
 if len(trace)!=31:raise ValueError('fixed32 requires 31 decode forwards')
 for i,step in enumerate(trace):
  if step['step']!=i or step['cache_before']!=n+i or step['cache_after']!=n+i+1 or len(step['layer_cache_after'])!=36 or any(v!=n+i+1 for v in step['layer_cache_after']):raise ValueError('decode/KV sequence mismatch')
  if step['cache_position']!=original+i or not isinstance(step['position_ids'],list) or len(step['position_ids'])!=3 or any(type(v)!=int for v in step['position_ids']):raise ValueError('decode position shape/cache mismatch')
  if step['rope_delta']!=record['adjusted_delta']:raise ValueError('decode rope mismatch')
 return True

def compare(reference,light):
 validate(reference);validate(light)
 keys=('task','qid','group_id','image_sha256','message_sha256','budget','ordered_ids','spatial_ids','greedy_calls','original_prefill_length','compact_prefill_length','protocol','generation_scalars','eos_ids','output_ids','tensors','adjusted_delta','decode_trace')
 if any(reference[k]!=light[k] for k in keys):raise ValueError('actual reference/light identity mismatch')
 return True
