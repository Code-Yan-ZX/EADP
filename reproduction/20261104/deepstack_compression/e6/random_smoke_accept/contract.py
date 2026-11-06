"""CPU-only random-smoke record and high-dimensional archive checks.

Call only after full source/exit acceptance and 18-arm inventory completion.
No GT, model, scoring or GPU access. KV contents remain fingerprints.
"""
import hashlib,re
import numpy as np

def tensor_bytes(arr):return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
def validate_record(r,choice,e1,arrays):
 for name in ('task','qid','group_id','message_sha256','image_sha256'):
  if str(r.get(name))!=str(choice.get(name)):raise ValueError('identity '+name)
 for name in ('task','qid','group_id','message_sha256'):
  if str(r.get(name))!=str(e1.get(name)):raise ValueError('E1 identity '+name)
 if r.get('budget')!=256 or r.get('ds_on') is not True or r.get('selection_mode')!='e6_random_fixed_seed' or r.get('greedy_calls')!=1 or r.get('actual_injection_verified') is not True or r.get('extra_author_greedy') is not True or r.get('timing_eligible') is not False:raise ValueError('random method identity')
 if r.get('seed')!=choice.get('seed') or r.get('derived_seed64')!=choice.get('derived_seed64'):raise ValueError('seed identity')
 ids=r.get('ordered_ids');sp=r.get('spatial_ids')
 if not isinstance(ids,list) or len(ids)!=256 or ids!=sorted(set(ids)) or any(type(i) is not int or not 0<=i<1024 for i in ids) or sp!=ids or ids!=choice.get('spatial_ids'):raise ValueError('random ordered selection')
 for actual,reference in (('processor','processor'),('main_visual','visual_sha256'),('full_deepstack','deepstack_full_sha256'),('full_position_ids','original_position_ids'),('full_delta','full_rope_delta')):
  if r.get(actual)!=e1.get(reference):raise ValueError('E1 full input/visual mismatch '+actual)
 start=r['image_start'];n=r['original_length'];ret=list(range(start))+[start+i for i in ids]+list(range(start+1024,n));pos=r['full_position_ids']
 if r.get('retained_ids')!=ret or len(pos)!=3 or any(len(axis)!=1 or len(axis[0])!=n for axis in pos) or r.get('prefill_position_ids')!=[[[axis[0][i] for i in ret]] for axis in pos]:raise ValueError('retained/P3 positions')
 delta=max(max(axis[0]) for axis in pos)+1-n
 if r.get('full_delta')!=[[delta]] or r.get('adjusted_delta')!=[[delta+768]]:raise ValueError('rope delta')
 mask=[False]*len(ret);mask[start:start+256]=[True]*256
 if r.get('injected_mask')!=[mask] or len(r.get('full_deepstack',[]))!=3 or len(r.get('selected_deepstack',[]))!=3 or len(r.get('injected_deepstack',[]))!=3:raise ValueError('DS inventory')
 required={'visual','prefill','prefill_logits',*[f'ds{i}' for i in range(3)],*[f'injected{i}' for i in range(3)]}
 if not required.issubset(set(arrays)):raise ValueError('archive missing tensors')
 for name,meta in [('visual',r['main_visual']),('prefill',r['prefill_embeds']),('prefill_logits',r['prefill_logits'])]+[(f'ds{i}',r['full_deepstack'][i]) for i in range(3)]+[(f'injected{i}',r['injected_deepstack'][i]) for i in range(3)]:
  if tensor_bytes(arrays[name])!=meta['sha256']:raise ValueError('tensor fingerprint '+name)
 if not np.array_equal(arrays['prefill'][0,start:start+256],arrays['visual'][ids]) or tensor_bytes(arrays['visual'][ids])!=r['pruned_visual']['sha256']:raise ValueError('actual selected prefill')
 for i in range(3):
  selected=arrays[f'ds{i}'][ids]
  if not np.array_equal(selected,arrays[f'injected{i}']) or tensor_bytes(selected)!=r['selected_deepstack'][i]['sha256']:raise ValueError('actual selected DS '+str(i))
 logits=np.ascontiguousarray(arrays['prefill_logits']).view(np.uint16)
 if r['prefill_logits']['dtype']!='torch.bfloat16' or np.any((logits&0x7f80)==0x7f80):raise ValueError('nonfinite logits')
 forced=r.get('forced');outputs=r.get('output_ids');trace=r.get('decode_trace');eos=r.get('eos_ids');controls=r.get('generation_scalars')
 if not isinstance(outputs,list) or not outputs or any(type(i) is not int for i in outputs) or not isinstance(trace,list) or len(trace)!=len(outputs):raise ValueError('generation inventory')
 expected=dict(e1['generation_scalars']);expected['max_new_tokens']=33 if forced else 2048
 if controls!=expected:raise ValueError('generation recipe')
 if r.get('eos')!=(outputs[-1] in eos) or r.get('truncated') is not (not forced and outputs[-1] not in eos and len(outputs)>=2048):raise ValueError('EOS/truncation flags')
 if forced:
  if r.get('protocol')!='forced33' or outputs!=[100]*33 or eos!=[]:raise ValueError('forced diagnostic')
  for key in ('input_ids','attention_mask','pixel_values','image_grid_thw'):
   if 'processor_'+key not in arrays or tensor_bytes(arrays['processor_'+key])!=r['processor'][key]['sha256']:raise ValueError('forced processor archive')
 else:
  if r.get('protocol')!='natural' or eos!=e1['eos_ids'] or not (outputs[-1] in eos or len(outputs)==2048):raise ValueError('natural termination')
 for i,t in enumerate(trace):
  prev=0 if i==0 else len(ret)+i-1;after=len(ret)+i;expected_pos=r['prefill_position_ids'] if i==0 else [[[prev+delta+768]]]*3
  if (t.get('position_ids')!=expected_pos or t.get('cache_position')!=(list(range(len(ret))) if i==0 else [prev]) or t.get('rope_delta')!=[[delta+768]] or (t.get('layer_cache_lengths') not in ([],[0]*36) if i==0 else t.get('layer_cache_lengths')!=[prev]*36) or t.get('layer_cache_lengths_after')!=[after]*36):raise ValueError('actual decode cache/position')
 if forced:
  caches=r.get('cache_fingerprints')
  if not isinstance(caches,list) or len(caches)!=33:raise ValueError('forced KV count')
  for i,layers in enumerate(caches):
   if len(layers)!=36:raise ValueError('forced KV layers')
   for pair in layers:
    if set(pair)!={'key','value'}:raise ValueError('forced KV pair')
    for meta in pair.values():
     if meta.get('dtype')!='torch.bfloat16' or meta.get('shape')!=[1,8,len(ret)+i,128] or not re.fullmatch('[0-9a-f]{64}',meta.get('sha256','')):raise ValueError('forced KV fingerprint')
 return True
