"""Diagnostic-only exact budget comparison; no GPU or scoring."""
import hashlib
from pathlib import Path
import numpy as np
COMMON=('task','qid','group_id','image_sha256','message_sha256','budget','ds_on','selection_mode','greedy_calls','processor','main_visual','pruned_visual','image_start','original_length','full_deepstack','ordered_ids','spatial_ids','retained_ids','full_position_ids','full_delta','prefill_position_ids','adjusted_delta','prefill_embeds','prefill_mask','injected_mask','injected_deepstack','prefill_logits','output_ids','answer','eos_ids','eos','truncated','max_new_tokens','generation_scalars','forced','forced_token','decode_trace','cache_fingerprints','block_calls','main_merger_calls','ds_calls')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def compare_metadata(actual,reference):
 k=actual.get('budget')
 if k not in (128,256,512,1024) or actual.get('ds_on') is not True or reference.get('actual_injection_verified') is not True:raise ValueError('invalid budget/DS reference')
 for name in COMMON:
  if name not in actual or name not in reference or actual[name]!=reference[name]:raise ValueError('identity mismatch: '+name)
 mode='keep_all' if k==1024 else 'author_greedy'
 if actual['selection_mode']!=mode or actual['greedy_calls']!=(0 if k==1024 else 1) or actual.get('ds_none')!=[False]*3:raise ValueError('actual method path')
 ordered=actual['ordered_ids'];spatial=actual['spatial_ids'];start=actual['image_start'];n=actual['original_length']
 if len(ordered)!=k or len(set(ordered))!=k or any(type(i) is not int or not 0<=i<1024 for i in ordered) or spatial!=sorted(ordered):raise ValueError('selection invalid')
 if k==1024 and ordered!=list(range(1024)):raise ValueError('keep-all order')
 retained=list(range(start))+[start+i for i in spatial]+list(range(start+1024,n))
 if actual['retained_ids']!=retained:raise ValueError('retained position identity')
 full=actual['full_position_ids'];pos=actual['prefill_position_ids']
 if len(full)!=3 or len(pos)!=3 or any(len(axis)!=1 or len(axis[0])!=n for axis in full):raise ValueError('position shape')
 if pos!=[[[axis[0][i] for i in retained]] for axis in full]:raise ValueError('P3 position mismatch')
 delta=max(max(axis[0]) for axis in full)+1-n
 if actual['full_delta']!=[[delta]] or actual['adjusted_delta']!=[[delta+1024-k]]:raise ValueError('independent full/adjusted delta mismatch')
 mask=actual['injected_mask'];expected=[False]*len(retained);expected[start:start+k]=[True]*k
 if mask!=[expected] or not isinstance(actual['injected_deepstack'],list) or len(actual['injected_deepstack'])!=3:raise ValueError('DS mask/streams')
 if not actual['prefill_logits']:raise ValueError('missing endpoint logits')
 ids=actual['output_ids'];forced=actual['forced'];eos=actual['eos_ids'];trace=actual['decode_trace'];seq=len(retained);adj=delta+1024-k
 if not isinstance(ids,list) or not ids or any(type(i) is not int for i in ids):raise ValueError('invalid output tokens')
 if forced:
  token=actual['forced_token']
  if type(token) is not int or ids!=[token]*33 or eos!=[] or actual['max_new_tokens']!=33:raise ValueError('forced prefix mismatch')
 elif not eos or actual['max_new_tokens']!=2048 or not (ids[-1] in eos or len(ids)==2048):raise ValueError('natural EOS/cap mismatch')
 if len(trace)!=len(ids):raise ValueError('decode call count')
 for i,step in enumerate(trace):
  previous=0 if i==0 else seq+i-1;after=seq+i
  expected_pos=pos if i==0 else [[[seq+i-1+adj]]]*3
  cache_pos=list(range(seq)) if i==0 else [seq+i-1]
  if step.get('position_ids')!=expected_pos or step.get('cache_position')!=cache_pos or step.get('rope_delta')!=[[adj]] or step.get('layer_cache_lengths')!=[previous]*36 or step.get('layer_cache_lengths_after')!=[after]*36:raise ValueError('decode position/cache trajectory')
 fps=actual['cache_fingerprints']
 if len(fps)!=(33 if forced else 1):raise ValueError('KV fingerprint step count')
 for i,layers in enumerate(fps):
  if len(layers)!=36:raise ValueError('KV fingerprint layers')
  for layer in layers:
   if set(layer)!={'key','value'}:raise ValueError('KV tensor pair')
   for value in layer.values():
    if not isinstance(value,dict) or len(value.get('shape',[]))<2 or value['shape'][-2]!=seq+i or not isinstance(value.get('sha256'),str) or len(value['sha256'])!=64:raise ValueError('KV fingerprint shape/hash')
 return {'status':'passed','budget':k,'forced':forced,'task':actual['task'],'qid':actual['qid']}

def compare_archives(actual,reference):
 a=actual['arrays'];b=reference['arrays'];ap=Path(a['path']);bp=Path(b['path'])
 if a.get('kind')!='full_arrays' or b.get('kind')!='full_arrays' or sha(ap)!=a.get('sha256') or sha(bp)!=b.get('sha256'):raise ValueError('array provenance')
 required={'visual','prefill','prefill_logits',*[f'ds{i}' for i in range(3)],*[f'injected{i}' for i in range(3)]}
 if actual['forced']:required.update('processor_'+k for k in ('input_ids','attention_mask','pixel_values','image_grid_thw'))
 with np.load(ap,allow_pickle=False) as left,np.load(bp,allow_pickle=False) as right:
  common=set(left.files)&set(right.files)
  if not required<=common or 'pruned' not in left.files:raise ValueError('missing diagnostic arrays')
  for name in common:
   if not np.array_equal(left[name],right[name]):raise ValueError('array differs: '+name)
  spatial=actual['spatial_ids'];start=actual['image_start'];k=actual['budget']
  if not np.array_equal(left['pruned'],left['visual'][spatial]):raise ValueError('selection/visual mismatch')
  if not np.array_equal(left['prefill'][0,start:start+k],left['pruned']):raise ValueError('actual prefill/selection mismatch')
  for i in range(3):
   if not np.array_equal(left['injected'+str(i)],left['ds'+str(i)][spatial]):raise ValueError('selected DS stream mismatch')
 return {'n_common_arrays':len(common),'compared':sorted(common)}
