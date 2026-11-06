"""No-GT checks for one full-panel record and retained anchor arrays.

For non-anchors, high dimensional values are not archived; E3 interventions
are checked only via runtime verification and logged fingerprints.
"""
import hashlib,random
import numpy as np

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def check(r,arrays=None):
 ordered=r['ordered_ids'];spatial=r['spatial_ids'];start=r['image_start'];length=r['original_length'];n=length-768
 if len(ordered)!=256 or len(set(ordered))!=256 or sorted(ordered)!=spatial or any(type(i) is not int or not 0<=i<1024 for i in ordered):raise ValueError('selection')
 kept=list(range(start))+[start+i for i in spatial]+list(range(start+1024,length))
 if r['retained_ids']!=kept:raise ValueError('retained ids')
 full=np.asarray(r['full_position_ids'],dtype=np.int64).reshape(3,length);delta=int(full.max())+1-length+768
 if r['prefill_position_ids']!=full[:,kept][:,None,:].tolist() or r['adjusted_delta']!=[[delta]]:raise ValueError('P3 positions')
 trace=r['decode_trace'];ids=r['output_ids']
 if len(trace)!=len(ids) or not ids:raise ValueError('decode call count')
 for i,x in enumerate(trace):
  cp=n+i-1
  if (x['rope_delta']!=[[delta]] or x['layer_cache_lengths_after']!=[n+i]*36 or
      x['layer_cache_lengths'] not in (([],[0]*36) if i==0 else ([cp]*36,)) or
      x['cache_position']!=(list(range(n)) if i==0 else [cp]) or
      x['position_ids']!=(r['prefill_position_ids'] if i==0 else [[[cp+delta]]]*3)):raise ValueError('decode position/KV lengths')
 if (r['actual_injection_verified'] is not True or r['injected_mask']!=[[False]*start+[True]*256+[False]*(length-start-1024)] or
     r['greedy_calls']!=1 or r['visual_calls']!=1 or r['main_merger_calls']!=1 or any(x!=1 for x in r['block_calls']) or r['ds_calls']!=[1,1,1]):raise ValueError('runtime injection or tower calls')
 family,arm=r['family'],r['arm'];info=r['arrays'];summaries=info.get('tensors') if info['kind']=='summary_only' else None
 if family=='E2' and (len(arm)!=3 or any(c not in '01' for c in arm)):raise ValueError('bad E2 arm')
 if family=='E3' and arm not in ('PERM','MEAN'):raise ValueError('bad E3 arm')
 if family=='E3' and arm=='PERM':
  seed=int.from_bytes(hashlib.sha256(('20261104|'+r['image_sha256']+'|perm').encode()).digest()[:8],'big');rng=random.Random(seed);perm=list(range(256))
  for i in range(255,0,-1):j=rng.randrange(i);perm[i],perm[j]=perm[j],perm[i]
  if r['intervention_metadata']['seed']!=seed or r['intervention_metadata']['permutation']!=perm or any(i==v for i,v in enumerate(perm)):raise ValueError('invalid fixed permutation')
 if summaries is not None:
  if summaries['visual']!=r['main_visual'] or summaries['prefill']!=r['prefill_embeds']:raise ValueError('summary visual/prefill')
  for i in range(3):
   if summaries['ds'+str(i)]!=r['full_deepstack'][i] or summaries['injected'+str(i)]!=r['injected_deepstack'][i] or r['selected_deepstack'][i]!=r['injected_deepstack'][i]:raise ValueError('summary DS metadata')
   if family=='E2' and arm[i]=='1' and summaries['selected_before'+str(i)]!=r['injected_deepstack'][i]:raise ValueError('kept stream mismatch')
   if family=='E2' and arm[i]=='0':
    meta=r['injected_deepstack'][i];size={'torch.bfloat16':2,'torch.float16':2,'torch.float32':4,'torch.float64':8}[meta['dtype']];nbytes=int(np.prod(meta['shape']))*size
    if meta['sha256']!=hashlib.sha256(bytes(nbytes)).hexdigest():raise ValueError('disabled stream not bitwise zero')
 if arrays is not None:
  if digest(arrays['visual'])!=r['main_visual']['sha256'] or digest(arrays['visual'][spatial])!=r['pruned_visual']['sha256'] or digest(arrays['prefill'])!=r['prefill_embeds']['sha256'] or not np.array_equal(arrays['prefill'][0,start:start+256],arrays['visual'][spatial]):raise ValueError('anchor visual/prefill')
  for i in range(3):
   full_ds=arrays['ds'+str(i)];before=arrays['selected_before'+str(i)];actual=arrays['injected'+str(i)]
   if digest(full_ds)!=r['full_deepstack'][i]['sha256'] or not np.array_equal(before,full_ds[spatial]) or digest(actual)!=r['injected_deepstack'][i]['sha256'] or r['selected_deepstack'][i]!=r['injected_deepstack'][i]:raise ValueError('anchor DS')
   if family=='E2':expected=before if arm[i]=='1' else np.zeros_like(before)
   elif arm=='PERM':expected=before[perm]
   else:
    import torch
    meta=r['full_deepstack'][i];dtype={'torch.bfloat16':torch.bfloat16,'torch.float16':torch.float16,'torch.float32':torch.float32}[meta['dtype']]
    t=torch.from_numpy(before.copy()).view(dtype).reshape(256,-1);expected=t.mean(0,keepdim=True).expand_as(t).contiguous().view(torch.uint8).numpy().reshape(actual.shape)
   if not np.array_equal(actual,expected):raise ValueError('anchor intervention differs')
 return True
