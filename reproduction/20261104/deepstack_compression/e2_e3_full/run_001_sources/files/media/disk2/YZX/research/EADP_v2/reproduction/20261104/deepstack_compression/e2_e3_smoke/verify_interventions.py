"""CPU-side tensor and trace reconstruction for the locked E2/E3 smoke draft.

These functions alone do not establish launch, E1, input, tokenizer or exit
provenance; an independent complete acceptance gate is still required.
"""
import hashlib
import random
import numpy as np
import torch


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def expected_permutation(image_hash):
    seed=int.from_bytes(hashlib.sha256(('20261104|'+image_hash+'|perm').encode()).digest()[:8],'big')
    r=random.Random(seed);ids=list(range(256))
    for i in range(255,0,-1):
        j=r.randrange(i);ids[i],ids[j]=ids[j],ids[i]
    return seed,ids


def validate_one(record, arrays):
    """Raise ValueError on any input, tensor, injection, or decode mismatch."""
    def req(ok,msg):
        if not ok:raise ValueError(msg)
    r=record;family=r['family'];arm=r['arm'];spatial=r['spatial_ids'];ordered=r['ordered_ids']
    ref_off=family=='REF' and arm=='D0'
    req(r['budget']==256 and r['ds_on'] is (not ref_off),'budget')
    req(len(ordered)==256 and len(set(ordered))==256 and sorted(ordered)==spatial and
        all(type(x) is int and 0<=x<1024 for x in ordered),'selection')
    start=r['image_start'];length=r['original_length']
    retained=list(range(start))+[start+x for x in spatial]+list(range(start+1024,length))
    req(r['retained_ids']==retained,'retained tokens')
    full=np.asarray(r['full_position_ids'],dtype=np.int64).reshape(3,length)
    delta=int(full.max())+1-length+768
    req(r['prefill_position_ids']==full[:,retained][:,None,:].tolist() and r['adjusted_delta']==[[delta]],'P3 positions')
    ids=r['output_ids'];trace=r['decode_trace'];n=len(retained)
    req(len(trace)==len(ids) and len(ids)>0,'forward count')
    for step,z in enumerate(trace):
        cp=n+step-1
        req(z['rope_delta']==[[delta]] and z['layer_cache_lengths_after']==[n+step]*36 and
            z['layer_cache_lengths'] in (([],[0]*36) if step==0 else ([cp]*36,)) and
            z['cache_position']==(list(range(n)) if step==0 else [cp]) and
            z['position_ids']==(r['prefill_position_ids'] if step==0 else [[[cp+delta]]]*3),'decode KV/position')
    mask=[[False]*start+[True]*256+[False]*(length-start-1024)]
    req(r['actual_injection_verified'] is True and r['injected_mask']==(None if ref_off else mask),'actual injection mask')
    if ref_off:req(r['injected_deepstack'] is None,'reference-off injection')
    req(digest(arrays['visual'])==r['main_visual']['sha256'] and
        digest(arrays['visual'][spatial])==r['pruned_visual']['sha256'],'visual arrays')
    req(digest(arrays['prefill'])==r['prefill_embeds']['sha256'] and
        np.array_equal(arrays['prefill'][0,start:start+256],arrays['visual'][spatial]),'prefill embeddings')
    for i in range(3):
        full_ds=arrays['ds'+str(i)];before=arrays['selected_before'+str(i)]
        req(digest(full_ds)==r['full_deepstack'][i]['sha256'] and np.array_equal(before,full_ds[spatial]),'selected DS')
        if ref_off:
            req(digest(before)==r['selected_deepstack'][i]['sha256'] and 'injected'+str(i) not in arrays,'reference-off selection')
            continue
        injected=arrays['injected'+str(i)]
        req(digest(injected)==r['injected_deepstack'][i]['sha256']==r['selected_deepstack'][i]['sha256'],'injected DS hash')
        if family=='REF' and arm=='D1':expected=before
        elif family=='E2':
            req(arm in ('000','111','100','010','001'),'unknown E2 arm')
            expected=before if arm[i]=='1' else np.zeros_like(before)
        elif family=='E3' and arm=='PERM':
            seed,perm=expected_permutation(r['image_sha256'])
            req(r['intervention_metadata']['seed']==seed and r['intervention_metadata']['permutation']==perm and
                all(j!=x for j,x in enumerate(perm)),'permutation')
            expected=before[perm]
        elif family=='E3' and arm=='MEAN':
            meta=r['full_deepstack'][i];dtype={'torch.bfloat16':torch.bfloat16,'torch.float16':torch.float16,
                                              'torch.float32':torch.float32}[meta['dtype']]
            t=torch.from_numpy(before.copy()).view(dtype).reshape(256,-1)
            expected=t.mean(dim=0,keepdim=True).expand_as(t).contiguous().view(torch.uint8).numpy().reshape(injected.shape)
        else:raise ValueError('unknown intervention')
        req(np.array_equal(injected,expected),'actual injected stream')
    req(r['greedy_calls']==1 and r['visual_calls']==1 and r['main_merger_calls']==1 and
        all(x==1 for x in r['block_calls']) and r['ds_calls']==[1,1,1],'visual call counts')
    fingerprints=r.get('cache_fingerprints')
    expected_kv=r['forced'] and (family=='REF' or (family=='E2' and arm in ('000','111')))
    req(isinstance(fingerprints,list) and len(fingerprints)==(len(ids) if expected_kv else 0),'KV fingerprint count')
    for step,layers in enumerate(fingerprints):
        req(len(layers)==36,'KV fingerprint layers')
        for layer in layers:
            req(set(layer)=={'key','value'},'KV components')
            for meta in layer.values():
                h=meta.get('sha256');shape=meta.get('shape')
                req(isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h) and
                    meta.get('dtype')=='torch.bfloat16' and shape==[1,8,n+step,128],'KV fingerprint metadata')
    return True
