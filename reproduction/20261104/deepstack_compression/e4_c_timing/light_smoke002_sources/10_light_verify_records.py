"""CPU-only semantic checks for a *completed* lightweight identity record.

This module does not itself admit a run or write a proof. A reviewed outer
verifier must first bind its inputs, original launcher and terminal evidence.
"""
import hashlib
from pathlib import Path
import numpy as np
from light_compare import sha

def tensor_hash(x):return hashlib.sha256(x.tobytes()).hexdigest()

def validate_light(record,arrays_path,forced_token):
    r=record;mode=r.get('mode')
    if mode not in ('C-ON','C-DROP','C-SKIP') or type(r.get('forced')) is not bool:raise ValueError('invalid mode or forced flag')
    ids=r['output_ids'];forced=r['forced']
    if not isinstance(ids,list) or not ids or any(type(x) is not int for x in ids):raise ValueError('invalid output tokens')
    recipe={'do_sample':False,'max_new_tokens':33 if forced else 2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0}
    if (r.get('generation_scalars')!=recipe or r['generation_scalars'].get('do_sample') is not False or
        r.get('min_new_tokens')!=(33 if forced else None) or r.get('max_new_tokens')!=recipe['max_new_tokens'] or len(ids)>recipe['max_new_tokens'] or
        not isinstance(r.get('eos_ids'),list) or any(type(x) is not int for x in r['eos_ids'])):raise ValueError('generation recipe mismatch')
    if forced:
        if type(forced_token) is not int or r['forced_token']!=forced_token or ids!=[forced_token]*33 or r['eos_ids']!=[]:raise ValueError('invalid forced tokens')
    elif r['forced_token'] is not None or r['max_new_tokens']!=2048 or (ids[-1] not in r['eos_ids'] and len(ids)!=2048):raise ValueError('invalid natural termination')
    if r['eos']!=(ids[-1] in r['eos_ids']) or r['truncated']!=(not forced and ids[-1] not in r['eos_ids'] and len(ids)>=2048):raise ValueError('EOS/truncation mismatch')
    kv=r.get('cache_fingerprints')
    if not isinstance(kv,list) or len(kv)!=(len(ids) if forced else 1) or any(not isinstance(step,list) or len(step)!=36 for step in kv):raise ValueError('incomplete KV fingerprints')
    for step in kv:
        for layer in step:
            for name in ('key','value'):
                meta=layer.get(name)
                if not isinstance(meta,dict) or not isinstance(meta.get('sha256'),str) or len(meta['sha256'])!=64 or not meta.get('shape') or not meta.get('dtype'):raise ValueError('invalid KV fingerprint')
    s,L=r['image_start'],r['original_length'];order=r['ordered_ids'];spatial=r['spatial_ids']
    if type(s) is not int or type(L) is not int or s<0 or s+1024>L or len(order)!=256 or len(set(order))!=256 or any(type(x) is not int or not 0<=x<1024 for x in order) or spatial!=sorted(order):raise ValueError('invalid token selection')
    retained=list(range(s))+[s+x for x in spatial]+list(range(s+1024,L));n=len(retained)
    if r['retained_ids']!=retained:raise ValueError('invalid retained token order')
    full=np.asarray(r['full_position_ids'],dtype=np.int64).reshape(3,L);delta=int(full.max())+1-L
    if r['full_delta']!=[[delta]] or r['adjusted_delta']!=[[delta+768]] or r['prefill_position_ids']!=full[:,retained][:,None,:].tolist():raise ValueError('invalid P3 position or delta')
    trace=r['decode_trace']
    if len(trace)!=len(ids):raise ValueError('incomplete decode trace')
    for step,z in enumerate(trace):
        cp=n+step-1
        if (z['rope_delta']!=[[delta+768]] or z['layer_cache_lengths'] not in (([],[0]*36) if step==0 else ([cp]*36,)) or
            z['layer_cache_lengths_after']!=[n+step]*36 or z['cache_position']!=(list(range(n)) if step==0 else [cp]) or
            z['position_ids']!=(r['prefill_position_ids'] if step==0 else [[[cp+delta+768]]]*3)):raise ValueError('decode/cache trajectory mismatch')
    arrays=r['arrays'];path=Path(arrays_path).resolve()
    if arrays.get('kind')!='full_arrays' or Path(arrays['path']).resolve()!=path or sha(path)!=arrays['sha256']:raise ValueError('archive identity mismatch')
    with np.load(path,allow_pickle=False) as z:
        required={'visual','pruned','prefill','prefill_logits','processor_input_ids','processor_attention_mask','processor_pixel_values','processor_image_grid_thw'}
        if not required<=set(z.files):raise ValueError('missing actual input array')
        checks={'visual':'main_visual','pruned':'pruned_visual','prefill':'prefill_embeds','prefill_logits':'prefill_logits'}
        for name,meta in checks.items():
            if tensor_hash(z[name])!=r[meta]['sha256']:raise ValueError('tensor hash mismatch: '+name)
        for name,meta in r['processor'].items():
            if tensor_hash(z['processor_'+name])!=meta['sha256']:raise ValueError('processor tensor hash mismatch')
        if not np.array_equal(z['pruned'],z['visual'][spatial]) or not np.array_equal(z['prefill'][0,s:s+256],z['pruned']):raise ValueError('selection not present in actual prefill')
        if mode=='C-ON':
            mask=[[False]*s+[True]*256+[False]*(n-s-256)]
            if r['injected_mask']!=mask or len(r['injected_deepstack'])!=3:raise ValueError('invalid DS mask')
            for j in range(3):
                ds='ds'+str(j);selected='injected'+str(j)
                if (ds not in z or selected not in z or tensor_hash(z[ds])!=r['full_deepstack'][j]['sha256'] or
                    tensor_hash(z[selected])!=r['injected_deepstack'][j]['sha256'] or not np.array_equal(z[selected],z[ds][spatial])):raise ValueError('actual DS selection mismatch')
        elif r['injected_mask'] is not None or r['injected_deepstack'] is not None:raise ValueError('DS off injection present')
        logits=z['prefill_logits'];dtype=r['prefill_logits']['dtype']
        if dtype=='torch.bfloat16':finite=((np.frombuffer(logits.tobytes(),dtype='<u2')&0x7f80)!=0x7f80).all()
        elif dtype=='torch.float32':finite=np.isfinite(np.frombuffer(logits.tobytes(),dtype='<f4')).all()
        else:raise ValueError('unexpected logits dtype')
        if not finite:raise ValueError('nonfinite first logits')
    return {'status':'passed','tokens':len(ids),'decode_steps':len(trace),'forced':forced,'mode':mode}
